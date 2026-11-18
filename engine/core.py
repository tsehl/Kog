"""JAX engine: Normal monsters, Spells (Normal, Quick-Play, Continuous), Traps (Normal, Counter), ignition effects,
chains and response windows (MR5 rules).

State is a struct of arrays (one game = N card slots). `step(bank, s, a)` applies the action, then advances to the
next decision that has at least two options: single-option decisions are played automatically (as the harness does
on the oracle side), and effects execute instruction by instruction (state D_RUN) until the next choice. Everything
is vmappable (no Python on values).

Zone, position and type codes reuse the ygopro-core values (LOCATION_*, POS_*, TYPE_*) to simplify comparison with
the oracle.

Effects: each card has up to E effects, each = a kind (Spell activation, ignition) + 3 bytecode programs (cost,
target, resolution) of P instructions (op, a, b, c) + a filter table. Register machine: R set registers (masks
[N]). The cards are written in `cards.py`.

Chain: a stack of CH chain links (card, effect, player, registers, negated). Response windows (decision D_CHAIN):
W_QUICK = ygopro-core's QuickEffect (alternating priority, two consecutive passes close the window), W_PHASE =
PhaseEvent (start/end of phase window). When the window closes, if there is a chain it resolves (top to bottom),
then a new window opens for the turn player; otherwise play resumes according to `ret` (R_*).
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# persistent compilation cache: no recompilation at each launch as long as the code does not change
jax.config.update("jax_compilation_cache_dir", os.environ.get("YGO_JAX_CACHE", os.path.expanduser("~/.cache/ygo-jax")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

# locations (= ygopro-core LOCATION_*)
Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED, Z_EXTRA = 1, 2, 4, 8, 16, 32, 64
Z_OVERLAY = 128  # attached Xyz material (ov_par: the Xyz monster, seq: rank among its materials, ctrl: its controller)
# positions (= POS_*); hand = 10 and GY = 5, as the oracle returns them
P_FU_ATK, P_FD_ATK, P_FU_DEF, P_FD_DEF = 1, 2, 4, 8
P_HAND, P_GRAVE = 10, 5
P_SFU, P_SFD = 5, 10  # face-up (POS_FACEUP) / face-down (POS_FACEDOWN) Spell/Trap
P_FACEUP = P_FU_ATK | P_FU_DEF
# types (= TYPE_*)
TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL = 1, 2, 4, 0x10
TYPE_QUICKPLAY, TYPE_CONTINUOUS, TYPE_COUNTER = 0x10000, 0x20000, 0x100000
TYPE_EQUIP = 0x40000
TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ, TYPE_LINK = 0x40, 0x80, 0x2000, 0x800000, 0x4000000
TYPE_EXTRA = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK  # Extra Deck monsters
TYPE_TUNER = 0x1000
TYPE_PENDULUM = 0x1000000
TYPE_SPSUMMON = 0x2000000  # no Normal Summon / Set (monster that can only be Special Summoned)
P_SPELL = TYPE_PENDULUM | TYPE_SPELL  # type of a Pendulum in the Pendulum Zone (card::get_type), and of its activation
# statuses (bits)
ST_SUMMON_TURN, ST_ATTACKED, ST_FORM_CHANGED, ST_SEL = 1, 2, 4, 8  # ST_SEL: chosen in the current selection
ST_SET_TURN, ST_LEAVE = 16, 32  # Set this turn (Spell/Trap); activated, will go to the GY at the end of the chain
ST_BDES = 64  # destroyed by battle, goes to the GY at the end of the damage step (if still on the field)
ST_ON = 128  # resolved Continuous Spell/Trap: its continuous effects apply (STATUS_EFFECT_ENABLED)
ST_SSFAIL = 256  # group Summon: not enough room, sent by rule after all placements
# 512, 1024, 2048: free (formerly ST_CNT / ST_CNTN when E = 2)
ST_FLIP_TURN = 4096  # Flip Summoned this turn (STATUS_FLIP_SUMMON_TURN)
ST_SPSUM, ST_LINKSUM = 8192, 16384  # Special Summoned / Link Summoned (IsSpecialSummoned, IsLinkSummoned): as long as
# it stays on the field
SUMMON_NORMAL, SUMMON_TRIBUTE, SUMMON_SPECIAL = 0x10000000, 0x11000000, 0x40000000
SUMMON_FUSION, SUMMON_RITUAL, SUMMON_SYNCHRO = 0x43000000, 0x45000000, 0x46000000
SUMMON_XYZ, SUMMON_PENDULUM, SUMMON_LINK = 0x49000000, 0x4a000000, 0x4c000000
# suminfo: native type (bits 24..30 and 0..15), origin location (16..23), player (31).
# The information survives moves; a new Summon replaces it.
# Composable native reasons (ocgapi_constants.h).
REASON_DESTROY, REASON_RELEASE, REASON_MATERIAL, REASON_SUMMON = 1, 2, 8, 16
REASON_BATTLE, REASON_EFFECT, REASON_COST, REASON_RULE = 32, 64, 128, 1024
REASON_LOST_TARGET, REASON_DISCARD = 512, 16384
REASON_FUSION, REASON_SYNCHRO, REASON_RITUAL, REASON_XYZ, REASON_LINK = 0x40000, 0x80000, 0x100000, 0x200000, 0x10000000
MOVE_REASON_EXPLICIT = 1 << 30

# activation negated (STATUS_ACTIVATE_DISABLED): a Pendulum then goes to the GY, not to the Extra Deck
ST_ACTDIS = 32768
ST_PSTEP = 65536  # placed by the group Summon in progress (EVENT_SPSUMMON for the group at the end)
# phases (= ygopro-core PHASE_*, as seen at decisions)
PH_DRAW, PH_STANDBY, PH_MAIN1, PH_BATTLE, PH_MAIN2, PH_END = 0x1, 0x2, 0x4, 0x8, 0x100, 0x200
# pending decisions (D_RUN: effect being executed, internal to the engine)
D_NONE, D_IDLE, D_BATTLE, D_TRIBUTE, D_ZONE, D_TARGET, D_DISCARD, D_SELECT, D_POS, D_RUN, D_CHAIN, D_TRIG, D_YESNO = range(13)
# D_YESNO: yes / no question during resolution (Duel.SelectYesNo): A_YES continues the effect, A_PASS stops it
# D_TRIG (internal): pending triggers to test, handled by settle outside the execution loop
# what the current zone choice is for. ZC_PROC: Summon by procedure (Synchro, Xyz, Link); ZC_PEND: Pendulum Summon
# (one card at a time, in uid order); ZC_REPLAY: the D_YESNO question is "replay the attack?" (SelectYesNo 30),
# not an effect's Duel.SelectYesNo
ZC_SUMMON, ZC_SSET, ZC_ACT, ZC_SPSUM, ZC_PROC, ZC_PEND, ZC_REPLAY = range(7)
ZC_INLINE = 7  # choice of an effect when activating a Continuous Trap (native description 94)
CF_ACT, CF_CONT = 1, 2  # nature of the chain link, independent of the kind of the chosen effect
CF_ACT_NEG = 4  # activation negated; distinct from DISABLE_CHAIN
CF_SRC_UNRELATED = 8  # the source moved after the activation started
PS_SYN, PS_XYZ, PS_LINK, PS_PEND = 1, 2, 3, 4  # psel
# Link markers (LINK_MARKER_*)
LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T, LM_TR = 1, 2, 4, 8, 0x20, 0x40, 0x80, 0x100
# response windows; W_TRIG: putting trigger effects on the chain (SEGOC)
W_QUICK, W_PHASE, W_TRIG = range(3)
# events (ygopro-core EVENT_*): available during the next window and the chain built on it;
# those that occur during the resolution of a chain link replace the previous ones (only the last link counts)
# The same events also feed `tev` (accumulated since the last event point) for triggers.
EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON, EV_ATTACK, EV_TO_GRAVE, EV_BATTLE_DESTROYED, EV_FLIP, EV_PRE_DMG = range(8)
NEV = 8  # EV_PRE_DMG: EVENT_PRE_DAMAGE_CALCULATE (attacker and target, at the start of the damage calculation window)
# continuation of a W_QUICK window closed without a chain (`chained`: a chain has resolved since it opened)
R_IDLE, R_BATTLE, R_TO_BP, R_TO_EP, R_TO_M2B, R_TO_EPB, R_DP, R_PHASE, R_ATK1, R_ATK2, R_DMG1, R_DMG2, \
    R_DS1, R_DS2, R_DS3, R_HAND = range(16)
# R_HAND: after the end-of-turn discard (PhaseEvent, steps 21-22: event point, then another check of the hand size
# limit; optional End Phase triggers are no longer offered, hand_adjusted)
# damage step (ygopro BattleCommand, MR5): R_DS1 start (step 20), R_DS2 after the target is flipped (22),
# R_DS3 damage calculation (24, PHASE_DAMAGE_CAL), then R_DMG1 after the calculation (32) and R_DMG2 after the
# send to the GY (34). dstep: DS_BEFORE (windows 1-2), DS_CAL (3), DS_AFTER (after the calculation)
DS_NONE, DS_BEFORE, DS_CAL, DS_AFTER = range(4)
# activation during the damage step (bank.edmg): EFFECT_FLAG_DAMAGE_STEP, EFFECT_FLAG_DAMAGE_CAL,
# condition aux.StatChangeDamageStepCondition (not after the damage calculation)
DMG_STEP, DMG_CAL, DMG_STAT = 1, 2, 4
# limits (bank.eopt): once per turn by name, once per turn per card, once while the card stays
# (SetCountLimit(1) + EFFECT_FLAG_NO_TURN_RESET)
OPT_NONE, OPT_NAME, OPT_CARD, OPT_CARD_NR = range(4)
# SetCountLimit(1, {id, k}) with k >= 1: a distinct per-name counter for each k (eopt = OPT_NAMEK + k - 1)
OPT_NAMEK = 4
# bank.etrig: optional trigger without EFFECT_FLAG_DELAY ("when ..."): can miss the timing (brk)
TRIG_WHEN = 1 << 30
# FIELD trigger: any card undergoes the event, the effect's card is in the erng locations (hand, GY, face-up
# monster / Spell/Trap zone); the condition on eg goes through OP_EVENT(c=1) in the cost
TRIG_FIELD = 1 << 29
# phase windows (PhaseEvent)
WP_DRAW, WP_STANDBY, WP_BSTART, WP_BEND, WP_END = range(5)

HAND_LIMIT = 6
N_MAIN_ZONES = 5

# effects
E = 4           # effects per card (shape of the banks; the network does not depend on it)
# per-effect counters in the high bits of st (not public, see obs.ST_PUBLIC)
# ST_CNT << e: effect e already used this turn (SetCountLimit(1) without a code: per card, cleared if it moves)
ST_CNT = 1 << 17
ST_CNTN = ST_CNT << E  # ST_CNTN << e: same, without reset on the next turn (EFFECT_FLAG_NO_TURN_RESET)
assert 17 + 2 * E <= 31, "not enough st bits for E"
ST_SST, ST_SSXT = 1 << 29, 1 << 30  # virtual filter bits (F_ST): Special Summoned this turn by the player
# virtual filter bit (F_ST): the card does not leave at the end of the chain (not ST_LEAVE, ygopro's
# STATUS_LEAVE_CONFIRMED). c:IsAbleToHand() / IsAbleToDeck() are false for such a card
# (card::is_capable_send_to_hand / _to_deck): a resolved Spell waiting for the GY cannot be targeted for a return
# to the hand.
ST_STAY = 1 << 28
# of the filter (ss_t), from the Extra Deck for ST_SSXT; read in filter_mask, never stored in st
assert ST_CNTN << (E - 1) < ST_SST
ST_TURN_RESET = ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN | (ST_CNT * ((1 << E) - 1)) | ST_FLIP_TURN
P = 12          # instructions per program
PT = 6          # instructions tested per cost / target program (test_ok)
R = 4           # set registers
NF = 4          # filters per effect
KC = 24         # candidate (card, effect) pairs for activation tested by legal (hand + field of the deciding player)
MAX_SLOTS = 192  # covers 60+15 cards per player and the token slots on the field
ACT_ROUNDS = (MAX_SLOTS * E + KC - 1) // KC  # constant bound: no N x N matrix
KA = 2          # candidates tested by auto_pass (beyond that: no automatic pass, legal decides)
K_NONE, K_ACTIVATE, K_IGNITION, K_TRIG_F, K_TRIG_O, K_QUICK, K_CONT, K_PROC = range(8)  # mandatory / optional trigger;
# K_QUICK: monster quick effect (speed 2), activatable from the erng locations; K_CONT: continuous effect, applied
# as long as its card is face-up in the erng locations (and resolved, for a Spell/Trap)
PG_COST, PG_TARGET, PG_RESOLVE = range(3)
CH = 12          # chain links (at most 1 ignition + 2 x 5 Spell/Trap zones)
(OP_END, OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_CHOOSE, OP_MOVE, OP_SPSUMMON, OP_DRAW,
 OP_DAMAGE, OP_RECOVER, OP_SHUFFLE_HAND, OP_COND_CHAIN, OP_CHAINCARD, OP_NEGATE, OP_PAY_LP, OP_EVENT, OP_COND_TURN,
 OP_COND_EVP, OP_COND_SELF, OP_LOCK, OP_MODIFY, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE, OP_FUSMAT, OP_RITMAT, OP_LABEL,
 OP_COND_LABEL, OP_YESNO, OP_EQUIP, OP_BLOCK, OP_SSLOCK) = range(34)
N_OPS = 34
# OP_SSLOCK: until the end of the turn, player a (relative) can no longer Special Summon cards that match
# filter b, except those of filter c - 1 if c > 0 (EFFECT_CANNOT_SPECIAL_SUMMON registered by
# Duel.RegisterEffect: target = b and not (c - 1)).
# Approximation: the Summoning player is the controller of the Summoned card.
# OP_BLOCK: start of a conditional block (if ... then ... end followed by instructions), a = index of the first
# instruction after the block. No effect at execution; a false test (or "no" at OP_YESNO) inside the block jumps to
# a instead of stopping the program.
# OP_LABEL: the chain link remembers whether register a is non-empty (e:SetLabel at activation); OP_COND_LABEL:
# condition on that label at resolution; OP_YESNO: the effect's player chooses whether to continue (D_YESNO);
# OP_EQUIP: Duel.Equip(tp, handler, first card of register a if it is face-up in the monster zone)
# filter fields; locations are relative to the effect's player (me / opponent)
F_LOC_ME, F_LOC_OP, F_POS, F_TYPE, F_NOT_SELF, F_ATK_LO, F_ATK_HI, F_DEF_LO, F_DEF_HI, F_ATTR, F_RACE, F_SET, \
    F_CODE1, F_CODE2, F_ST, F_LV_LO, F_LV_HI, F_SPSUM, F_NODIS, F_FUS, F_RIT, F_ZX, F_REASON, F_NOTTYPE = range(24)
FW = 24  # F_ZX: ZX_* bits (co-linked, main zone)
# ZX_OPFU: c:IsControler(p) or c:IsFaceup(); ZX_FREE: Duel.GetMZoneCount(p, c) > 0; ZX_FREEANY: ft > 0 or
# c:GetSequence() < 5; GetMutualLinkedGroupCount() > 0 (cache cl_c); GetSequence() < 5
# F_RIT: Ritual monster that can be Summoned now (cache rz_c) in RIT_GREATER or RIT_EQUAL mode
# F_FUS: Fusion monster of the Extra Deck that can be Fusion Summoned now (cache fz_c)
# F_ST: one of the status bits; F_SPSUM: can be Special Summoned (no FL_NO_SS);
ZX_COLINK, ZX_MAIN, ZX_EXTRA, ZX_FREE, ZX_FREEANY, ZX_OPFU = 1, 2, 4, 8, 16, 32
# F_NODIS: effects not negated (not IsDisabled)
NSET = 4  # archetypes (16-bit setcode) per card
# stat modifiers (EFFECT_*), continuous effects (bank.emod) and temporary ones (state tables tm_*)
# UPDATE_*, SET_*, SET_*_FINAL, SET_BASE_* for ATK then DEF
MC_NONE, MC_ATK, MC_SET_ATK, MC_FINAL_ATK, MC_BASE_ATK, MC_DEF, MC_SET_DEF, MC_FINAL_DEF, MC_BASE_DEF = range(9)
# prohibitions and protections (continuous effects): EFFECT_CANNOT_ATTACK, CANNOT_BE_BATTLE_TARGET,
# CANNOT_BE_EFFECT_TARGET, IMMUNE_EFFECT (Trap effects), CANNOT_SPECIAL_SUMMON (targeted cards, all locations),
# INDESTRUCTABLE_BATTLE (value: minimum ATK of the opposing monster, 0 = always), CANNOT_ACTIVATE (Trap cards,
# players M_VAL: 1 = me, 2 = opponent)
MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP, MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP = range(9, 16)
MC_NO_DIRECT = 17  # EFFECT_CANNOT_DIRECT_ATTACK
# EFFECT_DISABLE: the card's effects are negated (continuous ones inactive, no activation, chain links negated)
MC_DISABLE = 16
MC_FUS_SUB = 19  # named material substitute (passive bank metadata)
MC_INDES_EFF = 18  # EFFECT_INDESTRUCTABLE_EFFECT: not destroyed by effects (OP_MOVE with destruction)
FL_NO_ATTACK, FL_NO_BTARGET, FL_NO_ETARGET, FL_IMM_TRAP, FL_NO_SS, FL_DISABLED = 1, 2, 4, 8, 16, 32  # bits of fl_c
FL_NO_DIRECT = 128  # no direct attack
FL_INDES_EFF = 256  # indestructible by effects
FL_ONF = 64  # face-up on the field, effects enabled (STATUS_EFFECT_ENABLED) at the previous computation
PF_NO_TRAP = 1  # bits of pf_c
# card::add_effect: a temporary modifier of this code erases those of these codes already on the card (bit mask)
_b = lambda *cs: sum(1 << c for c in cs)
MC_KILL = [0] * 32
for _set, _fin, _upd, _base in ((MC_SET_ATK, MC_FINAL_ATK, MC_ATK, MC_BASE_ATK), (MC_SET_DEF, MC_FINAL_DEF, MC_DEF, MC_BASE_DEF)):
    MC_KILL[_set] = MC_KILL[_base] = _b(_set, _fin, _base)
    MC_KILL[_fin] = _b(_upd, _set, _fin)
# value of a temporary modifier (OP_MODIFY): constant, half of the base ATK / of the current ATK of the target
MV_CONST, MV_HALF_BASE_ATK, MV_HALF_ATK = range(3)
# duration: end of turn (RESETS_STANDARD_PHASE_END), end of damage calculation (RESET_PHASE + PHASE_DAMAGE_CAL)
TD_TURN, TD_DCAL, TD_PERM = range(3)  # TD_PERM: as long as the card stays on the field (RESETS_STANDARD)
# fields of emod: code, the card itself (SINGLE_RANGE) or filter 0 (FIELD), value, x number of cards of
# filter 1, condition: turn player (1: me, 2: opponent), phases (PH_* mask, 0: all), number of cards of
# filter 1 in [M_CMIN, M_CMAX] (if M_CNT; M_PER and M_CNT are mutually exclusive)
M_CODE, M_SELF, M_VAL, M_PER, M_TP, M_PH, M_CNT, M_CMIN, M_CMAX = range(9)
# M_SELF: 1 = the card itself (SINGLE_RANGE); 2 = the monster it equips (EFFECT_TYPE_EQUIP, sorted like FIELD)
M_EQUIP = 2
MW = 9
SSL = 4         # Special Summon restrictions of the turn (OP_SSLOCK); beyond that, the oldest is replaced
KM = 16         # active continuous effects taken into account (compaction, like KC)
TM = 16         # temporary modifiers per turn
FK = 4          # Fusion materials at most (one filter per material)
# Ritual: sum of the Levels of the Tributes >= Level with no superfluous Tribute (Greater), or equal (Equal)
RIT_GREATER, RIT_EQUAL = 1, 2
RIT_MON = TYPE_MONSTER | TYPE_RITUAL  # Ritual Spells also have TYPE_RITUAL
LVMAX, SMAX = 12, 26  # material Levels; tracked sums (bit masks) up to SMAX
EXK = 32        # Fusion monsters of the Extra Deck evaluated by refresh (fz_c)
EXP = 16        # Synchro / Xyz / Link monsters of the deciding player's Extra Deck evaluated by refresh_ex (15 at most)
BIG = 1 << 20

# actions: verb x slot, then special actions
V_SUMMON, V_MSET, V_REPOS, V_ATTACK, V_SELECT, V_SSET, V_ACT0 = range(7)
NV = V_ACT0 + E
A_TO_BP, A_TO_M2, A_TO_EP, A_ZONE0, A_POS_ATK, A_POS_DEF, A_PASS, A_YES = 0, 1, 2, 3, 10, 11, 12, 13  # A_ZONE0 + z
N_SPECIAL = 14  # A_YES: "yes" to D_YESNO ("no": A_PASS)


def n_actions(n):
    return NV * n + N_SPECIAL


def special(n, k):
    return NV * n + k


class Bank(NamedTuple):
    """Global card bank (static, shared by all games)."""
    code: jnp.ndarray   # [C] ygopro id (for comparison with the oracle)
    atk: jnp.ndarray
    df: jnp.ndarray
    level: jnp.ndarray
    ctype: jnp.ndarray  # [C] TYPE_* bits
    attr: jnp.ndarray   # [C] ATTRIBUTE_* bits
    race: jnp.ndarray   # [C] RACE_* bits
    ekind: jnp.ndarray  # [C, E] K_*
    prog: jnp.ndarray   # [C, E, 3, P, 4] (op, a, b, c)
    tprog: jnp.ndarray  # [C, E, PT2, 4] cost then target without the OP_END (test mode); PT2 = the longest in the bank
    econd: jnp.ndarray  # [C,E] length of the condition prefix in tprog (before cost/target)
    filt: jnp.ndarray   # [C, E, NF, FW]
    etrig: jnp.ndarray  # [C, E] bits 1 << EV_*: activation only in response to these events (0: free)
    eopt: jnp.ndarray   # [C, E] once per turn by name (SetCountLimit(1, id))
    # [C, E] locations from which a monster quick effect activates (Z_HAND | Z_MZONE); locations of a continuous effect
    erng: jnp.ndarray
    setcode: jnp.ndarray  # [C, NSET] archetypes (0: none)
    emod: jnp.ndarray   # [C, E, MW] continuous effect (K_CONT): M_*
    edmg: jnp.ndarray   # [C, E] activation during the damage step: DMG_* bits
    fmat: jnp.ndarray   # [C, FK, FW] Fusion materials (one filter per material, over the hand and the field)
    fmn: jnp.ndarray    # [C] number of materials (0: no recipe)
    fsub: jnp.ndarray   # [C] mask of the named slots that accept a substitute
    submat: jnp.ndarray  # [C,0 or 1] locations of the substitutes; width 0 eliminates the path at trace time
    erl: jnp.ndarray    # [C] 0: free; 1: EnableReviveLimit; 2: forbidden by effect even after a correct Summon
    smat: jnp.ndarray   # [C, 2, FW] Synchro: filters of the Tuner (exactly one) and of the non-Tuners (one or more)
    syn: jnp.ndarray    # [C] Synchro monster (Synchro Summon procedure)
    xmat: jnp.ndarray   # [C, FW] Xyz: material filter (in addition to Level = Rank)
    xct: jnp.ndarray    # [C] Xyz: number of materials (0: not an Xyz)
    lmark: jnp.ndarray  # [C] Link: arrows (LM_*); the Link Rating is in level, DEF is zero
    lmat: jnp.ndarray   # [C, FW] Link: material filter
    lmin: jnp.ndarray   # [C] Link: number of materials, at least
    lmax: jnp.ndarray   # [C] Link: at most (0: no Link procedure)
    lneed: jnp.ndarray  # [C] Link: the materials must include a Link monster
    ldiff: jnp.ndarray  # [C] Link: materials with different names (GetClassCount(Card.GetCode) == #g)
    lsc: jnp.ndarray    # [C] Pendulum: left Scale
    rsc: jnp.ndarray    # [C] Pendulum: right Scale
    einline: jnp.ndarray  # [C,0 or E] effect offered at activation (index + 1, 0: none)
    erefund: jnp.ndarray  # [C,0 or E] EFFECT_COUNT_CODE_OATH: give the count back if the activation is negated
    equip: jnp.ndarray   # [C,0 or 1] equipment card; width 0 (none in the bank) eliminates the path at trace time


def make_bank(code, atk, df, level, ctype=None, ekind=None, prog=None, filt=None, etrig=None, eopt=None, attr=None,
              race=None, erng=None, setcode=None, emod=None, edmg=None, fmat=None, fmn=None, erl=None,
              smat=None, syn=None, xmat=None, xct=None, lmat=None, lmin=None, lmax=None, lneed=None, ldiff=None,
              lsc=None, rsc=None, fsub=None, submat=None, einline=None, erefund=None, equip=None, econd=None):
    """df: DEF, or arrows for a Link monster (def field of the ygopro database): stored in lmark, DEF zero."""
    c = code.shape[0]
    i = lambda x, shape: jnp.zeros(shape, I32) if x is None else jnp.asarray(x, I32)
    is_link = (jnp.full(c, TYPE_MONSTER | TYPE_NORMAL, I32) if ctype is None else jnp.asarray(ctype, I32)) & TYPE_LINK != 0
    df = jnp.asarray(df, I32)
    tprog, cond_lengths = test_prog(prog, c, econd)
    return Bank(code=jnp.asarray(code, I32), atk=jnp.asarray(atk, I32), df=jnp.where(is_link, 0, df),
                tprog=tprog, econd=cond_lengths,
                lmark=jnp.where(is_link, df, 0), lmat=i(lmat, (c, FW)), lmin=i(lmin, (c,)), lmax=i(lmax, (c,)),
                lneed=i(lneed, (c,)), ldiff=i(ldiff, (c,)), lsc=i(lsc, (c,)), rsc=i(rsc, (c,)),
                level=jnp.asarray(level, I32),
                ctype=jnp.full(c, TYPE_MONSTER | TYPE_NORMAL, I32) if ctype is None else jnp.asarray(ctype, I32),
                attr=i(attr, (c,)), race=i(race, (c,)),
                ekind=i(ekind, (c, E)), prog=i(prog, (c, E, 3, P, 4)), filt=i(filt, (c, E, NF, FW)),
                etrig=i(etrig, (c, E)), eopt=i(eopt, (c, E)), erng=i(erng, (c, E)), setcode=i(setcode, (c, NSET)),
                emod=i(emod, (c, E, MW)), edmg=i(edmg, (c, E)), fmat=i(fmat, (c, FK, FW)), fmn=i(fmn, (c,)),
                fsub=i(fsub, (c,)), submat=i(submat, (c, 0)),
                erl=i(erl, (c,)), smat=i(smat, (c, 2, FW)), syn=i(syn, (c,)),
                xmat=i(xmat, (c, FW)), xct=i(xct, (c,)), einline=i(einline, (c, 0)), erefund=i(erefund, (c, 0)),
                equip=i(equip, (c, 0)))


def test_prog(prog, c, econd=None):
    """Compact test programs (numpy, when the bank is built): cost then target, without the padding OP_END (no effect
    in test mode). test_ok only walks the length of the longest, not 2 x PT."""
    import numpy as np
    pr = np.zeros((c, E, 3, P, 4), np.int32) if prog is None else np.asarray(prog, np.int32)
    prefix = np.zeros((c, E), np.int32) if econd is None else np.asarray(econd, np.int32)
    cond_lengths = np.zeros((c, E), np.int32)
    rows=[]
    for card in range(c):
        effects=[]
        for effect in range(E):
            result=[]
            boundary = int(prefix[card, effect])
            if not 0 <= boundary <= PT:
                raise ValueError("condition prefix outside the cost program")
            for index, (opcode,a,b,cc) in enumerate(pr[card,effect,:PG_RESOLVE,:PT].reshape(-1,4)):
                if index == boundary:
                    cond_lengths[card, effect] = len(result)
                opcode,a,b,cc=map(int,(opcode,a,b,cc))
                if opcode==OP_END:
                    continue
                if opcode==OP_FILTER:
                    # copy, complement and relation; other high bits: derived test / assertions
                    cc &= 15 | (1 << 20) | (1 << 21) | (1 << 22)
                can_join = index != boundary and result and result[-1][0]==OP_FILTER and result[-1][1]==a
                minimum = (opcode==OP_REQ and cc==0) or (opcode==OP_CHOOSE and a==b)
                count = cc&15 if opcode==OP_CHOOSE else b
                if can_join and (opcode==OP_REQ or opcode==OP_CHOOSE and a==b) and 0<=count<=(255 if minimum else 254):
                    previous=list(result[-1]);packed=previous[3]
                    if minimum:
                        packed=(packed & ~(255<<4)) | (max((packed>>4)&255,count)<<4)
                    else:
                        old=(packed>>12)&255
                        packed=(packed & ~(255<<12)) | ((min(old,count+1) if old else count+1)<<12)
                    previous[3]=packed;result[-1]=tuple(previous)
                else:
                    result.append((opcode,a,b,cc))
            effects.append(result)
        rows.append(effects)
    n=max(1,max(len(effect) for card in rows for effect in card))
    out=np.zeros((c,E,n,4),np.int32)
    for card,effects in enumerate(rows):
        for effect,instructions in enumerate(effects):
            if instructions:out[card,effect,:len(instructions)]=instructions
    return jnp.asarray(out), jnp.asarray(cond_lengths)


class State(NamedTuple):
    reason: jnp.ndarray  # [N] last known native reason mask
    suminfo: jnp.ndarray  # [N] last Summon: type, origin location and player
    ss_t: jnp.ndarray    # [N] Special Summoned this turn (ACTIVITY_SPSUMMON activity counters), kept when moved
    cid: jnp.ndarray     # [N] index in the bank
    zone: jnp.ndarray    # [N]
    seq: jnp.ndarray     # [N] zone (field) or rank in the location (deck: top = max seq)
    ctrl: jnp.ndarray    # [N]
    owner: jnp.ndarray   # [N]
    pos: jnp.ndarray     # [N]
    st: jnp.ndarray      # [N] ST_* bits
    lp: jnp.ndarray      # [2]
    turn: jnp.ndarray
    tp: jnp.ndarray      # turn player
    phase: jnp.ndarray
    ns_used: jnp.ndarray
    dec: jnp.ndarray     # D_*
    dec_player: jnp.ndarray
    pend: jnp.ndarray    # slot in progress (monster to Summon, attacker, Spell to Set/activate), -1 otherwise
    pend_set: jnp.ndarray  # the Summon in progress is a Set
    need: jnp.ndarray    # remaining selections (Tributes, discards, effect choices)
    picks: jnp.ndarray   # selections already made in the current decision
    smin: jnp.ndarray    # variable-size CHOOSE: minimum number of cards (A_PASS ends the selection beyond that)
    zctx: jnp.ndarray    # ZC_*: use of the current zone choice
    zsel: jnp.ndarray    # zone chosen for the Special Summon in progress
    ch_src: jnp.ndarray  # effect in progress (activation or resolution): source card (-1 if none)
    ch_src_mv: jnp.ndarray  # clock of the source at the start of the activation
    ch_tgt_t: jnp.ndarray  # clock of the creation of the target relations
    ch_targets: jnp.ndarray  # [N] bool: announced targets, relation dated by ch_tgt_t
    ch_eff: jnp.ndarray  # effect index
    ch_p: jnp.ndarray    # activating player
    ch_k: jnp.ndarray    # index of its chain link (= n_chain during the activation)
    # activated by a card under EFFECT_DISABLE_EFFECT: the chain link will be negated (DISABLE_CHAIN)
    ch_dis: jnp.ndarray
    ch_lab: jnp.ndarray  # label of the effect in progress (OP_LABEL)
    ch_flags: jnp.ndarray  # CF_* of the current chain link
    pg: jnp.ndarray      # program in progress (PG_*)
    pc: jnp.ndarray      # instruction in progress
    reg: jnp.ndarray     # [R, N] set registers
    done: jnp.ndarray
    odraw: jnp.ndarray   # impossible rule draw: loss at the next "adjust" (activation, end of a phase window)
    winner: jnp.ndarray  # -1 if not finished
    # chain stack
    lk_src: jnp.ndarray  # [CH] card
    lk_eff: jnp.ndarray  # [CH]
    lk_p: jnp.ndarray    # [CH]
    lk_reg: jnp.ndarray  # [CH, R, N] registers at activation time (a card that moves drops out of it)
    lk_neg: jnp.ndarray  # [CH] activation negated
    lk_t: jnp.ndarray    # [CH] move clock at activation
    lk_tgt_t: jnp.ndarray  # [CH] clock of the creation of the target relations
    lk_targets: jnp.ndarray  # [CH, N] bool: announced targets (public observation)
    lk_lab: jnp.ndarray  # [CH] label (OP_LABEL)
    lk_flags: jnp.ndarray  # [CH] CF_* kept after the choice of an inline effect
    mv: jnp.ndarray      # [N] clock of the card's last move
    clk: jnp.ndarray     # move clock: a card moved after the activation drops out of lk_reg (without copying it)
    en_t: jnp.ndarray    # [N] clock at the activation of the card's continuous effects (enable_field_effect: arrival on
    # the field, flip face-up, resolution); orders the modifiers like ygopro's effect ids
    cav: jnp.ndarray     # [N, E] continuous effect condition met at the last computation (EFFECT_STATUS_AVAILABLE)
    cav_t: jnp.ndarray   # [N, E] clock at which it became met again (is_available then gives the effect a new id)
    n_chain: jnp.ndarray
    # windows
    wk: jnp.ndarray      # W_*
    cw_passed: jnp.ndarray  # the other player has passed
    ret: jnp.ndarray     # R_*: continuation of the W_QUICK window
    chained: jnp.ndarray
    ph_restart: jnp.ndarray  # player who resumes the phase window after a chain
    wph: jnp.ndarray     # WP_*: current phase window
    bend_ep: jnp.ndarray  # end of the Battle Phase toward the End Phase (otherwise Main 2)
    atk: jnp.ndarray     # current attacker
    atk_tgt: jnp.ndarray  # target (-1: direct attack)
    # recent events
    ev_g: jnp.ndarray    # [NEV, N] cards concerned (eg)
    ev_p: jnp.ndarray    # [NEV] originating player (ep)
    # triggers
    pzone: jnp.ndarray   # [N] previous location (IsPreviousLocation)
    tev: jnp.ndarray     # [NEV, N] events undergone since the last event point
    tew: jnp.ndarray     # [NEV, N] same, since the last break_effect (optional "when" triggers)
    teg: jnp.ndarray     # [NEV, N] events of the last event point (eg of the triggers, OP_EVENT c=1)
    tg: jnp.ndarray      # [N] pending trigger effect
    tg_e: jnp.ndarray    # [N] SEGOC bits (1 << e), private hand (1 << (E+e))
    tst: jnp.ndarray     # SEGOC step: 0/1 mandatory (turn player, opponent), 2/3 optional
    pt_p: jnp.ndarray    # window to open if no trigger is activated: player, other player has passed
    pt_passed: jnp.ndarray
    dstep: jnp.ndarray   # DS_*: where the damage step stands
    ds_t: jnp.ndarray    # clock at the start of the damage step (attacker or target moved since: attack negated)
    ds_flip: jnp.ndarray  # the target was flipped at the start of the damage step (FLIP event at damage calculation)
    rb_m: jnp.ndarray    # [N] opposing monsters at the (re)declaration of the attack (ygopro-core's opp_mzone)
    rb_t: jnp.ndarray    # clock at the (re)declaration: a monster moved since is no longer the same one (fieldid_r)
    rpl: jnp.ndarray     # attack replayed (is_replaying_attack: no new EVENT_ATTACK_ANNOUNCE)
    opt_cid: jnp.ndarray   # [2, N*E] keys + 1 of the "once per turn" effects already used
    lock_cid: jnp.ndarray  # [2, N] cid + 1 of the cards whose effects the player can no longer activate this turn
    ssl: jnp.ndarray       # [SSL, 5] Special Summon restrictions of the turn: (source card + 1, effect, filter of the
    #                        targeted cards, filter of the exceptions + 1 or 0, player)
    # current stats (cache recomputed by refresh after each micro-step) and temporary modifiers
    atk_c: jnp.ndarray   # [N] current ATK (outside the monster zone: base ATK)
    df_c: jnp.ndarray    # [N]
    batk_c: jnp.ndarray  # [N] current base ATK (GetBaseAttack)
    tm_code: jnp.ndarray  # [TM] MC_* (in creation order; table emptied at end of turn)
    tm_src: jnp.ndarray  # [TM] source card of the modifier (owner of the effect)
    tm_m: jnp.ndarray    # [TM, N] cards concerned (a card moved since tm_t drops out, RESETS_STANDARD)
    tm_v: jnp.ndarray    # [TM, N] value per card
    tm_t: jnp.ndarray    # [TM] move clock at creation
    tm_dur: jnp.ndarray  # [TM] TD_*
    tm_rd: jnp.ndarray   # [TM] RESET_DISABLE (effect of the card on itself): cleared when the card is negated
    tm_n: jnp.ndarray
    fl_c: jnp.ndarray    # [N] current prohibitions / protections (FL_*), computed by refresh
    ib_c: jnp.ndarray    # [N] indestructible by battle against a monster with ATK >= ib_c (BIG: no)
    pf_c: jnp.ndarray    # [2] player prohibitions (PF_*)
    proc: jnp.ndarray    # [N] properly Summoned (STATUS_PROC_COMPLETE: Fusion Summon...); cleared in hand / deck
    fz_c: jnp.ndarray    # [N] Fusion monster of the Extra Deck achievable with its owner's hand and field
    fsel: jnp.ndarray    # material selection in progress (OP_FUSMAT): never played automatically if a choice remains
    rz_c: jnp.ndarray    # [N] Ritual monster in hand that can be Summoned: bit 0 in Greater mode, bit 1 in Equal mode
    rsel: jnp.ndarray    # Ritual Tribute selection in progress (OP_RITMAT): RIT_* mode, 0 otherwise
    sz_c: jnp.ndarray    # [N] Synchro monster of the Extra Deck that can be Summoned now (materials on the field)
    xz_c: jnp.ndarray    # [N] Xyz monster of the Extra Deck that can be Summoned now
    lz_c: jnp.ndarray    # [N] Link monster of the Extra Deck that can be Summoned now
    cl_c: jnp.ndarray    # [N] co-linked Link monster (GetMutualLinkedGroupCount() > 0), computed by refresh
    psel: jnp.ndarray    # material selection of a procedure in progress (D_TRIBUTE, card pend): PS_*, 0 otherwise
    ov_par: jnp.ndarray  # [N] Z_OVERLAY: index of the Xyz monster that holds this material (-1 otherwise)
    eq_t: jnp.ndarray    # [N] equipment card: index of the equipped monster (equiping_target, -1 otherwise)
    eq_mv: jnp.ndarray   # [N] clock of the equipment (the card or its target moved since: relation broken)
    ps_used: jnp.ndarray  # Pendulum Summon done this turn (flag 10000000 of proc_pendulum.lua)
    uid: jnp.ndarray     # [N] card creation order (ygopro's cardid: deck 0, Extra 0, deck 1, Extra 1)
    ovf: jnp.ndarray     # table overflow (bits): 1 = more than KM active continuous effects, 2 = more than TM modifiers
    # Per-slot copy of the card attributes (bank.X[cid]). A slot keeps its card for the whole game, so these are
    # written once by new_game (and by with_cid). Reading them is elementwise; bank.X[s.cid] was a gather over the
    # N slots, the most expensive operation of the engine on TPU.
    k_ctype: jnp.ndarray
    k_level: jnp.ndarray
    k_attr: jnp.ndarray
    k_race: jnp.ndarray
    k_code: jnp.ndarray
    k_setcode: jnp.ndarray
    k_erl: jnp.ndarray
    k_fmn: jnp.ndarray
    k_xct: jnp.ndarray
    k_syn: jnp.ndarray
    k_lmax: jnp.ndarray
    k_lmark: jnp.ndarray
    k_ekind: jnp.ndarray
    k_erng: jnp.ndarray
    k_emod: jnp.ndarray
    k_etrig: jnp.ndarray
    k_edmg: jnp.ndarray
    k_submat: jnp.ndarray
    k_atk: jnp.ndarray
    k_df: jnp.ndarray


I32 = jnp.int32
SLOT_FIELDS = ("ctype", "level", "attr", "race", "code", "setcode", "erl", "fmn", "xct", "syn", "lmax", "lmark", "ekind", "erng", "emod", "etrig", "edmg", "submat", "atk", "df")


def slot_fields(bank, cid):
    """{k_X: bank.X[cid]} for the per-slot card attributes of State."""
    return {f"k_{f}": getattr(bank, f)[cid] for f in SLOT_FIELDS}


def with_cid(bank, s, cid):
    """s with other cards in its slots: cid and the per-slot attributes together."""
    return s._replace(cid=cid, **slot_fields(bank, cid))


def _merge(i, outs):
    """Selection by i among the outputs of the branches, field by field. A field that no branch modifies (same
    object as the input) passes through unchanged, and only the distinct values enter the selection."""
    flat = [jax.tree.flatten(o) for o in outs]
    res = []
    for vals in zip(*[f[0] for f in flat]):
        uniq, pos = [], []
        for v in vals:
            j = next((j for j, u in enumerate(uniq) if u is v), None)
            if j is None:
                j = len(uniq)
                uniq.append(v)
            pos.append(j)
        if len(uniq) == 1:
            res.append(uniq[0])
        else:
            dt = jnp.result_type(uniq[0])
            w = i if pos == list(range(len(vals))) else jnp.asarray(pos, I32)[i]
            res.append(lax.select_n(w, *[jnp.asarray(u, dt) for u in uniq]))
    return flat[0][1].unflatten(res)


def _switch(i, fs, *ops):
    """Replaces lax.switch: computes all the branches (which lax.switch already does under vmap) but only selects
    the modified fields. Under vmap, lax.switch selects the entire state for each branch."""
    i = jnp.clip(jnp.asarray(i), 0, len(fs) - 1).astype(I32)
    return _merge(i, [f(*ops) for f in fs])


def _cond(p, f, g, *ops):
    return _switch(jnp.asarray(p).astype(I32), [g, f], *ops)


def _i(x):
    return jnp.asarray(x, I32)


# ---------------------------------------------------------------- dense replacements for scatters (TPU)
# On TPU, scatters and gathers with computed indices run almost serially (~100-300 us each, measured), while dense
# compares and reductions over small axes are nearly free. The helpers below replace the scatter idioms of the
# engine by exact dense equivalents. The pairwise ones cost O(N^2) per game, which is small for N ~ 110.

def onehot(idx, size):
    """[..., size] bool: idx == k. Out-of-range indices match nothing (same as a scatter with mode="drop")."""
    return idx[..., None] == jnp.arange(size)


def set_true(idx, size, where=True):
    """[size] bool: k is True iff some entry with idx == k (and `where`) exists. Equals
    jnp.zeros(size, bool).at[idx].set(True, mode="drop") restricted to `where`."""
    return (onehot(idx, size) & jnp.asarray(where)[..., None]).reshape(-1, size).any(0)


def set_unique(idx, vals, size):
    """[size] bool: vals[i] written at idx[i], False elsewhere; idx must be unique among the entries whose index is
    in range. Equals jnp.zeros(size, bool).at[idx].set(vals, mode="drop")."""
    return (onehot(idx, size) & vals[:, None]).any(0)


def count_below(key, w, same=None):
    """[N] int32: number of j with w[j] and key[j] < key[i] (and same[i, j] if given). Exact dense replacement of
    the rank idiom bins = zeros.at[key].add(w); rank = (cumsum(bins) - bins)[key]."""
    m = (key[None, :] < key[:, None]) & w[None, :]
    if same is not None:
        m &= same
    return m.sum(1).astype(I32)


def rel(p, r):
    """Relative player (0 = me, 1 = opponent) -> absolute."""
    return jnp.where(r == 0, p, 1 - p).astype(I32)


# ---------------------------------------------------------------- O(N) primitives

def count(s, zone, p):
    return ((s.zone == zone) & (s.ctrl == p)).sum().astype(I32)


def zone_used(s, p, zone=Z_MZONE):
    """[8]: occupied zones of player p in the location (monsters 0-6, Spells/Traps 0-7)."""
    m = (s.zone == zone) & (s.ctrl == p)
    return set_true(s.seq, 8, m)


def free_main(s, p, zone=Z_MZONE):
    return (~zone_used(s, p, zone)[:N_MAIN_ZONES]).sum().astype(I32)


def leave_hand(s, c):
    """Removes c from the hand: the following cards shift."""
    p, sq = s.ctrl[c], s.seq[c]
    shift = (s.zone == Z_HAND) & (s.ctrl == p) & (s.seq > sq)
    return s._replace(seq=s.seq - shift.astype(I32))


ORDERED = (Z_DECK, Z_HAND, Z_GRAVE, Z_REMOVED, Z_EXTRA)


def compact(s, leaving):
    """The `leaving` cards are about to leave their location: tightens the seq of the ordered locations (deck, hand,
    GY, banished). O(N)."""
    n = s.cid.shape[0]
    oz = jnp.full(n, -1, I32)
    for i, z in enumerate(ORDERED):
        oz = jnp.where(s.zone == z, i, oz)
    grp = s.ctrl * len(ORDERED) + jnp.maximum(oz, 0)
    hit = leaving & (oz >= 0)
    shift = count_below(s.seq, hit, grp[None, :] == grp[:, None])  # departing cards of the group with lower seq
    return s._replace(seq=jnp.where((oz >= 0) & ~leaving, s.seq - shift, s.seq))


def summon_mask(typ, negate=False):
    """Negative selector for F_ST / COND_SELF: compacted native type bits."""
    assert typ & 0x00ff0000 == 0
    return -(((typ >> 24) & 255) | ((typ & 65535) << 8) | (int(negate) << 24))


def summon_matches(info, encoded):
    packed = -encoded
    mask = ((packed & 255) << 24) | ((packed >> 8) & 65535)
    return (((info & 0x7f00ffff) & mask) == mask) ^ ((packed & (1 << 24)) != 0)


def effect_summon_type(mode):
    """SPSUMMON: positive procedure; generic marker encoded as -(tag+1)."""
    ordinary = jnp.where(mode == 1, SUMMON_RITUAL, jnp.where(mode == 2, SUMMON_FUSION, SUMMON_SPECIAL))
    return jnp.where(mode < 0, SUMMON_SPECIAL | (-mode - 1), ordinary)


def summon_info(typ, location, player):
    return _i(typ) | (location << 16) | (player << 31)


def move(s, c, zone, seq, ctrl, pos):
    return s._replace(pzone=s.pzone.at[c].set(s.zone[c]), zone=s.zone.at[c].set(zone), seq=s.seq.at[c].set(seq), ctrl=s.ctrl.at[c].set(ctrl),
                      pos=s.pos.at[c].set(pos), st=s.st.at[c].set(0), mv=s.mv.at[c].set(s.clk + 1),
                      en_t=s.en_t.at[c].set(s.clk + 1), clk=s.clk + 1)


def enable(s, mask):
    """The continuous effects of the cards in mask are enabled now (new effect id in ygopro)."""
    return s._replace(en_t=jnp.where(mask, s.clk + 1, s.en_t), clk=s.clk + 1)


LOC_RANK = (Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED)


def group_send(s, mask, zone, pos, zone2=None, pos2=None, alt=None, reason=0):
    """Sends the cards in `mask` to the ordered location `zone` (GY, hand) of their owner, all at once.
    Like ygopro-core (card_operation_sort), they arrive there sorted by controller (turn player first),
    origin location, then seq (ascending on the field and in hand, descending for deck/GY/banished). O(N).
    alt [N]: these cards go to zone2 / pos2 instead (an Extra Deck monster returned to the hand), in the same
    pass."""
    n = s.cid.shape[0]
    if alt is not None:
        dest = jnp.where(alt, zone2, zone)
        pos = jnp.where(alt, pos2, pos)
    else:
        dest = zone
    zr = jnp.zeros(n, I32)
    for i, z in enumerate(LOC_RANK):
        zr = jnp.where(s.zone == z, i, zr)
    desc = (s.zone == Z_DECK) | (s.zone == Z_GRAVE) | (s.zone == Z_REMOVED)
    key = ((s.ctrl != s.tp) * len(LOC_RANK) + zr) * n + jnp.where(desc, n - 1 - s.seq, s.seq)
    nk = 2 * len(LOC_RANK) * n
    if alt is not None:
        key = key + jnp.where(alt, nk, 0)  # rank counted separately for each destination
        nk *= 2
    seq = s.seq
    for o in (0, 1):
        mo = mask & (s.owner == o)
        rank = count_below(key, mo)
        if alt is not None:
            rank = rank - jnp.where(alt, (mo & ~alt).sum(), 0)
            base2 = count(s, zone2, o)
            if zone2 == Z_EXTRA and pos2 == P_FD_DEF:  # face-down: before the face-up Pendulums (end - extra_p_count)
                base2 = base2 - ((s.zone == Z_EXTRA) & (s.ctrl == o) & (s.pos == P_FU_DEF)).sum()
            base = jnp.where(alt, base2, count(s, zone, o))
        else:
            base = count(s, zone, o)
        seq = jnp.where(mo, base + rank, seq)
    s = compact(s, mask)
    s = s._replace(reason=jnp.where(mask, reason, s.reason))
    if alt is not None and zone2 == Z_EXTRA and pos2 == P_FD_DEF:  # the face-up Pendulums shift
        fu = (s.zone == Z_EXTRA) & (s.pos == P_FU_DEF) & ~mask
        nin = jnp.stack([(mask & alt & (s.owner == o)).sum() for o in (0, 1)])
        s = s._replace(seq=jnp.where(fu, s.seq + nin[s.ctrl], s.seq))
    out = (dest == Z_HAND) | (dest == Z_DECK) | (dest == Z_EXTRA)  # STATUS_PROC_COMPLETE lost
    s = s._replace(proc=jnp.where(mask & out, False, s.proc))
    return s._replace(pzone=jnp.where(mask, s.zone, s.pzone), zone=jnp.where(mask, dest, s.zone),
                      seq=jnp.where(mask, seq, s.seq), ctrl=jnp.where(mask, s.owner, s.ctrl),
                      pos=jnp.where(mask, pos, s.pos), st=jnp.where(mask, 0, s.st), mv=jnp.where(mask, s.clk + 1, s.mv),
                      en_t=jnp.where(mask, s.clk + 1, s.en_t),
                      clk=s.clk + 1)


def detach_fix(s):
    """After a detach (a material left Z_OVERLAY): ov_par cleared, ranks of the remaining materials tightened
    (card::xyz_remove). O(8N)."""
    n = s.cid.shape[0]
    gone = (s.zone != Z_OVERLAY) & (s.ov_par >= 0)

    def fix(s):
        ov = s.zone == Z_OVERLAY
        par = jnp.clip(s.ov_par, 0, n - 1)
        rank = count_below(jnp.clip(s.seq, 0, 7), ov, par[None, :] == par[:, None])  # same Xyz, lower rank
        return s._replace(ov_par=jnp.where(gone, -1, s.ov_par), seq=jnp.where(ov, rank, s.seq))
    return _cond(gone.any(), fix, lambda s: s, s)


def drop_equips(bank, s):
    """Broken equip relations (card::unequip): an equipment card that left its zone loses its target; a card still
    in place whose target left the monster zone, was turned face-down or moved since the equipping is
    destroyed (REASON_LOST_TARGET + REASON_RULE, after the move's events, like field::send_to and
    change_position). Called by refresh, before drop_overlays (ygopro's order: destroy of the equipment cards
    then send_to of the materials). With no equipment card in the bank (width 0): nothing at trace time."""
    if not bank.equip.shape[1]:
        return s
    n = s.cid.shape[0]
    t = jnp.clip(s.eq_t, 0, n - 1)
    own = s.eq_t >= 0
    gone = own & ((s.zone != Z_SZONE) | (s.mv > s.eq_mv))
    lost = own & ~gone & ((s.zone[t] != Z_MZONE) | ((s.pos[t] & P_FACEUP) == 0) | (s.mv[t] > s.eq_mv))
    s = s._replace(eq_t=jnp.where(gone | lost, -1, s.eq_t))

    def drop(s):
        s = group_send(s, lost, Z_GRAVE, P_GRAVE, reason=REASON_RULE | REASON_LOST_TARGET | REASON_DESTROY)
        return raise_ev(s, EV_TO_GRAVE, lost, s.tp)
    return _cond(lost.any(), drop, lambda s: s, s)


def drop_overlays(s):
    """The materials of Xyz monsters that left the field (anywhere but the field, moved after being attached: not
    an Extra Deck Xyz being Summoned) go to their owner's GY, all at once (ygopro: send_to of the overlays after
    the group move), sorted by controller, Xyz seq then rank. Called by refresh: a single copy in the graph,
    instead of one per group_send call."""
    par = jnp.clip(s.ov_par, 0, s.cid.shape[0] - 1)
    ovm = (s.zone == Z_OVERLAY) & (s.ov_par >= 0) & (s.zone[par] != Z_MZONE) & (s.mv[par] > s.mv)

    def drop(s):  # only if there are any: refresh is called more often in step than in step_batch (clk, ev_p)
        s2 = s._replace(seq=jnp.where(ovm, s.seq[par] * 8 + s.seq, s.seq), ctrl=jnp.where(ovm, s.ctrl[par], s.ctrl))
        s2 = group_send(s2, ovm, Z_GRAVE, P_GRAVE, reason=REASON_RULE | REASON_LOST_TARGET)
        return raise_ev(s2._replace(ov_par=jnp.where(ovm, -1, s2.ov_par)), EV_TO_GRAVE, ovm, s.tp)
    return _cond(ovm.any(), drop, lambda s: s, s)


def attach(s, sel, par):
    """The cards in sel become the materials of the Xyz monster par, after those it already has, in creation order
    (card_set sorted by cardid: Duel.Overlay with sending the materials to the GY does not sort). O(N)."""
    n = s.cid.shape[0]
    rank = count_below(s.uid, sel)
    base = ((s.zone == Z_OVERLAY) & (s.ov_par == par)).sum()
    return s._replace(pzone=jnp.where(sel, s.zone, s.pzone), zone=jnp.where(sel, Z_OVERLAY, s.zone),
                      seq=jnp.where(sel, base + rank, s.seq), ov_par=jnp.where(sel, par, s.ov_par),
                      st=jnp.where(sel, 0, s.st), mv=jnp.where(sel, s.clk + 1, s.mv),
                      en_t=jnp.where(sel, s.clk + 1, s.en_t), clk=s.clk + 1)


def group_to_owner_zone(bank, s, mask, zone, reason=0):
    """Hand or top of the owner's Deck; Extra Deck monsters return to the Extra Deck face-down.
    ST_LEAVE stays in place; SendtoDeck does not move a card already in its Deck or a monster already in the Extra."""
    mask = mask & ((s.st & ST_LEAVE) == 0)
    ex = mask & ((s.k_ctype & TYPE_EXTRA) != 0)
    mask &= (zone != Z_DECK) | ~((s.zone == Z_DECK) | (ex & (s.zone == Z_EXTRA)))
    return group_send(s, mask, zone, P_HAND, Z_EXTRA, P_FD_DEF, alt=ex, reason=reason)


def group_to_grave(bank, s, mask, reason=0):
    """Send to the GY. A Pendulum that leaves the field (monster zone, or Spell/Trap zone without a negated
    activation) goes to the Extra Deck instead, face-up, after the other cards (field::send_to, no EVENT_TO_GRAVE)."""
    pen = (s.k_ctype & TYPE_PENDULUM) != 0
    red = mask & pen & ((s.zone == Z_MZONE) | ((s.zone == Z_SZONE) & ((s.st & ST_ACTDIS) == 0)))
    return raise_ev(group_send(s, mask, Z_GRAVE, P_GRAVE, Z_EXTRA, P_FU_DEF, alt=red, reason=reason), EV_TO_GRAVE, mask & ~red, s.tp)


def selected_to_grave(bank, s, reason=0):
    return group_to_grave(bank, s, (s.st & ST_SEL) != 0, reason)._replace(picks=_i(0))


def lose(s, p, cond):
    """If cond: p loses (without overriding a victory already secured)."""
    hit = cond & ~s.done
    return s._replace(done=s.done | hit, winner=jnp.where(hit, 1 - p, s.winner))


def damage(s, p, amount):
    lp = s.lp.at[p].set(jnp.maximum(0, s.lp[p] - amount))
    s = s._replace(lp=lp)
    return lose(s, p, lp[p] <= 0)


def draw(s, p):
    m = (s.zone == Z_DECK) & (s.ctrl == p)
    c = jnp.argmax(jnp.where(m, s.seq, -1))
    empty = ~m.any()
    s2 = move(s, c, Z_HAND, count(s, Z_HAND, p), p, P_HAND)
    s = jax.tree.map(lambda a, b: jnp.where(empty, a, b), s, s2)
    return lose(s, p, empty)


def rule_draw(s, p):
    """Draw Phase draw: ygopro-core only checks for the loss at the next adjust."""
    empty = ~((s.zone == Z_DECK) & (s.ctrl == p)).any()
    return _cond(empty, lambda s: s._replace(odraw=jnp.bool_(True)), lambda s: draw(s, p), s)


def adjust(s):
    """Adjust point (ygopro's Processors::Adjust, victory check): draw from an empty deck, LP at 0 after a cost paid
    during an activation (the loss is only noted here, after the targets are chosen)."""
    s = lose(s, s.tp, s.odraw)._replace(odraw=jnp.bool_(False))
    s = lose(s, 0, (s.lp[0] <= 0) & (s.lp[1] > 0))
    return lose(s, 1, (s.lp[1] <= 0) & (s.lp[0] > 0))


def raise_ev(s, ev, mask, p):
    return s._replace(ev_g=s.ev_g.at[ev].set(s.ev_g[ev] | mask), ev_p=s.ev_p.at[ev].set(p),
                      tev=s.tev.at[ev].set(s.tev[ev] | mask), tew=s.tew.at[ev].set(s.tew[ev] | mask))


def brk(s):
    """ygopro-core's break_effect: optional triggers without EFFECT_FLAG_DELAY ("when", TRIG_WHEN) of the previous
    events miss the timing."""
    return s._replace(tew=jnp.zeros_like(s.tew))


def name_in(table, cid):
    """Is cid [...] present in the by-name register `table`?"""
    return (table == (cid + 1)[..., None]).any(-1)


def opt_key(cid, opt, oath=0):
    """Key by name: identifier, native 8-bit index and OATH flag."""
    k = jnp.where(opt >= OPT_NAMEK, opt - OPT_NAMEK + 1, 0)
    return jnp.where((opt == OPT_NAME) | (opt >= OPT_NAMEK), cid + (k << 16) + (_i(oath) << 24), -1)


def name_add(table, cid):
    """Adds a name without duplicating entries; the capacities cover all the names of the duel."""
    i = jnp.where((table == 0).any(), jnp.argmax(table == 0), table.shape[0] - 1)
    return jnp.where(name_in(table, cid), table, table.at[i].set(cid + 1))


def clear_ev(s):
    return s._replace(ev_g=jnp.zeros_like(s.ev_g))


def ev_bits(s):
    return (s.ev_g.any(axis=1) * (1 << jnp.arange(NEV))).sum().astype(I32)


def tributes_needed(bank, s, c):
    lv = bank.level[s.cid[c]]
    return jnp.where(lv >= 7, 2, jnp.where(lv >= 5, 1, 0)).astype(I32)


# ---------------------------------------------------------------- windows and flow

def open_quick(s, p, ret=None, passed=False, chained=None):
    """W_QUICK window (QuickEffect): p has priority; `passed`: the other player has already passed."""
    return s._replace(dec=_i(D_CHAIN), dec_player=_i(p), wk=_i(W_QUICK), cw_passed=jnp.bool_(passed),
                      ret=s.ret if ret is None else _i(ret),
                      chained=s.chained if chained is None else jnp.bool_(chained))


def open_phase(s, wph, phase):
    """W_PHASE window (PhaseEvent): the turn player first."""
    return s._replace(phase=_i(phase), dec=_i(D_CHAIN), dec_player=s.tp, wk=_i(W_PHASE), cw_passed=jnp.bool_(False),
                      wph=_i(wph))


def act_ok(bank, s, h, e, p, enabled=None, condition_only=False):
    """Effect e of h can be activated by p: cost + target executable, no prohibition or per-name limit."""
    cid = s.cid[h]
    # a card with negated effects can still activate them (is_activateable does not test STATUS_DISABLED): they are
    # negated at resolution (load_link)
    def check():
        oath = bank.erefund[cid, e] if bank.erefund.shape[1] else 0
        key = opt_key(cid, bank.eopt[cid, e], oath)
        available = (((bank.ekind[cid, e] == K_PROC) | ~name_in(s.lock_cid[p], cid))
                & ~((key >= 0) & name_in(s.opt_cid[p], key))
                & ~(((bank.eopt[cid, e] == OPT_CARD) | (bank.eopt[cid, e] == OPT_CARD_NR))
                    & ((s.st[h] & ((ST_CNT | ST_CNTN) << e)) != 0)))
        return test_ok(bank, s, h, e, p, enabled=enabled, condition_only=condition_only) & (condition_only | available)
    if enabled is None:
        return check()
    return lax.fori_loop(0, jnp.where(enabled, 1, 0), lambda _, previous: check(), jnp.bool_(True))


def trig_next(bank, s):
    """Defers trig_eval to settle: under vmap, calling it here would compute it at every executed instruction."""
    return s._replace(dec=_i(D_TRIG))


def trig_eval(bank, s):
    """Next decision for putting triggers on the chain (from step s.tst); if there are none left, a response
    window (after the chain is built: the opponent of the last chain link has priority)."""
    n = s.cid.shape[0]
    ebits = (1 << jnp.arange(E))[:, None]
    stage = (s.ctrl != s.tp)[None, :] + 2 * (s.k_ekind.T == K_TRIG_O)
    hidden = ((s.tg_e[None, :] & (ebits << E)) != 0)
    pending = (((s.tg_e[None, :] & ebits) != 0) | hidden) & s.tg[None, :] & (s.dec == D_TRIG) & ~s.done
    stage = jnp.where(hidden, 4, stage)  # private hand: QuickEffect after the public SEGOC

    def evaluate(carry):
        pending, phase, _ = carry
        initial = phase < 0
        # The native engine discards false conditions at the event point, then
        # revalidates cost/target of the mandatory ones (two players), then the optional ones.
        candidates = pending & (initial | (stage // 2 == phase // 2))
        def more(batch_carry):
            remaining, _, rounds = batch_carry
            return remaining.any() & (rounds < ACT_ROUNDS)
        def batch(batch_carry):
            remaining, accepted, rounds = batch_carry
            ce, ch = jnp.nonzero(remaining, size=KC, fill_value=0)
            valid = jnp.arange(KC) < remaining.sum()
            passed = jax.vmap(lambda h, e, active: act_ok(
                bank, s, h, e, s.ctrl[h], enabled=active, condition_only=initial))(ch, ce, valid) & valid
            hit = onehot(ce, E)[:, :, None] & onehot(ch, n)[:, None, :]      # [KC, E, N]
            accepted |= (hit & passed[:, None, None]).any(0)
            return remaining & ~hit.any(0), accepted, rounds + 1
        _, ok, _ = lax.while_loop(more, batch, (candidates, jnp.zeros((E, n), bool), _i(0)))
        pending &= ~candidates | ok
        avail = (ok[None, :, :] & (stage[None, :, :] == jnp.arange(4)[:, None, None])).any((1, 2))
        avail &= jnp.arange(4) >= phase
        found = ~initial & avail.any()
        nxt = jnp.where(initial, 0, jnp.where(found, jnp.argmax(avail), (phase // 2 + 1) * 2)).astype(I32)
        return pending, nxt, found

    pending, nxt, found = lax.while_loop(lambda carry: (carry[1] < 4) & ~carry[2], evaluate,
                                         (pending, s.tst, jnp.bool_(False)))
    pending &= stage >= nxt
    s = s._replace(tg_e=(pending * jnp.where(hidden, ebits << E, ebits)).sum(0).astype(I32), tg=pending.any(0))

    def go(s):
        return s._replace(tst=nxt, dec=_i(D_CHAIN), dec_player=jnp.where(nxt % 2 == 0, s.tp, 1 - s.tp).astype(I32),
                          wk=_i(W_TRIG), cw_passed=jnp.bool_(False))

    def finish(s):
        s = s._replace(wk=_i(W_QUICK), dec=_i(D_CHAIN))
        top = jnp.clip(s.n_chain - 1, 0, CH - 1)
        return _cond(s.n_chain > 0, lambda s: s._replace(dec_player=1 - s.lk_p[top], cw_passed=jnp.bool_(False)),
                        lambda s: s._replace(dec_player=s.pt_p, cw_passed=s.pt_passed), s)
    return _cond(found, go, finish, s)


def in_range(s):
    """[N] location of each card in the sense of effect ranges (erng): hand, GY, monster zone or face-up
    Spell/Trap zone; 0 elsewhere."""
    up = (s.pos & P_FACEUP) != 0
    return jnp.where((s.zone == Z_HAND) | (s.zone == Z_GRAVE) | (((s.zone == Z_MZONE) | (s.zone == Z_SZONE)) & up),
                     s.zone, 0)


def trig_point(bank, s):
    """Event point, right after a W_QUICK window opens (empty chain): the effects triggered by the events
    accumulated in `tev` go first (SEGOC), the window only opens afterwards."""
    bits = (s.tev * (1 << jnp.arange(NEV))[:, None]).sum(0)
    wbits = (s.tew * (1 << jnp.arange(NEV))[:, None]).sum(0)
    abits, awbits = (s.tev.any(1) * (1 << jnp.arange(NEV))).sum(), (s.tew.any(1) * (1 << jnp.arange(NEV))).sum()
    zr = in_range(s)
    tg_e = jnp.zeros_like(s.tg_e)
    for e in range(E):
        k = s.k_ekind[:, e]
        et = s.k_etrig[:, e]
        when = (k == K_TRIG_O) & ((et & TRIG_WHEN) != 0)
        fld = (et & TRIG_FIELD) != 0
        evs = jnp.where(fld, jnp.where(when, awbits, abits), jnp.where(when, wbits, bits))
        # effect::is_activateable: during the damage step, an optional FIELD trigger without
        # EFFECT_FLAG_DAMAGE_STEP only responds to battle events (codes 1132-1149)
        ds_lock = (k == K_TRIG_O) & fld & (s.dstep != DS_NONE) & ((s.k_edmg[:, e] & DMG_STEP) == 0)
        evs = jnp.where(ds_lock, evs & ((1 << EV_BATTLE_DESTROYED) | (1 << EV_PRE_DMG)), evs)
        hit = ((k == K_TRIG_F) | (k == K_TRIG_O)) & ((et & evs) != 0) & (~fld | ((zr & s.k_erng[:, e]) != 0))
        private = (k == K_TRIG_O) & (s.zone == Z_HAND) & ((s.pos & P_FACEUP) == 0)
        tg_e |= jnp.where(hit, jnp.where(private, 1 << (e + E), 1 << e), 0).astype(I32)
    tg = tg_e != 0
    s = s._replace(tg=tg, tg_e=tg_e, teg=s.tev, tev=jnp.zeros_like(s.tev), tew=jnp.zeros_like(s.tew), pt_p=s.dec_player, pt_passed=s.cw_passed, tst=_i(-1))
    return _cond(tg.any(), lambda s: trig_next(bank, s), lambda s: s, s)


def keep_perm(s):
    """End of turn for the modifier table: only the TD_PERM ones still on a field card that has not moved since
    remain, packed at the front (the application order comes from tm_t, not from the rank)."""
    alive = (s.tm_m & (s.mv[None, :] <= s.tm_t[:, None]) & (s.zone == Z_MZONE)[None, :]).any(1)
    keep = (s.tm_dur == TD_PERM) & (jnp.arange(TM) < s.tm_n) & alive
    o = jnp.argsort(jnp.where(keep, 0, 1) * TM + jnp.arange(TM))
    k = keep[o]
    return s._replace(tm_code=jnp.where(k, s.tm_code[o], 0), tm_m=s.tm_m[o] & k[:, None], tm_v=s.tm_v[o],
                      tm_t=s.tm_t[o], tm_dur=s.tm_dur[o], tm_rd=s.tm_rd[o], tm_src=s.tm_src[o],
                      tm_n=keep.sum().astype(I32))


def start_turn(s):
    tp = jnp.where(s.turn == 0, 0, 1 - s.tp).astype(I32)
    s = s._replace(turn=s.turn + 1, tp=tp, ns_used=jnp.bool_(False), ps_used=jnp.bool_(False), st=s.st & ~ST_TURN_RESET, phase=_i(PH_DRAW),
                   tev=jnp.zeros_like(s.tev), tew=jnp.zeros_like(s.tew), opt_cid=jnp.zeros_like(s.opt_cid), lock_cid=jnp.zeros_like(s.lock_cid),
                   ssl=jnp.zeros_like(s.ssl), ss_t=jnp.zeros_like(s.ss_t))
    s = keep_perm(s)
    # draw (except on the first turn) then window; then phase window
    return _cond(s.turn > 1, lambda s: open_quick(rule_draw(s, tp), tp, R_DP, chained=False),
                        lambda s: open_phase(s, WP_DRAW, PH_DRAW), s)


def end_phase(s):
    over = count(s, Z_HAND, s.tp) - HAND_LIMIT
    return _cond(over > 0,
                        lambda s: s._replace(phase=_i(PH_END), dec=_i(D_DISCARD), dec_player=s.tp, need=over,
                                             picks=_i(0)),
                        start_turn, s)


def to_idle(s):
    return s._replace(dec=_i(D_IDLE), dec_player=s.tp)


def to_battle(s):
    return s._replace(dec=_i(D_BATTLE), dec_player=s.tp)


def atk_canceled(s):
    """Attack negated before the damage step (adjust, step 14: STATUS_ATTACK_CANCELED): the attacker left the
    field, moved since the declaration (ds_t) or is no longer face-up in Attack Position."""
    a = s.atk
    return (s.zone[a] != Z_MZONE) | (s.mv[a] > s.ds_t) | (s.pos[a] != P_FU_ATK)


def declare(s, tgt, replay):
    """Declaration or replay of the attack (steps 6-8): target, snapshot of the opposing monsters (opp_mzone);
    EVENT_ATTACK_ANNOUNCE only at the first declaration, then the attack window."""
    n = s.cid.shape[0]
    s = s._replace(atk_tgt=tgt, rb_m=(s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE), rb_t=s.clk, rpl=replay)
    r = raise_ev(s, EV_ATTACK, (jnp.arange(n) == s.atk) & ~replay, s.tp)
    return open_quick(r._replace(ev_p=jnp.where(replay, s.ev_p, r.ev_p)), s.tp, R_ATK1, chained=False)


def atk_rollback(s):
    """attack_rollback (adjust, step 14, outside the damage step): the opposing monsters changed since the
    declaration (arrival, departure or move), or the direct attack is no longer possible (confirm_attack_target,
    without the CANNOT_BE_BATTLE_TARGET test). Computed when the attack windows close: ygopro-core freezes it at the
    first adjust where it becomes true, which only differs if the condition becomes false again (direct attack
    forbidden then allowed again)."""
    opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
    same = opp & (s.mv <= s.rb_t)
    nodir = (s.atk_tgt < 0) & ((s.fl_c[s.atk] & FL_NO_DIRECT) != 0)
    return ((s.rb_m | opp) & ~same).any() | nodir


def replay_ask(s):
    """Step 11, replay: with no possible target and no direct attack, the attack stops (it counts); otherwise the
    turn player chooses whether to replay it (SelectYesNo 30; EFFECT_MUST_ATTACK does not exist in the engine)."""
    opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
    tg = (opp & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    dr = ~opp.any() & ((s.fl_c[s.atk] & FL_NO_DIRECT) == 0)
    return _cond(tg | dr, lambda s: s._replace(dec=_i(D_YESNO), dec_player=s.tp, zctx=_i(ZC_REPLAY)), battle_over, s)


def replay_go(s):
    """Replay accepted (step 4): direct attack if there is no opposing monster left (without EFFECT_DIRECT_ATTACK,
    question 31 "direct attack?" never arises), otherwise choice of the target."""
    dr = ~((s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)).any()
    return _cond(dr, lambda s: declare(s, _i(-1), jnp.bool_(True)),
                 lambda s: s._replace(dec=_i(D_TARGET), dec_player=s.tp, rpl=jnp.bool_(True)), s)


def damage_step(bank, s):
    """Start of the damage step (steps 19-20): first window."""
    start = lambda s: trig_point(bank, open_quick(s._replace(dstep=_i(DS_BEFORE), ds_t=s.clk, ds_flip=jnp.bool_(False),
                                                             rpl=jnp.bool_(False)),
                                                  s.tp, R_DS1, chained=False))
    return _cond(atk_canceled(s), battle_over, start, s)  # step 11: attack negated, no battle


def ds_canceled(s):
    """Attack negated during the damage step (adjust, step 14): attacker or target moved (or no longer on the
    field), attacker in Defense Position."""
    a, t = s.atk, jnp.maximum(s.atk_tgt, 0)
    gone = lambda c: (s.zone[c] != Z_MZONE) | (s.mv[c] > s.ds_t)
    return gone(a) | ((s.pos[a] & (P_FU_DEF | P_FD_DEF)) != 0) | ((s.atk_tgt >= 0) & gone(t))


def end_dcal(s):
    """reset_phase(PHASE_DAMAGE_CAL): end of the "until the end of the damage calculation" modifiers."""
    return s._replace(tm_m=s.tm_m & (s.tm_dur != TD_DCAL)[:, None])


def ds_next(bank, s, f):
    """Continuation of the damage step after a window: f, or directly the end (step 33) if the attack is negated."""
    return _cond(ds_canceled(s), lambda s: dmg_send(bank, end_dcal(s)._replace(dstep=_i(DS_AFTER))), f, s)


def ds_flip(bank, s):
    """Steps 21-22: the face-down target is flipped (its FLIP effects wait for the end of the calculation), second
    window."""
    t = jnp.maximum(s.atk_tgt, 0)
    fd = (s.atk_tgt >= 0) & (s.pos[t] == P_FD_DEF)
    s = enable(s._replace(pos=s.pos.at[t].set(jnp.where(fd, P_FU_DEF, s.pos[t])), ds_flip=fd),
               fd & (jnp.arange(s.cid.shape[0]) == t))
    return trig_point(bank, open_quick(s, s.tp, R_DS2, chained=False))


def ds_cal(bank, s):
    """Steps 23-24: PHASE_DAMAGE_CAL, PRE_DAMAGE_CALCULATE event, third window."""
    n = s.cid.shape[0]
    m = (jnp.arange(n) == s.atk) | ((s.atk_tgt >= 0) & (jnp.arange(n) == s.atk_tgt))
    s = raise_ev(s._replace(dstep=_i(DS_CAL)), EV_PRE_DMG, m, s.tp)
    return trig_point(bank, open_quick(s, s.tp, R_DS3, chained=False))


def ds_damage(bank, s):
    """Steps 26-32: damage calculation, then event point (FLIP effects); the destroyed monsters stay on the field
    until `dmg_send`."""
    # no refresh here: the state is up to date on entry (each settle step and each instruction ends with a
    # refresh), and settle does another one afterwards. Under vmap, this branch of quick_end is computed at every
    # pass_priority and every apply: a refresh here almost doubled the cost of the engine.
    direct = lambda s: damage(s, 1 - s.tp, s.atk_c[s.atk])
    s = s._replace(dstep=_i(DS_AFTER))
    s = _cond(s.atk_tgt < 0, direct, lambda s: battle(bank, s, s.atk, s.atk_tgt), s)
    return trig_point(bank, open_quick(end_dcal(s), s.tp, R_DMG1, chained=False))


def dmg_send(bank, s):
    """End of the damage step: monsters destroyed by battle still on the field go to the GY, then event point
    ("destroyed by battle", "sent to the GY")."""
    m = ((s.st & ST_BDES) != 0) & (s.zone == Z_MZONE)
    s = raise_ev(group_to_grave(bank, s, m, REASON_BATTLE | REASON_DESTROY), EV_BATTLE_DESTROYED, m, s.tp)
    # materials of a destroyed Xyz: to the GY before the collection, EVENT_TO_GRAVE of the same event point. Elsewhere,
    # each instruction ends with a refresh (drop_overlays) before the next event point.
    return trig_point(bank, drop_overlays(drop_equips(bank, open_quick(s, s.tp, R_DMG2, chained=False))))


def battle_over(s):
    return to_battle(s._replace(atk=_i(-1), atk_tgt=_i(-1), dstep=_i(DS_NONE), rpl=jnp.bool_(False)))


def phase_end(s):
    """End of a phase window (two consecutive passes)."""
    def bend(s):
        # toward the End Phase: we still go through Main 2 (ygopro-core simulates "to_ep" from idle)
        s = s._replace(phase=_i(PH_MAIN2))
        return _cond(s.bend_ep, lambda s: open_quick(s, 1 - s.tp, R_TO_EP, passed=True, chained=False), to_idle, s)
    return _switch(s.wph, [
        lambda s: open_phase(s, WP_STANDBY, PH_STANDBY),
        lambda s: to_idle(s._replace(phase=_i(PH_MAIN1))),
        to_battle, bend, end_phase], clear_ev(adjust(s)))


def quick_end(bank, s):
    """End of a W_QUICK window without a chain."""
    ch = s.chained
    sel = lambda a, b: lambda s: _cond(ch, a, b, s)
    return _switch(s.ret, [
        to_idle,                                                                    # R_IDLE
        to_battle,                                                                  # R_BATTLE
        sel(to_idle, lambda s: open_phase(s, WP_BSTART, PH_BATTLE)),                # R_TO_BP
        sel(to_idle, lambda s: open_phase(s, WP_END, PH_END)),                      # R_TO_EP
        sel(to_battle, lambda s: open_phase(s._replace(bend_ep=jnp.bool_(False)), WP_BEND, PH_BATTLE)),
        sel(to_battle, lambda s: open_phase(s._replace(bend_ep=jnp.bool_(True)), WP_BEND, PH_BATTLE)),
        lambda s: open_phase(s, WP_DRAW, PH_DRAW),                                  # R_DP
        lambda s: s._replace(dec=_i(D_CHAIN), dec_player=s.ph_restart, wk=_i(W_PHASE),
                             cw_passed=jnp.bool_(False)),                           # R_PHASE
        lambda s: _cond(atk_canceled(s), battle_over,                               # R_ATK1 (step 9)
                        lambda s: _cond(atk_rollback(s), replay_ask,                # replay (step 11)
                                        lambda s: open_quick(s, s.tp, R_ATK2, chained=False), s), s),
        # R_ATK2 (after a chain: the SolveChain window, then the damage step; step 11: replay)
        lambda s: _cond(~atk_canceled(s) & atk_rollback(s), replay_ask, lambda s: damage_step(bank, s), s),
        lambda s: dmg_send(bank, s),                                                # R_DMG1
        battle_over,                                                                # R_DMG2
        lambda s: ds_next(bank, s, lambda s: ds_flip(bank, s)),                     # R_DS1
        lambda s: ds_next(bank, s, lambda s: ds_cal(bank, s)),                      # R_DS2
        lambda s: ds_next(bank, s, lambda s: ds_damage(bank, s)),                   # R_DS3
        end_phase,                                                                  # R_HAND
    ], clear_ev(s))


def pass_priority(bank, s):
    p = s.dec_player
    other = lambda s: s._replace(cw_passed=jnp.bool_(True), dec_player=1 - p)

    def close(s):
        # PointEvent case 10: the unchosen private triggers expire
        # before the resolution or the return to free play.
        s = s._replace(tg=jnp.zeros_like(s.tg), tg_e=jnp.zeros_like(s.tg_e))
        return _cond(s.wk == W_PHASE, phase_end,
                        lambda s: _cond(s.n_chain > 0, lambda s: load_link(bank, s, s.n_chain - 1),
                                           lambda s: quick_end(bank, s), s), s)
    return _cond(s.cw_passed, close, other, s)


def pass_close(bank, s):
    """Second pass of a window (cw_passed already true): the closing of pass_priority, alone. Deliberate copy:
    pass_priority (step path, verified against the oracle) is not touched."""
    return _cond(s.wk == W_PHASE, phase_end,
                 lambda s: _cond(s.n_chain > 0, lambda s: load_link(bank, s, s.n_chain - 1),
                                 lambda s: quick_end(bank, s), s), s)


def place_pending(bank, s, z):
    c = s.pend
    s = leave_hand(s, c)
    pos = jnp.where(s.pend_set, P_FD_DEF, P_FU_ATK)
    s = move(s, c, Z_MZONE, z, s.tp, pos)
    s = s._replace(st=s.st.at[c].set(ST_SUMMON_TURN), ns_used=jnp.bool_(True), pend=_i(-1))
    s = brk(s)  # end of the Summon (after the Tributes): break_effect
    s = _cond(s.pend_set, lambda s: s, lambda s: raise_ev(s, EV_SUMMON, jnp.arange(s.cid.shape[0]) == c, s.tp), s)
    return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))


def battle(bank, s, a, t):
    """Battle of a against t (already flipped by ds_flip). The destroyed ones are marked ST_BDES."""
    idx = jnp.arange(s.cid.shape[0])
    s = raise_ev(s, EV_FLIP, (idx == t) & s.ds_flip, s.tp)
    A, T, D = s.atk_c[a], s.atk_c[t], s.df_c[t]
    pa, pt = s.ctrl[a], s.ctrl[t]
    atk_pos = s.pos[t] == P_FU_ATK
    kill_t = jnp.where(atk_pos, (A > T) | ((A == T) & (A > 0)), A > D)
    kill_a = atk_pos & ((T > A) | ((A == T) & (A > 0)))
    s = damage(s, pt, jnp.where(atk_pos & (A > T), A - T, 0))
    s = damage(s, pa, jnp.where(atk_pos, jnp.maximum(T - A, 0), jnp.maximum(D - A, 0)))
    kill_t &= A < s.ib_c[t]  # INDESTRUCTABLE_BATTLE (value: the opponent's ATK)
    kill_a &= T < s.ib_c[a]
    kill = ((idx == t) & kill_t) | ((idx == a) & kill_a)
    return s._replace(st=jnp.where(kill, s.st | ST_BDES, s.st))


# ---------------------------------------------------------------- effect interpreter

def speed(bank, cid, e):
    """Spell speed: 1 (Normal Spell, ignition), 2 (Quick-Play, Trap, monster quick effect), 3 (Counter Trap)."""
    return speed_of(bank.ctype[cid], bank.ekind[cid, e])


def speed_of(ct, kind):
    """speed() from the card type and the effect kind."""
    act = kind == K_ACTIVATE
    fast = ((ct & TYPE_TRAP) != 0) | ((ct & TYPE_QUICKPLAY) != 0)
    quick = kind == K_QUICK
    return jnp.where(act & fast, jnp.where((ct & TYPE_COUNTER) != 0, 3, 2), jnp.where(quick, 2, 1)).astype(I32)


def set_match(bank, cid, sc):
    """Card cid belongs to archetype sc (ygopro's rule: same archetype, sc is a subset of the card's sub-archetype)."""
    return set_match_cs(bank.setcode[cid], sc)


def set_match_cs(cs, sc):
    """set_match on setcode rows cs [..., NSET]."""
    return ((cs != 0) & ((cs & 0xFFF) == (sc & 0xFFF)) & ((cs & sc) == sc)).any(-1)


def ss_locked(bank, s):
    """[N] cards that their controller cannot Special Summon (OP_SSLOCK restrictions of the turn).
    The active rows are at the head of the table (OP_SSLOCK inserts at the head): a loop over them only, a single
    filter_mask in the body (even j: filter of the restriction, odd: filter of the exceptions)."""
    rows = (s.ssl[:, 0] > 0).sum().astype(I32)

    def body(j, carry):
        out, hit = carry
        k, alt = j // 2, j % 2
        h, e, fi, fb, q = s.ssl[k, 0] - 1, s.ssl[k, 1], s.ssl[k, 2], s.ssl[k, 3], s.ssl[k, 4]
        filt = bank.filt[s.cid[jnp.maximum(h, 0)], jnp.clip(e, 0, E - 1)]
        m = filter_mask(bank, s, filt[jnp.clip(jnp.where(alt == 0, fi, fb - 1), 0, NF - 1)], q, jnp.maximum(h, 0),
                        bdes=False)
        done = out | (hit & ~((fb > 0) & m) & (s.ctrl == q))
        return jnp.where(alt == 0, out, done), jnp.where(alt == 0, m, hit)

    zero = jnp.zeros(s.cid.shape[0], bool)
    return lax.fori_loop(0, 2 * rows, body, (zero, zero))[0]


def special_allowed(bank, s, proper=False):
    """Current Summonability per effect; a real procedure bypasses only the revive limit."""
    revive = (s.k_erl == 0) | ((s.k_erl == 1) &
             (s.proc | ((s.zone == Z_EXTRA) & ((s.pos & P_FACEUP) == 0))))
    return ((s.zone != Z_MZONE) & ((s.fl_c & FL_NO_SS) == 0)
            & ((s.k_ctype & TYPE_MONSTER) != 0) & (revive | proper))


def reason_matches(actual, selector):
    """Positive: at least one bit; negative: all the requested bits."""
    mask = jnp.abs(selector)
    return jnp.where(selector < 0, (actual & mask) == mask, (actual & mask) != 0)


def filter_mask(bank, s, f, p, h, bdes=True, targ=False):
    """Evaluates a filter (fixed record) for player p, source card h. O(N).
    bdes: excludes monsters destroyed by battle (selections); False for the targets of continuous effects.
    targ: selection with targeting (IsCanBeEffectTarget): excludes cards that cannot be targeted."""
    n = s.cid.shape[0]
    loc = jnp.where(s.ctrl == p, f[F_LOC_ME], f[F_LOC_OP])
    m = (s.zone & loc) != 0
    m &= (s.zone != Z_OVERLAY) | (s.ov_par == h)  # Xyz materials: only those of the source card
    # monster placed by the Pendulum Summon in progress (STATUS_SUMMONING): outside the groups
    # (field::filter_matching_card)
    m &= ~((s.zone == Z_MZONE) & ((s.st & ST_PSTEP) != 0))
    if bdes:
        m &= (s.st & ST_BDES) == 0  # destroyed by battle: no longer targetable (ygopro is_capable_be_effect_target)
    m &= (f[F_POS] == 0) | ((s.pos & f[F_POS]) != 0)
    ct = s.k_ctype
    ct = jnp.where(((ct & TYPE_PENDULUM) != 0) & (s.zone == Z_SZONE), P_SPELL, ct)  # Pendulum Zone: Pendulum Spell
    m &= (f[F_TYPE] == 0) | ((ct & f[F_TYPE]) != 0)
    m &= (ct & f[F_NOTTYPE]) == 0
    m &= ~((f[F_NOT_SELF] != 0) & (jnp.arange(n) == h))
    atk, df = s.atk_c, s.df_c
    m &= (f[F_ATTR] == 0) | ((s.k_attr & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((s.k_race & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match_cs(s.k_setcode, f[F_SET])
    code = s.k_code
    # F_CODE1 < 0: everything except the name -F_CODE1 (not c:IsCode(id))
    m &= jnp.where(f[F_CODE1] < 0, code != -f[F_CODE1], (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2]))
    by_p = s.ss_t & (((s.suminfo >> 31) & 1) == p)
    vst = (s.st | jnp.where(by_p, ST_SST, 0) | jnp.where(by_p & (((s.suminfo >> 16) & Z_EXTRA) != 0), ST_SSXT, 0)
           | jnp.where((s.st & ST_LEAVE) == 0, ST_STAY, 0))
    m &= (f[F_ST] == 0) | jnp.where(f[F_ST] < 0, summon_matches(s.suminfo, f[F_ST]), (vst & f[F_ST]) != 0)
    lv = s.k_level
    nolv = (s.k_ctype & (TYPE_XYZ | TYPE_LINK)) != 0  # without Level: only an unbounded filter
    m &= jnp.where(nolv, (f[F_LV_LO] <= -BIG) & (f[F_LV_HI] >= BIG), (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI]))
    # IsCanBeSpecialSummoned: current conditions, and Link only in attack.
    m &= (f[F_SPSUM] == 0) | (special_allowed(bank, s)
                              & ~((f[F_SPSUM] == 2) & ((s.k_ctype & TYPE_LINK) != 0)))
    # Fusion / Ritual: SummonEffFilter and Ritual.Filter also require IsCanBeSpecialSummoned (fl_c is up to date here,
    # not necessarily when fz_c / rz_c are computed)
    nss = (s.fl_c & FL_NO_SS) == 0
    m &= (f[F_FUS] == 0) | (s.fz_c & nss)
    m &= (f[F_RIT] == 0) | ((((s.rz_c >> jnp.maximum(f[F_RIT] - 1, 0)) & 1) != 0) & nss)
    m &= (f[F_NODIS] == 0) | ((s.fl_c & FL_DISABLED) == 0)
    m &= (f[F_REASON] == 0) | reason_matches(s.reason, f[F_REASON])
    m &= ((f[F_ZX] & ZX_COLINK) == 0) | s.cl_c
    m &= ((f[F_ZX] & ZX_MAIN) == 0) | (s.seq < N_MAIN_ZONES)
    need_extra = (f[F_ZX] & ZX_EXTRA) != 0
    extra_space = lax.fori_loop(0, need_extra.astype(I32), lambda _, previous: ex_zones(s, p).any(), jnp.bool_(True))
    faceup_pend = (s.zone == Z_EXTRA) & ((ct & TYPE_PENDULUM) != 0) & ((s.pos & P_FACEUP) != 0)
    need_pend = need_extra & (m & faceup_pend).any()
    pend_space = lax.fori_loop(0, need_pend.astype(I32),
        lambda _, previous: pend_extra_zones(bank, s, p).any(), jnp.bool_(True))
    m &= jnp.where(faceup_pend, pend_space, extra_space)
    # ZX_FREE: Duel.GetMZoneCount(p, c) > 0: p has a free main zone, or c occupies one and frees it
    # ZX_FREEANY: same test without a controller condition on the card (ft>0 or c:GetSequence()<5)
    need_free = (f[F_ZX] & (ZX_FREE | ZX_FREEANY)) != 0
    free = lax.fori_loop(0, need_free.astype(I32), lambda _, previous: free_main(s, p) > 0, jnp.bool_(True))
    m &= ~need_free | free | ((s.zone == Z_MZONE) & (s.seq < N_MAIN_ZONES) & ((s.ctrl == p) | ((f[F_ZX] & ZX_FREEANY) != 0)))
    m &= ((f[F_ZX] & ZX_OPFU) == 0) | (s.ctrl == p) | ((s.pos & P_FACEUP) != 0)
    if targ:
        m &= ~untargetable(bank, s, h)
    # a Link monster has no DEF (IsDefense* false): only an unbounded filter
    nodf = (s.k_ctype & TYPE_LINK) != 0
    dok = jnp.where(nodf, (f[F_DEF_LO] <= -BIG) & (f[F_DEF_HI] >= BIG), (df >= f[F_DEF_LO]) & (df <= f[F_DEF_HI]))
    return m & (atk >= f[F_ATK_LO]) & (atk <= f[F_ATK_HI]) & dok


def untargetable(bank, s, h):
    """[N]: cards that the effect of card h cannot target (is_capable_be_effect_target: CANNOT_BE_EFFECT_TARGET;
    immunity does not prevent targeting, the effect simply does not apply at resolution)."""
    return (s.fl_c & FL_NO_ETARGET) != 0


def is_trap(bank, s, h):
    """The effect of card h is a Trap effect (IsTrapEffect)."""
    return (bank.ctype[s.cid[h]] & TYPE_TRAP) != 0


def stats(bank, s):
    """Current ATK / DEF, current base ATK (ygopro's card::get_attack / get_defense): the modifiers are applied in
    the order of ygopro's effect ids. A continuous effect gets an id when its card's effect is activated (en_t) and
    when its condition becomes true again (cav_t); a temporary modifier at its creation (tm_t). Outside the monster
    zone: base stats. O((KM + TM) x N). Also returns the updated (cav, cav_t), the clock and the KM overflow."""
    n = s.cid.shape[0]
    idx = jnp.arange(n)
    onf = s.zone == Z_MZONE
    faceup = (s.pos & P_FACEUP) != 0
    # continuous effects in place (effect::is_available before the condition): card face-up in its locations, resolved
    # Spell/Trap; a FIELD effect of a monster destroyed by battle no longer applies
    k = s.k_ekind                                          # [N, E]
    md = s.k_emod                                          # [N, E, MW]
    # a card placed by the Pendulum Summon in progress still has STATUS_SUMMONING: effects not yet enabled
    # (STATUS_EFFECT_ENABLED at step 27) and base ATK / DEF (card::get_attack)
    pst = (s.st & ST_PSTEP) != 0
    srcok = faceup & ~pst & ((s.zone == Z_MZONE) | ((s.zone == Z_SZONE) & ((s.st & ST_ON) != 0)))
    bdok = (md[..., M_SELF] != 0) | ((s.st & ST_BDES) == 0)[:, None]
    act = (k == K_CONT) & ((s.k_erng & s.zone[:, None]) != 0) & srcok[:, None] & bdok
    if bank.submat.shape[1]:
        act &= md[..., M_CODE] != MC_FUS_SUB  # passive property, no slot of stats()/KM
    flat = jnp.nonzero(act.reshape(-1), size=KM, fill_value=0)[0]  # card then effect order
    hs, es = flat // E, flat % E
    live = jnp.arange(KM) < act.sum()

    def one(h, e, ok):
        f = bank.filt[s.cid[h], e]
        m = bank.emod[s.cid[h], e]
        p = s.ctrl[h]
        cnt = filter_mask(bank, s, f[1], p, h, bdes=False).sum()  # filter 1: count (M_PER) or condition (M_CNT)
        cond = (((m[M_TP] == 0) | (s.tp == jnp.where(m[M_TP] == 1, p, 1 - p)))
                & ((m[M_PH] == 0) | ((m[M_PH] & s.phase) != 0))
                & ((m[M_CNT] == 0) | ((cnt >= m[M_CMIN]) & (cnt <= m[M_CMAX]))))
        ok &= cond
        raw = jnp.where(m[M_SELF] != 0, idx == jnp.where(m[M_SELF] == M_EQUIP, s.eq_t[h], h),
                        filter_mask(bank, s, f[0], p, h, bdes=False))
        tgt = raw & (faceup | (m[M_SELF] != 0))
        val = m[M_VAL] * jnp.where(m[M_PER] != 0, cnt, 1)
        return (tgt & onf & ok, jnp.where(ok, m[M_CODE], MC_NONE), val, m[M_SELF] == 1, cond, raw & ok, p,
                is_trap(bank, s, h))
    ctgt, ccode, cval, cself, ccond, craw, cp, ctrap = jax.vmap(one)(hs, es, live)
    ttgt = s.tm_m & (s.mv <= s.tm_t[:, None]) & onf
    # negations first (EFFECT_DISABLE): temporary ones, then continuous ones (Skill Drain) of non-negated cards,
    # which do not affect an immune monster; a negated card no longer applies its continuous effects
    dis_t = (ttgt & (s.tm_code == MC_DISABLE)[:, None]).any(0)
    # the immunity (SINGLE_RANGE) of a card that has just arrived or was negated does not count yet
    # (refresh_disable_status before STATUS_EFFECT_ENABLED, and is_available excludes a negated card): history
    prev_ok = ((s.fl_c & FL_ONF) != 0) & ((s.fl_c & FL_DISABLED) == 0)
    imm0 = (ctgt & ((ccode == MC_IMM_TRAP) & ~dis_t[hs] & prev_ok[hs])[:, None]).any(0)
    dis = dis_t | (ctgt & ((ccode == MC_DISABLE) & ~dis_t[hs])[:, None] & ~(ctrap[:, None] & imm0[None, :])).any(0)
    off = dis[hs] & (ccode != MC_DISABLE)
    ctgt &= ~off[:, None]
    craw &= ~off[:, None]
    ccode = jnp.where(off, MC_NONE, ccode)
    tm_m = s.tm_m & ~(s.tm_rd[:, None] & dis[None, :])  # RESET_DISABLE
    ttgt &= tm_m
    # a modifier set by a card on itself is suspended while it is negated (effect::is_available: owner == handler,
    # not EFFECT_FLAG_CANNOT_DISABLE)
    ttgt &= ~((jnp.arange(s.cid.shape[0])[None, :] == s.tm_src[:, None]) & dis[None, :]
              & (s.tm_code != MC_DISABLE)[:, None])
    # immunity: continuous Trap effects do not apply to immune monsters
    imm = (ctgt & (ccode == MC_IMM_TRAP)[:, None]).any(0)
    ctgt &= ~(ctrap[:, None] & imm[None, :] & (ccode != MC_IMM_TRAP)[:, None] & (ccode != MC_DISABLE)[:, None])
    craw &= ~(ctrap[:, None] & imm[None, :] & onf[None, :])
    fl = (jnp.where(dis, FL_DISABLED, 0) | jnp.where(onf & faceup, FL_ONF, 0)).astype(I32)
    for c, b in ((MC_NO_ATTACK, FL_NO_ATTACK), (MC_NO_BTARGET, FL_NO_BTARGET), (MC_NO_ETARGET, FL_NO_ETARGET),
                 (MC_IMM_TRAP, FL_IMM_TRAP), (MC_NO_DIRECT, FL_NO_DIRECT), (MC_INDES_EFF, FL_INDES_EFF)):
        # continuous and temporary (EFFECT_CANNOT_ATTACK set by a resolution, Silent Doom...)
        fl |= jnp.where((ctgt & (ccode == c)[:, None]).any(0) | (ttgt & (s.tm_code == c)[:, None]).any(0), b, 0)
    fl |= jnp.where((craw & (ccode == MC_NO_SS)[:, None]).any(0), FL_NO_SS, 0)  # targeted card, wherever it is
    fl |= jnp.where(ss_locked(bank, s), FL_NO_SS, 0)
    nt = ccode == MC_P_NO_TRAP
    pf = jnp.stack([(nt & ((cval & jnp.where(cp == q, 1, 2)) != 0)).any() for q in (0, 1)]) * PF_NO_TRAP
    # condition true again: new id (clock advanced one notch for all those of this computation)
    hw = jnp.where(live, hs, n)
    new = live & ccond & ~s.cav[hs, es]
    clk = s.clk + new.any()
    cav = s.cav.at[hw, es].set(ccond, mode="drop")
    cav_t = s.cav_t.at[jnp.where(new, hs, n), es].set(clk, mode="drop")
    # entries sorted like the effect ids: (targets, code, value [N], SINGLE type, SINGLE without SINGLE_RANGE)
    key = jnp.concatenate([jnp.maximum(s.en_t[hs], cav_t[hs, es]) * 4 + es, s.tm_t * 4 + 3])
    o = jnp.argsort(key)
    tgt = jnp.concatenate([ctgt, ttgt])[o]
    code = jnp.concatenate([ccode, s.tm_code])[o]
    val = jnp.concatenate([jnp.broadcast_to(cval[:, None], (KM, n)), s.tm_v])[o]
    ib = jnp.where(tgt & (code == MC_INDES_BATTLE)[:, None], val, BIG).min(0)
    single = jnp.concatenate([cself, jnp.ones(TM, bool)])[o]
    sgl = jnp.concatenate([jnp.zeros(KM, bool), jnp.ones(TM, bool)])[o]
    K = KM + TM

    jj = jnp.arange(K)[:, None]
    sg2 = sgl[:, None]

    def last(m):  # [K, N] -> [N]: index of the last true entry (-1: none)
        return jnp.where(m.any(0), K - 1 - jnp.argmax(m[::-1], axis=0), -1)

    def at(i):  # value of entry i for each card
        return val[jnp.maximum(i, 0), idx]

    def fold(base, upd, set_, fin, sbase):
        """Closed-form fold of card::get_attack (parallel over the entries, no sequential loop):
        temporary SET / SET_FINAL / SET_BASE fix the value (SET_BASE: the base); a FIELD SET resets the temporary
        UPDATEs to zero, a temporary SET_FINAL all UPDATEs; continuous SET_FINALs are applied last;
        continuous SET_BASEs before everything."""
        t = tgt
        is_ = lambda c: t & (code == c)[:, None]
        sb_s, sb_c = is_(sbase) & sg2, is_(sbase) & ~sg2
        fin_s, fin_c = is_(fin) & sg2, is_(fin) & ~sg2
        set_m = is_(set_)
        lc = last(sb_c)
        b0 = jnp.where(lc >= 0, at(lc), jnp.maximum(base, 0))
        lb = last(sb_s)
        b = jnp.where(lb >= 0, at(lb), b0)
        lx = last(set_m | fin_s | sb_s)
        x = jnp.where((lx < 0) | sb_s[jnp.maximum(lx, 0), idx], -1, at(lx))
        r_up = last((set_m & ~single[:, None]) | fin_s)
        r_upc = last(fin_s)
        u = is_(upd)
        up = jnp.where(u & sg2 & (jj > r_up), val, 0).sum(0)
        upc = jnp.where(u & ~sg2 & (jj > r_upc), val, 0).sum(0)
        ll = last(fin_c)
        cur = jnp.maximum(jnp.where(x < 0, b, x) + up + upc, 0)
        onx = onf & ~pst
        return jnp.where(onx, jnp.maximum(jnp.where(ll >= 0, at(ll), cur), 0), base), jnp.where(onx, b, base)
    atk, batk = fold(s.k_atk, MC_ATK, MC_SET_ATK, MC_FINAL_ATK, MC_BASE_ATK)
    df, _ = fold(s.k_df, MC_DEF, MC_SET_DEF, MC_FINAL_DEF, MC_BASE_DEF)
    df = jnp.where((s.k_ctype & TYPE_LINK) != 0, 0, df)  # card::get_defense: always 0 for a Link
    return (atk.astype(I32), df.astype(I32), batk.astype(I32), fl, ib.astype(I32), pf.astype(I32), cav, cav_t, clk,
            tm_m, act.sum() > KM)


def stats_key(s):
    """Entries of stats() that change during the game (excluding its own outputs): if they have not moved, refresh
    is useless (it is idempotent on an unchanged state)."""
    return (s.zone, s.pos, s.st, s.ctrl, s.phase, s.tp, s.turn, s.clk, s.tm_n, s.tm_m, s.reason, s.suminfo)


def changed(prev, s):
    """[...]: the entries of stats() changed between prev and s (per game if the states are batched)."""
    nb = s.done.shape
    d = jnp.zeros(nb, bool)
    for x, y in zip(stats_key(prev), stats_key(s)):
        d |= (x != y).reshape(nb + (-1,)).any(-1)
    return d


def cont_reads_stats(bank):
    """The bank has a continuous effect with a filter (target or count) that bounds ATK or DEF."""
    f = bank.filt[:, :, :2]
    bounded = ((f[..., F_ATK_LO] > -BIG) | (f[..., F_ATK_HI] < BIG) | (f[..., F_DEF_LO] > -BIG) | (f[..., F_DEF_HI] < BIG))
    return ((bank.ekind == K_CONT)[..., None] & bounded).any()


TB = 16  # distinct ATK / DEF thresholds kept per bound (beyond that: conservative test, any changed ATK / DEF)


def cont_bounds(bank):
    """Thresholds of the bank's continuous-effect filters: [4, TB] (ATK_LO, ATK_HI, DEF_LO, DEF_HI), padded with
    +-BIG (never crossed), and an overflow flag."""
    f = bank.filt[:, :, :2]
    cont = (bank.ekind == K_CONT)[..., None]
    out, over = [], jnp.bool_(False)
    for i, lo in ((F_ATK_LO, True), (F_ATK_HI, False), (F_DEF_LO, True), (F_DEF_HI, False)):
        v = f[..., i]
        ok = cont & ((v > -BIG) if lo else (v < BIG))
        u = jnp.where(ok, v, BIG if lo else -BIG).reshape(-1)
        u = jnp.sort(u) if lo else -jnp.sort(-u)
        first = jnp.concatenate([jnp.ones(1, bool), u[1:] != u[:-1]])      # distinct values first
        idx = jnp.nonzero(first, size=TB + 1, fill_value=u.shape[0] - 1)[0]
        w = u[idx]
        over |= (w[TB] != (BIG if lo else -BIG))
        out.append(w[:TB])
    return jnp.stack(out), over


def stats_moved(bank, a, b):
    """[...]: a cached ATK / DEF crossed a threshold of a continuous-effect filter between a and b (per game if the
    states are batched); the only case where the second pass can change a result."""
    nb = a.done.shape
    th, over = cont_bounds(bank)
    flat = lambda x: x.reshape(nb + (-1, 1))
    oa, na, od, nd = flat(a.atk_c), flat(b.atk_c), flat(a.df_c), flat(b.df_c)
    x = (((oa >= th[0]) != (na >= th[0])).any((-1, -2)) | ((oa <= th[1]) != (na <= th[1])).any((-1, -2))
         | ((od >= th[2]) != (nd >= th[2])).any((-1, -2)) | ((od <= th[3]) != (nd <= th[3])).any((-1, -2)))
    moved = ((a.atk_c != b.atk_c) | (a.df_c != b.df_c)).reshape(nb + (-1,)).any(-1)
    return jnp.where(over, moved, x)


# Second refresh pass. The filters of continuous effects read atk_c / df_c, the result of the previous refresh;
# ygopro evaluates them with the current ATK (get_attack): a bonus that has just applied (phase change, card
# arrived) already counts for "cannot attack if it has 1500 ATK or more". If the bank has such filters and
# the refresh changed an ATK / DEF, it is redone once (without Fusion / Ritual, like refresh_b). Only one extra
# pass: a circular dependency (ATK bonus filtered on ATK) is not iterated to the fixed point.
# A full second pass of stats() at every refresh cost 16% of the CPU throughput (procedural bank 1, B=64):
# in step_batch, it only covers the sub-batch of the games where an ATK / DEF moved.
def refresh2(bank, s):
    """refresh, then the second pass if needed (one game; under vmap, the cond computes both branches)."""
    s1 = refresh(bank, s)
    need = cont_reads_stats(bank) & stats_moved(bank, s, s1)
    return lax.cond(need, lambda x: refresh(bank, x, fus=False), lambda x: x, s1)


def refresh_if(bank, prev, s):
    """refresh only if the state changed since prev (one game; under vmap, the cond computes both branches)."""
    return lax.cond(changed(prev, s), lambda s: refresh2(bank, s), lambda s: s, s)


def refresh_b(bank, prev, s, frac=8):
    """Batch version: only the games whose state changed are recomputed, gathered in a sub-batch of B/frac
    (in practice a few % of the games change at each step); if there are more, the whole batch is recomputed."""
    nb = s.done.shape[0]
    d = changed(prev, s)
    k = max(1, nb // frac)

    def ref(s):  # refresh of a batch; fusion_ok only if a game of the batch has Fusion monsters in the Extra Deck
        fus = jax.vmap(lambda x: has_fusion(bank, x))(s).any()
        s = lax.cond(fus, lambda s: jax.vmap(lambda x: refresh_fus(bank, x))(s), lambda s: s, s)
        rit = jax.vmap(lambda x: has_ritual(bank, x))(s).any()
        s = lax.cond(rit, lambda s: jax.vmap(lambda x: refresh_rit(bank, x))(s), lambda s: s._replace(rz_c=s.rz_c * 0), s)
        return jax.vmap(lambda x: refresh(bank, x, fus=False))(s)

    def sub(s):  # the whole state is rewritten: rewriting only the refresh fields is slower (measured on CPU)
        idx = jnp.nonzero(d, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda x: x[jnp.minimum(idx, nb - 1)], s))
        return jax.tree.map(lambda x, y: x.at[idx].set(y, mode="drop"), s, part)
    s1 = lax.cond(d.sum() > k, ref, lambda s: lax.cond(d.any(), sub, lambda s: s, s), s)
    return lax.cond(cont_reads_stats(bank), lambda s1: _second_b(bank, s, s1, k), lambda s1: s1, s1)


def _second_b(bank, s, s1, k):
    """Second pass of refresh2 over a batch: only the games where an ATK / DEF moved (sub-batch of k)."""
    nb = s.done.shape[0]
    d = stats_moved(bank, s, s1)
    ref = lambda x: jax.vmap(lambda y: refresh(bank, y, fus=False))(x)

    def sub(x):
        idx = jnp.nonzero(d, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda y: y[jnp.minimum(idx, nb - 1)], x))
        return jax.tree.map(lambda y, z: y.at[idx].set(z, mode="drop"), x, part)
    return lax.cond(d.sum() > k, lambda x: _where(d, ref(x), x), lambda x: lax.cond(d.any(), sub, lambda x: x, x), s1)


SUBS = [[(q >> j) & 1 for j in range(FK)] for q in range(1 << FK)]  # subsets of the material slots


# fields allowed in a recipe
MAT_FIELDS = (F_TYPE, F_NOTTYPE, F_ATTR, F_RACE, F_SET, F_CODE1, F_CODE2, F_LV_LO, F_LV_HI)


def mat_mask(bank, s, f):
    """Lightweight material filter (only the MAT_FIELDS fields, without the location): cheaper than filter_mask."""
    ct, code, lv = s.k_ctype, s.k_code, s.k_level
    m = (f[F_TYPE] == 0) | ((ct & f[F_TYPE]) != 0)
    m &= (ct & f[F_NOTTYPE]) == 0
    m &= (f[F_ATTR] == 0) | ((s.k_attr & f[F_ATTR]) != 0)
    m &= (f[F_RACE] == 0) | ((s.k_race & f[F_RACE]) != 0)
    m &= (f[F_SET] == 0) | set_match_cs(s.k_setcode, f[F_SET])
    # F_CODE1 < 0: everything except the name -F_CODE1 (not c:IsCode(id))
    m &= jnp.where(f[F_CODE1] < 0, code != -f[F_CODE1], (f[F_CODE1] == 0) | (code == f[F_CODE1]) | (code == f[F_CODE2]))
    nolv = (ct & (TYPE_XYZ | TYPE_LINK)) != 0
    return m & jnp.where(nolv, (f[F_LV_LO] <= -BIG) & (f[F_LV_HI] >= BIG), (lv >= f[F_LV_LO]) & (lv <= f[F_LV_HI]))


def fus_cands(bank, s, c):
    """[FK, N]: cards usable for each material of Fusion monster c (hand and field of its owner, monsters, not
    destroyed by battle)."""
    p = s.owner[c]
    mats = (((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & (s.ctrl == p) & ((s.k_ctype & TYPE_MONSTER) != 0)
            & ((s.st & ST_BDES) == 0))
    k = bank.fmn[s.cid[c]]
    return jnp.stack([mat_mask(bank, s, bank.fmat[s.cid[c], j]) & mats & (j < k) for j in range(FK)])


def fusion_ok(bank, s):
    """[N] fz_c: Fusion monsters of the Extra Deck that can be Summoned (distinct materials in the hand and the field,
    and a zone for the Summon: a free zone, or a material on the field that will free one). Each card is summarized
    by the mask of the slots it can fill; Hall's test is done on the histogram of these masks
    (hand / field), in O(2^FK x 2^FK) once the histogram is built. O(EXK x (FK + 1) x N)."""
    if bank.submat.shape[1]:
        return fusion_sub_ok(bank, s)
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (s.k_fmn > 0)
    idx = jnp.nonzero(want, size=EXK, fill_value=0)[0]
    live = jnp.arange(EXK) < want.sum()
    nq = 1 << FK
    q = jnp.arange(nq)
    qsize = jnp.asarray([bin(x).count("1") for x in range(nq)])
    field = s.zone == Z_MZONE

    def one(c, ok):
        cand = fus_cands(bank, s, c)                               # [FK, N]
        bits = (cand * (1 << jnp.arange(FK))[:, None]).sum(0)      # [N] possible slots of each card
        hist = onehot(bits + nq * field, 2 * nq).sum(0, dtype=I32)  # [hand | field]
        hh, hf = hist[:nq], hist[nq:]
        k = bank.fmn[s.cid[c]]
        use = (1 << k) - 1
        inter = (q[:, None] & q[None, :]) != 0                     # [value, Q]: the card serves a slot of Q

        def hall_h(hh, hf):  # every Q subset of use has at least |Q| candidate cards
            cnt = ((hh + hf)[:, None] * inter).sum(0)              # [Q]
            return ((q & ~use) != 0) | (cnt >= qsize)
        ok_all = hall_h(hh, hf).all()
        # without a free zone: a slot j must be held by a field monster (the cards in hand lose j)
        with_field = jnp.zeros((), bool)
        for j in range(FK):
            hj = (onehot(q & ~(1 << j), nq) * hh[:, None]).sum(0, dtype=I32)
            with_field |= (j < k) & hall_h(hj, hf).all()
        return ok & ok_all & (ex_zones(s, s.owner[c]).any() | with_field)
    oks = jax.vmap(one)(idx, live)
    return set_unique(jnp.where(live, idx, n), oks, n), want.sum() > EXK


PERMS = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)]


def fus_picks(bank, s, f):
    """[N]: materials that can still be chosen for Fusion monster f, the ST_SEL cards being already chosen: there
    exists an assignment of the chosen materials and of the candidate to distinct slots, and the remaining slots
    have distinct materials among the other cards (Hall). All the permutations of the FK slots are
    tried (FK! = 24). O(FK! x 2^FK x N)."""
    if bank.submat.shape[1]:
        return fusion_sub_picks(bank, s, f)
    n = s.cid.shape[0]
    k = bank.fmn[s.cid[f]]
    cand = fus_cands(bank, s, f)                                   # [FK, N]
    sel = (s.st & ST_SEL) != 0
    t = sel.sum()
    chosen = jnp.nonzero(sel, size=FK, fill_value=0)[0]            # cards already chosen, in slot order
    free = cand & ~sel[None, :]
    sub = jnp.asarray(SUBS, bool)                                  # [Q, FK]
    field = s.zone == Z_MZONE

    def hallq(fr):  # [Q, N]: Hall for Q with the candidates fr, without card c
        unionm = (sub[:, :, None] & fr[None]).any(1)
        return (unionm.sum(-1)[:, None] - unionm) >= sub.sum(1)[:, None]
    hq = hallq(free)
    hqf = [hallq(free.at[j].set(free[j] & field)) for j in range(FK)]  # slot j taken on the field
    # no free zone for the Fusion monster: at least one material must come from the field
    room = ex_zones(s, s.owner[f]).any() | (sel & field).any()
    out = jnp.zeros(n, bool)
    for pm in PERMS:
        pm_ = jnp.asarray(pm)
        pos = jnp.arange(FK)
        ok_s = ((pos >= t) | cand[pm_, chosen]).all()              # the chosen ones go to the first slots
        slot_c = pm_[jnp.minimum(t, FK - 1)]
        rest = jnp.zeros(FK, bool).at[pm_].set((pos > t) & (pm_ < k))
        need_q = ~(sub & ~rest[None, :]).any(1)                    # Q is a subset of the remaining slots
        ok_h = (~need_q[:, None] | hq).all(0)
        ok_f = jnp.stack([rest[j] & (~need_q[:, None] | hqf[j]).all(0) for j in range(FK)]).any(0)
        okc = ok_s & (t < k) & (slot_c < k) & cand[slot_c] & jnp.where(room | field, ok_h, ok_f)
        out |= okc
    return out & ~sel



def fusion_sub_cards(bank, s, f):
    """SINGLE but not SINGLE_RANGE: also available face-down, except effective negation."""
    return ((s.k_submat[:, 0] & s.zone) != 0) & ((s.k_ctype & TYPE_MONSTER) != 0) & (s.ctrl == s.owner[f]) & \
           ((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & ((s.fl_c & FL_DISABLED) == 0) & ((s.st & ST_BDES) == 0)


def fusion_sub_ok(bank, s):
    from fusion import completions
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (s.k_fmn > 0)
    idx = jnp.nonzero(want, size=EXK, fill_value=0)[0]
    live = jnp.arange(EXK) < want.sum()
    def one(f):
        cand = fus_cands(bank, s, f)
        _, any_sub, _, field_sub = completions(cand, fusion_sub_cards(bank, s, f), s.zone == Z_MZONE,
                                              bank.fsub[s.cid[f]], jnp.array([-1], I32))
        use = (1 << bank.fmn[s.cid[f]]) - 1
        return jnp.where(ex_zones(s, s.owner[f]).any(), any_sub[use, 0], field_sub[use, 0])
    ok = jax.vmap(one)(idx) & live
    return set_unique(jnp.where(live, idx, n), ok, n), want.sum() > EXK


def fusion_sub_picks(bank, s, f):
    from fusion import completions
    n, k = s.cid.shape[0], bank.fmn[s.cid[f]]
    cand = fus_cands(bank, s, f)
    subs = fusion_sub_cards(bank, s, f)
    eligible = (bank.fsub[s.cid[f]] & (1 << jnp.arange(FK))) != 0
    sel = (s.st & ST_SEL) != 0
    t = sel.sum()
    chosen = jnp.nonzero(sel, size=FK, fill_value=0)[0]
    field = s.zone == Z_MZONE
    plain, enhanced, plain_field, enhanced_field = completions(cand & ~sel[None, :], subs & ~sel, field,
                                                               bank.fsub[s.cid[f]], jnp.arange(n))
    room = ex_zones(s, s.owner[f]).any() | (sel & field).any()
    def assignment(pm):
        pos = jnp.arange(FK)
        ordinary = cand[pm, chosen]
        alternative = eligible[pm] & subs[chosen]
        ok_s = ((pos >= t) | ordinary | alternative).all()
        used = ((pos < t) & ~ordinary).sum()
        slot_c = pm[jnp.minimum(t, FK - 1)]
        rest = jnp.where((pos > t) & (pm < k), 1 << pm, 0).sum()
        budget = used + (~cand[slot_c]).astype(I32)
        can = cand[slot_c] | (eligible[slot_c] & subs)
        complete = jnp.where(budget == 0, enhanced[rest], plain[rest])
        complete_field = jnp.where(budget == 0, enhanced_field[rest], plain_field[rest])
        return ok_s & (t < k) & (slot_c < k) & can & (budget <= 1) & jnp.where(room | field, complete, complete_field)
    return jax.vmap(assignment)(jnp.asarray(PERMS)).any(0) & ~sel


def ex_zones(s, p):
    """[7]: zones where p can Summon an Extra Deck monster (MR5 without Link: the 5 free main zones; extra
    zones 5 and 6 if p occupies neither and the opponent does not occupy the symmetric zone)."""
    me, op = zone_used(s, p, Z_MZONE), zone_used(s, 1 - p, Z_MZONE)
    emz = ~me[5] & ~me[6]
    return jnp.concatenate([~me[:5], jnp.stack([emz & ~op[6], emz & ~op[5]])])


def pend_extra_zones(bank, s, p):
    """Face-up Pendulum of the Extra Deck: designated main zones and one extra zone."""
    ez = ex_zones(s, p)
    return jnp.concatenate([ez[:N_MAIN_ZONES] & link_arrows(bank, s, p).any(0), ez[N_MAIN_ZONES:]])


def has_fusion(bank, s):
    """Fusion monsters are waiting in an Extra Deck (otherwise fusion_ok is useless)."""
    return ((s.zone == Z_EXTRA) & (s.k_fmn > 0)).any()


def refresh_fus(bank, s):
    fz, fover = fusion_ok(bank, s)
    return s._replace(fz_c=fz, ovf=s.ovf | jnp.where(fover, 4, 0).astype(I32))


def refresh_rit(bank, s):
    return s._replace(rz_c=ritual_ok(bank, s))


def rit_mats(bank, s, p):
    """[N]: Ritual materials of p (monsters of its hand and field, not destroyed by battle)."""
    return (((s.zone == Z_HAND) | (s.zone == Z_MZONE)) & (s.ctrl == p) & ((s.k_ctype & TYPE_MONSTER) != 0)
            & ((s.st & ST_BDES) == 0) & (s.k_level > 0) & ((s.k_ctype & (TYPE_XYZ | TYPE_LINK)) == 0))


def lv_hist(bank, s, mask):
    """Histograms [LVMAX + 1] of the Levels of the cards in mask: hand, field."""
    lv = jnp.clip(s.k_level, 0, LVMAX)
    field = s.zone == Z_MZONE
    oh = lv[None, :] == jnp.arange(LVMAX + 1)[:, None]  # one-hot (a scatter is slow under vmap)
    hh = (oh & (mask & ~field)[None, :]).sum(1, dtype=I32)
    hf = (oh & (mask & field)[None, :]).sum(1, dtype=I32)
    return hh, hf


def sum_ok(hh, hf, req, u, mu, has_f):
    """Can the cards already taken (sum u, smallest Level mu, one from the field if has_f) be completed with
    cards from the histograms hh (hand) / hf (field) into a valid set? Returns (Greater, Equal). Greater:
    sum >= req and sum - smallest < req; Equal: sum == req. With has_f < 0: a material must come from the field
    (no free zone). Reachable sums tracked as bit masks (A: without a field card, B: with), Levels
    from LVMAX down to 1: with the Levels >= v, the smallest one added is >= v. The counts are taken in packets
    1, 2, 4... (binary split). O(LVMAX x log SMAX), independent of N."""
    need_f = has_f < 0
    hasf0 = has_f > 0
    full = (1 << SMAX) - 1
    A, B = jnp.int32(1), jnp.int32(0)
    need = req - u
    fok = ~need_f | hasf0
    # empty T: what is taken already suffices
    g = (need <= 0) & (u - mu < req) & fok
    e = (need == 0) & fok
    sig = jnp.arange(SMAX)
    for L in range(LVMAX, 0, -1):
        cmax = SMAX // L
        nb = cmax.bit_length()  # packets 1, 2, ..., 2^(nb-1): cover 0..cmax
        ch, cf = jnp.minimum(hh[L], cmax), jnp.minimum(hf[L], cmax)
        for i in range(nb):
            k = jnp.clip(ch - (2 ** i - 1), 0, 2 ** i)  # packet i of the hand (0: nothing)
            A = A | ((A << (k * L)) & full)
            B = B | ((B << (k * L)) & full)
        for i in range(nb):
            k = jnp.clip(cf - (2 ** i - 1), 0, 2 ** i)  # packet i of the field: goes to B
            B = jnp.where(k > 0, B | (((A | B) << (k * L)) & full), B)
        reach = (jnp.where(need_f & ~hasf0, B, A | B) >> sig) & 1
        win = jnp.minimum(mu, L)                                   # the smallest of the set is min(mu, >= L)
        g |= ((reach != 0) & (sig >= need) & (sig < need + win) & (need > 0)).any()
        e |= ((reach != 0) & (sig == need) & (need > 0)).any()
    return g, e


def ritual_ok(bank, s):
    """[N] rz_c: Ritual monsters in hand that can be Ritual Summoned (materials: hand and field, without the monster
    itself; required Level: its Level). Bit 0: Greater; bit 1: Equal."""
    n = s.cid.shape[0]
    out = jnp.zeros(n, I32)
    lvc = jnp.clip(s.k_level, 0, LVMAX)
    for p in (0, 1):
        mats = rit_mats(bank, s, p)
        hh, hf = lv_hist(bank, s, mats)
        room = ex_zones(s, p)[:N_MAIN_ZONES].any()

        def cls(lv):  # the Ritual monster (Level lv, in hand) does not Tribute itself
            h2 = hh - (jnp.arange(LVMAX + 1) == lv)
            g, e = sum_ok(h2, hf, lv, 0, LVMAX + 1, jnp.where(room, 0, -1))
            return g.astype(I32) | (e.astype(I32) << 1)
        res = jax.vmap(cls)(jnp.arange(LVMAX + 1))
        mine = (s.zone == Z_HAND) & (s.ctrl == p) & ((s.k_ctype & RIT_MON) == RIT_MON)
        out = jnp.where(mine, res[lvc], out)
    return out


def rit_picks(bank, s, r, mode):
    """[N]: Tributes that can still be chosen for Ritual monster r: there exists a valid set that contains the
    cards already chosen (ST_SEL) and this one. Computed per class (Level, hand / field)."""
    p = s.ctrl[r]
    sel = (s.st & ST_SEL) != 0
    mats = rit_mats(bank, s, p) & (jnp.arange(s.cid.shape[0]) != r)
    lv = jnp.clip(s.k_level, 0, LVMAX)
    field = s.zone == Z_MZONE
    hh, hf = lv_hist(bank, s, mats & ~sel)
    u0 = jnp.where(sel, lv, 0).sum()
    mu0 = jnp.where(sel, lv, LVMAX + 1).min()
    f0 = (sel & field).any()
    room = ex_zones(s, p)[:N_MAIN_ZONES].any()
    req = bank.level[s.cid[r]]

    def cls(l, isf):
        one = jnp.arange(LVMAX + 1) == l
        h2 = hh - (one & ~isf)
        f2 = hf - (one & isf)
        hasf = f0 | isf
        g, e = sum_ok(h2, f2, req, u0 + l, jnp.minimum(mu0, l), jnp.where(room | hasf, 1, -1))
        return jnp.where(mode == RIT_GREATER, g, e)
    tab = jax.vmap(lambda isf: jax.vmap(lambda l: cls(l, isf))(jnp.arange(LVMAX + 1)))(jnp.array([False, True]))
    return mats & ~sel & tab[field.astype(I32), lv]


def rit_done(bank, s, r, mode):
    """The Ritual selection is complete (sum reached)."""
    sel = (s.st & ST_SEL) != 0
    return jnp.where(sel, jnp.clip(s.k_level, 0, LVMAX), 0).sum() >= bank.level[s.cid[r]]


def has_ritual(bank, s):
    return ((s.zone == Z_HAND) & ((s.k_ctype & RIT_MON) == RIT_MON)).any()


def syn_mats(bank, s, c):
    """[N], [N]: Tuners and non-Tuners usable for Synchro monster c (face-up monsters on its owner's field,
    of Level >= 1, neither Xyz nor Link; filters of the bank)."""
    ct, lv = s.k_ctype, s.k_level
    base = ((s.zone == Z_MZONE) & (s.ctrl == s.owner[c]) & ((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0)
            & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0) & (lv >= 1) & ((s.st & ST_BDES) == 0))
    tun = (ct & TYPE_TUNER) != 0
    f = bank.smat[s.cid[c]]
    return base & tun & mat_mask(bank, s, f[0]), base & ~tun & mat_mask(bank, s, f[1])


def reach(h, lo=0):
    """Bit mask of the reachable sums (< SMAX) with at least lo (0 or 1) cards of the Level histogram h
    [LVMAX + 1]. Counts taken in packets 1, 2, 4... O(LVMAX x log SMAX)."""
    full = (1 << SMAX) - 1
    A, B = jnp.int32(1), jnp.int32(0)  # A: no card; B: at least one
    for L in range(LVMAX, 0, -1):
        cmax = SMAX // L
        c = jnp.minimum(h[L], cmax)
        for i in range(cmax.bit_length()):
            k = jnp.clip(c - (2 ** i - 1), 0, 2 ** i)
            B = jnp.where(k > 0, B | (((A | B) << (k * L)) & full), B)
    return jnp.where(lo > 0, B, A | B)


def bit(x, i):
    return ((x >> jnp.clip(i, 0, 31)) & 1).astype(bool) & (i >= 0) & (i < SMAX)


def synchro_ok(bank, s):
    """[N] sz_c: Synchro monsters of the Extra Deck that can be Summoned now: a Tuner of Level t and
    non-Tuners (at least one) whose Levels sum to exactly L - t. Since the materials are on the field, a zone
    is always freed. O(EXK x (N + LVMAX log SMAX))."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (s.k_syn != 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    lvs = jnp.arange(LVMAX + 1)
    lv = jnp.clip(s.k_level, 0, LVMAX)

    def one(c, ok):
        t, nt = syn_mats(bank, s, c)
        oh = lv[None, :] == lvs[:, None]
        tl = (oh & t[None, :]).any(1)                              # [LVMAX + 1] Tuner Levels present
        b = reach((oh & nt[None, :]).sum(1, dtype=I32), 1)
        L = bank.level[s.cid[c]]
        return ok & (tl & jax.vmap(lambda l: bit(b, L - l))(lvs)).any()
    oks = jax.vmap(one)(idx, live)
    return set_unique(jnp.where(live, idx, n), oks, n)


def syn_picks(bank, s, c):
    """[N]: materials that can still be chosen for Synchro monster c: first the Tuner (non-Tuners remain that
    complete the sum), then the non-Tuners (the rest of the sum stays reachable)."""
    t, nt = syn_mats(bank, s, c)
    sel = (s.st & ST_SEL) != 0
    lv = jnp.clip(s.k_level, 0, LVMAX)
    lvs = jnp.arange(LVMAX + 1)
    oh = lv[None, :] == lvs[:, None]
    h = (oh & (nt & ~sel)[None, :]).sum(1, dtype=I32)
    L = bank.level[s.cid[c]]
    u = jnp.where(sel, lv, 0).sum()
    has_t = (sel & t).any()
    b1 = reach(h, 1)
    tun = jax.vmap(lambda l: bit(b1, L - l))(lvs)                  # Tuner of Level l
    ntv = jax.vmap(lambda l: bit(reach(h - (lvs == l), 0), L - u - l))(lvs)  # non-Tuner of Level l
    return jnp.where(has_t, nt & ~sel & ntv[lv], t & ~sel & tun[lv])


def syn_done(bank, s):
    sel = (s.st & ST_SEL) != 0
    return jnp.where(sel, jnp.clip(s.k_level, 0, LVMAX), 0).sum() >= bank.level[s.cid[s.pend]]


def has_synchro(bank, s):
    return ((s.zone == Z_EXTRA) & (s.k_syn != 0)).any()


def refresh_syn(bank, s):
    return s._replace(sz_c=synchro_ok(bank, s))


def xyz_mats(bank, s, c):
    """[N]: possible materials of Xyz c: face-up monsters on its owner's field, with Level equal to its
    Rank (neither Xyz nor Link: no Level), filter of the bank."""
    ct, lv = s.k_ctype, s.k_level
    return ((s.zone == Z_MZONE) & (s.ctrl == s.owner[c]) & ((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0)
            & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0) & (lv == bank.level[s.cid[c]]) & ((s.st & ST_BDES) == 0)
            & mat_mask(bank, s, bank.xmat[s.cid[c]]))


def xyz_ok(bank, s):
    """[N] xz_c: Xyz monsters of the Extra Deck that can be Summoned (at least xct possible materials). O(EXK x N)."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (s.k_xct > 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    oks = jax.vmap(lambda c, ok: ok & (xyz_mats(bank, s, c).sum() >= bank.xct[s.cid[c]]))(idx, live)
    return set_unique(jnp.where(live, idx, n), oks, n)


def has_xyz(bank, s):
    return ((s.zone == Z_EXTRA) & (s.k_xct > 0)).any()


def refresh_xyz(bank, s):
    return s._replace(xz_c=xyz_ok(bank, s))


SUB7 = [[(q >> z) & 1 == 1 for z in range(7)] for q in range(128)]  # subsets of the 7 monster zones
# 21 pairs of zones
PAIRS7 = ([i for i in range(7) for j in range(i + 1, 7)], [j for i in range(7) for j in range(i + 1, 7)])


def link_arrows(bank, s, p):
    """[N, 5]: main zones of p designated by the arrows of each Link monster on the field (card::get_linked_zone,
    field::get_linked_zone): its own (left / right from a main zone, down from an extra zone) and those
    of the opponent in an extra zone (top arrows). Arrows toward the extra zones do not count: they are
    always within the allowed zones."""
    mk = s.k_lmark
    on = (s.zone == Z_MZONE) & ((s.k_ctype & TYPE_LINK) != 0) & ((s.pos & P_FACEUP) != 0) & ((s.st & ST_PSTEP) == 0)
    q = s.seq[:, None]
    j = jnp.arange(5)[None, :]
    has = lambda b: ((mk & b) != 0)[:, None]
    e = 2 * (q - 5)  # extra zone 5: zones 0-2; 6: zones 2-4
    own = (((q < 5) & (((j == q - 1) & has(LM_L)) | ((j == q + 1) & has(LM_R))))
           | ((q >= 5) & (((j == e) & has(LM_BL)) | ((j == e + 1) & has(LM_B)) | ((j == e + 2) & has(LM_BR)))))
    o = 4 - e  # opponent's extra zone 5: top-left -> my zone 4, top -> 3, top-right -> 2; zone 6: 2, 1, 0
    opp = (q >= 5) & (((j == o) & has(LM_TL)) | ((j == o - 1) & has(LM_T)) | ((j == o - 2) & has(LM_TR)))
    return on[:, None] & jnp.where((s.ctrl == p)[:, None], own, opp)


def link_zones(bank, s, p, c):
    """[7]: zones where p can Summon Link monster c (get_rule_zone_fromex): free main zones designated by
    an arrow, the extra zones (ex_zones rule), and the other extra zone via Extra Link."""
    ez = ex_zones(s, p)
    e, x = extra_link(bank, s, p, jnp.zeros((1, 7), bool))
    ok = (arrows_at(bank.lmark[s.cid[c]], e[0]) & x[0]) != 0
    return jnp.concatenate([ez[:N_MAIN_ZONES] & link_arrows(bank, s, p).any(0),
                            ez[N_MAIN_ZONES:] | (ok & (jnp.arange(5, 7) == e[0]))])


def pend_info(bank, s, p):
    """Pendulum Summon of p (proc_pendulum.lua): (left Scale card, Summon possible, candidates from the
    hand, candidates from the Extra Deck, ft1, ft2, ft, free designated zones, usable extra zone). Candidates:
    monsters of the hand and face-up Pendulums of the Extra Deck with Level strictly between the Scales, that can be
    Special Summoned; those of the hand only with a free main zone (ft1 > 0), those of the Extra Deck only with an
    allowed zone (ft2 > 0: free main zones designated by an arrow, plus an extra zone). ft: usable zones
    (GetUsableMZoneCount). ft1 and ft2 are bounded by the number of candidates (Pendulum.Operation)."""
    ct = s.k_ctype
    pen = (ct & TYPE_PENDULUM) != 0
    sz = (s.zone == Z_SZONE) & (s.ctrl == p) & pen
    lm, rm = sz & (s.seq == 0), sz & (s.seq == 4)
    l, r = jnp.argmax(lm), jnp.argmax(rm)
    a, b = bank.lsc[s.cid[l]], bank.rsc[s.cid[r]]
    lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
    lv = s.k_level
    ok = ((lv > lo) & (lv < hi) & ((ct & TYPE_MONSTER) != 0) & ((ct & (TYPE_XYZ | TYPE_LINK)) == 0)
          & ((s.fl_c & FL_NO_SS) == 0) & ((s.k_erl == 0) | ((s.k_erl == 1) & s.proc)) & (s.ctrl == p))
    ez = ex_zones(s, p)
    free = ez[:N_MAIN_ZONES]
    lz = free & link_arrows(bank, s, p).any(0)
    emz = ez[N_MAIN_ZONES] | ez[N_MAIN_ZONES + 1]
    f1, f2 = free.sum(), lz.sum() + emz
    ch = ok & (s.zone == Z_HAND) & (f1 > 0)
    cx = ok & (s.zone == Z_EXTRA) & (s.pos == P_FU_DEF) & pen & (f2 > 0)
    can = lm.any() & rm.any() & ~s.ps_used & (ch | cx).any()
    return l, can, ch, cx, jnp.minimum(f1, ch.sum()), jnp.minimum(f2, cx.sum()), f1 + emz, lz, emz


def pend_zones(bank, s, p, c, lz):
    """[7]: zones for card c of the Pendulum Summon (operations.cpp, step 23). ct1 / ct2: free main zones
    / zones allowed from the Extra Deck, minus the remaining cards of the hand / of the Extra Deck (c included). A
    card from the hand avoids the designated zones if ct2 = 0; a card from the Extra Deck goes to an extra zone
    if ct1 = 0."""
    ez = ex_zones(s, p)
    free = ez[:N_MAIN_ZONES]
    rest = (s.st & ST_SEL) != 0
    ct1 = free.sum() - (rest & (s.zone == Z_HAND)).sum()
    ct2 = lz.sum() + (ez[N_MAIN_ZONES] | ez[N_MAIN_ZONES + 1]) - (rest & (s.zone == Z_EXTRA)).sum()
    no2 = jnp.zeros(2, bool)
    return jnp.where(s.zone[c] == Z_EXTRA, jnp.concatenate([lz & (ct1 != 0), ez[N_MAIN_ZONES:]]),
                     jnp.concatenate([free & ~((ct2 == 0) & lz), no2]))


def arrows_at(mk, q):
    """Zones designated by a Link monster with arrows mk in zone q (0-4 main, 5-6 extra zones), seen from its
    controller: 12-bit mask (0-4 its main zones, 5-6 its extra zones, 7-11 the opponent's main zones).
    card::get_linked_zone without the Spell/Trap zones."""
    h = lambda b: (mk & b) != 0
    bit = lambda i: jnp.left_shift(jnp.int32(1), jnp.clip(i, 0, 11))
    main, ex = q < 5, q >= 5
    e2 = 2 * (q - 5)
    return (jnp.where(main & (q > 0) & h(LM_L), bit(q - 1), 0) | jnp.where(main & (q < 4) & h(LM_R), bit(q + 1), 0)
            | jnp.where(((q == 0) & h(LM_TR)) | ((q == 1) & h(LM_T)) | ((q == 2) & h(LM_TL)), 1 << 5, 0)
            | jnp.where(((q == 2) & h(LM_TR)) | ((q == 3) & h(LM_T)) | ((q == 4) & h(LM_TL)), 1 << 6, 0)
            | jnp.where(ex & h(LM_BL), bit(e2), 0) | jnp.where(ex & h(LM_B), bit(e2 + 1), 0)
            | jnp.where(ex & h(LM_BR), bit(e2 + 2), 0) | jnp.where(ex & h(LM_TL), bit(11 - e2), 0)
            | jnp.where(ex & h(LM_T), bit(10 - e2), 0) | jnp.where(ex & h(LM_TR), bit(9 - e2), 0)).astype(I32)


def _swap12(m):
    """12-zone mask seen from the opponent: main zones swapped, extra zones 5 and 6 swapped."""
    return ((m & 0x1F) << 7) | ((m >> 7) & 0x1F) | (((m >> 5) & 1) << 6) | (((m >> 6) & 1) << 5)


def link_cells(bank, s, p):
    """Face-up Link monsters on the field over the 12 zones seen from p (see arrows_at): zone of each card [N]
    (12: none), zones designated by each zone [12], zones held by a Link [12], co-linked neighbors of each zone [12] (masks)."""
    on = (s.zone == Z_MZONE) & ((s.k_ctype & TYPE_LINK) != 0) & ((s.pos & P_FACEUP) != 0) & ((s.st & ST_PSTEP) == 0)
    mine = s.ctrl == p
    q = s.seq
    m = arrows_at(s.k_lmark, q)
    m = jnp.where(mine, m, _swap12(m))
    cell = jnp.where(mine, q, jnp.where(q < 5, q + 7, 11 - q))
    cell = jnp.where(on, cell, 12)
    oc = onehot(cell, 12)                                                # [N, 12] (cell 12: none)
    pt = (oc * m[:, None]).sum(0, dtype=I32)                             # [12] zones designated by each zone
    lk = oc.any(0)                                                       # zones held by a face-up Link
    j = jnp.arange(12)
    mut = (((pt[:, None] >> j[None, :]) & 1) != 0) & (((pt[None, :] >> j[:, None]) & 1) != 0)
    M = jnp.where(mut, 1 << j[None, :], 0).sum(1).astype(I32)          # [12] co-linked neighbors
    return cell, pt, lk, M


def colinked(bank, s):
    """[N]: Link monsters co-linked with at least one other (card::get_mutual_linked_cards non-empty). O(N + 144)."""
    cell, _, _, M = link_cells(bank, s, 0)
    return (cell < 12) & (M[jnp.clip(cell, 0, 11)] != 0)


def extra_link(bank, s, p, sub):
    """Extra Link (field::get_spsummonable_count_fromex_rule4, check_extra_link, card::is_extra_link_state), for
    each subset sub [S, 7] of p's zones whose monsters leave (materials). If p occupies a single extra
    zone o and the other zone e is free (neither for p nor for the opponent), a Link monster placed in e is allowed
    there when its co-link component contains o. Without this card, the component of o is fixed; the card joins it
    through a co-linked neighbor. Returns (e [S]: 5 or 6, X [S]: mask of the zones of the component of o that
    designate e; 0 if the rule does not apply); card c is allowed in e if arrows_at(lmark[c], e) & X != 0. 12-bit
    zones seen from p (see arrows_at). O(N + S x 12 x 11)."""
    _, pt, lk, M = link_cells(bank, s, p)
    j = jnp.arange(12)
    me, op = zone_used(s, p, Z_MZONE)[:7], zone_used(s, 1 - p, Z_MZONE)[:7]
    left = me[None, :] & ~sub                                             # [S, 7]
    l5, l6 = left[:, 5], left[:, 6]
    pre = (l5 & ~l6 & ~op[5]) | (l6 & ~l5 & ~op[6])                      # the opponent's symmetric extra zone is free
    o = jnp.where(l5, 5, 6)
    e = 11 - o
    lkb = jnp.where(lk, 1 << j, 0).sum().astype(I32)
    subb = jnp.where(sub, 1 << jnp.arange(7), 0).sum(1).astype(I32)
    alive = lkb & ~subb                                                   # [S]
    # component of o among the living zones: 11 unrolled propagation rounds (elementary operations on [S],
    # fusable; a fori_loop forces a sequential loop, an integer matrix product is slow on CPU)
    reach = (1 << o) & alive
    for _ in range(11):
        for a in range(12):
            reach = reach | jnp.where(((reach >> a) & 1) != 0, M[a], 0)
        reach = reach & alive
    back = jnp.where(((pt[None, :] >> e[:, None]) & 1) != 0, 1 << j[None, :], 0).sum(1).astype(I32)  # [S] designate e
    return e.astype(I32), jnp.where(pre, reach & back & alive, 0).astype(I32)


def link_table(bank, s, p):
    """Link materials of p, by zone of its monster zone and by subset of the 7 zones (128). Returns: zones -> cards
    [7, N]; number of cards [128]; possible sums [128] (bit mask: each monster counts 1, a Link with Link Rating
    L > 1 counts 1 or L, Link.GetLinkCount); contains a Link [128]; subset of face-up monsters, and a zone
    remains for the Link monster once these materials are gone (GetLocationCountFromEx with the materials: neither
    their zones nor their arrows) [128]; Extra Link per subset (extra_link). O(7 N + 128 x 7 x 5 + 128 x 12 x 11)."""
    ct = s.k_ctype
    mine = (s.zone == Z_MZONE) & (s.ctrl == p)
    oh = mine[None, :] & (s.seq[None, :] == jnp.arange(7)[:, None])      # [7, N]
    used = oh.any(1)
    ok = (oh & (((s.pos & P_FACEUP) != 0) & ((ct & TYPE_MONSTER) != 0))[None, :]).any(1)
    islk = (oh & ((ct & TYPE_LINK) != 0)[None, :]).any(1)
    r = jnp.where(oh, s.k_level[None, :], 0).sum(1)
    arw = link_arrows(bank, s, p)
    own = (oh[:, :, None] & arw[None, :, :]).any(1)                       # [7, 5] arrows of the monster of each zone
    opp = (~mine[:, None] & arw).any(0)                                  # [5]
    sub = jnp.asarray(SUB7)                                               # [128, 7]
    lk = opp[None, :] | (~sub[:, :, None] & own[None]).any(1)             # [128, 5]
    left = used[None, :] & ~sub
    opu = zone_used(s, 1 - p, Z_MZONE)
    emz = ~left[:, 5] & ~left[:, 6] & (~opu[6] | ~opu[5])
    zok = (lk & ~left[:, :N_MAIN_ZONES]).any(1) | emz
    m = jnp.ones(128, I32)
    for z in range(7):
        alt = jnp.where(islk[z] & (r[z] > 1), m << jnp.clip(r[z], 0, 15), 0)
        m = jnp.where(sub[:, z], ((m << 1) | alt) & 0xFFFF, m)
    hasl = (sub & islk[None, :]).any(1)
    base = ~(sub & ~ok[None, :]).any(1)
    ee, ex = extra_link(bank, s, p, sub)
    # two materials with the same name (different names required: ldiff)
    cc = jnp.where(oh, s.k_code[None, :], 0).sum(1)              # [7]
    pi, pj = jnp.asarray(PAIRS7[0]), jnp.asarray(PAIRS7[1])
    same = used[pi] & used[pj] & (cc[pi] == cc[pj])                       # [21]
    dup = (sub[:, pi] & sub[:, pj] & same[None, :]).any(1)
    return oh, sub.sum(1), m, hasl, zok & base, ee, jnp.where(base, ex, 0), dup


def link_valid(bank, s, c, tab):
    """[128]: valid material sets for Link monster c (Link.CheckGoal): filter, number of materials,
    sum equal to the Link Rating, a Link among them if requested, a zone for c."""
    oh, cnt, m, hasl, zok, ee, ex, dup = tab
    k = s.cid[c]
    mm = (oh & mat_mask(bank, s, bank.lmat[k])[None, :]).any(1)          # [7]
    sub = jnp.asarray(SUB7)
    zok = zok | ((arrows_at(bank.lmark[k], ee) & ex) != 0)               # Extra Link
    return (zok & ~(sub & ~mm[None, :]).any(1) & (cnt >= bank.lmin[k]) & (cnt <= bank.lmax[k])
            & (((m >> jnp.clip(bank.level[k], 0, 15)) & 1) != 0) & ((bank.lneed[k] == 0) | hasl)
            & ((bank.ldiff[k] == 0) | ~dup))


def link_ok(bank, s):
    """[N] lz_c: Link monsters of the deciding player's Extra Deck that can be Summoned now. O(EXP x (N + 128 x 7))."""
    n = s.cid.shape[0]
    want = (s.zone == Z_EXTRA) & (s.k_lmax > 0) & (s.owner == s.dec_player)
    idx = jnp.nonzero(want, size=EXP, fill_value=0)[0]
    live = jnp.arange(EXP) < want.sum()
    tab = link_table(bank, s, s.dec_player)

    def one(c, ok):
        return ok & link_valid(bank, s, c, tab).any()
    oks = jax.vmap(one)(idx, live)
    return set_unique(jnp.where(live, idx, n), oks, n)


def link_picks(bank, s, c):
    """([N], bool): materials that can still be chosen for Link monster c (a valid set contains the choices
    made and this candidate: Link.CheckRecursive); the selection made is itself valid ("finish")."""
    tab = link_table(bank, s, s.owner[c])
    v = link_valid(bank, s, c, tab)
    oh = tab[0]
    sel = (oh & ((s.st & ST_SEL) != 0)[None, :]).any(1)                  # [7]
    bits = 1 << jnp.arange(7, dtype=I32)
    smask = jnp.where(sel, bits, 0).sum()
    need = smask | bits
    q = jnp.arange(128, dtype=I32)
    can = ~sel & (v[None, :] & ((q[None, :] & need[:, None]) == need[:, None])).any(1)
    return (oh & can[:, None]).any(0), v[smask]


def has_link(bank, s):
    return ((s.zone == Z_EXTRA) & (s.k_lmax > 0)).any()


def refresh_link(bank, s):
    return s._replace(lz_c=link_ok(bank, s))


def refresh_ex(bank, s):
    """Caches of procedure Summons from the Extra Deck (Synchro, Xyz, Link): only legal reads them, for the
    deciding player and in a free decision (D_IDLE). They are therefore only computed at decision points (end of step /
    step_batch, new_game), only in D_IDLE and for the deciding player's Extra Deck (false elsewhere)."""
    idle = (s.dec == D_IDLE) & ~s.done
    s = lax.cond(idle & has_synchro(bank, s), lambda s: refresh_syn(bank, s), lambda s: s._replace(sz_c=s.sz_c & False), s)
    s = lax.cond(idle & has_xyz(bank, s), lambda s: refresh_xyz(bank, s), lambda s: s._replace(xz_c=s.xz_c & False), s)
    return lax.cond(idle & has_link(bank, s), lambda s: refresh_link(bank, s), lambda s: s._replace(lz_c=s.lz_c & False), s)


def refresh_ex_b(bank, s, mask, frac=8):
    """refresh_ex at the batch level, for the games in mask only (gathered in a sub-batch of B/frac if there are
    few), each computation only if one of them needs it."""
    nb = s.done.shape[0]
    k = max(1, nb // frac)

    def ref(s, m):
        for has, fn, fld in ((has_synchro, refresh_syn, "sz_c"), (has_xyz, refresh_xyz, "xz_c"), (has_link, refresh_link, "lz_c")):
            h = jax.vmap(lambda x: has(bank, x))(s) & (s.dec == D_IDLE) & ~s.done
            s = lax.cond((h & m).any(), lambda s: _where(h, jax.vmap(lambda x: fn(bank, x))(s), s), lambda s: s, s)
            s = s._replace(**{fld: getattr(s, fld) & h[:, None]})
        return s

    def sub(s):
        idx = jnp.nonzero(mask, size=k, fill_value=nb)[0]
        part = ref(jax.tree.map(lambda x: x[jnp.minimum(idx, nb - 1)], s), idx < nb)
        return jax.tree.map(lambda x, y: x.at[idx].set(y, mode="drop"), s, part)
    full = lambda s: _where(mask, ref(s, mask), s)
    return lax.cond(mask.sum() > k, full, lambda s: lax.cond(mask.any(), sub, lambda s: s, s), s)


def refresh(bank, s, fus=True):
    """Recomputes the caches (stats, flags, achievable Fusions; not those of refresh_ex). fus=False: without fz_c
    (refresh_b does it at the batch level, only if a game has Fusion monsters in its Extra Deck)."""
    s = drop_overlays(drop_equips(bank, s))
    s = detach_fix(s)
    if fus:
        s = lax.cond(has_fusion(bank, s), lambda s: refresh_fus(bank, s), lambda s: s, s)
        s = lax.cond(has_ritual(bank, s), lambda s: refresh_rit(bank, s), lambda s: s._replace(rz_c=s.rz_c * 0), s)
    s = s._replace(cl_c=colinked(bank, s))
    atk, df, batk, fl, ib, pf, cav, cav_t, clk, tm_m, over = stats(bank, s)
    return s._replace(atk_c=atk, df_c=df, batk_c=batk, fl_c=fl, ib_c=ib, pf_c=pf, cav=cav, cav_t=cav_t, clk=clk, tm_m=tm_m,
                      ovf=s.ovf | jnp.where(over, 1, 0).astype(I32))


def _pure(bank, s, reg, ok, h, e, p, k, ins, count_asserts=False):
    """Side-effect-free instructions (queries and conditions), shared by test mode and execution.
    k: index of the effect's chain link (the link it responds to is k-1).
    In test mode, CHOOSE only checks that there are enough candidates (the register receives all the candidates)."""
    op, a, b, c = ins[0], ins[1], ins[2], ins[3]
    n = s.cid.shape[0]
    ra, rb = jnp.clip(a, 0, R - 1), jnp.clip(b, 0, R - 1)
    lk = jnp.clip(k - 1, 0, CH - 1)
    lc = s.cid[jnp.maximum(s.lk_src[lk], 0)]
    # each branch returns (row [n] written into reg[a] if wr, condition): under vmap, the switch computes all
    # the branches, so we avoid having each of them return the whole register
    none = jnp.zeros(n, bool)
    put = lambda row: (row, jnp.bool_(True), jnp.bool_(True))
    cond = lambda x: (none, jnp.bool_(False), x)
    branches = [lambda: cond(jnp.bool_(True))] * N_OPS
    cw = c & 7
    within = jnp.where(cw > 0, reg[jnp.clip(cw - 1, 0, R - 1)], True)  # c & 7 > 0: intersection with the register
    # c & 8: selection with targeting
    # Queries / Group:Filter include the monsters destroyed by battle still on the field.
    # Only targeting excludes them here (field::filter_matching_card, is_capable_be_effect_target).
    def filter_():
        row = filter_mask(bank, s, bank.filt[s.cid[h], e, jnp.clip(b, 0, NF - 1)], p, h, bdes=False)
        row = (jnp.where((c & (1 << 21)) != 0, ~row, row)  # c & (1 << 21): complement of the filter
                                      & within & (((c & 8) == 0) | ((s.st & ST_BDES) == 0))
                                      & ~(((c & 8) != 0) & untargetable(bank, s, h)))
        row = jnp.where((c & (1 << 20)) != 0, within, row)  # full copy of the register
        row &= ((c & (1 << 22)) == 0) | (s.mv <= s.lk_tgt_t[jnp.clip(k, 0, CH - 1)])
        if count_asserts:
            count = row.sum()
            minimum, maximum = (c >> 4) & 255, (c >> 12) & 255
            return row, jnp.bool_(True), (count >= minimum) & ((maximum == 0) | (count < maximum))
        return put(row)
    branches[OP_FILTER] = filter_
    branches[OP_SELF] = lambda: put((jnp.arange(n) == h) & ((b == 0) | (((s.lk_flags[k] & CF_SRC_UNRELATED) == 0) & (s.mv[h] <= s.lk_t[k]))))
    def req():
        related = (c & 2) != 0
        count = (reg[ra] & (~related | (s.mv <= s.lk_tgt_t[jnp.clip(k, 0, CH - 1)]))).sum()
        return cond(jnp.where((c & 1) == 0, count >= b, count <= b))
    branches[OP_REQ] = req
    branches[OP_REQ_FREE] = lambda: cond(free_main(s, rel(p, a), jnp.where(b == Z_SZONE, Z_SZONE, Z_MZONE)) >= c)

    def minatk():
        original = (c & 2) != 0
        candidates = jnp.where(original, s.lk_reg[jnp.clip(k, 0, CH - 1), rb], reg[rb])
        value = jnp.where((c & 1) != 0, s.uid, s.atk_c)
        rank = jnp.where(candidates, value, BIG)
        return put(candidates & (rank == rank.min()))
    branches[OP_MINATK] = minatk
    branches[OP_CHOOSE] = lambda: (reg[rb], jnp.bool_(True), reg[rb].sum() >= (c & 15))
    lkind = bank.ekind[lc, s.lk_eff[lk]]
    lfield = (s.zone[jnp.maximum(s.lk_src[lk], 0)] & (Z_MZONE | Z_SZONE)) != 0
    lct = jnp.where((bank.ctype[lc] & TYPE_PENDULUM) != 0, P_SPELL, bank.ctype[lc])  # activation of a Pendulum: Spell
    branches[OP_COND_CHAIN] = lambda: cond((k > 0) & jnp.where(
        b == 0, ((s.lk_flags[lk] & CF_ACT) != 0) & ((lct & a) != 0),
        lfield & ((s.lk_flags[lk] & CF_ACT) == 0)))
    branches[OP_CHAINCARD] = lambda: put((jnp.arange(n) == s.lk_src[lk]) & (k > 0)
                                         & ((s.zone & jnp.where(b == 0, Z_SZONE, b)) != 0))
    branches[OP_PAY_LP] = lambda: cond(s.lp[p] >= a)
    evm = ((a >> jnp.arange(NEV)) & 1) != 0
    evb = ((b >> jnp.arange(NEV)) & 1) != 0
    branches[OP_EVENT] = lambda: put((jnp.where(c == 1, s.teg, s.ev_g) & evb[:, None]).any(axis=0))
    branches[OP_COND_TURN] = lambda: cond(s.tp == rel(p, a))
    branches[OP_COND_EVP] = lambda: cond((evm & s.ev_g.any(axis=1) & (s.ev_p == rel(p, b))).any())
    branches[OP_COND_SELF] = lambda: cond(((a == 0) | ((s.zone[h] & a) != 0)) & ((b == 0) | ((s.pzone[h] & b) != 0))
                                          & ((c == 0) | jnp.where(c < 0,
                                                jnp.where((-c & (1 << 30)) != 0, reason_matches(s.reason[h], -c & 0x1fffffff),
                                                          summon_matches(s.suminfo[h], c)), (s.st[h] & c) != 0)))
    branches[OP_COND_LABEL] = lambda: cond(s.ch_lab != 0)
    branches[OP_COND_PHASE] = lambda: cond((s.phase & a) != 0)
    branches[OP_COND_ATK] = lambda: cond(jnp.where(a == 0, s.atk, s.atk_tgt) == h)  # the card attacks / is attacked
    branches[OP_ATTACKER] = lambda: put((jnp.arange(n) == s.atk) & (s.atk >= 0) & ((s.st & ST_BDES) == 0)
                                        & ~((b != 0) & untargetable(bank, s, h)))  # b: targeting
    row, wr, c_ok = _switch(op, [lambda _, f=f: f() for f in branches], None)
    return jnp.where(wr, reg.at[ra].set(row), reg), ok & c_ok


def test_ok(bank, s, h, e, p, enabled=None, condition_only=False):
    """Test mode (chk==0 of the scripts): cost + target executable by p with card h, effect e?"""
    n = s.cid.shape[0]
    progs = bank.tprog[s.cid[h], e]

    initial = (jnp.zeros((R, n), bool), jnp.bool_(True))
    length = (progs[:, 0] != OP_END).sum()
    length = jnp.where(condition_only, bank.econd[s.cid[h], e], length)
    if enabled is not None:
        length = jnp.where(enabled, length, 0)
    def body(i, carry):
        reg, ok = carry
        return _pure(bank, s, reg, ok, h, e, p, s.n_chain, progs[i], count_asserts=True)
    _, ok = lax.fori_loop(0, length, body, initial)
    return ok


def src_retarget(s, target, sel):
    """Duel.SetTargetCard recreates the relation of each target with the effect: a source sent as a cost and then
    taken as a target (Ignition "send this card to the GY; target 1 monster in your GY") stays linked to the effect
    (IsRelateToEffect). Its activation start clock is realigned (CF_SRC_UNRELATED)."""
    src = jnp.maximum(s.ch_src, 0)
    hit = target & (s.ch_src >= 0) & sel[src]
    return jnp.where(hit, s.mv[src], s.ch_src_mv)


def sel_finish(bank, s):
    """End of a selection during an effect (D_SELECT): the chosen cards go into the instruction's register."""
    sel = (s.st & ST_SEL) != 0
    ins = current_ins(bank, s)
    dst = jnp.clip(ins[1], 0, R - 1)
    target = (s.pg == PG_TARGET) & (ins[0] == OP_CHOOSE) & ((ins[3] & (1 << 10)) != 0)
    return s._replace(ch_tgt_t=jnp.where(target, s.clk, s.ch_tgt_t), ch_src_mv=src_retarget(s, target, sel),
                      ch_targets=s.ch_targets | (target & ((ins[3] & (1 << 11)) == 0) & sel),
                      reg=s.reg.at[dst].set(sel), st=s.st & ~ST_SEL, picks=_i(0), pc=s.pc + 1,
                      dec=_i(D_RUN), fsel=jnp.bool_(False), rsel=_i(0))


def sel_var(s):
    """Variable-size selection (CHOOSE, bit 9) in progress: A_PASS ends it as soon as smin cards are taken."""
    return (s.dec == D_SELECT) & ~s.fsel & (s.rsel == 0) & (s.smin < s.picks + s.need)


def current_ins(bank, s):
    return bank.prog[s.cid[s.ch_src], s.ch_eff, s.pg, s.pc]


def block_end(bank, s):
    """End of the block (OP_BLOCK) that contains the current instruction, or -1: a false test jumps there instead of
    stopping."""
    prog = bank.prog[s.cid[s.ch_src], s.ch_eff, s.pg]
    i = jnp.arange(prog.shape[0])
    m = (prog[:, 0] == OP_BLOCK) & (i < s.pc) & (prog[:, 1] > s.pc)
    return jnp.where(m.any(), jnp.max(jnp.where(m, prog[:, 1], 0)), -1).astype(I32)


def use_count(bank, s, c, e, p, enabled=True):
    """Consumes the limit of an activation or a procedure, without building a chain link."""
    opt = bank.eopt[s.cid[c], e]
    oath = bank.erefund[s.cid[c], e] if bank.erefund.shape[1] else 0
    key = opt_key(s.cid[c], opt, oath)
    def add(_, table):
        return table.at[p].set(name_add(table[p], key))
    counts = lax.fori_loop(0, jnp.where(enabled & (key >= 0), 1, 0), add, s.opt_cid)
    bits = jnp.where(opt == OPT_CARD, ST_CNT << e, jnp.where(opt == OPT_CARD_NR, ST_CNTN << e, 0))
    return s._replace(opt_cid=counts, st=s.st.at[c].set(s.st[c] | jnp.where(enabled, bits, 0)))


def proc_ready(s, card):
    """Procedure costs paid: placement, without a chain link or persistent group register."""
    return s._replace(pend=card, pend_set=jnp.bool_(False), picks=_i(0), psel=_i(0), need=_i(0),
                      zctx=_i(ZC_PROC), dec=_i(D_ZONE), ch_src=_i(-1), pc=_i(0), pg=_i(0),
                      ch_lab=_i(0), reg=jnp.zeros_like(s.reg))


def begin_act(bank, s, c, e, enabled=True):
    """Activation of effect e of c by the deciding player: cost then target, then put on the chain."""
    n = s.cid.shape[0]
    stay = TYPE_CONTINUOUS | TYPE_PENDULUM | TYPE_EQUIP  # stay on the field after resolution
    leave = (bank.ekind[s.cid[c], e] == K_ACTIVATE) & ((bank.ctype[s.cid[c]] & stay) == 0)
    p = s.dec_player
    s = use_count(bank, s, c, e, p, enabled=enabled)
    # AddChain: a field card under a temporary negation (Forbidden Chalice, Effect Veiler:
    # EFFECT_DISABLE_EFFECT) has its chain link negated, even if it later leaves the field (before the cost)
    tdis = (s.tm_m[:, c] & (s.mv[c] <= s.tm_t) & (s.tm_code == MC_DISABLE)).any() & (s.zone[c] == Z_MZONE)
    card_activation = bank.ekind[s.cid[c], e] == K_ACTIVATE
    continuous = (s.zone[c] == Z_SZONE) & ((bank.ctype[s.cid[c]] & stay) != 0)
    flags = jnp.where(card_activation, CF_ACT, 0) | jnp.where(continuous, CF_CONT, 0)
    s = s._replace(st=s.st.at[c].set(jnp.where(leave, s.st[c] | ST_LEAVE, s.st[c])), ch_dis=tdis, ch_lab=_i(0), ch_flags=flags,
                      ch_src=_i(c), ch_src_mv=s.mv[c], ch_tgt_t=s.clk, ch_targets=jnp.zeros_like(s.ch_targets), ch_eff=_i(e), ch_p=s.dec_player, ch_k=s.n_chain, pg=_i(PG_COST), pc=_i(0),
                      reg=jnp.zeros((R, n), bool), dec=_i(D_RUN), pend=_i(-1))
    if bank.einline.shape[1]:
        alternate = bank.einline[s.cid[c], e] - 1
        candidate = enabled & card_activation & (alternate >= 0) & (s.dstep == DS_NONE)
        available = act_ok(bank, s, c, jnp.maximum(alternate, 0), p, enabled=candidate)
        s = s._replace(dec=jnp.where(candidate & available, D_YESNO, s.dec),
                       zctx=jnp.where(candidate & available, ZC_INLINE, ZC_ACT))
    return s


def push_link(bank, s):
    s = brk(adjust(s))  # ygopro: break_effect after cost and target (AddChain step 8)
    k, src = s.n_chain, s.ch_src
    s = s._replace(lk_src=s.lk_src.at[k].set(s.ch_src), lk_eff=s.lk_eff.at[k].set(s.ch_eff),
                   lk_p=s.lk_p.at[k].set(s.ch_p), lk_reg=s.lk_reg.at[k].set(s.reg),
                   lk_targets=s.lk_targets.at[k].set(s.ch_targets),
                   lk_neg=s.lk_neg.at[k].set(s.ch_dis), lk_t=s.lk_t.at[k].set(s.clk), lk_tgt_t=s.lk_tgt_t.at[k].set(s.ch_tgt_t), lk_lab=s.lk_lab.at[k].set(s.ch_lab),
                   lk_flags=s.lk_flags.at[k].set(s.ch_flags | jnp.where(s.mv[src] != s.ch_src_mv, CF_SRC_UNRELATED, 0)),
                   n_chain=k + 1, ch_src=_i(-1))
    remaining = s.tg_e[src] & ~((1 << s.ch_eff) | (1 << (s.ch_eff + E)))
    s = s._replace(tg_e=s.tg_e.at[src].set(remaining), tg=s.tg.at[src].set(remaining != 0))
    return _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s),
                    lambda s: open_quick(s, 1 - s.ch_p), s)


def load_link(bank, s, k):
    """Resolution of chain link k; a negated link jumps straight to the end of its program, like that of a
    monster whose effects are negated and that is still linked to the chain link (solve_chain: STATUS_DISABLED)."""
    c = jnp.maximum(s.lk_src[k], 0)
    if bank.erefund.shape[1]:
        e, p = s.lk_eff[k], s.lk_p[k]
        key = opt_key(s.cid[c], bank.eopt[s.cid[c], e], bank.erefund[s.cid[c], e])
        refund = (bank.erefund[s.cid[c], e] != 0) & ((s.lk_flags[k] & CF_ACT_NEG) != 0) & (key >= 0)
        def restore(_, table):
            return table.at[p].set(jnp.where(table[p] == key + 1, 0, table[p]))
        s = s._replace(opt_cid=lax.fori_loop(0, refund.astype(I32), restore, s.opt_cid))
    dis = ((s.fl_c[c] & FL_DISABLED) != 0) & (s.zone[c] == Z_MZONE) & (s.mv[c] <= s.lk_t[k])
    continuous = (s.lk_flags[k] & CF_CONT) != 0
    dis |= continuous & ((s.zone[c] != Z_SZONE) | ((s.pos[c] & P_FACEUP) == 0) | (s.mv[c] > s.lk_t[k]))
    s = brk(s)  # break_effect before each chain link (solve_chain)
    return s._replace(ch_k=k, ch_src=s.lk_src[k], ch_eff=s.lk_eff[k], ch_p=s.lk_p[k], reg=s.lk_reg[k] & (s.mv <= s.lk_t[k]),
                      ch_lab=s.lk_lab[k], ch_flags=s.lk_flags[k],
                      pg=_i(PG_RESOLVE), pc=jnp.where(s.lk_neg[k] | dis, P - 1, 0).astype(I32), dec=_i(D_RUN),
                      ev_g=jnp.zeros_like(s.ev_g))


def chain_end(bank, s):
    """Chain resolved: the activated Spells/Traps still on the field go to the GY together, then a
    new window for the turn player."""
    n = s.cid.shape[0]
    s = group_to_grave(bank, s, (s.st & ST_LEAVE) != 0, REASON_RULE)
    s = s._replace(n_chain=_i(0), ch_src=_i(-1), ch_k=_i(0), pg=_i(0), pc=_i(0), reg=jnp.zeros((R, n), bool))
    return trig_point(bank, open_quick(s, s.tp, chained=True))


def exec_one(bank, s):
    """Executes one instruction of the current effect (state D_RUN)."""
    h, e, p = s.ch_src, s.ch_eff, s.ch_p
    ins = current_ins(bank, s)
    op, a, b, c = ins[0], ins[1], ins[2], ins[3]
    ra = jnp.clip(a, 0, R - 1)
    adv = lambda s: s._replace(pc=s.pc + 1)
    # resolution of a Trap effect: immune monsters are neither moved nor modified (is_affect_by_effect)
    imm = (s.pg == PG_RESOLVE) & is_trap(bank, s, h) & ((s.fl_c & FL_IMM_TRAP) != 0) & (s.zone == Z_MZONE)
    target = lambda s: s.reg[ra] & ~imm

    def next_link(s):
        # resolved (non-negated) chain link: the continuous effects of its card apply (enable_field_effect)
        on = ~s.lk_neg[s.ch_k]
        on &= ((s.lk_flags[s.ch_k] & CF_CONT) == 0) | ((s.zone[h] == Z_SZONE) &
               ((s.pos[h] & P_FACEUP) != 0) & (s.mv[h] <= s.lk_t[s.ch_k]))
        s = enable(s._replace(st=s.st.at[h].set(jnp.where(on, s.st[h] | ST_ON, s.st[h]))),
                   on & (jnp.arange(s.cid.shape[0]) == h))
        if bank.equip.shape[1]:  # resolved equip activation with no equipped target: to the GY at the end of the chain
            free = (((s.lk_flags[s.ch_k] & CF_ACT) != 0) & ((bank.ctype[s.cid[h]] & TYPE_EQUIP) != 0) & (s.eq_t[h] < 0)
                    & (s.zone[h] == Z_SZONE) & (s.mv[h] <= s.lk_t[s.ch_k]))
            s = s._replace(st=s.st.at[h].set(jnp.where(free, s.st[h] | ST_LEAVE, s.st[h])))
        return _cond(s.ch_k > 0, lambda s: load_link(bank, s, s.ch_k - 1), lambda s: chain_end(bank, s), s)

    def end(s):
        return _switch(s.pg, [lambda s: s._replace(pg=_i(PG_TARGET), pc=_i(0)),
                                 lambda s: _cond(bank.ekind[s.cid[h], e] == K_PROC,
                                                 lambda s: proc_ready(s, h), lambda s: push_link(bank, s), s),
                                 next_link], s)

    def pure(s):
        reg, ok = _pure(bank, s, s.reg, jnp.bool_(True), h, e, p, s.ch_k, ins)
        # false condition at resolution: the effect stops, or jumps to the end of the block that contains it
        skip = block_end(bank, s)
        return _cond(ok, adv, lambda s: _cond(skip >= 0, lambda s: s._replace(pc=skip), end, s), s._replace(reg=reg))

    def choose(s):  # at most (c >> 4) & 15 cards, fewer if there are not enough candidates (none: no decision)
        need = jnp.minimum((c >> 4) & 15, s.reg[jnp.clip(b, 0, R - 1)].sum()).astype(I32)
        # variable size (bit 9): at least c & 15 cards (as many as possible), A_PASS then ends the selection
        smin = jnp.where((c >> 9) & 1, jnp.minimum(c & 15, need), need).astype(I32)
        return _cond(need > 0,
                        lambda s: s._replace(dec=_i(D_SELECT), dec_player=rel(p, (c >> 8) & 1), need=need, picks=_i(0),
                                             smin=smin),
                        lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def move_(s):  # GY, hand / top of the owner's Deck, or banished
        # c = 1: destruction (Duel.Destroy): cards indestructible by effects stay; the register only keeps
        # the destroyed cards (Destroy(...) ~= 0)
        destroying = (c & 1) != 0
        supplied = (c & MOVE_REASON_EXPLICIT) != 0
        raw_reason = (c >> 1) & 0x1fffffff
        base = jnp.where(supplied, raw_reason, jnp.where(s.pg == PG_COST, REASON_COST, REASON_EFFECT))
        reason = base | jnp.where(destroying, REASON_DESTROY, 0)
        reason |= jnp.where((c < 0) & (s.zone[h] == Z_MZONE), REASON_RELEASE, 0)
        t = target(s) & ~(destroying & ((s.fl_c & FL_INDES_EFF) != 0) & (s.zone == Z_MZONE))
        s = s._replace(reg=jnp.where(destroying, s.reg.at[ra].set(t), s.reg))
        return adv(_switch(jnp.where((b == Z_HAND) | (b == Z_DECK), 1, jnp.where(b == Z_REMOVED, 2, 0)),
                              [lambda s: group_to_grave(bank, s, t, reason),
                               lambda s: group_to_owner_zone(bank, s, t, b, reason),
                               lambda s: group_send(s, t, Z_REMOVED, P_GRAVE, reason=reason)], s))

    def fusmat(s):  # materials of the Fusion monster of register b, chosen one by one by p (D_SELECT), stored in a
        f = jnp.argmax(s.reg[jnp.clip(b, 0, R - 1)])
        go = s.reg[jnp.clip(b, 0, R - 1)].any()
        return _cond(go, lambda s: s._replace(dec=_i(D_SELECT), dec_player=p, need=bank.fmn[s.cid[f]], picks=_i(0),
                                                fsel=jnp.bool_(True)),
                     lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def ritmat(s):  # Tributes for the Ritual monster of register b (mode c), chosen one by one by p, stored in a
        go = s.reg[jnp.clip(b, 0, R - 1)].any()
        return _cond(go, lambda s: s._replace(dec=_i(D_SELECT), dec_player=p, need=_i(99), picks=_i(0), rsel=c),
                     lambda s: adv(s._replace(reg=s.reg.at[ra].set(False))), s)

    def spsummon(s):
        # ST_SEL keeps the remaining placements without emptying the result register.
        # A placed card keeps ST_PSTEP until the group's simultaneous success.
        resumed = s.zctx == ZC_SPSUM
        pending = jnp.where(resumed, (s.st & ST_SEL) != 0, s.reg[ra])
        active = (op == OP_SPSUMMON) & pending.any()
        typ = effect_summon_type(c)
        # suminfo: only at the first pass of a group Summon (not on resumption); ss_t keeps
        # the cards whose Special Summon was attempted this turn (OP_SSLOCK).
        attempted = s.reg[ra] & ~resumed & (op == OP_SPSUMMON)
        s = s._replace(suminfo=jnp.where(attempted, summon_info(typ, s.zone, p), s.suminfo), ss_t=s.ss_t | attempted)
        def prepare(_, carry):
            group, failed_space, selected, ready = carry
            lk = (s.k_ctype & TYPE_LINK) != 0
            group &= special_allowed(bank, s, proper=c > 0) & ~(lk & ((b & P_FU_ATK) == 0))
            # A response may have filled the zones or forbidden the Summon after targeting.
            space = jnp.where(s.zone == Z_EXTRA, ex_zones(s, p).any(), free_main(s, p) > 0)
            faceup_pend = ((s.zone == Z_EXTRA) & ((s.k_ctype & TYPE_PENDULUM) != 0)
                           & ((s.pos & P_FACEUP) != 0))
            pend_space = lax.fori_loop(0, (group & faceup_pend).any().astype(I32),
                lambda _, previous: pend_extra_zones(bank, s, p).any(), jnp.bool_(True))
            space = jnp.where(faceup_pend, pend_space, space)
            # card_operation_sort: turn controller, location, then sequence
            # (descending for the ordered locations other than the hand).
            desc = (s.zone == Z_DECK) | (s.zone == Z_EXTRA) | (s.zone == Z_GRAVE) | (s.zone == Z_REMOVED)
            key = ((s.ctrl != s.tp) * 256 + s.zone) * s.cid.shape[0] + jnp.where(desc, s.cid.shape[0] - 1 - s.seq, s.seq)
            selected = jnp.argmin(jnp.where(group, key, BIG)).astype(I32)
            need_link = group.any() & (s.zone[selected] == Z_EXTRA) & lk[selected]
            link_space = lax.fori_loop(0, need_link.astype(I32),
                lambda _, previous: link_zones(bank, s, p, selected).any(), jnp.bool_(True))
            available = space & (~lk | link_space)
            chosen = (jnp.arange(s.cid.shape[0]) == selected) & group
            # A full main zone can no longer be freed during these
            # placements. Discard its candidates all at once, even when an
            # Extra Deck is present: no loop over N refusals before the next Extra.
            full = (s.zone != Z_EXTRA) & (free_main(s, p) == 0)
            blocked = group & (chosen | full) & ~available
            failed_space = blocked & (s.zone != Z_GRAVE)
            # Only the next card is tested for room: its placement
            # changes the free zones, without enabling the arrows of the pending Links.
            ready = (chosen & available).any()
            return group & ~blocked, failed_space, selected, ready
        group, failed_space, selected, ready = lax.fori_loop(0, active.astype(I32), prepare,
            (pending, jnp.zeros_like(pending), _i(0), jnp.bool_(False)))
        s = s._replace(st=(s.st & ~ST_SEL) | jnp.where(group, ST_SEL, 0) | jnp.where(failed_space, ST_SSFAIL, 0))
        def finish(s):
            # SpSummon / ss_tograve_set: send by rule after the placements,
            # then activation of the effects and a single event for the whole group.
            failed = (s.st & ST_SSFAIL) != 0
            s = lax.fori_loop(0, (failed.any() & (op == OP_SPSUMMON)).astype(I32),
                lambda _, state: group_to_grave(bank, state, failed, REASON_RULE), s)
            placed = (s.st & ST_PSTEP) != 0
            s = _cond(placed.any(), lambda s: raise_ev(enable(s, placed), EV_SPSUMMON, placed, _i(2)), lambda s: s, s)
            return adv(s._replace(reg=s.reg.at[ra].set(placed), st=s.st & ~(ST_SEL | ST_PSTEP | ST_SSFAIL),
                                  zctx=_i(ZC_ACT), pend=_i(-1)))
        return _cond(group.any(), lambda s: s._replace(dec=jnp.where(ready, D_ZONE, D_RUN), zctx=_i(ZC_SPSUM),
                                                   pend=selected, dec_player=p), finish, s)

    def lock(s):  # p can no longer activate the effects of cards with this name this turn
        return adv(_cond(s.reg[ra].any(), lambda s: s._replace(
            lock_cid=s.lock_cid.at[p].set(name_add(s.lock_cid[p], s.cid[jnp.argmax(s.reg[ra])]))), lambda s: s, s))

    def draw_n(s):
        q = rel(p, a)
        # under vmap, the switch evaluates all the branches with the b of the current instruction (1000 for DAMAGE...):
        # the number of loop rounds is that of the game that asks for the most, hence the mask on op
        nb = jnp.where(op == OP_DRAW, jnp.clip(b, 0, s.cid.shape[0]), 0)
        return adv(lax.fori_loop(0, nb, lambda i, s: draw(s, q), s))

    # temporary modifier on the cards of register a: b = MC_* | MV_* << 5 | TD_* << 9 | RD << 13 | NRC << 14
    def modify(s):
        j = jnp.minimum(s.tm_n, TM - 1)
        m = target(s) & (s.zone == Z_MZONE)
        kill = ((jnp.asarray(MC_KILL, I32)[b & 31] >> s.tm_code) & 1) != 0
        s = s._replace(tm_m=s.tm_m & ~(kill[:, None] & m[None, :]))
        mode = (b >> 5) & 15
        # Lua x/2 is a float, rounded by std::round in effect::get_value: upper half
        v = jnp.where(mode == MV_HALF_BASE_ATK, (s.batk_c + 1) // 2, jnp.where(mode == MV_HALF_ATK, (s.atk_c + 1) // 2, c))
        # NRC: Duel.NegateRelatedChain(tc) before the negation: the chain links in progress of tc, still linked to their
        # card, are negated (EFFECT_DISABLE_CHAIN, kept even if tc later leaves the field)
        src = jnp.maximum(s.lk_src, 0)
        rel = (((b >> 14) & 1) != 0) & (jnp.arange(CH) < s.ch_k) & (s.lk_src >= 0) & m[src] & (s.mv[src] <= s.lk_t)
        s = s._replace(lk_neg=s.lk_neg | rel)
        return adv(s._replace(tm_code=s.tm_code.at[j].set(b & 31), tm_m=s.tm_m.at[j].set(m),
                              tm_rd=s.tm_rd.at[j].set(((b >> 13) & 1) != 0),
                              tm_v=s.tm_v.at[j].set(jnp.where(m, v, 0)), tm_t=s.tm_t.at[j].set(s.clk + 1), clk=s.clk + 1,
                              tm_dur=s.tm_dur.at[j].set((b >> 9) & 15), tm_src=s.tm_src.at[j].set(h),
                              tm_n=jnp.minimum(s.tm_n + 1, TM), ovf=s.ovf | jnp.where(s.tm_n >= TM, 2, 0).astype(I32)))

    def damage_(s):
        mul = jnp.where(c > 0, s.reg[jnp.clip(c - 1, 0, R - 1)].sum(), 1)
        return adv(damage(s, rel(p, a), b * mul))

    branches = [lambda s: s] * N_OPS
    branches[OP_END] = end
    branches[OP_BLOCK] = adv
    for o in (OP_FILTER, OP_SELF, OP_REQ, OP_REQ_FREE, OP_MINATK, OP_COND_CHAIN, OP_CHAINCARD, OP_EVENT, OP_COND_TURN,
              OP_COND_EVP, OP_COND_SELF, OP_COND_ATK, OP_ATTACKER, OP_COND_PHASE, OP_COND_LABEL):
        branches[o] = pure
    branches[OP_CHOOSE] = choose
    branches[OP_LOCK] = lock

    def sslock(s):  # insertion at the head of the table (the oldest drops out if the table is full)
        row = jnp.stack([h + 1, e, b, c, rel(p, a)]).astype(I32)
        return adv(s._replace(ssl=jnp.concatenate([row[None], s.ssl[:-1]])))
    branches[OP_SSLOCK] = sslock
    branches[OP_MODIFY] = modify
    branches[OP_FUSMAT] = fusmat
    branches[OP_RITMAT] = ritmat
    branches[OP_MOVE] = move_
    branches[OP_SPSUMMON] = spsummon
    branches[OP_DRAW] = draw_n
    branches[OP_DAMAGE] = damage_
    branches[OP_RECOVER] = lambda s: adv(s._replace(lp=s.lp.at[rel(p, a)].add(b)))
    branches[OP_SHUFFLE_HAND] = adv  # the order of the hand is not observed: nothing to do
    def negate(s):  # negate_chain: the card of an activation still linked in the Spell/Trap zone goes to the GY
        k1 = jnp.clip(s.ch_k - 1, 0, CH - 1)
        lc = jnp.maximum(s.lk_src[k1], 0)
        hit = (((s.lk_flags[k1] & CF_ACT) != 0) & (s.zone[lc] == Z_SZONE) & (s.mv[lc] <= s.lk_t[k1])
               & (s.lk_src[k1] >= 0))
        return adv(s._replace(lk_neg=s.lk_neg.at[k1].set(True),
                              lk_flags=s.lk_flags.at[k1].set(s.lk_flags[k1] | CF_ACT_NEG),
                              st=s.st.at[lc].set(jnp.where(hit, s.st[lc] | ST_LEAVE | ST_ACTDIS, s.st[lc]))))
    branches[OP_NEGATE] = negate
    # LP cost of an activation: loss noted by adjust when put on the chain (push_link), after the targets;
    # procedure (K_PROC, no chain link): immediately
    branches[OP_PAY_LP] = lambda s: adv(_cond(bank.ekind[s.cid[h], e] == K_PROC, lambda s: damage(s, p, a),
                                              lambda s: s._replace(lp=s.lp.at[p].set(jnp.maximum(0, s.lp[p] - a))), s))
    branches[OP_LABEL] = lambda s: adv(s._replace(
        ch_lab=jnp.where(b == 0, s.reg[ra].any().astype(I32), s.ch_lab),
        ch_targets=s.ch_targets | ((b == 1) & (c == 0) & s.reg[ra]),
        ch_tgt_t=jnp.where(b == 1, s.clk, s.ch_tgt_t), ch_src_mv=src_retarget(s, b == 1, s.reg[ra])))
    branches[OP_YESNO] = lambda s: s._replace(dec=_i(D_YESNO), dec_player=p)
    if bank.equip.shape[1]:
        def equip_(s):  # EquipOperation: the target (linked to the chain link) must be face-up; the card still in place
            n = s.cid.shape[0]
            t = jnp.argmax(target(s))
            ok = (target(s).any() & (s.zone[t] == Z_MZONE) & ((s.pos[t] & P_FACEUP) != 0) & (s.zone[h] == Z_SZONE)
                  & (s.mv[h] <= s.lk_t[s.ch_k]) & (t != h))
            s2 = enable(s._replace(eq_t=s.eq_t.at[h].set(t), eq_mv=s.eq_mv.at[h].set(s.clk + 1)), jnp.arange(n) == h)
            return adv(jax.tree.map(lambda x, y: jnp.where(ok, x, y), s2, s))
        branches[OP_EQUIP] = equip_
    else:
        branches[OP_EQUIP] = adv
    return _switch(op, branches, s)


# ---------------------------------------------------------------- legal actions

def act_cand(bank, s):
    """[E, N]: effects that the deciding player could activate, before the cost and target test (act_ok)."""
    p = s.dec_player
    mine = s.ctrl == p
    ct = s.k_ctype
    spell, trap = (ct & TYPE_SPELL) != 0, (ct & TYPE_TRAP) != 0
    quick = spell & ((ct & TYPE_QUICKPLAY) != 0)
    in_hand = mine & (s.zone == Z_HAND)
    my_mon = mine & (s.zone == Z_MZONE)
    free_s = free_main(s, p, Z_SZONE)
    idle, bat, chn = s.dec == D_IDLE, s.dec == D_BATTLE, s.dec == D_CHAIN
    set_st = mine & (s.zone == Z_SZONE) & (s.pos == P_SFD)
    ready = set_st & ((s.st & ST_SET_TURN) == 0)       # Set before this turn
    from_hand = in_hand & (free_s > 0)
    us = zone_used(s, p, Z_SZONE)
    pend_hand = in_hand & ((ct & TYPE_PENDULUM) != 0) & (~us[0] | ~us[4])  # toward a free Pendulum Zone
    faceup_mon = my_mon & ((s.pos & P_FACEUP) != 0)
    in_grave = mine & (s.zone == Z_GRAVE)
    faceup_st = mine & (s.zone == Z_SZONE) & ((s.pos & P_FACEUP) != 0) & ((s.st & ST_ON) != 0)
    trig_w = chn & (s.wk == W_TRIG)
    # hand adjusted in the End Phase (hand_adjusted): no free window, only responses to a chain
    fast = bat | (chn & ~trig_w & ~((s.wk == W_QUICK) & (s.ret == R_HAND) & (s.n_chain == 0)))  # speed 2 minimum
    top = jnp.clip(s.n_chain - 1, 0, CH - 1)
    top_spd = jnp.where(s.n_chain > 0, speed(bank, s.cid[jnp.maximum(s.lk_src[top], 0)], s.lk_eff[top]), 0)
    evb = ev_bits(s)
    tkind = jnp.where(s.tst < 2, K_TRIG_F, K_TRIG_O)
    cand = []
    for e in range(E):
        kind = s.k_ekind[:, e]
        trig = s.k_etrig[:, e]
        act = kind == K_ACTIVATE
        rng = s.k_erng[:, e]
        # locations of a monster quick effect / ignition effect (erng; 0 for an ignition effect: monster zone)
        rng = jnp.where((kind == K_IGNITION) & (rng == 0), Z_MZONE, rng)
        inr = ((in_hand & ((rng & Z_HAND) != 0)) | (faceup_mon & ((rng & Z_MZONE) != 0))
               | (in_grave & ((rng & Z_GRAVE) != 0)) | (faceup_st & ((rng & Z_SZONE) != 0)))
        qm = (kind == K_QUICK) & inr
        where = ((idle & spell & ~quick & act & (from_hand | set_st))
                 | (idle & act & pend_hand)
                 | ((idle | fast) & qm)
                 | ((idle | fast) & quick & act & ((from_hand & (p == s.tp)) | ready))
                 | ((idle | fast) & trap & act & ready & ((s.pf_c[p] & PF_NO_TRAP) == 0))
                 | (idle & inr & (kind == K_IGNITION)))
        where &= speed_of(s.k_ctype, s.k_ekind[:, e]) >= top_spd
        where &= (trig == 0) | ((trig & evb) != 0)
        # damage step (effect::is_activateable): DAMAGE_STEP / DAMAGE_CAL flags, except Counter Traps and effects
        # on a battle event; after the calculation, the StatChangeDamageStepCondition condition forbids
        dmg = s.k_edmg[:, e]
        dsok = jnp.where(s.dstep == DS_BEFORE, (dmg & DMG_STEP) != 0,
                         jnp.where(s.dstep == DS_CAL, (dmg & DMG_CAL) != 0,
                                   ((dmg & DMG_STEP) != 0) & ((dmg & DMG_STAT) == 0)))
        dsok |= (s.dstep == DS_NONE) | (act & ((ct & TYPE_COUNTER) != 0)) | ((trig & (1 << EV_PRE_DMG)) != 0)
        where &= dsok
        # The optional private-hand ones respond in QuickEffect, even at
        # speed 2; never to a Counter Trap, nor with an empty chain after
        # the hand adjustment (fast is then false).
        private = ((s.tg_e & (1 << (e + E))) != 0) & in_hand & inr & (kind == K_TRIG_O)
        where |= chn & (s.wk == W_QUICK) & fast & private & (top_spd <= 2) & dsok
        where |= trig_w & mine & s.tg & ((s.tg_e & (1 << e)) != 0) & (kind == tkind)  # current SEGOC step
        # Inherent procedure from the hand: neither chain speed nor monster activation.
        where |= idle & in_hand & (kind == K_PROC) & (free_main(s, p) > 0) & ((s.fl_c & FL_NO_SS) == 0)
        cand.append(where)
    return jnp.stack(cand)


def auto_pass(bank, s):
    """Response window where the player cannot activate anything: passing is the only action (played by settle, without
    going through legal + apply). W_TRIG windows are only opened if there is an activatable trigger.
    With at most KA candidates, their cost and target are tested (act_ok, like legal): in mid-game, most
    windows have one or two candidates that fail act_ok, and the pass, the only legal action, would otherwise take
    a full round of step (apply, settle, legal, forced). Same game: A_PASS also plays pass_priority."""
    cand = act_cand(bank, s)
    nc = cand.sum()
    ce, ch = jnp.nonzero(cand, size=KA, fill_value=0)
    valid = (jnp.arange(KA) < nc) & (nc <= KA) & (s.dec == D_CHAIN) & (s.wk != W_TRIG) & ~s.done
    okc = jax.vmap(lambda h, e, active: act_ok(bank, s, h, e, s.dec_player, enabled=active))(ch, ce, valid) & valid
    return (s.dec == D_CHAIN) & (s.wk != W_TRIG) & ~s.done & ((nc == 0) | ((nc <= KA) & ~okc.any()))


def legal(bank, s, fus=True):
    """Mask of legal actions. fus=False: without the choice of Fusion materials (step_batch only computes it if
    a game of the batch is at that point)."""
    n = s.cid.shape[0]
    p = s.dec_player
    mine = s.ctrl == p
    ct = s.k_ctype
    monster, spell, trap = (ct & TYPE_MONSTER) != 0, (ct & TYPE_SPELL) != 0, (ct & TYPE_TRAP) != 0
    in_hand = mine & (s.zone == Z_HAND)
    my_mon = mine & (s.zone == Z_MZONE)
    op_mon = (s.ctrl == 1 - p) & (s.zone == Z_MZONE)
    free_m, free_s = free_main(s, p), free_main(s, p, Z_SZONE)
    nmon = my_mon.sum()
    lv = s.k_level
    need = jnp.where(lv >= 7, 2, jnp.where(lv >= 5, 1, 0))
    # Ritual monsters and Extra Deck monsters: no Normal Summon
    can_ns = (in_hand & monster & ~s.ns_used & jnp.where(need == 0, free_m > 0, nmon >= need)
              & ((ct & (TYPE_RITUAL | TYPE_EXTRA | TYPE_SPSUMMON)) == 0))
    can_repos = (my_mon & ((s.st & (ST_SUMMON_TURN | ST_ATTACKED | ST_FORM_CHANGED | ST_SET_TURN)) == 0)
                 & ((ct & TYPE_LINK) == 0))  # a Link does not change position
    unsel = (s.st & ST_SEL) == 0
    # get_attack_target (select_target true by default): at least one possible target, or direct attack
    atk_ok = ~op_mon.any() | (op_mon & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    # without direct attack (FL_NO_DIRECT): there must be an opposing monster that can be attacked
    tgt_ok = (op_mon & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
    can_atk = (my_mon & (s.pos == P_FU_ATK) & ((s.st & ST_ATTACKED) == 0) & ((s.fl_c & FL_NO_ATTACK) == 0)
               & jnp.where((s.fl_c & FL_NO_DIRECT) != 0, tgt_ok, atk_ok))
    ins = current_ins(bank, s)
    eff_cand = s.reg[jnp.clip(ins[2], 0, R - 1)]
    if fus:
        src = jnp.argmax(eff_cand)
        eff_cand = jnp.where(s.fsel, fus_picks(bank, s, src),
                             jnp.where(s.rsel > 0, rit_picks(bank, s, src, s.rsel), eff_cand))
        syn_c = syn_picks(bank, s, jnp.maximum(s.pend, 0))
        xyz_c = xyz_mats(bank, s, jnp.maximum(s.pend, 0)) & unsel
        lnk_c, lgoal = link_picks(bank, s, jnp.maximum(s.pend, 0))
    else:
        syn_c = xyz_c = lnk_c = jnp.zeros(n, bool)
        lgoal = jnp.bool_(False)
    # Pendulum Summon: V_SUMMON on the left Scale; selection (PS_PEND): loop of Pendulum.Operation
    pl, can_pend, pch, pcx, ft1, ft2, ft, plz, _ = pend_info(bank, s, p)
    psel_n = (~unsel & pch).sum(), (~unsel & pcx).sum()
    pen_c = unsel & ((pch & (ft1 > psel_n[0])) | (pcx & (ft2 > psel_n[1]))) & (ft > psel_n[0] + psel_n[1])

    # Tributes: with no free main zone, one of them must free a main zone (not an extra zone)
    mainz = my_mon & (s.seq < N_MAIN_ZONES)
    trib_ok = ((free_m > 0) | mainz | (mainz & ~unsel).any()
               | ((s.need >= 2) & (mainz & unsel).any()))
    idle, bat, chn = s.dec == D_IDLE, s.dec == D_BATTLE, s.dec == D_CHAIN
    # Summon by procedure (Synchro, Xyz, Link): V_SUMMON on the Extra Deck monster, unless forbidden (FL_NO_SS)
    can_syn = (s.zone == Z_EXTRA) & (s.ctrl == p) & (s.sz_c | s.xz_c | s.lz_c) & ((s.fl_c & FL_NO_SS) == 0)
    verbs = [
        idle & (can_ns | can_syn | ((jnp.arange(n) == pl) & can_pend)),
        idle & can_ns,
        idle & can_repos,
        bat & can_atk,
        ((s.dec == D_TRIBUTE) & jnp.where(s.psel == PS_SYN, syn_c, jnp.where(s.psel == PS_XYZ, xyz_c, jnp.where(s.psel == PS_LINK, lnk_c, jnp.where(s.psel == PS_PEND, pen_c, my_mon & unsel & trib_ok))))) | ((s.dec == D_TARGET) & op_mon & ((s.fl_c & FL_NO_BTARGET) == 0))
        | ((s.dec == D_DISCARD) & in_hand & unsel) | ((s.dec == D_SELECT) & eff_cand & unsel),
        idle & in_hand & (spell | trap) & (free_s > 0),
    ]
    trig_w = chn & (s.wk == W_TRIG)
    cand = act_cand(bank, s)
    # All the candidates, in blocks of KC; 16 blocks at most in the bounded arena.
    # The kernel stays KC x N, and ordinary games only make one block.
    def more(carry):
        remaining, _, rounds = carry
        return remaining.any() & (rounds < ACT_ROUNDS)
    def batch(carry):
        remaining, accepted, rounds = carry
        ce, ch = jnp.nonzero(remaining, size=KC, fill_value=0)
        valid = jnp.arange(KC) < remaining.sum()
        passed = jax.vmap(lambda h, e, active: act_ok(bank, s, h, e, p, enabled=active))(ch, ce, valid) & valid
        hit = onehot(ce, E)[:, :, None] & onehot(ch, n)[:, None, :]          # [KC, E, N]
        accepted |= (hit & passed[:, None, None]).any(0)
        return remaining & ~hit.any(0), accepted, rounds + 1
    _, ok, _ = lax.while_loop(more, batch, (cand, jnp.zeros((E, n), bool), _i(0)))
    verbs = jnp.concatenate([jnp.stack(verbs), cand & ok])
    sp = jnp.zeros(N_SPECIAL, bool)
    sp = sp.at[A_TO_BP].set(idle & (s.phase == PH_MAIN1) & (s.turn > 1))
    sp = sp.at[A_TO_M2].set(bat)
    sp = sp.at[A_TO_EP].set(idle | bat)
    mz = (s.zctx == ZC_SUMMON) | (s.zctx == ZC_SPSUM) | (s.zctx == ZC_PROC) | (s.zctx == ZC_PEND)
    used = jnp.where(mz, zone_used(s, p, Z_MZONE), zone_used(s, p, Z_SZONE))[:7]
    spc = jnp.maximum(s.pend, 0)             # card Special Summoned (ZC_SPSUM)
    fromex = ((s.zctx == ZC_SPSUM) & (s.zone[spc] == Z_EXTRA)) | ((s.zctx == ZC_PROC) & (s.zone[jnp.maximum(s.pend, 0)] == Z_EXTRA))
    pc = jnp.where((s.zctx == ZC_PROC) | (s.zctx == ZC_PEND), jnp.maximum(s.pend, 0), spc)  # card Summoned
    is_lk = (ct[pc] & TYPE_LINK) != 0
    exz = lax.cond(is_lk & (s.dec == D_ZONE), lambda s: link_zones(bank, s, p, pc), lambda s: ex_zones(s, p), s)
    faceup_pend = ((ct[pc] & TYPE_PENDULUM) != 0) & ((s.pos[pc] & P_FACEUP) != 0) & ~is_lk
    exz = jnp.where(faceup_pend, jnp.concatenate([exz[:N_MAIN_ZONES] & plz, exz[N_MAIN_ZONES:]]), exz)
    zok = jnp.where(fromex, exz, ~used & (jnp.arange(7) < N_MAIN_ZONES))
    zok = jnp.where(s.zctx == ZC_PEND, pend_zones(bank, s, p, pc, plz), zok)
    # activation of a Pendulum: Pendulum Zones (zones 0 and 4)
    pact = (s.zctx == ZC_ACT) & ((ct[jnp.maximum(s.pend, 0)] & TYPE_PENDULUM) != 0)
    zok &= ~pact | (jnp.arange(7) == 0) | (jnp.arange(7) == 4)
    sp = sp.at[A_ZONE0:A_ZONE0 + 7].set((s.dec == D_ZONE) & zok)
    pos_ok = jnp.where(s.zctx == ZC_SPSUM, current_ins(bank, s)[2], P_FU_ATK | P_FU_DEF)  # allowed positions
    pos_ok &= jnp.where(is_lk, P_FU_ATK, -1)  # a Link is always in Attack Position
    sp = sp.at[A_POS_ATK].set((s.dec == D_POS) & ((pos_ok & P_FU_ATK) != 0))
    sp = sp.at[A_POS_DEF].set((s.dec == D_POS) & ((pos_ok & P_FU_DEF) != 0))
    sp = sp.at[A_PASS].set((chn & ~(trig_w & (s.tst < 2)))  # mandatory triggers cannot be declined
                           | ((s.dec == D_TRIBUTE) & (s.psel == PS_LINK) & lgoal)  # Link: "finish" the selection
                           | ((s.dec == D_TRIBUTE) & (s.psel == PS_PEND) & ~unsel.all())  # Pendulum: at least one card
                           | (s.dec == D_YESNO)
                           | (sel_var(s) & (s.picks >= s.smin)))  # variable-size selection: "finish"
    sp = sp.at[A_YES].set(s.dec == D_YESNO)
    m = jnp.concatenate([verbs.reshape(-1), sp])
    return m & ~s.done & (s.dec != D_NONE)


# ---------------------------------------------------------------- applying an action

def apply(bank, s, a):
    n = s.cid.shape[0]
    verb, c = a // n, a % n
    k = a - NV * n
    s_id = lambda s: s

    def summon(s, is_set):
        need = tributes_needed(bank, s, c)
        info = summon_info(jnp.where(need > 0, SUMMON_TRIBUTE, SUMMON_NORMAL), _i(Z_HAND), s.tp)
        s = s._replace(suminfo=s.suminfo.at[c].set(jnp.where(s.zone[c] == Z_HAND, info, s.suminfo[c])))
        s = s._replace(pend=c, pend_set=jnp.bool_(is_set), need=need, picks=_i(0), zctx=_i(ZC_SUMMON))
        s = s._replace(dec=jnp.where(need > 0, D_TRIBUTE, D_ZONE).astype(I32))
        # from the Extra Deck: Synchro Summon (materials chosen one by one in D_TRIBUTE, then zone and position)
        ps = jnp.where(bank.syn[s.cid[c]] != 0, PS_SYN, jnp.where(bank.xct[s.cid[c]] > 0, PS_XYZ, PS_LINK)).astype(I32)
        # from the left Scale: Pendulum Summon (cards chosen one by one, then placed one by one)
        return _switch(jnp.where(s.zone[c] == Z_EXTRA, 1, jnp.where(s.zone[c] == Z_SZONE, 2, 0)), [
            lambda s: s,
            lambda s: s._replace(psel=ps, need=_i(99), zctx=_i(ZC_PROC), dec=_i(D_TRIBUTE)),
            lambda s: s._replace(psel=_i(PS_PEND), need=_i(99), zctx=_i(ZC_PEND), dec=_i(D_TRIBUTE))], s)

    def repos(s):
        new = jnp.where(s.pos[c] == P_FU_ATK, P_FU_DEF, P_FU_ATK)
        fd = (jnp.arange(n) == c) & (s.pos[c] == P_FD_DEF)  # face-down -> face-up: Flip Summon
        s = raise_ev(raise_ev(s, EV_FLIP_SUMMON, fd, s.tp), EV_FLIP, fd, s.tp)
        info = (s.suminfo[c] & 0x7fffffff) | (s.tp << 31)
        s = s._replace(suminfo=s.suminfo.at[c].set(jnp.where(fd[c], info, s.suminfo[c])))
        s = enable(s._replace(pos=s.pos.at[c].set(new),
                              st=s.st.at[c].set(s.st[c] | ST_FORM_CHANGED | jnp.where(fd[c], ST_FLIP_TURN, 0))), fd)
        return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

    def attack(s):
        s = s._replace(st=s.st.at[c].set(s.st[c] | ST_ATTACKED), atk=c, atk_tgt=_i(-1), ds_t=s.clk, rpl=jnp.bool_(False))
        opp = (s.ctrl == 1 - s.tp) & (s.zone == Z_MZONE)
        has_target = opp.any()
        # no possible target (CANNOT_BE_BATTLE_TARGET): attack_announce_failed, the attack counts, no window
        ok = (opp & ((s.fl_c & FL_NO_BTARGET) == 0)).any()
        return _cond(has_target, lambda s: _cond(ok, lambda s: s._replace(dec=_i(D_TARGET)), to_battle, s),
                            lambda s: trig_point(bank, declare(s, _i(-1), jnp.bool_(False))), s)

    def select(s):
        def pick(s):
            return s._replace(st=s.st.at[c].set(s.st[c] | ST_SEL), need=s.need - 1, picks=s.picks + 1)

        def tribute(s):
            s = pick(s)
            sel = (s.st & ST_SEL) != 0
            xdone = sel.sum() >= bank.xct[s.cid[jnp.maximum(s.pend, 0)]]
            # Link: the selection ends with A_PASS (played automatically when no candidate remains)
            fin = jnp.where(s.psel == PS_SYN, syn_done(bank, s),
                            jnp.where(s.psel == PS_XYZ, xdone, (s.psel == 0) & (s.need == 0)))
            done = lambda s: _cond(s.psel == PS_XYZ,
                                   lambda s: attach(s, sel, s.pend)._replace(picks=_i(0)),  # Xyz materials attached
                                   lambda s: selected_to_grave(bank, s, jnp.where(s.psel == PS_SYN,
                                       REASON_SYNCHRO | REASON_MATERIAL, REASON_SUMMON | REASON_MATERIAL | REASON_RELEASE)), s)._replace(dec=_i(D_ZONE), psel=_i(0))
            return _cond(fin, done, lambda s: s, s)

        def target(s):
            return trig_point(bank, declare(s, c, s.rpl))

        def discard(s):
            s = pick(s)
            return _cond(s.need == 0, lambda s: trig_point(bank, open_quick(selected_to_grave(bank, s, REASON_RULE | REASON_DISCARD | 0x100), s.tp, R_HAND,
                                                                            chained=False)), lambda s: s, s)

        def eff_select(s):
            done = lambda s: sel_finish(bank, s)
            s = pick(s)
            rit = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[2], 0, R - 1)])
            fin = jnp.where(s.rsel > 0, rit_done(bank, s, rit, s.rsel), s.need == 0)
            return _cond(fin, done, lambda s: s, s)

        return _switch(jnp.clip(s.dec - D_TRIBUTE, 0, 4), [tribute, s_id, target, discard, eff_select], s)

    def sset(s):
        return s._replace(pend=c, zctx=_i(ZC_SSET), dec=_i(D_ZONE))

    def activate(s):
        # a single branch for the E effects (dynamic e): under vmap, _switch computes all its branches
        e = jnp.clip(verb - V_ACT0, 0, E - 1)
        p = s.dec_player
        # where to resume once the chain is resolved
        s = _switch(jnp.where(s.dec == D_IDLE, 0, jnp.where(s.dec == D_BATTLE, 1, jnp.where(s.wk == W_PHASE, 2, 3))), [
            lambda s: s._replace(ret=_i(R_IDLE), chained=jnp.bool_(False)),
            lambda s: s._replace(ret=_i(R_BATTLE), chained=jnp.bool_(False)),
            lambda s: s._replace(ret=_i(R_PHASE), ph_restart=p),
            s_id], s)
        s = s._replace(ch_eff=_i(e))
        from_hand = lambda s: s._replace(pend=c, zctx=_i(ZC_ACT), dec=_i(D_ZONE))
        active = (verb >= V_ACT0) & (verb < NV)
        flip = lambda s: begin_act(bank, s._replace(pos=s.pos.at[c].set(P_SFU)), c, e, enabled=active)
        card = bank.ekind[s.cid[c], e] == K_ACTIVATE  # card activation (not a monster effect)
        return _switch(jnp.where(card & (s.zone[c] == Z_HAND), 0, jnp.where(card & (s.zone[c] == Z_SZONE), 1, 2)),
                          [from_hand, flip, lambda s: _cond((bank.ekind[s.cid[c], e] == K_PROC) &
                              (bank.prog[s.cid[c], e, PG_TARGET, 0, 0] == OP_END),
                              lambda s: proc_ready(s, c), lambda s: s, begin_act(bank, s, c, e, enabled=active))], s)

    def zone(s):
        z = k - A_ZONE0

        def to_szone(s, pos):
            return move(leave_hand(s, s.pend), s.pend, Z_SZONE, z, s.dec_player, pos)

        def set_done(s):
            s = to_szone(s, P_SFD)
            s = s._replace(st=s.st.at[s.pend].set(ST_SET_TURN), pend=_i(-1))
            return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

        return _switch(s.zctx, [
            lambda s: place_pending(bank, s, z),
            set_done,
            lambda s: begin_act(bank, to_szone(s, P_SFU), s.pend, s.ch_eff,
                                 enabled=(verb == NV) & (s.dec == D_ZONE) & (s.zctx == ZC_ACT)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
            lambda s: s._replace(zsel=z, dec=_i(D_POS)),
        ], s)

    def proc_place(s):  # end of a Summon by procedure (Synchro): no chain, window like a Summon
        c = s.pend
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        ct = bank.ctype[s.cid[c]]
        typ = jnp.where((ct & TYPE_LINK) != 0, SUMMON_LINK,
                        jnp.where((ct & TYPE_XYZ) != 0, SUMMON_XYZ,
                                  jnp.where((ct & TYPE_SYNCHRO) != 0, SUMMON_SYNCHRO, SUMMON_SPECIAL)))
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(typ,s.zone[c],s.tp)), ss_t=s.ss_t.at[c].set(True))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.tp, pos)
        s = raise_ev(brk(s), EV_SPSUMMON, jnp.arange(n) == c, s.tp)  # break_effect after the materials
        lk = (bank.ctype[s.cid[c]] & TYPE_LINK) != 0
        s = s._replace(proc=s.proc.at[c].set(True), st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM | jnp.where(lk, ST_LINKSUM, 0)),
                       pend=_i(-1))
        return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))

    def pend_next(s):
        """Pendulum Summon: next chosen card to place (smallest uid: card_sort), or end: EVENT_SPSUMMON
        of the group, then window like a Summon."""
        sel = (s.st & ST_SEL) != 0

        def go(s):
            return s._replace(pend=jnp.argmin(jnp.where(sel, s.uid, BIG)).astype(I32), zctx=_i(ZC_PEND), dec=_i(D_ZONE),
                              dec_player=s.tp, psel=_i(0))

        def fin(s):
            g = (s.st & ST_PSTEP) != 0
            s = raise_ev(brk(enable(s, g)), EV_SPSUMMON, g, s.tp)
            s = s._replace(st=s.st & ~ST_PSTEP, psel=_i(0), pend=_i(-1), ps_used=jnp.bool_(True))
            return trig_point(bank, open_quick(s, s.tp, R_IDLE, chained=False))
        return _cond(sel.any(), go, fin, s)

    def pend_place(s):
        c = s.pend
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(SUMMON_PENDULUM,s.zone[c],s.tp)), ss_t=s.ss_t.at[c].set(True))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.tp, pos)
        s = s._replace(proc=s.proc.at[c].set(True), st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM | ST_PSTEP))
        return pend_next(s)

    def position(s):
        return _switch(jnp.where(s.zctx == ZC_PROC, 1, jnp.where(s.zctx == ZC_PEND, 2, 0)),
                       [eff_position, proc_place, pend_place], s)

    def eff_position(s):
        c = s.pend
        pos = jnp.where(k == A_POS_ATK, P_FU_ATK, P_FU_DEF)
        # Only an explicit procedure Summon is properly completed.
        proper = current_ins(bank, s)[3]
        completed = proper > 0
        typ = effect_summon_type(proper)
        s = s._replace(suminfo=s.suminfo.at[c].set(summon_info(typ,s.zone[c],s.ch_p)), ss_t=s.ss_t.at[c].set(True))
        s = compact(s, jnp.arange(n) == c)
        s = move(s, c, Z_MZONE, s.zsel, s.ch_p, pos)
        s = s._replace(proc=s.proc.at[c].set(s.proc[c] | completed))
        # Same instruction until ST_SEL is exhausted; move removed the placed card
        # from this set. Its effects stay suspended until the end.
        return s._replace(st=s.st.at[c].set(ST_SUMMON_TURN | ST_SPSUM | ST_PSTEP), dec=_i(D_RUN))

    def to_ep(s):
        return open_quick(s, 1 - s.tp, jnp.where(s.dec == D_BATTLE, R_TO_EPB, R_TO_EP), passed=True, chained=False)

    def spec(s):
        yn = s.dec == D_YESNO  # yes: the effect continues; no (A_PASS): it stops (last instruction, OP_END)
        rq = s.zctx == ZC_REPLAY  # attack replay: yes -> new target, no -> end of the attack
        inline = s.zctx == ZC_INLINE
        def inline_yes(s):
            if not bank.einline.shape[1]:
                return s
            e = bank.einline[s.cid[s.ch_src], s.ch_eff] - 1
            s = use_count(bank, s, s.ch_src, e, s.ch_p, enabled=yn & inline & (k == A_YES))
            return s._replace(ch_eff=e, pc=_i(0), pg=_i(PG_COST), dec=_i(D_RUN), zctx=_i(ZC_ACT))
        return _switch(jnp.where(yn, jnp.where(inline, jnp.where(k == A_YES, 10, 11),
                                 jnp.where(rq, jnp.where(k == A_YES, 8, 9), jnp.where(k == A_YES, 6, 7))),
                                 jnp.where(k < A_ZONE0, k, jnp.where(k < A_POS_ATK, 3, jnp.where(k < A_PASS, 4, 5)))), [
            lambda s: open_quick(s, 1 - s.tp, R_TO_BP, passed=True, chained=False),
            lambda s: open_quick(s, 1 - s.tp, R_TO_M2B, passed=True, chained=False),
            to_ep, zone, position,
            lambda s: _cond(s.dec == D_TRIBUTE,  # end of the Link material selection: to the GY, then the zone
                            lambda s: _cond(s.psel == PS_PEND, pend_next,  # Pendulum: placement card by card
                                            lambda s: selected_to_grave(bank, s, REASON_LINK | REASON_MATERIAL)._replace(dec=_i(D_ZONE), psel=_i(0)), s),
                            lambda s: _cond(s.dec == D_SELECT, lambda s: sel_finish(bank, s), lambda s: _cond(s.wk == W_TRIG, lambda s: trig_next(bank, s._replace(tst=s.tst + 1)),
                                            lambda s: pass_priority(bank, s), s), s), s),
            lambda s: s._replace(pc=s.pc + 1, dec=_i(D_RUN)),
            lambda s: s._replace(pc=jnp.where(block_end(bank, s) >= 0, block_end(bank, s), P - 1), dec=_i(D_RUN)),
            replay_go, battle_over,
            inline_yes, lambda s: s._replace(pg=_i(PG_TARGET), pc=_i(P - 1), dec=_i(D_RUN), zctx=_i(ZC_ACT)),
        ], s)

    s = _cond((s.dec == D_IDLE) | (s.dec == D_BATTLE), clear_ev, lambda s: s, s)
    return _switch(jnp.where(verb < V_ACT0, verb, jnp.where(verb < NV, V_ACT0, V_ACT0 + 1)), [
        lambda s: summon(s, False), lambda s: summon(s, True), repos, attack, select, sset,
        activate, spec], s)


MAX_ITER = 256
# share of playing games below which the settle of the first round of step_batch switches to sub-batches
# (0 = never; 0.25 = size of the sub-batch, optimum measured on L4)
SETTLE_SUB = float(os.environ.get("YGO_SETTLE_SUB", "0.25"))
# the two passes of a settle round in a single heavy call (passes_merged); YGO_PASS_MERGE=0: old path
PASS_MERGE = os.environ.get("YGO_PASS_MERGE", "1") != "0"


def forced(bank, s):
    """Forced action (single option, or selection where all the possible cards must be taken), otherwise -1."""
    return forced_m(bank, s, legal(bank, s))


def forced_m(bank, s, m):
    k = m.sum()
    sel = ((s.dec == D_TRIBUTE) & (s.psel == 0)) | (s.dec == D_DISCARD) | ((s.dec == D_SELECT) & ~s.fsel & (s.rsel == 0)
                                                                            & ~sel_var(s))
    # Ritual Tributes: played automatically if the union of the possible choices is itself the only valid set
    # (the oracle then has a single option): it reaches the sum with no superfluous card (Greater) or exactly (Equal)
    n = s.cid.shape[0]
    cand = m[V_SELECT * n:(V_SELECT + 1) * n] | ((s.st & ST_SEL) != 0)
    lv = jnp.clip(s.k_level, 0, LVMAX)
    tot = jnp.where(cand, lv, 0).sum()
    mn = jnp.where(cand, lv, LVMAX + 1).min()
    rit = jnp.argmax(s.reg[jnp.clip(current_ins(bank, s)[2], 0, R - 1)])
    req = bank.level[s.cid[rit]]
    # only with a free main zone (SelectWithSum, one decision); without a free zone, ygopro chooses card by
    # card (SelectUnselectGroup) and offers each card of the set
    room = ex_zones(s, s.dec_player)[:N_MAIN_ZONES].any()
    uniq = (s.rsel > 0) & (s.dec == D_SELECT) & room & jnp.where(s.rsel == RIT_GREATER, (tot >= req) & (tot - mn < req),
                                                          tot == req)
    return jnp.where((k == 1) | (sel & (k > 0) & (k <= s.need)) | (uniq & (k > 0)), jnp.argmax(m), -1).astype(I32)


def run(bank, s):
    """Executes the effects (D_RUN) until the next choice."""
    def body(carry):
        s, i = carry
        return refresh_if(bank, s, exec_one(bank, s)), i + 1
    return lax.while_loop(lambda c: (c[0].dec == D_RUN) & ~c[0].done & (c[1] < MAX_ITER), body, (s, _i(0)))[0]


def settle(bank, s):
    """Until the next real choice: run, pending triggers (trig_eval), forced passes (auto_pass)."""
    def body(carry):
        s, i = carry
        s = run(bank, s)
        s = lax.cond(s.dec == D_TRIG, lambda s: trig_eval(bank, s), lambda s: s, s)
        return refresh_if(bank, s, lax.cond(auto_pass(bank, s), lambda s: pass_priority(bank, s), lambda s: s, s)), i + 1
    more = lambda c: (((c[0].dec == D_RUN) | (c[0].dec == D_TRIG) | auto_pass(bank, c[0])) & ~c[0].done
                      & (c[1] < MAX_ITER))
    return lax.while_loop(more, body, (s, _i(0)))[0]


def step(bank, s, a):
    """Applies a, then executes the effects and automatically plays the single-option decisions.
    Two loops: under vmap, a lax.cond computes both its branches, so legal (expensive) is only evaluated at the
    decision points, not at every executed instruction."""
    def body(carry):
        s, a, i = carry
        s = refresh_ex(bank, settle(bank, refresh_if(bank, s, apply(bank, s, a))))
        return s, forced(bank, s), i + 1

    # no initial settle: s is always at a decision (output of step or new_game), and a second copy of
    # settle would almost double the size of the compiled program (about 40% of the HLO of step)
    s, _, _ = lax.while_loop(lambda c: (c[1] >= 0) & ~c[0].done & (c[2] < MAX_ITER), body, (s, _i(a), _i(0)))
    return s


def _where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


def step_batch(bank, s, a, frac=0.0, stats=False):
    """Batch version of step, for training: a[b] = action of game b (policy, or the forced action returned
    by the previous call). Returns (states, legal masks, pending forced actions or -1).

    The loops and the rare computations (trig_eval) are decided at the batch level. The forced-action loop stops
    as soon as the share of games that still have one is <= frac: otherwise the whole batch waits for the slowest.
    The games stopped along the way resume at the next call with their forced action. Likewise, the execution and
    settle loops stop as soon as only the share frac of the playing games remains in progress (otherwise the whole
    batch waits for the tail of the long resolutions, which grows with B): these games are returned mid-resolution,
    with the forced action n_actions(N) ("continue", without apply). frac = 0: same result as vmap(step).
    stats: also returns an int32[11] counter (throughput diagnostic): rounds of the outer loop, of settle_b, of run_b,
    trig_eval taken, passes taken, second passes taken, legal with fusion; then the sum over the rounds of the number
    of active games: run_b, trig_eval, first pass, second pass.

    After the first round (full batch), the forced-action rounds only cover a sub-batch of B/4 games
    (those that have a forced action, the others wait for the next round). In settle, when the first player passes
    without being able to do anything, the second pass (opponent who cannot do anything either) is played in the same
    round."""
    nb = s.done.shape[0]
    na = n_actions(s.cid.shape[1])

    def apb(s, mask, prev):
        """auto_pass recomputed only for the games in mask (sub-batch of B/8, or the whole batch if there are more),
        prev elsewhere: auto_pass tests act_ok, too expensive for the whole batch at every settle round."""
        n = s.done.shape[0]
        ks = max(1, n // 8)
        full = lambda s: jax.vmap(lambda s: auto_pass(bank, s))(s)

        def sub(s):
            idx = jnp.nonzero(mask, size=ks, fill_value=n)[0]
            part = full(jax.tree.map(lambda x: x[jnp.minimum(idx, n - 1)], s))
            return jnp.zeros(n, bool).at[idx].set(part, mode="drop")
        r = lax.cond(mask.sum() > ks, full, lambda s: lax.cond(mask.any(), sub, lambda s: jnp.zeros(n, bool), s), s)
        return jnp.where(mask, r, prev)

    def todo(s, ap):  # ap: auto_pass(s) up to date for all the games
        return ((s.dec == D_RUN) | (s.dec == D_TRIG) | ap) & ~s.done

    def run_b(s, thr):  # batch-level execution loop: refresh only for the modified games
        def body(c):
            s, i, n = c
            r = (s.dec == D_RUN) & ~s.done
            return refresh_b(bank, s, _where(r, jax.vmap(lambda s: exec_one(bank, s))(s), s)), i + 1, n + r.sum()
        c = lax.while_loop(lambda c: (((c[0].dec == D_RUN) & ~c[0].done).sum() > thr) & (c[1] < MAX_ITER), body,
                           (s, _i(0), _i(0)))
        return c[0], c[1], c[2]

    def settle_body(c):  # invariant: ap == auto_pass(s) for all the games, on entry and on exit
        s, i, thr, k, ap = c
        r0 = ((s.dec == D_RUN) | (s.dec == D_TRIG)) & ~s.done   # the only games that run_b and trig_eval modify
        s, nr, ar = run_b(s, thr)
        need = s.dec == D_TRIG
        k = k + jnp.array([0, 1, nr, need.any(), 0, 0, 0, ar, need.sum(), 0, 0], I32)
        s = lax.cond(need.any(), lambda s: _where(need, jax.vmap(lambda s: trig_eval(bank, s))(s), s),
                     lambda s: s, s)
        ap = apb(s, r0 | need, ap)

        def passes_merged(s):
            """Same states as `passes`, with a single heavy call per round. A player's first pass only
            hands over priority (cw_passed, dec_player): it is written directly, without vmap(pass_priority), which
            also computes the window closing for the whole batch. The closing (pass_close) then serves, in a single
            call, the games for which it is the second pass and those that chain the two."""
            first = ap & ~s.cw_passed
            s1 = s._replace(cw_passed=s.cw_passed | first, dec_player=jnp.where(first, 1 - s.dec_player, s.dec_player))
            ap2 = first & apb(s1, first, jnp.zeros_like(ap))
            cl = (ap & s.cw_passed) | ap2
            s2 = lax.cond(cl.any(), lambda s1: _where(cl, jax.vmap(lambda s: pass_close(bank, s))(s1), s1), lambda s1: s1, s1)
            s2 = refresh_b(bank, s, s2)
            return s2, apb(s2, ap, ap & False), ap2.sum()

        def passes(s):  # first pass, then the second right away if the opponent cannot do anything either
            if PASS_MERGE:
                return passes_merged(s)
            s1 = _where(ap, jax.vmap(lambda s: pass_priority(bank, s))(s), s)
            ap2 = ap & ~s.cw_passed & apb(s1, ap & ~s.cw_passed, jnp.zeros_like(ap))
            s2 = lax.cond(ap2.any(), lambda s1: _where(ap2, jax.vmap(lambda s: pass_priority(bank, s))(s1), s1), lambda s1: s1, s1)
            s2 = refresh_b(bank, s, s2)
            return s2, apb(s2, ap, ap & False), ap2.sum()
        s, ap_new, p2 = lax.cond(ap.any(), passes, lambda s: (s, ap, _i(0)), s)
        return s, i + 1, thr, k + jnp.array([0, 0, 0, 0, ap.any(), p2 > 0, 0, 0, 0, ap.sum(), p2], I32), ap_new

    def settle_b(s, thr, k, ap):
        c = lax.while_loop(lambda c: (todo(c[0], c[4]).sum() > thr) & (c[1] < MAX_ITER), settle_body,
                           (s, _i(0), thr, k, ap))
        return c[0], c[3], c[4]

    def rnd(s, m, a, k, sub=0.0):  # one round: actions a (>= 0) applied, resolution, next masks and forced actions
        # sub: settle stops as soon as the share sub of the playing games remains in progress (see SETTLE_SUB)
        act = (a >= 0) & ~s.done
        s1 = jax.vmap(lambda s, a: apply(bank, s, a))(s, jnp.clip(a, 0, na - 1))
        s2 = refresh_b(bank, s, _where(act & (a < na), s1, s))
        s2, k, ap = settle_b(s2, max(frac, sub) * act.sum(), k, apb(s2, act, jnp.zeros_like(act)))
        s = _where(act, s2, s)
        left = act & todo(s, ap)                  # stopped mid-resolution: "continue" at the next call
        s = refresh_ex_b(bank, s, act & ~left)
        fl = (s.fsel | (s.rsel > 0) | (s.psel > 0)).any()
        lg = lax.cond(fl, lambda s: jax.vmap(lambda s: legal(bank, s))(s),
                      lambda s: jax.vmap(lambda s: legal(bank, s, fus=False))(s), s)
        m = jnp.where(act[:, None], lg, m)
        f = jnp.where(left, na, jax.vmap(lambda s, m: forced_m(bank, s, m))(s, m))
        return s, m, jnp.where(act, f, -1), k + jnp.array([1, 0, 0, 0, 0, 0, fl, 0, 0, 0, 0], I32)

    ks = max(1, nb // 4)

    def body(carry):  # following rounds: only the games that have a forced action, in a sub-batch of B/4 (the
        s, m, f, i, k = carry  # others wait for the next round; after the first round, ~30% of the games have one)
        idx = jnp.nonzero((f >= 0) & ~s.done, size=ks, fill_value=nb)[0]
        g = lambda x: x[jnp.minimum(idx, nb - 1)]
        ss, ms, fs, k = rnd(jax.tree.map(g, s), g(m), jnp.where(idx < nb, g(f), -1), k)
        put = lambda x, y: x.at[idx].set(y, mode="drop")
        return jax.tree.map(put, s, ss), put(m, ms), put(f, fs), i + 1, k

    def more(c):
        pend = ((c[2] >= 0) & ~c[0].done).sum()
        return (pend > frac * nb) & (c[3] < MAX_ITER)

    # no initial settle_b (the input states are at a decision). rnd is compiled twice (full batch, sub-batch):
    # compilation ~2x longer and memory peak ~2x (5.8 GB on CPU), for ~+30% throughput on L4
    # SETTLE_SUB > 0: the first round (full batch) hands back as soon as only the share SETTLE_SUB of the playing
    # games remains in mid-resolution (relative share, like frac: an absolute threshold blocked the last games
    # of an almost finished batch); they continue ("continue") in the following rounds, in a sub-batch of B/4, where a
    # settle round costs less. Same sequence of decisions per game (sb_check).
    s, m, f, k = rnd(s, jnp.zeros((nb, n_actions(s.cid.shape[1])), bool), a.astype(I32), jnp.zeros(11, I32),
                     SETTLE_SUB)
    s, m, f, _, k = lax.while_loop(more, body, (s, m, f, _i(1), k))
    return (s, m, f, k) if stats else (s, m, f)

def new_game(bank, deck0, deck1, key=None, extra0=None, extra1=None, extra_counts=None):
    """deck0/deck1: [D] indices into the bank, in order of addition (last = top of the deck).
    extra0/extra1: [X] Extra Deck (same capacity for both players; None: none), in order of addition.
    extra_counts: [2] optional real sizes (0..X); the remaining slots have zone=0.
    Allows a single compilation for Extra Decks of different sizes, without dummy cards in play.
    Returns the state at the start of the first turn, whose first decision may be single-option: the caller plays it
    with step (forced(bank, s) if >= 0), or draws a legal action (which is then the only one).

    key=None: fixed order, slots in deck order (to match the oracle in DUEL_PSEUDO_SHUFFLE).
    With a key: decks shuffled and cards spread at random over the slots, so that neither the order of the deck
    nor the slot number gives away a hidden card.
    """
    d = deck0.shape[0]
    x = 0 if extra0 is None else extra0.shape[0]
    nd = 2 * d
    n = nd + 2 * x
    if n > MAX_SLOTS:
        raise ValueError(f"arena capacity exceeded: {n} > {MAX_SLOTS}")
    cid = jnp.concatenate([deck0, deck1] + ([] if x == 0 else [extra0, extra1])).astype(I32)
    owner = jnp.concatenate([jnp.repeat(jnp.arange(2, dtype=I32), d), jnp.repeat(jnp.arange(2, dtype=I32), x)])
    seq = jnp.concatenate([jnp.tile(jnp.arange(d, dtype=I32), 2), jnp.tile(jnp.arange(x, dtype=I32), 2)])
    zone0 = jnp.concatenate([jnp.full(nd, Z_DECK, I32), jnp.full(2 * x, Z_EXTRA, I32)])
    if extra_counts is not None and x:
        active_extra = jnp.tile(jnp.arange(x, dtype=I32), 2) < jnp.repeat(jnp.asarray(extra_counts, I32), x)
        zone0 = zone0.at[nd:].set(jnp.where(active_extra, Z_EXTRA, 0))
    ar = lambda k: jnp.arange(k, dtype=I32)
    uid = jnp.concatenate([ar(d), d + x + ar(d), d + ar(x), 2 * d + x + ar(x)])
    if key is not None:
        k1, k2 = jax.random.split(key)
        own = owner[:nd]
        order = jnp.lexsort((jax.random.uniform(k1, (nd,)), own))  # shuffle within each deck
        seq = seq.at[:nd].set(jnp.zeros(nd, I32).at[order].set(jnp.arange(nd, dtype=I32)) - own * d)
        perm = jax.random.permutation(k2, n)
        cid, owner, seq, zone0, uid = cid[perm], owner[perm], seq[perm], zone0[perm], uid[perm]
    z = jnp.zeros(n, I32)
    f = jnp.bool_(False)
    s = State(reason=z, suminfo=z, cid=cid, zone=zone0,
              seq=seq, ctrl=owner, owner=owner, pos=jnp.full(n, P_FD_DEF, I32),
              st=z, lp=jnp.full(2, 8000, I32), turn=_i(0), tp=_i(1), phase=_i(PH_DRAW), ns_used=f,
              dec=_i(D_NONE), dec_player=_i(0), pend=_i(-1), pend_set=f, need=_i(0), picks=_i(0), smin=_i(0),
              zctx=_i(ZC_SUMMON), zsel=_i(0), ch_src=_i(-1), ch_src_mv=_i(0), ch_tgt_t=_i(0), ch_targets=jnp.zeros(n, bool), ch_eff=_i(0), ch_p=_i(0), ch_k=_i(0), ch_dis=f, ch_lab=_i(0), ch_flags=_i(0), pg=_i(0),
              pc=_i(0), reg=jnp.zeros((R, n), bool), done=f, odraw=f, winner=_i(-1),
              lk_src=jnp.full(CH, -1, I32), lk_eff=jnp.zeros(CH, I32), lk_p=jnp.zeros(CH, I32),
              lk_reg=jnp.zeros((CH, R, n), bool), lk_targets=jnp.zeros((CH, n), bool), lk_neg=jnp.zeros(CH, bool), lk_t=jnp.zeros(CH, I32), lk_tgt_t=jnp.zeros(CH, I32), lk_lab=jnp.zeros(CH, I32), lk_flags=jnp.zeros(CH, I32), mv=z, clk=_i(0), n_chain=_i(0),
              wk=_i(W_QUICK), cw_passed=f, ret=_i(R_IDLE), chained=f, ph_restart=_i(0), wph=_i(WP_DRAW),
              bend_ep=f, atk=_i(-1), atk_tgt=_i(-1), ev_g=jnp.zeros((NEV, n), bool), ev_p=jnp.zeros(NEV, I32),
              pzone=zone0, tev=jnp.zeros((NEV, n), bool), tew=jnp.zeros((NEV, n), bool), teg=jnp.zeros((NEV, n), bool), tg=jnp.zeros(n, bool), tg_e=z,
              tst=_i(0), pt_p=_i(0), pt_passed=f, dstep=_i(DS_NONE), ds_t=_i(0), ds_flip=f, rb_m=jnp.zeros(n, bool), rb_t=_i(0), rpl=f, opt_cid=jnp.zeros((2, n * E), I32),
              lock_cid=jnp.zeros((2, n), I32), ssl=jnp.zeros((SSL, 5), I32), ss_t=jnp.zeros(n, bool), atk_c=bank.atk[cid], df_c=bank.df[cid], batk_c=bank.atk[cid], tm_code=jnp.zeros(TM, I32),
              tm_m=jnp.zeros((TM, n), bool), tm_v=jnp.zeros((TM, n), I32), tm_t=jnp.zeros(TM, I32), tm_dur=jnp.zeros(TM, I32), tm_rd=jnp.zeros(TM, bool), tm_src=jnp.full(TM, -1, I32), tm_n=_i(0),
              ovf=_i(0), en_t=z, fl_c=z, ib_c=jnp.full(n, BIG, I32), pf_c=jnp.zeros(2, I32), cav=jnp.zeros((n, E), bool), cav_t=jnp.zeros((n, E), I32),
              proc=jnp.zeros(n, bool), fz_c=jnp.zeros(n, bool), fsel=f, rz_c=z, rsel=_i(0), sz_c=jnp.zeros(n, bool),
              xz_c=jnp.zeros(n, bool), lz_c=jnp.zeros(n, bool), cl_c=jnp.zeros(n, bool), psel=_i(0), ov_par=jnp.full(n, -1, I32), eq_t=jnp.full(n, -1, I32), eq_mv=z, ps_used=f, uid=uid, **slot_fields(bank, cid))
    for p in (0, 1):
        for _ in range(5):
            s = draw(s, p)
    # refresh after the draw: fz_c / rz_c depend on the hand; before, they stayed stale until the first state
    # change (step kept them, step_batch recomputed them by refreshing the whole batch)
    s = refresh(bank, s)
    # no step here: it would recompile the whole engine a second time (about 2 GB more compilation). The first
    # decision may be single-option (Draw Phase window): the caller plays it with its own step
    # (forced, or a randomly drawn legal action, which is then the only one)
    return refresh_ex(bank, start_turn(s))
