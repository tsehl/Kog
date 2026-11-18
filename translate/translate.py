"""Translator: official Lua scripts -> brick effects (format of engine/cards.py).

Symbolic interpreter over the luaparser tree. It reads initial_effect (Effect.CreateEffect, Set*, Clone,
RegisterEffect), then the condition, cost, target and operation functions, which it translates pattern by pattern:
- activation tests (`if chk==0 then return A and B end`) -> FILTER / REQ / REQ_FREE / COND_*;
- selections (SelectMatchingCard, SelectTarget, DiscardHand...) -> FILTER + CHOOSE;
- actions (Destroy, SendtoHand, SendtoGrave, Remove, SpecialSummon, Draw, Damage, Recover...) -> bricks;
- filter functions (`return c:IsX() and c:IsY()`) -> fields of F(...).
Anything outside the known patterns raises Unsupported(reason): the ranking of reasons (by number of cards) says
what to add next, to the translator or to the engine. A translated card is only "verified" after identical games
against the oracle (translate/validate.py).

Usage: uv run --with luaparser --with jax python translate/translate.py [--codes 55144522,...] [--top 60] [--show]"""
import argparse
import collections
import glob
import os
import re
import sys
import sqlite3
from functools import lru_cache
from contextlib import closing

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "engine")]
SCRIPTS = os.path.join(HERE, "..", "data", "CardScripts")
OFFICIAL = os.path.join(SCRIPTS, "official")

from luaparser import ast as LA  # noqa: E402
from luaparser import astnodes as A  # noqa: E402

import cards as C  # noqa: E402
import core  # noqa: E402

ME, OPP = 0, 1
ALLZ = core.Z_DECK | core.Z_HAND | core.Z_MZONE | core.Z_SZONE | core.Z_GRAVE | core.Z_REMOVED | core.Z_EXTRA
MON_TYPES = (core.TYPE_MONSTER | core.TYPE_NORMAL | 0x20 | core.TYPE_FUSION | core.TYPE_RITUAL | core.TYPE_SYNCHRO
             | core.TYPE_XYZ | core.TYPE_LINK | core.TYPE_TUNER | core.TYPE_PENDULUM | 0x200 | 0x400 | 0x800 | 0x4000
             | 0x200000 | 0x400000 | 0x2000000)  # SPIRIT, UNION, GEMINI, TOKEN, FLIP, TOON, SPSUMMON
SPELL_SUB = 0x10000 | 0x20000 | 0x40000 | 0x80000 | core.TYPE_RITUAL  # QUICKPLAY, CONTINUOUS, EQUIP, FIELD
TRAP_SUB = 0x20000 | 0x100000


@lru_cache(maxsize=1)
def card_metadata():
    path = os.path.join(HERE, "..", "data", "BabelCDB", "cards.cdb")
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return {code:(typ,level&255) for code,typ,level in db.execute("SELECT id,type,level FROM datas")}


def card_type(code):
    row = card_metadata().get(code)
    need(row is not None, "card type missing for EnableReviveLimit")
    return row[0]


class Unsupported(Exception):
    pass


def need(cond, reason):
    if not cond:
        raise Unsupported(reason)


# ---------------------------------------------------------------- constants

def _const_eval(node, env):
    if isinstance(node, A.Number):
        return node.n
    if isinstance(node, A.Name):
        return env[node.id]
    if isinstance(node, A.BOrOp) or isinstance(node, A.AddOp):
        return _const_eval(node.left, env) | _const_eval(node.right, env) if isinstance(node, A.BOrOp) \
            else _const_eval(node.left, env) + _const_eval(node.right, env)
    if isinstance(node, A.BAndOp):
        return _const_eval(node.left, env) & _const_eval(node.right, env)
    if isinstance(node, A.BShiftLOp):
        return _const_eval(node.left, env) << _const_eval(node.right, env)
    if isinstance(node, A.SubOp):
        return _const_eval(node.left, env) - _const_eval(node.right, env)
    if isinstance(node, A.MultOp):
        return _const_eval(node.left, env) * _const_eval(node.right, env)
    if isinstance(node, A.UMinusOp):
        return -_const_eval(node.operand, env)
    if isinstance(node, A.UBNotOp):
        return ~_const_eval(node.operand, env)
    raise KeyError(type(node).__name__)


def load_constants():
    env = {}
    for f in ("constant.lua", "archetype_setcode_constants.lua", "card_counter_constants.lua"):
        p = os.path.join(SCRIPTS, f)
        if not os.path.exists(p):
            continue
        for st in LA.parse(open(p, encoding="utf-8", errors="replace").read()).body.body:
            if isinstance(st, (A.Assign, A.LocalAssign)) and len(st.targets) == 1 and isinstance(st.targets[0], A.Name):
                try:
                    env[st.targets[0].id] = _const_eval(st.values[0], env)
                except (KeyError, AttributeError, TypeError):
                    pass
    return env


K = load_constants()

# ---------------------------------------------------------------- symbolic values


class Player:  # player relative to the activating player (ME / OPP)
    def __init__(self, p): self.p = p
    def __repr__(self): return f"P{self.p}"


class Desc(int):  # aux.Stringid(id, k) = (id << 20) | (k & 0xfffff) (effect description, the one the oracle shows)
    pass


class Handler:  # the effect's card (c, e:GetHandler())
    def __repr__(self): return "c"


class Eff:  # the current effect (e)
    pass


class Arg:  # parameter without a symbolic value (eg, ep, ev, re, r, rp, chk…)
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name


class Fn:  # Lua function (node) or known reference (Card.IsFaceup, aux.TRUE…)
    def __init__(self, node=None, name=None, bound=()):
        self.node, self.name, self.bound = node, name, tuple(bound)
    def __repr__(self): return self.name or "fn"


class Reg:  # set of cards in a program register; originating locations (me, op)
    def __init__(self, r, me=ALLZ, op=ALLZ, one=False, handler=False, target=False):
        self.r, self.me, self.op, self.one, self.handler, self.target = r, me, op, one, handler, target
    def __repr__(self): return f"R{self.r}"


class Lazy:  # GetMatchingGroup not yet evaluated (only emitted if used)
    def __init__(self, F): self.F = F


class Count:  # number of cards in a register x mul
    def __init__(self, reg, mul=1): self.reg, self.mul = reg, mul


class FreeCount:  # number of free zones, kept symbolically in a local variable
    def __init__(self, player, zone): self.player, self.zone = player, zone


class ChainInfo:  # CHAININFO_TARGET_PLAYER / PARAM
    def __init__(self, what): self.what = what


class Attacker:
    pass


class EventGroup:  # eg
    pass


class ChainEff:  # re (effect of the previous chain link); re:GetHandler()
    def __init__(self, card=False): self.card = card


class NewEff:  # Effect.CreateEffect inside an operation (temporary modifier)
    def __init__(self): self.f = {}


# ---------------------------------------------------------------- filters

def merge_typ(a, b):
    if a == 0 or a == b:
        return b
    if b == 0:
        return a
    for gen, sub in ((core.TYPE_MONSTER, MON_TYPES), (core.TYPE_SPELL, SPELL_SUB), (core.TYPE_TRAP, TRAP_SUB)):
        if a == gen and b & ~sub == 0:
            return b
        if b == gen and a & ~sub == 0:
            return a
    raise Unsupported("filter: two type conditions")


class Filt:
    """Fields of an F(...) under construction (all ANDed)."""

    def __init__(self):
        self.d = dict(pos=0, typ=0, not_self=0, atk=[-core.BIG, core.BIG], df=[-core.BIG, core.BIG], attr=0, race=0,
                      setc=0, code=None, st=0, lv=[-core.BIG, core.BIG], spsum=0, nodis=0, zx=0, reason=0, not_typ=0)
        self.loc = ALLZ  # location restriction (IsLocation)
        self.stay = False  # IsAbleToHand / IsAbleToDeck: not a card that leaves at the end of the chain (core.ST_STAY)
        self.ctrl = None  # IsControler
        self.tgt = False

    def set(self, k, v):
        d = self.d
        if k == "typ":
            d["typ"] = merge_typ(d["typ"], v)
        elif k == "pos":
            d["pos"] = v if d["pos"] == 0 else d["pos"] & v
            need(d["pos"] != 0, "filter: incompatible positions")
        elif k in ("atk", "df", "lv"):
            lo, hi = v
            d[k] = [max(d[k][0], lo), min(d[k][1], hi)]
            if k != "lv" and d["typ"] == 0:
                d["typ"] = core.TYPE_MONSTER
        elif k in ("attr", "race", "setc", "st"):
            need(d[k] == 0 or d[k] == v, f"filter: two conditions {k}")
            d[k] = v
        elif k == "not_typ":
            need(type(v) is int and 0 < v < (1 << 31), "filter: excluded types")
            d[k] |= v
        elif k == "reason":
            need(type(v) is int and 0 < v <= 0x1fffffff, "filter: reason mask")
            old = d[k]
            if old == 0:
                d[k] = v
            elif old == v:
                pass
            elif old > 0 and old & v == v:
                d[k] = v
            elif old > 0 and old & v == old:
                pass
            elif old > 0 and old & (old - 1) == 0 and v & (v - 1) == 0:
                d[k] = -(old | v)
            elif old < 0 and (-old & v) != 0:
                pass
            elif old < 0 and v & (v - 1) == 0:
                d[k] = -(-old | v)
            else:
                raise Unsupported("filter: complex conjunction of reasons")
        elif k == "code":
            need(d["code"] is None, "filter: two name conditions")
            d["code"] = v
        else:
            d[k] = v

    def build(self, me, op):
        d = dict(self.d)
        me, op = me & self.loc, op & self.loc
        if self.ctrl == ME:
            op = 0
        elif self.ctrl == OPP:
            me = 0
        if d["code"] is None:
            d["code"] = (0, 0)
        if self.stay and d["st"] == 0 and (me | op) & (core.Z_MZONE | core.Z_SZONE):
            # only a card on the field can leave at the end of a chain; a status already requested wins
            # (an old approximation, rare)
            d["st"] = core.ST_STAY
        for k in ("atk", "df", "lv"):
            d[k] = tuple(d[k])
        if d["reason"]:
            tracked = core.Z_GRAVE | core.Z_REMOVED | core.Z_EXTRA
            need((me | op) & ~tracked == 0, "filter: reason outside tracked locations")
        if d["zx"] & core.ZX_EXTRA:
            need(d["typ"] != 0 and d["typ"] & ~(core.TYPE_FUSION | core.TYPE_SYNCHRO | core.TYPE_XYZ) == 0,
                 "GetLocationCountFromEx: filter outside Fusion / Synchro / Xyz")
        return C.F(me=me, op=op, **d)


SUMMON_METHODS = {"IsNormalSummoned": core.SUMMON_NORMAL, "IsTributeSummoned": core.SUMMON_TRIBUTE,
                  "IsSpecialSummoned": core.SUMMON_SPECIAL, "IsFusionSummoned": core.SUMMON_FUSION,
                  "IsRitualSummoned": core.SUMMON_RITUAL, "IsSynchroSummoned": core.SUMMON_SYNCHRO,
                  "IsXyzSummoned": core.SUMMON_XYZ, "IsPendulumSummoned": core.SUMMON_PENDULUM,
                  "IsLinkSummoned": core.SUMMON_LINK}


CARD_METHODS_IGNORED = {
    "IsAbleToHand", "IsAbleToGrave", "IsAbleToRemove", "IsAbleToDeck", "IsAbleToExtra", "IsAbleToGraveAsCost",
    "IsAbleToHandAsCost", "IsAbleToRemoveAsCost", "IsAbleToDeckAsCost", "IsDestructable", "IsDiscardable",
    "IsReleasable", "IsReleasableByEffect", "IsCanBeEffectTarget", "IsAbleToDeckOrExtraAsCost", "IsOnField",
    "CanAttack"}


# IsCompositeType captures the native predicate (all bits), before the
# Lua redefinition of IsExactType. In the catalog, these subtypes imply
# their primary type; same reduction as merge_typ(MONSTER, subtype).
COMPOSITE_TYPES = {"IsFusionMonster": core.TYPE_FUSION, "IsSynchroMonster": core.TYPE_SYNCHRO,
                   "IsXyzMonster": core.TYPE_XYZ, "IsLinkMonster": core.TYPE_LINK,
                   "IsFieldSpell": K["TYPE_FIELD"]}
# RITUAL and CONTINUOUS are shared: the bit alone does not tell the
# primary type. The other primary types are excluded via F_NOTTYPE.
# Invariant of cards carrying these bits: exactly one primary type, non-zero.
SHARED_COMPOSITE_TYPES = {
    "IsRitualMonster": (core.TYPE_RITUAL, core.TYPE_MONSTER),
    "IsRitualSpell": (core.TYPE_RITUAL, core.TYPE_SPELL),
    "IsContinuousSpell": (core.TYPE_CONTINUOUS, core.TYPE_SPELL),
    "IsContinuousTrap": (core.TYPE_CONTINUOUS, core.TYPE_TRAP),
}
FILTER_METHODS = {"IsCanBeSpecialSummoned", "IsFaceup", "IsFacedown", "IsAttackPos", "IsDefensePos", "IsPosition"} | COMPOSITE_TYPES.keys() | SHARED_COMPOSITE_TYPES.keys()


def generic_summon_tag(value):
    need(type(value) is int and value >= 0 and (value & ~(core.SUMMON_SPECIAL | 65535)) == 0,
         "Special Summon: type not represented")
    return value & 65535


def apply_method(fl, name, args):
    """c:name(args) in a filter; args already evaluated (constants)."""
    def num(i):
        need(len(args) > i and isinstance(args[i], int), f"filter: {name} without constant")
        return args[i]
    args = list(args)
    while args and args[-1] is None:  # c:IsAttribute(a, lc, sumtype, tp) (materials): unbound context
        args.pop()
    if name == "IsReason":
        need(len(args) == 1, "filter: IsReason arguments")
        fl.set("reason", num(0))
        return
    if name in SUMMON_METHODS or name == "IsSummonType":
        need(len(args) == (1 if name == "IsSummonType" else 0), "summon type: arguments")
        typ = SUMMON_METHODS[name] if name in SUMMON_METHODS else num(0)
        need(typ & 0x80ff0000 == 0, "summon type: mask not represented")
        fl.set("st", core.summon_mask(typ))
        return
    if name in CARD_METHODS_IGNORED:
        if name == "IsCanBeEffectTarget":
            fl.tgt = True
        if name in ("IsAbleToHand", "IsAbleToDeck", "IsAbleToExtra", "IsAbleToHandAsCost", "IsAbleToDeckAsCost",
                    "IsAbleToDeckOrExtraAsCost"):
            fl.stay = True
        return
    if name in COMPOSITE_TYPES:
        need(not args, f"filter: {name} arguments")
        fl.set("typ", COMPOSITE_TYPES[name])
    elif name in SHARED_COMPOSITE_TYPES:
        need(not args, f"filter: {name} arguments")
        specialty, primary = SHARED_COMPOSITE_TYPES[name]
        fl.set("typ", specialty)
        fl.set("not_typ", (core.TYPE_MONSTER | core.TYPE_SPELL | core.TYPE_TRAP) ^ primary)
    elif name == "IsNonEffectMonster":
        need(not args, "filter: IsNonEffectMonster arguments")
        fl.set("typ", core.TYPE_MONSTER)
        fl.set("not_typ", K["TYPE_EFFECT"])
    elif name == "IsMonster":
        fl.set("typ", core.TYPE_MONSTER)
    elif name == "IsSpell":
        fl.set("typ", core.TYPE_SPELL)
    elif name == "IsTrap":
        fl.set("typ", core.TYPE_TRAP)
    elif name == "IsSpellTrap":
        fl.set("typ", core.TYPE_SPELL | core.TYPE_TRAP)
    elif name == "IsType":
        fl.set("typ", num(0))
    elif name == "IsFaceup":
        fl.set("pos", core.P_FACEUP)
    elif name == "IsFacedown":
        fl.set("pos", core.P_FD_ATK | core.P_FD_DEF)
    elif name == "IsAttackPos":
        fl.set("pos", core.P_FU_ATK | core.P_FD_ATK)
    elif name == "IsDefensePos":
        fl.set("pos", core.P_FU_DEF | core.P_FD_DEF)
    elif name == "IsPosition":
        fl.set("pos", num(0))
    elif name == "IsSetCard":
        need(len(args) == 1, "filter: IsSetCard with several arguments")
        fl.set("setc", num(0))
    elif name == "IsRace":
        fl.set("race", num(0))
    elif name == "IsAttribute":
        fl.set("attr", num(0))
    elif name == "IsCode":
        need(1 <= len(args) <= 2 and all(isinstance(x, int) for x in args), "filter: IsCode")
        fl.set("code", (args[0], args[-1]))
    elif name == "IsAttackBelow":
        fl.set("atk", (-core.BIG, num(0)))
    elif name == "IsAttackAbove":
        fl.set("atk", (num(0), core.BIG))
    elif name == "IsDefenseBelow":
        fl.set("df", (-core.BIG, num(0)))
    elif name == "IsDefenseAbove":
        fl.set("df", (num(0), core.BIG))
    elif name == "IsLevelBelow":
        fl.set("lv", (1, num(0)))  # Level > 0 (not Spells / Traps, Xyz, Link)
    elif name == "IsLevelAbove":
        fl.set("lv", (max(num(0), 1), core.BIG))
    elif name == "IsLevel":
        need(len(args) == 1, "filter: IsLevel with several values")
        fl.set("lv", (num(0), num(0)))
    elif name == "IsLocation":
        fl.loc &= num(0)
    elif name == "IsControler":
        need(len(args) == 1 and isinstance(args[0], Player), "filter: IsControler without a bound player")
        fl.ctrl = args[0].p
    elif name == "IsCanBeSpecialSummoned":
        # (e, sumtype, sp, nocheck, nolimit[, pos[, toplayer]])
        need(len(args) >= 2, "filter: missing summon type")
        generic_summon_tag(args[1])
        need(len(args) >= 5 and args[3] is False and args[4] is False, "filter: IsCanBeSpecialSummoned(nocheck/nolimit)")
        need(len(args) < 7, "filter: IsCanBeSpecialSummoned to the opponent")
        need(isinstance(args[2], Player) and args[2].p == ME, "filter: IsCanBeSpecialSummoned by the opponent")
        pos = args[5] if len(args) > 5 else core.P_FACEUP
        need(pos in (core.P_FACEUP, core.P_FU_ATK | core.P_FU_DEF | core.P_FD_DEF, 0xf, core.P_FU_DEF,
                     core.P_FU_DEF | core.P_FD_DEF), "filter: IsCanBeSpecialSummoned(position)")
        fl.set("spsum", 2 if pos in (core.P_FU_DEF, core.P_FU_DEF | core.P_FD_DEF) else 1)
    elif name == "IsNegatableMonster" or name == "IsNegatable":
        fl.set("nodis", 1)
        fl.set("pos", core.P_FACEUP)
    else:
        raise Unsupported(f"filter: c:{name}")


# ---------------------------------------------------------------- script

def dotted(node):
    if isinstance(node, A.Name):
        return node.id
    if isinstance(node, A.Index) and isinstance(node.idx, A.Name) and node.notation == A.IndexNotation.DOT:
        b = dotted(node.value)
        return f"{b}.{node.idx.id}" if b else None
    if isinstance(node, A.Index) and isinstance(node.idx, A.String):
        b = dotted(node.value)
        return f"{b}.{node.idx.s}" if b else None
    return None


class Script:
    def __init__(self, code):
        self.code = code
        path = os.path.join(OFFICIAL, f"c{code}.lua")
        with open(path, encoding="utf-8", errors="replace") as source:
            self.src = source.read()
        try:
            self.tree = LA.parse(self.src)
        except Exception as e:  # noqa: BLE001
            raise Unsupported(f"script not parsed ({type(e).__name__})")
        fold_helpers(self.tree)
        self.fns = {}
        self.vals = {}
        for st in self.tree.body.body:
            if isinstance(st, (A.Function, A.LocalFunction)):
                n = dotted(st.name)
                self.fns[n.split(".")[-1] if n.startswith("s.") else n] = st
            elif isinstance(st, (A.Assign, A.LocalAssign)):
                for t, v in zip(st.targets, st.values):
                    n = dotted(t)
                    if n and n.startswith("s."):
                        if isinstance(v, A.AnonymousFunction):
                            self.fns[n[2:]] = v
                        else:
                            self.vals[n[2:]] = v


# ---------------------------------------------------------------- programs

class Prog:
    """Program under construction (cost, target or resolution); registers are shared by the three programs of an effect."""

    def __init__(self, tr, kind):
        self.tr, self.kind, self.ins = tr, kind, []

    def reg(self):
        r = self.tr.nreg
        need(r < 256, "more than 256 virtual registers")
        self.tr.nreg += 1
        return r

    def emit(self, *ins):
        self.ins.extend(ins)


class Effect:
    def __init__(self):
        self.f = {}
        self.desc = None

    def copy(self):
        e = Effect()
        e.f = dict(self.f)
        e.desc = self.desc
        return e


EVENTS = {
    "EVENT_SUMMON_SUCCESS": core.EV_SUMMON, "EVENT_FLIP_SUMMON_SUCCESS": core.EV_FLIP_SUMMON,
    "EVENT_SPSUMMON_SUCCESS": core.EV_SPSUMMON, "EVENT_ATTACK_ANNOUNCE": core.EV_ATTACK,
    "EVENT_TO_GRAVE": core.EV_TO_GRAVE, "EVENT_BATTLE_DESTROYED": core.EV_BATTLE_DESTROYED, "EVENT_FLIP": core.EV_FLIP,
    "EVENT_PRE_DAMAGE_CALCULATE": core.EV_PRE_DMG,
}
EVCODE = {K[k]: v for k, v in EVENTS.items() if k in K}

MODCODES = {"EFFECT_UPDATE_ATTACK": core.MC_ATK, "EFFECT_UPDATE_DEFENSE": core.MC_DEF,
            "EFFECT_SET_ATTACK": core.MC_SET_ATK, "EFFECT_SET_ATTACK_FINAL": core.MC_FINAL_ATK,
            "EFFECT_SET_BASE_ATTACK": core.MC_BASE_ATK, "EFFECT_SET_DEFENSE": core.MC_SET_DEF,
            "EFFECT_SET_DEFENSE_FINAL": core.MC_FINAL_DEF, "EFFECT_SET_BASE_DEFENSE": core.MC_BASE_DEF,
            "EFFECT_DISABLE": core.MC_DISABLE, "EFFECT_CANNOT_ATTACK": core.MC_NO_ATTACK,
            "EFFECT_INDESTRUCTABLE_BATTLE": core.MC_INDES_BATTLE, "EFFECT_INDESTRUCTABLE_EFFECT": core.MC_INDES_EFF,
            "EFFECT_CANNOT_BE_BATTLE_TARGET": core.MC_NO_BTARGET, "EFFECT_CANNOT_BE_EFFECT_TARGET": core.MC_NO_ETARGET,
            "EFFECT_CANNOT_DIRECT_ATTACK": core.MC_NO_DIRECT}
MODC = {K[k]: v for k, v in MODCODES.items()}
STAT_CODES = {core.MC_ATK, core.MC_DEF, core.MC_SET_ATK, core.MC_FINAL_ATK, core.MC_BASE_ATK, core.MC_SET_DEF,
              core.MC_FINAL_DEF, core.MC_BASE_DEF}

BOUND_FNS = {"Cost.PayLP", "Cost.DetachFromSelf", "aux.FaceupFilter", "aux.FilterBoolFunction", "aux.FilterBoolFunctionEx",
             "aux.FaceupFilterEx", "aux.TargetBoolFunction"}

IGNORED_CALLS = {"Duel.Hint", "Duel.HintSelection", "Duel.ConfirmCards", "Duel.SetOperationInfo",
                 "Duel.SetPossibleOperationInfo", "Duel.ShuffleDeck", "Duel.BreakEffect", "aux.RegisterClientHint",
                 "Duel.AdjustInstantly"}


def _children(node):
    for k, v in vars(node).items():
        if isinstance(v, A.Node):
            yield node, k, None, v
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, A.Node):
                    yield node, k, i, x


def _walk(node, loop=False):
    for parent, k, i, ch in _children(node):
        yield parent, k, i, ch, loop
        yield from _walk(ch, loop or isinstance(ch, (A.While, A.Repeat, A.Fornum, A.Forin, A.Function,
                                                         A.LocalFunction, A.AnonymousFunction, A.Method)))


# Player effects that no translated card sets: the test is false. Blue-Eyes Spirit Dragon (forbids Special Summoning
# more than one monster at a time) sets a player effect, which the translator rejects.
ABSENT_PLAYER_EFFECTS = {"CARD_BLUEEYES_SPIRIT"}


def _pure(n):
    def act(x):
        return isinstance(x, A.Call) and not re.match(r"Duel\.(Is|Get)", dotted(x.func) or "")
    return not act(n) and not any(act(ch) for *_, ch, _ in _walk(n))


def _fold(n):
    for parent, k, i, ch in list(_children(n)):
        new = _fold(ch)
        if new is not ch:
            if i is None:
                setattr(parent, k, new)
            else:
                getattr(parent, k)[i] = new
    T, F = A.TrueExpr, A.FalseExpr
    if isinstance(n, A.Call) and dotted(n.func) == "Duel.IsPlayerAffectedByEffect" and len(n.args) == 2 \
            and dotted(n.args[1]) in ABSENT_PLAYER_EFFECTS:
        return F()
    if isinstance(n, A.ULNotOp) and isinstance(n.operand, (T, F)):
        return F() if isinstance(n.operand, T) else T()
    if isinstance(n, A.AndLoOp):
        if isinstance(n.left, F) or isinstance(n.right, F) and _pure(n.left):
            return F()
        if isinstance(n.left, T):
            return n.right
        if isinstance(n.right, T):
            return n.left
    if isinstance(n, A.OrLoOp):
        if isinstance(n.left, T) or isinstance(n.right, T) and _pure(n.left):
            return T()
        if isinstance(n.left, F):
            return n.right
        if isinstance(n.right, F):
            return n.left
    if isinstance(n, A.Block):
        n.body = [st for st in n.body if not (isinstance(st, A.If) and isinstance(st.test, F) and st.orelse is None)]
    return n


def _flag(x):
    """Constant boolean argument (true / false / nil) or None."""
    if isinstance(x, A.TrueExpr):
        return True
    if isinstance(x, (A.FalseExpr, A.Nil)):
        return False
    return None


def fold_helpers(tree):
    """Helpers from utility whose result depends only on effects that no translated card sets.
    aux.NecroValleyFilter(f) = f without EFFECT_NECRO_VALLEY. aux.SpElimFilter(c, faceup, mzone) without Spirit
    Elimination (69832741): without mzone, c:IsLocation(LOCATION_GRAVE); with mzone, not face-down in the monster
    zone if faceup, otherwise true. aux.DoubleSnareValidity: mark read only by Double Snare, removed."""
    folded = False

    def loc(c, z):
        return A.Invoke(c, A.Name("IsLocation"), [A.Name(z)])

    def rewrite(n):
        nonlocal folded
        if isinstance(n, A.Block):
            n.body = [st for st in n.body if not (isinstance(st, A.Call) and dotted(st.func) in
                                                  ("aux.DoubleSnareValidity", "Auxiliary.DoubleSnareValidity"))]
        for parent, k, i, ch in list(_children(n)):
            new = rewrite(ch)
            if new is not ch:
                if i is None:
                    setattr(parent, k, new)
                else:
                    getattr(parent, k)[i] = new
        name = dotted(n.func) if isinstance(n, A.Call) else None
        if name in ("aux.NecroValleyFilter", "Auxiliary.NecroValleyFilter") and len(n.args) == 1:
            return n.args[0]
        if name in ("aux.SpElimFilter", "Auxiliary.SpElimFilter") and 1 <= len(n.args) <= 3 \
                and isinstance(n.args[0], A.Name):
            faceup, mzone = ([_flag(a) for a in n.args[1:]] + [False, False])[:2]
            if faceup is None or mzone is None:
                return n
            c = n.args[0]
            if not mzone:
                return loc(c, "LOCATION_GRAVE")
            folded = True
            if not faceup:
                return A.TrueExpr()
            return A.OrLoOp(A.Invoke(c, A.Name("IsFaceup"), []), A.ULNotOp(loc(c, "LOCATION_MZONE")))
        return n

    rewrite(tree)
    if folded:
        _fold(tree)


def fold_absent(fn):
    """Replace tests for absent player effects (ABSENT_PLAYER_EFFECTS) with false and simplify."""
    if any(isinstance(ch, A.Call) and dotted(ch.func) == "Duel.IsPlayerAffectedByEffect" for *_, ch, _ in _walk(fn.body)):
        _fold(fn.body)


def negate(t):
    """Negation of a Lua test, by inverting the comparisons."""
    inv = {A.EqToOp: A.NotEqToOp, A.NotEqToOp: A.EqToOp, A.LessThanOp: A.GreaterOrEqThanOp,
           A.GreaterOrEqThanOp: A.LessThanOp, A.GreaterThanOp: A.LessOrEqThanOp, A.LessOrEqThanOp: A.GreaterThanOp}
    if type(t) in inv:
        return inv[type(t)](t.left, t.right)
    if isinstance(t, A.ULNotOp):
        return t.operand
    return A.ULNotOp(t)


class _LabelFold:
    """e:SetLabel / e:GetLabel with constant values, between the cost, target and operation of one activation.

    ygopro calls cost(chk=0) then target(chk=0) to test the activation, then cost, target and operation.
    The label is simulated along these two paths; each read must follow a constant write from the same
    activation (no dependence on previous activations). Reads become numbers, decided tests are replaced by
    their branch and the writes disappear."""

    def __init__(self):
        self.reads, self.tests, self.writes = {}, {}, set()

    @staticmethod
    def param(fn):
        return fn.args[0].id if fn.args and isinstance(fn.args[0], A.Name) else None

    @staticmethod
    def is_label(n, e, m):
        return isinstance(n, A.Invoke) and isinstance(n.source, A.Name) and n.source.id == e and n.func.id == m

    def number(self, node):
        if isinstance(node, A.Number):
            return node.n
        if isinstance(node, A.UMinusOp) and isinstance(node.operand, A.Number):
            return -node.operand.n
        raise Unsupported("SetLabel: non-constant value")

    def read(self, node, e, v):
        for *_, ch, _ in [(None, None, None, node, False)] + list(_walk(node)):
            if self.is_label(ch, e, "SetLabel"):
                raise Unsupported("SetLabel in an expression")
            if self.is_label(ch, e, "GetLabel"):
                need(not ch.args and v is not None, "GetLabel without a prior constant SetLabel")
                need(self.reads.setdefault(id(ch), v) == v, "GetLabel: different values depending on the path")

    def value(self, t, e, v):
        """Boolean value of the test after substitution, or None."""
        def ev(n):
            if self.is_label(n, e, "GetLabel"):
                return v
            if isinstance(n, A.Number):
                return n.n
            if isinstance(n, A.UMinusOp):
                x = ev(n.operand)
                return None if x is None else -x
            if isinstance(n, A.TrueExpr):
                return True
            if isinstance(n, A.FalseExpr):
                return False
            ops = {A.EqToOp: lambda a, b: a == b, A.NotEqToOp: lambda a, b: a != b,
                   A.LessThanOp: lambda a, b: a < b, A.GreaterThanOp: lambda a, b: a > b,
                   A.LessOrEqThanOp: lambda a, b: a <= b, A.GreaterOrEqThanOp: lambda a, b: a >= b}
            if type(n) in ops:
                a, b = ev(n.left), ev(n.right)
                return None if a is None or b is None or isinstance(a, bool) or isinstance(b, bool) else ops[type(n)](a, b)
            if isinstance(n, A.ULNotOp):
                x = ev(n.operand)
                return None if x is None else not x
            if isinstance(n, A.AndLoOp):
                a, b = ev(n.left), ev(n.right)
                return False if a is False or (b is False and _pure(n.left)) else (b if a is True else None)
            if isinstance(n, A.OrLoOp):
                a, b = ev(n.left), ev(n.right)
                return True if a is True or (b is True and _pure(n.left)) else (b if a is False else None)
            return None
        return ev(t)

    def run(self, stmts, e, v, check):
        """Simulate a sequence of statements; returns (label, finished)."""
        for st in stmts:
            if isinstance(st, A.If) and isinstance(st.test, A.EqToOp) and isinstance(st.test.left, A.Name) \
                    and st.test.left.id == "chk":
                if check:
                    v, _ = self.run(st.body.body, e, v, check)
                    return v, True
                if st.orelse is not None:
                    need(isinstance(st.orelse, A.Block), "chk==0 with elseif")
                    v, done = self.run(st.orelse.body, e, v, check)
                    if done:
                        return v, True
                continue
            if self.is_label(st, e, "SetLabel"):
                need(len(st.args) == 1, "SetLabel: several values")
                v = self.number(st.args[0])
                self.writes.add(id(st))
                continue
            if isinstance(st, A.If):
                self.read(st.test, e, v)
                b = self.value(st.test, e, v)
                if b is not None and any(self.is_label(ch, e, "GetLabel") for *_, ch, _ in _walk(st.test)) | \
                        self.is_label(st.test, e, "GetLabel"):
                    need(self.tests.setdefault(id(st), b) == b, "label test: different values depending on the path")
                    branch = st.body.body if b else (st.orelse.body if isinstance(st.orelse, A.Block) else
                                                     [st.orelse] if st.orelse is not None else [])
                    v, done = self.run(branch, e, v, check)
                    if done:
                        return v, True
                    continue
                for part in [st.body] + ([st.orelse] if st.orelse is not None else []):
                    need(not any(self.is_label(ch, e, "SetLabel") for *_, ch, _ in _walk(part)), "conditional SetLabel")
                    self.read(part, e, v)
                continue
            if isinstance(st, A.Return):
                self.read(st, e, v)
                if check:
                    return v, True
                continue
            need(not any(self.is_label(ch, e, "SetLabel") for *_, ch, _ in _walk(st)), "nested SetLabel")
            self.read(st, e, v)
        return v, False

    def rewrite(self, block, e):
        out = []
        for st in block.body:
            if id(st) in self.writes:
                continue
            if id(st) in self.tests:
                b = self.tests[id(st)]
                branch = st.body if b else st.orelse
                if branch is None:
                    continue
                if isinstance(branch, A.Block):
                    self.rewrite(branch, e)
                    out.extend(branch.body)
                    continue
                st = branch  # elseif
                if id(st) in self.tests or id(st) in self.writes:
                    tmp = A.Block([st])
                    self.rewrite(tmp, e)
                    out.extend(tmp.body)
                    continue
            self.subst(st, e)
            out.append(st)
        block.body = out

    def subst(self, node, e):
        for parent, k, i, ch in list(_children(node)):
            if self.is_label(ch, e, "GetLabel") and id(ch) in self.reads:
                new = A.Number(self.reads[id(ch)]) if self.reads[id(ch)] >= 0 else A.UMinusOp(A.Number(-self.reads[id(ch)]))
                if i is None:
                    setattr(parent, k, new)
                else:
                    getattr(parent, k)[i] = new
            elif isinstance(ch, A.Block):
                self.rewrite(ch, e)
            else:
                self.subst(ch, e)


def fold_labels(f):
    """Copy of the cost / target / operation callbacks with the constant label folded (see _LabelFold)."""
    import copy
    keys = [k for k in ("SetCost", "SetTarget", "SetOperation") if k in f and isinstance(f[k][0], Fn) and f[k][0].node is not None]
    def uses(node):
        e = _LabelFold.param(node)
        return e is not None and any(_LabelFold.is_label(ch, e, m) for *_, ch, _ in _walk(node.body)
                                     for m in ("SetLabel", "GetLabel"))
    if not any(uses(f[k][0].node) for k in keys):
        return f
    f = dict(f)
    nodes = {}
    for k in keys:
        nodes[k] = copy.deepcopy(f[k][0].node)
        f[k] = [Fn(node=nodes[k], name=f[k][0].name, bound=f[k][0].bound), *f[k][1:]]
    lf = _LabelFold()
    v = None
    for check in (True, False):
        for k in ("SetCost", "SetTarget") + (() if check else ("SetOperation",)):
            if k in nodes:
                e = _LabelFold.param(nodes[k])
                v, _ = lf.run(nodes[k].body.body, e, v, check)
    for k, node in nodes.items():
        lf.rewrite(node.body, _LabelFold.param(node))
        # cost without a chk test: "e:SetLabel(1) return true" pays nothing
        if k == "SetCost" and node.body.body and isinstance(node.body.body[-1], A.Return) and \
                len(node.body.body[-1].values) == 1 and isinstance(node.body.body[-1].values[0], A.TrueExpr):
            node.body.body = node.body.body[:-1]
        e = _LabelFold.param(node)
        need(not any(_LabelFold.is_label(ch, e, m) for *_, ch, _ in _walk(node.body) for m in ("SetLabel", "GetLabel")),
             "label not folded")
    return f


BLOCK_END = (core.OP_BLOCK, -1, 0, 0)  # translator marker, removed by allocate


def _walk_nofn(node):
    """Nodes under node, without entering nested functions."""
    for _, _, _, ch in _children(node):
        yield ch
        if not isinstance(ch, (A.Function, A.LocalFunction, A.AnonymousFunction, A.Method)):
            yield from _walk_nofn(ch)


def _dead_getfirst_loop_prelude(st, rest):
    """Detect a dead GetFirst, immediately replaced by aux.Next of the same group."""
    if not isinstance(st, A.LocalAssign) or len(st.targets) != 1 or len(st.values) != 1:
        return False
    target, value = st.targets[0], st.values[0]
    if not isinstance(target, A.Name) or not isinstance(value, A.Invoke):
        return False
    name = target.id
    if not isinstance(value.func, A.Name) or value.func.id != "GetFirst" or value.args:
        return False
    if not isinstance(value.source, A.Name) or value.source.id == name or not rest:
        return False
    group = value.source.id
    loop = rest[0]
    if not isinstance(loop, A.Forin) or len(loop.targets) != 1 or len(loop.iter) != 1:
        return False
    if not isinstance(loop.targets[0], A.Name) or loop.targets[0].id != name:
        return False
    iterator = loop.iter[0]
    if not isinstance(iterator, A.Call) or dotted(iterator.func) != "aux.Next" or len(iterator.args) != 1:
        return False
    if not isinstance(iterator.args[0], A.Name) or iterator.args[0].id != group:
        return False
    # Includes nested functions: an outer-local capture after the loop is a live read.
    return not any(isinstance(ch, A.Name) and ch.id == name
                   for statement in rest[1:] for _, _, _, ch, _ in _walk(statement))


def _path(root, target):
    """Ancestors of target under root (root included), from the highest to the closest."""
    for _, _, _, ch in _children(root):
        if ch is target:
            return [root]
        sub = _path(ch, target)
        if sub is not None:
            return [root] + sub
    return None


def step_to_summon(fn):
    """Duel.SpecialSummonStep followed by Duel.SpecialSummonComplete, a single summon outside a loop: rewritten as
    Duel.SpecialSummon (same signature; in a test, SpecialSummon(...)>0) and the Complete is removed. Effects
    registered between the two then apply after the move, as in ygopro; only the summon events arrive earlier
    (to be validated against the oracle)."""
    steps, comps = [], []
    for parent, k, i, ch, loop in _walk(fn.body):
        if isinstance(ch, A.Call) and dotted(ch.func) == "Duel.SpecialSummonStep":
            steps.append((parent, k, i, ch, loop))
        elif isinstance(ch, A.Call) and dotted(ch.func) == "Duel.SpecialSummonComplete":
            comps.append((parent, k, i, ch, loop))
    if not steps:
        return
    need(len(steps) == 1, "multiple SpecialSummonStep")
    parent, k, i, call, _ = steps[0]
    # In a for loop over a group (aux.Next / :Iter), the body is translated once for the whole group:
    # the Step becomes a simultaneous summon of the group, which is what Complete does.
    path = _path(fn.body, call)
    loops = [n for n in path if isinstance(n, (A.While, A.Repeat, A.Fornum, A.Forin, A.Function, A.LocalFunction,
                                               A.AnonymousFunction, A.Method))]
    need(not loops or len(loops) == 1 and isinstance(loops[0], A.Forin) and isinstance(parent, A.Block),
         "SpecialSummonStep in a loop")
    need(len(comps) == 1 and not comps[0][4] and isinstance(comps[0][0], A.Block), "SpecialSummonComplete: shape")
    blk, j = comps[0][0], comps[0][2]
    def has(n):
        return n is call or any(ch is call for *_, ch, _ in _walk(n))
    # Complete after the Step in the same block, or in the body of an if whose test performs the Step
    owner = next((p for p, kk, _, ch, _ in _walk(fn.body) if ch is blk and kk == "body"), None)
    need(any(has(st) for st in blk.body[:j]) or isinstance(owner, (A.If, A.ElseIf)) and has(owner.test),
         "SpecialSummonComplete outside the SpecialSummonStep block")
    call.func.idx.id = "SpecialSummon"
    if not isinstance(parent, A.Block):  # in an expression: true if the summon happened
        new = A.GreaterThanOp(call, A.Number(0))
        if i is None:
            setattr(parent, k, new)
        else:
            getattr(parent, k)[i] = new
    del blk.body[j]


class Translator:
    def __init__(self, code):
        self.code = code
        self.sc = Script(code)
        self.notes = set()  # approximations made
        self.proc = None  # summon recipe by procedure: ("xyz", F, ct, Rank) / ("synchro", F, F) / ("link", F, mn, mx)
        # Duel.AddCustomActivityCounter(id, ACTIVITY_SPSUMMON, f): id -> (bit ST_SST / ST_SSXT, filter or None)
        self.counters = {}

    # ------------------------------------------------ initial_effect
    def effects(self):
        fn = self.sc.fns.get("initial_effect")
        need(fn is not None, "no initial_effect")
        env = {fn.args[0].id: Handler()} if fn.args else {}
        regs = []
        for st in fn.body.body:
            self.ie_stmt(st, env, regs)
        return regs

    def ie_val(self, node, env):
        if isinstance(node, A.Name):
            if node.id in env:
                return env[node.id]
            if node.id in K:
                return K[node.id]
            if node.id == "id":
                return self.code
            raise Unsupported(f"initial_effect: name {node.id}")
        if isinstance(node, A.Number):
            return node.n
        if isinstance(node, (A.TrueExpr, A.FalseExpr)):
            return isinstance(node, A.TrueExpr)
        if isinstance(node, A.Nil):
            return None
        if isinstance(node, (A.BOrOp, A.AddOp, A.BAndOp, A.SubOp, A.MultOp)):
            a, b = self.ie_val(node.left, env), self.ie_val(node.right, env)
            need(isinstance(a, int) and isinstance(b, int), "initial_effect: non-constant operation")
            return {A.BOrOp: a | b, A.AddOp: a + b, A.BAndOp: a & b, A.SubOp: a - b, A.MultOp: a * b}[type(node)]
        d = dotted(node)
        if d and d.startswith("s."):
            n = d[2:]
            if n in self.sc.fns:
                return Fn(self.sc.fns[n], n)
            if n in self.sc.vals:
                return self.ie_val(self.sc.vals[n], env)
        if d:
            return Fn(name=d)
        if isinstance(node, A.Call) and dotted(node.func) == "aux.Stringid":
            a = [self.ie_val(x, env) for x in node.args]
            return Desc((a[0] << 20) | (a[1] & 0xfffff)) if len(a) == 2 and all(isinstance(x, int) for x in a) else None
        if isinstance(node, A.UMinusOp):
            v = self.ie_val(node.operand, env)
            need(isinstance(v, int), "initial_effect: -x")
            return -v
        if isinstance(node, A.Call) and dotted(node.func) in BOUND_FNS:
            return Fn(name=dotted(node.func), bound=[self.ie_val(a, env) for a in node.args])
        if isinstance(node, A.AnonymousFunction):
            return Fn(node, "anon")
        if isinstance(node, A.Table):  # {id, k} (SetCountLimit)
            need(all(f.key is None or isinstance(f.key, A.Number) for f in node.fields), "initial_effect: table")
            return tuple(self.ie_val(f.value, env) for f in node.fields)
        if isinstance(node, A.Call) and dotted(node.func) in ("Effect.CreateEffect",):
            return Effect()
        if isinstance(node, A.Invoke) and node.func.id == "Clone":
            e = self.ie_val(node.source, env)
            need(isinstance(e, Effect), "Clone")
            return e.copy()
        if isinstance(node, (A.Call, A.Invoke)):
            raise Unsupported(f"initial_effect: call {dotted(node.func) if isinstance(node, A.Call) else ':' + node.func.id}")
        raise Unsupported(f"initial_effect: expression {type(node).__name__}")

    def ie_stmt(self, st, env, regs):
        if isinstance(st, (A.LocalAssign, A.Assign)):
            for t, v in zip(st.targets, st.values):
                need(isinstance(t, A.Name), "initial_effect: assignment")
                if isinstance(v, A.Call) and dotted(v.func) == "aux.AddEquipProcedure":
                    # the returned effect is not modeled: any later use is rejected (opaque value)
                    regs.append(self.equip_procedure(v.args, env))
                    env[t.id] = object()
                    continue
                env[t.id] = self.ie_val(v, env)
            return
        if isinstance(st, A.Invoke):
            obj = self.ie_val(st.source, env)
            m = st.func.id
            args = [self.ie_val(a, env) for a in st.args]
            if isinstance(obj, Effect):
                if m in ("SetHintTiming", "SetCategory"):
                    return
                if m == "SetDescription":
                    obj.desc = args[0] if args and isinstance(args[0], int) else None
                    return
                obj.f[m] = args
                return
            if isinstance(obj, Handler):
                if m == "RegisterEffect":
                    need(isinstance(args[0], Effect), "RegisterEffect")
                    regs.append(args[0].copy())
                    return
                if m in ("EnableReviveLimit",):
                    regs.append(("revive",))
                    return
                if m == "SetUniqueOnField":
                    raise Unsupported("SetUniqueOnField")
            raise Unsupported(f"initial_effect: :{m}")
        if isinstance(st, A.Call):
            d = dotted(st.func)
            if d == "Fusion.RegisterSummonEff":
                need(len(st.args) == 1 and isinstance(self.ie_val(st.args[0], env), Handler),
                     "Fusion.RegisterSummonEff: options")
                regs.append(("fusion_spell",))
                return
            if d in ("Ritual.AddProcGreaterCode", "Ritual.AddProcEqualCode", "Ritual.AddProcGreater", "Ritual.AddProcEqual"):
                need(st.args and isinstance(self.ie_val(st.args[0], env), Handler), "ritual: source card")
                regs.append(self.ritual_procedure(d, st.args[1:], env))
                return
            if d == "Pendulum.AddProcedure":
                regs.append(("pendulum",))
                return
            if d == "Duel.AddCustomActivityCounter":
                need(len(st.args) == 3 and dotted(st.args[1]) == "ACTIVITY_SPSUMMON", "activity counter: type")
                key = self.ie_val(st.args[0], env)
                fv = self.ie_val(st.args[2], env)
                need(isinstance(key, int) and isinstance(fv, Fn) and fv.node is not None, "activity counter: filter")
                self.counters[key] = self.counter_filter(fv.node)
                return
            if d == "aux.addContinuousLizardCheck":
                self.notes.add("Clock Lizard ignored")
                return
            if d == "aux.AddEquipProcedure":
                regs.append(self.equip_procedure(st.args, env))
                return
            if d in ("Xyz.AddProcedure", "Synchro.AddProcedure", "Link.AddProcedure", "Fusion.AddProcMix", "Fusion.AddProcMixN"):
                need(self.proc is None, "two procedures")
                self.proc = self.procedure(d, st.args[1:], env)
                return
            raise Unsupported(f"initial_effect: {d}")
        raise Unsupported(f"initial_effect: statement {type(st).__name__}")

    STATIC_FIELDS = {"typ", "attr", "race", "setc", "code", "not_typ"}

    def static_filter(self, terms, cvar, why):
        """Filter on fixed properties (type, attribute, race, archetype, name): equivalent when evaluated
        afterwards on an already summoned card."""
        fl = Filt()
        for x in terms:
            self.filter_term(fl, x, {cvar: "CARD"}, cvar=cvar)
        dflt = Filt().d
        need(fl.loc == ALLZ and fl.ctrl is None and not fl.tgt and
             all(fl.d[k] == dflt[k] for k in fl.d if k not in self.STATIC_FIELDS), why)
        return fl.build(ALLZ | core.Z_OVERLAY, ALLZ | core.Z_OVERLAY)

    def counter_filter(self, node):
        """Filter of an ACTIVITY_SPSUMMON activity counter (cards that do not count) -> (virtual status bit of the
        summons that count, filter of the exceptions or None): counted = summoned this turn [from the Extra Deck]
        and not exception."""
        need(len(node.args) >= 1 and len(node.body.body) == 1 and isinstance(node.body.body[0], A.Return),
             "activity counter: body")
        cvar = node.args[0].id
        x = node.body.body[0].values[0]

        def from_extra(t):
            if isinstance(t, A.ULNotOp) and isinstance(t.operand, A.Invoke) and t.operand.func.id == "IsSummonLocation":
                return dotted(t.operand.args[0]) == "LOCATION_EXTRA"
            if isinstance(t, A.NotEqToOp) and isinstance(t.left, A.Invoke) and t.left.func.id == "GetSummonLocation":
                return dotted(t.right) == "LOCATION_EXTRA"
            return False
        # not (c:IsSummonLocation(LOCATION_EXTRA) and not B) = not c:IsSummonLocation(LOCATION_EXTRA) or B
        if isinstance(x, A.ULNotOp) and isinstance(x.operand, A.AndLoOp):
            parts = self.conj(x.operand)
            need(len(parts) == 2, "activity counter: shape")
            loc = [t for t in parts if isinstance(t, A.Invoke) and t.func.id == "IsSummonLocation"]
            neg = [t for t in parts if isinstance(t, A.ULNotOp)]
            need(len(loc) == 1 and len(neg) == 1 and dotted(loc[0].args[0]) == "LOCATION_EXTRA", "activity counter: shape")
            return core.ST_SSXT, self.static_filter(self.conj(neg[0].operand), cvar, "activity counter: filter")
        terms = self.disj(x)
        fx = [t for t in terms if from_extra(t)]
        rest = [t for t in terms if not from_extra(t)]
        need(len(rest) <= 1 and (rest or fx), "activity counter: disjunction")
        allowed = self.static_filter(self.conj(rest[0]), cvar, "activity counter: filter") if rest else None
        return (core.ST_SSXT if fx else core.ST_SST), allowed

    def equip_procedure(self, args, env):
        """aux.AddEquipProcedure(c, p, f): activation that targets a face-up monster (of p: 0 me, 1 opponent, nil or
        PLAYER_ALL both) satisfying f, then Duel.Equip; equip limit f (EquipLimit), checked by the ygopro
        adjustment: only filters on fixed properties are accepted (the limit stays true)."""
        need(args and isinstance(self.ie_val(args[0], env), Handler), "equip: source card")
        need(all(isinstance(a, A.Nil) for a in args[3:]), "equip: limit, cost, target, operation or condition")
        pv = self.ie_val(args[1], env) if len(args) > 1 else None
        need(pv in (None, 0, 1, K["PLAYER_ALL"]), "equip: player")
        fv = self.ie_val(args[2], env) if len(args) > 2 else None
        fl = Filt()
        if fv is not None:
            self.apply_fn(fl, fv, [Eff(), Player(ME)], env)
        dflt = Filt().d
        static = {"typ", "attr", "race", "setc", "code", "lv", "not_typ", "pos"}
        need(all(v == dflt[k] for k, v in fl.d.items() if k not in static) and fl.ctrl is None
             and fl.loc & core.Z_MZONE, "equip: filter on a variable property")
        need(fl.d["pos"] in (0, core.P_FACEUP), "equip: position")
        fl.set("pos", core.P_FACEUP)
        return ("equip_spell", fl.build(core.Z_MZONE if pv != 1 else 0, core.Z_MZONE if pv != 0 else 0))

    def ritual_procedure(self, name, args, env):
        values = [self.ie_val(a, env) for a in args]
        mode = core.RIT_GREATER if "Greater" in name else core.RIT_EQUAL
        fl = Filt()
        if name.endswith("Code"):
            need(len(values) >= 3, "ritual: missing names")
            level, desc, *codes = values
            need(1 <= len(codes) <= 2 and all(type(c) is int and c > 0 for c in codes), "ritual: names")
            apply_method(fl, "IsCode", codes)
        else:
            values += [None] * max(0,3-len(values))
            filter_, level, desc, *options = values
            need(all(v is None for v in options), "ritual: custom materials, operation or location")
            self.apply_fn(fl, filter_, [], env)
        need(level is None or type(level) is int, "ritual: computed level")
        if level is not None:
            # Constant folding valid for the pinned data and the engine's static Levels.
            # Effects that modify Levels remain rejected by the translator.
            codes = fl.d["code"]
            need(codes and all(c > 0 and card_metadata().get(c,(0,-1))[1] == level for c in codes),
                 "ritual: fixed level different from the monster level")
        need(desc is None or type(desc) is int, "ritual: description")
        fl.set("typ", core.TYPE_RITUAL)
        f = list(fl.build(core.Z_HAND, 0));f[core.F_RIT] = mode
        return ("ritual_spell", tuple(f), mode, 1171 if desc is None else desc)

    # ------------------------------------------------ summon procedures (Extra Deck)
    def mat_fn(self, node, env):
        if isinstance(node, A.Call) and dotted(node.func) == "Synchro.NonTuner":
            need(len(node.args) == 1, "Synchro.NonTuner with arguments")
            return self.mat_fn(node.args[0], env)
        if isinstance(node, A.Call) and dotted(node.func) == "Synchro.NonTunerEx":
            return Fn(name="aux.FilterBoolFunctionEx", bound=[self.ie_val(a, env) for a in node.args])
        return self.ie_val(node, env)

    def mat_filter(self, node, env):
        """Material filter -> F(...) over the hand and the field, core.MAT_FIELDS fields only."""
        fl = Filt()
        self.apply_fn(fl, self.mat_fn(node, env), [], env)
        need(fl.loc == ALLZ and fl.ctrl is None and not fl.tgt, "material: location or controller")
        f = fl.build(core.Z_HAND | core.Z_MZONE, 0)
        dflt = C.MAT()
        need(all(f[k] == dflt[k] for k in range(len(f)) if k not in core.MAT_FIELDS), "material: filter outside MAT_FIELDS")
        return f

    def procedure(self, d, args, env):
        def num(i, dflt=None):
            if i >= len(args) or isinstance(args[i], A.Nil):
                return dflt
            v = self.ie_val(args[i], env)
            need(isinstance(v, int), f"{d}: argument {i + 1} non-constant")
            return v

        def nil(i):
            return i >= len(args) or isinstance(args[i], A.Nil)
        if d in ("Fusion.AddProcMix", "Fusion.AddProcMixN"):
            need(len(args) >= 4, "Fusion: missing arguments")
            sub, insf = self.ie_val(args[0], env), self.ie_val(args[1], env)
            need(type(sub) is bool and type(insf) is bool, "Fusion: non-boolean options")
            # insf only concerns summons without materials (Instant Fusion), whose effects are not
            # translated. The common recipes use materials.
            specs = args[2:]
            if d.endswith("MixN"):
                need(len(specs) % 2 == 0, "Fusion: material / number pairs")
                expanded = []
                for material, count in zip(specs[::2], specs[1::2]):
                    n = self.ie_val(count, env)
                    need(type(n) is int and 1 <= n <= core.FK, "Fusion: number of materials")
                    expanded.extend([material] * n)
                specs = expanded
            need(2 <= len(specs) <= core.FK, f"Fusion: between 2 and {core.FK} materials")
            filters = []
            substitute_mask = 0
            for j, material in enumerate(specs):
                value = self.ie_val(material, env)
                if type(value) is int:
                    need(value > 0, "Fusion: material code")
                    if sub:
                        substitute_mask |= 1 << j
                    filters.append(C.MAT(code=value))
                else:
                    need(isinstance(value, Fn), "Fusion: material neither code nor filter")
                    filters.append(self.mat_filter(material, env))
            return ("fusion", filters, substitute_mask)
        f0 = args[0] if args else A.Nil()
        if d == "Xyz.AddProcedure":  # (f, lv, ct, alterf, desc, maxct, op, mustbemat, exchk)
            need(all(nil(i) for i in (3, 6, 7, 8)), "Xyz: alternative material / operation")
            lv, ct = num(1), num(2)
            need(num(5, ct) == ct, "Xyz: variable number of materials")
            return ("xyz", self.mat_filter(f0, env), ct, lv)
        if d == "Synchro.AddProcedure":  # (f1, min1, max1, f2, min2, max2, ...)
            need(len(args) <= 6, "Synchro: options")
            need((num(1), num(2), num(4), num(5)) == (1, 1, 1, 99), "Synchro: numbers of materials")
            return ("synchro", self.mat_filter(f0, env), self.mat_filter(args[3], env))
        # Link.AddProcedure (f, min, max, specialchk, ...)
        need(all(nil(i) for i in range(3, len(args))), "Link: condition on the material group")
        mn = num(1)
        return ("link", self.mat_filter(f0, env), mn, num(2, 99))

    # ------------------------------------------------ effect -> bricks
    def translate(self):
        regs = self.effects()
        plain_activation = any(isinstance(e, Effect) and e.f.get("SetType") == [K["EFFECT_TYPE_ACTIVATE"]]
                               and e.f.get("SetCode") == [K["EVENT_FREE_CHAIN"]]
                               and not any(k in e.f for k in ("SetCost","SetTarget","SetOperation")) for e in regs)
        # field::get_cteffect: only TRIGGER / QUICK_O effects whose range includes the Spell/Trap zone are offered
        def offered(e):
            if not isinstance(e, Effect):
                return False
            typ, rng = e.f.get("SetType", [0])[0], e.f.get("SetRange", [0])[0]
            return isinstance(typ, int) and isinstance(rng, int) and bool(typ & (K["EFFECT_TYPE_QUICK_O"] | K["EFFECT_TYPE_TRIGGER_O"] | K["EFFECT_TYPE_TRIGGER_F"])) \
                and bool(rng & core.Z_SZONE)
        offers = [e for e in regs if offered(e)]
        inline = plain_activation and bool(offers) and card_type(self.code) == (core.TYPE_TRAP | core.TYPE_CONTINUOUS)
        alt_reg = None
        if inline:
            need(len(offers) == 1, "combined activation: several effects")
            activation = [e for e in regs if isinstance(e, Effect) and e.f.get("SetType") == [K["EFFECT_TYPE_ACTIVATE"]]]
            alt_reg = offers[0]
            need(alt_reg.f.get("SetType") == [K["EFFECT_TYPE_QUICK_O"]] and alt_reg.f.get("SetCode") == [K["EVENT_FREE_CHAIN"]]
                 and alt_reg.f.get("SetRange") == [core.Z_SZONE], "combined activation: trigger or event")
            need(len(activation) == 1 and "SetCountLimit" not in activation[0].f,
                 "combined activation: event or counter")
        hand_procs = [e for e in regs if isinstance(e, Effect) and e.f.get("SetCode") == [K["EFFECT_SPSUMMON_PROC"]]]
        need(len(hand_procs) <= 1, "several hand procedures")
        self.revive = int(("revive",) in regs)
        # aux.FALSE forbids summons by effect, even after a correct summon.
        # This static restriction goes into erl, without taking up an activation slot.
        if hand_procs:
            remaining = []
            for entry in regs:
                if isinstance(entry, Effect) and entry.f.get("SetCode") == [K["EFFECT_SPSUMMON_CONDITION"]]:
                    f = entry.f
                    value = f.get("SetValue", [None])[0]
                    need(f.get("SetType") == [K["EFFECT_TYPE_SINGLE"]] and isinstance(value, Fn) and value.name == "aux.FALSE"
                         and f.get("SetProperty", [0])[0] & ~(K["EFFECT_FLAG_UNCOPYABLE"] | K["EFFECT_FLAG_CANNOT_DISABLE"]) == 0
                         and set(f) <= {"SetCode", "SetType", "SetProperty", "SetValue"},
                         "conditional Special Summon restriction")
                    self.revive = 2
                else:
                    remaining.append(entry)
            regs = remaining
        out = []
        self.descs = []  # description (aux.Stringid) of each translated effect, for the oracle driver
        for e in regs:
            if isinstance(e, tuple):
                if e[0] == "fusion_spell":
                    out.append(C.act(target=[C.FILTER(0, C.F(me=core.Z_EXTRA, fus=1)), C.REQ(0, 1)],
                                     resolve=[C.FILTER(0, C.F(me=core.Z_EXTRA, fus=1)), C.CHOOSE(1, 0, 1),
                                              C.FUSMAT(2, 1), C.MOVE(2, reason=core.REASON_EFFECT | core.REASON_MATERIAL | core.REASON_FUSION), C.SPSUMMON(1, core.P_FU_ATK | core.P_FU_DEF, fusion=True)]))
                    self.descs.append(None)
                    continue
                if e[0] == "ritual_spell":
                    _, f, mode, desc = e
                    out.append(C.act(target=[C.FILTER(0,f),C.REQ(0,1)],
                                     resolve=[C.FILTER(0,f),C.CHOOSE(1,0,1),C.RITMAT(2,1,mode),
                                              C.MOVE(2, reason=core.REASON_EFFECT | core.REASON_MATERIAL | core.REASON_RITUAL | core.REASON_RELEASE),C.SPSUMMON(1,core.P_FU_ATK|core.P_FU_DEF,True)]))
                    self.descs.append(desc)
                    continue
                if e[0] == "pendulum":
                    out.append(C.act())
                    self.descs.append(None)
                    continue
                if e[0] == "equip_spell":
                    out.append(C.act(target=[C.FILTER(0, e[1], tgt=True), C.CHOOSE(1, 0, 1, tgt=True)],
                                     resolve=[C.EQUIP(1)]))
                    self.descs.append(1068)
                    continue
                # limit carried by the card
                if self.proc is not None or hand_procs or card_type(self.code) & core.TYPE_RITUAL:
                    continue
                raise Unsupported("monster with a restricted Summon (EnableReviveLimit)")
            compiled = self.allocate(self.effect(e))
            count = e.f.get("SetCountLimit", ())
            if len(count) == 3 and count[2] == K["EFFECT_COUNT_CODE_OATH"]:
                compiled = (*compiled[:7], dict(compiled[7], refund=1))
            # SINGLE clones equivalent for the three summon types, mutually exclusive
            # for this card at the same event point. Per-effect counters stay distinct.
            # No FIELD or arbitrary grouping.
            summon_events = (1 << core.EV_SUMMON) | (1 << core.EV_FLIP_SUMMON) | (1 << core.EV_SPSUMMON)
            merged = False
            if compiled[0] in (core.K_TRIG_F, core.K_TRIG_O) and compiled[2] not in (core.OPT_CARD, core.OPT_CARD_NR) and compiled[1] & summon_events and not compiled[1] & ~(summon_events | core.TRIG_WHEN):
                for index, previous in enumerate(out):
                    if previous[0] == compiled[0] and previous[2:] == compiled[2:] and self.descs[index] == e.desc and \
                            (previous[1] & ~summon_events) == (compiled[1] & ~summon_events) and \
                            not previous[1] & compiled[1] & summon_events:
                        out[index] = (previous[0], previous[1] | compiled[1], *previous[2:])
                        merged = True
                        break
            if not merged:
                if e is alt_reg:
                    alt_index = len(out)
                out.append(compiled)
                self.descs.append(e.desc)
        need(len(out) <= core.E, f"more than {core.E} effects")
        if inline:
            initial = next(i for i, e in enumerate(out) if e[0] == core.K_ACTIVATE)
            alternate = alt_index
            need(out[alternate][1] == 0 and out[alternate][7].get("dmg", 0) == 0,
                 "combined activation: damage step")
            effect = out[initial]
            out[initial] = (*effect[:7], dict(effect[7], inline=alternate + 1))
        return C.Effects(out, revive=self.revive) if self.revive else out

    def flag(self, e, k):
        return (e.f.get("SetProperty", [0])[0] or 0) & K[k]

    def count_limit(self, f):
        opt = 0
        if "SetCountLimit" in f:
            cl = f["SetCountLimit"]
            need(cl[0] == 1, "SetCountLimit > 1")
            # The pinned core returns the OATH counter when the activation is negated.
            # The refund metadata is added to the effect after compilation.
            if len(cl) == 3 and cl[2] == K["EFFECT_COUNT_CODE_OATH"]:
                cl = cl[:2]
            need(len(cl) in (1, 2), "SetCountLimit: unsupported flags")
            if len(cl) == 1:
                opt = core.OPT_CARD
            elif isinstance(cl[1], tuple):
                need(len(cl) == 2 and len(cl[1]) == 2 and cl[1][0] == self.code and type(cl[1][1]) is int and 0 <= cl[1][1] <= 255,
                     "SetCountLimit(1, {other code / flags})")
                opt = core.OPT_NAME if cl[1][1] == 0 else core.OPT_NAMEK + cl[1][1] - 1
            else:
                need(cl[1] == self.code or (isinstance(cl[1], int) and cl[1] & 0xFFFFFFF == self.code and cl[1] == self.code),
                     "SetCountLimit(1, other code / flags)")
                opt = core.OPT_NAME
        return opt

    def effect(self, e):
        f = e.f
        typ = f.get("SetType", [0])[0]
        code = f.get("SetCode", [0])[0] if "SetCode" in f else 0
        T = lambda n: typ & K[n]  # noqa: E731
        for k in ("SetLabel", "SetLabelObject", "SetValue", "SetTargetRange", "SetReset", "SetAbsoluteRange"):
            need(k not in f or T("EFFECT_TYPE_SINGLE") or T("EFFECT_TYPE_FIELD") or typ == K["EFFECT_TYPE_EQUIP"],
                 f"effect: {k}")
        if code == K["EFFECT_SPSUMMON_PROC"]:
            # Conditions and payments reuse the VM; procedures never create a chain.
            need(typ == K["EFFECT_TYPE_FIELD"] and f.get("SetRange") == [core.Z_HAND],
                 "procedure: type / location")
            need(f.get("SetProperty", [0])[0] & ~(K["EFFECT_FLAG_UNCOPYABLE"] | K["EFFECT_FLAG_CANNOT_DISABLE"]) == 0,
                 "procedure: properties")
            need(not any(k in f for k in ("SetCost", "SetReset",
                                         "SetTargetRange", "SetAbsoluteRange", "SetLabel", "SetLabelObject")),
                 "procedure: separate cost or metadata not represented")
            need(f.get("SetValue", [0]) == [0], "procedure: summon type")
            self.nreg = 0
            self.on = 0
            self.pending = []
            self.hreg = self.egreg = None
            self.chaining = self.field_trig = self.chain_activate = False
            self.hz = (core.Z_HAND, 0)
            prog = Prog(self, "cost")
            if "SetCondition" in f:
                fn = f["SetCondition"][0]
                need(isinstance(fn, Fn) and fn.node is not None and len(fn.node.args) == 2,
                     "procedure: condition with complex parameters")
                earg, carg = fn.node.args
                need(isinstance(earg, A.Name) and isinstance(carg, A.Name), "procedure: parameters")
                env = {earg.id: Eff(), carg.id: Handler()}
                body = fn.node.body.body
                # The API's c==nil test only announces the existence of the procedure.
                if body and isinstance(body[0], A.If):
                    first = body[0]
                    test = first.test
                    need(isinstance(test, A.EqToOp) and isinstance(test.left, A.Name) and
                         test.left.id == carg.id and isinstance(test.right, A.Nil) and first.orelse is None and
                         len(first.body.body) == 1 and isinstance(first.body.body[0], A.Return) and
                         len(first.body.body[0].values) == 1 and isinstance(first.body.body[0].values[0], A.TrueExpr),
                         "procedure: availability test")
                    body = body[1:]
                need(body and isinstance(body[-1], A.Return) and len(body[-1].values) == 1,
                     "procedure: condition shape")
                for st in body[:-1]:
                    self.stmt(st, env, prog, cond=True)
                self.check(body[-1].values[0], env, prog, cond=True)
                self.flush(prog)
            self.in_proc = True
            self.label_object = None
            self.hreg = self.egreg = None
            self.tinfo = {}
            self.treg = None
            target = Prog(self, "target")
            operation = Prog(self, "resolve")
            try:
                for method, program in (("SetTarget", target), ("SetOperation", operation)):
                    if method in f:
                        fn = f[method][0]
                        need(isinstance(fn, Fn) and fn.node is not None and len(fn.node.args) <= 9,
                             "procedure: complex callback")
                        env = self.fn_env(fn.node)
                        if len(fn.node.args) == 9:
                            env[fn.node.args[8].id] = Handler()
                        self.block(fn.node.body.body, env, program)
                        self.flush(program)
            finally:
                self.in_proc = False
            payment = target.ins + operation.ins
            # Payments that could free a zone need a dedicated check.
            # They are admitted here only if the condition already requires a free zone.
            zones = {}
            field_payment = False
            for op, a, b, cc in payment:
                if op == core.OP_FILTER:
                    within, _ = cc
                    zones[a] = (b[0] | b[1]) if within is None else zones.get(within, ALLZ) & (b[0] | b[1])
                elif op == core.OP_SELF:
                    zones[a] = core.Z_HAND
                elif op == core.OP_CHOOSE:
                    zones[a] = zones.get(b, ALLZ)
                elif op == core.OP_MOVE:
                    field_payment |= bool(zones.get(a, ALLZ) & core.Z_MZONE)
            require_free = any(op == core.OP_REQ_FREE and b == core.Z_MZONE and cc >= 1
                               for op, a, b, cc in prog.ins)
            need(not field_payment or require_free, "procedure: zone freed by the payment")
            return (core.K_PROC, 0, self.count_limit(f), core.Z_HAND, prog.ins, payment, [], {"revive": int(self.revive)})
        if code == K["EFFECT_FUSION_SUBSTITUTE"]:
            need(typ == K["EFFECT_TYPE_SINGLE"] and not f.get("SetProperty", [0])[0], "substitute: type / properties")
            need(not any(k in f for k in ("SetValue", "SetRange", "SetCountLimit", "SetReset")), "substitute: value / dynamic range")
            zones = ALLZ
            if "SetCondition" in f:
                fn = f["SetCondition"][0]
                need(isinstance(fn, Fn) and fn.node is not None and len(fn.node.body.body) == 1, "substitute: condition")
                statement = fn.node.body.body[0]
                need(isinstance(statement, A.Return) and len(statement.values) == 1, "substitute: condition")
                test = statement.values[0]
                need(isinstance(test, A.Invoke) and test.func.id == "IsLocation" and len(test.args) == 1,
                     "substitute: condition other than a zone")
                owner = test.source
                need(isinstance(owner, A.Invoke) and owner.func.id == "GetHandler" and not owner.args and
                     isinstance(owner.source, A.Name) and owner.source.id == fn.node.args[0].id, "substitute: another card")
                zones = self.ie_val(test.args[0], {})
                need(type(zones) is int, "substitute: non-constant zones")
            return C.cont(zones, core.MC_FUS_SUB, 0)
        prop = f.get("SetProperty", [0])[0] or 0
        okprop = K["EFFECT_FLAG_CARD_TARGET"] | K["EFFECT_FLAG_PLAYER_TARGET"] | K["EFFECT_FLAG_DELAY"]
        opt = self.count_limit(f)
        self.nreg = 0
        self.chaining = code == K["EVENT_CHAINING"]
        self.field_trig = False
        self.chain_activate = False
        on = 0
        # location of the card when the effect resolves (IsRelateToEffect of the card itself): (locations, position)
        rng = f.get("SetRange", [0])[0]
        if rng and rng & K["LOCATION_PZONE"]:  # Pendulum Zone = zones 0 / 4 of the Spell/Trap zone (Pendulum card)
            rng = (rng & ~K["LOCATION_PZONE"]) | K["LOCATION_SZONE"]
        if T("EFFECT_TYPE_ACTIVATE"):
            self.hz = (core.Z_SZONE, 0)
        elif rng == K["LOCATION_MZONE"]:
            self.hz = (core.Z_MZONE, core.P_FACEUP)
        elif rng == K["LOCATION_SZONE"]:
            self.hz = (core.Z_SZONE, core.P_FACEUP)
        elif rng in (K["LOCATION_HAND"], K["LOCATION_GRAVE"]):
            self.hz = (rng, 0)
        else:
            self.hz = None
        if T("EFFECT_TYPE_ACTIVATE"):
            need(typ == K["EFFECT_TYPE_ACTIVATE"], "compound effect type")
            if code != K["EVENT_FREE_CHAIN"]:
                need(code in EVCODE or code == K["EVENT_CHAINING"], f"event {self.kname(code, 'EVENT_')}")
                on = 1 << EVCODE[code] if code in EVCODE else 0
            need(opt != core.OPT_CARD, "activated card with per-card limit")
            need(prop & ~(okprop | K["EFFECT_FLAG_DAMAGE_STEP"] | K["EFFECT_FLAG_DAMAGE_CAL"]) == 0,
                 f"properties {prop:#x}")
            dmg = 0
            if prop & K["EFFECT_FLAG_DAMAGE_STEP"]:
                need(self.fn_name(f.get("SetCondition")) == "aux.StatChangeDamageStepCondition", "DAMAGE_STEP without StatChange")
                dmg = C.STAT_DS
            need(not prop & K["EFFECT_FLAG_DAMAGE_CAL"], "DAMAGE_CAL")
            progs = self.programs(e, on, cond_skip=dmg != 0)
            if code == K["EVENT_CHAINING"]:
                need(any(i[0] == core.OP_COND_CHAIN for i in progs[0]), "EVENT_CHAINING without a chain link condition")
            return C.act(cost=progs[0], target=progs[1], resolve=progs[2], on=on, dmg=dmg, opt=opt)
        if T("EFFECT_TYPE_IGNITION"):
            need(typ == K["EFFECT_TYPE_IGNITION"], "compound effect type")
            zs = K["LOCATION_MZONE"] | K["LOCATION_HAND"] | K["LOCATION_GRAVE"] | K["LOCATION_SZONE"]
            need(rng and rng & ~zs == 0, f"ignition from {rng:#x}")
            need(prop & ~okprop == 0, f"properties {prop:#x}")
            progs = self.programs(e, 0)
            return C.ignition(cost=progs[0], target=progs[1], resolve=progs[2], opt=opt,
                              rng=0 if rng == K["LOCATION_MZONE"] else rng)
        if T("EFFECT_TYPE_TRIGGER_O") or T("EFFECT_TYPE_TRIGGER_F") or T("EFFECT_TYPE_FLIP"):
            fld = bool(T("EFFECT_TYPE_FIELD"))
            need(fld or T("EFFECT_TYPE_SINGLE") or T("EFFECT_TYPE_FLIP"), "trigger neither SINGLE nor FIELD")
            zs = K["LOCATION_MZONE"] | K["LOCATION_HAND"] | K["LOCATION_GRAVE"] | K["LOCATION_SZONE"]
            need(not fld or (rng and rng & ~zs == 0), f"FIELD trigger from {rng:#x}")
            self.field_trig = fld
            need(prop & ~(okprop | K["EFFECT_FLAG_DAMAGE_STEP"]) == 0, f"properties {prop:#x}")
            if T("EFFECT_TYPE_FLIP"):
                on = 1 << core.EV_FLIP
            else:
                need(code in EVCODE, f"event {self.kname(code, 'EVENT_')}")
                on = 1 << EVCODE[code]
            need(opt in (0, core.OPT_NAME) or opt >= core.OPT_NAMEK, "trigger with per-card limit")
            progs = self.programs(e, on)
            when = bool(T("EFFECT_TYPE_TRIGGER_O")) and not prop & K["EFFECT_FLAG_DELAY"]
            return C.trigger(on | (core.TRIG_WHEN if when else 0), cost=progs[0], target=progs[1], resolve=progs[2],
                             optional=bool(T("EFFECT_TYPE_TRIGGER_O")), opt=opt if opt != core.OPT_CARD else 0,
                             rng=rng if fld else 0, dmg=core.DMG_STEP if prop & K["EFFECT_FLAG_DAMAGE_STEP"] else 0,
                             cond=self.condition_length)
        if T("EFFECT_TYPE_QUICK_O"):
            need(typ == K["EFFECT_TYPE_QUICK_O"], "compound effect type")
            zs = K["LOCATION_MZONE"] | K["LOCATION_HAND"] | K["LOCATION_GRAVE"] | K["LOCATION_SZONE"]
            need(rng and rng & ~zs == 0, f"quick effect from {rng:#x}")
            need(code == K["EVENT_FREE_CHAIN"] or code == K["EVENT_CHAINING"], f"quick effect on {self.kname(code, 'EVENT_')}")
            need(prop & ~(okprop | K["EFFECT_FLAG_DAMAGE_STEP"]) == 0, f"properties {prop:#x}")
            dmg = 0
            if prop & K["EFFECT_FLAG_DAMAGE_STEP"]:
                need(self.fn_name(f.get("SetCondition")) == "aux.StatChangeDamageStepCondition", "DAMAGE_STEP without StatChange")
                dmg = C.STAT_DS
            progs = self.programs(e, 0, cond_skip=dmg != 0)
            return C.quick(rng, cost=progs[0], target=progs[1], resolve=progs[2], opt=opt, dmg=dmg)
        if typ in (K["EFFECT_TYPE_FIELD"], K["EFFECT_TYPE_SINGLE"], K["EFFECT_TYPE_EQUIP"]) and code in MODC:
            return self.continuous(e, typ, code)
        raise Unsupported(f"effect type {self.kname(typ, 'EFFECT_TYPE_')}" + (f" {self.kname(code, 'EFFECT_')}" if code else ""))

    def continuous(self, e, typ, code):
        """Continuous effect (FIELD on monsters / SINGLE on the card) -> cont(...)."""
        f = e.f
        mc = MODC[code]
        equip = typ == K["EFFECT_TYPE_EQUIP"]
        need(mc != core.MC_DISABLE or typ == K["EFFECT_TYPE_FIELD"] or equip, "DISABLE SINGLE")
        prop = f.get("SetProperty", [0])[0] or 0
        need(prop & ~(K["EFFECT_FLAG_SINGLE_RANGE"] | K["EFFECT_FLAG_IGNORE_IMMUNE"] | K["EFFECT_FLAG_CANNOT_DISABLE"]
                      | K["EFFECT_FLAG_SET_AVAILABLE"]) == 0, f"continuous: properties {prop:#x}")
        need(not prop & K["EFFECT_FLAG_SET_AVAILABLE"] or mc in (core.MC_INDES_BATTLE, core.MC_INDES_EFF),
             "SET_AVAILABLE")
        for k in ("SetReset", "SetLabel", "SetLabelObject", "SetCountLimit", "SetCost", "SetOperation"):
            need(k not in f, f"continuous: {k}")
        rng = f.get("SetRange", [0])[0]
        if rng == K["LOCATION_PZONE"]:
            rng = core.Z_SZONE
        if typ == K["EFFECT_TYPE_SINGLE"] and not rng:
            rng = core.Z_MZONE
        if equip:
            need(not rng and "SetTarget" not in f and "SetTargetRange" not in f, "equip: location or target")
            need(mc not in (core.MC_NO_SS, core.MC_P_NO_TRAP), "equip: player prohibition")
            rng = core.Z_SZONE
        need(rng in (core.Z_MZONE, core.Z_SZONE), "continuous: card location")
        tgt = None
        kw = {}
        if typ == K["EFFECT_TYPE_FIELD"]:
            tr = f.get("SetTargetRange")
            need(tr is not None and len(tr) == 2 and all(isinstance(x, int) for x in tr), "continuous: TargetRange")
            if mc == core.MC_NO_SS:
                raise Unsupported("continuous: CANNOT_SPECIAL_SUMMON")
            need(tr[0] & ~core.Z_MZONE == 0 and tr[1] & ~core.Z_MZONE == 0, "continuous: targets outside the monster zone")
            fl = Filt()
            if "SetTarget" in f:
                self.apply_fn(fl, f["SetTarget"][0], [], {}, two=True)
            tgt = fl.build(tr[0], tr[1])
        elif equip:
            tgt = C.EQUIPPED
        else:
            need(not "SetTarget" in f, "continuous SINGLE with target")
        val = f.get("SetValue", [None])[0]
        per = None
        if mc in STAT_CODES:
            if isinstance(val, Fn):
                per, mul = self.cont_value(val)
                val = mul
            need(isinstance(val, int) and not isinstance(val, bool), "continuous: value")
        elif mc == core.MC_INDES_BATTLE:
            if isinstance(val, Fn):
                val = self.indes_value(val)
            else:
                need(val in (1, True), "continuous: INDESTRUCTABLE_BATTLE value")
                val = 0
        elif mc == core.MC_NO_BTARGET:
            need(isinstance(val, Fn) and val.name == "aux.imval1" or val in (None, 1), "continuous: BTARGET value")
            val = 0
        elif mc == core.MC_NO_ETARGET or mc == core.MC_INDES_EFF:
            need(val in (1, True), f"continuous: value {val}")
            val = 0
        else:
            need(val in (None, 0, 1, True), f"continuous: value {val}")
            val = 0
        cnt = None
        if "SetCondition" in f:
            tp, ph, cnt = self.cont_cond(f["SetCondition"][0])
            if tp:
                kw["tp"] = tp
            if ph:
                kw["ph"] = ph
        need(per is None or cnt is None, "continuous: count and condition")
        return C.cont(rng, mc, val, tgt=tgt, per=per, cnt=cnt, **kw)

    def cont_value(self, fn):
        """SetValue(function(e,c) return Duel.GetMatchingGroupCount(...)*n end) -> (count filter, n)."""
        need(fn.node is not None, f"continuous: value {fn.name}")
        body = fn.node.body.body
        need(len(body) == 1 and isinstance(body[0], A.Return), "continuous: value (body)")
        x = body[0].values[0]
        need(isinstance(x, A.MultOp) and isinstance(x.right, A.Number) and isinstance(x.left, A.Call), "continuous: value")
        d = dotted(x.left.func)
        env = self.fn_env(fn.node)
        if len(fn.node.args) > 1:
            env[fn.node.args[1].id] = "SELFCARD"
        a = x.left.args
        if d == "Duel.GetMatchingGroupCount":
            F_, _ = self.mkfilter(a[0], a[5:], a[1], a[2], a[3], a[4] if len(a) > 4 else None, env)
        elif d == "Duel.GetFieldGroupCount":
            F_, _ = self.mkfilter(A.Nil(), [], a[0], a[1], a[2], None, env)
        else:
            raise Unsupported(f"continuous: value {d}")
        return F_, x.right.n

    def indes_value(self, fn):
        need(fn.node is not None, f"continuous: INDES value {fn.name}")
        body = fn.node.body.body
        need(len(body) == 1 and isinstance(body[0], A.Return), "continuous: INDES value")
        x = body[0].values[0]
        need(isinstance(x, A.GreaterOrEqThanOp) and isinstance(x.right, A.Number) and isinstance(x.left, A.Invoke)
             and x.left.func.id == "GetAttack" and isinstance(x.left.source, A.Name)
             and x.left.source.id == fn.node.args[1].id, "continuous: INDES value")
        return x.right.n

    def cont_cond(self, fn):
        """Condition of a continuous effect: (tp, ph, cnt)."""
        need(fn.node is not None, f"continuous: condition {fn.name}")
        env = self.fn_env(fn.node)
        body = fn.node.body.body
        need(len(body) == 1 and isinstance(body[0], A.Return), "continuous: condition")
        tp = ph = 0
        cnt = None
        for x in self.conj(body[0].values[0]):
            if isinstance(x, A.Call) and dotted(x.func) == "Duel.IsTurnPlayer":
                p = self.expr(x.args[0], env, None)
                need(isinstance(p, Player), "continuous: IsTurnPlayer")
                tp = p.p + 1
            elif isinstance(x, A.Call) and dotted(x.func) == "Duel.IsBattlePhase":
                ph |= core.PH_BATTLE
            elif isinstance(x, A.Call) and dotted(x.func) == "Duel.IsMainPhase":
                ph |= core.PH_MAIN1 | core.PH_MAIN2
            elif isinstance(x, (A.GreaterThanOp, A.GreaterOrEqThanOp, A.EqToOp, A.LessThanOp, A.LessOrEqThanOp)) \
                    and isinstance(x.left, A.Call) and dotted(x.left.func) in ("Duel.GetFieldGroupCount",
                                                                               "Duel.GetMatchingGroupCount") \
                    and isinstance(x.right, A.Number):
                need(cnt is None, "continuous: two counts")
                a = x.left.args
                if dotted(x.left.func) == "Duel.GetFieldGroupCount":
                    F_, _ = self.mkfilter(A.Nil(), [], a[0], a[1], a[2], None, env)
                else:
                    F_, _ = self.mkfilter(a[0], a[5:], a[1], a[2], a[3], a[4] if len(a) > 4 else None, env)
                n = x.right.n
                lo, hi = {A.GreaterThanOp: (n + 1, core.BIG), A.GreaterOrEqThanOp: (n, core.BIG), A.EqToOp: (n, n),
                          A.LessThanOp: (0, n - 1), A.LessOrEqThanOp: (0, n)}[type(x)]
                cnt = (F_, lo, hi)
            else:
                raise Unsupported(f"continuous: condition {self.show(x)}")
        return tp, ph, cnt

    def kname(self, v, prefix):
        names = [k for k, x in K.items() if k.startswith(prefix) and x == v]
        if names:
            return names[0]
        bits = [k for k, x in K.items() if k.startswith(prefix) and x and x & (x - 1) == 0 and v & x]
        return "+".join(bits) or f"{prefix}{v:#x}"

    def fn_name(self, v):
        if v is None:
            return None
        v = v[0]
        return v.name if isinstance(v, Fn) else None

    @staticmethod
    def vfilter(r, f, within=None, tgt=False, invert=False, related=False):
        # Do not pack the virtual register into the three bits of the ISA.
        # Fourth field / bit 21: complement of the filter (invert); fifth field / bit 22: relation to the target.
        return (core.OP_FILTER, r, f, (within, bool(tgt), False, bool(invert), bool(related)) if invert or related
                else (within, bool(tgt)))

    @staticmethod
    def vcopy(r, source, related=False):
        # Compiler-only marker: exact copy of the group currently in play.
        zones = ALLZ | core.Z_OVERLAY
        bounds = (-(2**31) + 1, 2**31 - 1)
        return (core.OP_FILTER, r, C.F(me=zones, op=zones, atk=bounds, df=bounds, lv=bounds),
                (source, False, True, False, True) if related else (source, False, True))

    @staticmethod
    def allocate(effect):
        """SSA then allocation of the live groups to the four physical registers.

        Identical queries are shared only without an intervening mutation.
        Resolution opens a new epoch: answers may have changed the field.
        No change to the JAX state shapes or to the generator programs.
        """
        if not any(effect[4:7]):
            if len(effect) > 7 and effect[7].get("cond", 0):
                extra = dict(effect[7], cond=0)
                return (*effect[:7], extra, *effect[8:])
            return effect
        condition_boundary = int(effect[7].get("cond", 0)) if len(effect) > 7 else 0
        pure = {core.OP_FILTER, core.OP_SELF, core.OP_MINATK, core.OP_EVENT,
                core.OP_CHAINCARD, core.OP_ATTACKER}
        definitions = pure | {core.OP_CHOOSE, core.OP_FUSMAT, core.OP_RITMAT}
        read_a = {core.OP_REQ, core.OP_MOVE, core.OP_SPSUMMON, core.OP_LOCK,
                  core.OP_MODIFY, core.OP_LABEL}
        read_b = {core.OP_MINATK, core.OP_CHOOSE, core.OP_FUSMAT, core.OP_RITMAT}
        # Any opcode that may observe/alter a decision invalidates the queries.
        predicates = {core.OP_REQ, core.OP_REQ_FREE, core.OP_COND_CHAIN, core.OP_COND_TURN,
                      core.OP_COND_EVP, core.OP_COND_SELF, core.OP_COND_LABEL,
                      core.OP_COND_PHASE, core.OP_COND_ATK, core.OP_LABEL}
        mutated = {a for program in effect[4:7] for op,a,b,c in program
                   if op == core.OP_SPSUMMON or op == core.OP_MOVE and c & 1}
        # Last read of a virtual register: a copy costs nothing if the input is no longer used after it,
        # including across the phases of a chain link.
        last_read = {}
        flat = [ins for program in effect[4:7] for ins in program]
        in_blk, blk_mut, done_mut = False, set(), set()
        for index, (op, a, b, c) in enumerate(flat):
            if op == core.OP_BLOCK:
                in_blk = a >= 0
                if not in_blk:
                    done_mut |= blk_mut
                    blk_mut = set()
            elif in_blk and (op == core.OP_SPSUMMON or op == core.OP_MOVE and c & 1):
                blk_mut.add(a)
            reads = []
            if op in read_a:
                reads.append(a)
            if op in read_b:
                reads.append(b)
            if op == core.OP_FILTER and c[0] is not None:
                reads.append(c[0])
            if op == core.OP_DAMAGE and c:
                reads.append(c - 1)
            for reg in reads:
                need(reg not in done_mut, "group modified in an if followed by statements, read afterwards")
                last_read[reg] = index
        current, queries, entries = {}, {}, []
        next_value = 0
        index = -1
        for phase, program in enumerate(effect[4:7]):
            if phase == core.PG_RESOLVE:
                queries.clear()
            for local_index, (op, a, b, c) in enumerate(program):
                index += 1
                # Plain copy, read for the last time here: the source register is taken over as is. A copy that
                # carries the complement (4th field) or the relation to the target (5th field) is not elided.
                if op == core.OP_FILTER and len(c) >= 3 and c[2] and not any(c[3:]) and last_read.get(c[0]) == index:
                    need(c[0] in current, "copy of an uninitialized register")
                    current[a] = current[c[0]]
                    continue
                operands = [a, b, c]
                reads = {}
                if op in read_a:
                    reads[0] = current[a]
                if op in read_b:
                    reads[1] = current[b]
                if op == core.OP_FILTER:
                    within, target = c[:2]
                    if within is not None:
                        reads[2] = current[within]
                    operands[2] = (int(target) | (2 if len(c) >= 3 and c[2] else 0)
                                   | (4 if len(c) >= 4 and c[3] else 0) | (8 if len(c) >= 5 and c[4] else 0))
                elif op == core.OP_DAMAGE and c:
                    reads[2] = current[c-1]
                writes = op in definitions or op == core.OP_SPSUMMON or (op == core.OP_MOVE and c & 1)
                key = (op, b, operands[2], tuple(sorted(reads.items())))
                reusable = op == core.OP_REQ or op in pure and a not in mutated
                if reusable and key in queries:
                    if op in pure:
                        current[a] = queries[key]
                    continue
                value = None
                if writes:
                    value = next_value
                    next_value += 1
                    current[a] = value
                is_condition = phase == core.PG_COST and local_index < condition_boundary
                entries.append((phase, op, operands, reads, value, is_condition))
                if reusable:
                    queries[key] = value
                elif op not in predicates:
                    queries.clear()
        # Remove unused pure results; selections remain observable.
        live, kept = set(), []
        for entry in reversed(entries):
            phase, op, operands, reads, value, is_condition = entry
            if op in pure and value not in live:
                continue
            if value is not None:
                live.discard(value)
            live.update(reads.values())
            kept.append(entry)
        kept.reverse()
        last = {v: i for i, entry in enumerate(kept) for v in entry[3].values()}
        slots, programs = {}, [[], [], []]
        condition_length = 0
        for i, (phase, op, operands, reads, value, is_condition) in enumerate(kept):
            lowered = operands.copy()
            for index, v in reads.items():
                need(v in slots, "uninitialized virtual register")
                lowered[index] = slots[v]
            # Operands are read before the write, allowing dst == src.
            slots = {v: slot for v, slot in slots.items() if last.get(v, -1) > i}
            if value is not None:
                free = sorted(set(range(core.R)) - set(slots.values()))
                need(bool(free), "more than 4 live registers")
                inplace = op == core.OP_SPSUMMON or op == core.OP_MOVE and operands[2] & 1
                destination = lowered[0] if inplace else free[0]
                need(destination not in slots.values(), "mutation of a shared group")
                lowered[0] = destination
                if last.get(value, -1) > i:
                    slots[value] = destination
            if op == core.OP_FILTER:
                lowered[2] = (((lowered[2]+1) if 2 in reads else 0) | (8 if operands[2] & 1 else 0)
                              | ((1 << 20) if operands[2] & 2 else 0) | ((1 << 21) if operands[2] & 4 else 0)
                              | ((1 << 22) if operands[2] & 8 else 0))
            elif op == core.OP_DAMAGE and 2 in reads:
                lowered[2] += 1
            programs[phase].append((op, *lowered))
            if is_condition:
                condition_length += 1
        for phase, program in enumerate(programs):
            out = []
            for ins in program:  # BLOCK: a = index of the first instruction after the block
                if ins[0] == core.OP_BLOCK and ins[1] < 0:
                    start = max(i for i, x in enumerate(out) if x[0] == core.OP_BLOCK and x[1] == 0)
                    out[start] = (core.OP_BLOCK, len(out), 0, 0)
                else:
                    out.append(ins)
            need(phase == core.PG_RESOLVE or not any(x[0] == core.OP_BLOCK for x in out), "BLOCK outside resolution")
            programs[phase] = program = out
            limit = core.P-1 if phase == core.PG_RESOLVE else core.PT
            need(len(program) <= limit, f"program {phase} > {limit} instructions")
        filters = {ins[2] for program in programs for ins in program if ins[0] in (core.OP_FILTER, core.OP_SSLOCK)}
        filters |= {ins[3] for program in programs for ins in program if ins[0] == core.OP_SSLOCK and ins[3]}  # exceptions
        if len(effect) > 7:
            filters.update(effect[7].get("filters", ()))
        need(len(filters) <= core.NF, f"more than {core.NF} filters")
        tail = effect[7:]
        if tail and "cond" in tail[0]:
            tail = (dict(tail[0], cond=condition_length), *tail[1:])
        return (*effect[:4], *programs, *tail)

    def programs(self, e, on, cond_skip=False):
        f = fold_labels(e.f)
        self.on = on
        self.tinfo = {}  # SetTargetPlayer / SetTargetParam
        self.treg = None  # target register (SelectTarget)
        self.public_targets = bool(self.flag(e, "EFFECT_FLAG_CARD_TARGET"))
        self.pending = []  # activation tests not yet emitted (merged with a following selection)
        self.nrc = None  # register of a Duel.NegateRelatedChain waiting for its EFFECT_DISABLE
        self.in_block = False  # inside an if followed by statements (BLOCK)
        self.hreg = self.egreg = None
        self.chain_field = self.chain_nact = False
        cost = Prog(self, "cost")
        if "SetCondition" in f and not cond_skip:
            self.run_cond(f["SetCondition"][0], cost)
        self.flush(cost)
        self.condition_length = len(cost.ins)
        if "SetCost" in f:
            self.run_fn(f["SetCost"][0], cost)
        self.flush(cost)
        tgt = Prog(self, "target")
        if "SetTarget" in f:
            self.run_fn(f["SetTarget"][0], tgt)
        self.flush(tgt)
        res = Prog(self, "resolve")
        need("SetOperation" in f or not ("SetTarget" in f), "target without operation")
        if "SetOperation" in f:
            self.run_fn(f["SetOperation"][0], res)
        self.flush(res)
        return cost.ins, tgt.ins, res.ins

    # ------------------------------------------------ known functions (Cost.*, aux.*)
    def run_known(self, name, prog):
        if name == "Cost.SelfToGrave" or name == "Cost.SelfTribute" or name == "Cost.SelfDiscardToGrave" \
                or name == "Cost.SelfDiscard":
            r = prog.reg()
            reason = core.REASON_COST
            if name == "Cost.SelfTribute":
                reason |= core.REASON_RELEASE
            if name in ("Cost.SelfDiscardToGrave", "Cost.SelfDiscard"):
                reason |= core.REASON_DISCARD
            prog.emit(C.SELF(r), C.MOVE(r, reason=reason))
            return
        if name == "Cost.PayLP":
            need(len(self.cur_fn.bound) == 1 and isinstance(self.cur_fn.bound[0], int), "Cost.PayLP")
            prog.emit(C.PAY_LP(self.cur_fn.bound[0]))
            return
        if name == "Cost.DetachFromSelf":  # (min[, max[, op]]): only min = max, without op
            b = list(self.cur_fn.bound) + [None, None]
            need(isinstance(b[0], int) and b[1] in (None, b[0]) and b[2] is None, "variable Cost.DetachFromSelf")
            r = prog.reg()
            r2 = prog.reg()
            prog.emit(self.vfilter(r, C.F(me=core.Z_OVERLAY)), C.CHOOSE(r2, r, b[0]), C.MOVE(r2, reason=core.REASON_COST))
            return
        if name == "Cost.SelfBanish":
            r = prog.reg()
            prog.emit(C.SELF(r), C.MOVE(r, core.Z_REMOVED, reason=core.REASON_COST))
            return
        raise Unsupported(f"function {name}")

    def run_cond(self, fn, prog):
        need(isinstance(fn, Fn), "condition")
        if fn.node is None:
            if fn.name in ("aux.StatChangeDamageStepCondition",):
                raise Unsupported("StatChangeDamageStepCondition outside DAMAGE_STEP")
            raise Unsupported(f"condition {fn.name}")
        env = self.fn_env(fn.node)
        body = fn.node.body.body
        need(len(body) >= 1, "empty condition")
        for st in body[:-1]:
            self.stmt(st, env, prog, cond=True)
        need(isinstance(body[-1], A.Return) and len(body[-1].values) == 1, "condition: shape")
        self.check(body[-1].values[0], env, prog, cond=True)

    def fn_env(self, node):
        names = ["e", "tp", "eg", "ep", "ev", "re", "r", "rp", "chk", "chkc"]
        env = {}
        for i, a in enumerate(node.args):
            need(isinstance(a, A.Name), "parameters")
            if i >= len(names):
                env[a.id] = Arg(a.id)
                continue
            n = names[i]
            env[a.id] = {"e": Eff(), "tp": Player(ME), "eg": EventGroup(), "re": ChainEff()}.get(n, Arg(n))
        return env

    def run_fn(self, fn, prog):
        need(isinstance(fn, Fn), f"{prog.kind}: not a function")
        if fn.node is None:
            self.cur_fn = fn
            return self.run_known(fn.name, prog)
        env = self.fn_env(fn.node)
        fold_absent(fn.node)
        step_to_summon(fn.node)
        self.nrc = None
        self.block(fn.node.body.body, env, prog)
        need(self.nrc is None, "NegateRelatedChain without EFFECT_DISABLE on the same card")

    def flush(self, prog):
        for chk in self.pending:
            if not chk["used"]:
                self.emit_check(chk, prog)
        self.pending = []

    def emit_check(self, chk, prog):
        r = prog.reg()
        prog.emit(self.vfilter(r, chk["F"], tgt=chk["tgt"]), C.REQ(r, chk["n"]))
        chk["used"] = True

    # ------------------------------------------------ statements
    def block(self, stmts, env, prog):
        for i, st in enumerate(stmts):
            rest = stmts[i + 1:]
            if _dead_getfirst_loop_prelude(st, rest):
                continue
            if self.stmt(st, env, prog, rest=rest) == "stop":
                return "stop"

    def is_name(self, node, env, cls):
        return isinstance(node, A.Name) and isinstance(env.get(node.id), cls)

    def stmt(self, st, env, prog, rest=(), cond=False):
        if isinstance(st, A.If):
            return self.stmt_if(st, env, prog, rest)
        if isinstance(st, (A.LocalAssign, A.Assign)):
            vals = []
            for i, v in enumerate(st.values):
                x = self.expr(v, env, prog, nret=len(st.targets) - i if i == len(st.values) - 1 else 1)
                vals.extend(x if isinstance(x, list) else [x])
            for t, v in zip(st.targets, vals):
                need(isinstance(t, A.Name), "non-local assignment")
                env[t.id] = v
            return
        if isinstance(st, A.Call) and dotted(st.func) == "Duel.SendtoDeck":
            self.expr(st, env, prog, nret=0, statement=True)
            return
        if isinstance(st, (A.Call, A.Invoke)):
            self.expr(st, env, prog, statement=True)
            return
        if isinstance(st, A.Return) and getattr(self, "in_proc", False) and len(st.values) == 1:
            need(isinstance(st.values[0], A.TrueExpr), "procedure: conditional return not represented")
            return "stop"
        if isinstance(st, A.Return) and not st.values:
            return "stop"
        if isinstance(st, A.Forin) and len(st.targets) == 1 and len(st.iter) == 1:
            it = st.iter[0]
            if isinstance(it, A.Call) and dotted(it.func) == "aux.Next":
                g = self.expr(it.args[0], env, prog)
            elif isinstance(it, A.Invoke) and it.func.id == "Iter":
                g = self.expr(it.source, env, prog)
            else:
                raise Unsupported("for loop")
            env[st.targets[0].id] = self.as_reg(g, prog)
            need(not [s for s in rest if not self.ignorable(s)] or True, "")
            return self.block(st.body.body, env, prog)
        raise Unsupported(f"statement {type(st).__name__}")

    def stmt_if(self, st, env, prog, rest):
        t = st.test
        # if chkc then ... end: target redirection, no effect here
        if isinstance(t, A.Name) and isinstance(env.get(t.id), Arg) and env[t.id].name == "chkc":
            return
        # if chk==0 then return COND end
        if isinstance(t, A.EqToOp) and isinstance(t.left, A.Name) and isinstance(env.get(t.left.id), Arg) \
                and env[t.left.id].name == "chk":
            need(st.orelse is None or isinstance(st.orelse, A.Block), "chk==0 with elseif")
            b = st.body.body
            # if A then return false end ... return B  ->  return (not A) and ... and B
            while len(b) > 1 and isinstance(b[0], A.If) and b[0].orelse is None and len(b[0].body.body) == 1 \
                    and isinstance(b[0].body.body[0], A.Return) and len(b[0].body.body[0].values) == 1 \
                    and isinstance(b[0].body.body[0].values[0], A.FalseExpr):
                rest = b[1:]
                if len(rest) == 1 and isinstance(rest[0], A.Return) and len(rest[0].values) == 1:
                    b = [A.Return([A.AndLoOp(negate(b[0].test), rest[0].values[0])])]
                else:
                    break
            need(len(b) == 1 and isinstance(b[0], A.Return) and len(b[0].values) == 1, "chk==0: shape")
            need(prog.kind in ("cost", "target"), "chk==0 outside cost / target")
            self.check(b[0].values[0], env, prog)
            if st.orelse is not None:
                return self.block(st.orelse.body, env, prog)
            return
        body = st.body.body
        if getattr(self, "in_proc", False):
            if isinstance(t, A.ULNotOp) and self.is_name(t.operand, env, Reg) and len(body) == 1 and isinstance(body[0], A.Return) and not body[0].values:
                return  # a labeled and kept group always exists at this point
            if st.orelse is None and body and isinstance(body[-1], A.Return) and len(body[-1].values) == 1 and isinstance(body[-1].values[0], A.TrueExpr):
                self.check(t, env, prog)
                return self.block(body, env, prog)
        src = self.show(t)
        if st.orelse is None and all(self.ignorable(b) for b in body) and _pure(t):  # an action in the test counts
            return
        if len(body) == 1 and isinstance(body[0], A.Return) and len(body[0].values) == 1 and st.orelse is None and \
                src in ("not eg", "Duel.IsPlayerAffectedByEffect(tp, EFFECT_DISCARD_COST_CHANGE)"):
            return
        # if COND then return end: stop (resolution)
        if len(body) == 1 and isinstance(body[0], A.Return) and not body[0].values and st.orelse is None:
            need(prog.kind == "resolve", "if … then return end outside resolution")
            self.res_cond(t, env, prog, negate=True)
            return
        need(st.orelse is None, "if … else")
        # A MOVE on an empty group is already an empty operation. This local proof
        # allows keeping the statements after the if (Morphing Jar), without assuming
        # that a non-emptiness test also authorizes a constant Draw/Damage.
        if isinstance(t, A.GreaterThanOp) and isinstance(t.right, A.Number) and t.right.n == 0:
            source = t.left.operand if isinstance(t.left, A.ULengthOP) else (t.left.source if isinstance(t.left, A.Invoke) and t.left.func.id == "GetCount" else None)
            if isinstance(source, A.Name) and isinstance(env.get(source.id), (Reg,Lazy,EventGroup)) and all(
                    self.ignorable(st) or isinstance(st, A.Call) and dotted(st.func) in
                    ("Duel.SendtoGrave", "Duel.SendtoHand", "Duel.Remove", "Duel.Destroy", "Duel.Release") and
                    st.args and isinstance(st.args[0], A.Name) and st.args[0].id == source.id for st in body):
                return self.block(body, env, prog)
        # if COND then BODY end: resolution stops if COND is false; only if nothing follows
        need(prog.kind == "resolve" or self.always_true(t, env), f"if in {prog.kind}")
        if not self.always_true(t, env) and [s for s in rest if not self.ignorable(s)]:
            return self.if_block(st, env, prog)
        self.res_cond(t, env, prog)
        return self.block(body, env, prog)

    def if_block(self, st, env, prog):
        """if COND then BODY end followed by statements (resolution): BLOCK ... BLOCK_END. A false test (or a "no" to
        SelectYesNo) inside the block jumps to its end instead of stopping the effect. Names bound in the block (locals,
        restrictions from res_cond) are forgotten afterwards."""
        need(not self.in_block, "nested if followed by statements")
        inner = list(_walk_nofn(st.body))
        need(not any(isinstance(n, A.Return) for n in inner), "return in an if followed by statements")
        need(not any(isinstance(n, A.Assign) and any(isinstance(x, A.Name) and x.id in env for x in n.targets)
                     for n in inner), "assignment to an outer variable in an if followed by statements")
        saved = dict(env), self.hreg, self.egreg
        prog.emit(C.BLOCK())
        self.in_block = True
        try:
            self.res_cond(st.test, env, prog)
            self.block(st.body.body, env, prog)
        finally:
            self.in_block = False
        prog.emit(BLOCK_END)
        env.clear()
        env.update(saved[0])
        self.hreg, self.egreg = saved[1:]

    def ignorable(self, st):
        return isinstance(st, A.Call) and dotted(st.func) in IGNORED_CALLS

    def always_true(self, t, env):
        """Presence of an implicit reference / relation; group counts remain real tests."""
        if isinstance(t, A.Name):
            return isinstance(env.get(t.id), (Reg, Handler))
        if isinstance(t, A.Invoke) and t.func.id == "IsRelateToEffect":
            target = isinstance(t.source, A.Name) and isinstance(env.get(t.source.id), Reg) and env[t.source.id].target
            return self.handler_call(t, env) is None and not target
        if isinstance(t, A.AndLoOp):
            return self.always_true(t.left, env) and self.always_true(t.right, env)
        return False

    def res_cond(self, t, env, prog, negate=False):
        """Resolution condition: stop if false (or if true with negate)."""
        if self.always_true(t, env) and not negate:
            return
        if negate and isinstance(t, A.OrLoOp):
            # if a or b then return end: each guard stops the program
            # before evaluating the next one, including any actions it has.
            for term in self.disj(t):
                need(not self.always_true(term, env), "inverse condition of an implicit reference")
                self.res_cond(term, env, prog, negate=True)
            return
        terms = self.conj(t) if not negate else [t]
        # An action in a conjunction observes the conditions already evaluated.
        # Do not delay the card filters until after a destruction or
        # summon that may modify those cards.
        def action(x):
            return (isinstance(x, (A.GreaterThanOp, A.NotEqToOp, A.EqToOp, A.GreaterOrEqThanOp))
                    and isinstance(x.left, A.Call) and dotted(x.left.func) in
                    ("Duel.Destroy", "Duel.SendtoHand", "Duel.SendtoGrave", "Duel.Remove",
                     "Duel.SpecialSummon", "Duel.Draw", "Duel.Release"))
        if len(terms) > 1 and any(map(action, terms)):
            pending = []
            def flush():
                if pending:
                    combined = pending[0]
                    for term in pending[1:]:
                        combined = A.AndLoOp(combined, term)
                    self.res_cond(combined, env, prog)
                    pending.clear()
            for term in terms:
                if action(term):
                    flush()
                    self.res_cond(term, env, prog)
                else:
                    pending.append(term)
            flush()
            return
        refine = collections.defaultdict(list)
        hterms = []
        for x in terms:
            relation = x.operand if negate and isinstance(x, A.ULNotOp) else x
            call = self.handler_call(relation, env)
            target = (isinstance(relation, A.Invoke) and relation.func.id == "IsRelateToEffect" and
                      isinstance(relation.source, A.Name) and isinstance(env.get(relation.source.id), Reg) and
                      env[relation.source.id].target)
            if target:
                need(not negate or isinstance(x, A.ULNotOp), "inverse relation condition not represented")
                reg = env[relation.source.id]
                need(reg.one, "relation to several targets not represented")
                need(len(relation.args) == 1 and isinstance(self.expr(relation.args[0], env, prog), Eff),
                     "relation to another effect")
                prog.emit(C.REQ(reg.r, 1, related=True))
                continue
            if call is not None and call[0] == "IsRelateToEffect":
                need(not negate or isinstance(x, A.ULNotOp), "inverse relation condition not represented")
                need(len(call[1]) == 1 and isinstance(self.expr(call[1][0], env, prog), Eff), "relation to another effect")
                r = prog.reg()
                prog.emit(C.SELF(r, related=True), C.REQ(r, 1))
                continue
            if self.always_true(x, env):
                continue
            if self.handler_call(x, env) is not None and not negate:
                hterms.append(x)
                continue
            act = self.action_test(x, env, prog, negate=negate)
            if act:
                continue
            if isinstance(x, (A.EqToOp, A.NotEqToOp, A.GreaterThanOp, A.GreaterOrEqThanOp, A.LessThanOp, A.LessOrEqThanOp)) and isinstance(x.right, A.Number) and (
                    isinstance(x.left, A.ULengthOP) or
                    isinstance(x.left, A.Name) and isinstance(env.get(x.left.id), Count) or
                    isinstance(x.left, A.Invoke) and x.left.func.id in ("GetCount", "FilterCount") or
                    isinstance(x.left, A.Call) and dotted(x.left.func) in ("Duel.GetMatchingGroupCount", "Duel.GetFieldGroupCount",
                                                                           "Duel.GetCustomActivityCount", "Duel.GetActivityCount")):
                value = self.expr(x.left, env, prog)
                if isinstance(value, Count):
                    self.count_check(value, type(x), x.right.n, prog, negate=negate)
                    continue
            # tc:IsFaceup() etc.: restricts the register
            v = self.term_var(x, env)
            if v is not None and not negate:
                refine[v].append(x)
                continue
            # Duel.GetLocationCount(tp,LOCATION_MZONE)<=0 (negate) / >0
            lc = self.loccount(x, env, negate)
            if lc is not None:
                prog.emit(C.REQ_FREE(*lc))
                continue
            if isinstance(x, A.Invoke) and x.func.id == "IsRelateToEffect":
                continue
            if isinstance(x, A.Call) and dotted(x.func) == "Duel.NegateActivation":
                prog.emit(C.NEGATE())
                continue
            raise Unsupported(f"resolution condition {self.show(x)}")
        for name, xs in refine.items():
            reg = env[name]
            fl = Filt()
            for x in xs:
                self.filter_term(fl, x, env, cvar=name)
            r = prog.reg()
            prog.emit(self.vfilter(r, fl.build(reg.me, reg.op), within=reg.r))
            # The filter conditions the whole branch, including the operations that do not use the card.
            if reg.one:
                prog.emit(C.REQ(r, 1))
            env[name] = Reg(r, reg.me, reg.op, reg.one, reg.handler, reg.target)
            if reg is self.egreg:
                self.egreg = env[name]
        if hterms:
            # the card itself: still where the effect was activated (IsRelateToEffect), plus the conditions
            need(self.hz is not None, "IsRelateToEffect of the card (unknown location)")
            fl = Filt()
            fl.set("pos", self.hz[1]) if self.hz[1] else None
            for x in hterms:
                m, args = self.handler_call(x, env)
                apply_method(fl, m, [self.expr(a, env, prog) for a in args])
            base = self.as_reg(Handler(), prog)
            r = prog.reg()
            prog.emit(self.vfilter(r, fl.build(self.hz[0], 0), within=base.r))
            self.hreg = Reg(r, self.hz[0], 0, True, handler=True)
            for k, v in env.items():
                if isinstance(v, Handler):
                    env[k] = self.hreg

    def action_test(self, x, env, prog, negate=False):
        """Test the real result of Destroy / SpecialSummon, including for a group.
        For the other actions, the continuation is assumed to take place."""
        if isinstance(x, (A.GreaterThanOp, A.NotEqToOp, A.EqToOp, A.GreaterOrEqThanOp)) and isinstance(x.left, A.Call):
            d = dotted(x.left.func)
            if d in ("Duel.Destroy", "Duel.SendtoHand", "Duel.SendtoGrave", "Duel.Remove", "Duel.SpecialSummon",
                     "Duel.Draw", "Duel.Release"):
                need(isinstance(x.right, A.Number), "action test")
                n = self.call(d, x.left.args, env, prog, 1)
                if d in ("Duel.Destroy", "Duel.SpecialSummon") and isinstance(n, Count):
                    self.count_check(n, type(x), x.right.n, prog, negate=negate)
                else:
                    need(not negate, f"inverse test of {d} not represented")
                    self.notes.add(f"{d}(...) assumed successful")
                return True
        if isinstance(x, A.Call) and dotted(x.func) in ("Duel.SpecialSummonStep",):
            raise Unsupported("SpecialSummonStep")
        return False

    def show(self, node):
        try:
            return LA.to_lua_source(node)[:80].replace("\n", " ")
        except Exception:  # noqa: BLE001
            return type(node).__name__

    def loccount(self, x, env, negate):
        """Number of free zones, direct or via a local alias; no new instruction."""
        if not isinstance(x, (A.GreaterThanOp, A.GreaterOrEqThanOp, A.LessThanOp, A.LessOrEqThanOp)) or not isinstance(x.right, A.Number):
            return None
        left = x.left
        known = (isinstance(left, A.Call) and dotted(left.func) == "Duel.GetLocationCount") or \
                (isinstance(left, A.Name) and isinstance(env.get(left.id), FreeCount))
        if not known:
            return None
        value = self.expr(left, env, None)
        if negate and isinstance(x, A.LessThanOp):
            need(type(x.right.n) is int, "GetLocationCount: integer threshold")
            return (value.player.p, value.zone, x.right.n)
        if isinstance(x, A.LessOrEqThanOp if negate else A.GreaterThanOp):
            return (value.player.p, value.zone, x.right.n + 1)
        if not negate and isinstance(x, A.GreaterOrEqThanOp):
            return (value.player.p, value.zone, x.right.n)
        return None

    def conj(self, t):
        if isinstance(t, A.AndLoOp):
            return self.conj(t.left) + self.conj(t.right)
        return [t]

    # ------------------------------------------------ activation tests / conditions
    def term_var(self, x, env):
        """Name of the register variable the term is about (tc:IsX(), tc:GetAttack()>=n, not tc:IsX()), or None."""
        if isinstance(x, (A.GreaterOrEqThanOp, A.LessOrEqThanOp, A.GreaterThanOp, A.LessThanOp, A.EqToOp)):
            x = x.left
        if isinstance(x, A.ULNotOp):
            x = x.operand
        if isinstance(x, A.Invoke) and isinstance(x.source, A.Name) and isinstance(env.get(x.source.id), Reg) and x.func.id not in ("GetCount", "FilterCount", "IsExists"):
            return x.source.id
        return None

    def check(self, t, env, prog, cond=False):
        refine = collections.defaultdict(list)
        for x in self.conj(t):
            if isinstance(x, A.TrueExpr):
                continue
            v = self.term_var(x, env)
            if v is not None:
                refine[v].append(x)
                continue
            self.check_term(x, env, prog, cond)
        for name, xs in refine.items():
            reg = env[name]
            fl = Filt()
            for x in xs:
                self.filter_term(fl, x, env, cvar=name)
            r = prog.reg()
            prog.emit(self.vfilter(r, fl.build(reg.me, reg.op), within=reg.r, tgt=fl.tgt), C.REQ(r, 1))
            env[name] = Reg(r, reg.me, reg.op, reg.one, reg.handler, reg.target)
            if reg is self.egreg:
                self.egreg = env[name]

    def handler_call(self, x, env):
        """e:GetHandler():M(args) ou c:M(args) (c = handler) -> (M, args) otherwise None."""
        if isinstance(x, A.Invoke):
            src = x.source
            if isinstance(src, A.Invoke) and src.func.id == "GetHandler" and self.is_name(src.source, env, Eff):
                return x.func.id, x.args
            if self.is_name(src, env, Handler) or (isinstance(src, A.Name) and
                    isinstance(env.get(src.id), Reg) and env[src.id].handler):
                return x.func.id, x.args
        return None

    def count_check(self, value, comparator, n, prog, negate=False):
        need(value.mul == 1 and type(n) is int, "weighted count comparison")
        if negate:
            comparator = {A.EqToOp:A.NotEqToOp, A.NotEqToOp:A.EqToOp,
                          A.GreaterThanOp:A.LessOrEqThanOp, A.GreaterOrEqThanOp:A.LessThanOp,
                          A.LessThanOp:A.GreaterOrEqThanOp, A.LessOrEqThanOp:A.GreaterThanOp}[comparator]
        reg = self.as_reg(value.reg, prog)
        if comparator == A.NotEqToOp:
            need(n == 0, "count different from a non-zero constant")
            prog.emit(C.REQ(reg.r, 1))
            return
        if comparator in (A.EqToOp, A.GreaterThanOp, A.GreaterOrEqThanOp) and (n > 0 or comparator != A.EqToOp):
            prog.emit(C.REQ(reg.r, n + int(comparator == A.GreaterThanOp)))
        if comparator in (A.EqToOp, A.LessThanOp, A.LessOrEqThanOp):
            prog.emit(C.REQ(reg.r, n - int(comparator == A.LessThanOp), maximum=True))

    def check_term(self, x, env, prog, cond):
        # ft>-1 / GetLocationCount(..)>=0 : always true (the count is never negative)
        if isinstance(x, (A.GreaterThanOp, A.GreaterOrEqThanOp)) and isinstance(x.right, (A.Number, A.UMinusOp)) and (
                (isinstance(x.left, A.Name) and isinstance(env.get(x.left.id), FreeCount)) or
                (isinstance(x.left, A.Call) and dotted(x.left.func) == "Duel.GetLocationCount")):
            bound = self.fexpr(x.right, env)
            if bound == (-1 if isinstance(x, A.GreaterThanOp) else 0):
                return
        # Duel.GetMZoneCount(tp, e:GetHandler()) > 0 : free zone, or the card itself frees one
        if isinstance(x, (A.GreaterThanOp, A.GreaterOrEqThanOp)) and isinstance(x.left, A.Call) and \
                dotted(x.left.func) == "Duel.GetMZoneCount" and len(x.left.args) == 2:
            player, card = x.left.args
            handler = (isinstance(card, A.Invoke) and card.func.id == "GetHandler" and self.is_name(card.source, env, Eff)) \
                or self.is_name(card, env, Handler)
            need(handler and isinstance(env.get(getattr(player, "id", None)), Player) and env[player.id].p == ME
                 and self.fexpr(x.right, env) == (0 if isinstance(x, A.GreaterThanOp) else 1),
                 "GetMZoneCount: card or threshold")
            fl = Filt()
            fl.set("zx", core.ZX_FREE)
            r = prog.reg()
            prog.emit(C.SELF(r), self.vfilter(r, fl.build(ALLZ, ALLZ), within=r), C.REQ(r, 1))
            return
        hc = self.handler_call(x, env)
        if hc is not None:
            m, args = hc
            if m in ("IsAbleToGraveAsCost", "IsDiscardable", "IsAbleToRemoveAsCost", "IsReleasable", "IsAbleToHandAsCost",
                     "IsAbleToDeckAsCost", "IsAbleToGrave", "IsAbleToHand", "IsAbleToRemove", "IsDestructable",
                     "IsRelateToEffect"):
                return
            if m == "IsReason":
                need(len(args) == 1, "IsReason: arguments")
                mask = self.cnum(args[0], env)
                need(0 < mask <= 0x1fffffff, "IsReason: mask")
                tracked = (1 << core.EV_TO_GRAVE) | (1 << core.EV_BATTLE_DESTROYED)
                need(self.on != 0 and (self.on & ~tracked) == 0,
                     "IsReason: movement context not represented")
                prog.emit(C.COND_REASON(mask))
                return
            if m == "IsPreviousLocation":
                prog.emit(C.COND_SELF(prev=self.cnum(args[0], env)))
                return
            if m == "IsLocation":
                prog.emit(C.COND_SELF(zone=self.cnum(args[0], env)))
                return
            if m in SUMMON_METHODS or m == "IsSummonType":
                need(len(args) == (1 if m == "IsSummonType" else 0), "summon type: arguments")
                typ = SUMMON_METHODS[m] if m in SUMMON_METHODS else self.cnum(args[0],env)
                need(typ & 0x80ff0000 == 0, "summon type: mask not represented")
                prog.emit(C.COND_SELF(st=core.summon_mask(typ)))
                return
            if m in FILTER_METHODS:  # property of the card itself: SELF intersected with the filter, at least one card
                fl = Filt()
                apply_method(fl, m, [self.expr(a, env, prog) for a in args])
                r = prog.reg()
                prog.emit(C.SELF(r), self.vfilter(r, fl.build(ALLZ, ALLZ), within=r), C.REQ(r, 1))
                return
            raise Unsupported(f"condition: c:{m}")
        if (isinstance(x, A.Invoke) and isinstance(x.source, A.Call)
                and dotted(x.source.func) == "Duel.GetAttacker" and x.func.id == "IsControler"):
            need(self.on == (1 << core.EV_BATTLE_DESTROYED) and len(x.args) == 1,
                 "attacker controller outside a battle destruction event")
            player = self.expr(x.args[0], env, prog)
            need(isinstance(player, Player), "unbound attacker controller")
            r = prog.reg()
            filt = C.F(me=ALLZ if player.p == ME else 0, op=ALLZ if player.p == OPP else 0)
            prog.emit(C.ATTACKER(r), self.vfilter(r, filt, within=r), C.REQ(r, 1))
            return
        # eg:IsExists(f, n, ex, ...) : at least n cards of the event pass the filter
        if isinstance(x, A.Invoke) and isinstance(x.source, A.Name) and isinstance(env.get(x.source.id), (EventGroup,Reg,Lazy)) and x.func.id == "IsExists":
            a = x.args
            eg = self.as_reg(env[x.source.id], prog)
            fl = Filt()
            self.apply_fn(fl, self.expr(a[0], env, None), [self.expr(v, env, None) for v in a[3:]], env)
            if not isinstance(a[2], A.Nil):
                need(isinstance(self.expr(a[2], env, None), Handler), "eg:IsExists: exclusion")
                fl.set("not_self", 1)
            r = prog.reg()
            prog.emit(self.vfilter(r, fl.build(ALLZ, ALLZ), within=eg.r), C.REQ(r, self.cnum(a[1], env)))
            return
        if isinstance(x, A.Call):
            d = dotted(x.func)
            a = x.args
            if d == "aux.SelectUnselectGroup":
                need(len(a) >= 7 and isinstance(self.expr(a[2], env, None), Player) and
                     self.expr(a[5], env, None) is None and self.cnum(a[6], env) == 0,
                     "SelectUnselectGroup: group constraint")
                group = self.as_reg(self.expr(a[0], env, prog), prog)
                mn, mx = self.cnum(a[3], env), self.cnum(a[4], env)
                need(1 <= mn <= mx <= 15, "SelectUnselectGroup: size")
                prog.emit(C.REQ(group.r, mn))
                return
            if d == "Duel.IsPlayerCanDraw":
                p = self.expr(a[0], env, prog)
                n = self.cnum(a[1], env) if len(a) > 1 else 1
                need(isinstance(p, Player), "IsPlayerCanDraw")
                r = prog.reg()
                prog.emit(self.vfilter(r, C.F(me=core.Z_DECK) if p.p == ME else C.F(op=core.Z_DECK)), C.REQ(r, n))
                return
            if d in ("Duel.IsExistingMatchingCard", "Duel.IsExistingTarget"):
                f, p, s, o, n, ex = a[0], a[1], a[2], a[3], a[4], a[5] if len(a) > 5 else None
                F_, tgt = self.mkfilter(f, a[6:], p, s, o, ex, env, tgt=d == "Duel.IsExistingTarget")
                self.pending.append(dict(F=F_, n=self.cnum(n, env), tgt=tgt, used=False))
                return
            if d == "Duel.IsTurnPlayer":
                p = self.expr(a[0], env, prog)
                need(isinstance(p, Player), "IsTurnPlayer")
                prog.emit(C.COND_TURN(p.p))
                return
            if d == "Duel.IsMainPhase":
                prog.emit(C.COND_PHASE(core.PH_MAIN1 | core.PH_MAIN2))
                return
            if d == "Duel.IsBattlePhase":
                prog.emit(C.COND_PHASE(core.PH_BATTLE))
                return
            if d == "Duel.CheckLPCost":
                return  # PAY_LP checks the LP itself
            if d == "Duel.CheckReleaseGroupCost":
                need(len(a) >= 3 and self.cnum(a[2], env) == 1, "CheckReleaseGroupCost")
                need(len(a) < 4 or self.expr(a[3], env, prog) in (False, None), "Tribute from the hand")
                F_, _ = self.mkfilter(a[1], a[6:], a[0], A.Name("LOCATION_MZONE"), A.Number(0),
                                      a[5] if len(a) > 5 else None, env)
                self.pending.append(dict(F=F_, n=1, tgt=False, used=False))
                return
            if d == "Duel.IsChainNegatable":
                return
        lc = self.loccount(x, env, False)
        if lc is not None:
            prog.emit(C.REQ_FREE(*lc))
            return
        if isinstance(x, (A.EqToOp, A.NotEqToOp, A.GreaterThanOp, A.GreaterOrEqThanOp, A.LessThanOp, A.LessOrEqThanOp)):
            if isinstance(x.right, A.Number):
                value = self.expr(x.left, env, prog)
                if isinstance(value, Count):
                    self.count_check(value, type(x), x.right.n, prog)
                    return
        # turn: Duel.GetTurnPlayer()==tp / ~=tp
        if isinstance(x, (A.EqToOp, A.NotEqToOp)):
            l, r = x.left, x.right
            if isinstance(l, A.Call) and dotted(l.func) == "Duel.GetTurnPlayer":
                p = self.expr(r, env, prog)
                need(isinstance(p, Player), "GetTurnPlayer")
                prog.emit(C.COND_TURN(p.p if isinstance(x, A.EqToOp) else 1 - p.p))
                return
            if isinstance(l, A.Name) and isinstance(env.get(l.id), Arg) and env[l.id].name in ("ep", "rp"):
                p = self.expr(r, env, prog)
                need(isinstance(p, Player), "ep")
                need(self.on != 0, "ep without event")
                prog.emit(C.COND_EVP(self.on, p.p if isinstance(x, A.EqToOp) else 1 - p.p))
                return
        # chain link card: re:GetHandler():IsOnField() / re:GetOwner():IsTrap() ...
        if isinstance(x, A.Invoke) and isinstance(x.source, A.Invoke) and x.source.func.id in ("GetHandler", "GetOwner") \
                and self.is_name(x.source.source, env, ChainEff):
            m = x.func.id
            if m in ("IsDestructable", "IsRelateToEffect", "IsAbleToRemove", "IsAbleToHand", "IsAbleToGrave",
                     "IsAbleToDeck", "IsCanBeEffectTarget"):
                return
            if m == "IsOnField":
                self.chain_field = True
                self.emit_chain(prog)
                return
            t = {"IsTrap": core.TYPE_TRAP, "IsSpell": core.TYPE_SPELL}.get(m)
            if t:
                self.chain_typ = t
                self.emit_chain(prog)
                return
            raise Unsupported(f"chain link card test: {m}")
        if isinstance(x, A.OrLoOp) and self.show(x).replace("\n", " ").replace("  ", " ") in (
                "(re:IsMonsterEffect() or (re:IsSpellTrapEffect() and not re:IsHasType(EFFECT_TYPE_ACTIVATE)))",
                "re:IsMonsterEffect() or (re:IsSpellTrapEffect() and not re:IsHasType(EFFECT_TYPE_ACTIVATE))",):
            self.chain_nact = True
            self.emit_chain(prog)
            return
        if isinstance(x, A.ULNotOp) and isinstance(x.operand, A.Invoke) and self.is_name(x.operand.source, env, ChainEff) \
                and x.operand.func.id == "IsHasType" and dotted(x.operand.args[0]) == "EFFECT_TYPE_ACTIVATE":
            self.chain_nact = True
            self.emit_chain(prog)
            return
        # chain link: re:IsSpellEffect() / re:IsHasType(EFFECT_TYPE_ACTIVATE) / re:IsActiveType(TYPE_X)
        if isinstance(x, A.Invoke) and self.is_name(x.source, env, ChainEff):
            m = x.func.id
            if m == "IsHasType" and dotted(x.args[0]) == "EFFECT_TYPE_ACTIVATE":
                self.chain_act = True
                self.emit_chain(prog)
                return
            if m in ("IsSpellEffect", "IsTrapEffect", "IsActiveType", "IsSpellTrapEffect"):
                t = {"IsSpellEffect": core.TYPE_SPELL, "IsTrapEffect": core.TYPE_TRAP,
                     "IsSpellTrapEffect": core.TYPE_SPELL | core.TYPE_TRAP}.get(m) or self.cnum(x.args[0], env)
                self.chain_typ = t
                self.emit_chain(prog)
                return
        raise Unsupported(f"test {self.show(x)}")

    def emit_chain(self, prog):
        if getattr(self, "chain_act", False) and getattr(self, "chain_typ", None):
            need(self.chain_typ in (core.TYPE_SPELL, core.TYPE_TRAP), "chain link: type")
            prog.emit(C.COND_CHAIN(self.chain_typ))
            self.chain_act = self.chain_typ = None
            self.chain_activate = True
        if self.chain_field and self.chain_nact:
            prog.emit(C.COND_CHAIN_FIELD())
            self.chain_field = self.chain_nact = False

    def cnum(self, node, env):
        v = self.expr(node, env, None)
        need(isinstance(v, int) and not isinstance(v, bool), f"number expected: {self.show(node)}")
        return v

    # ------------------------------------------------ filters
    def mkfilter(self, f, extra, p, s, o, ex, env, tgt=False):
        """Filter f(c, extra...) over locations s (player p) / o (their opponent); ex: excluded card."""
        pv = self.expr(p, env, None)
        need(isinstance(pv, Player), "filter: player")
        s, o = self.cnum(s, env), self.cnum(o, env)
        if pv.p == OPP:
            s, o = o, s
        fl = Filt()
        fl.tgt = tgt
        fv = self.expr(f, env, None)
        xargs = [self.expr(a, env, None) for a in extra]
        self.apply_fn(fl, fv, xargs, env)
        if ex is not None and not isinstance(ex, A.Nil):
            exv = self.expr(ex, env, None)
            need(isinstance(exv, Handler), "filter: exclusion other than the card")
            fl.set("not_self", 1)
        need(s & ~(ALLZ | core.Z_OVERLAY) == 0 and o & ~(ALLZ | core.Z_OVERLAY) == 0, "symbolic zones (FZONE, PZONE…)")
        return fl.build(s, o), fl.tgt

    def apply_fn(self, fl, fv, xargs, env, two=False):
        if fv is None:
            return
        need(isinstance(fv, Fn), "filter: not a function")
        if fv.node is None:
            n = fv.name
            if n == "aux.TRUE":
                return
            if n.startswith("Card."):
                apply_method(fl, n[5:], list(fv.bound) + list(xargs))
                return
            if n in ("aux.FaceupFilter", "aux.FilterBoolFunction", "aux.FilterBoolFunctionEx", "aux.FaceupFilterEx",
                     "aux.TargetBoolFunction"):
                if n.startswith("aux.Faceup"):
                    fl.set("pos", core.P_FACEUP)
                inner = fv.bound[0]
                need(isinstance(inner, Fn), "compound filter")
                # FilterBoolFunction / FaceupFilter / TargetBoolFunction ignore the call arguments (utility.lua)
                self.apply_fn(fl, Fn(inner.node, inner.name, list(fv.bound[1:])), xargs if n.endswith("Ex") else [], env)
                return
            raise Unsupported(f"filter {n}")
        node = fv.node
        need(len(node.args) >= 1, "filter without parameter")
        args = list(node.args)
        fenv = {}
        if two:  # SetTarget of a continuous effect: function(e, c)
            need(len(args) >= 2, "continuous effect target")
            fenv[args[0].id] = Eff()
            args = args[1:]
        fenv[args[0].id] = "CARD"
        vals = list(fv.bound) + list(xargs)
        for i, a in enumerate(args[1:]):
            need(isinstance(a, A.Name), "filter: parameters")
            fenv[a.id] = vals[i] if i < len(vals) else None
        body = node.body.body
        need(len(body) == 1 and isinstance(body[0], A.Return) and len(body[0].values) == 1, "filter: body")
        for x in self.conj(body[0].values[0]):
            self.filter_term(fl, x, fenv, cvar=args[0].id)

    def mzone_free(self, x, env, cvar):
        """Duel.GetMZoneCount(tp, c[, tp]) > 0 (or >= 1) on the filter's card."""
        if not (isinstance(x, (A.GreaterThanOp, A.GreaterOrEqThanOp)) and isinstance(x.left, A.Call)
                and dotted(x.left.func) == "Duel.GetMZoneCount"):
            return False
        args = x.left.args
        me = lambda a: isinstance(self.fexpr(a, env), Player) and self.fexpr(a, env).p == ME  # noqa: E731
        need(len(args) in (2, 3) and me(args[0]) and (len(args) == 2 or me(args[2]))
             and isinstance(args[1], A.Name) and args[1].id == cvar
             and self.fexpr(x.right, env) == (0 if isinstance(x, A.GreaterThanOp) else 1),
             "filter: GetMZoneCount: options")
        return True

    def filter_term(self, fl, x, env, cvar):
        if isinstance(x, A.Invoke) and isinstance(x.source, A.Name) and x.source.id == cvar:
            args = [self.fexpr(a, env) for a in x.args]
            apply_method(fl, x.func.id, args)
            return
        if isinstance(x, A.Call):
            d = dotted(x.func)
            if d and d.startswith("s.") and x.args and isinstance(x.args[0], A.Name) and x.args[0].id == cvar:
                sub = self.sc.fns.get(d[2:])
                need(sub is not None, f"filter: {d}")
                self.apply_fn(fl, Fn(sub, d), [self.fexpr(a, env) for a in x.args[1:]], env)
                return
        # Duel.GetMZoneCount(tp, c) > 0, and its expanded form ft>0 or (c:IsControler(tp) and c:GetSequence()<5)
        if self.mzone_free(x, env, cvar):
            fl.set("zx", fl.d["zx"] | core.ZX_FREE)
            return
        if isinstance(x, A.OrLoOp) and (self.mzone_free(x.left, env, cvar) or
                isinstance(x.left, A.GreaterThanOp) and isinstance(x.left.left, A.Name)
                and isinstance(env.get(x.left.left.id), FreeCount) and self.fexpr(x.left.right, env) == 0
                and env[x.left.left.id].player.p == ME and env[x.left.left.id].zone == core.Z_MZONE):
            mine = self.conj(x.right)
            ctrl = [t for t in mine if isinstance(t, A.Invoke) and t.func.id == "IsControler" and isinstance(t.source, A.Name)
                    and t.source.id == cvar and len(t.args) == 1 and isinstance(self.fexpr(t.args[0], env), Player)
                    and self.fexpr(t.args[0], env).p == ME]
            main = [t for t in mine if isinstance(t, A.LessThanOp) and isinstance(t.left, A.Invoke) and t.left.func.id == "GetSequence"
                    and isinstance(t.left.source, A.Name) and t.left.source.id == cvar and self.fexpr(t.right, env) == 5]
            inmz = [t for t in mine if isinstance(t, A.Invoke) and t.func.id == "IsLocation" and isinstance(t.source, A.Name)
                    and t.source.id == cvar and len(t.args) == 1 and self.fexpr(t.args[0], env) == core.Z_MZONE]
            need(len(main) == 1 and len(ctrl) + len(main) + len(inmz) == len(mine), f"filter: or {self.show(x)}")
            if not inmz:
                self.notes.add("GetSequence()<5 read as a main Monster Zone cell")
            fl.set("zx", fl.d["zx"] | (core.ZX_FREE if ctrl else core.ZX_FREEANY))
            return
        # c:IsControler(tp) or c:IsFaceup() : mine, or face-up on the opponent's side
        if isinstance(x, A.OrLoOp):
            parts = self.disj(x)
            kinds = sorted("ctrl" if isinstance(t, A.Invoke) and t.func.id == "IsControler" and len(t.args) == 1
                           and isinstance(self.fexpr(t.args[0], env), Player) and self.fexpr(t.args[0], env).p == ME
                           else "fu" if isinstance(t, A.Invoke) and t.func.id == "IsFaceup" and not t.args else "?"
                           for t in parts if isinstance(t, A.Invoke) and isinstance(t.source, A.Name) and t.source.id == cvar)
            if kinds == ["ctrl", "fu"] and len(parts) == 2:
                fl.set("zx", fl.d["zx"] | core.ZX_OPFU)
                return
        if isinstance(x, A.OrLoOp):
            # A union of types or reasons reuses the "at least one bit" mask.
            axis, values = None, []
            for y in self.disj(x):
                g = Filt()
                self.filter_term(g, y, env, cvar)
                field = "typ" if g.d["typ"] > 0 else "reason" if g.d["reason"] > 0 else None
                need(field is not None and (axis is None or axis == field) and
                     all(g.d[k] == Filt().d[k] for k in g.d if k != field) and
                     g.loc == ALLZ and g.ctrl is None and not g.tgt,
                     f"filter: or {self.show(x)}")
                axis = field
                values.append(g.d[field])
            union = 0
            for value in values:
                union |= value
            fl.set(axis, union)
            return
        if isinstance(x, A.GreaterThanOp) and isinstance(x.left, A.Call) and dotted(x.left.func) == "Duel.GetLocationCountFromEx":
            args = x.left.args
            need(len(args) == 4 and self.fexpr(x.right, env) == 0,
                 "GetLocationCountFromEx: options / threshold")
            players = [self.fexpr(a, env) for a in args[:2]]
            need(all(isinstance(v, Player) and v.p == ME for v in players)
                 and isinstance(args[2], A.Nil) and isinstance(args[3], A.Name) and args[3].id == cvar,
                 "GetLocationCountFromEx: player / materials / card")
            fl.set("zx", fl.d["zx"] | core.ZX_EXTRA)
            return
        # Exclusion of the current card's code: c:GetCode() ~= positive id.
        if isinstance(x, A.NotEqToOp):
            l = x.left
            if isinstance(l, A.Invoke) and isinstance(l.source, A.Name) and l.source.id == cvar \
                    and l.func.id == "GetCode":
                need(not l.args, "filter: GetCode with arguments")
                v = self.fexpr(x.right, env)
                need(type(v) is int and 0 < v <= 0x7fffffff, "filter: GetCode code outside positive int32 or dynamic")
                fl.set("code", (-v, -v))
                return
        # Sequences 0..4: reuses the existing main-zones filter.
        if isinstance(x, A.LessThanOp) and isinstance(x.left, A.Invoke) \
                and isinstance(x.left.source, A.Name) and x.left.source.id == cvar \
                and x.left.func.id == "GetSequence" and not x.left.args and self.fexpr(x.right, env) == 5:
            fl.set("zx", fl.d["zx"] | core.ZX_MAIN)
            return
        # comparisons: c:GetAttack()>=n ...
        if isinstance(x, (A.GreaterOrEqThanOp, A.LessOrEqThanOp, A.GreaterThanOp, A.LessThanOp, A.EqToOp)):
            l = x.left
            if isinstance(l, A.Invoke) and isinstance(l.source, A.Name) and l.source.id == cvar \
                    and l.func.id in ("GetAttack", "GetDefense", "GetLevel"):
                v = self.fexpr(x.right, env)
                need(isinstance(v, int), "filter: non-constant comparison")
                k = {"GetAttack": "atk", "GetDefense": "df", "GetLevel": "lv"}[l.func.id]
                rng = {A.GreaterOrEqThanOp: (v, core.BIG), A.GreaterThanOp: (v + 1, core.BIG),
                       A.LessOrEqThanOp: (-core.BIG, v), A.LessThanOp: (-core.BIG, v - 1), A.EqToOp: (v, v)}[type(x)]
                fl.set(k, rng)
                return
        if isinstance(x, A.ULNotOp) and isinstance(x.operand, A.Invoke) and isinstance(x.operand.source, A.Name) \
                and x.operand.source.id == cvar:
            m = x.operand.func.id
            if m == "IsDisabled" and not x.operand.args:
                fl.set("nodis", 1)
                return
            if m == "IsStatus" and dotted(x.operand.args[0]) == "STATUS_ATTACK_CANCELED":
                self.notes.add("STATUS_ATTACK_CANCELED ignored")
                return
            if m == "IsImmuneToEffect":
                self.notes.add("IsImmuneToEffect ignored")
                return
            if m == "IsType" and len(x.operand.args) == 1:
                fl.set("not_typ", self.fexpr(x.operand.args[0], env))
                return
            if m == "IsCode" and len(x.operand.args) == 1:
                v = self.fexpr(x.operand.args[0], env)
                need(isinstance(v, int) and v > 0, "filter: non-constant not c:IsCode")
                fl.set("code", (-v, -v))
                return
            raise Unsupported(f"filter: not c:{m}")
        raise Unsupported(f"filter: {self.show(x)}")

    def disj(self, t):
        if isinstance(t, A.OrLoOp):
            return self.disj(t.left) + self.disj(t.right)
        return [t]

    def fexpr(self, node, env):
        """Expression in a filter: constants, bound parameters, e, tp."""
        if isinstance(node, A.Name):
            if node.id in env:
                v = env[node.id]
                need(v != "CARD", "filter: card as argument")
                return v
            if node.id in K:
                return K[node.id]
            if node.id == "id":
                return self.code
            raise Unsupported(f"filter: name {node.id}")
        if isinstance(node, A.UMinusOp):
            value = self.fexpr(node.operand, env)
            need(type(value) is int, "negation of a non-constant value")
            return -value
        if isinstance(node, A.BAndOp):
            left = self.fexpr(node.left, env)
            right = self.fexpr(node.right, env)
            need(type(left) is int and type(right) is int, "non-constant bitwise conjunction")
            return left & right
        if isinstance(node, A.Number):
            return node.n
        if isinstance(node, (A.TrueExpr, A.FalseExpr)):
            return isinstance(node, A.TrueExpr)
        if isinstance(node, A.Nil):
            return None
        if isinstance(node, (A.BOrOp, A.AddOp, A.SubOp)):
            a, b = self.fexpr(node.left, env), self.fexpr(node.right, env)
            if isinstance(node, A.SubOp) and isinstance(b, Player) and a == 1:
                return Player(1 - b.p)
            need(isinstance(a, int) and isinstance(b, int), "filter: operation")
            return a | b if isinstance(node, A.BOrOp) else a + b if isinstance(node, A.AddOp) else a - b
        raise Unsupported(f"filter: expression {self.show(node)}")

    # ------------------------------------------------ expressions
    def expr(self, node, env, prog, nret=1, statement=False):
        if isinstance(node, A.Name):
            if node.id in env:
                return env[node.id]
            if node.id in K:
                return K[node.id]
            if node.id == "id":
                return self.code
            raise Unsupported(f"name {node.id}")
        if isinstance(node, A.UMinusOp):
            value = self.expr(node.operand, env, prog)
            need(type(value) is int, "negation of a non-constant value")
            return -value
        if isinstance(node, A.BAndOp):
            left = self.expr(node.left, env, prog)
            right = self.expr(node.right, env, prog)
            need(type(left) is int and type(right) is int, "non-constant bitwise conjunction")
            return left & right
        if isinstance(node, A.Number):
            return node.n
        if isinstance(node, (A.TrueExpr, A.FalseExpr)):
            return isinstance(node, A.TrueExpr)
        if isinstance(node, A.Nil):
            return None
        if isinstance(node, A.SubOp):
            a, b = self.expr(node.left, env, prog), self.expr(node.right, env, prog)
            if a == 1 and isinstance(b, Player):
                return Player(1 - b.p)
            need(isinstance(a, int) and isinstance(b, int), "subtraction")
            return a - b
        if isinstance(node, (A.BOrOp, A.AddOp)):
            a, b = self.expr(node.left, env, prog), self.expr(node.right, env, prog)
            need(isinstance(a, int) and isinstance(b, int), "addition")
            return a | b if isinstance(node, A.BOrOp) else a + b
        if isinstance(node, A.MultOp):
            a, b = self.expr(node.left, env, prog), self.expr(node.right, env, prog)
            if isinstance(a, Count) and isinstance(b, int):
                return Count(a.reg, a.mul * b)
            need(isinstance(a, int) and isinstance(b, int), "multiplication")
            return a * b
        if isinstance(node, (A.FloatDivOp, A.FloorDivOp)):
            a, b = self.expr(node.left, env, prog), self.expr(node.right, env, prog)
            if isinstance(a, tuple) and a[0] == "stat" and b == 2 and a[1] in ("GetAttack", "GetBaseAttack"):
                return ("half", a[1], a[2])
            raise Unsupported("division")
        if isinstance(node, A.ULengthOP):
            v = self.expr(node.operand, env, prog)
            need(isinstance(v, (Reg, Lazy)), "#")
            return Count(v)
        d = dotted(node)
        if d and not isinstance(node, A.Call):
            if d.startswith("s."):
                n = d[2:]
                if n in self.sc.fns:
                    return Fn(self.sc.fns[n], n)
                if n in self.sc.vals:
                    return self.expr(self.sc.vals[n], env, prog)
                raise Unsupported(f"{d} unknown")
            return Fn(name=d)
        if isinstance(node, A.AnonymousFunction):
            return Fn(node, "anon")
        if isinstance(node, A.Call):
            return self.call(dotted(node.func), node.args, env, prog, nret)
        if isinstance(node, A.Invoke):
            return self.invoke(node, env, prog)
        raise Unsupported(f"expression {type(node).__name__}")

    def as_reg(self, v, prog):
        if isinstance(v, Reg):
            return v
        if isinstance(v, Lazy):
            r = prog.reg()
            prog.emit(self.vfilter(r, v.F))
            return Reg(r, v.F[0], v.F[1])
        if isinstance(v, Handler):
            if self.hreg is not None:
                return self.hreg
            r = prog.reg()
            prog.emit(C.SELF(r))
            return Reg(r, ALLZ, 0, True)
        if isinstance(v, EventGroup):
            if self.egreg is not None:
                return self.egreg
            r = prog.reg()
            if self.chaining:
                prog.emit(C.CHAINCARD(r, 0 if self.chain_activate else C.ONFIELD))
            else:
                need(self.on, "eg without event")
                prog.emit(C.EVENT(r, self.on, trig=self.field_trig))
            single_events = (1 << core.EV_SUMMON) | (1 << core.EV_FLIP_SUMMON)
            singleton = bool(self.on) and not (self.on & ~single_events) and not self.chaining
            self.egreg = Reg(r, one=singleton)
            return self.egreg
        if isinstance(v, Attacker):
            r = prog.reg()
            prog.emit(C.ATTACKER(r))
            return Reg(r, 0, core.Z_MZONE, True)
        raise Unsupported(f"group expected ({type(v).__name__})")

    def select(self, prog, F_, sp, mn, mx, tgt):
        """Selection: FILTER + CHOOSE; merges with an identical activation test (same filter)."""
        need(isinstance(sp, Player), "selection: player")
        need(isinstance(mn, int) and isinstance(mx, int) and 0 <= mn <= mx and 1 <= mx <= 15, f"selection of {mn} to {mx}")
        exact = False
        for chk in getattr(self, "pending", []):
            if not chk["used"] and chk["tgt"] == tgt and chk["n"] == mx and self.same_filter(chk["F"], F_):
                chk["used"] = True
                F_ = self.union_notself(chk["F"], F_)
                exact = True
                break
        r = prog.reg()
        r2 = prog.reg()
        upto = not exact and prog.kind != "resolve"
        # mn < mx: variable-size selection (the player stops whenever they want after mn cards). At least one card
        # (the oracle driver only offers the empty set when a cancellation is possible)
        lo = max(mn, 1)
        prog.emit(self.vfilter(r, F_, tgt=tgt), C.CHOOSE(r2, r, mx, who=sp.p, upto=upto or (mn == 0),
                                                     mn=lo if lo < mx else None, tgt=tgt,
                                                     public=self.public_targets if tgt else True))
        return r2

    def same_filter(self, a, b):
        return a[:4] == b[:4] and a[5:] == b[5:]

    def union_notself(self, a, b):
        return a[:4] + (a[4] | b[4],) + a[5:]

    def first_target(self, source, prog):
        if source.one:
            return source
        need(prog is not None and prog.kind == "resolve", "multiple first target outside resolution")
        # The register loaded at resolution excludes the moved cards. The
        # first element must be chosen BEFORE this filtering, in lk_reg.
        need(not any(a == source.r and (op == core.OP_SPSUMMON or op == core.OP_MOVE and c & 1)
                     for op, a, b, c in prog.ins), "first target after group mutation")
        r = prog.reg()
        prog.emit(C.FIRST(r, source.r, original=True))
        return Reg(r, source.me, source.op, one=True, target=True)

    def call(self, d, args, env, prog, nret):
        E_ = lambda i: self.expr(args[i], env, prog)  # noqa: E731
        if d in IGNORED_CALLS:
            return None
        if d == "Duel.SendtoDeck":
            need(not any(isinstance(ch, A.Call) and dotted(ch.func) == "Duel.ShuffleDeck"
                         for *_, ch, _ in _walk(self.sc.tree)),
                 "SendtoDeck: explicit shuffle not represented")
            need(nret == 0, "SendtoDeck: return value not represented")
            need(prog is not None and prog.kind in ("cost", "resolve"),
                 "SendtoDeck outside cost / resolution")
            need(len(args) == 4 and E_(1) is None and E_(2) == K.get("SEQ_DECKTOP"),
                 "SendtoDeck: player / sequence")
            reason = E_(3)
            need(type(reason) is int and 0 <= reason <= 0x1fffffff, "SendtoDeck: reason")
            source = E_(0)
            reg = self.as_reg(source, prog)
            if isinstance(source, Handler):
                need(self.hz is not None and not (self.hz[0] & (core.Z_DECK | core.Z_EXTRA))
                     and not (card_type(self.code) & core.TYPE_EXTRA),
                     "SendtoDeck: source / Extra Deck redirection")
            else:
                need(not ((reg.me | reg.op) & (core.Z_DECK | core.Z_EXTRA)),
                     "SendtoDeck: source / Extra Deck redirection")
            prog.emit(C.MOVE(reg.r, core.Z_DECK, reason=reason))
            for group in (reg, *(v for v in env.values() if isinstance(v, Reg) and v.r == reg.r)):
                group.me |= core.Z_DECK
                group.op |= core.Z_DECK
            return None
        if d == "Duel.GetChainInfo":
            out = []
            for a in args[1:]:
                k = dotted(a)
                if k == "CHAININFO_TARGET_PLAYER":
                    need("player" in self.tinfo, "GetChainInfo without SetTargetPlayer")
                    out.append(self.tinfo["player"])
                elif k == "CHAININFO_TARGET_PARAM":
                    need("param" in self.tinfo, "GetChainInfo without SetTargetParam")
                    out.append(self.tinfo["param"])
                elif k == "CHAININFO_TARGET_CARDS":
                    need(self.treg is not None, "unknown targets")
                    out.append(self.treg)
                else:
                    raise Unsupported(f"GetChainInfo {k}")
            return out if nret > 1 else out[0]
        if d == "Duel.SetTargetPlayer":
            self.tinfo["player"] = E_(0)
            return None
        if d == "Duel.SetTargetParam":
            self.tinfo["param"] = E_(0)
            return None
        if d == "Duel.GetFirstTarget" or d == "Duel.GetTargetCards":
            need(self.treg is not None, "unknown targets")
            need(d != "Duel.GetFirstTarget" or nret == 1, "several GetFirstTarget returns")
            if d == "Duel.GetFirstTarget":
                return self.first_target(self.treg, prog)
            need(prog is not None and prog.kind == "resolve", "GetTargetCards outside resolution")
            need(len(args) == 1 and isinstance(E_(0), Eff), "GetTargetCards: current effect expected")
            r = prog.reg()
            prog.emit(self.vcopy(r, self.treg.r, related=True))
            return Reg(r, self.treg.me, self.treg.op, one=self.treg.one, target=True)
        if d == "Duel.GetAttacker":
            r = prog.reg()
            prog.emit(C.ATTACKER(r))
            return Reg(r, core.Z_MZONE, core.Z_MZONE, True)
        if d == "Duel.SetTargetCard":
            need(prog is not None and prog.kind == "target", "SetTargetCard outside targeting")
            need(self.treg is None, "two target groups")
            self.treg = self.as_reg(E_(0), prog)
            need(self.treg.one, "multiple target group not represented")
            self.treg.target = True
            prog.emit(C.LABEL(self.treg.r, target=True, public=self.public_targets))
            return None
        if d == "Duel.GetMatchingGroup" or d == "Duel.GetFieldGroup" or d == "Duel.GetFieldGroupCount" \
                or d == "Duel.GetMatchingGroupCount":
            if d.startswith("Duel.GetField"):
                F_, _ = self.mkfilter(A.Nil(), [], args[0], args[1], args[2], None, env)
            else:
                F_, _ = self.mkfilter(args[0], args[5:], args[1], args[2], args[3], args[4] if len(args) > 4 else None, env)
            group = self.as_reg(Lazy(F_), prog) if prog is not None else Lazy(F_)
            return Count(group) if d.endswith("Count") else group
        if d == "Duel.SelectMatchingCard" or d == "Duel.SelectTarget":
            sp = E_(0)
            tgt = d == "Duel.SelectTarget"
            need(prog is not None and (not tgt or (prog.kind == "target" and not getattr(self, "in_proc", False))),
                 "SelectTarget outside the targeting of a chain")
            F_, _ = self.mkfilter(args[1], args[8:], args[2], args[3], args[4], args[7] if len(args) > 7 else None, env, tgt=tgt)
            mn, mx = self.cnum(args[5], env), self.cnum(args[6], env)
            r = self.select(prog, F_, sp, mn, mx, tgt)
            reg = Reg(r, F_[0], F_[1], one=(mx == 1), target=tgt)
            if tgt:
                need(self.treg is None, "two SelectTarget")
                self.treg = reg
            return reg
        if d == "Duel.DiscardHand":
            sp = E_(0)
            F_, _ = self.mkfilter(args[1], args[6:], args[0], A.Name("LOCATION_HAND"), A.Number(0),
                                  args[5] if len(args) > 5 else None, env)
            r = self.select(prog, F_, sp, self.cnum(args[2], env), self.cnum(args[3], env), False)
            reason = E_(4)
            need(type(reason) is int and 0 <= reason <= 0x1fffffff, "DiscardHand: reason")
            prog.emit(C.MOVE(r, reason=reason))
            return 1
        if d == "Duel.SelectReleaseGroupCost":
            need(self.cnum(args[2], env) == 1 and self.cnum(args[3], env) == 1, "SelectReleaseGroupCost")
            need(len(args) < 5 or self.expr(args[4], env, prog) in (False, None), "Tribute from the hand")
            self.notes.add("Tribute = send to the GY")
            F_, _ = self.mkfilter(args[1], args[7:], args[0], A.Name("LOCATION_MZONE"), A.Number(0),
                                  args[6] if len(args) > 6 else None, env)
            r = self.select(prog, F_, E_(0), 1, 1, False)
            return Reg(r, core.Z_MZONE, 0)
        if d in ("Duel.Destroy", "Duel.SendtoGrave", "Duel.SendtoHand", "Duel.Remove", "Duel.Release"):
            v = E_(0)
            reg = self.as_reg(v, prog)
            original = reg
            if d == "Duel.Destroy":
                need(len(args) == 2, "Destroy to another location")
                reason = E_(1)
                need(type(reason) is int and 0 <= reason <= 0x1fffffff, "Destroy: reason")
                r = prog.reg()
                prog.emit(self.vcopy(r, reg.r), C.DESTROY(r, reason=reason))
                reg = Reg(r, reg.me, reg.op, reg.one)
            elif d == "Duel.SendtoGrave" or d == "Duel.Release":
                need(len(args) == 2, f"{d}: player or options")
                reason = E_(1)
                need(type(reason) is int and 0 <= reason <= 0x1fffffff, f"{d}: reason")
                prog.emit(C.MOVE(reg.r, reason=reason | (core.REASON_RELEASE if d == "Duel.Release" else 0)))
            elif d == "Duel.SendtoHand":
                need(len(args) == 3 and E_(1) is None, "SendtoHand to a player or options")
                reason = E_(2)
                need(type(reason) is int and 0 <= reason <= 0x1fffffff, "SendtoHand: reason")
                prog.emit(C.MOVE(reg.r, core.Z_HAND, reason=reason))
            else:
                need(len(args) == 3 and E_(1) == core.P_FACEUP, "face-down banish or options")
                reason = E_(2)
                need(type(reason) is int and 0 <= reason <= 0x1fffffff, "Remove: reason")
                prog.emit(C.MOVE(reg.r, core.Z_REMOVED, reason=reason))
            destination = {"Duel.Destroy": core.Z_GRAVE, "Duel.SendtoGrave": core.Z_GRAVE,
                           "Duel.Release": core.Z_GRAVE, "Duel.SendtoHand": core.Z_HAND,
                           "Duel.Remove": core.Z_REMOVED}[d]
            for group in (original, *(v for v in env.values() if isinstance(v, Reg) and v.r == original.r)):
                group.me |= destination
                group.op |= destination
            return Count(reg)
        if d == "Duel.SpecialSummon":
            tag = generic_summon_tag(E_(1))
            reg = self.as_reg(E_(0), prog)
            sp, to = E_(2), E_(3)
            need(isinstance(sp, Player) and sp.p == ME and isinstance(to, Player) and to.p == ME,
                 "Special Summon for / on the opponent's side")
            need(E_(4) is False and E_(5) is False, "Special Summon nocheck / nolimit")
            pos = E_(6)
            need(pos in (core.P_FACEUP, core.P_FU_ATK, core.P_FU_DEF, core.P_FU_DEF | core.P_FD_DEF), "position")
            need(len(args) <= 7, "Special Summon in chosen zones")
            r = prog.reg()
            prog.emit(self.vcopy(r, reg.r), C.SPSUMMON(r, pos, tag=tag))
            for group in (reg, *(v for v in env.values() if isinstance(v, Reg) and v.r == reg.r)):
                group.me |= core.Z_MZONE
                group.op |= core.Z_MZONE
            return Count(Reg(r, reg.me, reg.op, reg.one))
        if d in ("Duel.Draw", "Duel.Damage", "Duel.Recover"):
            p, n = E_(0), E_(1)
            need(isinstance(p, Player), f"{d}: player")
            if d == "Duel.Draw":
                need(isinstance(n, int), "variable draw")
                prog.emit(C.DRAW(p.p, n))
            elif isinstance(n, Count):
                need(d == "Duel.Damage", "variable recovery")
                prog.emit(C.DAMAGE(p.p, n.mul, per=self.as_reg(n.reg, prog).r))
            else:
                need(isinstance(n, int), f"{d}: variable value")
                prog.emit(C.DAMAGE(p.p, n) if d == "Duel.Damage" else C.RECOVER(p.p, n))
            return n
        if d == "Duel.NegateRelatedChain":
            # pending chain links of the card are negated: nrc bit of the following MODIFY MC_DISABLE on the same card
            need(len(args) == 2 and dotted(args[1]) == "RESET_TURN_SET", "NegateRelatedChain: reset")
            obj = E_(0)
            need(isinstance(obj, Reg) and not obj.handler and self.nrc is None, "NegateRelatedChain: card")
            self.nrc = obj.r
            return None
        if d == "Duel.PayLPCost":
            p, n = E_(0), E_(1)
            need(isinstance(p, Player) and p.p == ME and isinstance(n, int), "PayLPCost")
            prog.emit(C.PAY_LP(n))
            return None
        if d == "Duel.ShuffleHand":
            p = E_(0)
            need(isinstance(p, Player), "ShuffleHand")
            prog.emit(C.SHUFFLE_HAND(p.p))
            return None
        if d in ("Duel.GetCustomActivityCount", "Duel.GetActivityCount"):
            custom = d == "Duel.GetCustomActivityCount"
            args_ = args[1:] if custom else args
            need(len(args_) == 2 and dotted(args_[1]) == "ACTIVITY_SPSUMMON", f"{d}: activity")
            who = self.expr(args_[0], env, prog)
            need(isinstance(who, Player) and who.p == ME and prog is not None, f"{d}: player")
            if custom:
                key = self.expr(args[0], env, prog)
                need(key in self.counters, "unknown activity counter")
                bit, allowed = self.counters[key]
            else:
                bit, allowed = core.ST_SST, None
            r = prog.reg()
            zones = ALLZ | core.Z_OVERLAY
            if allowed is not None:
                prog.emit(self.vfilter(r, allowed, invert=True))
            prog.emit(self.vfilter(r, C.F(me=zones, op=zones, st=bit), within=r if allowed is not None else None))
            return Count(Reg(r, zones, zones))
        if d == "aux.addTempLizardCheck":
            self.notes.add("Clock Lizard ignored")
            return None
        if d == "Duel.RegisterEffect":
            ne, p = E_(0), E_(1)
            need(isinstance(ne, NewEff) and isinstance(p, Player) and p.p == ME, "Duel.RegisterEffect")
            self.player_effect(ne, prog)
            return None
        if d == "Duel.NegateActivation":
            prog.emit(C.NEGATE())
            return True
        if d == "Effect.CreateEffect":
            return NewEff()
        if d in BOUND_FNS:
            return Fn(name=d, bound=[self.expr(a, env, prog) for a in args])
        if d == "aux.SelectUnselectGroup":
            need(getattr(self, "in_proc", False) and len(args) >= 7, "SelectUnselectGroup outside a procedure")
            need(E_(5) is None and E_(6) == 1, "SelectUnselectGroup: group constraint")
            need(all(E_(i) is None for i in (9,10) if i < len(args)),
                 "SelectUnselectGroup: conditional end / interruption")
            player, mn, mx = E_(7) if len(args) > 7 and E_(7) is not None else E_(2), E_(3), E_(4)
            need(isinstance(player, Player) and all(type(x) is int for x in (mn,mx)) and 1 <= mn <= mx <= 15,
                 "SelectUnselectGroup: size / player")
            group = self.as_reg(E_(0), prog)
            r = prog.reg()
            prog.emit(C.CHOOSE(r,group.r,mx,who=player.p,mn=mn if mn<mx else None))
            return Reg(r,group.me,group.op,mx==1)
        if d == "Duel.GetLocationCount":
            need(len(args) == 2, "GetLocationCount: options")
            player, zone = E_(0), E_(1)
            need(isinstance(player, Player) and zone in (core.Z_MZONE, core.Z_SZONE), "GetLocationCount: player / location")
            return FreeCount(player, zone)
        raise Unsupported(f"call {d}")

    def invoke(self, node, env, prog):
        m = node.func.id
        src = node.source
        if isinstance(src, A.Name) and src.id in env:
            obj = env[src.id]
        else:
            obj = self.expr(src, env, prog)
        args = node.args
        if getattr(self, "in_proc", False):
            if isinstance(obj, Eff) and m == "SetLabelObject":
                need(len(args) == 1, "SetLabelObject: arguments")
                self.label_object = self.as_reg(self.expr(args[0],env,prog),prog)
                return None
            if isinstance(obj, Eff) and m == "GetLabelObject":
                need(self.label_object is not None and not args, "GetLabelObject without a kept group")
                return self.label_object
            if isinstance(obj, (Reg,Lazy)) and m in ("KeepAlive","DeleteGroup"):
                need(not args, "group duration: arguments")
                return None
        if isinstance(obj, (Reg, Lazy, EventGroup)) and m in ("Filter", "FilterSelect", "FilterCount"):
            selecting = m == "FilterSelect"
            offset = int(selecting)
            need(len(args) >= (5 if selecting else 2), "Group.Filter: arguments")
            source = self.as_reg(obj, prog)
            fl = Filt()
            fv = self.expr(args[offset], env, None)
            ex_index = 4 if selecting else 1
            extra = [self.expr(a, env, None) for a in args[ex_index+1:]]
            related = isinstance(fv, Fn) and fv.node is None and fv.name == 'Card.IsRelateToEffect'
            if related:
                relation_args = list(fv.bound) + extra
                need(prog is not None and prog.kind == 'resolve' and source.target,
                     'Group.Filter: relation on a target group expected')
                need(len(relation_args) == 1 and isinstance(relation_args[0], Eff),
                     'Group.Filter: relation to another effect')
            else:
                self.apply_fn(fl, fv, extra, env)
            excluded = self.expr(args[ex_index], env, None)
            need(excluded is None or isinstance(excluded, Handler), "Group.Filter: exclusion of another card")
            if excluded is not None:
                fl.set("not_self", 1)
            r = prog.reg()
            # A group keeps its references, even if a card has changed location.
            prog.emit(self.vfilter(r, fl.build(ALLZ | core.Z_OVERLAY, ALLZ | core.Z_OVERLAY),
                                  within=source.r, tgt=fl.tgt, related=related))
            result = Reg(r, source.me & fl.loc, source.op & fl.loc, target=source.target)
            if not selecting:
                return Count(result) if m == "FilterCount" else result
            player, mn, mx = [self.expr(args[i], env, prog) for i in (0,2,3)]
            need(isinstance(player, Player) and type(mn) is int and type(mx) is int and 1 <= mn <= mx <= 15,
                 "Group.FilterSelect: size / player")
            selected = prog.reg()
            prog.emit(C.CHOOSE(selected, r, mx, who=player.p, mn=mn if mn<mx else None))
            return Reg(selected, result.me, result.op, mx==1, target=result.target)
        if isinstance(obj, (Reg,Lazy)) and m == "Select":
            values=[self.expr(a,env,prog) for a in args]
            need(len(values) >= 3 and (len(values)==3 or values[3] is None), "Group.Select: exclusion")
            player,mn,mx=values[:3]
            need(isinstance(player,Player) and type(mn) is int and type(mx) is int and 1 <= mn <= mx <= 15,
                 "Group.Select: size / player")
            source=self.as_reg(obj,prog);r=prog.reg()
            prog.emit(C.CHOOSE(r,source.r,mx,who=player.p,mn=mn if mn<mx else None))
            return Reg(r,source.me,source.op,mx==1)
        if isinstance(obj, Eff) and m == "GetHandler":
            return Handler()
        if isinstance(obj, Eff) and m == "GetHandlerPlayer":
            return Player(ME)
        if (obj == "SELFCARD" or isinstance(obj, Handler)) and m == "GetControler":
            return Player(ME)
        if isinstance(obj, ChainEff) and m == "GetHandler":
            return ChainEff(card=True)
        if isinstance(obj, (Reg, Lazy, EventGroup)) and m == "GetFirst":
            if obj is self.treg:
                return self.first_target(obj, prog)
            if isinstance(obj, Reg) and obj.one:
                return obj
            need(prog is not None, "GetFirst without representable program")
            source = self.as_reg(obj, prog)
            need(source.one, "GetFirst on a potentially multiple group not represented")
            return source
        if isinstance(obj, (Reg, Lazy, EventGroup)) and m == "Clone":
            need(not args, "Group.Clone: arguments")
            source = self.as_reg(obj, prog)
            r = prog.reg()
            prog.emit(self.vfilter(r, C.F(me=ALLZ | core.Z_OVERLAY, op=ALLZ | core.Z_OVERLAY), within=source.r))
            return Reg(r, source.me, source.op, source.one)
        if isinstance(obj, (Reg, Lazy, EventGroup)) and m == "GetCount":
            return Count(obj)
        if isinstance(obj, (Reg, Lazy)) and m == "Iter":
            return self.as_reg(obj, prog)
        if isinstance(obj, Reg) and m in ("GetAttack", "GetBaseAttack", "GetDefense", "GetBaseDefense"):
            return ("stat", m, obj)
        if isinstance(obj, Reg) and m == "GetCode":
            return ("code", obj)
        if isinstance(obj, NewEff):
            if m in ("SetDescription", "SetProperty", "SetCategory"):
                if m == "SetProperty":
                    obj.f[m] = [self.expr(args[0], env, prog)]
                return None
            obj.f[m] = [self.expr(a, env, prog) for a in args]
            return None
        if m == "RegisterEffect" and args:
            ne = self.expr(args[0], env, prog)
            need(isinstance(ne, NewEff), "RegisterEffect")
            need(isinstance(obj, (Reg, Handler)), "RegisterEffect on something other than a card")
            self.modifier(obj, ne, prog)
            return None
        raise Unsupported(f"method: {m} on {type(obj).__name__}")

    def player_effect(self, ne, prog):
        """Player effect set by a resolution: only "cannot activate cards with this name this turn"
        (Sangan, Witch...) -> LOCK."""
        f = ne.f
        if f.get("SetCode", [0])[0] == K["EFFECT_CANNOT_SPECIAL_SUMMON"]:
            return self.ss_lock(ne, prog)
        need(f.get("SetType", [0])[0] == K["EFFECT_TYPE_FIELD"] and f.get("SetCode", [0])[0] == K["EFFECT_CANNOT_ACTIVATE"],
             "player effect" + (f" {f.get('SetCode')}" if os.environ.get("TR_DIAG") else ""))
        need(f.get("SetTargetRange") == [1, 0], "player effect: players")
        need(f.get("SetReset", [0])[0] == K["RESET_PHASE"] | K["PHASE_END"], "player effect: duration")
        lab = f.get("SetLabel", [None])[0]
        need(isinstance(lab, tuple) and lab[0] == "code", "player effect: label")
        v = f.get("SetValue", [None])[0]
        need(isinstance(v, Fn) and v.node is not None and len(v.node.body.body) == 1, "player effect: value")
        need(self.show(v.node.body.body[0]).replace("\n", " ") == "return re:GetHandler():IsCode(e:GetLabel())",
             "player effect: value")
        prog.emit(C.LOCK(lab[1].r))

    def ss_lock(self, ne, prog):
        """"This turn, you cannot Special Summon monsters, except ...": player EFFECT_CANNOT_SPECIAL_SUMMON
        until the end of the turn, target = filter of the forbidden cards -> SSLOCK."""
        f = ne.f
        need(f.get("SetType", [0])[0] == K["EFFECT_TYPE_FIELD"], "summon prohibition: type")
        pr = f.get("SetProperty", [0])[0]
        ok = K["EFFECT_FLAG_PLAYER_TARGET"] | K["EFFECT_FLAG_CLIENT_HINT"] | K["EFFECT_FLAG_OATH"]
        need(pr & K["EFFECT_FLAG_PLAYER_TARGET"] and pr & ~ok == 0, f"summon prohibition: property {pr:#x}")
        # OATH set at cost: lifted if the activation is negated (not modeled)
        need(prog.kind != "cost" or not pr & K["EFFECT_FLAG_OATH"], "OATH summon prohibition at cost")
        need(f.get("SetReset", [0])[0] == K["RESET_PHASE"] | K["PHASE_END"], "summon prohibition: duration")
        tr = f.get("SetTargetRange")
        need(tr in ([1, 0], [0, 1], [1, 1]), "summon prohibition: players")
        need(not set(f) - {"SetType", "SetProperty", "SetCode", "SetReset", "SetTargetRange", "SetTarget",
                           "SetDescription"}, "summon prohibition: fields " + ",".join(sorted(f)))
        fl, unless = Filt(), None
        fv = f.get("SetTarget", [None])[0]
        if fv is not None:
            need(isinstance(fv, Fn), "summon prohibition: target")
            node = fv.node
            if node is None:
                self.apply_fn(fl, fv, [], {}, two=True)
            else:
                # target = positive terms and not (exception): the exception becomes a separate
                # filter (and not exception)
                need(len(node.args) >= 2 and len(node.body.body) == 1 and isinstance(node.body.body[0], A.Return),
                     "summon prohibition: body")
                cvar = node.args[1].id
                terms = self.conj(node.body.body[0].values[0])
                neg = [t for t in terms if isinstance(t, A.ULNotOp)]
                need(len(neg) <= 1, "summon prohibition: two exceptions")
                for t in terms:
                    if t not in neg:
                        self.filter_term(fl, t, {cvar: "CARD"}, cvar=cvar)
                if neg:
                    unless = self.static_filter(self.conj(neg[0].operand), cvar, "summon prohibition: exception")
        need(fl.ctrl is None and not fl.tgt, "summon prohibition: filter")
        flt = fl.build(ALLZ, 0)
        for who in (ME, OPP):
            if tr[who]:
                prog.emit(C.SSLOCK(flt, who, unless))

    def modifier(self, obj, ne, prog):
        f = ne.f
        pr = f.get("SetProperty", [0])[0]
        need(pr & ~(K["EFFECT_FLAG_CANNOT_DISABLE"] | K["EFFECT_FLAG_CLIENT_HINT"]) == 0, f"modifier: property {pr:#x}")
        need(f.get("SetType", [0])[0] == K["EFFECT_TYPE_SINGLE"], "non-SINGLE modifier")
        code = f.get("SetCode", [0])[0]
        need(code in MODC, f"modifier {self.kname(code, 'EFFECT_')}")
        mc = MODC[code]
        reset = f.get("SetReset", [0])
        need(len(reset) <= 2 and (len(reset) == 1 or type(reset[1]) is int and reset[1] in (0, 1)),
             "modifier: reset count not represented")
        rs = reset[0]
        std = K["RESET_EVENT"] | K["RESETS_STANDARD"]
        if rs & ~K["RESET_DISABLE"] == std | K["RESET_PHASE"] | K["PHASE_END"]:
            dur = core.TD_TURN
        elif rs & ~K["RESET_DISABLE"] == std:
            dur = core.TD_PERM
        else:
            raise Unsupported(f"modifier: reset {rs:#x}")
        self_ = isinstance(obj, Handler) or (isinstance(obj, Reg) and obj.handler)
        rd = bool(rs & K["RESET_DISABLE"])
        need(not rd or self_, "RESET_DISABLE on another card")
        val = f.get("SetValue", [0])[0]
        if mc == core.MC_DISABLE:
            need(val in (0, None), "DISABLE with value")
            val = 0
        mode = core.MV_CONST
        if isinstance(val, tuple) and val[0] == "half":
            mode = core.MV_HALF_ATK if val[1] == "GetAttack" else core.MV_HALF_BASE_ATK
            need(val[2] is obj or val[2].r == getattr(obj, "r", None), "modifier: half of another card's value")
            need((mode == core.MV_HALF_ATK) == (mc == core.MC_FINAL_ATK) or mode == core.MV_HALF_BASE_ATK and
                 mc == core.MC_BASE_ATK, "modifier: half and code")
            val = 0
        need(isinstance(val, int), "modifier: variable value")
        if mc == core.MC_INDES_BATTLE:
            need(val == 1, "modifier: INDESTRUCTABLE_BATTLE value")
            val = 0  # ATK threshold 0: unconditional protection, including against a 0 ATK
        reg = self.as_reg(obj, prog)
        if isinstance(obj, Reg) and obj.handler:
            self_ = True
        elif self_:
            r = prog.reg()
            prog.emit(self.vfilter(r, C.F(me=core.Z_MZONE, pos=core.P_FACEUP), within=reg.r))
            reg = Reg(r)
        if mc == core.MC_DISABLE:
            # EFFECT_DISABLE is always followed by EFFECT_DISABLE_EFFECT: a single MODIFY
            if prog.ins and prog.ins[-1][0] == core.OP_MODIFY and prog.ins[-1][1] == reg.r and \
                    (prog.ins[-1][2] & 31) == core.MC_DISABLE:
                return
        nrc = mc == core.MC_DISABLE and self.nrc is not None and self.nrc == reg.r
        if nrc:
            self.nrc = None
        prog.emit(C.MODIFY(reg.r, mc, val, mode=mode, dur=dur, rd=rd, nrc=nrc))


# EFFECT_DISABLE_EFFECT : companion of EFFECT_DISABLE (same MODIFY)
MODC[K["EFFECT_DISABLE_EFFECT"]] = core.MC_DISABLE


def translate(code):
    t = Translator(code)
    effs = t.translate()
    return effs, t.notes, t.proc, t.descs


def all_codes():
    return sorted(int(os.path.basename(p)[1:-4]) for p in glob.glob(os.path.join(OFFICIAL, "c*.lua")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    codes = [int(x) for x in a.codes.split(",")] if a.codes else all_codes()
    ok, why = [], collections.Counter()
    ex = {}
    for c in codes:
        try:
            effs, notes, *_ = translate(c)
            ok.append(c)
            if a.show:
                print(c, effs, sorted(notes))
        except Unsupported as e:
            k = str(e)
            why[k] += 1
            ex.setdefault(k, c)
            if a.show:
                print(c, "REJECTED:", k)
        except Exception as e:  # noqa: BLE001
            k = f"BUG {type(e).__name__}: {e}"[:100]
            why[k] += 1
            ex.setdefault(k, c)
            if a.show:
                print(c, k)
    print(f"{len(ok)}/{len(codes)} translated")
    for k, v in why.most_common(a.top):
        print(f"{v:6d}  {k}  (e.g. {ex[k]})")


if __name__ == "__main__":
    main()
