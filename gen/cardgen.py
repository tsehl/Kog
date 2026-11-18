"""Procedural generator of cards and decks.

A generated card is first a specification (dict): a frame (card type, stats, summon recipe), then
0 to 2 effects described by template: activation timing (kind, events, once per turn), conditions, cost,
clauses. A clause is either a verb applied to a selection of cards (target at activation, all the cards
of a filter, choice at resolution, cards of the event, the card itself), or an action without cards (draw,
damage, LP gain, negate an activation). A second clause follows the first ("then") and applies only if
the first one has hit at least one card.

The specification is translated:
- into engine bricks (`bricks`, then `cards.assemble` in `to_bank`);
- into a Lua script, to check the generated cards against ygopro-core (gen/tolua.py).
The templates reuse the shapes of the real cards verified against the oracle (cards.CARDS); only their
combinations are new.

Power budget (`value`, in "cards of advantage"): rough, used only as bounds to discard absurd cards.
To be recalibrated on the self-play win rates.

All generated cards have codes >= CODE0: they never mix with the real cards.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

import cards  # noqa: E402
import core  # noqa: E402
from cards import (F, FILTER, SELF, REQ, REQ_FREE, CHOOSE, MOVE, DESTROY, SPSUMMON, DRAW,  # noqa: E402
                   DAMAGE, RECOVER, COND_CHAIN, COND_CHAIN_FIELD, CHAINCARD, NEGATE, PAY_LP, EVENT, COND_TURN,
                   COND_EVP, COND_SELF, MODIFY, COND_PHASE, LOCK, MINATK, act, ignition, trigger, quick, cont, ev, ritual_spell,
                   ME, OPP)
from core import (BIG, E, P, PT, NF, NSET, Z_DECK, Z_HAND, Z_MZONE, Z_SZONE, Z_GRAVE, Z_REMOVED, Z_EXTRA,  # noqa: E402
                  Z_OVERLAY, P_FU_ATK, P_FU_DEF, P_FD_ATK, P_FACEUP, TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL,
                  TYPE_QUICKPLAY, TYPE_CONTINUOUS, TYPE_COUNTER, TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ,
                  TYPE_LINK, TYPE_TUNER, TYPE_PENDULUM, TYPE_EXTRA, EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON, EV_ATTACK,
                  EV_TO_GRAVE, EV_BATTLE_DESTROYED, EV_FLIP, OPT_NONE, OPT_NAME, OPT_CARD, PH_MAIN1, PH_MAIN2,
                  PH_BATTLE, MC_ATK, MC_DEF, MC_DISABLE, MC_NO_ATTACK, MC_NO_BTARGET, MC_NO_ETARGET, MC_IMM_TRAP,
                  MC_NO_SS, MC_INDES_BATTLE, MC_P_NO_TRAP, MC_INDES_EFF, MC_NO_DIRECT, DMG_STEP, DMG_STAT,
                  MC_FINAL_ATK, MC_BASE_ATK, MV_HALF_BASE_ATK, MV_HALF_ATK, TD_PERM,
                  RIT_GREATER, RIT_EQUAL, ST_SUMMON_TURN, ST_FLIP_TURN, ST_LINKSUM, LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T,
                  LM_TR)

CODE0 = 600_000_000
TYPE_EFFECT, TYPE_FLIP = 0x20, 0x200000
ONFIELD = Z_MZONE | Z_SZONE
ATTRS = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20]          # EARTH, WATER, FIRE, WIND, LIGHT, DARK
RACES = [1 << i for i in range(20)]
SETS = [0x7A0 + i for i in range(8)]             # procedural archetypes
ARROWS = [LM_BL, LM_B, LM_BR, LM_L, LM_R, LM_TL, LM_T, LM_TR]
STAT_BUDGET = {1: 1200, 2: 1900, 3: 2500, 4: 3200, 5: 3700, 6: 4200, 7: 4700, 8: 5200, 9: 5500, 10: 5800,
               11: 6000, 12: 6200}                # ATK + DEF of a Normal Monster of this Level (~1000 per "card")
LV_W = np.array([5, 8, 10, 24, 8, 8, 8, 9, 4, 4, 1, 1], float)  # Levels of the main deck monsters
N_EFF_W = [0.45, 0.30, 0.17, 0.08]  # 1, 2, 3, 4 effects
CARD_CAP = 2.6                                    # maximum value of a card (Spell / Trap: net advantage + 1)


def _r100(x):
    return int(max(0, round(x / 100)) * 100)


class Gen:
    """Generator (numpy, host side): a seed -> a bank of procedural cards (`make_set`)."""

    def __init__(self, seed, revive=0, arch=0.0, skel=0.0):
        """revive: reinforcement of the resurrections with choice (0 = reference distribution, draws unchanged). Each
        unit adds an "ss" to the verbs of a clause; the source is then the GY, without self-summon.
        arch: share of the cards attached to an archetype (0 = reference distribution, draws unchanged). An
        archetype has a favourite attribute and type; its monsters carry its setcode, and the filters of its
        cards mostly aim at the archetype, its attribute or its type (support cards of a single deck)."""
        self.rng = np.random.default_rng(seed)
        # skel: share of the effect monsters, Spells and Traps drawn from the effect skeletons (gen/skeleton.py) instead
        # of the hand-written templates. 0 = reference distribution, draws unchanged (separate random stream).
        self.skel = skel
        self.srng = np.random.default_rng([seed, 7919])
        self.revive = revive
        self.arch_p = arch
        self.arch = None                               # archetype of the card being generated
        if arch:
            self.arch_attr = {a: self.pick(ATTRS) for a in SETS}
            self.arch_race = {a: self.pick(RACES) for a in SETS}

    # ---------------------------------------------------------------- basic draws
    def p(self, x):
        return self.rng.random() < x

    def pick(self, xs, w=None):
        w = None if w is None else np.asarray(w, float) / np.sum(w)
        return xs[self.rng.choice(len(xs), p=w)]

    def qual(self, kw, n=1, stats=True, lv=True):
        """Adds n qualifiers to the monster filter kw (ATK, attribute, type, Level, archetype)."""
        for _ in range(n):
            # archetype: only for filters on its own cards (or the materials of a recipe)
            A = self.arch if (kw.get("me", 0) and not kw.get("op", 0)) or not (kw.get("me", 0) | kw.get("op", 0)) else None
            q = self.pick(["atk_lo", "atk_hi", "attr", "race", "lv", "setc"], [2, 3, 3, 2, 2 if lv else 0, 6 if A else 1.5])
            if q == "atk_lo" and stats:
                kw["atk"] = (self.pick([1000, 1500, 1800, 2000, 2400]), BIG)
            elif q == "atk_hi" and stats:
                kw["atk"] = (-BIG, self.pick([500, 1000, 1500, 1800]))
            elif q == "attr":
                kw["attr"] = self.arch_attr[A] if A and self.p(0.7) else self.pick(ATTRS)
            elif q == "race":
                kw["race"] = self.arch_race[A] if A and self.p(0.7) else self.pick(RACES)
            elif q == "lv":
                kw["lv"] = self.pick([(-BIG, 4), (5, BIG), (-BIG, 3), (7, BIG)])
            elif q == "setc":
                kw["setc"] = A if A else self.pick(SETS)
        return kw

    def mon(self, nq=None, **kw):
        kw.setdefault("typ", TYPE_MONSTER)
        if kw["typ"] == TYPE_MONSTER and self.p(0.15):
            kw["not_typ"] = kw.get("not_typ", 0) | TYPE_EFFECT
        nq = self.pick([0, 1, 1, 2]) if nq is None else nq
        if (kw.get("me", 0) | kw.get("op", 0)) == Z_GRAVE and self.p(0.15):
            kw["reason"] = self.pick([core.REASON_EFFECT, core.REASON_DESTROY, core.REASON_BATTLE,
                                      core.REASON_COST, -(core.REASON_EFFECT | core.REASON_DESTROY)])
        kw = self.qual(kw, nq)
        if self.arch and kw.get("me", 0) and not kw.get("op", 0) and "setc" not in kw and self.p(0.7):
            kw["setc"] = self.arch                     # archetype support: "an X monster of your ..."
        return kw

    # ---------------------------------------------------------------- clauses
    def card_clause(self, ctx):
        """"Verb on a selection" clause. ctx: effect kind, event, location of the card at activation."""
        kind, evs, zone = ctx["kind"], ctx.get("on", 0), ctx.get("zone")
        opts = ["destroy", "destroy", "banish", "bounce", "tohand", "tohand", "ss", "atk", "disable", "tograve",
                "decktop", "indes_battle"]
        if self.arch:  # archetype card: more searches and Summons of its own monsters
            opts = opts + ["tohand", "tohand", "ss", "ss"]
        verb = self.pick(opts + ["ss_gy"] * self.revive) if self.revive else self.pick(opts)
        if verb == "ss_gy" and evs:
            verb = "ss"
        if evs:  # Trap / quick effect on an event: the verb applies to the cards of the event
            if self.p(0.75):
                verb = self.pick(["destroy", "destroy", "banish", "bounce", "atk", "disable"])
                f = dict(op=Z_MZONE, pos=P_FACEUP) if not evs & ev(EV_ATTACK) else dict(op=Z_MZONE)
                if verb == "disable":
                    f = dict(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_EFFECT, nodis=1)
                elif verb == "atk":
                    f = dict(op=Z_MZONE, pos=P_FACEUP)
                elif self.p(0.4):
                    f = self.mon(1, **f)
                return dict(verb=verb, sel="event", f=f, k=1, v=self.neg_atk() if verb == "atk" else 0)
        if verb in ("destroy", "banish", "bounce", "tograve", "decktop"):
            if verb == "decktop":
                # placing on top of the deck is deterministic: take the cards only from the field or the GY.
                where = self.pick(["opp_mon", "any_mon", "opp_gy"], [4, 2, 1.5])
            else:
                where = self.pick(["opp_mon", "opp_st", "any_mon", "opp_card", "opp_gy"],
                                  [4, 2, 2, 1, 1.5 if verb == "banish" else 0])
            f = {"opp_mon": lambda: self.mon(op=Z_MZONE), "any_mon": lambda: self.mon(me=Z_MZONE, op=Z_MZONE, not_self=1),
                 "opp_st": lambda: dict(op=Z_SZONE, typ=TYPE_SPELL | TYPE_TRAP),
                 "opp_card": lambda: dict(op=ONFIELD), "opp_gy": lambda: dict(op=Z_GRAVE)}[where]()
            if verb == "decktop":
                f["not_typ"] = f.get("not_typ", 0) | TYPE_EXTRA
            if verb == "bounce" and where in ("opp_gy", "opp_st"):
                f = self.mon(op=Z_MZONE)
            if where == "opp_mon" and verb != "tograve" and self.p(0.15):  # Fissure: the opponent monster with the lowest ATK
                fmin = self.mon(op=Z_MZONE, pos=P_FACEUP)
                if verb == "decktop":
                    fmin["not_typ"] = fmin.get("not_typ", 0) | TYPE_EXTRA
                return dict(verb=verb, sel="min", f=fmin, k=1, v=0)
            sel = "target" if self.p(0.6) else "all"
            if sel == "all" and where in ("opp_card", "any_mon") and verb != "destroy":
                sel = "target"
            k = 2 if sel == "target" and self.p(0.15) else 1
            clause = dict(verb=verb, sel=sel, f=f, k=k, v=0)
            if k == 2 and self.p(0.5):
                clause["first_original"] = True
            return clause
        if verb == "tohand":
            if self.p(0.6):  # search in the deck
                f = self.mon(self.pick([1, 1, 2]), me=Z_DECK) if self.p(0.7) else dict(
                    me=Z_DECK, typ=self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_SPELL | TYPE_TRAP]))
                return dict(verb="tohand", sel="pick", f=f, k=1, v=0, lock=self.p(0.3))  # lock: Sangan
            f = self.mon(me=Z_GRAVE)  # recovery from the GY
            return dict(verb="tohand", sel="target" if self.p(0.6) else "pick", f=f, k=1, v=0)
        if verb == "ss_gy":  # resurrection with choice (see revive)
            pos = self.pos()
            sp = 1 if pos & P_FU_ATK else 2
            base = self.pick([dict(me=Z_GRAVE), dict(me=Z_GRAVE, op=Z_GRAVE)], [3, 1])
            f = self.mon(self.pick([1, 1, 2]), spsum=sp, **base)
            return dict(verb="ss", sel="target" if self.p(0.7) else "pick", f=f, k=self.pick([1, 1, 1, 2, 3]), v=0, pos=pos)
        if verb == "ss":
            pos = self.pos()
            sp = 1 if pos & P_FU_ATK else 2  # in Defense Position only: never a Link (IsCanBeSpecialSummoned)
            if zone in (Z_GRAVE, Z_HAND) and self.p(0.5):  # the card Summons itself
                return dict(verb="ss", sel="self", f=dict(me=zone, spsum=sp), k=1, v=0, pos=pos)
            src = self.pick(["gy", "gy_any", "deck", "hand", "removed"], [3, 1, 2, 2, 1])
            base = {"gy": dict(me=Z_GRAVE), "gy_any": dict(me=Z_GRAVE, op=Z_GRAVE), "deck": dict(me=Z_DECK),
                    "hand": dict(me=Z_HAND, not_self=1), "removed": dict(me=Z_REMOVED)}[src]
            f = self.mon(self.pick([1, 1, 2]), spsum=sp, **base)
            if src in ("deck", "hand") and "lv" not in f and "atk" not in f:
                f["lv"] = (-BIG, 4)
            sel = "target" if src in ("gy", "gy_any", "removed") and self.p(0.7) else "pick"
            return dict(verb="ss", sel=sel, f=f, k=self.pick([1, 1, 1, 2, 3]), v=0, pos=pos)
        if verb == "atk":
            r = self.rng.random()
            if zone == Z_MZONE and r < 0.4:  # on itself; rd: erased if its effects are negated (Little Swordsman),
                rd = self.p(0.5)              # perm: as long as it stays on the field (Number 17)
                return dict(verb="atk", sel="self", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pos_atk(), rd=rd,
                            perm=rd and self.p(0.4))
            r2 = self.rng.random()
            if r2 < 0.15:  # Castle Walls
                return dict(verb="def", sel="target", f=dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP), k=1,
                            v=self.pos_atk())
            if r2 < 0.27:  # Shrink: base ATK / 2
                return dict(verb="halve_base", sel="target", f=self.mon(1, op=Z_MZONE, pos=P_FACEUP), k=1, v=0)
            if r2 < 0.35:  # Reverse Glasses: ATK / 2 (all the opponent monsters, or of the whole field)
                f = dict(op=Z_MZONE, pos=P_FACEUP) if self.p(0.6) else dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP)
                return dict(verb="halve", sel="all", f=f, k=1, v=0)
            if r < 0.6:
                if self.p(0.5):
                    return dict(verb="atk", sel="target", f=dict(me=Z_MZONE, op=Z_MZONE, pos=P_FACEUP), k=1,
                                v=self.pos_atk() if self.p(0.6) else self.neg_atk())
                return dict(verb="atk", sel="target", f=self.mon(1, op=Z_MZONE, pos=P_FACEUP), k=1, v=self.neg_atk())
            if self.p(0.6):
                return dict(verb="atk", sel="all", f=self.mon(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pos_atk())
            return dict(verb="atk", sel="all", f=self.mon(op=Z_MZONE, pos=P_FACEUP), k=1, v=self.neg_atk())
        if verb == "indes_battle":
            return dict(verb=verb, sel="target", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=0)
        if verb == "disable":
            return dict(verb="disable", sel="target", f=dict(op=Z_MZONE, pos=P_FACEUP, typ=TYPE_EFFECT, nodis=1),
                        k=1, v=0)
        # tograve from the deck (mill)
        return dict(verb="tograve", sel="pick", f=self.mon(me=Z_DECK), k=self.pick([1, 1, 2]), v=0)

    def pos(self):
        return self.pick([P_FU_ATK | P_FU_DEF, P_FU_ATK | P_FU_DEF, P_FU_DEF, P_FU_ATK])

    def pos_atk(self):
        return self.pick([300, 500, 500, 700, 800, 1000, 1500])

    def neg_atk(self):
        return -self.pick([300, 500, 700, 1000, 1500])

    def plain_clause(self):
        """Clause without a card selection."""
        verb = self.pick(["draw", "damage", "recover", "burn_per"], [4, 3, 1.5, 1])
        if verb == "draw":
            return dict(verb="draw", n=self.pick([1, 1, 1, 2]))
        if verb == "damage":
            return dict(verb="damage", v=self.pick([300, 500, 800, 1000, 1500]))
        if verb == "recover":
            return dict(verb="recover", v=self.pick([500, 1000, 1500, 2000]))
        f = self.pick([dict(op=Z_MZONE), dict(op=Z_GRAVE, typ=TYPE_MONSTER), dict(me=Z_GRAVE, typ=TYPE_MONSTER),
                       dict(op=ONFIELD)])
        return dict(verb="burn_per", f=f, v=self.pick([100, 200, 300, 500]))

    def then_clause(self):
        """Second clause ("then"): no target, resolved without a check at activation (except the draw)."""
        r = self.pick(["draw", "damage", "recover", "atk_all", "mill"], [3, 3, 1, 1.5, 1])
        if r == "draw":
            return dict(verb="draw", n=1)
        if r == "damage":
            return dict(verb="damage", v=self.pick([300, 500, 800]))
        if r == "recover":
            return dict(verb="recover", v=self.pick([500, 1000]))
        if r == "atk_all":
            return dict(verb="atk", sel="all", f=dict(me=Z_MZONE, pos=P_FACEUP), k=1, v=self.pick([300, 500]))
        return dict(verb="tograve", sel="pick", f=self.mon(1, me=Z_DECK), k=1, v=0)

    def clauses(self, ctx):
        c1 = self.card_clause(ctx) if self.p(0.75) else self.plain_clause()
        out = [c1]
        if c1["verb"] == "ss" and self.p(0.5):
            c1["tag"] = self.pick([1, 2, 3, 7, 15, 31, 63, 127, 255])
        if self.p(0.25) and c1["verb"] not in ("burn_per",):
            out.append(self.then_clause())
        return out

    # ---------------------------------------------------------------- costs and effects
    def cost(self, choices, w):
        c = self.pick(choices, w)
        if c is None:
            return None
        if c == "lp":
            return ("lp", self.pick([500, 800, 1000, 1500, 2000]))
        if c == "detach":
            return ("detach", 1)
        return (c,)

    def effect(self, kind, card):
        """An effect of kind `kind` for the card `card` (specification, without bricks)."""
        mon = card["ctype"] & TYPE_MONSTER
        e = dict(kind=kind, on=0, opt=OPT_NONE, optional=False, rng=0, dmg=0, conds=[], cost=None, clauses=[])
        if kind in ("act", "ignition", "quick") and self.p(0.12):
            e["related"] = True  # the effect still depends on its source at resolution
        if kind == "act":  # Spell / Trap
            ct = card["ctype"]
            if ct & TYPE_COUNTER:
                typ = self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_MONSTER, TYPE_SPELL | TYPE_TRAP])
                e["cost"] = self.cost([None, "lp", "discard"], [2, 2, 1])
                if typ == TYPE_MONSTER:
                    e["conds"] = [("chain_field",)]
                    e["clauses"] = [dict(verb="chain_destroy")]
                else:
                    e["conds"] = [("chain", typ)]
                    e["clauses"] = [dict(verb="negate")]
                return e
            if ct & TYPE_TRAP and self.p(0.35):
                if self.p(0.6):
                    e["on"] = self.pick([ev(EV_SUMMON, EV_FLIP_SUMMON), ev(EV_SUMMON, EV_FLIP_SUMMON, EV_SPSUMMON),
                                         ev(EV_SPSUMMON)])
                    e["conds"] = [("evp", e["on"], OPP)]
                else:
                    e["on"] = ev(EV_ATTACK)
                    e["conds"] = [("turn", OPP)]
            e["clauses"] = self.clauses(dict(kind="act", on=e["on"]))
            if e["on"] == ev(EV_ATTACK) and e["clauses"][0].get("sel") != "event" and self.p(0.5):
                e["clauses"][0] = dict(verb="destroy", sel="all", f=dict(op=Z_MZONE, pos=P_FU_ATK | P_FD_ATK), k=1, v=0)
            e["cost"] = self.cost([None, "lp", "discard"], [6, 2, 1])
            c1 = e["clauses"][0]  # stat modifier: can be activated during the damage step (Shrink, Reverse Glasses)
            if (ct & (TYPE_QUICKPLAY | TYPE_TRAP)) and len(e["clauses"]) == 1 and self.p(0.5) and \
                    c1["verb"] in ("atk", "def", "halve_base", "halve") and c1["sel"] in ("target", "all"):
                e["dmg"] = DMG_STEP | DMG_STAT
            return e
        if kind == "hand_proc":
            e["cost"] = self.cost([None, "lp", "discard", "banish_gy"], [4, 1, 2, 1])
            e["rng"] = Z_HAND
            e["opt"] = self.pick([OPT_NONE, OPT_NAME])
            mode = self.pick(["own_empty", "op_empty", "cyber", "ally"])
            if mode == "ally":
                f = dict(me=Z_MZONE, pos=P_FACEUP)
                self.qual(f, stats=False)
                e["counts"] = [(f, 1, BIG)]
            else:
                e["counts"] = [(dict(**({"op": Z_MZONE} if mode == "op_empty" else {"me": Z_MZONE})),0,0)]
                if mode == "cyber":
                    e["counts"].append((dict(op=Z_MZONE),1,BIG))
            return e
        if kind == "ignition":
            e["cost"] = self.cost([None, "lp", "discard", "self", "tribute", "banish_gy"] +
                                  (["detach"] if card["ctype"] & TYPE_XYZ else []),
                                  [4, 1.5, 1.5, 1, 1, 1] + ([4] if card["ctype"] & TYPE_XYZ else []))
            # repeatable effect without cost that consumes a card: once per turn
            if e["cost"] is None or e["cost"][0] == "lp" or self.p(0.5):
                e["opt"] = self.pick([OPT_NAME, OPT_NAME, OPT_CARD])
            e["clauses"] = self.clauses(dict(kind=kind, zone=None if e["cost"] == ("self",) else Z_MZONE))
            return e
        if kind == "trigger":
            trig = self.pick(["summon", "spsummon", "flip", "to_gy_field", "to_gy", "battle_destroyed"],
                             [3, 1.5, 1.5 if not card.get("extra") else 0, 2, 1, 1.5])
            e["on"], zone = {"summon": (ev(EV_SUMMON), Z_MZONE), "spsummon": (ev(EV_SPSUMMON), Z_MZONE),
                             "flip": (ev(EV_FLIP), Z_MZONE), "to_gy_field": (ev(EV_TO_GRAVE), Z_GRAVE),
                             "to_gy": (ev(EV_TO_GRAVE), Z_GRAVE),
                             "battle_destroyed": (ev(EV_BATTLE_DESTROYED), Z_GRAVE)}[trig]
            if trig == "flip":
                card["flip"] = True
            if trig == "to_gy_field":
                e["conds"] = [("self", 0, ONFIELD, 0)]
            elif trig == "battle_destroyed":
                e["conds"] = [("self", Z_GRAVE, 0, 0)]
            elif trig == "spsummon" and self.p(0.5):
                typ = core.SUMMON_SPECIAL
                for flag, value in ((TYPE_FUSION, core.SUMMON_FUSION), (TYPE_RITUAL, core.SUMMON_RITUAL),
                                    (TYPE_SYNCHRO, core.SUMMON_SYNCHRO), (TYPE_XYZ, core.SUMMON_XYZ),
                                    (TYPE_LINK, core.SUMMON_LINK)):
                    if card["ctype"] & flag:
                        typ = value
                e["conds"] = [("self", 0, 0, core.summon_mask(typ))]
            if trig in ("to_gy_field", "to_gy", "battle_destroyed") and self.p(0.35):
                mask = self.pick([core.REASON_EFFECT, core.REASON_DESTROY, core.REASON_BATTLE,
                                  core.REASON_COST, core.REASON_COST | core.REASON_EFFECT])
                e["conds"].append(("self", 0, 0, -((1 << 30) | mask)))
            e["optional"] = self.p(0.6)
            e["opt"] = OPT_NAME if self.p(0.5) else OPT_NONE
            e["cost"] = self.cost([None, "lp"], [6, 1]) if e["optional"] else None
            e["clauses"] = self.clauses(dict(kind=kind, zone=zone))
            return e
        if kind == "quick":
            hand = mon and not card.get("extra") and self.p(0.5)
            e["rng"] = Z_HAND if hand else Z_MZONE
            if hand:
                e["cost"] = ("self",)
            else:
                e["opt"] = self.pick([OPT_NAME, OPT_NAME, OPT_CARD])
                e["cost"] = self.cost([None, "lp", "discard", "tribute"] + (["detach"] if card["ctype"] & TYPE_XYZ else []),
                                      [4, 1, 1, 0.5] + ([3] if card["ctype"] & TYPE_XYZ else []))
            if self.p(0.5):
                e["conds"].append(("turn", OPP))
            if self.p(0.2):
                e["conds"].append(("phase", PH_MAIN1 | PH_MAIN2))
            if self.p(0.15):  # counter-effect: negate a Spell / Trap activation, or destroy the card of an effect
                if self.p(0.5):
                    e["conds"] = [("chain", self.pick([TYPE_SPELL, TYPE_TRAP, TYPE_SPELL | TYPE_TRAP]))]
                    e["clauses"] = [dict(verb="negate")]
                else:
                    e["conds"] = [("chain_field",)]
                    e["clauses"] = [dict(verb="chain_destroy")]
                return e
            e["clauses"] = self.clauses(dict(kind=kind, zone=None if hand else Z_MZONE))
            return e
        if kind == "cont":
            e["rng"] = Z_MZONE if mon else Z_SZONE
            e["cont"] = self.cont_spec(mon)
            return e
        raise ValueError(kind)

    def cont_spec(self, mon):
        """Continuous effect: stat modifier, prohibition or protection."""
        r = self.pick(["atk_field", "atk_self", "debuff", "no_attack", "no_btarget", "indes_battle", "no_etarget",
                       "imm_trap", "no_ss", "p_no_trap", "disable", "indes_eff", "no_direct"],
                      [4, 2 if mon else 0, 2, 1.5, 1 if mon else 0, 1 if mon else 0, 1, 1 if mon else 0, 1, 0.7,
                       0.5 if not mon else 0, 1, 0.5])
        c = dict(code=MC_ATK, val=0, tgt=None, per=None, cnt=None, tp=0, ph=0)
        # stat modifiers: no filter on the ATK / DEF of the targets (ygopro evaluates the target during the calculation
        # of the ATK of the card, GetAttack returns the partial value temp.attack there: not modelled by the engine)
        if r == "atk_field":
            c.update(code=self.pick([MC_ATK, MC_ATK, MC_DEF]), val=self.pick([200, 300, 400, 500]),
                     tgt=self.qual(dict(me=Z_MZONE, typ=TYPE_MONSTER), 1, stats=False))
            if self.p(0.2):
                c["tp"], c["ph"] = ME + 1, PH_BATTLE
        elif r == "atk_self":
            c.update(code=MC_ATK, val=self.pick([100, 200, 300, 500]))
            if self.p(0.5):
                c["per"] = self.pick([dict(me=Z_GRAVE, typ=TYPE_MONSTER), dict(me=Z_MZONE, not_self=1),
                                      self.mon(1, me=Z_GRAVE, op=Z_GRAVE)])
            else:
                c["val"] = self.pick([500, 800, 1000])
                c["cnt"] = (self.pick([dict(me=Z_MZONE, not_self=1), dict(op=Z_MZONE)]), self.pick([1, 2]), BIG)
        elif r == "debuff":
            c.update(code=self.pick([MC_ATK, MC_DEF]), val=-self.pick([200, 300, 500]),
                     tgt=self.qual(dict(op=Z_MZONE, typ=TYPE_MONSTER), 1, stats=False))
        elif r == "no_attack":
            c.update(code=MC_NO_ATTACK, tgt=self.pick([dict(op=Z_MZONE, lv=(self.pick([4, 5, 7]), BIG)),
                                                      dict(op=Z_MZONE, st=ST_SUMMON_TURN | ST_FLIP_TURN),
                                                      dict(op=Z_MZONE, atk=(self.pick([1500, 2000]), BIG))]))
        elif r == "no_btarget":
            c.update(code=MC_NO_BTARGET, cnt=(dict(me=Z_MZONE), 2, BIG))
        elif r == "indes_battle":
            c.update(code=MC_INDES_BATTLE, val=self.pick([0, 1900, 2000]))
        elif r == "no_etarget":
            c.update(code=MC_NO_ETARGET, tgt=self.mon(1, me=Z_MZONE))
        elif r == "imm_trap":
            c.update(code=MC_IMM_TRAP)
        elif r == "no_ss":
            c.update(code=MC_NO_SS, tgt=self.qual(dict(me=Z_DECK | Z_HAND | Z_GRAVE | Z_REMOVED | Z_EXTRA,
                                                       op=Z_DECK | Z_HAND | Z_GRAVE | Z_REMOVED | Z_EXTRA),
                                                  1, stats=False, lv=False))
        elif r == "p_no_trap":
            c.update(code=MC_P_NO_TRAP, val=2, ph=PH_BATTLE)
        elif r == "disable":
            c.update(code=MC_DISABLE, tgt=dict(op=Z_MZONE, typ=TYPE_EFFECT))
        elif r == "indes_eff":
            c.update(code=MC_INDES_EFF, tgt=None if mon and self.p(0.5) else self.mon(1, me=Z_MZONE))
        elif r == "no_direct":
            c.update(code=MC_NO_DIRECT, tgt=dict(op=Z_MZONE))
        return c

    # ---------------------------------------------------------------- cards
    def stats(self, lv, val):
        tot = max(0, STAT_BUDGET[lv] - 900 * val)
        a = self.rng.uniform(0.25, 0.8)
        return _r100(tot * a), _r100(tot * (1 - a))

    def base(self, ctype, lv=0):
        return dict(ctype=ctype, atk=0, df=0, lv=lv, attr=0, race=0, setc=[], lsc=0, rsc=0, effects=[],
                    recipe=None, flip=False, extra=False)

    def monster_frame(self, extra=False, lv=None):
        lv = int(self.rng.choice(12, p=LV_W / LV_W.sum())) + 1 if lv is None else lv
        c = self.base(TYPE_MONSTER, lv)
        c.update(attr=self.pick(ATTRS), race=self.pick(RACES), extra=extra)
        A = self.arch
        if A:
            c["setc"] = [A]
            if self.p(0.7):
                c["attr"] = self.arch_attr[A]
            if self.p(0.7):
                c["race"] = self.arch_race[A]
        elif self.p(0.1 if self.arch_p else 0.3):
            c["setc"] = [self.pick(SETS)]
        return c

    def with_effects(self, c, kinds_w, n=None):
        """Adds 1 to E effects (kinds drawn according to kinds_w), within the card budget. Number of effects drawn
        according to N_EFF_W (close to the real effect cards, cut at E); if it does not fit in the budget, one fewer."""
        if n is None:
            w = N_EFF_W[:E]
            n = 1 + self.pick(list(range(len(w))), w)
        kinds = list(kinds_w)
        for attempt in range(40):
            if attempt and attempt % 10 == 0 and n > 1:
                n -= 1
            effs, used = [], []
            for _ in range(n):
                # A single hand procedure; several trigger effects can
                # share the same event point and build a chain.
                k = self.pick(kinds, [kinds_w[x] * (0 if x == "hand_proc" and x in used else 0.3 if x in used else 1)
                                      for x in kinds])
                used.append(k)
                effs.append(self.effect(k, c))
            v = sum(value(e) for e in effs)
            if v <= CARD_CAP and all(fits(e) for e in effs):
                c["effects"] = effs
                return c, v
        c["effects"] = []
        return c, 0.0

    def skel_card(self, k):
        """Card whose effects come from a skeleton (structure of a translated card, parameters re-drawn); None if no
        skeleton of the group fits in the engine arrays."""
        import skeleton
        group = dict(effect="monster", spell="spell", quick="quick-play spell", cspell="continuous spell", trap="trap",
                     ctrap="continuous trap", counter="counter trap")[k]
        r = self.srng
        A = self.arch
        pick = lambda xs: xs[int(r.integers(len(xs)))]
        for _ in range(20):
            if k == "effect":
                c = self.monster_frame()
                c["ctype"] |= TYPE_EFFECT
            else:
                c = self.base(dict(spell=TYPE_SPELL, quick=TYPE_SPELL | TYPE_QUICKPLAY, cspell=TYPE_SPELL | TYPE_CONTINUOUS,
                                   trap=TYPE_TRAP, ctrap=TYPE_TRAP | TYPE_CONTINUOUS, counter=TYPE_TRAP | TYPE_COUNTER)[k])
            # filter qualifiers: those of the card's archetype most of the time (support cards of a deck)
            attr = lambda: self.arch_attr[A] if A and r.random() < 0.7 else pick(ATTRS)
            race = lambda: self.arch_race[A] if A and r.random() < 0.7 else pick(RACES)
            setc = lambda: A if A and r.random() < 0.8 else pick(SETS)
            unit = skeleton.draw(r, group, attr, race, setc, E)
            if unit is None:
                return None
            effs = [dict(kind="raw", raw=e, val=0.8, sup=_raw_support(e)) for e in unit]
            if all(fits(e) for e in effs):
                c["effects"] = effs
                c["skel"] = True
                if k == "effect":
                    c["atk"], c["df"] = self.stats(c["lv"], min(CARD_CAP, 0.8 * len(effs)))
                return c
        return None

    def effect_monster(self, lv=None):
        c = self.monster_frame(lv=lv)
        c["ctype"] |= TYPE_EFFECT
        c, v = self.with_effects(c, dict(ignition=3, trigger=4, quick=2, cont=2, hand_proc=1))
        c["atk"], c["df"] = self.stats(c["lv"], v)
        if not c["effects"]:
            c["ctype"] = (c["ctype"] & ~TYPE_EFFECT) | TYPE_NORMAL
        return c

    def normal_monster(self, tuner=False, pend=False):
        c = self.monster_frame(lv=self.pick([1, 2, 3, 4]) if tuner else None)
        c["ctype"] |= TYPE_NORMAL | (TYPE_TUNER if tuner else 0) | (TYPE_PENDULUM if pend else 0)
        c["atk"], c["df"] = self.stats(c["lv"], 0.4 if tuner else 0.0)
        if pend:
            c["lsc"] = c["rsc"] = int(self.rng.integers(0, 11))
            c["effects"] = [dict(kind="pend")]
        return c

    def spell(self, sub):
        ct = TYPE_SPELL | {"normal": 0, "quick": TYPE_QUICKPLAY, "cont": TYPE_CONTINUOUS}[sub]
        c = self.base(ct)
        if sub == "cont":
            for _ in range(20):
                e = self.effect("cont", c)
                if value(e) <= CARD_CAP:
                    break
            c["effects"] = [dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], cost=None,
                                 clauses=[]), e]
            return c
        return self.with_effects(c, dict(act=1), n=1)[0]

    def trap(self, sub):
        ct = TYPE_TRAP | {"normal": 0, "cont": TYPE_CONTINUOUS, "counter": TYPE_COUNTER}[sub]
        c = self.base(ct)
        if sub == "cont":
            if self.p(0.25):
                for _ in range(20):
                    e = dict(kind="quick", on=0, opt=self.pick([OPT_NAME, OPT_CARD]), optional=False,
                             rng=Z_SZONE, dmg=0, conds=[],
                             cost=self.cost([None, "lp", "discard", "tribute"], [4, 1, 1, 1]),
                             clauses=self.clauses(dict(kind="quick", zone=Z_SZONE)))
                    e['refund'] = e['opt'] == OPT_NAME and self.p(0.5)
                    if fits(e) and value(e) <= CARD_CAP:
                        break
                a = dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0,
                         conds=[], cost=None, clauses=[], inline=2)
                c["effects"] = [a, e]
                return c
            e = self.effect("cont", c)
            a = dict(kind="act", on=0, opt=0, optional=False, rng=0, dmg=0, conds=[], clauses=[],
                     cost=("lp", 1000) if e["cont"]["code"] == MC_DISABLE else None)
            c["effects"] = [a, e]
            return c
        return self.with_effects(c, dict(act=1), n=1)[0]

    def extra_monster(self, kind, main):
        """Extra Deck monster; main: main deck monsters (codes named by the Fusion recipes)."""
        mats = lambda: self.qual(dict(), 1, stats=False, lv=False) if self.p(0.4) else dict()
        if kind == "fusion":
            lv = self.pick([4, 5, 6, 7, 8, 9, 10])
        elif kind == "synchro":
            lv = self.pick([3, 4, 5, 6, 7, 8, 9, 10])
        elif kind == "xyz":
            lv = self.pick([2, 3, 4, 4, 4, 5, 6, 7, 8])
        else:
            lv = self.pick([1, 2, 2, 3, 3, 4])
        c = self.monster_frame(extra=True, lv=lv)
        c["ctype"] |= {"fusion": TYPE_FUSION, "synchro": TYPE_SYNCHRO, "xyz": TYPE_XYZ, "link": TYPE_LINK}[kind]
        if kind == "fusion":
            n = self.pick([2, 2, 2, 3])
            rec = []
            for _ in range(n):
                if self.p(0.35) and main:
                    rec.append(dict(code=self.pick(main)))
                else:
                    rec.append(self.qual(dict(), 1, stats=False))
            c["recipe"] = ("fusion", rec)
        elif kind == "synchro":
            c["recipe"] = ("synchro", (mats(), mats()))
        elif kind == "xyz":
            c["recipe"] = ("xyz", (mats(), self.pick([2, 2, 2, 3])))
        else:
            arrows = self.rng.choice(8, size=lv, replace=False)
            c["df"] = int(sum(ARROWS[i] for i in arrows))
            c["recipe"] = ("link", (mats(), 1 if lv == 1 else 2, max(2, lv) if lv > 1 else 1, 0))
        v = 0.0
        if self.p(0.7):
            c["ctype"] |= TYPE_EFFECT
            c, v = self.with_effects(c, dict(ignition=3, trigger=3, quick=2, cont=2))
        bonus = {"fusion": 400, "synchro": 500, "xyz": 500, "link": 600}[kind]
        tot = STAT_BUDGET[max(1, min(12, lv if kind != "link" else 2 * lv + 1))] + bonus - 900 * v
        if kind == "link":
            c["atk"] = _r100(max(0, tot * self.rng.uniform(0.4, 0.55)))
        else:
            c["atk"], c["df"] = self.stats(lv, 0)
            s = (tot / max(1, c["atk"] + c["df"]))
            c["atk"], c["df"] = _r100(c["atk"] * s), _r100(c["df"] * s)
        return c

    def ritual_pair(self):
        """Ritual Monster + its Ritual Spell (Greater or Equal); the monster code is set by make_set."""
        m = self.monster_frame(lv=self.pick([4, 5, 6, 7, 8, 8, 10]))
        m["ctype"] |= TYPE_RITUAL
        v = 0.0
        if self.p(0.6):
            m["ctype"] |= TYPE_EFFECT
            m, v = self.with_effects(m, dict(ignition=3, trigger=3, quick=2, cont=2))
        m["atk"], m["df"] = self.stats(m["lv"], v - 0.6)
        s = self.base(TYPE_SPELL | TYPE_RITUAL)
        s["effects"] = [dict(kind="ritual", mode=self.pick([RIT_GREATER, RIT_EQUAL]))]
        m["recipe"] = ("ritual", None)
        return m, s

    def fusion_spell(self):
        s = self.base(TYPE_SPELL)
        f = dict(me=Z_EXTRA, fus=1)
        if self.arch:
            f["setc"] = self.arch
        elif self.p(0.3):
            f["setc"] = self.pick(SETS)
        s["effects"] = [dict(kind="fusion_spell", f=f)]
        return s

    def make_set(self, n_main=400, n_extra=80):
        """Procedural bank: n_main main deck cards (index 0..), then n_extra Extra Deck cards."""
        main = []
        kinds = ["normal", "tuner", "pend", "effect", "spell", "quick", "cspell", "fusion_spell", "ritual",
                 "trap", "ctrap", "counter"]
        w = [14, 4, 2, 32, 12, 5, 3, 2, 2, 12, 3, 2]
        while len(main) < n_main:
            k = self.pick(kinds, w)
            if self.arch_p:
                self.arch = self.pick(SETS) if self.p(self.arch_p) else None
            n0 = len(main)
            sk = None
            if self.skel and k in ("effect", "spell", "quick", "cspell", "trap", "ctrap", "counter") \
                    and self.srng.random() < self.skel:
                sk = self.skel_card(k)
            if sk is not None:
                main.append(sk)
            elif k == "normal":
                main.append(self.normal_monster())
            elif k == "tuner":
                main.append(self.effect_monster(lv=self.pick([1, 2, 3, 4])) if self.p(0.5) else self.normal_monster(True))
                main[-1]["ctype"] |= TYPE_TUNER
            elif k == "pend":
                main.append(self.normal_monster(pend=True))
            elif k == "effect":
                main.append(self.effect_monster())
            elif k in ("spell", "quick", "cspell"):
                main.append(self.spell({"spell": "normal", "quick": "quick", "cspell": "cont"}[k]))
            elif k == "fusion_spell":
                main.append(self.fusion_spell())
            elif k == "ritual" and len(main) + 2 <= n_main:
                m, s = self.ritual_pair()
                main += [m, s]
            elif k in ("trap", "ctrap", "counter"):
                main.append(self.trap({"trap": "normal", "ctrap": "cont", "counter": "counter"}[k]))
            if self.arch:
                for c in main[n0:]:
                    c["arch"] = self.arch
        for i, c in enumerate(main):
            c["code"] = CODE0 + i
        for i, c in enumerate(main):  # Ritual Spell: code of the Ritual Monster that precedes it
            if c["effects"] and c["effects"][0]["kind"] == "ritual":
                c["effects"][0]["code"] = main[i - 1]["code"]
        mon_codes = [c["code"] for c in main if c["ctype"] & TYPE_MONSTER and not c["ctype"] & TYPE_RITUAL]
        extra = []
        for _ in range(n_extra):
            if self.arch_p:
                self.arch = self.pick(SETS) if self.p(self.arch_p) else None
            extra.append(self.extra_monster(self.pick(["fusion", "synchro", "xyz", "link"]), mon_codes))
            if self.arch:
                extra[-1]["arch"] = self.arch
        self.arch = None
        for i, c in enumerate(extra):
            c["code"] = CODE0 + n_main + i
        return main + extra


# ---------------------------------------------------------------------- value (rough budget)

def clause_value(c):
    v, k = c["verb"], c.get("k", 1)
    if c.get("first_original"):
        k = 1
    sel = c.get("sel")
    if v == "draw":
        return 1.0 * c["n"]
    if v == "damage":
        return c["v"] / 1500
    if v == "recover":
        return c["v"] / 3000
    if v == "burn_per":
        return 2.5 * c["v"] / 1500
    if v in ("negate", "chain_destroy"):
        return 1.3
    mult = {"target": 1.0, "pick": 1.0, "event": 0.9, "self": 0.8, "all": 1.8, "min": 0.9}[sel]
    base = {"destroy": 1.0, "banish": 1.15, "bounce": 0.8, "tograve": 0.9, "decktop": 0.9, "tohand": 0.9, "ss": 1.0,
            "atk": abs(c.get("v", 0)) / 1500 * (1.6 if c.get("perm") else 1.0), "def": c.get("v", 0) / 3000,
            "halve_base": 0.6, "halve": 0.5, "disable": 0.7, "indes_battle": 0.6}[v]
    if c.get("lock"):
        base -= 0.1
    if v == "tograve" and c["f"].get("me") == Z_DECK:
        base = 0.3
    if v in ("tohand", "ss") and sel == "all":
        mult = 1.0
    return base * mult * k


def cost_value(cost):
    if cost is None:
        return 0.0
    return {"lp": cost[1] / 2500 if cost[0] == "lp" else 0, "discard": 0.9, "self": 0.6, "tribute": 0.9,
            "banish_gy": 0.3, "detach": 0.4}[cost[0]]


def value(e):
    """Rough value of an effect, in cards of advantage (before the value of the card itself)."""
    k = e["kind"]
    if k == "raw":
        return e.get("val", 0.8)
    if k == "hand_proc":
        return 0.7
    if k == "cont":
        c = e["cont"]
        if c["code"] in (MC_ATK, MC_DEF):
            return abs(c["val"]) / 1000 * (2.0 if c["tgt"] is not None else 1.0) * (2.5 if c["per"] else 1.0)
        return {MC_DISABLE: 1.8, MC_NO_ATTACK: 1.0, MC_NO_SS: 0.8, MC_P_NO_TRAP: 0.6}.get(c["code"], 0.6)
    if k in ("pend", "ritual", "fusion_spell"):
        return 0.0
    v = sum(clause_value(c) for c in e["clauses"])
    speed = {"act": 1.0, "ignition": 1.0, "trigger": 0.9, "quick": 1.25}[k]
    if k == "act" and e["on"]:
        speed = 0.9
    if e["opt"] == OPT_NONE and k in ("ignition", "quick") and e["cost"] not in (("self",),):
        speed *= 1.3  # repeatable
    return max(0.0, v * speed - cost_value(e["cost"]) - 0.15 * len(e["conds"]))


# ---------------------------------------------------------------------- translation into bricks

def _F(kw):
    return F(**kw)


COST_REGS = (2, 3)


def cost_bricks(cost):
    a, b = COST_REGS
    if cost is None:
        return []
    t = cost[0]
    if t == "lp":
        return [PAY_LP(cost[1])]
    if t == "discard":
        return [FILTER(a, F(me=Z_HAND, not_self=1)), CHOOSE(b, a, 1), MOVE(b, reason=core.REASON_COST | core.REASON_DISCARD)]
    if t == "self":
        return [SELF(a), MOVE(a, reason=core.REASON_COST, release_if_field=True)]
    if t == "tribute":
        return [FILTER(a, F(me=Z_MZONE, not_self=1)), CHOOSE(b, a, 1), MOVE(b, reason=core.REASON_COST | core.REASON_RELEASE)]
    if t == "banish_gy":
        return [FILTER(a, F(me=Z_GRAVE, typ=TYPE_MONSTER)), CHOOSE(b, a, 1), MOVE(b, Z_REMOVED, reason=core.REASON_COST)]
    if t == "detach":
        return [FILTER(a, F(me=Z_OVERLAY)), CHOOSE(b, a, cost[1]), MOVE(b, reason=core.REASON_COST)]
    raise ValueError(cost)


def cond_bricks(conds):
    out = []
    for c in conds:
        if c[0] == "turn":
            out.append(COND_TURN(c[1]))
        elif c[0] == "phase":
            out.append(COND_PHASE(c[1]))
        elif c[0] == "self":
            out.append(COND_SELF(zone=c[1], prev=c[2], st=c[3]))
        elif c[0] == "evp":
            out.append(COND_EVP(c[1], c[2]))
        elif c[0] == "chain":
            out.append(COND_CHAIN(c[1]))
        elif c[0] == "chain_field":
            out.append(COND_CHAIN_FIELD())
        else:
            raise ValueError(c)
    return out


def verb_bricks(c, r):
    v = c["verb"]
    if v == "destroy":
        return [DESTROY(r)]
    if v == "banish":
        return [MOVE(r, Z_REMOVED)]
    if v in ("bounce", "tohand"):
        return [MOVE(r, Z_HAND)] + ([LOCK(r)] if c.get("lock") else [])
    if v == "tograve":
        return [MOVE(r, Z_GRAVE)]
    if v == "decktop":
        return [MOVE(r, Z_DECK)]
    if v == "ss":
        return [SPSUMMON(r, c["pos"], tag=c.get("tag", 0))]
    if v == "atk":
        return [MODIFY(r, MC_ATK, c["v"], dur=TD_PERM if c.get("perm") else core.TD_TURN, rd=bool(c.get("rd")))]
    if v == "def":
        return [MODIFY(r, MC_DEF, c["v"])]
    if v == "halve_base":
        return [MODIFY(r, MC_BASE_ATK, mode=MV_HALF_BASE_ATK)]
    if v == "halve":
        return [MODIFY(r, MC_FINAL_ATK, mode=MV_HALF_ATK)]
    if v == "disable":
        return [MODIFY(r, MC_DISABLE, nrc=True)]  # procgen.lua: NegateRelatedChain then DISABLE
    if v == "indes_battle":
        return [MODIFY(r, MC_INDES_BATTLE, 0)]
    raise ValueError(v)


def clause_bricks(c, on, first):
    """(target, resolution, result register or None) of a clause. The first clause can target (registers 0-2;
    the target keeps register 1 until resolution); the second one only uses register 3 at activation."""
    v = c["verb"]
    if v == "draw":
        rr = 0 if first else 3
        return [FILTER(rr, F(me=Z_DECK)), REQ(rr, c["n"])], [DRAW(ME, c["n"])], None
    if v == "damage":
        return [], [DAMAGE(OPP, c["v"])], None
    if v == "recover":
        return [], [RECOVER(ME, c["v"])], None
    if v == "burn_per":
        return [FILTER(0, _F(c["f"])), REQ(0, 1)], [FILTER(0, _F(c["f"])), DAMAGE(OPP, c["v"], per=0)], 0
    if v == "negate":
        return [], [NEGATE(), CHAINCARD(0), DESTROY(0)], None
    if v == "chain_destroy":
        return [CHAINCARD(1, ONFIELD)], [DESTROY(1)], 1
    f, sel, k = c["f"], c["sel"], c["k"]
    ss = v == "ss"
    pre_t = [REQ_FREE(ME, Z_MZONE, 1)] if ss else []
    pre_r = [REQ_FREE(ME, Z_MZONE, 1)] if ss else []
    if not first:  # "then": selection at resolution only
        assert sel in ("all", "pick")
        if sel == "all":
            return [], [FILTER(0, _F(f))] + verb_bricks(c, 0), 0
        return [], pre_r + [FILTER(0, _F(f)), CHOOSE(1, 0, k)] + verb_bricks(c, 1), 1
    if sel == "target":
        select = ([cards.FIRST(2, 1, original=True), REQ(2, 1, related=True)]
                  if c.get("first_original") else [])
        return (pre_t + [FILTER(0, _F(f), tgt=True), CHOOSE(1, 0, k, tgt=True)],
                pre_r + select + [FILTER(2, _F(f), within=2 if select else 1, related=True)] + verb_bricks(c, 2), 2)
    if sel == "all":
        return [FILTER(0, _F(f)), REQ(0, 1)], [FILTER(2, _F(f))] + verb_bricks(c, 2), 2
    if sel == "pick":
        return (pre_t + [FILTER(0, _F(f)), REQ(0, k)],
                pre_r + [FILTER(2, _F(f)), CHOOSE(1, 2, k)] + verb_bricks(c, 1), 1)
    if sel == "min":  # Fissure
        return ([FILTER(0, _F(f)), REQ(0, 1)],
                [FILTER(0, _F(f)), REQ(0, 1), MINATK(1, 0), CHOOSE(2, 1, 1)] + verb_bricks(c, 2), 2)
    if sel == "event":
        return ([EVENT(0, on), FILTER(1, _F(f), within=0, tgt=True), REQ(1, 1), cards.LABEL(1, target=True)],
                [FILTER(2, _F(f), within=1, related=True)] + verb_bricks(c, 2), 2)
    if sel == "self":
        return (pre_t + [SELF(0), FILTER(1, _F(f), within=0), REQ(1, 1)],
                pre_r + [SELF(0), FILTER(2, _F(f), within=0)] + verb_bricks(c, 2), 2)
    raise ValueError(sel)


def bricks(e):
    """Effect (specification) -> effect tuple of cards.py."""
    k = e["kind"]
    if k == "raw":  # effect of a skeleton: already in bricks
        return e["raw"]
    if k == "hand_proc":
        program = [REQ_FREE(ME,Z_MZONE,1)]
        for r,(f,mn,mx) in enumerate(e["counts"]):
            program.append(FILTER(r,_F(f)))
            if mn > 0: program.append(REQ(r,mn))
            if mx < BIG: program.append(REQ(r,mx,maximum=True))
        return (core.K_PROC,0,e["opt"],Z_HAND,program,cost_bricks(e.get("cost")),[],
                {"refund": int(e["opt"] == OPT_NAME)})
    if k == "pend":
        return act()
    if k == "cont":
        c = e["cont"]
        return cont(e["rng"], c["code"], c["val"], tgt=None if c["tgt"] is None else _F(c["tgt"]),
                    per=None if c["per"] is None else _F(c["per"]),
                    cnt=None if c["cnt"] is None else (_F(c["cnt"][0]), c["cnt"][1], c["cnt"][2]), tp=c["tp"], ph=c["ph"])
    if k == "fusion_spell":
        f = _F(e["f"])
        return act(target=[FILTER(0, f), REQ(0, 1)],
                   resolve=[FILTER(0, f), CHOOSE(1, 0, 1), cards.FUSMAT(2, 1), MOVE(2, reason=core.REASON_EFFECT | core.REASON_MATERIAL | core.REASON_FUSION), SPSUMMON(1, P_FU_ATK | P_FU_DEF, fusion=True)])
    if k == "ritual":
        return ritual_spell(e["code"], e["mode"])[0]
    conditions = cond_bricks(e["conds"])
    cost = conditions + cost_bricks(e["cost"])
    target, resolve = [], [SELF(0, related=True), REQ(0, 1)] if e.get("related") else []
    first = True
    for c in e["clauses"]:
        t, r, out = clause_bricks(c, e["on"], first)
        target += t
        resolve += r
        if first and out is not None and len(e["clauses"]) > 1:
            resolve.append(REQ(out, 1))  # "then": only if the first clause has hit a card
        first = False
    if k == "act":
        result = act(cost=cost, target=target, resolve=resolve, on=e["on"], dmg=e["dmg"], opt=e["opt"])
        if e.get("inline"):
            result = (*result[:7], dict(result[7], inline=e["inline"]))
    elif k == "ignition":
        result = ignition(cost=cost, target=target, resolve=resolve, opt=e["opt"])
    elif k == "trigger":
        result = trigger(e["on"], cost=cost, target=target, resolve=resolve, optional=e["optional"],
                         opt=e["opt"] == OPT_NAME, cond=len(conditions))
    elif k == "quick":
        result = quick(e["rng"], cost=cost, target=target, resolve=resolve, opt=e["opt"] if e["opt"] else False, on=e["on"])
    else:
        raise ValueError(k)
    if e.get('refund'):
        result = (*result[:7], dict(result[7], refund=1))
    return result


def _raw_support(e):
    """Qualified filters of an effect in bricks that concern its own monsters, in the form read by
    structured_decks and _fmatch (deck support: a card that matches these filters)."""
    out = []
    fl = [i[2] for p in e[4:7] for i in p if i[0] == core.OP_FILTER and isinstance(i[2], tuple) and len(i[2]) == 24]
    for f in fl:
        me, op, attr, race, setc, lo, hi = f[0], f[1], f[9], f[10], f[11], f[15], f[16]
        if me and not op and (attr or race or setc or lo > -BIG or hi < BIG):
            d = dict(me=me)
            if attr:
                d["attr"] = attr
            if race:
                d["race"] = race
            if setc:
                d["setc"] = setc
            if lo > -BIG or hi < BIG:
                d["lv"] = (max(lo, 0), min(hi, 12))
            out.append(d)
    return out


def fits(e):
    """The effect fits in the engine arrays (PT cost and target instructions, P - 1 resolution instructions, NF
    filters)."""
    b = bricks(e)
    progs = b[4:7]
    fl = set()
    for pr in progs:
        for ins in pr:
            if ins[0] == core.OP_FILTER:
                fl.add(ins[2])
    extra = b[7] if len(b) > 7 else {}
    for f in extra.get("filters", ()):
        fl.add(f)
    return len(progs[0]) <= PT and len(progs[1]) <= PT and len(progs[2]) < P and len(fl) <= NF


# ---------------------------------------------------------------------- bank and decks

def to_bank(cset):
    """List of cards (specifications) -> core.Bank. tprog is padded up to 2 x PT instructions: all the procedural
    banks then have the same shapes (a single compilation for all the seeds)."""
    import jax.numpy as jnp
    codes = [c["code"] for c in cset]
    effects = {c["code"]: [bricks(e) for e in c["effects"]] for c in cset}
    rec = dict(fusion={}, synchro={}, xyz={}, link={}, ritual={})
    for c in cset:
        if c["recipe"] is None:
            continue
        kind, r = c["recipe"]
        mat = lambda kw: cards.MAT(**kw)
        if kind == "fusion":
            # Only raw names allow a substitute in Fusion.AddProcMix.
            mask = 0
            for j, material in enumerate(r):
                code = material.get("code")
                same_code = isinstance(code, (int, np.integer)) or isinstance(code, (list, tuple)) and len(code) == 2 and code[0] == code[1]
                if code and same_code and not any(material.get(k) for k in ("attr", "race", "setc", "lv", "typ", "not_typ")):
                    mask |= 1 << j
            rec["fusion"][c["code"]] = cards.FusionRecipe([mat(x) for x in r], mask)
        elif kind == "synchro":
            rec["synchro"][c["code"]] = (mat(r[0]), mat(r[1]))
        elif kind == "xyz":
            rec["xyz"][c["code"]] = (mat(r[0]), r[1])
        elif kind == "link":
            rec["link"][c["code"]] = (mat(r[0]), r[1], r[2], r[3])
        elif kind == "ritual":
            rec["ritual"][c["code"]] = (c["atk"], c["df"], c["lv"])
    asm = cards.assemble(codes, cards=effects, **rec)
    # Stable shape even if this bank drew no Trap with a combined activation.
    if not asm['einline'].shape[1]:
        asm['einline'] = np.zeros((len(cset), E), np.int32)
    sc = np.zeros((len(cset), NSET), np.int32)
    for i, c in enumerate(cset):
        sc[i, :len(c["setc"])] = c["setc"]
    bank = core.make_bank(np.array(codes), [c["atk"] for c in cset], [c["df"] for c in cset],
                          [c["lv"] for c in cset], ctype=[c["ctype"] for c in cset], attr=[c["attr"] for c in cset],
                          race=[c["race"] for c in cset], setcode=sc, lsc=[c["lsc"] for c in cset],
                          rsc=[c["rsc"] for c in cset], **asm)
    tp = bank.tprog
    pad = 2 * PT - tp.shape[2]
    return bank._replace(tprog=jnp.pad(tp, ((0, 0), (0, 0), (0, pad), (0, 0))),
                         erefund=bank.erefund if bank.erefund.shape[1] else jnp.zeros_like(bank.ekind))


def is_extra(c):
    return bool(c["ctype"] & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK))


def random_decks(cset, rng, b, d=40, x=15):
    """Random decks (without structure): [b, 2, d] and Extra Decks [b, 2, x], indices into the bank."""
    main = np.array([i for i, c in enumerate(cset) if not is_extra(c)])
    ext = np.array([i for i, c in enumerate(cset) if is_extra(c)])
    return _draw3(main, rng, (b, 2), d), _draw3(ext, rng, (b, 2), x)


def _draw3(pool, rng, shape, n):
    """n cards drawn at random from pool, at most 3 copies of each (game rule)."""
    p3 = np.repeat(pool, 3)
    return np.array([rng.choice(p3, n, replace=False) for _ in range(int(np.prod(shape)))]).reshape(*shape, n)


def _own_filters(x, out):
    """Qualified monster filters (attribute / type / archetype / Level) that concern its own cards."""
    if isinstance(x, dict):
        if x.get("me", 0) and not x.get("op", 0) and any(k in x for k in ("attr", "race", "setc", "lv")):
            out.append(x)
        for v in x.values():
            _own_filters(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _own_filters(v, out)
    return out


def _fmatch(f, c):
    return bool(c["ctype"] & TYPE_MONSTER and (not f.get("attr") or c["attr"] & f["attr"])
                and (not f.get("race") or c["race"] & f["race"])
                and ("setc" not in f or f["setc"] in c["setc"])
                and ("lv" not in f or f["lv"][0] <= c["lv"] <= f["lv"][1]))


def _cap3(cards):
    """At most 3 copies of a card (game rule), order preserved."""
    n, out = {}, []
    for c in cards:
        n[c] = n.get(c, 0) + 1
        if n[c] <= 3:
            out.append(c)
    return out


def _is_high(c):
    """Monster of Level 5 or more that requires a Tribute: neither Ritual nor summonable by its own procedure."""
    return bool(c["ctype"] & TYPE_MONSTER and not c["ctype"] & TYPE_RITUAL and c["lv"] >= 5
                and not any(e.get("kind") == "hand_proc" for e in c["effects"]))


def _self_summons(c):
    """The monster Summons itself (hand procedure, or an effect that Summons the card itself from the hand or the
    GY): it needs no support."""
    for e in c["effects"]:
        if e.get("kind") == "hand_proc":
            return True
        last = {}                                # register -> opcode that last wrote it
        for i in bricks(e)[6]:
            if i[0] == core.OP_SPSUMMON and last.get(i[1]) == core.OP_SELF:
                return True
            if i[0] in (core.OP_SELF, core.OP_FILTER, core.OP_CHOOSE, core.OP_EVENT):
                last[i[1]] = i[0]
    return False


def _enables_high(c):
    """The card Special Summons a monster of Level 5 or more other than itself, from the hand, the GY
    or the banished zone (read from the bricks: also works for the skeleton cards)."""
    for e in c["effects"]:
        if e.get("kind") in ("pend", "ritual", "fusion_spell", "cont", "hand_proc"):
            continue
        b = bricks(e)
        if not any(i[0] == core.OP_SPSUMMON for i in b[6]):
            continue
        for pr in b[4:7]:
            for i in pr:
                f = i[2]
                if i[0] == core.OP_FILTER and isinstance(f, tuple) and len(f) == 24:
                    me, typ, not_self, lv_hi = f[0], f[3], f[4], f[16]
                    if me & (Z_HAND | Z_GRAVE | Z_REMOVED) and (not typ or typ & TYPE_MONSTER) and lv_hi >= 5:
                        return True
    return False


def _apply_profile(cset, rng, cards, keep, low_pool, high_pool, st_pool, enablers, A, arch_main):
    """Randomly drawn composition profile (see structured_decks), then support of the big monsters."""
    cards = list(cards)
    n_mon = int(rng.integers(10, 33))
    n_high = int(round(n_mon * rng.beta(1.0, 2.5)))
    mon = lambda k: bool(cset[k]["ctype"] & TYPE_MONSTER)
    high = lambda k: _is_high(cset[k])
    arch = set(arch_main[A].tolist()) if A is not None else set()

    def swap(victims, pool, prefer=()):
        """A card of victims replaced by a card of pool (from prefer first) that still has fewer than 3 copies."""
        if not victims or not len(pool):
            return False
        v = victims[int(rng.integers(len(victims)))]
        cand = [k for k in prefer if cards.count(k) < 3] if prefer and rng.random() < 0.6 else []
        for _ in range(20):
            k = int(cand[int(rng.integers(len(cand)))]) if cand else int(pool[int(rng.integers(len(pool)))])
            if cards.count(k) < 3:
                cards[cards.index(v)] = k
                return True
        return False

    free = lambda pred: [k for k in cards if k not in keep and pred(k)]
    pref = lambda pool: [k for k in pool if k in arch]
    for _ in range(3 * len(cards)):                       # number of monsters
        m = sum(mon(k) for k in cards)
        if m > n_mon:
            ok = swap(free(mon), st_pool, pref(st_pool))
        elif m < n_mon:
            ok = swap(free(lambda k: not mon(k)), low_pool, pref(low_pool))
        else:
            break
        if not ok:
            break
    for _ in range(3 * len(cards)):                       # share of the big monsters
        h = sum(high(k) for k in cards)
        if h > n_high:
            ok = swap(free(high), low_pool, pref(low_pool))
        elif h < n_high and len(high_pool):
            ok = swap(free(lambda k: mon(k) and not high(k)), high_pool, pref(high_pool))
        else:
            break
        if not ok:
            break
    # support: a card that Summons a big monster from the hand / the GY for three big monsters that do not
    # Summon themselves; replaces cards that are neither big monster nor support, outside the packs
    hard = sum(high(k) and not _self_summons(cset[k]) for k in cards)
    need = -(-hard // 3) if hard >= 3 else 0
    en = set(enablers.tolist()) if enablers is not None else set()
    for _ in range(need):
        if sum(k in en for k in cards) >= need:
            break
        if not swap(free(lambda k: not high(k) and k not in en), enablers, pref(enablers)):
            break
    return cards


def _apply_caps(cset, rng, cards, keep, low_pool, st_pool):
    """Composition caps of a deck (see structured_decks); cards: indices into the bank, at most 3 copies."""
    cards = list(cards)
    max_high, max_mon = int(rng.integers(2, 7)), int(rng.integers(16, 25))
    count = lambda k: cards.count(k)

    def swap(victims, pool):
        """Replaces a card taken at random from victims by a card of pool that does not yet have 3 copies."""
        v = victims[int(rng.integers(len(victims)))]
        for _ in range(20):
            k = int(pool[int(rng.integers(len(pool)))])
            if count(k) < 3:
                cards[cards.index(v)] = k
                return True
        return False

    for _ in range(4 * len(cards)):
        high = [k for k in cards if _is_high(cset[k]) and k not in keep]
        if sum(_is_high(cset[k]) for k in cards) <= max_high or not high or not len(low_pool) or not swap(high, low_pool):
            break
    for _ in range(4 * len(cards)):
        mons = [k for k in cards if cset[k]["ctype"] & TYPE_MONSTER and k not in keep]
        if sum(bool(cset[k]["ctype"] & TYPE_MONSTER) for k in cards) <= max_mon or not mons or not len(st_pool) \
                or not swap(mons, st_pool):
            break
    return cards


def _xsup(e, deck):
    """Extra Deck monster summonable with the cards of the main deck (deck: cards, copies included)."""
    kind, r = e["recipe"]
    ok = lambda f, c: _fmatch(f, c) and ("code" not in f or c["code"] == f["code"])
    if kind == "fusion":
        return all(any(ok(f, c) for c in deck) for f in r)
    if kind == "xyz":
        return sum(c["lv"] == e["lv"] and ok(r[0], c) for c in deck) > r[1]
    if kind == "synchro":
        tl = {c["lv"] for c in deck if c["ctype"] & TYPE_TUNER and ok(r[0], c)}
        nl = {c["lv"] for c in deck if not c["ctype"] & TYPE_TUNER and ok(r[1], c)}
        return any(t + k * n == e["lv"] for t in tl for n in nl for k in (1, 2))
    return True


def structured_decks(cset, rng, b, d=40, x=15, p_rand=0.2, p_sup=0.6, p_arch=0.6, p_lv=0.5, caps=False, profile=False):
    """Structured decks [b, 2, d] / [b, 2, x], at most 3 copies per card. A share p_rand is still drawn at random.
    The others are built around a theme, like real decks:
    - archetype core (probability p_arch, banks Gen(arch > 0)): 8 to 14 cards of the archetype in 1 to 3
      copies, and Extra Deck monsters of the archetype;
    - 0 to 2 mechanic packs (1 to 2 without an archetype) among: Pendulum Monsters, Ritual pairs, Fusion with its
      named materials and its Spell, monsters of the same Level with Xyz of that Rank, Tuners;
    - one or two dominant Levels (Level 4 favoured, like real decks; Rank of the Xyz pack if there is one): a
      share p_lv of the filler is a monster of these Levels;
    - support: a share p_sup of the filler matches a qualified filter that still has no target in the deck
      (attribute / type / archetype / Level of the searches, targets, costs), and a share p_sup of the Extra Deck is
      summonable with the main deck (_xsup).
    profile: composition profile drawn at random per deck, in a wide distribution (10 to 32 monsters, share of
      Level 5 or higher monsters drawn from a Beta(1, 2.5) law, from 0 to ~80%), then made playable by the support:
      a deck loaded with big monsters receives cards that Summon them from the hand or the GY, at the rate
      of one per three big monsters that do not Summon themselves (_apply_profile). No strategy is
      named: the variety comes from the draw, the coherence from the support.
    caps (not recommended: makes the decks uniform): composition caps, drawn per deck like in real decks: 16 to 24
    monsters and 2 to 6 monsters of Level 5 or more (excluding Ritual Monsters and monsters with a hand Summon
    procedure). Without caps, a deck has ~29 monsters, ~10 of them high-Level, dead in hand in 60 to 70% of cases
    (train/deck_quality.py). The excess cards are replaced by monsters of Level 4 or less, then by Spells and Traps;
    the mechanic packs (Ritual, Fusion, Pendulum, Xyz, Tuners) are not touched."""
    idx = np.arange(len(cset))
    ext_m = np.array([is_extra(c) for c in cset])
    main, ext = idx[~ext_m], idx[ext_m]
    ct = np.array([c["ctype"] for c in cset])
    k0 = lambda c: c["effects"][0]["kind"] if c["effects"] else ""
    pend = idx[~ext_m & (ct & TYPE_PENDULUM != 0)]
    tuner = idx[~ext_m & (ct & TYPE_TUNER != 0) & (ct & TYPE_MONSTER != 0)]
    rit = [(i, c["effects"][0]["code"] - CODE0) for i, c in enumerate(cset) if k0(c) == "ritual"]
    fspell = idx[[k0(c) == "fusion_spell" for c in cset]]
    fus = [(i, [r["code"] - CODE0 for r in c["recipe"][1] if "code" in r]) for i, c in enumerate(cset)
           if c.get("recipe") and c["recipe"][0] == "fusion"]
    fus = [(i, m) for i, m in fus if m]
    lvl = np.array([c.get("lv", 0) for c in cset])
    xyz = idx[ext_m & (ct & TYPE_XYZ != 0)]
    nmon = idx[~ext_m & (ct & TYPE_MONSTER != 0) & (ct & TYPE_RITUAL == 0)]
    xyz_lv = [L for L in sorted(set(lvl[xyz])) if (lvl[nmon] == L).sum() >= 3]
    lv_w = {L: (lvl[nmon] == L).sum() * (6.0 if L == 4 else 1.0) for L in sorted(set(lvl[nmon]))}
    arch_of = [c.get("arch") or (c["setc"][0] if c["setc"] else None) for c in cset]
    arch_main = {A: np.array([i for i in main if arch_of[i] == A]) for A in SETS}
    arch_ext = {A: np.array([i for i in ext if arch_of[i] == A]) for A in SETS}
    archs = [A for A in SETS if len(arch_main[A]) >= 10 and any(cset[i].get("arch") for i in arch_main[A])]
    packs = [k for k, ok in (("pend", len(pend) >= 2), ("rit", bool(rit)), ("fus", bool(fus) and len(fspell) > 0),
                             ("xyz", bool(xyz_lv)), ("tuner", len(tuner) > 0)) if ok]
    cop = lambda: int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
    filt = [_own_filters(c["effects"], []) for c in cset]       # filters of each card
    sup = {}                                                    # id(filter) -> main deck cards that match it
    for fs in filt:
        for f in fs:
            sup[id(f)] = np.array([i for i in main if _fmatch(f, cset[i])], np.int64)

    low_pool = nmon[lvl[nmon] <= 4]
    st_pool = idx[~ext_m & (ct & TYPE_MONSTER == 0)]
    if profile:
        high_pool = np.array([i for i in nmon if _is_high(cset[i])], np.int64)
        enablers = np.array([i for i in main if _enables_high(cset[i])], np.int64)
    else:
        high_pool = enablers = None

    def one():
        if rng.random() < p_rand:
            rd, rx = _draw3(main, rng, (), d), _draw3(ext, rng, (), x)
            return (np.array(_apply_caps(cset, rng, rd, set(), low_pool, st_pool)) if caps else rd), rx
        cards, extra, Ls = [], [], []
        A = archs[rng.integers(len(archs))] if archs and rng.random() < p_arch else None
        if A is not None:                          # archetype core
            pool = arch_main[A]
            for c in rng.choice(pool, min(len(pool), int(rng.integers(8, 15))), replace=False):
                cards += [c] * cop()
        npk = int(rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25]) if A is not None else rng.choice([1, 2], p=[0.6, 0.4]))
        for k in rng.choice(packs, min(npk, len(packs)), replace=False) if packs else []:
            if k == "pend":
                for c in rng.choice(pend, min(len(pend), int(rng.integers(3, 6))), replace=False):
                    cards += [c] * int(rng.integers(2, 4))
            elif k == "rit":
                for j in rng.choice(len(rit), min(len(rit), int(rng.integers(1, 3))), replace=False):
                    cards += [rit[j][0]] * 2 + [rit[j][1]] * int(rng.integers(1, 3))
            elif k == "fus":
                for j in rng.choice(len(fus), min(len(fus), int(rng.integers(1, 3))), replace=False):
                    extra += [fus[j][0]] * 2
                    for mt in fus[j][1]:
                        cards += [mt] * 2
                cards += [rng.choice(fspell)] * int(rng.integers(2, 4))
            elif k == "xyz":                       # monsters of the same Level + Xyz of that Rank
                L = int(rng.choice(xyz_lv))
                Ls.append(L)
                same = nmon[lvl[nmon] == L]
                for c in rng.choice(same, min(len(same), int(rng.integers(3, 5))), replace=False):
                    cards += [c] * 2
                xs = xyz[lvl[xyz] == L]
                extra += list(rng.choice(xs, min(len(xs), 3), replace=False))
            elif k == "tuner":
                for c in rng.choice(tuner, min(len(tuner), int(rng.integers(2, 4))), replace=False):
                    cards += [c] * 2
        if not Ls or rng.random() < 0.4:           # dominant Levels
            ks = list(lv_w)
            w = np.array([lv_w[k] for k in ks], float)
            Ls += [int(L) for L in rng.choice(ks, int(rng.choice([1, 2])), replace=False, p=w / w.sum())]
        lv_pool = nmon[np.isin(lvl[nmon], Ls)]
        if A is not None:
            la = np.array([i for i in arch_main[A] if cset[i]["ctype"] & TYPE_MONSTER and lvl[i] in Ls])
            lv_pool = la if len(la) >= 3 and rng.random() < 0.5 else lv_pool
        cards = _cap3(cards)[:d - 10]              # at least 10 filler cards
        keep = set(cards) if A is None else set(cards) - set(arch_main[A].tolist())   # mechanic packs: untouchable
        while len(cards) < d:
            open_ = [f for c in set(cards) for f in filt[c]
                     if len(sup[id(f)]) and not any(_fmatch(f, cset[k]) for k in cards if k != c)]
            r = rng.random()
            if open_ and r < p_sup:
                f = open_[rng.integers(len(open_))]
                cards += [rng.choice(sup[id(f)])] * cop()
            elif len(lv_pool) and r < p_sup + p_lv * (1 - p_sup):
                cards += [rng.choice(lv_pool)] * cop()
            elif A is not None and rng.random() < 0.3:
                cards += [rng.choice(arch_main[A])] * cop()
            else:
                cards += [rng.choice(main)] * cop()
            cards = _cap3(cards)
        cards = cards[:d]
        if caps:
            cards = _apply_caps(cset, rng, cards, keep, low_pool, st_pool)
        if profile:
            cards = _apply_profile(cset, rng, cards, keep, low_pool, high_pool, st_pool, enablers, A, arch_main)
        deck = [cset[k] for k in cards]
        xs = [k for k in ext if cset[k]["recipe"][0] != "link" and _xsup(cset[k], deck)]
        if A is not None and len(arch_ext[A]):
            xa = [k for k in arch_ext[A] if cset[k]["recipe"][0] == "link" or _xsup(cset[k], deck)]
            if xa:
                extra += list(rng.choice(xa, min(len(xa), int(rng.integers(2, 6))), replace=False))
        extra = _cap3(extra)[:x]
        while len(extra) < x:
            if xs and rng.random() < p_sup:           # support: Extra Deck Monster that can be summoned with this deck
                extra += [rng.choice(xs)] * int(rng.choice([1, 2]))
            else:
                extra += [rng.choice(ext)] * cop()
            extra = _cap3(extra)
        return np.array(cards[:d]), np.array(extra[:x])

    out = [[one() for _ in range(2)] for _ in range(b)]
    return (np.array([[p[0] for p in r] for r in out], np.int64), np.array([[p[1] for p in r] for r in out], np.int64))


# ---------------------------------------------------------------------- text (debugging)

def describe(c):
    t = c["ctype"]
    names = [(TYPE_FUSION, "fusion"), (TYPE_SYNCHRO, "synchro"), (TYPE_XYZ, "xyz"), (TYPE_LINK, "link"),
             (TYPE_RITUAL, "ritual"), (TYPE_PENDULUM, "pendulum"), (TYPE_TUNER, "tuner"), (TYPE_QUICKPLAY, "quick-play"),
             (TYPE_CONTINUOUS, "continuous"), (TYPE_COUNTER, "counter"), (TYPE_EFFECT, "effect"), (TYPE_NORMAL, "normal")]
    kind = "monster" if t & TYPE_MONSTER else "spell" if t & TYPE_SPELL else "trap"
    tags = " ".join(n for b, n in names if t & b)
    head = f"{c['code']} {kind} {tags}"
    if t & TYPE_MONSTER:
        head += f" lv {c['lv']} {c['atk']}/{c['df']} attr {c['attr']:#x} race {c['race']:#x}"
        if c["setc"]:
            head += f" set {c['setc'][0]:#x}"
    if c["recipe"]:
        head += f" recipe {c['recipe']}"
    lines = [head] + [f"  - {e['kind']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k != "kind" and v)
                      for e in c["effects"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Draws a procedural bank and prints it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main", type=int, default=40)
    ap.add_argument("--extra", type=int, default=10)
    a = ap.parse_args()
    cs = Gen(a.seed).make_set(a.main, a.extra)
    for c in cs:
        print(describe(c))
    vals = [sum(value(e) for e in c["effects"] if e["kind"] not in ("pend",)) for c in cs]
    print(f"{len(cs)} cards; mean effect value {np.mean(vals):.2f}, max {np.max(vals):.2f}")
