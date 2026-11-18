"""Brick audit: which effect characteristics of the benchmark's real cards (train/realbank.npz) does the procedural
generator never produce? Characteristics per effect: kind (K_*), opcodes by part of the program (cost / target /
resolution; the COND_* conditions have no part) with their significant arguments, registers removed, trigger events,
continuous modifiers (MC_*), damage step, per-turn limits, filters by name / archetype.

Usage: python gen/brick_audit.py [--banks 10]"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "train")]
import numpy as np

import core
import cardgen
import realcards

OPS = ("END FILTER SELF REQ REQ_FREE MINATK CHOOSE MOVE SPSUMMON DRAW DAMAGE RECOVER SHUFFLE_HAND COND_CHAIN CHAINCARD "
       "NEGATE PAY_LP EVENT COND_TURN COND_EVP COND_SELF LOCK MODIFY COND_ATK ATTACKER COND_PHASE FUSMAT RITMAT LABEL "
       "COND_LABEL YESNO EQUIP BLOCK").split()
KINDS = "NONE ACTIVATE IGNITION TRIG_F TRIG_O QUICK CONT PROC".split()
PARTS = ("cost", "target", "resolve")
EVS = "SUMMON FLIP_SUMMON SPSUMMON ATTACK TO_GRAVE BATTLE_DESTROYED FLIP PRE_DMG".split()
# significant arguments per opcode (register numbers are assembly details: they are removed)
ARGS = {"MINATK": lambda a, b, c: (c,), "REQ": lambda a, b, c: (c,), "MOVE": lambda a, b, c: (b, c), "SPSUMMON": lambda a, b, c: (b, c), "MODIFY": lambda a, b, c: (b,),
        "EVENT": lambda a, b, c: (b,), "COND_PHASE": lambda a, b, c: (a,), "COND_TURN": lambda a, b, c: (a,),
        "COND_EVP": lambda a, b, c: (a, b), "COND_SELF": lambda a, b, c: (a, b, c), "COND_CHAIN": lambda a, b, c: (a, b),
        "COND_ATK": lambda a, b, c: (a,),
        "FILTER": lambda a, b, c: filter_modes(c)}


def filter_modes(c):
    """FILTER modes without exposing the register numbers used by the assembler."""
    return tuple(name for bit, name in ((7, "within"), (8, "target"), (1 << 20, "copy"),
                                         (1 << 22, "related")) if c & bit)


def features(bank, c):
    """Set of characteristics (strings) of the card of index c."""
    out = set()
    ek, prog, etrig = np.asarray(bank.ekind[c]), np.asarray(bank.prog[c]), np.asarray(bank.etrig[c])
    emod, edmg, eopt = np.asarray(bank.emod[c]), np.asarray(bank.edmg[c]), np.asarray(bank.eopt[c])
    filt = np.asarray(bank.filt[c])
    for e in range(ek.shape[0]):
        if ek[e] == core.K_NONE:
            continue
        k = KINDS[ek[e]]
        out.add(f"kind {k}")
        if k == "CONT":
            out.add(f"continuous MC={int(emod[e, core.M_CODE])}")
        for pt in range(3):
            for op, a1, b1, c1 in prog[e, pt]:
                if op == core.OP_END:
                    continue
                nm = OPS[op]
                part = "condition" if nm.startswith("COND_") else PARTS[pt]   # cost or target: checked at activation
                out.add(f"{part} {nm}" + (f" {ARGS[nm](int(a1), int(b1), int(c1))}" if nm in ARGS else ""))
        for i, ev in enumerate(EVS):
            if etrig[e] >> i & 1:
                out.add(f"trigger {ev}")
        if edmg[e]:
            out.add(f"damage step {int(edmg[e])}")
        if eopt[e]:
            out.add(f"per-turn limit {int(eopt[e])}")
        for f in filt[e]:
            if f[core.F_CODE1] or f[core.F_CODE2]:
                out.add("name filter")
            if f[core.F_SET]:
                out.add("archetype filter")
    if np.asarray(bank.erl[c]):
        out.add("EnableReviveLimit")
    if np.asarray(bank.ldiff[c]):
        out.add("Link: different names")
    if np.asarray(bank.lneed[c]):
        out.add("Link: Link material required")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", type=int, default=10)
    a = ap.parse_args()
    seen = collections.Counter()
    ncards = 0
    for sd in range(a.banks):
        cset = cardgen.Gen(sd).make_set(400, 80)
        bank = cardgen.to_bank(cset)
        for c in range(len(cset)):
            for ft in features(bank, c):
                seen[ft] += 1
        ncards += len(cset)
    rbank, *_ = realcards.load()
    names = realcards.card_names()
    missing = collections.defaultdict(list)
    for c, nm in enumerate(names):
        if not nm:
            continue
        for ft in features(rbank, c):
            if seen[ft] == 0:
                missing[ft].append(nm)
    print(f"{ncards} procedural cards ({a.banks} banks), {sum(1 for n in names if n)} real cards\n")
    print("features of real cards never produced by the generator:")
    for ft, nms in sorted(missing.items(), key=lambda x: -len(x[1])):
        print(f"  {ft:42s} {len(nms):3d}  {', '.join(sorted(set(nms)))}")
    full = [nm for c, nm in enumerate(names) if nm and all(seen[ft] for ft in features(rbank, c))]
    print(f"\nreal cards fully covered: {len(full)} / {sum(1 for n in names if n)}")
    print("\nfeatures rare in training (< 0.2% of cards) but present in real cards:")
    real = collections.Counter(ft for c, nm in enumerate(names) if nm for ft in features(rbank, c))
    for ft, n in sorted(real.items(), key=lambda x: seen[x[0]]):
        if 0 < seen[ft] < 0.002 * ncards:
            print(f"  {ft:42s} generator {seen[ft]:4d} / {ncards}, real cards {n}")


if __name__ == "__main__":
    main()
