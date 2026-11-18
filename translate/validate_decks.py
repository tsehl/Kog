"""The proto-archetype decks (train/realcards.py, PROTO_DECKS) against the ygopro-core oracle: each game pits two
decks against each other (mirrors included), played at random on both sides by the engine and by the oracle, compared
decision by decision. The evaluation decks serve as a reference only if these games are identical.

Usage: uv run --with luaparser --with jax python translate/validate_decks.py [--rounds 1] [--seed0 0]
        (--rounds 1: the 16 pairings once, each deck shuffled according to the seed)"""
import argparse
import collections
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "oracle"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "gen")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--only", default="", help="restrict to pairings containing this deck")
    ap.add_argument("--pair", default="", help="replay a single pairing x,y (with --seed0)")
    a = ap.parse_args()
    import jax
    import compare
    import core
    import driver
    import realcards
    db = driver.CardDB()
    by_name = {}
    for c, nm in db.names.items():
        if c in db.rows and db.rows[c][1] == 0:
            by_name.setdefault(nm, c)
    decks = {nm: ([by_name[n] for n in realcards._expand(d["main"])], [by_name[n] for n in realcards._expand(d["extra"])])
             for nm, d in realcards.PROTO_DECKS.items()}
    codes = sorted({c for m, x in decks.values() for c in m + x})
    import cards
    tr = realcards.translated(set(codes))
    bank, cmap = realcards.proto_bank(db, codes, tr)
    codes = [c for c in codes if c not in cards.CARDS]     # oracle descriptions: translated cards only
    compare.DESC.clear()
    compare.DESC.update({c: tr[c][2] for c in codes})
    compare.EFFECT_EVENTS.clear()
    compare.EFFECT_EVENTS.update({c: [e[1] for e in tr[c][0]] for c in codes})
    compare.EFFECT_RANGES.clear()
    compare.EFFECT_RANGES.update({c: [e[3] for e in tr[c][0]] for c in codes})
    compare.EFFECT_KINDS.clear()
    compare.EFFECT_KINDS.update({c: [e[0] for e in tr[c][0]] for c in codes})
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    names = sorted(decks)
    pairs = [(x, y) for x in names for y in names if not a.only or a.only in (x, y)]
    if a.pair:
        pairs = [tuple(a.pair.split(","))]
    ok, n, errs = 0, 0, collections.Counter()
    activated = collections.Counter()
    seed = a.seed0
    for r in range(a.rounds):
        for x, y in pairs:
            rng = random.Random(1000 + seed)
            d = [list(decks[x][0]), list(decks[y][0])]
            for z in d:
                rng.shuffle(z)
            t0 = time.time()
            try:
                out = compare.play(db, bank, cmap, seed, step, legal, mode="tr", given=(d, [list(decks[x][1]), list(decks[y][1])]),
                                   extra_capacity=15)
            except Exception as e:  # noqa: BLE001
                out = dict(ok=False, status="error", err=f"{type(e).__name__}: {e}")
            n += 1
            ok += out["ok"]
            for ev in out.get("coverage", []):
                if out["ok"] and ev["stage"] in ("activated", "summoned"):
                    activated[ev["stage"]] += ev["count"]
            summary = {k: v for k, v in out.items() if k not in ("coverage", "trace", "oracle_log")}
            print(f"game {seed} {x} vs {y} ({time.time() - t0:.0f} s) {str(summary)[:400]}", flush=True)
            if not out["ok"]:
                errs[str(out.get("err", ""))[:80]] += 1
                print("   difference:", str(out.get("err", ""))[:4000], flush=True)
            seed += 1
    print(f"{ok}/{n} identical games; in these games: {activated['activated']} activations, "
          f"{activated['summoned']} Summons recorded")
    for k, v in errs.most_common():
        print(f"  {v} × {k}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
