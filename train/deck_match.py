"""Pairing decks of similar strength: does it make the games closer?

A pool of P structured decks per bank. The network plays against itself.
1. Ranking: rounds of games between randomly drawn pairs give a strength per deck (Bradley-Terry).
2. Test: 128 pairs played 8 times each (like train/deck_effect.py), formed either at random or between ranking
   neighbors, with a ranking learned on few games (1 round) or more (all rounds).
The share of the result decided by the decks and the share of pairs where one deck wins its 8 games are compared.

Usage: python train/deck_match.py CKPT [--b 1024 --eval-steps 500 --bf16]   (DM_BANKS=2, DM_POOL=256, DM_ROUNDS=3)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "gen")]
import jax  # noqa: E402
import optax  # noqa: E402

import cardgen  # noqa: E402
import model  # noqa: E402
import ppo  # noqa: E402


def bradley_terry(i, j, win_i, n, iters=200):
    """Strengths (log) of n decks from games i against j (win_i: i won). A fictitious draw against a deck of
    strength 1 regularizes the decks that are rarely played."""
    w = np.bincount(i, win_i, n) + np.bincount(j, ~win_i, n) + 0.5
    p = np.ones(n)
    for _ in range(iters):
        d = 1.0 / (p[i] + p[j])
        p = w / (np.bincount(i, d, n) + np.bincount(j, d, n) + 1.0 / (p + 1.0))
        p /= np.exp(np.log(p).mean())
    return np.log(p)


def icc(k, n):
    p = k / n
    pbar = k.sum() / n.sum()
    return 1 - (p * (1 - p) * n / (n - 1)).mean() / (pbar * (1 - pbar))


def main():
    path = sys.argv.pop(1)
    a = ppo.parse()
    model.set_compute(a.bf16)
    params = ppo.load_params(path)
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    P, R = int(os.environ.get("DM_POOL", "256")), int(os.environ.get("DM_ROUNDS", "3"))
    NP = a.b // 8
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; pool of {P} decks ; {path}", flush=True)
    res_all = {}
    for bi in range(int(os.environ.get("DM_BANKS", "2"))):
        cset, bank = ppo.bank_of(a.eval_bank + bi, a, arch=a.arch or 0.5)
        rng = np.random.default_rng(900 + bi)
        d, x = cardgen.structured_decks(cset, rng, (P + 1) // 2)
        pd, px = d.reshape(-1, d.shape[-1])[:P], x.reshape(-1, x.shape[-1])[:P]
        key = jax.random.PRNGKey(60 + bi)
        gi, gj, gw, ratings = [], [], [], {}
        for r in range(R):
            i = rng.integers(0, P, a.b)
            j = (i + rng.integers(1, P, a.b)) % P
            key, k = jax.random.split(key)
            fin, res = ppo.play_eval(a, env, fns, params, bank, np.stack([pd[i], pd[j]], 1), np.stack([px[i], px[j]], 1),
                                     np.zeros(a.b, int), k, 1)
            ok = fin & (res != 0)
            gi.append(i[ok]); gj.append(j[ok]); gw.append(res[ok] == 1)
            ratings[r + 1] = bradley_terry(np.concatenate(gi), np.concatenate(gj), np.concatenate(gw), P)
            print(f"  bank {bi}, ranking round {r + 1}: {2 * sum(len(v) for v in gi) / P:.0f} games per deck, "
                  f"strength std {ratings[r + 1].std():.2f}", flush=True)

        def test(name, pairs):
            dd, xx = np.stack([pd[pairs[:, 0]], pd[pairs[:, 1]]], 1), np.stack([px[pairs[:, 0]], px[pairs[:, 1]]], 1)
            D, X, side = ppo.paired(np.concatenate([dd, dd]), np.concatenate([xx, xx]))
            nonlocal key
            key, k = jax.random.split(key)
            fin, res = ppo.play_eval(a, env, fns, params, bank, D, X, side, k, 1)
            g = np.arange(len(side))
            winner = np.where(res == 1, side, 1 - side)
            ok = fin & (res != 0)
            pair = (g // 4) % len(pairs)
            kk, nn = np.bincount(pair, ok & (winner == (g % 4) // 2), len(pairs)), np.bincount(pair, ok, len(pairs))
            keep = nn >= 2
            res_all.setdefault(name, []).append((kk[keep], nn[keep]))
            print(f"  bank {bi}, {name}: deck share {100 * icc(kk[keep], nn[keep]):.0f} %", flush=True)

        perm = rng.permutation(P)[:2 * NP].reshape(NP, 2)
        test("random pairs", perm)
        for r in sorted({1, R}):
            order = np.argsort(ratings[r])
            # ranking neighbors, with a little randomness: each deck plays a deck at most 3 ranks away
            idx = order[:2 * (P // 2)].reshape(-1, 2)
            test(f"neighbours, ranking over {r} round(s)", idx[rng.permutation(len(idx))[:NP]])
    print("\nmatching                                  pairs    share decided by the decks   a deck wins its 8 games")
    bs = np.random.default_rng(0)
    for name, v in res_all.items():
        K, N = np.concatenate([q[0] for q in v]), np.concatenate([q[1] for q in v])
        boot = [icc(K[i], N[i]) for i in (bs.integers(0, len(K), len(K)) for _ in range(1000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{name:40s} {len(K):6d}   {100 * icc(K, N):5.1f} %  [{100 * lo:.1f} ; {100 * hi:.1f}]        "
              f"{100 * ((K == 0) | (K == N)).mean():5.1f} % of pairs")


if __name__ == "__main__":
    main()
