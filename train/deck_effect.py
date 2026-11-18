"""What share of the result of a game is decided by the decks?

The network plays against itself (same parameters on both sides, sampled actions). Each pair of decks (A, B) is
played 8 times: A first or second, different shuffles. If the same deck almost always wins, the win / loss signal of
a training game mostly says which deck was stronger, not who played better.

Decomposition: the variance of the result of a game (A wins or not) = variance between pairs of the probability that
A wins (share of the decks) + variance within a pair (shuffle, action sampling, play). The share of the decks is
estimated without bias (intra-class correlation); interval by bootstrap over the pairs. The first player's advantage
is measured separately.

Usage: python train/deck_effect.py CKPT [--b 1024 --eval-steps 500 --bf16]   (DE_BANKS=4 evaluation banks)"""
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

path = sys.argv.pop(1)
a = ppo.parse()
model.set_compute(a.bf16)
params = ppo.load_params(path)
env = ppo.Env(a.b, a.frac)
fns = ppo.make_fns(a, optax.adam(1e-4))
R = 8
P = a.b // R
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; {P} pairs × {R} games per bank ; {path}", flush=True)


def icc(k, n):
    """k [pairs] wins of A, n [pairs] finished games (>= 2). Share of the variance due to the pair."""
    p = k / n
    within = (p * (1 - p) * n / (n - 1)).mean()          # unbiased estimator of E[p (1 - p)]
    pbar = k.sum() / n.sum()
    return 1 - within / (pbar * (1 - pbar))


rows = {}
for kind in ("structured", "random"):
    K, N, first, tot, sweep = [], [], 0, 0, []
    for bi in range(int(os.environ.get("DE_BANKS", "4"))):
        cset, bank = ppo.bank_of(a.eval_bank + bi, a, arch=a.arch or 0.5)
        rng = np.random.default_rng(777 + bi)
        d, x = (cardgen.structured_decks if kind == "structured" else cardgen.random_decks)(cset, rng, P)
        D, X, side = ppo.paired(np.concatenate([d, d]), np.concatenate([x, x]))        # 2 x 4 games per pair
        fin, res = ppo.play_eval(a, env, fns, params, bank, D, X, side, jax.random.PRNGKey(31 + bi), 1)
        g = np.arange(len(side))
        a_player = (g % 4) // 2                         # paired: games 0-1 = (A, B), 2-3 = (B, A)
        winner = np.where(res == 1, side, 1 - side)
        ok = fin & (res != 0)
        a_win = ok & (winner == a_player)
        pair = (g // 4) % P
        k, n = np.bincount(pair, a_win, P), np.bincount(pair, ok, P)
        keep = n >= 2
        K.append(k[keep]); N.append(n[keep])
        first += int((ok & (winner == 0)).sum()); tot += int(ok.sum())
        sweep.append(((k[keep] == 0) | (k[keep] == n[keep])).mean())
        print(f"  {kind} decks, bank {bi}: {int(ok.sum())} games decided out of {len(g)}, deck share "
              f"{100 * icc(k[keep], n[keep]):.0f} %", flush=True)
    K, N = np.concatenate(K), np.concatenate(N)
    bs = np.random.default_rng(0)
    boot = [icc(K[i], N[i]) for i in (bs.integers(0, len(K), len(K)) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows[kind] = (icc(K, N), lo, hi, first / tot, float(np.mean(sweep)), len(K))
print("\ndecks        pairs    share of outcome decided by the decks   a deck wins all its games   first player wins")
for kind, (v, lo, hi, f, sw, n) in rows.items():
    print(f"{kind:12s} {n:6d}   {100 * v:5.1f} %  [{100 * lo:.1f} ; {100 * hi:.1f}]                 {100 * sw:5.1f} % of pairs"
          f"            {100 * f:5.1f} %")
print("(reference: with 8 games and two decks of equal strength, one deck wins everything in 0.8% of pairs)")
