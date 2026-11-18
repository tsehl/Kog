"""Several networks on the two sets of real-card decks, against the heuristic (and against the random player):
the 4 reference decks (basic play) and the 4 proto-archetype decks (synergies, translated cards). A single
compilation: the two banks have the shapes of the procedural banks. Reference point: the heuristic against the random
player.

Usage: python train/real_compare.py CKPT [CKPT...] [--b 1024 --bf16]"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import jax  # noqa: E402
import optax  # noqa: E402

import bench_eval  # noqa: E402
import model  # noqa: E402
import ppo  # noqa: E402
import realcards  # noqa: E402

paths = []
while len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    paths.append(sys.argv.pop(1))
a = ppo.parse()
model.set_compute(a.bf16)
env = ppo.Env(a.b, a.frac)
fns = ppo.make_fns(a, optax.adam(1e-4))
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b}", flush=True)
sets = [("reference", realcards.PATH), ("proto", realcards.PATH_PROTO)]
for label, p in sets:
    ppo._REAL["v"] = realcards.load(p)
    rbank, names, main_, extra = ppo._REAL["v"]
    nd = len(names)
    cfg = np.array([(i, j, s) for i in range(nd) for j in range(nd) for s in (0, 1)])[np.arange(env.b) % (2 * nd * nd)]
    p0 = np.where(cfg[:, 2] == 0, cfg[:, 0], cfg[:, 1])
    p1 = np.where(cfg[:, 2] == 0, cfg[:, 1], cfg[:, 0])
    t0 = time.time()
    ev = ppo.summary(*bench_eval.heur_vs_random(a, env, rbank, np.stack([main_[p0], main_[p1]], 1),
                                                np.stack([extra[p0], extra[p1]], 1), cfg[:, 2], jax.random.PRNGKey(1)))
    print(f"[{label}] heuristic vs random ({time.time() - t0:.0f} s) : {ppo.fmt(ev)}", flush=True)
for path in paths:
    params = ppo.load_params(path)
    for label, p in sets:
        ppo._REAL["v"] = realcards.load(p)
        for kind, opp in ((0, "random"), (2, "heuristic")):
            t0 = time.time()
            ev = ppo.real_bench(a, env, fns, params, jax.random.PRNGKey(5 + kind), kind, pre="r")
            print(f"[{label}] {os.path.basename(path)} vs {opp} ({time.time() - t0:.0f} s) : {ppo.fmt(ev)}", flush=True)
print("END", flush=True)
