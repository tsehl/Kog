"""Shape of the rollout and end-of-game signal: what share of the network's decisions sees a real reward?

Replays the rollout loop of train/ppo.py (no update, fixed network on both sides) for several shapes (calls per
iteration), with the same bank change every --bank-every iterations. For each decision of the network:
- does its game end (and in the same rollout as the decision: the real reward then enters its GAE advantage;
  otherwise the advantage rests on the value estimated at the end of the segment);
- or is it abandoned at the bank change (never any reward for any of its decisions).
Only complete banks count (the last one, cut by the end of the simulation, is ignored).

The result depends on B only through the turnover of the pool of new games: a small B on CPU is enough.

Usage: python bench/rollout_cut.py CKPT [--b 64] [--shapes 16,64] [--banks 3] [--bank-every 10]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("train", "engine", "gen")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import model  # noqa: E402
import ppo  # noqa: E402

path = sys.argv.pop(1)
shapes, nbanks = [16, 64], 3
for flag in ("--shapes", "--banks"):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if flag == "--shapes":
            shapes = [int(x) for x in v.split(",")]
        else:
            nbanks = int(v)
a = ppo.parse()
model.set_compute(False)
params = ppo.load_params(path)
fns = ppo.make_fns(a, optax.adam(1e-4))
act_step, enc = fns[2], fns[5]
env = ppo.Env(a.b, a.frac, "structured")
B = a.b
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={B} ; bank every {a.bank_every} iterations ; {path}",
      flush=True)
rows = []
for steps in shapes:
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    side = jnp.full(B, -1, jnp.int32)
    gid = np.arange(B)
    nxt = B
    status = {}                   # gid -> ("done", it, t) or ("cut",)
    decs = []                     # (gid, it, t)
    bank_of_gid = {}
    cur_bank = -1
    for it in range(nbanks * a.bank_every):
        if it % a.bank_every == 0:
            if it:
                for g in gid:     # games in progress abandoned at the bank change
                    status.setdefault(int(g), ("cut",))
                gid = np.arange(nxt, nxt + B)
                nxt += B
            cur_bank += 1
            cset, bank = ppo.bank_of(1000 + cur_bank, a, 0, a.arch)
            tok, msk = model.card_tokens(bank)
            key, k = jax.random.split(key)
            s, m = env.games(bank, cset, rng, k)
            f = jnp.full(B, -1, jnp.int32)
            cemb = enc(params, tok, msk)
            opp0 = (params, cemb, jnp.asarray(0))
        for g in gid:
            bank_of_gid.setdefault(int(g), cur_bank)
        key, k = jax.random.split(key)
        pool, pm = env.games(bank, cset, rng, k)
        used = jnp.zeros(B, bool)
        for t in range(steps):
            key, k = jax.random.split(key)
            used0 = np.asarray(used)
            s, m, f, used, tr, d, w = act_step(params, cemb, bank, s, m, f, k, side, pool, pm, used, opp0, jnp.asarray(False))
            need, d = np.asarray(tr["need"]), np.asarray(d)
            for i in np.nonzero(need)[0]:
                decs.append((int(gid[i]), it, t))
            for i in np.nonzero(d)[0]:
                status.setdefault(int(gid[i]), ("done", it, t))
                if not used0[i]:  # replaced by a game from the pool
                    gid[i] = nxt
                    bank_of_gid[nxt] = cur_bank
                    nxt += 1
            if t % 8 == 7 and float(np.asarray(used).mean()) > 0.25:
                key, k = jax.random.split(key)
                pool, pm = env.games(bank, cset, rng, k)
                used = jnp.zeros(B, bool)
        print(f"  {steps} calls: iteration {it} done", flush=True)
    last = nbanks - 1
    n = fin = same = cut = 0
    lens = []
    first = {}
    for g, it, t in decs:
        if bank_of_gid[g] == last:
            continue
        n += 1
        st = status.get(g, ("cut",))
        if st[0] == "done":
            fin += 1
            same += st[1] == it
        else:
            cut += 1
        first.setdefault(g, (it, t))
    for g, st in status.items():
        if st[0] == "done" and g in first and bank_of_gid[g] != last:
            lens.append((st[1] - first[g][0]) * steps + st[2] - first[g][1] + 1)
    rows.append((steps, n, fin / n, same / n, cut / n, np.mean(lens), len(lens)))
print(f"\n{'calls / iteration':>18s} {'decisions':>10s} {'game finished':>13s} {'end in the same rollout':>25s} "
      f"{'game cut':>14s} {'game length (calls)':>28s}")
for steps, n, fin, same, cut, ln, ng in rows:
    print(f"{steps:18d} {n:10d} {100 * fin:12.1f} % {100 * same:24.1f} % {100 * cut:13.1f} % {ln:20.1f} ({ng} games)")
