"""Evaluation of a saved network: never-seen procedural bank (decks in pairs) and real cards, against the
random player and against the heuristic; plus the heuristic-against-random references.

Usage: python train/bench_eval.py train/ckpt/run2_it200_f16.pkl.gz [--b 1024] [--d 128 ...] (same options as ppo.py);
BENCH_GREEDY=1 adds the deterministic network (argmax)."""
import os
import sys
import time

import numpy as np

import ppo
import jax
import jax.numpy as jnp
import optax

import cardgen
import core
import model
import realcards


def heur_vs_random(a, env, bank, d, x, side, key):
    """Heuristic (player side) against random: (finished, result) like ppo.play_eval."""
    step = env.step
    s, m = env.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    sd = np.asarray(side)
    fin, res = np.zeros(env.b, bool), np.zeros(env.b, int)
    pick = jax.jit(lambda bank, s, m, f, k, side: jnp.where(
        s.dec_player == side, ppo.heuristic(bank, s, m, f, k),
        jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))))
    side = jnp.asarray(side)
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        s, m, f = step(bank, s, pick(bank, s, m, f, k, side))
        dn, w = np.asarray(s.done), np.asarray(s.winner)
        new = dn & ~fin
        res[new] = (w[new] == sd[new]).astype(int) - (w[new] == 1 - sd[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fin, res


def main():
    path = sys.argv[1]
    sys.argv = sys.argv[:1] + sys.argv[2:]
    a = ppo.parse()
    model.set_compute(a.bf16)
    params = ppo.load_params(path)
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    key = jax.random.PRNGKey(7)
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; {path}", flush=True)
    rows = []

    def run(name, fn):
        t0 = time.time()
        ev = fn()
        rows.append((name, ev))
        print(f"{name} ({time.time() - t0:.0f} s) : {ppo.fmt(ev)}", flush=True)

    ks = jax.random.split(key, 8)
    # references: heuristic against random
    cset, pbank = ppo.bank_of(a.eval_bank, a)
    d, x = cardgen.random_decks(cset, np.random.default_rng(12345), env.b // 4)
    D, X, side = ppo.paired(d, x)
    run("procedural, heuristic vs random", lambda: ppo.summary(*heur_vs_random(a, env, pbank, D, X, side, ks[0])))
    rbank, names, main_, extra = realcards.load()
    nd = len(names)
    cfg = np.array([(i, j, s) for i in range(nd) for j in range(nd) for s in (0, 1)])[np.arange(env.b) % (2 * nd * nd)]
    p0 = np.where(cfg[:, 2] == 0, cfg[:, 0], cfg[:, 1])
    p1 = np.where(cfg[:, 2] == 0, cfg[:, 1], cfg[:, 0])
    run("real cards, heuristic vs random", lambda: ppo.summary(*heur_vs_random(
        a, env, rbank, np.stack([main_[p0], main_[p1]], 1), np.stack([extra[p0], extra[p1]], 1), cfg[:, 2], ks[1])))
    # the network
    run("procedural, network vs random", lambda: ppo.evaluate(a, env, fns, params, ks[2], 0))
    run("procedural, network vs heuristic", lambda: ppo.evaluate(a, env, fns, params, ks[3], 2))
    run("structured procedural (archetypes), network vs heuristic",
        lambda: ppo.evaluate(a, env, fns, params, ks[6], 2, struct=True))
    run("real cards, network vs random", lambda: ppo.real_bench(a, env, fns, params, ks[4], 0))
    run("real cards, network vs heuristic", lambda: ppo.real_bench(a, env, fns, params, ks[5], 2))
    if os.environ.get("BENCH_GREEDY"):   # deterministic policy (argmax)
        run("procedural, deterministic network vs random", lambda: ppo.evaluate(a, env, fns, params, ks[2], 0, greedy=True))
        run("procedural, deterministic network vs heuristic",
            lambda: ppo.evaluate(a, env, fns, params, ks[3], 2, greedy=True))
        run("real cards, deterministic network vs heuristic",
            lambda: ppo.real_bench(a, env, fns, params, ks[5], 2, greedy=True))
    print("END", flush=True)


if __name__ == "__main__":
    main()
