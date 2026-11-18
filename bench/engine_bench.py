"""Engine throughput under step_batch (policy decisions per second), random policy.

Usage: python engine_bench.py ENGINE_DIR [--b 1024,4096] [--frac 0.05] [--calls 60] [--old-cards] [--extra X]
       [--compile-only]
ENGINE_DIR: directory containing core.py and cards.py (current engine, or an old engine extracted from git).
--old-cards: the effect cards of the decks are drawn among the first hand-written cards, with no continuous effects or
Extra Deck (isolates the engine cost).
A decision = an action chosen by the policy (the forced actions returned by step_batch are not counted).
Finished games are replaced by new games (pre-generated pool), to keep the batch full.
--compile-only: compile without measuring. Several engines on the same GPU: run the --compile-only runs in parallel
(XLA_PYTHON_CLIENT_PREALLOCATE=false), then the measurements one by one, which read the cache (YGO_JAX_CACHE).
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("--b", default="1024,4096")
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--calls", type=int, default=60)
ap.add_argument("--old-cards", action="store_true")
ap.add_argument("--compile-only", action="store_true", help="compile only (fills the YGO_JAX_CACHE cache of core.py)")
ap.add_argument("--extra", type=int, default=0, help="Extra Deck of X Fusion Monsters per player (cards.test_decks)")
ap.add_argument("--bank-arg", action="store_true", help="bank passed as an argument of step_batch (as in train/ppo.py) "
                "instead of a compile-time constant")
ap.add_argument("--proc", type=int, default=-1, help="procedural bank of seed X (gen/cardgen.py, 400 + 80 cards, "
                "random decks of 40 + 15) instead of the real cards")
args = ap.parse_args()
sys.path.insert(0, args.engine)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import cards  # noqa: E402
import core  # noqa: E402

D, C = 40, 670
key = jax.random.PRNGKey(0)
if args.proc >= 0:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen"))
    import cardgen  # noqa: E402
    cset = cardgen.Gen(args.proc).make_set(400, 80)
    bank = cardgen.to_bank(cset)
    args.extra = args.extra or 15
else:
    bank = cards.test_bank(key, C)
codes = np.asarray(bank.code)
new = set()
for name in ("J4", "J4B", "J4C", "J4D"):
    new |= getattr(cards, name, set())
fus = set(getattr(cards, "FUSION", {})) | set(getattr(cards, "SYNCHRO", {})) | set(getattr(cards, "XYZ", {})) | set(getattr(cards, "LINK", {}))  # Extra Deck: never in the main deck
eff = np.array([i for i in range(C, len(codes)) if not (args.old_cards and int(codes[i]) in new) and int(codes[i]) not in fus])
print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; bank {len(codes)} cards, {len(eff)} drawable effect cards",
      flush=True)


def decks(k, b):
    if args.proc >= 0:
        d, x = cardgen.random_decks(cset, np.random.default_rng(int(jax.random.randint(k, (), 0, 1 << 30))), b, D, args.extra)
        return jnp.asarray(d), jnp.asarray(x)
    if args.extra:
        return cards.test_decks(k, bank, b, D, C, 0.3, args.extra)
    k1, k2, k3 = jax.random.split(k, 3)
    normal = jax.random.randint(k1, (b, 2, D), 0, C)
    effc = jnp.asarray(eff)[jax.random.randint(k2, (b, 2, D), 0, len(eff))]
    return jnp.where(jax.random.uniform(k3, (b, 2, D)) < 0.3, effc, normal)  # 30 % effect cards


def run(b):
    if args.extra:
        ng0 = jax.jit(jax.vmap(lambda d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])))
        ng = lambda dx, k: ng0(dx[0], dx[1], k)
    else:
        ng = jax.jit(jax.vmap(lambda d, k: core.new_game(bank, d[0], d[1], k)))
    legal = jax.jit(jax.vmap(lambda s: core.legal(bank, s)))
    if args.bank_arg:
        sb0 = jax.jit(lambda bk, s, a: core.step_batch(bk, s, a, args.frac))
        sb = lambda s, a: sb0(bank, s, a)
        sb.lower = lambda *x: sb0.lower(bank, *x)
    else:
        sb = jax.jit(lambda s, a: core.step_batch(bank, s, a, args.frac))
    if args.compile_only:
        t0 = time.time()
        d = decks(jax.random.fold_in(key, 1), b)
        k = jax.random.split(jax.random.fold_in(key, 2), b)
        ng0.lower(d[0], d[1], k).compile() if args.extra else ng.lower(d, k).compile()
        sa = jax.eval_shape(ng, d, k)
        legal.lower(sa).compile()
        sb.lower(sa, jax.ShapeDtypeStruct((b,), jnp.int32)).compile()
        print(f"  B={b}: compiled in {time.time() - t0:.0f} s", flush=True)
        return
    t0 = time.time()
    s = ng(decks(jax.random.fold_in(key, 1), b), jax.random.split(jax.random.fold_in(key, 2), b))
    pool = ng(decks(jax.random.fold_in(key, 3), b), jax.random.split(jax.random.fold_in(key, 4), b))
    m = legal(s)
    pm = legal(pool)
    jax.block_until_ready(pm)
    t_new = time.time() - t0
    f = jnp.full(b, -1, jnp.int32)

    @jax.jit
    def policy(k, m, f, done):
        a = jax.random.categorical(k, jnp.where(m, 0.0, -1e9))
        return jnp.where(f >= 0, f, a).astype(jnp.int32), ((f < 0) & ~done).sum()

    @jax.jit
    def refill(s, m, f, pool, pm):
        d = s.done
        w = lambda x, y: jnp.where(d.reshape(d.shape + (1,) * (x.ndim - 1)), y, x)
        return jax.tree.map(w, s, pool), w(m, pm), jnp.where(d, -1, f)

    kk = jax.random.fold_in(key, 5)
    times, dec = [], 0
    for i in range(args.calls + 3):
        a, nd = policy(jax.random.fold_in(kk, i), m, f, s.done)
        t1 = time.time()
        s, m, f = sb(s, a)
        jax.block_until_ready(f)
        dt = time.time() - t1
        if i == 0:
            print(f"  B={b}: step_batch compilation + 1st call {dt:.0f} s (new_game + legal {t_new:.0f} s)", flush=True)
        if i >= 3:
            times.append(dt)
            dec += int(nd)
        s, m, f = refill(s, m, f, pool, pm)
    tot = sum(times)
    print(f"  B={b}: {dec / tot:,.0f} decisions/s ; {1000 * tot / len(times):.1f} ms/call ; "
          f"{dec / len(times):.0f} decisions/call", flush=True)


for b in map(int, args.b.split(",")):
    run(b)
