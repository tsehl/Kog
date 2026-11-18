"""Engine throughput under step_batch (policy decisions per second), random policy.

Usage: python engine_bench.py ENGINE_DIR [--b 1024,4096] [--frac 0.05] [--calls 60] [--old-cards] [--extra X]
       [--compile-only]
ENGINE_DIR: directory containing core.py and cards.py (current engine, or an old engine extracted from git).
--old-cards: the effect cards of the decks are drawn among the first hand-written cards, with no continuous effects or
Extra Deck (isolates the engine cost).
A decision = an action chosen by the policy (the forced actions returned by step_batch are not counted).
Finished games are replaced by new games (pre-generated pool), to keep the batch full.
--profile DIR: after the measurement, records a profiler trace of --profile-calls calls in DIR (summarise it with
bench/trace_summary.py). The memory and cost analysis of the compiled step_batch is always printed.
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
ap.add_argument("--profile", default="", help="directory for a profiler trace of a few calls after the measurement")
ap.add_argument("--profile-calls", type=int, default=5)
ap.add_argument("--layout", default="default", choices=("default", "auto", "bminor"),
                help="memory layout of the arguments and outputs of step_batch: XLA's default (batch major), chosen by "
                "the compiler (Layout.AUTO), or batch minor (batch axis last in memory: on TPU, the per-game ops fill "
                "the 128 lanes whatever the trailing dimensions). The conversion of the inputs is not timed")
ap.add_argument("--hlo", default="", help="writes the compiled HLO of step_batch to this file (bench/op_census.py "
                "maps the ops of a trace to their engine source lines with it)")
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
    elif args.layout != "default":
        from jax.experimental.layout import Format, Layout
        from jax.sharding import SingleDeviceSharding
        dev = SingleDeviceSharding(jax.devices()[0])

        def fmt(x):
            if args.layout == "auto":
                return Format(Layout.AUTO, dev)
            nd = len(x.shape)
            return Format(Layout(major_to_minor=tuple(range(1, nd)) + (0,) if nd > 1 else (0,)), dev)
        step = lambda s, a: core.step_batch(bank, s, a, args.frac)

        def sb_lower(s, a):
            out = jax.eval_shape(step, s, a)
            return jax.jit(step, in_shardings=jax.tree.map(fmt, (s, a)),
                           out_shardings=jax.tree.map(fmt, out)).lower(s, a)
        sb = None
    else:
        sb = jax.jit(lambda s, a: core.step_batch(bank, s, a, args.frac))
    if args.compile_only:
        t0 = time.time()
        d = decks(jax.random.fold_in(key, 1), b)
        k = jax.random.split(jax.random.fold_in(key, 2), b)
        ng0.lower(d[0], d[1], k).compile() if args.extra else ng.lower(d, k).compile()
        sa = jax.eval_shape(ng, d, k)
        legal.lower(sa).compile()
        (sb.lower if sb is not None else sb_lower)(sa, jax.ShapeDtypeStruct((b,), jnp.int32)).compile()
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

    # compile ahead of time: the compiled object gives the memory / cost analysis, and the calls below use it
    t1 = time.time()
    if args.bank_arg:
        comp0 = sb0.lower(bank, s, f).compile()
        comp = lambda s, a: comp0(bank, s, a)
    elif sb is None:
        comp0 = sb_lower(s, f).compile()
        comp = comp0
        print(f"  B={b}: layout {args.layout}: input layouts of the state "
              f"{sorted({str(x.layout.major_to_minor) for x in jax.tree.leaves(comp0.input_formats[0][0])})}", flush=True)
    else:
        comp0 = sb.lower(s, f).compile()
        comp = comp0
    # inputs in the layouts of the compiled step_batch (converted outside the timed region)
    fin = comp0.input_formats[0] if args.layout != "default" and not args.bank_arg else None
    prep = (lambda s, a: jax.block_until_ready(jax.device_put((s, a), fin))) if fin else (lambda s, a: (s, a))
    if args.hlo:
        with open(args.hlo, "w") as fh:
            fh.write(comp0.as_text())
    print(f"  B={b}: step_batch compiled in {time.time() - t1:.0f} s (new_game + legal {t_new:.0f} s)", flush=True)
    try:
        ma = comp0.memory_analysis()
        mb = lambda x: f"{x / 2**20:.0f} MB"
        print(f"  B={b}: memory: arguments {mb(ma.argument_size_in_bytes)}, outputs {mb(ma.output_size_in_bytes)}, "
              f"temporaries {mb(ma.temp_size_in_bytes)}, generated code {mb(ma.generated_code_size_in_bytes)}",
              flush=True)
    except Exception as e:  # noqa: BLE001 (not available on every backend)
        print(f"  B={b}: memory analysis unavailable ({e})", flush=True)
    try:
        ca = comp0.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        print(f"  B={b}: cost analysis: {ca.get('flops', 0):.3g} flops, {ca.get('bytes accessed', 0):.3g} bytes "
              "accessed (while-loop bodies counted once)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  B={b}: cost analysis unavailable ({e})", flush=True)
    sb = comp

    kk = jax.random.fold_in(key, 5)
    times, dec = [], 0
    for i in range(args.calls + 3):
        a, nd = policy(jax.random.fold_in(kk, i), m, f, s.done)
        s, a = prep(s, a)
        t1 = time.time()
        s, m, f = sb(s, a)
        jax.block_until_ready(f)
        dt = time.time() - t1
        if i == 0:
            print(f"  B={b}: 1st call {dt:.1f} s", flush=True)
        if i >= 3:
            times.append(dt)
            dec += int(nd)
        s, m, f = refill(s, m, f, pool, pm)
    tot = sum(times)
    print(f"  B={b}: {dec / tot:,.0f} decisions/s ; {1000 * tot / len(times):.1f} ms/call ; "
          f"{dec / len(times):.0f} decisions/call", flush=True)
    if args.profile:
        out = f"{args.profile}/B{b}"
        with jax.profiler.trace(out, create_perfetto_trace=True):
            for i in range(args.profile_calls):
                a, _ = policy(jax.random.fold_in(kk, 10_000 + i), m, f, s.done)
                s, m, f = sb(*prep(s, a))
                jax.block_until_ready(f)
                s, m, f = refill(s, m, f, pool, pm)
        print(f"  B={b}: trace of {args.profile_calls} calls in {out} (bench/trace_summary.py {out})", flush=True)


for b in map(int, args.b.split(",")):
    run(b)
