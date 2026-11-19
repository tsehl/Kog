"""Accelerator == CPU: step_batch gives bit-identical states on the accelerator and on the CPU.

Usage: python bench/tpu_parity.py [ENGINE_DIR] [--b 64] [--calls 80] [--extra 15] [--frac 0.05]
The oracle checks (engine/regress.sh) run on the CPU. This script plays the same games on both backends, with the same
actions (drawn from the CPU legal mask), and compares every field of the state, the legal mask and the forced action
after each call. A difference shows a backend-dependent computation (e.g. a float matmul computed in bfloat16 by default
on TPU). Exit code 1 on the first difference, with the field, the call and the games concerned.
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("engine", nargs="?", default="engine")
ap.add_argument("--b", type=int, default=64)
ap.add_argument("--calls", type=int, default=80)
ap.add_argument("--extra", type=int, default=15)
ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--dry", action="store_true", help="rehearsal on two CPU devices (checks the script, not the backend)")
args = ap.parse_args()
sys.path.insert(0, args.engine)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import cards  # noqa: E402
import core  # noqa: E402

acc = jax.devices()[0]
cpu = jax.devices("cpu")[0]
if args.dry:  # rehearsal of the script without an accelerator: two CPU devices
    acc = jax.devices("cpu")[1]
print(f"jax {jax.__version__}: {acc.device_kind} against {cpu.device_kind}", flush=True)
assert acc != cpu, "no accelerator: nothing to compare (--dry with XLA_FLAGS=--xla_force_host_platform_device_count=2)"

D, C, B = 40, 670, args.b
key = jax.random.PRNGKey(0)
bank = cards.test_bank(key, C)
decks, extras = cards.test_decks(jax.random.fold_in(key, 1), bank, B, D, C, 0.3, args.extra)
ks = jax.random.split(jax.random.fold_in(key, 2), B)
ng = jax.jit(jax.vmap(lambda d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1])))
legal = jax.jit(jax.vmap(lambda s: core.legal(bank, s)))
sb = jax.jit(lambda s, a: core.step_batch(bank, s, a, args.frac))
put = lambda t, d: jax.device_put(t, d)

# the same starting states on both backends (computed on the CPU, copied)
s_c = ng(*put((decks, extras, ks), cpu))
s_a = put(s_c, acc)
m_c = legal(s_c)
m_a = legal(s_a)
f = jnp.full(B, -1, jnp.int32)
rng = np.random.default_rng(0)


def compare(tag, xc, xa):
    xc, xa = np.asarray(xc), np.asarray(xa)
    if np.array_equal(xc, xa):
        return
    games = np.unique(np.argwhere(xc != xa)[:, 0]) if xc.ndim else []
    print(f"DIFFERENCE {tag}: games {games[:10].tolist()} ({len(games)} in total)", flush=True)
    sys.exit(1)


compare("legal mask at the start", m_c, m_a)
t0, n_dec = time.time(), 0
f_c = f_a = f
for i in range(args.calls):
    m = np.asarray(m_c)
    # random legal action (or the forced one), the same for both backends
    logits = np.where(m, rng.gumbel(size=m.shape), -np.inf)
    a = np.where(np.asarray(f_c) >= 0, np.asarray(f_c), logits.argmax(-1)).astype(np.int32)
    a = np.where(m.any(-1) | (np.asarray(f_c) >= 0), a, 0).astype(np.int32)
    n_dec += int(((np.asarray(f_c) < 0) & ~np.asarray(s_c.done)).sum())
    s_c, m_c, f_c = sb(s_c, put(jnp.asarray(a), cpu))
    s_a, m_a, f_a = sb(s_a, put(jnp.asarray(a), acc))
    for name, xc, xa in zip(core.State._fields, s_c, s_a):
        compare(f"in the state field {name} after call {i}", xc, xa)
    compare(f"in the legal mask after call {i}", m_c, m_a)
    compare(f"in the forced action after call {i}", f_c, f_a)
done = int(np.asarray(s_c.done).sum())
print(f"ok: {args.calls} calls, {n_dec} decisions on {B} games ({done} finished): every state field, legal mask and "
      f"forced action identical on {acc.device_kind} and CPU ({time.time() - t0:.0f} s)", flush=True)
