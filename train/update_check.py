"""Check that cutting an observation to L tokens (>= its valid tokens) does not change the update: same loss
statistics and same parameters after one step, with 64 tokens and with the smallest class.
CPU, without step_batch (early-game states). Usage: python train/update_check.py [--b 16]"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

import ppo
import model
import obs as obsm

a = ppo.parse()
model.set_compute(False)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
fns = ppo.make_fns(a, opt)
act, update, enc = fns[0], fns[4], fns[5]
params = model.init(jax.random.PRNGKey(0), a.d, a.card_layers, a.layers)
# non-zero heads, otherwise the truncation would be tested only on almost constant logits
params = jax.tree.map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(1), x.shape), params)
cset, bank = ppo.bank_of(a.eval_bank, a)
tok, msk = model.card_tokens(bank)
env = ppo.Env(a.b, a.frac, "structured")
s, m = env.games(bank, cset, np.random.default_rng(0), jax.random.PRNGKey(2))
f = jnp.full(a.b, -1, jnp.int32)
_, tr = act(params, enc(params, tok, msk), bank, s, m, f, jax.random.PRNGKey(3), jnp.asarray(False))
fb = dict(tr, adv=jax.random.normal(jax.random.PRNGKey(4), (a.b,)), ret=jax.random.uniform(jax.random.PRNGKey(5), (a.b,)))
nval = np.asarray(fb["o"].valid.sum(-1))
classes = ppo.tok_classes(a)
L = min(c for c in classes if c >= nval.max())
assert L < obsm.T, f"states too crowded for the test: {nval.max()} tokens"
idx = jnp.arange(a.b)
ost = opt.init(params)
out = [update(params, ost, tok, msk, ppo.take_batch(fb, idx, c, a.b)) for c in (obsm.T, L)]
for k in out[0][2]:
    u, v = float(out[0][2][k]), float(out[1][2][k])
    print(f"{k:10s} {u:+.6f} {v:+.6f}")
    assert abs(u - v) <= 1e-4 * max(1.0, abs(u)), k
dp = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree.leaves(out[0][0]), jax.tree.leaves(out[1][0])))
print(f"valid tokens: {nval.min()}-{nval.max()}; classes {classes}; cut at {L}; max parameter difference after one step: {dp:.2e}")
assert dp < 1e-5
print("ok: the update does not depend on the cut length")
