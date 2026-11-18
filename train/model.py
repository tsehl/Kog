"""Policy and value network.

Two stages:
1. `encode_bank`: each card of the bank is read from its bricks (header, effects, instructions of its programs,
   filters), never from an identifier. A small per-card transformer (~90 tokens: header, one token per effect, one
   per instruction, one per filter, one per materials filter) gives a vector [d] per card.
   Computed once per rollout (fixed parameters), recomputed at each minibatch of the update (gradient).
2. `policy`: the T tokens of the observation (engine/obs.py) = the card's vector (or "unknown") + its visible state
   (location, position, current ATK / DEF, statuses, chain...), plus a global token. Transformer, then
   verb x token logits, logits of the special actions (on the global token), value (global token).
   Its remaining deck (`o.deck`, sorted list): vectors of the cards -> MLP -> sum, added to the global token
   (parameters `deck1` / `deck2`, deck2 initialized to zero: a checkpoint without them behaves exactly as before).

The integers of the bricks (parameters a, b, c, filter fields) are given by symlog(x) and their 16 low-order bits:
most of them are bit fields (locations, types, positions, events) or codes (archetype, name), which the network must
be able to compare between cards (same archetype) without a learned identifier.
Parameters as a pytree of dicts; pure jax (no flax), optax for the optimizer.
"""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import core  # noqa: E402
from core import P, NF, FW, MW, NSET, FK, NV, N_SPECIAL, CH, V_ACT0  # noqa: E402

NBITS = 16
N_OPS = 48  # margin: the opcode table grows without changing the weight shapes (complete() pads old ones)
N_KIND = 8
N_DEC = 13
N_PHASE = 16


# ---------------------------------------------------------------------------------------------------- bricks

def ifeat(x):
    """Integer(s) -> [..., 1 + NBITS]: symlog and low-order bits."""
    x = x.astype(jnp.int32)
    sl = jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0
    bits = ((jnp.abs(x)[..., None] >> jnp.arange(NBITS)) & 1).astype(jnp.float32)
    return jnp.concatenate([sl[..., None], bits], -1)


def stat(x):
    """ATK / DEF -> [..., NSTAT]: linear (/ 5000) and thermometer in steps of 250 up to 5000.
    Symlog alone squashes the useful gaps (1500 and 2000 differ by 0.036): comparing two ATK values, which decides
    an attack, a Summon or a target, required huge weights."""
    x = jnp.clip(x.astype(jnp.float32), 0, 5000)
    th = jnp.clip((x[..., None] - jnp.arange(0, 5000, 250, dtype=jnp.float32)) / 250, 0, 1)
    return jnp.concatenate([x[..., None] / 5000, th], -1) * NUM_ON


def lvl(x):
    """Level / Rank / Link Rating -> [..., NLVL]: one-hot (0-13) and linear."""
    return jnp.concatenate([onehot(x, 14), jnp.clip(x.astype(jnp.float32), 0, 13)[..., None] / 13], -1) * NUM_ON


NUM_ON = float(os.environ.get("YGO_NO_NUM", "0") != "1")   # YGO_NO_NUM=1: features at zero (control run)
NSTAT = 1 + 20
NLVL = 15
NUM = 2 * NSTAT + NLVL   # added at the end of the header and of the dynamic features


def bits(x, n):
    return ((x.astype(jnp.int32)[..., None] >> jnp.arange(n)) & 1).astype(jnp.float32)


def onehot(x, n):
    return jax.nn.one_hot(jnp.clip(x, 0, n - 1), n)


def dense_init(k, i, o, scale=1.0):
    return dict(w=jax.random.normal(k, (i, o)) * scale / jnp.sqrt(i), b=jnp.zeros(o))


CDT = jnp.float32   # compute type (jnp.bfloat16: mixed precision, parameters in f32); see set_compute


def set_compute(bf16):
    global CDT
    CDT = jnp.bfloat16 if bf16 else jnp.float32


def dense(p, x):
    return x.astype(CDT) @ p["w"].astype(CDT) + p["b"].astype(CDT)


def ln_init(d):
    return dict(g=jnp.ones(d), b=jnp.zeros(d))


def ln(p, x):
    x = x.astype(jnp.float32)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return ((x - m) * jax.lax.rsqrt(v + 1e-5) * p["g"] + p["b"]).astype(CDT)


def block_init(k, d, h):
    ks = jax.random.split(k, 4)
    return dict(ln1=ln_init(d), qkv=dense_init(ks[0], d, 3 * d), o=dense_init(ks[1], d, d, 0.5),
                ln2=ln_init(d), f1=dense_init(ks[2], d, 4 * d), f2=dense_init(ks[3], 4 * d, d, 0.5))


def block(p, x, mask, h):
    """Pre-LN block. x [..., L, d], mask [..., L] (masked tokens are not read)."""
    d = x.shape[-1]
    y = ln(p["ln1"], x)
    q, k, v = jnp.split(dense(p["qkv"], y), 3, -1)
    sh = lambda t: t.reshape(t.shape[:-1] + (h, d // h))
    q, k, v = sh(q), sh(k), sh(v)
    att = jnp.einsum("...qhc,...khc->...hqk", q, k) / jnp.sqrt(d // h)
    att = jnp.where(mask[..., None, None, :], att.astype(jnp.float32), -1e9)
    y = jnp.einsum("...hqk,...khc->...qhc", jax.nn.softmax(att, -1).astype(CDT), v).reshape(x.shape)
    x = x + dense(p["o"], y)
    return x + dense(p["f2"], jax.nn.gelu(dense(p["f1"], ln(p["ln2"], x))))


# ---------------------------------------------------------------------------------------------------- cards

F_INS = 3 * (1 + NBITS) + 16                       # a, b, c
F_FILT = FW * (1 + NBITS) + 16
F_EFF = 32 + 3 * 8 + MW * (1 + NBITS) + 3     # etrig, eopt, erng, edmg; emod; inline, refund
F_HEAD = 3 + 32 + 8 + 32 + NSET * NBITS + 9 + 2 + 8 + FK + NUM   # stats, ctype, attr, race, setcode, arrows, scales, procedures
N_MATF = FK + 2 + 1 + 1                        # materials filters: Fusion, Synchro (2), Xyz, Link


def card_tokens(bank):
    """Bank -> (features per token type, masks). Fixed shapes for all procedural banks."""
    sl = lambda x: (jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0)[..., None]
    head = jnp.concatenate([
        sl(bank.atk), sl(bank.df), sl(bank.level), bits(bank.ctype, 32), bits(bank.attr, 8), bits(bank.race, 32),
        bits(bank.setcode, NBITS).reshape(bank.setcode.shape[0], -1), bits(bank.lmark, 9), sl(bank.lsc), sl(bank.rsc),
        sl(bank.fmn), sl(bank.xct), sl(bank.lmin), sl(bank.lmax), sl(bank.lneed), sl(bank.ldiff), sl(bank.syn), sl(bank.erl), bits(bank.fsub, FK),
        stat(bank.atk), stat(bank.df), lvl(bank.level),
    ], -1)
    inline = bank.einline if bank.einline.shape[1] else jnp.zeros_like(bank.ekind)
    refund = bank.erefund if bank.erefund.shape[1] else jnp.zeros_like(bank.ekind)
    eff = jnp.concatenate([bits(bank.etrig, 32), bits(bank.eopt, 8), bits(bank.erng, 8), bits(bank.edmg, 8),
                           ifeat(bank.emod).reshape(bank.emod.shape[:2] + (-1,)), bits(inline, 2), bits(refund, 1)], -1)
    prog = bank.prog                                                                          # [C, E, 3, P, 4]
    ins_op = prog[..., 0]
    ins = jnp.concatenate([ifeat(prog[..., 1:]).reshape(prog.shape[:-1] + (-1,)), bits(prog[..., 3], 32)[..., 16:]], -1)                               # [C, E, 3, P, F_INS]
    filt = jnp.concatenate([ifeat(bank.filt[..., :-1]).reshape(bank.filt.shape[:-1] + (-1,)), bits(jnp.abs(bank.filt[..., core.F_REASON]), 32)[..., 16:], ifeat(bank.filt[..., -1])], -1)                             # [C, E, NF, F_FILT]
    mats = jnp.concatenate([bank.fmat, bank.smat, bank.xmat[:, None], bank.lmat[:, None]], 1)   # [C, N_MATF, FW]
    matf = jnp.concatenate([ifeat(mats[..., :-1]).reshape(mats.shape[:-1] + (-1,)), bits(jnp.abs(mats[..., core.F_REASON]), 32)[..., 16:], ifeat(mats[..., -1])], -1)
    has = jnp.concatenate([jnp.arange(FK)[None] < bank.fmn[:, None], jnp.repeat(bank.syn[:, None] != 0, 2, 1),
                           bank.xct[:, None] > 0, bank.lmax[:, None] > 0], 1)
    emask = bank.ekind != core.K_NONE                                                         # [C, E]
    imask = (ins_op != core.OP_END) & emask[:, :, None, None]
    fmask = (bank.filt != 0).any(-1) & emask[:, :, None]
    return dict(head=head, eff=eff, ekind=bank.ekind, ins=ins, ins_op=ins_op, filt=filt, matf=matf), \
        dict(eff=emask, ins=imask, filt=fmask, mat=has)


def card_enc_init(k, d, layers):
    ks = jax.random.split(k, 12 + layers)
    return dict(head=dense_init(ks[0], F_HEAD, d), eff=dense_init(ks[1], F_EFF, d), kind=jax.random.normal(ks[2], (N_KIND, d)) * 0.1,
                ins=dense_init(ks[3], F_INS, d), op=jax.random.normal(ks[4], (N_OPS, d)) * 0.1,
                filt=dense_init(ks[5], F_FILT, d), matf=dense_init(ks[6], F_FILT, d),
                pos_e=jax.random.normal(ks[7], (E_MAX, d)) * 0.1, pos_g=jax.random.normal(ks[8], (3, d)) * 0.1,
                pos_i=jax.random.normal(ks[9], (P, d)) * 0.1, pos_f=jax.random.normal(ks[10], (NF, d)) * 0.1,
                pos_m=jax.random.normal(ks[11], (N_MATF, d)) * 0.1,
                blocks=[block_init(ks[12 + i], d, 4) for i in range(layers)], out=ln_init(d))


E_MAX = 16  # learned effect positions (pos_e): bound of core.E, the rest of the network does not depend on E


def pos_e_rows(pe, n):
    """First n effect positions; old checkpoints (pos_e [2, d]): the following rows are drawn from a fixed key,
    used only by effects that did not exist for them."""
    if pe.shape[0] >= n:
        return pe[:n]
    extra = jax.random.normal(jax.random.PRNGKey(7), (n - pe.shape[0], pe.shape[1])) * 0.1
    return jnp.concatenate([pe, extra.astype(pe.dtype)])


LC = 64   # useful tokens kept per card (header included); E = 4: ~9 on average, 46 at most over 10 banks


def n_tokens(msk):
    """Useful tokens per card (header included), [C]: must stay <= LC (see `encode_cards`)."""
    C = msk["eff"].shape[0]
    return 1 + sum(v.reshape(C, -1).sum(1) for v in msk.values())


def encode_cards(p, tok, msk, heads=4):
    """[C, 1 + E, d]: per card, the output of the header token then that of each effect token (zero for an empty or
    truncated effect). The policy scores the activation of an effect from its token: no weight depends on E.
    Of the 91 possible tokens, ~8 are useful: they are grouped at the front (stable sort, the header stays first) and
    the sequence is cut to LC before the transformer. The positions are added beforehand; the result is the same as
    on the 91 tokens as long as no card has more than LC useful tokens."""
    C, NE = tok["ekind"].shape
    pe = pos_e_rows(p["pos_e"], NE)
    # Old checkpoints predate the FK substitution permissions.
    # Older checkpoints do not have the NUM numeric features at the end of the header.
    head = tok["head"]
    if p["head"]["w"].shape[0] in (F_HEAD - NUM, F_HEAD - NUM - FK):
        head = head[:, :p["head"]["w"].shape[0]]  # same matrix product as before; no weight invented
    elif p["head"]["w"].shape[0] != F_HEAD:
        raise ValueError("incompatible checkpoint header dimension")
    x_head = dense(p["head"], head)[:, None]
    eff = tok["eff"]
    if p["eff"]["w"].shape[0] == F_EFF - 3:
        eff = eff[..., :-3]
    elif p["eff"]["w"].shape[0] != F_EFF:
        raise ValueError("incompatible checkpoint effect dimension")
    x_eff = dense(p["eff"], eff) + p["kind"][jnp.clip(tok["ekind"], 0, N_KIND - 1)] + pe
    if p["ins"]["w"].shape[0] not in (F_INS, F_INS - 16):
        raise ValueError("incompatible checkpoint instruction dimension")
    if any(p[key]["w"].shape[0] not in (F_FILT, F_FILT - 17, F_FILT - 50) for key in ("filt", "matf")):
        raise ValueError("incompatible checkpoint filter dimension")
    ins = tok["ins"][..., :p["ins"]["w"].shape[0]]
    filt = tok["filt"][..., :p["filt"]["w"].shape[0]]
    matf = tok["matf"][..., :p["matf"]["w"].shape[0]]
    x_ins = (dense(p["ins"], ins) + p["op"][jnp.clip(tok["ins_op"], 0, N_OPS - 1)]
             + pe[:, None, None] + p["pos_g"][:, None] + p["pos_i"])
    x_filt = dense(p["filt"], filt) + pe[:, None] + p["pos_f"]
    x_mat = dense(p["matf"], matf) + p["pos_m"]
    x = jnp.concatenate([x_head, x_eff, x_ins.reshape(C, -1, x_head.shape[-1]), x_filt.reshape(C, -1, x_head.shape[-1]),
                         x_mat], 1)
    m = jnp.concatenate([jnp.ones((C, 1), bool), msk["eff"], msk["ins"].reshape(C, -1), msk["filt"].reshape(C, -1),
                         msk["mat"]], 1)
    order = jnp.argsort(~m, axis=1, stable=True)[:, :LC]
    x = jnp.take_along_axis(x, order[..., None], 1)
    m = jnp.take_along_axis(m, order, 1)
    for b in p["blocks"]:
        x = block(b, x, m, heads)
    x = ln(p["out"], x).astype(jnp.float32) * m[..., None]
    # back to the original positions: header (0) then effects (1..E); the truncated tokens stay zero
    full = jnp.zeros((C, 1 + NE, x.shape[-1]), x.dtype)
    keep = order <= NE
    full = full.at[jnp.arange(C)[:, None], jnp.where(keep, order, 1 + NE)].set(x, mode="drop")
    return full


# ---------------------------------------------------------------------------------------------------- policy

F_DYN = 3 + 8 + 16 + 4 + 17 + 1 + (CH + 1) + 1 + 1 + 1 + 64 + NUM + 1 + CH   # additions at the end: source_related, targets per chain link
F_GLOB = 2 + 2 + N_PHASE + 3 + N_DEC + 1 + 8 + 1 + 12  # LP, turn, phase, flags, decision, need, zctx, chain, counters


def dyn_feats(o):
    sl = lambda x: (jnp.sign(x) * jnp.log1p(jnp.abs(x).astype(jnp.float32)) / 8.0)[..., None]
    zone = bits(o.zone, 8)
    return jnp.concatenate([sl(o.atk), sl(o.df), sl(o.level), zone, onehot(o.seq, 16), bits(o.pos, 4), bits(o.st, 17),
                            o.mine[..., None].astype(jnp.float32), onehot(o.chain, CH + 1),
                            o.negated[..., None].astype(jnp.float32), o.pending[..., None].astype(jnp.float32),
                            o.card_activation[..., None].astype(jnp.float32), bits(o.suminfo, 32), bits(o.reason, 32),
                            stat(o.atk), stat(o.df), lvl(o.level), o.source_related[..., None].astype(jnp.float32),
                            o.targeted.astype(jnp.float32)], -1)


def glob_feats(o):
    g = o.glob
    f = lambda x: x.astype(jnp.float32)[..., None]
    return jnp.concatenate([f(g[..., 0]) / 8000, f(g[..., 1]) / 8000, f(jnp.minimum(g[..., 2], 100)) / 50, f(g[..., 3]),
                            bits(g[..., 4], N_PHASE), f(g[..., 5]), f(g[..., 6]), f(g[..., 3]) * 0,
                            onehot(g[..., 7], N_DEC), f(g[..., 8]) / 4, onehot(g[..., 9], 8), f(g[..., 10]) / CH,
                            g[..., 11:23].astype(jnp.float32) / 20], -1)


def init(k, d=128, card_layers=2, layers=3):
    ks = jax.random.split(k, 8 + layers)
    return dict(cards=card_enc_init(ks[0], d, card_layers), unk=jax.random.normal(ks[1], (d,)) * 0.1,
                dyn=dense_init(ks[2], F_DYN, d), glob=dense_init(ks[3], F_GLOB, d),
                blocks=[block_init(ks[8 + i], d, 4) for i in range(layers)], out=ln_init(d),
                act=dense_init(ks[4], d, V_ACT0 + 1, 0.01), **eff_head_init(ks[4], d), spec=dense_init(ks[5], d, N_SPECIAL, 0.01),
                v1=dense_init(ks[6], d, d), v2=dense_init(ks[7], d, 1, 0.01), **deck_init(ks[1], d))


def complete(p):
    """Brings old parameters to the current structure and shapes, without changing the output:
    - before the remaining deck: deck1 / deck2 zero (deck2 zero, zero contribution);
    - shorter inputs (features added at the end: dyn, head, eff, ins, filt, matf): zero rows at the end of the weights,
      same product as the truncation of the features done in policy / encode.
    An old opponent then shares the compilations of the trained network."""
    def pad(q, n):
        w = q["w"]
        return q if w.shape[0] >= n else dict(q, w=jnp.concatenate([w, jnp.zeros((n - w.shape[0],) + w.shape[1:], w.dtype)]))
    p = dict(p, dyn=pad(p["dyn"], F_DYN))
    c = dict(p["cards"])
    for k, n in dict(head=F_HEAD, eff=F_EFF, ins=F_INS, filt=F_FILT, matf=F_FILT).items():
        c[k] = pad(c[k], n)
    if c["op"].shape[0] < N_OPS:  # opcodes added: zero embeddings
        c["op"] = jnp.concatenate([c["op"], jnp.zeros((N_OPS - c["op"].shape[0], c["op"].shape[1]), c["op"].dtype)])
    p["cards"] = c
    w = p["act"]["w"]
    if w.shape[1] != V_ACT0 + 1:
        # before the per-effect-token head: one column per effect number. The non-effect verbs are kept; the
        # activation starts from the mean of the old effect columns (same score for all the effects of a card), the
        # term that reads the effect token (eo) is zero: nothing else is invented.
        b = p["act"]["b"]
        p = dict(p, act=dict(w=jnp.concatenate([w[:, :V_ACT0], w[:, V_ACT0:].mean(1, keepdims=True)], 1),
                             b=jnp.concatenate([b[:V_ACT0], b[V_ACT0:].mean(keepdims=True)])))
    if "eq" not in p:
        d = w.shape[0]
        p = dict(p, **jax.tree.map(lambda x: x.astype(w.dtype), eff_head_init(jax.random.PRNGKey(11), d)))
    if "deck1" in p:
        return p
    d = p["glob"]["w"].shape[1]
    z = lambda: dict(w=jnp.zeros((d, d), p["glob"]["w"].dtype), b=jnp.zeros(d, p["glob"]["w"].dtype))
    return dict(p, deck1=z(), deck2=z())


def eff_head_init(k, d):
    """Activation of effect e of the card of token t: act[:, V_ACT0](x_t) (common to all effects) +
    eo(gelu(eq(x_t) + ek(effect_e))). eo zero at the start."""
    k1, k2 = jax.random.split(jax.random.fold_in(k, 202))
    return dict(eq=dense_init(k1, d, d), ek=dense_init(k2, d, d), eo=dense_init(k, d, 1, 0.0))


def deck_init(k, d):
    """Encoder of the remaining deck; deck2 zero at the start (no effect as long as it has learned nothing)."""
    return dict(deck1=dense_init(jax.random.fold_in(k, 101), d, d), deck2=dense_init(k, d, d, 0.0))


def deck_feats(p, cemb, o):
    """[.., d]: sum over its remaining deck of MLP(card vector), / 20."""
    e = cemb[jnp.clip(o.deck, 0, cemb.shape[0] - 1), 0]
    h = jax.nn.gelu(dense(p["deck1"], e)) * (o.deck >= 0)[..., None]
    return dense(p["deck2"], h.sum(-2) / 20)


def policy(p, cemb, o, heads=4):
    """o: Obs (engine/obs.py) of a batch [..]. Returns (masked logits [.., T*NV + N_SPECIAL], value [..])."""
    known = o.known[..., None]
    dyn = dyn_feats(o)
    previous = F_DYN - CH  # historical prefixes, before the per-link targets
    if p["dyn"]["w"].shape[0] in (previous, previous - 1, previous - NUM - 1, previous - NUM - 33, previous - NUM - 65, previous - NUM - 66):
        dyn = dyn[..., :p["dyn"]["w"].shape[0]]
    elif p["dyn"]["w"].shape[0] != F_DYN:
        raise ValueError("incompatible checkpoint dynamic dimension")
    cid = jnp.clip(o.cid, 0, cemb.shape[0] - 1)
    x = jnp.where(known, cemb[cid, 0], p["unk"]) + dense(p["dyn"], dyn)
    g = dense(p["glob"], glob_feats(o))
    if "deck1" in p:  # older checkpoints have no deck encoder: skip it
        g = g + deck_feats(p, cemb, o)
    g = g[..., None, :]
    x = jnp.concatenate([g, x], -2)
    m = jnp.concatenate([jnp.ones(o.valid.shape[:-1] + (1,), bool), o.valid], -1)
    for b in p["blocks"]:
        x = block(b, x, m, heads)
    x = ln(p["out"], x)
    xt = x[..., 1:, :]
    base = dense(p["act"], xt)                                        # [.., T, V_ACT0 + 1]
    ee = jnp.where(known[..., None], cemb[cid, 1:], 0.0)              # [.., T, E, d]
    h = jax.nn.gelu(dense(p["eq"], xt)[..., None, :] + dense(p["ek"], ee.astype(xt.dtype)))
    le = base[..., V_ACT0:] + dense(p["eo"], h)[..., 0]               # [.., T, E]
    la = jnp.concatenate([base[..., :V_ACT0], le], -1)                # [.., T, NV]
    ls = dense(p["spec"], x[..., 0, :])                               # [.., N_SPECIAL]
    logits = jnp.concatenate([la.reshape(la.shape[:-2] + (-1,)), ls], -1)
    legal = jnp.concatenate([o.act.reshape(o.act.shape[:-2] + (-1,)), o.special], -1)
    logits = jnp.where(legal, logits.astype(jnp.float32), -1e9)
    v = jnp.tanh(dense(p["v2"], jax.nn.gelu(dense(p["v1"], x[..., 0, :])))[..., 0].astype(jnp.float32))
    return logits, v


def to_engine(o, n, idx):
    """Index of the network output -> engine action (verb x slot, or special action)."""
    T = o.slot.shape[-1]
    tok, verb = idx // NV, idx % NV
    slot = jnp.take_along_axis(o.slot, jnp.clip(tok, 0, T - 1)[..., None], -1)[..., 0]
    return jnp.where(idx < T * NV, verb * n + slot, NV * n + (idx - T * NV)).astype(jnp.int32)
