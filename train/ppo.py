"""Self-play PPO on procedural cards and decks.

Loop:
- a procedural bank (gen/cardgen.py) every `--bank-every` iterations, decks of 40 + Extra Deck of 15 drawn
  without structure; B games in parallel (`core.step_batch`); a finished game is replaced by a new one;
- both players play with the same policy; each decision is stored with its player. The advantages (GAE)
  are computed per player: going back in time, each game keeps for each of the two players the value and
  the advantage of their next decision; the end of the game gives +1 to the winner, -1 to the loser (0: draw);
- PPO (clip), minibatches along the game axis; the encoding of the bank cards is recomputed in every
  minibatch (gradient down to the bricks);
- evaluation: the network (sampling) against the uniform random policy, half of the games on each
  side, on a held-out bank (seed `--eval-bank`), never seen in training.

Usage (CPU test): uv run --with jax --with optax python train/ppo.py --b 32 --steps 16 --iters 3 --d 64
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gen"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402
import model  # noqa: E402
import obs as obsm  # noqa: E402

if os.environ.get("YGO_JAX_CACHE"):
    jax.config.update("jax_compilation_cache_dir", os.environ["YGO_JAX_CACHE"])


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=256, help="games in parallel")
    ap.add_argument("--steps", type=int, default=64, help="step_batch calls per rollout")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--card-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=4, help="minibatches per epoch")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--frac", type=float, default=0.1, help="early stop of step_batch (optimum measured on an L4)")
    ap.add_argument("--bank-every", type=int, default=10)
    ap.add_argument("--main", type=int, default=400)
    ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--old-lag", type=int, default=50, help="also evaluate against the policy from X iterations ago")
    ap.add_argument("--eval-b", type=int, default=0, help="evaluation games (default: b)")
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--bench-every", type=int, default=50, help="benchmark on the real cards (multiple of --eval-every)")
    ap.add_argument("--eval-bank", type=int, default=1_000_000)
    ap.add_argument("--ref-ckpt", default="", help="frozen reference network (pickle, .gz accepted): evaluation against it "
                    "at every --eval-every (archetype bank, structured decks) and on the real cards at every --bench-every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--stop-file", default="", help="clean stop at the end of an iteration if this file exists (budget)")
    ap.add_argument("--snap-every", type=int, default=0, help="also keep the parameters every X iterations (SAVE.itN.gz, f16)")
    ap.add_argument("--gen-revive", type=int, default=0, help="training banks: more revival effects with a choice (cardgen.Gen)")
    ap.add_argument("--arch", type=float, default=0.5,
                    help="training banks and structured evaluation: share of archetype cards (cardgen.Gen; 0 = banks from before archetypes)")
    ap.add_argument("--deck-pool", type=int, default=2048, help="pairs of structured decks drawn per bank (2 × X decks)")
    ap.add_argument("--match", type=int, default=0, help="strength-based matching: a deck that has played at least X games "
                    "faces a neighbour in the pool's Elo ranking (0 = random pairs). The ranking is worthless "
                    "before ~24 games per deck (train/deck_match.py): use a small pool (--deck-pool 64)")
    ap.add_argument("--deck-caps", action="store_true", help="discouraged (makes decks uniform): capped structured decks, "
                    "16 to 24 monsters, 2 to 6 of Level 5 or higher (cardgen.structured_decks(caps=True))")
    ap.add_argument("--deck-profile", action="store_true", help="composition profile drawn at random per deck, big monsters "
                    "supported (cardgen.structured_decks(profile=True))")
    ap.add_argument("--skel", type=float, default=0.0, help="share of effect cards drawn from the skeletons (gen/skeleton.py)")
    ap.add_argument("--tok-classes", default="32,40,48,64", help="update: observation lengths (tokens) of the batches; "
                    "each decision goes into the smallest class that holds its valid tokens (one compilation per class)")
    ap.add_argument("--update-old", action="store_true", help="old update: the whole buffer, split by games, "
                    "64 tokens (including the slots without a network decision)")
    ap.add_argument("--bf16", action="store_true", help="network computation in bfloat16 (parameters and optimiser in f32)")
    ap.add_argument("--decks", default="structured", choices=["structured", "random"],
                    help="training decks (the procedural evaluation 'rnd / heur' keeps the reference bank and random decks; 'sheur': archetype bank, structured decks)")
    ap.add_argument("--mix-heur", type=float, default=0.0, help="share of games against the styled heuristic")
    ap.add_argument("--mix-old", type=float, default=0.0, help="share of games against an older version (league)")
    ap.add_argument("--league-every", type=int, default=10, help="one version added to the league every X iterations")
    ap.add_argument("--league-size", type=int, default=5)
    ap.add_argument("--init", default="", help="starting parameters (pickle) instead of a random initialisation; "
                    "the optimiser restarts from zero")
    ap.add_argument("--it0", type=int, default=0, help="first iteration (resuming: banks, snapshots and the log line follow)")
    ap.add_argument("--eval-first", action="store_true", help="full evaluation also after the first iteration (checks "
                    "every path from the start)")
    ap.add_argument("--precompile", type=int, default=0,
                    help="precompile the heavy functions with X threads in parallel (0: on the fly; memory: ~X times the peak)")
    ap.add_argument("--prof", type=int, default=0, help="at the end: time act, step_batch, act_step... X times on the rollout state")
    return ap.parse_args()


class Env:
    """Compiled functions of the engine (the bank is an argument: one compilation for all banks)."""

    def __init__(self, b, frac, decks="random", pool=2048, match=0, caps=False, profile=False):
        self.b = b
        self.decks = decks
        self.pool_n, self.match, self.caps, self.profile = pool, match, caps, profile
        self.last_ids = None                       # [b, 2] decks (index into the pool) of the last games drawn
        self._pool = (None, None)
        ng = jax.vmap(lambda bank, d, x, k: core.new_game(bank, d[0], d[1], k, x[0], x[1]), in_axes=(None, 0, 0, 0))
        lg = jax.vmap(lambda bank, s: core.legal(bank, s), in_axes=(None, 0))
        self.new = jax.jit(lambda bank, d, x, k: (lambda s: (s, lg(bank, s)))(ng(bank, d, x, k)))
        self.step = jax.jit(lambda bank, s, a: core.step_batch(bank, s, a, frac))

    def games(self, bank, cset, rng, key):
        if self.decks == "structured":
            # pool of structured decks drawn once per bank (~0.5 ms per deck in Python), then drawn from
            if self._pool[0] is not cset:
                self._pool = (cset, [z.reshape(-1, z.shape[-1]) for z in cardgen.structured_decks(cset, rng, self.pool_n, caps=self.caps,
                                                                                         profile=self.profile)])
                n = len(self._pool[1][0])
                self.elo, self.played = np.zeros(n), np.zeros(n, int)
            pd, px = self._pool[1]
            i = rng.integers(0, len(pd), (self.b, 2))
            if self.match:
                assert self.decks == "structured"
                # opponent among the ranking neighbours (+- 1/16 of the pool), for decks played enough
                order = np.argsort(self.elo + rng.normal(0, 1e-3, len(pd)))
                rank = np.empty(len(pd), int)
                rank[order] = np.arange(len(pd))
                w = max(2, len(pd) // 16)
                off = rng.integers(1, w + 1, self.b) * rng.choice([-1, 1], self.b)
                near = order[np.clip(rank[i[:, 0]] + off, 0, len(pd) - 1)]
                ok = (self.played[i[:, 0]] >= self.match) & (near != i[:, 0])
                i[:, 1] = np.where(ok, near, i[:, 1])
                flip = rng.random(self.b) < 0.5        # the deck drawn first does not always play first
                i = np.where(flip[:, None], i[:, ::-1], i)
            self.last_ids = i
            d, x = pd[i], px[i]
        else:
            d, x = cardgen.random_decks(cset, rng, self.b)
        return self.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, self.b))

    def report(self, ids, winner):
        """Results of finished games: ids [k, 2] pool decks, winner [k] winning player (0 / 1). Elo, K = 24."""
        for (i, j), w in zip(ids, winner):
            e = 1.0 / (1.0 + 10 ** ((self.elo[j] - self.elo[i]) / 400))
            g = 24 * ((w == 0) - e)
            self.elo[i] += g
            self.elo[j] -= g
            self.played[i] += 1
            self.played[j] += 1


def bank_of(seed, a, revive=0, arch=0.0):
    if getattr(a, "skel", 0) and seed < a.eval_bank:     # training banks only: the evaluation stays comparable
        cset = cardgen.Gen(seed, revive, arch, skel=a.skel).make_set(a.main, a.extra)
    else:
        cset = cardgen.Gen(seed, revive, arch).make_set(a.main, a.extra)
    bank = cardgen.to_bank(cset)
    nt = int(model.n_tokens(model.card_tokens(bank)[1]).max())
    assert nt <= model.LC, f"bank {seed}: a card has {nt} useful tokens > model.LC = {model.LC}"
    return cset, bank


def _where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


# heuristic opponent: weight per verb (V_SUMMON, V_MSET, V_REPOS, V_ATTACK, V_SELECT, V_SSET, then V_ACT0 + e)
H_VERB = [4.0, 1.5, 0.3, 3.0, 1.0, 2.0] + [5.0] * core.E
# special actions: to BP, to M2, to EP, 7 zones, ATK / DEF position, pass, yes
H_SPECIAL = [2.5, 0.5, 0.2] + [1.0] * 7 + [1.2, 0.8, 0.1, 1.0]


def heuristic(bank, s, m, f, key):
    """Simple reference opponent, more demanding than the random one: activates everything that can be, summons (the
    strongest first), attacks with everything, Sets its Spells / Traps, then passes; choices (targets, materials, zones)
    at random. Returns the engine action [B] (the forced action if f >= 0)."""
    n = s.cid.shape[1]
    assert len(H_VERB) == core.NV and len(H_SPECIAL) == core.N_SPECIAL
    w = jnp.concatenate([jnp.repeat(jnp.asarray(H_VERB), n), jnp.asarray(H_SPECIAL)])          # [A]
    atk = bank.atk[s.cid].astype(jnp.float32) / 10000.0                                          # [B, n]
    vb = jnp.zeros((s.cid.shape[0], core.NV, n)).at[:, core.V_SUMMON].set(atk).at[:, core.V_ATTACK].set(atk)
    bonus = jnp.concatenate([vb.reshape(s.cid.shape[0], -1), jnp.zeros((s.cid.shape[0], core.N_SPECIAL))], -1)
    sc = w + bonus + 0.01 * jax.random.gumbel(key, m.shape)
    return jnp.where(f >= 0, f, jnp.argmax(jnp.where(m, sc, -1e9), -1).astype(jnp.int32))


# styled heuristic (training opponent): one style vector per game
# aggressiveness (1: attacks even when the attack is lost in advance), choice (1: reasoned targets / costs, 0: random),
# caution (1: Sets in Defense Position a monster weaker than the opponent's field), zeal (> 0.5: also activates during
# its own turn in the chain windows), noise temperature
S_AGGRO, S_SMART, S_PRUDENT, S_EAGER, S_TEMP = range(5)
N_STYLE = 5
STYLE_REF = (1.0, 0.0, 0.0, 1.0, 0.01)                   # = heuristic (the reference opponent of the evaluation)


def sample_style(key, b):
    """Random styles [b, N_STYLE]."""
    k = jax.random.split(key, 5)
    u = lambda k: jax.random.uniform(k, (b,))
    return jnp.stack([u(k[0]), (u(k[1]) < 0.7).astype(jnp.float32), u(k[2]), u(k[3]),
                      jnp.exp(jnp.log(0.01) + u(k[4]) * (jnp.log(0.5) - jnp.log(0.01)))], -1)


def heuristic_style(bank, s, m, f, key, sty):
    """heuristic, modulated by a per-game style sty [B, N_STYLE] (STYLE_REF: identical to heuristic). The rules only
    read the state (ATK / DEF, positions, locations), never the card identity: valid on procedural cards."""
    b, n = s.cid.shape
    w = jnp.concatenate([jnp.repeat(jnp.asarray(H_VERB), n), jnp.asarray(H_SPECIAL)])
    batk = bank.atk[s.cid].astype(jnp.float32) / 10000.0                                         # [B, n]
    col = lambda i: sty[:, i][:, None]
    me = s.dec_player[:, None]
    mon = s.zone == core.Z_MZONE
    my_mon, op_mon = mon & (s.ctrl == me), mon & (s.ctrl != me)
    fu = (s.pos & core.P_FACEUP) != 0
    fatk = s.pos == core.P_FU_ATK
    atk, df = s.atk_c.astype(jnp.float32) / 10000.0, s.df_c.astype(jnp.float32) / 10000.0
    # attack lost in advance: all opponent face-up monsters in attack and stronger
    lose = op_mon.any(-1, keepdims=True) & (~op_mon[:, None, :]
                                            | (fatk[:, None, :] & (atk[:, None, :] > atk[:, :, None]))).all(-1)
    v_atk = batk - 6.0 * (1 - col(S_AGGRO)) * lose
    # summon: hand monster weaker than the strongest face-up opponent monster -> Set (caution)
    op_best = jnp.where(op_mon & fu, atk, 0.0).max(-1, keepdims=True)
    hand = (s.zone == core.Z_HAND) & (s.ctrl == me)
    v_sum = batk - 4.0 * col(S_PRUDENT) * (hand & (batk < op_best))
    # card choices: attack targets, selections, Tributes, discards
    ak = jnp.take_along_axis(atk, jnp.maximum(s.atk, 0)[:, None], -1)
    kill = op_mon & fu & ((fatk & (atk < ak)) | (~fatk & (df < ak)))
    suicide = op_mon & fu & fatk & (atk > ak)
    t_att = 2.0 * kill + atk - 2.0 * suicide
    val = jnp.where(mon, atk, batk)
    high = (s.ctrl != me) | (s.zone == core.Z_GRAVE) | (s.zone == core.Z_REMOVED) | (s.zone == core.Z_DECK) \
        | (s.zone == core.Z_EXTRA)
    dsel = (s.dec == core.D_SELECT)[:, None] & high
    dpen = ((s.dec == core.D_TRIBUTE) & (s.psel == core.PS_PEND))[:, None]
    t_sel = jnp.where(dsel | dpen, val, -val)
    v_selc = col(S_SMART) * jnp.where((s.dec == core.D_TARGET)[:, None], t_att, t_sel)
    vb = jnp.zeros((b, core.NV, n)).at[:, core.V_SUMMON].set(v_sum).at[:, core.V_ATTACK].set(v_atk) \
        .at[:, core.V_SELECT].set(v_selc)
    # chain window during its own turn: without zeal, it does not activate (passing is worth 0.1)
    calm = ((s.dec == core.D_CHAIN) & (s.tp == s.dec_player) & (sty[:, S_EAGER] <= 0.5))[:, None, None]
    vb = vb.at[:, core.V_ACT0:].add(jnp.where(calm, -5.0, 0.0))
    bonus = jnp.concatenate([vb.reshape(b, -1), jnp.zeros((b, core.N_SPECIAL))], -1)
    sc = w + bonus + col(S_TEMP) * jax.random.gumbel(key, m.shape)
    return jnp.where(f >= 0, f, jnp.argmax(jnp.where(m, sc, -1e9), -1).astype(jnp.int32))


def make_fns(a, opt):
    obs_b = jax.vmap(lambda bank, s, m, p: obsm.observe(bank, s, p, m), in_axes=(None, 0, 0, 0))

    def _act(params, cemb, bank, s, m, f, key, greedy=False):
        n = s.cid.shape[1]
        o = obs_b(bank, s, m, s.dec_player)
        logits, v = model.policy(params, cemb, o)
        idx = jnp.where(greedy, jnp.argmax(logits, -1), jax.random.categorical(key, logits))  # greedy: evaluation
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), idx[:, None], -1)[:, 0]
        need = (f < 0) & ~s.done
        ae = jnp.where(f >= 0, f, model.to_engine(o, n, idx))
        return ae, dict(o=o, idx=idx, logp=logp, v=v, need=need, pl=s.dec_player)

    act = jax.jit(_act)

    @jax.jit
    def values(params, cemb, bank, s, m):
        vs = [model.policy(params, cemb, obs_b(bank, s, m, jnp.full_like(s.dec_player, p)))[1] for p in (0, 1)]
        return jnp.stack(vs, -1)                                                       # [B, 2]

    def refill(s, m, f, s2, m2, used):
        """Finished games replaced by the pool game with the same index, if it has not already been used."""
        d = s.done & ~used
        return _where(d, s2, s), _where(d, m2, m), jnp.where(d, -1, f), used | d

    @jax.jit
    def act_step(params, cemb, bank, s, m, f, key, side, pool, pm, used, opp=None, greedy=False):
        """One rollout call, in a single compiled function (no host synchronisation): policy, step_batch,
        then replacement of the finished games by the pool. side [B]: -1 = the network plays both sides; otherwise the
        network plays player `side` and the opponent the other (evaluation, with used = true: no replacement).
        opp = (parameters, cemb, kind[, styles [B, N_STYLE]]): uniform random opponent (kind 0), network (1),
        reference heuristic (2) or styled heuristic (3). Scalar kind: lax.switch outside vmap, a single
        branch computed; kind [B] (training, opponent per game): network and heuristic computed on the whole
        batch. Only the decisions of the tested network count (tr["need"]). Also returns done / winner just after
        step_batch."""
        k1, k2 = jax.random.split(key)
        ae, tr = _act(params, cemb, bank, s, m, f, k1, greedy)
        tr = dict(tr, need=tr["need"] & ((side < 0) | (s.dec_player == side)))
        rnd = lambda: jnp.where(f >= 0, f, jax.random.categorical(k2, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))
        sty = opp[3] if opp is not None and len(opp) > 3 else jnp.broadcast_to(jnp.asarray(STYLE_REF), (s.cid.shape[0], N_STYLE))
        if opp is None:
            ar = rnd()
        elif jnp.ndim(opp[2]) == 0:
            ar = jax.lax.switch(opp[2], [rnd, lambda: _act(opp[0], opp[1], bank, s, m, f, k2)[0],
                                         lambda: heuristic(bank, s, m, f, k2),
                                         lambda: heuristic_style(bank, s, m, f, k2, sty)])
        else:
            kind = opp[2]
            sty = jnp.where((kind == 2)[:, None], jnp.asarray(STYLE_REF), sty)
            ar = jnp.where(kind == 1, _act(opp[0], opp[1], bank, s, m, f, k2)[0],
                           jnp.where(kind >= 2, heuristic_style(bank, s, m, f, k2, sty), rnd()))
        s, m, f = core.step_batch(bank, s, jnp.where((side < 0) | (s.dec_player == side), ae, ar), a.frac)
        done, win = s.done, s.winner
        s, m, f, used = refill(s, m, f, pool, pm, used)
        return s, m, f, used, tr, done, win

    @jax.jit
    def gae(v, need, pl, done, win, last):
        """v, need, pl, done, win: [S, B]; last: [B, 2]. Returns (advantages, returns) [S, B]."""
        g, lam = a.gamma, a.lam
        pid = jnp.arange(2)

        def body(c, x):
            nv, adv, pend = c
            v, need, pl, done, win = x
            r = jnp.where(win[:, None] < 0, 0.0, jnp.where(win[:, None] == pid, 1.0, -1.0))
            dd = done[:, None]
            nv, adv, pend = jnp.where(dd, 0.0, nv), jnp.where(dd, 0.0, adv), jnp.where(dd, r, pend)
            sel = (pid == pl[:, None]) & need[:, None]                                 # [B, 2]
            pick = lambda t: (t * sel).sum(-1)
            delta = pick(pend) + g * pick(nv) - v
            A = delta + g * lam * pick(adv)
            nv, adv, pend = jnp.where(sel, v[:, None], nv), jnp.where(sel, A[:, None], adv), jnp.where(sel, 0.0, pend)
            return (nv, adv, pend), jnp.where(need, A, 0.0)
        z = jnp.zeros_like(last)
        _, A = jax.lax.scan(body, (last, z, z), (v, need, pl, done, win), reverse=True)
        return A, A + v

    def loss_fn(params, tok, msk, bt):
        cemb = model.encode_cards(params["cards"], tok, msk)
        logits, v = model.policy(params, cemb, bt["o"])
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, bt["idx"][..., None], -1)[..., 0]
        w = bt["need"].astype(jnp.float32)
        nw = jnp.maximum(w.sum(), 1.0)
        A = bt["adv"]
        ratio = jnp.exp(lp - bt["logp"])
        pg = -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * A)
        p_all = jnp.exp(lp_all)
        ent = -(p_all * jnp.where(p_all > 0, lp_all, 0.0)).sum(-1)
        vl = (v - bt["ret"]) ** 2
        tot = ((pg + a.vf * vl - a.ent * ent) * w).sum() / nw
        stats = dict(pg=(pg * w).sum() / nw, vl=(vl * w).sum() / nw, ent=(ent * w).sum() / nw,
                     kl=((bt["logp"] - lp) * w).sum() / nw, clipfrac=((jnp.abs(ratio - 1) > a.clip) * w).sum() / nw)
        return tot, stats

    @jax.jit
    def update(params, opt_state, tok, msk, bt):
        (l, st), gr = jax.value_and_grad(loss_fn, has_aux=True)(params, tok, msk, bt)
        up, opt_state = opt.update(gr, opt_state, params)
        st["gnorm"] = optax.global_norm(gr)
        return optax.apply_updates(params, up), opt_state, st

    enc = jax.jit(lambda params, tok, msk: model.encode_cards(params["cards"], tok, msk))
    return act, values, act_step, gae, update, enc, obs_b


# observation fields indexed by token (axis T): the valid tokens come first (obs.observe sorts the shown cards
# before the others), so cutting at L >= number of valid tokens does not change the network output
OBS_T = tuple(k for k in obsm.Obs._fields if k not in ("special", "glob", "deck"))


def tok_classes(a):
    c = sorted({int(v) for v in a.tok_classes.split(",") if v} | {obsm.T})
    return [v for v in c if v <= obsm.T]


def mb_size(a):
    """Decisions per batch of the compact update: ~75% of the buffer cells hold a decision of the network, so
    3/4 of the old batch gives about the same number of gradient steps, each with as many useful decisions."""
    return max(1, a.steps * a.b // a.mb * 3 // 4)


def plan_batches(need, nvalid, classes, msize, rng):
    """Batches of the compact update. need [N] bool (decision of the trained network), nvalid [N] valid tokens.
    Returns [(L, indices [msize], number of real indices)] in a random order: only the useful cells, sorted
    by length; the remainder of a class (less than one batch) passes to the next class, so that no gradient step
    runs on an almost empty batch. The last remainder is padded with zero-weight cells if it is at least
    half a batch, otherwise dropped (random draw: no decision is systematically discarded)."""
    out, carry = [], np.zeros(0, np.int64)
    for i, L in enumerate(classes):
        lo = classes[i - 1] if i else -1
        idx = rng.permutation(np.concatenate([carry, np.nonzero(need & (nvalid > lo) & (nvalid <= L))[0]]))
        k = len(idx) // msize
        out += [(L, idx[j * msize:(j + 1) * msize], msize) for j in range(k)]
        carry = idx[k * msize:]
    if len(carry) and (len(carry) >= msize // 2 or not out):
        out.append((classes[-1], np.concatenate([carry, np.repeat(carry[:1], msize - len(carry))]), len(carry)))
    return [out[i] for i in rng.permutation(len(out))]


def take_batch(fb, idx, L, n_real):
    """Update batch: cells idx of the flattened buffer fb ([N, ...]), observations cut to L tokens. The index of the
    chosen action (token x verb, then special actions) is re-based onto the new length."""
    mb = jax.tree.map(lambda x: x[idx], fb)
    o = mb["o"]
    T = o.valid.shape[-1]
    o = o._replace(**{k: getattr(o, k)[:, :L] for k in OBS_T})
    ai = mb["idx"]
    return dict(mb, o=o, idx=jnp.where(ai >= T * core.NV, ai - (T - L) * core.NV, ai),
                need=mb["need"] & (jnp.arange(len(idx)) < n_real))


def paired(d, x):
    """Decks [P, 2, ...] -> [4P, 2, ...] and network side [4P]: each pair of decks (A, B) is played 4 times, the network
    with A then with B, first then second. The strength of the decks and the first-player advantage cancel out."""
    sw = lambda z: z[:, ::-1]
    D = np.stack([d, d, sw(d), sw(d)], 1).reshape((-1,) + d.shape[1:])
    X = np.stack([x, x, sw(x), sw(x)], 1).reshape((-1,) + x.shape[1:])
    return D, X, np.tile([0, 1, 0, 1], len(d))


def play_eval(a, env, fns, params, bank, d, x, side, key, kind, oparams=None, greedy=False):
    """Games d [B, 2, 40], x [B, 2, 15]; the network plays player side[b], the opponent is random (kind 0), the
    network oparams (1) or the heuristic (2). Returns (finished [B], result [B]: 1 win, -1 loss, 0 draw / not finished)."""
    act, values, act_step, gae, update, enc, obs_b = fns
    tok, msk = model.card_tokens(bank)
    cemb = enc(params, tok, msk)
    opp = (params if oparams is None else oparams, cemb if oparams is None else enc(oparams, tok, msk), jnp.asarray(kind))
    s, m = env.new(bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    side = jnp.asarray(side)
    sd = np.asarray(side)
    fin = np.zeros(env.b, bool)
    res = np.zeros(env.b, int)
    used = jnp.ones(env.b, bool)                       # no replacement of finished games
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        s, m, f, used, _, dn, w = act_step(params, cemb, bank, s, m, f, k, side, s, m, used, opp, jnp.asarray(greedy))
        dn, w = np.asarray(dn), np.asarray(w)
        new = dn & ~fin
        res[new] = (w[new] == sd[new]).astype(int) - (w[new] == 1 - sd[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fin, res


def summary(fin, res, pre="eval"):
    n = max(int(fin.sum()), 1)
    return {f"{pre}_games": int(fin.sum()), f"{pre}_win": float((res[fin] == 1).sum() / n),
            f"{pre}_loss": float((res[fin] == -1).sum() / n)}


def evaluate(a, env, fns, params, key, kind, oparams=None, pre="eval", greedy=False, struct=False):
    """Held-out procedural evaluation bank (never seen), decks played in pairs (see paired), fixed seed. By default a
    reference bank (Gen without archetypes) and random decks, comparable from one run to the next; struct: bank with
    archetypes (a.arch) and structured decks, closer to the training decks and to real decks."""
    if struct:
        cset, bank = bank_of(a.eval_bank, a, arch=a.arch or 0.5)
        d, x = cardgen.structured_decks(cset, np.random.default_rng(12345), env.b // 4)
    else:
        cset, bank = bank_of(a.eval_bank, a)
        d, x = cardgen.random_decks(cset, np.random.default_rng(12345), env.b // 4)
    D, X, side = paired(d, x)
    return summary(*play_eval(a, env, fns, params, bank, D, X, side, key, kind, oparams, greedy), pre)


_REAL = {}


def real_bench(a, env, fns, params, key, kind, oparams=None, pre="real", greedy=False):
    """Real cards (train/realcards.py): all the matchups (network deck, opponent deck, side), repeated to fill the
    batch. Returns the overall win rate and the win rate per network deck."""
    import realcards
    if not _REAL:
        _REAL["v"] = realcards.load()
    bank, names, main, extra = _REAL["v"]
    nd = len(names)
    cfg = np.array([(i, j, sd) for i in range(nd) for j in range(nd) for sd in (0, 1)])
    cfg = cfg[np.arange(env.b) % len(cfg)]
    i, j, side = cfg[:, 0], cfg[:, 1], cfg[:, 2]
    p0, p1 = np.where(side == 0, i, j), np.where(side == 0, j, i)
    D = np.stack([main[p0], main[p1]], 1)
    X = np.stack([extra[p0], extra[p1]], 1)
    fin, res = play_eval(a, env, fns, params, bank, D, X, side, key, kind, oparams, greedy)
    out = summary(fin, res, pre)
    for k, nm in enumerate(names):
        sel = fin & (i == k)
        out[f"{pre}_{nm}"] = float((res[sel] == 1).sum() / max(int(sel.sum()), 1))
    return out


def precompile(a, env, eval_env, fns, params, opt_state, side_tr, opp_tr, n):
    """Pre-compiles, n at a time, the heavy functions with the exact shapes of the run: training and evaluation
    act_step (full engine), new games, values, GAE, update. Tracing stays sequential (Python); XLA compilation
    releases the GIL, hence the parallelism. The executables go into the persistent cache (core.py) where the real
    calls find them; total time ~ the longest compilation instead of the sum.
    opp_tr: training opponent, in the form passed to act_step."""
    from concurrent.futures import ThreadPoolExecutor
    act, values, act_step, gae, update, enc, obs_b = fns
    t0 = time.time()
    cset, bank = bank_of(a.eval_bank, a)
    tok, msk = model.card_tokens(bank)
    sds = lambda t: jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, weak_type=getattr(x, "weak_type", False)), t)
    cemb = sds(jax.eval_shape(enc, params, tok, msk))
    key = jax.random.PRNGKey(0)
    jobs = []
    banks = [bank]
    if a.bench_every:
        import realcards
        if not _REAL:
            _REAL["v"] = realcards.load()
        rb = _REAL["v"][0]
        if jax.tree.map(lambda x: (x.shape, x.dtype), rb) != jax.tree.map(lambda x: (x.shape, x.dtype), bank):
            banks.append(rb)
    for e in ([env] if eval_env is env else [env, eval_env]):
        d, x = cardgen.random_decks(cset, np.random.default_rng(0), e.b)
        ga = (bank, jnp.asarray(d), jnp.asarray(x), jax.random.split(key, e.b))
        jobs.append((f"new games B={e.b}", e.new, ga))
        s, m = sds(jax.eval_shape(e.new, *ga))
        f = jax.ShapeDtypeStruct((e.b,), jnp.int32)
        side = jax.ShapeDtypeStruct((e.b,), jnp.int32)
        for i, bk in enumerate(banks if e is eval_env and a.eval_every else []):   # no evaluation: nothing to compile
            jobs.append((f"act_step eval B={e.b} bank {i}", act_step,
                         (params, cemb, bk, s, m, f, key, side, s, m, jax.ShapeDtypeStruct((e.b,), bool),
                          (params, cemb, jnp.asarray(1)), jnp.asarray(False))))
        if e is env:
            tr_args = (params, cemb, bank, s, m, f, key, side_tr, s, m, jax.ShapeDtypeStruct((e.b,), bool),
                       opp_tr, jnp.asarray(False))
            jobs.append((f"act_step training B={e.b}", act_step, tr_args))
            out = jax.eval_shape(act_step, *tr_args)
            tr, done, win = out[4], out[5], out[6]
            jobs.append(("values", values, (params, cemb, bank, s, m)))
            last = jax.eval_shape(values, params, cemb, bank, s, m)
            st = lambda x: jax.ShapeDtypeStruct((a.steps,) + x.shape, x.dtype)
            jobs.append(("gae", gae, (st(tr["v"]), st(tr["need"]), st(tr["pl"]), st(done), st(win), last)))
            if a.update_old:
                mb = jax.tree.map(lambda x: jax.ShapeDtypeStruct((a.steps, e.b // a.mb) + x.shape[1:], x.dtype), tr)
                mb["adv"] = mb["ret"] = jax.ShapeDtypeStruct((a.steps, e.b // a.mb), jnp.float32)
                jobs.append(("update", update, (params, opt_state, tok, msk, mb)))
            else:
                M = mb_size(a)
                for L in tok_classes(a):
                    mb = jax.tree.map(lambda x: jax.ShapeDtypeStruct((M,) + x.shape[1:], x.dtype), tr)
                    mb["o"] = mb["o"]._replace(**{k: jax.ShapeDtypeStruct((M, L) + getattr(tr["o"], k).shape[2:],
                                                                         getattr(tr["o"], k).dtype) for k in OBS_T})
                    mb["adv"] = mb["ret"] = jax.ShapeDtypeStruct((M,), jnp.float32)
                    jobs.append((f"update, {L} tokens", update, (params, opt_state, tok, msk, mb)))
    low = []
    for name, fn, args in jobs:
        t1 = time.time()
        low.append((name, fn.lower(*args)))
        print(f"precompilation: {name} traced ({time.time() - t1:.0f} s)", flush=True)

    def comp(x):
        t1 = time.time()
        x[1].compile()
        return x[0], time.time() - t1
    with ThreadPoolExecutor(n) as ex:
        for name, dt in ex.map(comp, low):
            print(f"precompilation: {name} compiled ({dt:.0f} s)", flush=True)
    print(f"precompilation: {len(jobs)} functions in {time.time() - t0:.0f} s ({n} threads)", flush=True)


def load_params(path):
    """Saved parameters (pickle, possibly gzip, f16 or f32) -> f32, completed (model.complete)."""
    import gzip
    import pickle
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as fh:
        p = pickle.load(fh)
    return model.complete(jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), p))


def fmt(ev):
    return " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items())


def main():
    a = parse()
    a.eval_b = a.eval_b or a.b
    model.set_compute(a.bf16)
    key = jax.random.PRNGKey(a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} steps={a.steps} d={a.d}", flush=True)
    params = model.init(jax.random.fold_in(key, 0), a.d, a.card_layers, a.layers)
    if a.init:
        params = load_params(a.init)
        print("start:", a.init, flush=True)
    print("parameters:", sum(x.size for x in jax.tree.leaves(params)), flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = opt.init(params)
    ref = load_params(a.ref_ckpt) if a.ref_ckpt else None
    snaps = {a.it0: params}                                # past versions for the evaluation (iteration -> parameters)
    env = Env(a.b, a.frac, a.decks, a.deck_pool, a.match, a.deck_caps, a.deck_profile)
    eval_env = env if a.eval_b == a.b else Env(a.eval_b, a.frac)
    fns = act, values, act_step, gae, update, enc, obs_b = make_fns(a, opt)
    bank_seed = a.seed * 100_000 + a.it0 // a.bank_every
    tot_dec, tot_games = 0, 0
    # opponents per batch slot, fixed for the whole run: self-play (side -1), styled heuristic (kind 3)
    # or an older version of the league (kind 1); the network plays side b % 2 of these games
    cat = np.zeros(a.b, int)
    nh, no = int(round(a.mix_heur * a.b)), int(round(a.mix_old * a.b))
    perm0 = rng.permutation(a.b)
    cat[perm0[:nh]], cat[perm0[nh:nh + no]] = 1, 2
    side_tr = jnp.asarray(np.where(cat == 0, -1, np.arange(a.b) % 2), jnp.int32)
    kind_tr = jnp.asarray(np.where(cat == 1, 3, np.where(cat == 2, 1, 0)), jnp.int32)
    league = [params]
    key, k = jax.random.split(key)
    sty = sample_style(k, a.b)
    if a.precompile:
        ce = jax.eval_shape(enc, params, *model.card_tokens(bank_of(a.eval_bank, a)[1]))
        opp_tr = (params, ce, kind_tr, sty) if a.mix_heur + a.mix_old > 0 else (params, ce, jnp.asarray(0))
        precompile(a, env, eval_env, fns, params, opt_state, side_tr, opp_tr, a.precompile)
    for it in range(a.it0, a.iters):
        t0 = time.time()
        t_games = 0.0                         # new games: NumPy decks + new_game + legal (excluding the loop itself)

        def games(k, prev):
            """env.games timed separately: first wait for the GPU work already queued (prev), otherwise it would be
            counted."""
            jax.block_until_ready(prev)
            t1 = time.time()
            r = env.games(bank, cset, rng, k)
            jax.block_until_ready(r)
            return r, time.time() - t1

        if it % a.bank_every == 0 or it == a.it0:
            cset, bank = bank_of(bank_seed, a, a.gen_revive, a.arch)
            bank_seed += 1
            tok, msk = model.card_tokens(bank)
            key, k = jax.random.split(key)
            (s, m), dt = games(k, params)
            t_games += dt
            gd, counted = (env.last_ids.copy(), np.zeros(a.b, bool)) if a.match else (None, None)
            f = jnp.full(a.b, -1, jnp.int32)
            key, k = jax.random.split(key)
            sty = sample_style(k, a.b)
        key, k = jax.random.split(key)
        (pool, pm), dt = games(k, params)
        t_games += dt
        used = jnp.zeros(a.b, bool)
        pool_ids, epochs, used0s = [env.last_ids], [], []      # pairing: the pool in use at each call
        cemb = enc(params, tok, msk)
        buf, dones, wins = [], [], []
        side = side_tr
        if a.mix_heur + a.mix_old > 0:
            op = league[int(rng.integers(len(league)))]
            opp0 = (op, enc(op, tok, msk), kind_tr, sty)
        else:
            opp0 = (params, cemb, jnp.asarray(0))
        jax.block_until_ready(cemb)
        t1 = time.time()
        t_in = 0.0                            # new games during the loop (to subtract from the loop)
        for t in range(a.steps):
            key, k = jax.random.split(key)
            used0 = used
            s, m, f, used, tr, d, w = act_step(params, cemb, bank, s, m, f, k, side, pool, pm, used, opp0, jnp.asarray(False))
            if len(opp0) > 3:                 # new style for the replaced games
                key, k = jax.random.split(key)
                sty = jnp.where((d & ~used0)[:, None], sample_style(k, a.b), sty)
                opp0 = opp0[:3] + (sty,)
            buf.append(tr)
            dones.append(d)
            wins.append(w)
            if a.match:
                epochs.append(len(pool_ids) - 1)
                used0s.append(used0)
            if t % 8 == 7 and float(used.mean()) > 0.25:      # fresh pool: each game of the pool is used only once
                key, k = jax.random.split(key)
                (pool, pm), dt = games(k, f)
                t_in += dt
                pool_ids.append(env.last_ids)
                used = jnp.zeros(a.b, bool)
        jax.block_until_ready(f)
        t_games += t_in
        t_loop = time.time() - t1 - t_in      # the act_step calls alone (+ the synchronisations every 8 calls)
        last = values(params, cemb, bank, s, m)
        bt = jax.tree.map(lambda *x: jnp.stack(x), *buf)
        done, win = jnp.stack(dones), jnp.stack(wins)
        if a.match:      # deck ranking: finished games of this rollout, then decks of the replacement games
            dn_h, wn_h, u0_h = np.asarray(done), np.asarray(win), np.asarray(jnp.stack(used0s))
            for t in range(a.steps):
                fin_t = dn_h[t] & ~counted & (wn_h[t] >= 0)
                env.report(gd[fin_t], wn_h[t][fin_t])
                counted |= dn_h[t]
                rf = dn_h[t] & ~u0_h[t]
                gd[rf] = pool_ids[epochs[t]][rf]
                counted[rf] = False
        adv, ret = gae(bt["v"], bt["need"], bt["pl"], done, win, last)
        need = bt["need"]
        nd = int(need.sum())
        mu = (adv * need).sum() / max(nd, 1)
        sd = jnp.sqrt((((adv - mu) * need) ** 2).sum() / max(nd, 1)) + 1e-8
        bt["adv"], bt["ret"] = (adv - mu) / sd, ret
        t_roll = time.time() - t0
        st = None
        n_up = 0
        if a.update_old:
            for ep in range(a.epochs):
                key, k = jax.random.split(key)
                perm = jax.random.permutation(k, a.b)
                for chunk in jnp.split(perm, a.mb):
                    mb = jax.tree.map(lambda x: x[:, chunk], bt)
                    params, opt_state, st = update(params, opt_state, tok, msk, mb)
                    n_up += 1
        else:
            # compact update: only the decisions of the trained network, in batches sorted by number of tokens
            fb = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), bt)
            valid = np.asarray(fb["o"].valid)
            assert (valid[:, :-1] >= valid[:, 1:]).all(), "valid tokens not at the head of the observation"
            need_np, nval = np.asarray(fb["need"]), valid.sum(-1)
            if it == a.it0:
                # Each batch length has its own GPU workspace (9.7 GB at 64 tokens, B x steps = 65536). If the small
                # ones go first, the allocator no longer has a contiguous block for the large one (RESOURCE_EXHAUSTED
                # on L4): a dummy step (zero weights, result discarded) on the largest reserves it first, and the
                # others fit inside it.
                c = tok_classes(a)[-1]
                jax.block_until_ready(update(params, opt_state, tok, msk,
                                             take_batch(fb, jnp.zeros(mb_size(a), jnp.int32), c, 0))[2])
            for ep in range(a.epochs):
                for L, idx, nr in plan_batches(need_np, nval, tok_classes(a), mb_size(a), rng):
                    params, opt_state, st = update(params, opt_state, tok, msk, take_batch(fb, jnp.asarray(idx), L, nr))
                    n_up += 1
        jax.block_until_ready(params)
        t_up = time.time() - t0 - t_roll
        if (it + 1) % a.league_every == 0:
            league = (league + [params])[-a.league_size:]
        ng = int(done.sum())
        tot_dec += nd
        tot_games += ng
        line = (f"it {it} : {nd} decisions ({nd / (time.time() - t0):.0f}/s), {ng} games finished ({time.time() - t0:.1f} s ; "
                f"rollout {t_roll:.1f} s, of which new games {t_games:.1f} s, act_step {t_loop:.1f} s = "
                f"{1000 * t_loop / a.steps:.0f} ms/call ; update {t_up:.1f} s, {n_up} batches"
                + (f" ; decks : {env.played.mean():.0f} games per deck, Elo ±{env.elo.std():.0f}" if a.match else "")
                + ") | " + " ".join(f"{k} {float(v):.3f}" for k, v in st.items()))
        first = a.eval_first and it == a.it0
        if a.eval_every and ((it + 1) % a.eval_every == 0 or first):
            key, k1, k2, k3 = jax.random.split(key, 4)
            line += " | " + fmt(evaluate(a, eval_env, fns, params, k1, 0, pre="rnd"))
            line += " | " + fmt(evaluate(a, eval_env, fns, params, k2, 2, pre="heur"))
            key, k4 = jax.random.split(key)
            line += " | " + fmt(evaluate(a, eval_env, fns, params, k4, 2, pre="sheur", struct=True))
            # against the version --old-lag iterations ago (or the oldest one kept, at the start: the initialisation)
            old = max([i for i in snaps if i <= it + 1 - a.old_lag] or [min(snaps)])
            ev = evaluate(a, eval_env, fns, params, k3, 1, snaps[old])
            line += f" | vs it {old} : win {ev['eval_win']:.3f} loss {ev['eval_loss']:.3f}"
            if ref is not None:
                key, k5 = jax.random.split(key)
                line += " | " + fmt(evaluate(a, eval_env, fns, params, k5, 1, ref, pre="ref", struct=True))
            if a.bench_every and ((it + 1) % a.bench_every == 0 or first):
                key, k1, k2 = jax.random.split(key, 3)
                line += " | " + fmt(real_bench(a, eval_env, fns, params, k1, 0, pre="real_rnd"))
                line += " | " + fmt(real_bench(a, eval_env, fns, params, k2, 2, pre="real_heur"))
                if ref is not None:
                    key, k6 = jax.random.split(key)
                    line += " | " + fmt(real_bench(a, eval_env, fns, params, k6, 1, ref, pre="real_ref"))
            snaps[it + 1] = params
            snaps = {i: q for i, q in snaps.items() if i >= it + 1 - 2 * a.old_lag}
        print(line, flush=True)
        if a.save:
            import pickle
            with open(a.save, "wb") as fh:
                pickle.dump(jax.device_get(params), fh)
            if a.snap_every and (it + 1) % a.snap_every == 0:
                import gzip
                with gzip.open(f"{a.save}.it{it + 1}.gz", "wb") as fh:
                    pickle.dump(jax.tree.map(lambda x: np.asarray(x, np.float16), jax.device_get(params)), fh)
        if a.stop_file and os.path.exists(a.stop_file):
            print(f"stop requested ({a.stop_file}) after iteration {it}", flush=True)
            break
    print(f"total: {tot_dec} decisions, {tot_games} games", flush=True)
    if a.prof:
        profile(a, fns, params, bank, tok, msk, s, m, f, pool, pm, used, key)


def profile(a, fns, params, bank, tok, msk, s, m, f, pool, pm, used, key):
    """Each piece of a rollout call timed alone, a.prof times on the same state (pure functions): the end-of-rollout
    state (mid-game) and the pool (game starts)."""
    act, values, act_step, gae, update, enc, obs_b = fns
    cemb = enc(params, tok, msk)
    sb = jax.jit(lambda bank, s, x: core.step_batch(bank, s, x, a.frac))
    side = jnp.full(a.b, -1, jnp.int32)
    opp0 = (params, cemb, jnp.asarray(0))

    def tm(name, fn, *xs):
        r = fn(*xs)
        jax.block_until_ready(r)
        t0 = time.time()
        for _ in range(a.prof):
            r = fn(*xs)
        jax.block_until_ready(r)
        print(f"prof {name:45s} {1000 * (time.time() - t0) / a.prof:7.1f} ms", flush=True)
        return r

    for lab, (s1, m1) in (("mid-game", (s, m)), ("start", (pool, pm))):
        f1 = f if lab == "mid-game" else jnp.full(a.b, -1, jnp.int32)
        print(f"prof {lab} of game: mean turn {float(s1.turn.mean()):.1f}, finished {int(s1.done.sum())}", flush=True)
        ae, _ = tm(f"{lab} act", act, params, cemb, bank, s1, m1, f1, key)
        rnd = jax.jit(lambda k, m, f: jnp.where(f >= 0, f, jax.random.categorical(k, jnp.where(m, 0.0, -1e9)).astype(jnp.int32)))
        ar = rnd(key, m1, f1)
        tm(f"{lab} step_batch (network actions)", sb, bank, s1, ae)
        tm(f"{lab} step_batch (random actions)", sb, bank, s1, ar)
        tm(f"{lab} act_step", act_step, params, cemb, bank, s1, m1, f1, key, side, pool, pm, used, opp0, jnp.asarray(False))


if __name__ == "__main__":
    main()
