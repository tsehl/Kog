"""Behavior tests on real cards: the tested player (randomly drawn network, deterministic network,
heuristic or random) plays the 4 reference decks against the heuristic. At each of its decisions, the opportunities
and the basic-play errors are counted (attacks lost in advance, ending the turn without Summoning, wrong target of
Monster Reborn, defensive Trap not activated...), plus, per card, the activations and the end-of-turn moments where
it was still playable.

Usage: python train/behavior.py CKPT [--b 1024 --eval-steps 600 --bf16 ...] (same options as ppo.py);
BEHAV_WHO=net,random to test only some players"""
import os
import sys
import time

import numpy as np

import ppo
import jax
import jax.numpy as jnp
import optax

import core
import model
import realcards
from ppo import load_params

C = dict(raigeki=12580477, dark_hole=53129443, mst=5318639, hfd=18144506, reborn=83764718,
         mirror=44095762, sakuretsu=56120475, trap_hole=4206964)
PROBES = [  # (name, description of the error)
    ("stronger_target", "attacks a target stronger (ATK, face-up) than the attacker"),
    ("missed_target", "does not take a target it would destroy, although there is one"),
    ("losing_attack", "declares an attack although all opposing monsters (face-up, ATK) are stronger"),
    ("missed_direct", "opposing field empty, attack possible, but leaves the Battle Phase"),
    ("end_without_summon", "ends its turn although it could Summon or Set a monster"),
    ("not_strongest", "Summons from the hand a monster weaker than another summonable one"),
    ("spell_kept", "ends its turn with an activatable Spell in hand"),
    ("dark_hole_losing", "Dark Hole although its total ATK on the field exceeds the opponent's"),
    ("mst_no_target", "MST with no opposing Spell / Trap on the field"),
    ("reborn_target", "Monster Reborn: does not take back the strongest monster available"),
    ("trap_passed", "passes although Mirror Force / Sakuretsu / Trap Hole is activatable"),
    ("set_dominant", "Sets in Defense Position a monster stronger than the whole face-up opposing field"),
]
K = len(PROBES)


def probe(bank, s, m, a):
    """One game: flags [K, 2] (opportunity, error) for the decision a taken in the state s (legal mask m);
    then [2, C] per card of the bank: activated, kept playable in hand at end of turn."""
    n = s.cid.shape[0]
    me, op = s.dec_player, 1 - s.dec_player
    code = bank.code[s.cid]
    mon = s.zone == core.Z_MZONE
    my_mon, op_mon = mon & (s.ctrl == me), mon & (s.ctrl == op)
    fu = (s.pos & core.P_FACEUP) != 0
    fatk = s.pos == core.P_FU_ATK
    atk, df = s.atk_c, s.df_c
    batk = bank.atk[s.cid]
    hand = (s.zone == core.Z_HAND) & (s.ctrl == me)
    op_st = (s.zone == core.Z_SZONE) & (s.ctrl == op)
    nva = core.NV * n
    mv = m[:nva].reshape(core.NV, n)
    verb, slot = jnp.where(a < nva, a // n, -1), a % n
    sp = a - nva
    idle, bat = s.dec == core.D_IDLE, s.dec == core.D_BATTLE
    to_ep = idle & (sp == core.A_TO_EP)
    act = (verb >= core.V_ACT0)
    act_on = lambda c: act & (code[slot] == c)
    legal_act = mv[core.V_ACT0:].any(0)                                  # [n] activatable card (one of its effects)

    # attack: choice of the target (D_TARGET, V_SELECT on the target)
    tg = (s.dec == core.D_TARGET) & (verb == core.V_SELECT)
    ak = atk[jnp.maximum(s.atk, 0)]
    kills = op_mon & fu & ((fatk & (atk < ak)) | (~fatk & (df < ak))) & mv[core.V_SELECT]
    f = [
        (tg, tg & fatk[slot] & (atk[slot] > ak)),
        (tg & kills.any(), tg & kills.any() & ~kills[slot]),
        (bat & (verb == core.V_ATTACK), bat & (verb == core.V_ATTACK) & op_mon.any()
         & (~op_mon | (fatk & (atk > atk[slot]))).all()),
        (bat & ~op_mon.any() & mv[core.V_ATTACK].any(), bat & ~op_mon.any() & mv[core.V_ATTACK].any()
         & ((sp == core.A_TO_M2) | (sp == core.A_TO_EP))),
        (to_ep, to_ep & ((mv[core.V_SUMMON] & hand).any() | mv[core.V_MSET].any())),
        (idle & (verb == core.V_SUMMON) & hand[slot], idle & (verb == core.V_SUMMON) & hand[slot]
         & (mv[core.V_SUMMON] & hand & (batk > batk[slot])).any()),
        (to_ep, to_ep & (legal_act & hand & ((bank.ctype[s.cid] & core.TYPE_SPELL) != 0)).any()),
        (act_on(C["dark_hole"]), act_on(C["dark_hole"]) & ((atk * my_mon).sum() > (atk * op_mon).sum())),
        (act_on(C["mst"]), act_on(C["mst"]) & ~op_st.any()),
    ]
    # Monster Reborn: chosen target (D_SELECT during the activation / resolution of Reborn)
    rb = (s.dec == core.D_SELECT) & (verb == core.V_SELECT) & (s.ch_src >= 0) \
        & (bank.code[s.cid[jnp.maximum(s.ch_src, 0)]] == C["reborn"])
    f.append((rb, rb & (mv[core.V_SELECT] & (batk > batk[slot])).any()))
    trap = (code == C["mirror"]) | (code == C["sakuretsu"]) | (code == C["trap_hole"])
    tw = (s.dec == core.D_CHAIN) & (legal_act & trap).any()
    f.append((tw, tw & (sp == core.A_PASS)))
    ms = idle & (verb == core.V_MSET)
    f.append((ms, ms & (batk[slot] >= 1000) & ~(op_mon & fu & (atk >= batk[slot])).any()))
    flags = jnp.array([[o, e & o] for o, e in f], jnp.int32)
    nc = bank.code.shape[0]
    used = jnp.zeros(nc, jnp.int32).at[s.cid[slot]].add(act.astype(jnp.int32))
    held = jnp.zeros(nc, jnp.int32).at[s.cid].add((to_ep & legal_act & hand).astype(jnp.int32))
    return flags, jnp.stack([used, held])


def run(a, env, fns, params, bank, D, X, side, key, who):
    """who: 'net', 'net_det', 'heuristic' or 'random' (player side) against the heuristic."""
    act = fns[0]
    tok, msk = model.card_tokens(bank)
    cemb = fns[5](params, tok, msk)

    @jax.jit
    def tick(s, m, f, key):
        k1, k2, k3 = jax.random.split(key, 3)
        if who.startswith("net"):
            mine = act(params, cemb, bank, s, m, f, k1, jnp.asarray(who == "net_det"))[0]
        elif who == "heuristic":
            mine = ppo.heuristic(bank, s, m, f, k1)
        else:
            mine = jnp.where(f >= 0, f, jax.random.categorical(k1, jnp.where(m, 0.0, -1e9)).astype(jnp.int32))
        sd = jnp.asarray(side)
        a_ = jnp.where(s.dec_player == sd, mine, ppo.heuristic(bank, s, m, f, k2))
        dec = (s.dec_player == sd) & (f < 0) & ~s.done
        fl, pc = jax.vmap(lambda s, m, a: probe(bank, s, m, a))(s, m, a_)
        fl = (fl * dec[:, None, None]).sum(0)
        pc = (pc * dec[:, None, None]).sum(0)
        s, m, f = core.step_batch(bank, s, a_, a.frac)
        return s, m, f, fl, pc

    s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, env.b))
    f = jnp.full(env.b, -1, jnp.int32)
    fl_t, pc_t = 0, 0
    fin, res = np.zeros(env.b, bool), np.zeros(env.b, int)
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        s, m, f, fl, pc = tick(s, m, f, k)
        fl_t, pc_t = fl_t + np.asarray(fl), pc_t + np.asarray(pc)
        dn, w = np.asarray(s.done), np.asarray(s.winner)
        new = dn & ~fin
        res[new] = (w[new] == side[new]).astype(int) - (w[new] == 1 - side[new]).astype(int)
        fin |= dn
        if fin.all():
            break
    return fl_t, pc_t, fin, res


def main():
    path = sys.argv[1]
    sys.argv = sys.argv[:1] + sys.argv[2:]
    a = ppo.parse()
    model.set_compute(a.bf16)
    params = load_params(path)
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    bank, names, main_, extra = realcards.load()
    nd = len(names)
    cfg = np.array([(i, j, s) for i in range(nd) for j in range(nd) for s in (0, 1)])[np.arange(env.b) % (2 * nd * nd)]
    p0 = np.where(cfg[:, 2] == 0, cfg[:, 0], cfg[:, 1])
    p1 = np.where(cfg[:, 2] == 0, cfg[:, 1], cfg[:, 0])
    D, X, side = np.stack([main_[p0], main_[p1]], 1), np.stack([extra[p0], extra[p1]], 1), cfg[:, 2]
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; {path}", flush=True)
    cnames = realcards.card_names()
    out = {}
    for who in os.environ.get("BEHAV_WHO", "net,net_det,heuristic,random").split(","):
        t0 = time.time()
        fl, pc, fin, res = run(a, env, fns, params, bank, D, X, side, jax.random.PRNGKey(3), who)
        out[who] = (fl, pc)
        win = (res[fin] == 1).mean() if fin.any() else float("nan")
        print(f"== {who} ({time.time() - t0:.0f} s) : {int(fin.sum())} games finished, win rate against the heuristic "
              f"{win:.3f}", flush=True)
    print("\nerror (% of opportunities; opportunities):", flush=True)
    print(f"{'':20s}" + "".join(f"{w:>22s}" for w in out), flush=True)
    for k, (nm, desc) in enumerate(PROBES):
        cells = "".join(f"{100 * o[0][k, 1] / max(o[0][k, 0], 1):14.1f} ({o[0][k, 0]:>6d})" for o in out.values())
        print(f"{nm:20s}{cells}   {desc}", flush=True)
    code = np.asarray(bank.code)
    print("\nper card (activations / end of turn where it stayed activatable in hand):", flush=True)
    tot = sum(o[1] for o in out.values())
    for c in np.argsort(-tot[1]):
        if tot[0, c] + tot[1, c] == 0 or not cnames[c]:
            continue
        nm = cnames[c]
        print(f"{nm[:32]:32s}" + "".join(f"{o[1][0, c]:>10d} / {o[1][1, c]:<9d}" for o in out.values()), flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
