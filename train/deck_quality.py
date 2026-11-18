"""Quality of the decks as seen from the games: what the network actually gets the opportunity to play.

The network plays against itself (sampled actions), on evaluation banks with archetypes and structured or random
decks, and on the 4 decks of real cards (reference point). Per finished game:
- dead cards: cards of the main deck that reached the hand and were never played (neither Set, nor Summoned, nor
  activated);
- Extra Deck: monsters actually Summoned, by type; Ritual Monsters that reached the field;
- effects: activations, distinct effect shapes activated (shape = kind + sequence of opcodes of the three programs,
  read from the bank), share of the shapes present in the decks that are activated at least once in the batch,
  concentration of the activations on the most played shapes;
- length: turns, decisions of the network.

The activations are read from the applied actions (activation verb on a card); the locations are sampled between two
calls of step_batch (a card Set then destroyed within the same call is not seen: rare).

Usage: python train/deck_quality.py CKPT [--b 1024 --eval-steps 500 --bf16]   (DQ_BANKS=2; DQ_SKEL=p: banks
        with a share p of skeleton cards, if the generator offers them)"""
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "gen")]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import cardgen  # noqa: E402
import core  # noqa: E402
import model  # noqa: E402
import ppo  # noqa: E402
import realcards  # noqa: E402


def shapes_of(bank):
    """Shape identifier of each (card, effect): kind + opcodes of the programs; -1 for an empty effect."""
    ek, prog = np.asarray(bank.ekind), np.asarray(bank.prog)[..., 0]          # [C, E], [C, E, 3, P]
    ids, table = np.full(ek.shape, -1), {}
    for c in range(ek.shape[0]):
        for e in range(ek.shape[1]):
            if ek[c, e] != core.K_NONE:
                key = (int(ek[c, e]),) + tuple(tuple(int(v) for v in pr if v != core.OP_END) for pr in prog[c, e])
                ids[c, e] = table.setdefault(key, len(table))
    return ids, len(table)


def play(a, env, fns, params, bank, D, X, key):
    act, _, act_step = fns[0], fns[1], fns[2]
    cemb = fns[5](params, *model.card_tokens(bank))
    s, m = env.new(bank, jnp.asarray(D), jnp.asarray(X), jax.random.split(key, env.b))
    B, N = s.cid.shape
    f = jnp.full(B, -1, jnp.int32)
    side = jnp.full(B, -1, jnp.int32)
    used = jnp.ones(B, bool)
    opp = (params, cemb, jnp.asarray(0))
    cid, zone0, owner = np.asarray(s.cid), np.asarray(s.zone), np.asarray(s.owner)
    nva = core.NV * N
    hand, field, acted = np.zeros((B, N), bool), np.zeros((B, N), bool), np.zeros((B, N), bool)
    actv = np.zeros((B, N, core.E), np.int32)          # activations per (card, effect)
    verbs = np.zeros((B, core.NV), np.int64)
    dec = np.zeros(B, np.int64)
    fin = np.zeros(B, bool)
    turns = np.zeros(B, np.int64)
    rows = np.arange(B)
    for t in range(a.eval_steps):
        key, k = jax.random.split(key)
        f0, alive = np.asarray(f), ~fin
        s, m, f, used, tr, dn, w = act_step(params, cemb, bank, s, m, f, k, side, s, m, used, opp, jnp.asarray(False))
        idx, slot_of = np.asarray(tr["idx"]), np.asarray(tr["o"].slot)
        T = slot_of.shape[1]
        net = np.where(idx < T * core.NV, (idx % core.NV) * N + slot_of[rows, np.minimum(idx // core.NV, T - 1)], nva)
        ae = np.where(f0 >= 0, f0, net)                               # applied action (forced, otherwise the network's)
        on = alive & (ae >= 0) & (ae < nva)
        v, sl = ae // N, ae % N
        np.add.at(verbs, (rows[on], v[on]), 1)
        acted[rows[on], sl[on]] = True
        isact = on & (v >= core.V_ACT0)
        np.add.at(actv, (rows[isact], sl[isact], v[isact] - core.V_ACT0), 1)
        dec += alive & (f0 < 0)
        z = np.asarray(s.zone)
        live = alive[:, None]
        hand |= live & (z == core.Z_HAND)
        field |= live & ((z == core.Z_MZONE) | (z == core.Z_SZONE))
        dn = np.asarray(dn)
        turns = np.where(dn & ~fin, np.asarray(s.turn), turns)
        fin |= dn
        if fin.all():
            break
    return dict(cid=cid, zone0=zone0, owner=owner, hand=hand, field=field, acted=acted, actv=actv, verbs=verbs, dec=dec,
                fin=fin, turns=turns)


def report(name, bank, runs, skel_mask=None):
    ct = np.asarray(bank.ctype)
    shp, nshape = shapes_of(bank)
    out = collections.OrderedDict()
    cat = lambda r, k: np.concatenate([x[k][x["fin"]] for x in r])
    cid, zone0 = cat(runs, "cid"), cat(runs, "zone0")
    hand, field, acted, actv = cat(runs, "hand"), cat(runs, "field"), cat(runs, "acted"), cat(runs, "actv")
    G = len(cid)
    t = ct[cid]
    main = zone0 == core.Z_DECK
    played = field | acted
    drawn = main & hand
    dead = drawn & ~played
    out["games finished"] = f"{G}"
    out["turns per game"] = f"{cat(runs, 'turns').mean():.1f}"
    out["decisions per game"] = f"{cat(runs, 'dec').mean():.0f}"
    out["drawn cards never played"] = f"{100 * dead.sum() / max(drawn.sum(), 1):.1f} %"
    for lab, bit in (("  monsters", core.TYPE_MONSTER), ("  Spells", core.TYPE_SPELL), ("  Traps", core.TYPE_TRAP)):
        k = (t & bit) != 0
        out[lab] = f"{100 * (dead & k).sum() / max((drawn & k).sum(), 1):.1f} %"
    lv = np.asarray(bank.level)[cid]
    mon = (t & core.TYPE_MONSTER) != 0
    for lab, k in (("  monsters of Level 1 to 4", mon & (lv <= 4)), ("  monsters of Level 5 and 6", mon & (lv >= 5) & (lv <= 6)),
                   ("  monsters of Level 7 and up", mon & (lv >= 7))):
        out[lab] = (f"{100 * (dead & k).sum() / max((drawn & k).sum(), 1):.1f} % "
                    f"({100 * (drawn & k).sum() / max((drawn & mon).sum(), 1):.0f} % of monsters)")
    # big monsters (Level 5+, excluding Ritual Monsters): loaded decks (>= 9) or light ones, per player
    owner = cat(runs, "owner")
    big = main & mon & (lv >= 5) & ((t & core.TYPE_RITUAL) == 0)
    for p in (0, 1):
        own = owner == p
        nb = (big & own).sum(1, keepdims=True)
        for lab, sel in (("loaded decks (9 big monsters or more)", nb >= 9), ("light decks (fewer than 9)", nb < 9)):
            key_d, key_n = f"_{lab}_d", f"_{lab}_n"
            out[key_d] = out.get(key_d, 0) + int((dead & big & own & sel).sum())
            out[key_n] = out.get(key_n, 0) + int((drawn & big & own & sel).sum())
            out[f"_{lab}_f"] = out.get(f"_{lab}_f", 0) + int((field & big & own & sel).sum())
            out[f"_{lab}_g"] = out.get(f"_{lab}_g", 0) + int(sel.sum())
    for lab in ("loaded decks (9 big monsters or more)", "light decks (fewer than 9)"):
        d_, n_, f_, g_ = (out.pop(f"_{lab}_{k}") for k in "dnfg")
        out[f"dead big monsters, {lab}"] = f"{100 * d_ / max(n_, 1):.1f} % ({100 * g_ / (2 * G):.0f} % of decks)"
        out[f"big monsters on the field per deck, {lab}"] = f"{f_ / max(g_, 1):.2f}"
    ext = (zone0 == core.Z_EXTRA) & (zone0 != 0)
    summ = ext & field
    if skel_mask is not None and skel_mask.any():
        sk = skel_mask[cid]
        for lab, k in (("  skeleton cards", sk & ~mon), ("  hand-written cards", ~sk & ~mon)):
            out[lab + " (Spells / Traps)"] = f"{100 * (dead & k).sum() / max((drawn & k).sum(), 1):.1f} %"
        for lab, k in (("  skeleton monsters, Level 1 to 4", sk & mon & (lv <= 4)),
                       ("  hand-written effect monsters, Level 1 to 4", ~sk & mon & (lv <= 4) & ((t & 0x20) != 0))):
            out[lab] = f"{100 * (dead & k).sum() / max((drawn & k).sum(), 1):.1f} %"
        a_sk = (actv.sum(-1) * (sk & played)).sum() / max((sk & played).sum(), 1)
        a_hm = (actv.sum(-1) * (~sk & played & ((t & (0x20 | core.TYPE_SPELL | core.TYPE_TRAP)) != 0))).sum() / \
            max((~sk & played & ((t & (0x20 | core.TYPE_SPELL | core.TYPE_TRAP)) != 0)).sum(), 1)
        out["activations per card played: skeleton / hand-written"] = f"{a_sk:.2f} / {a_hm:.2f}"
    out["Extra Deck monsters Summoned per game"] = f"{summ.sum() / G:.2f} of {ext.sum() / G:.0f}"
    out["games with an Extra Deck Summon"] = f"{100 * summ.any(1).mean():.1f} %"
    for lab, bit in (("  Fusion", core.TYPE_FUSION), ("  Synchro", core.TYPE_SYNCHRO), ("  Xyz", core.TYPE_XYZ),
                     ("  Link", core.TYPE_LINK)):
        out[lab] = f"{100 * (summ & ((t & bit) != 0)).any(1).mean():.1f}% of games"
    rit = main & ((t & core.TYPE_RITUAL) != 0) & ((t & core.TYPE_MONSTER) != 0)
    out["games with a Ritual Monster on the field"] = f"{100 * (rit & field).any(1).mean():.1f} %"
    verbs = cat(runs, "verbs")
    out["effect activations per game"] = f"{verbs[:, core.V_ACT0:].sum() / G:.1f}"
    sid = shp[cid]                                                      # [G, N, E]
    has = (sid >= 0) & (main | ext)[:, :, None]
    per_game = [len(set(sid[g][(actv[g] > 0) & (sid[g] >= 0)].tolist())) for g in range(G)]
    out["distinct effect shapes activated per game"] = f"{np.mean(per_game):.1f}"
    present = np.unique(sid[has])
    hit = (actv > 0) & (sid >= 0)
    counts = np.bincount(sid[hit], weights=actv[hit], minlength=nshape)
    done = (counts[present] > 0).sum()
    out["shapes present in the decks, activated at least once"] = f"{done} / {len(present)} ({100 * done / max(len(present), 1):.0f} %)"
    top = np.sort(counts)[::-1]
    out["share of activations on the 10 most played shapes"] = f"{100 * top[:10].sum() / max(top.sum(), 1):.0f} %"
    return name, out


def main():
    path = sys.argv.pop(1)
    a = ppo.parse()
    model.set_compute(a.bf16)
    params = ppo.load_params(path)
    env = ppo.Env(a.b, a.frac)
    fns = ppo.make_fns(a, optax.adam(1e-4))
    nb = int(os.environ.get("DQ_BANKS", "2"))
    skel = float(os.environ.get("DQ_SKEL", "0"))
    print(f"jax {jax.__version__} {jax.devices()[0].device_kind} ; B={a.b} ; {path}", flush=True)
    cols = []
    kinds = [("structured", cardgen.structured_decks), ("random", cardgen.random_decks)]
    for name, build in kinds:
        runs, bank0 = [], None
        for bi in range(nb):
            if skel:
                cset = cardgen.Gen(a.eval_bank + bi, 0, a.arch or 0.5, skel=skel).make_set(a.main, a.extra)
                bank = cardgen.to_bank(cset)
            else:
                cset, bank = ppo.bank_of(a.eval_bank + bi, a, arch=a.arch or 0.5)
            kw = {}
            if build is cardgen.structured_decks:
                kw = dict(caps=os.environ.get("DQ_CAPS", "0") != "0", profile=os.environ.get("DQ_PROFILE", "0") != "0")
            caps = any(kw.values())
            d, x = build(cset, np.random.default_rng(555 + bi), a.b, **kw)
            r = play(a, env, fns, params, bank, d, x, jax.random.PRNGKey(41 + bi))
            cols.append(report(f"{name} b{bi}" + (" prof." if kw.get("profile") else " caps" if caps else ""), bank, [r], np.array([bool(c.get("skel")) for c in cset])))
            print(f"  {name}, bank {bi}: {int(r['fin'].sum())} games finished", flush=True)
    if not skel:
        bank, names, main_, extra = realcards.load()
        nd = len(names)
        cfg = np.array([(i, j) for i in range(nd) for j in range(nd)])[np.arange(a.b) % (nd * nd)]
        D, X = np.stack([main_[cfg[:, 0]], main_[cfg[:, 1]]], 1), np.stack([extra[cfg[:, 0]], extra[cfg[:, 1]]], 1)
        r = play(a, env, fns, params, bank, D, X, jax.random.PRNGKey(77))
        cols.append(report("real cards", bank, [r]))
    keys = list(cols[0][1])
    print("\n" + f"{'':62s}" + "".join(f"{n:>24s}" for n, _ in cols))
    for k in keys:
        print(f"{k:62s}" + "".join(f"{o[k]:>24s}" for _, o in cols))


if __name__ == "__main__":
    main()
