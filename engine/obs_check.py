"""Tests of the observation (mixed games, random slots):
1. every legal action corresponds to a token (nothing is truncated);
2. the observation does not depend on hidden information: the identities of the cards the observer cannot know are
   shuffled (its own deck: only permuted, since it knows the list) and the slots are renumbered; the observation
   must remain identical (except `slot`);
3. positive control: changing a visible card does change the observation;
4. `deck` = the player's starting list minus its cards that it sees elsewhere (computation of an external client),
   and positive control: changing a card of its deck changes `deck`.
"""
import collections

import jax
import jax.numpy as jnp
import numpy as np

import cards
import core
import obs as O

B, D, C, TT = 32, 40, 670, 300
key = jax.random.PRNGKey(0)
bank = cards.test_bank(key, C)
CT = bank.code.shape[0]
decks, extras = cards.test_decks(jax.random.fold_in(key, 3), bank, B, D, C, 0.3, 6)
N = 2 * (D + 6)


def rollout(d, x, k):
    s = core.new_game(bank, d[0], d[1], jax.random.fold_in(k, 7), x[0], x[1])

    def body(s, kt):
        m = core.legal(bank, s)
        a = jax.random.categorical(kt, jnp.where(m, 0.0, -1e9))
        s2 = core.step(bank, s, a)
        s2 = jax.tree.map(lambda x, y: jnp.where(s.done, x, y), s, s2)
        return s2, s
    return jax.lax.scan(body, s, jax.random.split(k, TT))[1]


hist = jax.jit(jax.vmap(rollout))(decks, extras, jax.random.split(jax.random.fold_in(key, 4), B))
hist = jax.tree.map(np.asarray, hist)
live = ~hist.done
print("live states:", int(live.sum()))

observe = jax.jit(O.observe)


def state_at(b, t):
    return core.State(*[jnp.asarray(x[b, t]) for x in hist])


# 1. no legal action lost
legal_cnt = jax.jit(jax.vmap(jax.vmap(lambda s: core.legal(bank, s)[:core.NV * N].sum())))(hist)
tok_cnt = jax.jit(jax.vmap(jax.vmap(lambda s: O.observe(bank, s).act.sum())))(hist)
bad = live & (np.asarray(legal_cnt) != np.asarray(tok_cnt))
assert not bad.any(), f"{bad.sum()} states with legal actions without a token"
print("1. ok: all legal actions have a token")

# 2. invariance to hidden information
rng = np.random.default_rng(0)
FIELDS = [f for f in O.Obs._fields if f != "slot"]


legal_j = jax.jit(lambda s: core.legal(bank, s))


def perturb(s, p):
    s = jax.tree.map(np.array, s)
    faceup = (s.pos & (core.P_FU_ATK | core.P_FU_DEF)) != 0
    # during a search in its deck, p sees its whole deck
    search = (int(s.dec_player) == p) & np.asarray(legal_j(core.State(*[jnp.asarray(x) for x in s]))[:core.NV * N]
                                                    .reshape(core.NV, N).any(0) & (s.zone == core.Z_DECK)).any()
    hidden = (((s.zone == core.Z_DECK) & ~(search & (s.ctrl == p)))
              | (((s.zone == core.Z_HAND) | ((s.zone == core.Z_EXTRA) & ~faceup)) & (s.ctrl != p))  # face-up Pendulum Monsters: public
              | (((s.zone == core.Z_MZONE) | (s.zone == core.Z_SZONE)) & (s.ctrl != p) & ~faceup))
    own_deck = hidden & (s.zone == core.Z_DECK) & (s.ctrl == p)  # list known, order hidden: permutation only
    hidden = hidden & ~own_deck
    h = np.where(hidden)[0]
    cid = s.cid.copy()
    od = np.where(own_deck)[0]
    cid[od] = cid[rng.permutation(od)]
    # arbitrary hidden identities (only the location of a Set card is public: Spell/Trap or monster)
    spells = np.where(np.asarray(bank.ctype) & (core.TYPE_SPELL | core.TYPE_TRAP))[0]
    mons = np.where(np.asarray(bank.ctype) & core.TYPE_MONSTER)[0]
    cid[h] = np.select([s.zone[h] == core.Z_SZONE, s.zone[h] == core.Z_MZONE],
                       [rng.choice(spells, len(h)), rng.choice(mons, len(h))], rng.integers(0, CT, len(h)))
    s = s._replace(cid=cid)
    perm = rng.permutation(N)                      # renumbering of the slots
    inv = np.argsort(perm)
    sl = lambda x: np.where(x >= 0, inv[x], -1)
    s = s._replace(**{f: getattr(s, f)[perm] for f in ("cid", "zone", "seq", "ctrl", "owner", "pos", "st", "pzone", "suminfo", "reason", "tg", "tg_e",
                                                  "atk_c", "df_c", "batk_c", "en_t", "cav", "cav_t", "fl_c", "ib_c",
                                                  "proc", "fz_c", "rz_c", "sz_c", "xz_c", "lz_c", "uid")},
                   ov_par=sl(s.ov_par)[perm], eq_t=sl(s.eq_t)[perm], eq_mv=s.eq_mv[perm], tm_m=s.tm_m[:, perm], tm_v=s.tm_v[:, perm],
                   pend=sl(s.pend), ch_src=sl(s.ch_src), atk=sl(s.atk), atk_tgt=sl(s.atk_tgt), lk_src=sl(s.lk_src),
                   reg=s.reg[:, perm], lk_reg=s.lk_reg[:, :, perm], ch_targets=s.ch_targets[perm], lk_targets=s.lk_targets[:, perm], mv=s.mv[perm], ev_g=s.ev_g[:, perm], tev=s.tev[:, perm], tew=s.tew[:, perm], teg=s.teg[:, perm])
    return core.State(*[jnp.asarray(x) for x in s]), len(h)


nchk = 0
samples = [tuple(x) for x in np.argwhere(live)]
for i in rng.choice(len(samples), 400, replace=False):
    b, t = samples[i]
    s = state_at(b, t)
    for p in (0, 1):
        o1 = observe(bank, s, p)
        s2, nh = perturb(s, p)
        o2 = observe(bank, s2, p)
        diff = [f for f in FIELDS if not np.array_equal(getattr(o1, f), getattr(o2, f))]
        assert not diff, f"information leak (b={b}, t={t}, p={p}) : {diff}"
        # the slots differ but designate the same cards
        v = np.asarray(o1.valid)
        assert np.array_equal(np.asarray(s.zone)[np.asarray(o1.slot)[v]], np.asarray(s2.zone)[np.asarray(o2.slot)[v]])
        nchk += 1
print(f"2. ok: {nchk} observations invariant to hidden information and to slots")

# 3. positive control: changing a card of one's own hand changes the observation
changed = 0
for i in rng.choice(len(samples), 100, replace=False):
    b, t = samples[i]
    s = state_at(b, t)
    p = int(s.dec_player)
    hand = np.where((np.asarray(s.zone) == core.Z_HAND) & (np.asarray(s.ctrl) == p))[0]
    if not len(hand):
        continue
    cid = np.array(s.cid)
    cid[hand[0]] = (cid[hand[0]] + 1) % CT
    o1, o2 = observe(bank, s, p), observe(bank, s._replace(cid=jnp.asarray(cid)), p)
    changed += not np.array_equal(o1.cid, o2.cid)
print(f"3. ok: {changed} positive controls detected" if changed else "3. FAILED: nothing detected")

# 4. its deck: public subtraction (starting list - its cards visible elsewhere) and positive control
main_origin = lambda uid: (uid < D) | ((uid >= D + 6) & (uid < 2 * D + 6))  # uid: deck 0, Extra 0, deck 1, Extra 1
dk_changed = nsub = 0
for i in rng.choice(len(samples), 200, replace=False):
    b, t = samples[i]
    s = state_at(b, t)
    p = int(s.dec_player)
    z, ctrl, own, pos = (np.asarray(x) for x in (s.zone, s.ctrl, s.owner, s.pos))
    cid0, uid = np.array(s.cid), np.asarray(s.uid)
    fu = (pos & core.P_FACEUP) != 0
    fld = (z == core.Z_MZONE) | (z == core.Z_SZONE)
    sees = (((ctrl == p) & ((z == core.Z_HAND) | fld | (z == core.Z_EXTRA))) | (fld & fu)
            | np.isin(z, [core.Z_GRAVE, core.Z_REMOVED, core.Z_OVERLAY]))
    mo = main_origin(uid) & (own == p)
    start = collections.Counter(cid0[mo].tolist())
    start.subtract(collections.Counter(cid0[mo & (z != core.Z_DECK) & sees].tolist()))
    o1 = observe(bank, s, p)
    got = collections.Counter(int(c) for c in np.asarray(o1.deck) if c >= 0)
    assert +start == got, f"4. deck != public subtraction (b={b}, t={t}, p={p})"
    nsub += 1
    dk = np.where((z == core.Z_DECK) & (ctrl == p))[0]
    if len(dk):
        cid = cid0.copy()
        cid[dk[0]] = (cid[dk[0]] + 1) % CT
        o2 = observe(bank, s._replace(cid=jnp.asarray(cid)), p)
        dk_changed += not np.array_equal(o1.deck, o2.deck)
print(f"4. ok: deck = public subtraction on {nsub} states; {dk_changed} positive controls")
assert dk_changed, "4. FAILED"
