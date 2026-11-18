"""Observation of a player: what they are allowed to see, as fixed-size tokens.

- One token per visible card: their hand, the whole field (the opponent's face-down cards without their identity),
  the GYs, the banished cards and the face-up Pendulum Monsters of the Extra Decks. The decks and the opponent's hand
  are only counters (global token).
- The identity (cid, ATK/DEF/Level) is given only if the player can know it.
- The order of the tokens depends only on visible information (location, controller, seq), never on the slot.
- `slot` is used only to convert the network's choice (verb x token) into an engine action. It is not a network
  input: the slot is drawn at random in each game and must teach nothing.
- `deck`: identities of one's own remaining deck, sorted, -1 beyond. The player knows the list, not the order: it is
  the starting list minus the cards visible elsewhere (reproducible by an external client; checked by obs_check).
- Only public information and events (plus one's own cards) enter: an external client (third-party server) must be
  able to rebuild the observation identically. No tracking of the identity of a hidden opponent card.
- At most T tokens. The cards targeted by a legal action come first, then hand and field, then GYs and banished
  cards (most recent first). The surplus cards are summarized only by the counters.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import core
from core import I32, NV

T = 64
DK = 40   # cards of its remaining deck given to the network (decks of 40; beyond that, the smallest cids)
ST_PUBLIC = (core.ST_SUMMON_TURN | core.ST_ATTACKED | core.ST_FORM_CHANGED | core.ST_SET_TURN | core.ST_LEAVE | core.ST_SPSUM
             | core.ST_LINKSUM)
ZONE_RANK = {core.Z_HAND: 0, core.Z_MZONE: 1, core.Z_SZONE: 2, core.Z_GRAVE: 3, core.Z_REMOVED: 4, core.Z_DECK: 5,
             core.Z_EXTRA: 6, core.Z_OVERLAY: 7}


class Obs(NamedTuple):
    reason: jnp.ndarray # [T] visible reason of the last move
    suminfo: jnp.ndarray # [T] type, origin and player of the last visible Summon
    valid: jnp.ndarray     # [T] bool
    known: jnp.ndarray     # [T] bool: identity visible
    cid: jnp.ndarray       # [T] index in the bank, -1 if unknown
    atk: jnp.ndarray       # [T] 0 if unknown
    df: jnp.ndarray
    level: jnp.ndarray
    zone: jnp.ndarray      # [T]
    seq: jnp.ndarray
    mine: jnp.ndarray      # [T] bool: controlled by the observer
    pos: jnp.ndarray
    st: jnp.ndarray        # [T] public statuses (+ ST_SEL for its own choices in progress)
    pending: jnp.ndarray   # [T] bool: card in progress (monster to Summon, attacker, effect being resolved)
    chain: jnp.ndarray     # [T] rank of the chain link it is the source of (1 = first), 0 otherwise
    negated: jnp.ndarray   # [T] bool: this chain link is negated
    card_activation: jnp.ndarray # [T] bool: the link activates a Spell/Trap card
    source_related: jnp.ndarray # [T] bool: the link's source is still related to it
    targeted: jnp.ndarray # [T, CH] int8: 0 not linked, 1 target, 2 first announced target
    act: jnp.ndarray       # [T, NV] bool: legal verb x token actions
    special: jnp.ndarray   # [N_SPECIAL] bool: legal special actions
    glob: jnp.ndarray      # [G] int: LP, turn, phase, decision, counters
    deck: jnp.ndarray      # [DK] cid of the cards of its remaining deck, sorted, -1 beyond
    slot: jnp.ndarray      # [T] NOT a network input: used to rebuild the action


def _count(s, zone, p):
    return ((s.zone == zone) & (s.ctrl == p)).sum().astype(I32)


def observe(bank, s, p=None, m=None):
    """Observation of player p (by default the one who must decide). m: legal mask already computed (step_batch)."""
    p = s.dec_player if p is None else jnp.asarray(p, I32)
    n = s.cid.shape[0]
    mine = s.ctrl == p
    faceup = (s.pos & (core.P_FU_ATK | core.P_FU_DEF)) != 0
    field = (s.zone == core.Z_MZONE) | (s.zone == core.Z_SZONE)
    ovl = s.zone == core.Z_OVERLAY  # Xyz materials: public; seq = zone of the Xyz x 8 + rank of the material
    # face-up Pendulum Monsters of the Extra Deck are public
    public = (s.zone == core.Z_GRAVE) | (s.zone == core.Z_REMOVED) | ovl | ((s.zone == core.Z_EXTRA) & faceup)
    vseq = jnp.where(ovl, s.seq[jnp.clip(s.ov_par, 0, n - 1)] * 8 + s.seq, s.seq)
    m = core.legal(bank, s) if m is None else m
    act_slot = m[:NV * n].reshape(NV, n).T            # [N, NV]
    acted = act_slot.any(1)
    # p observes its actions only if it is the one deciding
    acted = acted & (s.dec_player == p)
    act_slot = act_slot & (s.dec_player == p)

    # a card that p can choose is shown to p (search in its deck), but not its position in the deck
    deck = s.zone == core.Z_DECK
    own_ex = mine & (s.zone == core.Z_EXTRA)  # its own Extra Deck is known (face-down for the opponent)
    shown = (mine & (s.zone == core.Z_HAND)) | field | public | acted | own_ex
    known = shown & ((mine & ((s.zone == core.Z_HAND) | field | deck | own_ex)) | (field & faceup) | public)

    zrank = jnp.zeros(n, I32)
    for z, r in ZONE_RANK.items():
        zrank = jnp.where(s.zone == z, r, zrank)
    cls = jnp.where(acted, 0, jnp.where(public, 2, 1))
    seqkey = jnp.where(public, 999 - vseq, jnp.where(deck, s.cid, s.seq))
    score = cls * 1_000_000 + (~mine) * 100_000 + zrank * 1_000 + seqkey
    score = jnp.where(shown, score, 10_000_000)
    _, idx = jax.lax.top_k(-score, T)                 # order fixed by visible information only
    valid = shown[idx]
    kn = known[idx] & valid

    c = s.cid[idx]
    own_sel = jnp.where(mine[idx], s.st[idx] & core.ST_SEL, 0)
    pend = (((s.pend >= 0) & (idx == s.pend)) | ((s.ch_src >= 0) & (idx == s.ch_src))
            | ((s.atk >= 0) & (idx == s.atk)) | ((s.atk_tgt >= 0) & (idx == s.atk_tgt))) & valid
    live = jnp.arange(core.CH) < s.n_chain
    link = jnp.zeros(n, I32).at[jnp.where(live, s.lk_src, n)].set(jnp.arange(1, core.CH + 1, dtype=I32), mode="drop")
    neg = jnp.zeros(n, bool).at[jnp.where(live, s.lk_src, n)].set(s.lk_neg, mode="drop")
    card_activation = jnp.zeros(n, bool).at[jnp.where(live, s.lk_src, n)].set(
        (s.lk_flags & core.CF_ACT) != 0, mode="drop")
    src = jnp.clip(s.lk_src, 0, n - 1)
    related = ((s.lk_flags & core.CF_SRC_UNRELATED) == 0) & (s.mv[src] <= s.lk_t)
    source_related = jnp.zeros(n, bool).at[jnp.where(live, s.lk_src, n)].set(related, mode="drop")
    # Do not track a card after it leaves, even if it becomes visible again:
    # its identity may have been hidden in the meantime. The native announcements follow
    # card_sort: only the first announced element is exposed, never the raw UID.
    # A single announced group per chain link: all its targets share lk_tgt_t.
    # Keeping the original membership avoids designating another first target.
    original = s.lk_targets
    first = s.uid[None, :] == jnp.min(jnp.where(original, s.uid[None, :], core.BIG), axis=1)[:, None]
    targeted = (s.lk_targets & (s.mv[None, :] <= s.lk_tgt_t[:, None]) & live[:, None]).astype(jnp.int8) * (1 + first.astype(jnp.int8))
    glob = jnp.stack([
        s.lp[p], s.lp[1 - p], s.turn, (s.tp == p).astype(I32), s.phase, s.ns_used.astype(I32), s.ps_used.astype(I32),
        jnp.where(s.dec_player == p, s.dec, 0), jnp.where(s.dec_player == p, s.need, 0),
        jnp.where(s.dec_player == p, s.zctx, 0), s.n_chain,
        *[_count(s, z, q) for q in (p, 1 - p)
          for z in (core.Z_HAND, core.Z_DECK, core.Z_MZONE, core.Z_SZONE, core.Z_GRAVE, core.Z_REMOVED)],
    ]).astype(I32)
    zero = lambda x: jnp.where(kn, x, 0)
    tracked_reason = ((s.zone[idx] & (core.Z_GRAVE | core.Z_REMOVED | core.Z_EXTRA)) != 0)
    return Obs(reason=jnp.where(tracked_reason, zero(s.reason[idx]), 0), suminfo=zero(s.suminfo[idx]), valid=valid, known=kn, cid=jnp.where(kn, c, -1), atk=zero(s.atk_c[idx]), df=zero(s.df_c[idx]),
               level=zero(bank.level[c]), zone=jnp.where(valid, s.zone[idx], 0), seq=jnp.where(valid & ~deck[idx], vseq[idx], 0),
               mine=mine[idx] & valid, pos=jnp.where(valid, s.pos[idx], 0),
               st=jnp.where(valid, (s.st[idx] & ST_PUBLIC) | own_sel, 0), pending=pend,
               chain=jnp.where(valid, link[idx], 0), negated=neg[idx] & valid,
               card_activation=card_activation[idx] & valid, source_related=source_related[idx] & valid,
               targeted=jnp.where(valid[:, None], targeted[:, idx].T, 0),
               act=act_slot[idx] & valid[:, None], special=m[NV * n:] & (s.dec_player == p), glob=glob,
               deck=own_deck(s, mine & deck), slot=jnp.where(valid, idx, -1).astype(I32))


def own_deck(s, m):
    """[DK] cid of the cards m, sorted (order of the hidden deck), -1 beyond."""
    big = jnp.iinfo(I32).max
    v = -jax.lax.top_k(-jnp.where(m, s.cid, big), min(DK, m.shape[0]))[0]
    v = jnp.where(v == big, -1, v).astype(I32)
    return jnp.pad(v, (0, DK - v.shape[0]), constant_values=-1)


def to_action(obs, n, verb, tok):
    """Network choice (verb, token) -> engine action."""
    return (verb * n + obs.slot[tok]).astype(I32)


def special_action(n, k):
    return jnp.asarray(NV * n + k, I32)
