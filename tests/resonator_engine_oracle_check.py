"""Resonator Engine: targets two Resonators in the GY, adds Masaki to the hand and returns the targets to the
top of the Deck."""
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
from oracle_bank import build

ENGINE, FOOLISH, RESONATORS, MASAKI = 15576074, 81439173, (5780210, 13764881), 44287299


class WatchedDuel(driver.Duel):
    """Record native moves with their source and destination locations."""
    def __init__(self, state, *args, **kwargs):
        self.probe_state = state
        super().__init__(*args, **kwargs)

    def _handle(self, msg):
        reader = driver.Reader(msg)
        kind = reader.take('B')
        if kind == driver.K['MSG_MOVE']:
            code = reader.take('I')
            pctrl, pzone, pseq, _ = reader.take('BBII')
            ctrl, zone, seq, _ = reader.take('BBII')
            self.probe_state['native_moves'].append(
                (code, pctrl, pzone, pseq, ctrl, zone, seq))
        return super()._handle(msg)


db, bank, cmap = build([ENGINE, FOOLISH, *RESONATORS, MASAKI])
state = dict(foolish_activations=0, foolish_selections=[], engine_activated=False,
             engine_targets=(), target_snapshot={}, masaki_slot=None, masaki_mv=None,
             masaki_searched=False, targets_returned=False, targets_at_top=False,
             native_moves=[])

# Opening hand: two Foolish Burials, Resonator Engine, and two normal monsters.
# The two Resonators and the remaining level-four Masakis stay in the Deck.
deck0 = [*RESONATORS] + [MASAKI] * 33 + [FOOLISH, FOOLISH, ENGINE, MASAKI, MASAKI]
deck1 = [MASAKI] * 40
assert len(deck0) == len(deck1) == 40
NATIVE_DUEL = driver.Duel


def record_engine_targets(current):
    """Record the two targets from the active native-equivalent chain link."""
    for link in range(int(current.n_chain)):
        src = int(current.lk_src[link])
        if src < 0 or int(bank.code[current.cid[src]]) != ENGINE:
            continue
        related = current.lk_targets[link] & (current.mv <= current.lk_tgt_t[link])
        slots = tuple(i for i in range(current.cid.shape[0]) if bool(related[i]))
        if not slots:
            continue
        assert len(slots) == 2, ('Resonator Engine must target two cards', slots)
        assert {int(bank.code[current.cid[i]]) for i in slots} == set(RESONATORS), slots
        assert all(int(current.ctrl[i]) == 0 and int(current.zone[i]) == core.Z_GRAVE
                   for i in slots), slots
        state['engine_targets'] = slots
        state['target_snapshot'] = {i: int(current.mv[i]) for i in slots}


def chooser(dec, indices):
    current = compare.LAST_STATE
    turn = int(current.turn)
    record_engine_targets(current)

    if state['engine_targets'] and not state['targets_returned']:
        targets = state['engine_targets']
        state['targets_returned'] = all(
            int(current.zone[i]) == core.Z_DECK
            and int(current.mv[i]) > state['target_snapshot'][i] for i in targets)
        if state['targets_returned']:
            deck_seqs = [int(current.seq[i]) for i in range(current.cid.shape[0])
                         if int(current.zone[i]) == core.Z_DECK and int(current.ctrl[i]) == 0]
            top_two = set(sorted(deck_seqs)[-2:])
            state['targets_at_top'] = {int(current.seq[i]) for i in targets} == top_two
    if state['masaki_slot'] is not None and not state['masaki_searched']:
        i = state['masaki_slot']
        state['masaki_searched'] = (int(current.zone[i]) == core.Z_HAND
                                    and int(current.mv[i]) > state['masaki_mv'])

    def find(kind, code):
        return next((i for i in indices if dec.options[i][0] == kind
                     and dec.options[i][1] == code), None)

    choice = None
    if dec.options and dec.options[0][0] == 'cards':
        src = int(current.ch_src)
        source_code = int(bank.code[current.cid[src]]) if src >= 0 else None
        if int(current.pg) == core.PG_RESOLVE and source_code == FOOLISH:
            assert state['foolish_activations'] > len(state['foolish_selections']), state
            expected = RESONATORS[len(state['foolish_selections'])]
            candidates = [i for i in indices if len(dec.options[i][1]) == 1
                          and dec.options[i][1][0][0] == expected]
            assert candidates, ('Foolish must offer the intended Resonator', expected, dec.options)
            state['foolish_selections'].append(expected)
            choice = candidates[0]
        elif int(current.pg) == core.PG_TARGET and source_code == ENGINE:
            groups = [dec.options[i][1] for i in indices]
            assert len(groups) == 1 and len(groups[0]) == 2, ('Engine exact-two target set', groups)
            assert {card[0] for card in groups[0]} == set(RESONATORS), groups
            assert all(card[1] == 0 and card[2] == 'grave' for card in groups[0]), groups
            choice = indices[0]
        elif int(current.pg) == core.PG_RESOLVE and source_code == ENGINE:
            candidates = [i for i in indices if len(dec.options[i][1]) == 1
                          and dec.options[i][1][0][0] == MASAKI]
            assert candidates, ('Resonator Engine must offer Masaki from the Deck', dec.options)
            option = dec.options[candidates[0]][1][0]
            assert option[1] == 0 and option[2] == 'deck', option
            matches = [i for i in range(current.cid.shape[0])
                       if int(bank.code[current.cid[i]]) == MASAKI
                       and int(current.ctrl[i]) == 0 and int(current.zone[i]) == core.Z_DECK
                       and int(current.seq[i]) == option[3]]
            assert len(matches) == 1, ('ambiguous Masaki search result', option, matches)
            state['masaki_slot'] = matches[0]
            state['masaki_mv'] = int(current.mv[matches[0]])
            choice = candidates[0]

    if choice is None and dec.player == 0:
        if state['foolish_activations'] < 2:
            choice = find('activate', FOOLISH)
            if choice is not None:
                state['foolish_activations'] += 1
        elif len(state['foolish_selections']) == 2 and not state['engine_activated']:
            choice = find('activate', ENGINE)
            if choice is not None:
                state['engine_activated'] = True

    if choice is None:
        choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
    return choice


driver.Duel = lambda *args, **kwargs: WatchedDuel(state, *args, **kwargs)
try:
    result = compare.play(db, bank, cmap, 5912, jax.jit(core.step), jax.jit(core.legal),
                          mode='tr', chooser=chooser, given=([deck0, deck1], [[], []]))
finally:
    driver.Duel = NATIVE_DUEL

assert result['ok'], result.get('err')
assert state['foolish_activations'] == 2 and state['foolish_selections'] == list(RESONATORS), state
assert state['engine_activated'] and len(state['engine_targets']) == 2, state
assert state['targets_returned'] and state['targets_at_top'], state
assert state['masaki_slot'] is not None and state['masaki_searched'], state

deck_loc, grave_loc, hand_loc = (driver.K['LOCATION_DECK'], driver.K['LOCATION_GRAVE'],
                                 driver.K['LOCATION_HAND'])
for code in RESONATORS:
    assert any(move[0] == code and move[1] == 0 and move[2] == grave_loc
               and move[4] == 0 and move[5] == deck_loc
               for move in state['native_moves']), ('native Graveyard-to-Deck move missing', code,
                                                     state['native_moves'])
assert any(move[0] == MASAKI and move[2] == deck_loc and move[4] == 0 and move[5] == hand_loc
           for move in state['native_moves']), ('native Masaki search to hand missing', state['native_moves'])

for code, minimum in ((FOOLISH, 2), (ENGINE, 1)):
    for stage in ('activated', 'resolved'):
        count = sum(e['count'] for e in result['coverage']
                    if e['code'] == code and e['stage'] == stage)
        assert count >= minimum, (code, stage, count, result['coverage'])
print('Resonator Engine 15576074: two targets on top, Masaki in hand;',
      result['decisions'], 'identical decisions', flush=True)
