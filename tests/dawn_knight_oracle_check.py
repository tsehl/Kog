"""Dawn Knight: Foolish Burial sends the card from the Deck to the GY, then its effect puts it back on top."""
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
from oracle_bank import build

DAWN, FOOLISH, LIGHT_TARGET, NORMAL = 6351548, 81439173, 89631139, 44287299


class WatchedDuel(driver.Duel):
    """Keep native destination moves so the expected Deck return is explicit."""
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


db, bank, cmap = build([DAWN, FOOLISH, LIGHT_TARGET, NORMAL])
state = dict(activations=0, selected_for_foolish=[], dawn_target=False,
             dawn_slot=None, dawn_mv=None, dawn_returned=False, dawn_top=False,
             dawn_yes=0,
             native_moves=[])

# The opening five include two Foolish Burial copies. Dawn Knight and one other
# LIGHT monster remain in the Deck for the two real Foolish Burial resolutions.
deck0 = [DAWN, LIGHT_TARGET] + [NORMAL] * 33 + [FOOLISH, FOOLISH, NORMAL, NORMAL, NORMAL]
deck1 = [NORMAL] * 40
assert len(deck0) == len(deck1) == 40
NATIVE_DUEL = driver.Duel


def chooser(dec, indices):
    current = compare.LAST_STATE
    turn = int(current.turn)
    if state['dawn_slot'] is not None and not state['dawn_returned']:
        i = state['dawn_slot']
        state['dawn_returned'] = (int(current.zone[i]) == core.Z_DECK
                                  and int(current.mv[i]) > state['dawn_mv'])
        if state['dawn_returned']:
            own_deck = [int(current.seq[j]) for j in range(current.cid.shape[0])
                        if int(current.zone[j]) == core.Z_DECK and int(current.ctrl[j]) == 0]
            state['dawn_top'] = int(current.seq[i]) == max(own_deck)

    def find(kind, code):
        return next((i for i in indices if dec.options[i][0] == kind
                     and dec.options[i][1] == code), None)

    choice = None
    if (dec.options and dec.options[0][0] == 'yes' and int(current.ch_src) >= 0
            and int(bank.code[current.cid[current.ch_src]]) == DAWN):
        choice = next(i for i in indices if dec.options[i][0] == 'yes')
        state['dawn_yes'] += 1
    if dec.player == 0 and turn >= 1 and state['activations'] < 2:
        activation = find('activate', FOOLISH)
        if activation is not None:
            choice = activation
            state['activations'] += 1

    if dec.options and dec.options[0][0] == 'cards':
        if int(current.pg) == core.PG_RESOLVE:
            source = int(current.ch_src)
            assert source >= 0 and int(bank.code[current.cid[source]]) == FOOLISH, (
                'unexpected resolution selection source', source, dec.options)
            expected = LIGHT_TARGET if len(state['selected_for_foolish']) == 0 else DAWN
            candidates = [i for i in indices if len(dec.options[i][1]) == 1
                          and dec.options[i][1][0][0] == expected]
            assert candidates, ('Foolish Burial must offer the intended monster', expected,
                                dec.options)
            state['selected_for_foolish'].append(expected)
            choice = candidates[0]
        elif (int(current.pg) == core.PG_TARGET and int(current.ch_src) >= 0
              and int(bank.code[current.cid[current.ch_src]]) == DAWN):
            groups = [dec.options[i][1] for i in indices]
            target_codes = {group[0][0] for group in groups if len(group) == 1}
            assert target_codes == {DAWN, LIGHT_TARGET}, ('Dawn target set', groups)
            candidates = [i for i in indices if len(dec.options[i][1]) == 1
                          and dec.options[i][1][0][0] == DAWN]
            assert len(candidates) == 1, ('Dawn Knight must be a selectable target', groups)
            card = dec.options[candidates[0]][1][0]
            assert card[1] == 0 and card[2] == 'grave', card
            matches = [i for i in range(current.cid.shape[0])
                       if int(bank.code[current.cid[i]]) == DAWN
                       and int(current.ctrl[i]) == 0 and int(current.zone[i]) == core.Z_GRAVE
                       and int(current.seq[i]) == card[3]]
            assert len(matches) == 1, ('ambiguous Dawn Knight target', card, matches)
            state['dawn_slot'] = matches[0]
            state['dawn_mv'] = int(current.mv[matches[0]])
            state['dawn_target'] = True
            choice = candidates[0]

    if choice is None:
        choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
    return choice


driver.Duel = lambda *args, **kwargs: WatchedDuel(state, *args, **kwargs)
try:
    result = compare.play(db, bank, cmap, 4821, jax.jit(core.step), jax.jit(core.legal),
                          mode='tr', chooser=chooser, given=([deck0, deck1], [[], []]))
finally:
    driver.Duel = NATIVE_DUEL

assert result['ok'], result.get('err')
assert state['activations'] == 2, state
assert state['selected_for_foolish'] == [LIGHT_TARGET, DAWN], state
assert state['dawn_target'] and state['dawn_returned'] and state['dawn_top'], state
assert any(code == DAWN and pzone == driver.K['LOCATION_GRAVE']
           and zone == driver.K['LOCATION_DECK'] and ctrl == 0
           for code, _pctrl, pzone, _pseq, ctrl, zone, _seq in state['native_moves']), (
               'native move must return Dawn from p0 Graveyard to p0 Deck', state['native_moves'])
for code in (FOOLISH, DAWN):
    for stage in ('activated', 'resolved'):
        count = sum(e['count'] for e in result['coverage']
                    if e['code'] == code and e['stage'] == stage)
        assert count >= (2 if code == FOOLISH and stage == 'resolved' else 1), (
            code, stage, count, result['coverage'])
print('Dawn Knight 6351548: return from the Deck to the top of the Deck confirmed;',
      result['decisions'], 'identical decisions', flush=True)
