"""Dawn Knight: from the field, only the first effect sends Blue-Eyes to the GY."""
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
from oracle_bank import build

DAWN, HOLE, BLUE, NORMAL = 6351548, 53129443, 89631139, 44287299
DESC0, DESC1 = DAWN << 20, (DAWN << 20) | 1
db, bank, cmap = build([DAWN, HOLE, BLUE, NORMAL])
state = dict(native_moves=[], snapshots=[], summoned=False, hole=False, blue_sent=False)


class WatchedDuel(driver.Duel):
    def _handle(self, msg):
        reader = driver.Reader(msg)
        kind = reader.take('B')
        if kind == driver.K['MSG_MOVE']:
            code = reader.take('I')
            pctrl, pzone, pseq, _ = reader.take('BBII')
            ctrl, zone, seq, _ = reader.take('BBII')
            state['native_moves'].append((code, pctrl, pzone, pseq, ctrl, zone, seq))
        return super()._handle(msg)


NATIVE_DUEL = driver.Duel


def chooser(dec, indices):
    current = compare.LAST_STATE
    assert current is not None
    dawn_slots = [i for i in range(current.cid.shape[0])
                  if int(bank.code[current.cid[i]]) == DAWN and int(current.ctrl[i]) == 0]
    blue_slots = [i for i in range(current.cid.shape[0])
                  if int(bank.code[current.cid[i]]) == BLUE and int(current.ctrl[i]) == 0]
    snap = {
        'dec': int(current.dec), 'wk': int(current.wk), 'tst': int(current.tst),
        'n_chain': int(current.n_chain),
        'dawn': [(int(current.zone[i]), int(current.mv[i])) for i in dawn_slots],
        'blue': [(int(current.zone[i]), int(current.mv[i])) for i in blue_slots],
        'ch_src': int(current.ch_src), 'ch_eff': int(current.ch_eff),
    }
    state['snapshots'].append(snap)
    # A unique target may be selected automatically, without a chooser call.
    if any(zone == core.Z_GRAVE for zone, _ in snap['blue']):
        state['blue_sent'] = True

    def find(kind, code=None, desc=None):
        return next((i for i in indices
                     if dec.options[i][0] == kind
                     and (code is None or dec.options[i][1] == code)
                     and (desc is None or len(dec.options[i]) > 5 and dec.options[i][5] == desc)), None)

    choice = None
    if dec.options and dec.options[0][0] == 'cards' and int(current.pg) == core.PG_RESOLVE:
        source = int(current.ch_src)
        assert source >= 0 and int(bank.code[current.cid[source]]) == DAWN, (
            'unexpected resolution selection', dec.options, state['snapshots'][-1])
        candidates = [i for i in indices if len(dec.options[i][1]) == 1
                      and dec.options[i][1][0][0] == BLUE]
        assert candidates, ('Blue-Eyes must remain selectable in Deck', dec.options)
        choice = candidates[0]

    # YGOPro's sole-optional-trigger prompt uses generic description 221;
    # activation coverage still records aux.Stringid(id, 0) as DESC0.
    if choice is None:
        choice = find('chain', DAWN, 221)

    # Summon Dawn first, then activate Dark Hole while Dawn is still on the field.
    if choice is None and dec.player == 0:
        if not state['summoned']:
            choice = find('summon', DAWN)
            if choice is not None:
                state['summoned'] = True
        if choice is None and state['summoned'] and not state['hole']:
            choice = find('activate', HOLE)
            if choice is not None:
                state['hole'] = True

    if choice is None:
        choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
    return choice


def has_move(code, source_zone, destination_zone):
    return any(m[0] == code and m[2] == source_zone and m[5] == destination_zone
               for m in state['native_moves'])


# Last additions are the top of the pseudo-shuffled Deck: opening hand contains
# Dawn Knight and Dark Hole; Blue-Eyes remains in the Deck until Dawn resolves.
# Pseudo-shuffle: the final five cards are the opening hand, in reverse draw
# order. Keep Blue-Eyes at the bottom of the Deck for Dawn's resolution.
deck0 = [BLUE] + [NORMAL] * 34 + [NORMAL] * 3 + [HOLE, DAWN]
deck1 = [NORMAL] * 40
assert len(deck0) == len(deck1) == 40
driver.Duel = lambda *args, **kwargs: WatchedDuel(*args, **kwargs)
try:
    result = compare.play(db, bank, cmap, 6351548, jax.jit(core.step), jax.jit(core.legal),
                          mode='tr', chooser=chooser, given=([deck0, deck1], [[], []]))
finally:
    driver.Duel = NATIVE_DUEL

assert result['ok'], result.get('err')
assert state['summoned'] and state['hole'], state
coverage = result['coverage']
assert any(e['code'] == DAWN and e['stage'] == 'activated' and e['description'] == DESC0
           for e in coverage), coverage
assert any(e['code'] == DAWN and e['stage'] == 'resolved' and e['description'] == DESC0
           for e in coverage), coverage
assert any(e['code'] == HOLE and e['stage'] == 'activated' for e in coverage), coverage
assert any(e['code'] == HOLE and e['stage'] == 'resolved' for e in coverage), coverage
assert not any(e['code'] == DAWN and e['description'] == DESC1
               and e['stage'] in ('activated', 'resolved') for e in coverage), coverage
assert has_move(DAWN, driver.K['LOCATION_MZONE'], driver.K['LOCATION_GRAVE']), state['native_moves']
assert has_move(BLUE, driver.K['LOCATION_DECK'], driver.K['LOCATION_GRAVE']), state['native_moves']
assert not has_move(DAWN, driver.K['LOCATION_GRAVE'], driver.K['LOCATION_DECK']), state['native_moves']
assert any(s['dawn'] and s['dawn'][0][0] == core.Z_GRAVE
           and s['blue'] and s['blue'][0][0] == core.Z_DECK for s in state['snapshots']), state['snapshots']
assert any(s['dawn'] and s['dawn'][0][0] == core.Z_GRAVE
           and s['blue'] and s['blue'][0][0] == core.Z_GRAVE for s in state['snapshots']), state['snapshots']
assert state['blue_sent'], state['snapshots']
print('Dawn field → GY; desc0 activated/resolved; Blue-Eyes Deck → GY; desc1 absent;',
      result['decisions'], 'identical decisions')
