"""Official Twin Twisters: two targets, zero/one/two removed in response."""
from pathlib import Path
import sys
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare, core, driver
from oracle_bank import build

TWIN, MST, NORMAL = 43898403, 5318639, 44287299
NATIVE_DUEL = driver.Duel
ZONE_ID = {'szone': driver.K['LOCATION_SZONE'], 'mzone': driver.K['LOCATION_MZONE'],
           'grave': driver.K['LOCATION_GRAVE'], 'hand': driver.K['LOCATION_HAND']}


def watched_duel(state, reply_code):
    class WatchedDuel(NATIVE_DUEL):
        def _handle(self, msg):
            reader = driver.Reader(msg)
            kind = reader.take('B')
            if kind == driver.K['MSG_MOVE']:
                code = reader.take('I')
                ctrl, zone, seq, _ = reader.take('BBII')
                state['moved_cards'].add((code, ctrl, zone, seq))
            elif kind == driver.K['MSG_CHAIN_SOLVED'] and not state['reply_returned']:
                index = reader.take('B')
                chain = self.chains.get(index)
                if chain and chain[0] == reply_code:
                    keys = state['reply_keys']
                    state['reply_departed_at_return'] = bool(keys) and all(
                        key in state['moved_cards'] for key in keys)
                    state['reply_returned'] = True
            return super()._handle(msg)
    return WatchedDuel


db, bank, cmap = build([TWIN, MST, NORMAL])
step, legal = jax.jit(core.step), jax.jit(core.legal)
for hit in (None, 0, 1, 'both'):
    state = dict(fired=False, replied=False, pair=False, target_slots=(), target_mv={},
                 reply_slots=(), reply_mv={}, reply_keys=(),
                 moved_cards=set(), reply_returned=False, reply_departed_at_return=False)
    targets = set()
    reply = TWIN if hit == 'both' else MST
    def slots_for(cards, current):
        slots = []
        for card in cards:
            zone = ZONE_ID.get(card[2])
            matches = [i for i in range(current.cid.shape[0])
                       if int(bank.code[current.cid[i]]) == card[0]
                       and int(current.ctrl[i]) == card[1] and int(current.zone[i]) == zone
                       and int(current.seq[i]) == card[3]]
            assert len(matches) == 1, ('card option absent or ambiguous', card, matches)
            slots.append(matches[0])
        assert len(set(slots)) == len(cards), ('duplicate card options', cards, slots)
        return tuple(slots)

    def chooser(dec, indices):
        current = compare.LAST_STATE
        turn = int(current.turn)
        for link in range(int(current.n_chain)):
            if int(current.lk_p[link]) == 0 and int(bank.code[current.cid[current.lk_src[link]]]) == TWIN:
                targets.update(i for i in range(current.cid.shape[0]) if bool(current.lk_targets[link, i] > 0)
                               and int(bank.code[current.cid[i]]) == MST and int(current.ctrl[i]) == 0
                               and int(current.zone[i]) == core.Z_SZONE)
                state['pair'] |= len(targets) == 2
        def find(kind, code):
            return next((i for i in indices if dec.options[i][0] == kind and dec.options[i][1] == code), None)
        choice = None
        if turn == 1 and dec.player == 0:
            choice = find('sset', MST)
        if turn == 2 and dec.player == 1:
            choice = find('sset', reply)
        if turn >= 3 and dec.player == 0 and not state['fired']:
            choice = find('activate', TWIN)
            if choice is not None:
                state['fired'] = True
        if state['fired'] and hit is not None and dec.player == 1 and not state['replied']:
            choice = find('chain', reply)
            if choice is not None:
                state['replied'] = True
        if choice is None and dec.options[0][0] == 'cards':
            if int(current.pg) == core.PG_COST:
                choice = next((i for i in indices if any(c[0] == NORMAL for c in dec.options[i][1])), None)
            else:
                candidates = [i for i in indices if all(c[0] == MST and c[1] == 0 for c in dec.options[i][1])]
                if candidates:
                    size = max(len(dec.options[i][1]) for i in candidates)
                    candidates = [i for i in candidates if len(dec.options[i][1]) == size]
                    offset = hit if dec.player == 1 and isinstance(hit, int) else 0
                    choice = candidates[min(offset, len(candidates)-1)]
                    chosen_cards = dec.options[choice][1]
                    if (dec.player == 0 and state['fired'] and not state['target_slots']
                            and int(current.pg) != core.PG_COST):
                        state['target_slots'] = slots_for(chosen_cards, current)
                        state['target_mv'] = {i: int(current.mv[i]) for i in state['target_slots']}
                    elif (dec.player == 1 and state['replied'] and not state['reply_slots']
                          and int(current.pg) != core.PG_COST):
                        state['reply_slots'] = slots_for(chosen_cards, current)
                        state['reply_mv'] = {i: int(current.mv[i]) for i in state['reply_slots']}
                        state['reply_keys'] = tuple((c[0], c[1], ZONE_ID[c[2]], c[3])
                                                    for c in chosen_cards)
        if choice is None:
            choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
        return choice
    driver.Duel = watched_duel(state, reply)
    try:
        result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                              given=([[TWIN, MST, MST, NORMAL] * 10, [reply, NORMAL] * 20], [[], []]))
    finally:
        driver.Duel = NATIVE_DUEL
    assert result['ok'], (hit, result.get('err'))
    assert state['fired'] and state['pair'], state
    assert state['replied'] == (hit is not None), state
    assert len(state['target_slots']) == 2, state
    final = compare.LAST_STATE
    assert all(int(final.mv[i]) > state['target_mv'][i] for i in state['target_slots']), state
    assert any(e['code'] == TWIN and e['stage'] == 'resolved' and e['count'] for e in result['coverage'])
    if hit is not None:
        assert len(state['reply_slots']) == (2 if hit == 'both' else 1), state
        assert all(int(final.mv[i]) > state['reply_mv'][i] for i in state['reply_slots']), state
        assert state['reply_returned'] and state['reply_departed_at_return'], state
        assert any(e['code'] == reply and e['stage'] == 'resolved' and e['count']
                   for e in result['coverage']), result['coverage']
    print('Twin Twisters response', hit, ':', result['decisions'], 'identical decisions', flush=True)
