"""GetTargetCards: filter the relations before handling a multiple group."""
from pathlib import Path
import ast
import itertools
import shutil
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
import translate
from oracle_bank import build

SPELL, NORMAL, MST = 600930051, 44287299, 5318639
TWIN = 43898403
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


driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Multiple targets probe')
module = ast.parse((ROOT / 'tests/relation_multiple_targets_oracle_reference.py').read_text())
script = next(ast.literal_eval(node.value) for node in module.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == 'script')
# The reference script already requires exactly two selected targets.
assert 'Duel.SelectTarget(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,2,2,e:GetHandler())' in script
step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    shutil.copy(Path(translate.OFFICIAL, f'c{MST}.lua'), folder)
    shutil.copy(Path(translate.OFFICIAL, f'c{TWIN}.lua'), folder)
    old = translate.OFFICIAL
    translate.OFFICIAL = folder
    driver.SCRIPTS.insert(0, folder)
    try:
        getters = ('Duel.GetTargetCards(e)',
                   'Duel.GetChainInfo(0,CHAININFO_TARGET_CARDS):Filter(Card.IsRelateToEffect,nil,e)')
        guards = ('positive', 'exact_two', 'not_exactly_two')
        cases = [(*case, False) for case in itertools.product(getters, guards, (False, True))]
        cases.append(('Duel.GetTargetCards(e)', 'requery_after_move', False, True))
        for getter, guard, shifted, requery_case in cases:
            action = 'Duel.SendtoHand(tg,nil,REASON_EFFECT) Duel.Damage(tp,#tg*1234,REASON_EFFECT)'
            if requery_case:
                guarded_action = ('Duel.SendtoHand(tg,nil,REASON_EFFECT)\n'
                                  '  local requery=Duel.GetTargetCards(e):Filter(Card.IsType,nil,TYPE_SPELL)\n'
                                  '  Duel.Damage(tp,#requery*1234,REASON_EFFECT)')
            elif guard == 'positive':
                guarded_action = f'if #tg>0 then {action} end'
            elif guard == 'exact_two':
                guarded_action = f'if #tg==2 then {action} end'
            else:
                guarded_action = f'if #tg~=2 then return end\n  {action}'
            old_operation = ('local tc=Duel.GetFirstTarget()\n'
                             '  if tc and tc:IsRelateToEffect(e) then Duel.Recover(tp,1234,REASON_EFFECT) end')
            source = script.replace(old_operation,
                                    f'local tg={getter}:Filter(Card.IsType,nil,TYPE_SPELL)\n  {guarded_action}')
            assert source != script, (getter, guard, 'operation replacement did not apply')
            if shifted:
                select = 'Duel.SelectTarget(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,2,2,e:GetHandler())'
                source = source.replace(select, f'local chosen={select}\n  Duel.SendtoGrave(chosen,REASON_COST)')
            Path(folder, f'c{SPELL}.lua').write_text(source)
            db, bank, cmap = build([SPELL, NORMAL, MST, TWIN])
            for hit in ((None,) if shifted or requery_case else (None, 0, 1, 'both')):
                reply = TWIN if hit == 'both' else MST
                state = dict(fired=False, replied=False, selected=False, extra_set=False,
                             target_slots=(), target_mv={}, reply_slots=(), reply_mv={},
                             reply_keys=(), moved_cards=set(), reply_returned=False,
                             reply_departed_at_return=False)
                observed_lp = set()
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
                    turn = int(compare.LAST_STATE.turn)
                    observed_lp.add(int(compare.LAST_STATE.lp[0]))
                    def find(kind, code):
                        return next((i for i in indices if dec.options[i][0] == kind
                                     and dec.options[i][1] == code), None)
                    choice = None
                    if turn == 1 and dec.player == 0:
                        choice = find('sset', MST)
                    if turn == 2 and dec.player == 1:
                        choice = find('sset', reply)
                    if turn >= 3 and dec.player == 0 and not state['extra_set']:
                        choice = find('sset', MST)
                        if choice is not None:
                            state['extra_set'] = True
                    if (choice is None and turn >= 3 and dec.player == 0
                            and not state['fired']):
                        choice = find('activate', SPELL)
                        if choice is not None:
                            state['fired'] = True
                    if state['fired'] and hit is not None and not state['replied'] and dec.player == 1:
                        choice = find('chain', reply)
                        if choice is not None:
                            state['replied'] = True
                    if choice is None and dec.options[0][0] == 'cards':
                        candidates = [i for i in indices if all(c[0] == MST and c[1] == 0
                                                              for c in dec.options[i][1])]
                        if int(compare.LAST_STATE.pg) == core.PG_COST:
                            choice = next((i for i in indices if any(c[0] == NORMAL for c in dec.options[i][1])), None)
                        if candidates:
                            size = max(len(dec.options[i][1]) for i in candidates)
                            candidates = [i for i in candidates if len(dec.options[i][1]) == size]
                            offset = hit if dec.player == 1 and isinstance(hit, int) else 0
                            choice = candidates[min(offset, len(candidates)-1)]
                            if dec.player == 0 and state['fired']:
                                state['selected'] = True
                                if not state['target_slots']:
                                    state['target_slots'] = slots_for(dec.options[choice][1], compare.LAST_STATE)
                                    state['target_mv'] = {i: int(compare.LAST_STATE.mv[i])
                                                          for i in state['target_slots']}
                            elif dec.player == 1 and state['replied'] and not state['reply_slots']:
                                state['reply_slots'] = slots_for(dec.options[choice][1], compare.LAST_STATE)
                                state['reply_mv'] = {i: int(compare.LAST_STATE.mv[i])
                                                     for i in state['reply_slots']}
                                state['reply_keys'] = tuple((c[0], c[1], ZONE_ID[c[2]], c[3])
                                                            for c in dec.options[choice][1])
                    if choice is None:
                        choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
                    return choice
                driver.Duel = watched_duel(state, reply)
                try:
                    result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                                          given=([[SPELL, MST, MST, NORMAL] * 10,
                                                  [reply, NORMAL] * 20], [[], []]))
                finally:
                    driver.Duel = NATIVE_DUEL
                assert result['ok'], (getter, hit, result.get('err'))
                assert state['fired'] and state['selected'], state
                assert len(state['target_slots']) == 2, state
                final = compare.LAST_STATE
                target_mv_final = tuple(int(final.mv[i]) for i in state['target_slots'])
                moved_targets = {i for i, after in zip(state['target_slots'], target_mv_final)
                                 if after > state['target_mv'][i]}
                expected_moved = set(state['target_slots'])
                if guard in ('exact_two', 'not_exactly_two') and hit is not None:
                    expected_moved &= set(state['reply_slots'])
                assert moved_targets == expected_moved, (
                    getter, guard, hit, shifted, state['target_slots'],
                    state['target_mv'], target_mv_final, expected_moved,
                    int(final.lp[0]), int(final.pg), result['coverage'], state)
                assert state['replied'] == (hit is not None), state
                if requery_case:
                    assert observed_lp == {8000}, observed_lp
                elif guard == 'positive':
                    if hit == 'both' or shifted:
                        assert observed_lp == {8000}, observed_lp
                    else:
                        assert (5532 if hit is None else 6766) in observed_lp, (hit, observed_lp)
                elif hit is None and not shifted:
                    assert 5532 in observed_lp, (guard, observed_lp)
                else:
                    assert observed_lp == {8000}, (guard, hit, shifted, observed_lp)
                for code in ((SPELL, reply) if hit is not None else (SPELL,)):
                    assert any(e['code'] == code and e['stage'] == 'resolved' and e['count']
                               for e in result['coverage']), result['coverage']
                if hit is not None:
                    assert len(state['reply_slots']) == (2 if hit == 'both' else 1), state
                    assert all(int(final.mv[i]) > state['reply_mv'][i] for i in state['reply_slots']), state
                    assert state['reply_returned'] and state['reply_departed_at_return'], state
                print(getter, guard, 'departure before stacking', shifted, 'response', hit,
                      ':', result['decisions'], 'identical decisions', flush=True)
    finally:
        translate.OFFICIAL = old
        driver.SCRIPTS.remove(folder)
