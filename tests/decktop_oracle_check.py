"""SendtoDeck(SEQ_DECKTOP): order of the selected cards and drawing from the top."""
from math import comb
from pathlib import Path
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

SPELL, NORMALS = 991000101, (991000102, 991000103, 991000104)
for code, label in zip(NORMALS, ('Deck top normal A', 'Deck top normal B', 'Deck top normal C')):
    driver.TEST_CARDS[code] = ((0, 0, core.TYPE_MONSTER | core.TYPE_NORMAL, 800, 800, 4, 1, 1), label)
driver.TEST_CARDS[SPELL] = ((0, 0, core.TYPE_SPELL, 0, 0, 0, 0, 0), 'Deck top probe')

script = '''local s,id=GetID()
function s.filter(c)
 return c:IsType(TYPE_NORMAL)
end
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chkc then return chkc:IsControler(tp) and chkc:IsLocation(LOCATION_MZONE) and s.filter(chkc) end
  if chk==0 then return Duel.IsExistingTarget(s.filter,tp,LOCATION_MZONE,0,TARGET_COUNT,e:GetHandler()) end
  Duel.SelectTarget(tp,s.filter,tp,LOCATION_MZONE,0,TARGET_COUNT,TARGET_COUNT,e:GetHandler())
 end)
 e:SetOperation(function(e,tp)
  local g=Duel.GetTargetCards(e)
  if #g==TARGET_COUNT then
   Duel.SendtoDeck(g,nil,SEQ_DECKTOP,REASON_EFFECT)
   Duel.Draw(tp,1,REASON_EFFECT)
  end
 end)
 c:RegisterEffect(e)
end'''

ZONE_ID = {'deck': core.Z_DECK, 'hand': core.Z_HAND, 'mzone': core.Z_MZONE,
           'szone': core.Z_SZONE, 'grave': core.Z_GRAVE, 'removed': core.Z_REMOVED,
           'extra': core.Z_EXTRA}
NATIVE_DUEL = driver.Duel


def watched_duel(state):
    class WatchedDuel(NATIVE_DUEL):
        def _handle(self, msg):
            reader = driver.Reader(msg)
            kind = reader.take('B')
            if kind == driver.K['MSG_MOVE']:
                code = reader.take('I')
                reader.take('BBII')  # Previous location.
                ctrl, zone, seq, _ = reader.take('BBII')
                state['native_moves'].append((code, ctrl, zone, seq))
            elif kind == driver.K['MSG_DRAW']:
                player = reader.take('B')
                count = reader.take('I')
                for _ in range(count):
                    code, _position = reader.take('II')
                    state['native_moves'].append((code, player, driver.K['LOCATION_HAND'], -1))
            return super()._handle(msg)
    return WatchedDuel


def slots_for(cards, current, bank):
    slots = []
    for card in cards:
        code, ctrl, zone_name, seq = card[:4]
        zone = ZONE_ID[zone_name]
        matches = [i for i in range(current.cid.shape[0])
                   if int(bank.code[current.cid[i]]) == code and int(current.ctrl[i]) == ctrl
                   and int(current.zone[i]) == zone and int(current.seq[i]) == seq]
        assert len(matches) == 1, ('selected card absent or ambiguous', card, matches)
        slots.append(matches[0])
    assert len(set(slots)) == len(cards), ('duplicate selected cards', cards, slots)
    return tuple(slots)


with tempfile.TemporaryDirectory() as folder:
    old_official = translate.OFFICIAL
    old_scripts = list(driver.SCRIPTS)
    try:
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        for target_count in (1, 2):
            Path(folder, f'c{SPELL}.lua').write_text(script.replace('TARGET_COUNT', str(target_count)))
            db, bank, cmap = build([SPELL, *NORMALS])
            state = dict(fired=False, spell_set=False, summoned={}, target_prompts=0,
                         selected=(), target_snapshot={}, expected_top=None,
                         operation_snapshot=None, native_moves=[])
            summon_at = {1: NORMALS[0], 3: NORMALS[1], 5: NORMALS[2]}

            def chooser(dec, indices):
                current = compare.LAST_STATE
                turn = int(current.turn)

                # The first decision after the operation observes the completed
                # deck move and draw; no decision occurs between these Lua calls.
                if state['selected'] and state['operation_snapshot'] is None:
                    moved = all(int(current.mv[i]) > state['target_snapshot'][i]['mv']
                                and int(current.zone[i]) in (core.Z_DECK, core.Z_HAND)
                                for i in state['selected'])
                    if moved:
                        state['operation_snapshot'] = {
                            i: (int(current.zone[i]), int(current.seq[i]), int(current.mv[i]),
                                int(bank.code[current.cid[i]])) for i in state['selected']}

                def find(kind, code):
                    return next((i for i in indices if dec.options[i][0] == kind
                                 and dec.options[i][1] == code), None)

                choice = None
                if dec.player == 0 and turn in summon_at \
                        and summon_at[turn] not in state['summoned']:
                    choice = find('summon', summon_at[turn])
                    if choice is not None:
                        state['summoned'][summon_at[turn]] = turn
                if (choice is None and dec.player == 0 and turn == 1
                        and not state['spell_set']):
                    choice = find('sset', SPELL)
                    if choice is not None:
                        state['spell_set'] = True
                if dec.player == 0 and turn >= 7 and not state['fired']:
                    choice = find('activate', SPELL)
                    if choice is not None:
                        state['fired'] = True

                if (dec.options[0][0] == 'cards' and int(current.pg) == core.PG_TARGET
                        and int(current.ch_src) >= 0
                        and int(bank.code[current.cid[current.ch_src]]) == SPELL):
                    groups = [dec.options[i][1] for i in indices]
                    assert len(groups) == comb(3, target_count), (target_count, groups)
                    assert all(len(group) == target_count and all(
                        card[0] in NORMALS and card[1] == 0 and card[2] == 'mzone'
                        for card in group) for group in groups), (target_count, groups)
                    on_field = [i for i in range(current.cid.shape[0])
                                if int(bank.code[current.cid[i]]) in NORMALS
                                and int(current.ctrl[i]) == 0 and int(current.zone[i]) == core.Z_MZONE]
                    assert len(on_field) == 3, ('expected three normal monsters', on_field)
                    state['target_prompts'] += 1
                    choice = indices[0]
                    state['selected'] = slots_for(dec.options[choice][1], current, bank)
                    assert len(state['selected']) == target_count
                    state['target_snapshot'] = {
                        i: dict(code=int(bank.code[current.cid[i]]), ctrl=int(current.ctrl[i]),
                                zone=int(current.zone[i]), seq=int(current.seq[i]),
                                mv=int(current.mv[i])) for i in state['selected']}
                    # The last card in operation-sort order has the highest
                    # original field sequence and is expected on top of deck.
                    state['expected_top'] = max(state['selected'],
                        key=lambda i: state['target_snapshot'][i]['seq'])

                if dec.player == 1 and choice is None:
                    choice = next((i for i in indices
                                   if dec.options[i][0] in ('pass', 'to_ep')), None)
                if choice is None:
                    choice = next((i for i in indices
                                   if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
                return choice

            driver.Duel = watched_duel(state)
            try:
                result = compare.play(db, bank, cmap, 3400, jax.jit(core.step), jax.jit(core.legal),
                                      mode='tr', chooser=chooser,
                                      given=([[SPELL, *NORMALS] * 10,
                                              [NORMALS[0]] * 40], [[], []]))
            finally:
                driver.Duel = NATIVE_DUEL

            assert result['ok'], (target_count, result.get('err'))
            assert state['fired'] and state['spell_set'], state
            assert state['summoned'] == dict(zip(NORMALS, (1, 3, 5))), state['summoned']
            assert state['target_prompts'] == 1 and len(state['selected']) == target_count, state
            assert state['operation_snapshot'] is not None, state
            final = state['operation_snapshot']
            assert all(final[i][2] > state['target_snapshot'][i]['mv'] for i in state['selected']), state
            drawn = [i for i in state['selected'] if final[i][0] == core.Z_HAND]
            assert drawn == [state['expected_top']], (target_count, drawn, state)
            assert final[state['expected_top']][3] == state['target_snapshot'][state['expected_top']]['code'], state
            assert all(final[i][0] == core.Z_DECK for i in state['selected'] if i != state['expected_top']), state
            native_deck = driver.K['LOCATION_DECK']
            native_hand = driver.K['LOCATION_HAND']
            selected_codes = {state['target_snapshot'][i]['code'] for i in state['selected']}
            native_moves = state['native_moves']
            assert all(any(code == selected_code and zone == native_deck
                           for code, _ctrl, zone, _seq in native_moves)
                       for selected_code in selected_codes), (target_count, native_moves, state)
            top_code = state['target_snapshot'][state['expected_top']]['code']
            top_deck_move = next(i for i, move in enumerate(native_moves)
                                 if move[0] == top_code and move[2] == native_deck)
            assert any(code == top_code and zone == native_hand
                       for code, _ctrl, zone, _seq in native_moves[top_deck_move + 1:]), (
                           target_count, top_code, native_moves)
            for stage in ('activated', 'resolved'):
                assert any(e['code'] == SPELL and e['stage'] == stage and e['count']
                           for e in result['coverage']), result['coverage']
            print('SendtoDeck top', target_count, 'target(s), draw',
                  state['target_snapshot'][state['expected_top']]['code'],
                  ':', result['decisions'], 'identical decisions', flush=True)
    finally:
        translate.OFFICIAL = old_official
        driver.SCRIPTS[:] = old_scripts
