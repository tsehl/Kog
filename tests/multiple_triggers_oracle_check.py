"""Native + JAX probe for two SINGLE EVENT_SUMMON_SUCCESS effects on one monster.

Run with the project's Python environment from the repository.
"""
from pathlib import Path
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import cards
import core
import driver
import translate
from oracle_bank import build

CARD = 590930080  # below compare.GEN_CODE0: exercise translated DESC mapping
NORMAL = 44287299
DESC0, DESC1 = CARD << 20, (CARD << 20) | 1


def make_script(kind):
    mandatory0 = kind in ('mandatory_optional', 'mandatory_cost_target',
                         'mandatory_cost_condition')
    type0 = 'EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_F' if mandatory0 else \
            'EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_O'
    limits = {
        'independent': ('', ''),
        'shared_hopt': ('e0:SetCountLimit(1,id)', 'e1:SetCountLimit(1,id)'),
        'distinct_keys': ('e0:SetCountLimit(1,{id,0})', 'e1:SetCountLimit(1,{id,1})'),
        'mandatory_optional': ('', ''),
        'mandatory_cost_target': ('', ''),
        'mandatory_cost_condition': ('', ''),
    }[kind]
    cost = '''e0:SetCost(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then
   return Duel.IsExistingMatchingCard(Card.IsType,tp,LOCATION_HAND,0,1,nil,TYPE_MONSTER)
  end
  Duel.DiscardHand(tp,Card.IsType,1,1,REASON_COST+REASON_DISCARD,nil,TYPE_MONSTER)
 end)''' if mandatory0 and kind != 'mandatory_optional' else ''
    target_condition = ''
    if kind == 'mandatory_cost_target':
        target_condition = '''e1:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then
   return Duel.IsExistingMatchingCard(Card.IsType,tp,LOCATION_GRAVE,0,1,nil,TYPE_MONSTER)
  end
 end)'''
    elif kind == 'mandatory_cost_condition':
        target_condition = '''e1:SetCondition(function(e,tp)
  return Duel.IsExistingMatchingCard(Card.IsType,tp,LOCATION_GRAVE,0,1,nil,TYPE_MONSTER)
 end)'''
    return f'''local s,id=GetID()
function s.initial_effect(c)
 local e0=Effect.CreateEffect(c)
 e0:SetDescription(aux.Stringid(id,0))
 e0:SetType({type0})
 e0:SetCode(EVENT_SUMMON_SUCCESS)
 {limits[0]}
 {cost}
 e0:SetOperation(function(e,tp) Duel.Damage(1-tp,111,REASON_EFFECT) end)
 c:RegisterEffect(e0)
 local e1=Effect.CreateEffect(c)
 e1:SetDescription(aux.Stringid(id,1))
 e1:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_O)
 e1:SetCode(EVENT_SUMMON_SUCCESS)
 {limits[1]}
 {target_condition}
 e1:SetOperation(function(e,tp) Duel.Damage(1-tp,222,REASON_EFFECT) end)
 c:RegisterEffect(e1)
end
'''


CASES = (
    # Desired activation order, then expected resolved effect indices.
    ('optional 0 then 1', 'independent', (0, 1), (0, 1)),
    ('optional 1 then 0', 'independent', (1, 0), (1, 0)),
    ('shared HOPT key', 'shared_hopt', (0,), (0,)),
    ('distinct HOPT keys', 'distinct_keys', (1, 0), (1, 0)),
    ('mandatory 0 plus optional 1', 'mandatory_optional', (), (0, 1)),
    ('mandatory cost opens optional target', 'mandatory_cost_target', (), (0, 1)),
    ('mandatory cost leaves failed optional condition unavailable',
     'mandatory_cost_condition', (), (0,)),
)
CASES_TO_RUN = CASES[-2:] if '--conditions-only' in sys.argv else CASES

driver.TEST_CARDS[CARD] = ((0, 0, core.TYPE_MONSTER | cards.TYPE_EFFECT,
                            7500, 2500, 4, 1, 1), 'Multi-trigger probe')


for case_index, (label, kind, preferred_order, expected_indices) in enumerate(CASES_TO_RUN,
                                                                          start=len(CASES)-len(CASES_TO_RUN)):
    with tempfile.TemporaryDirectory(prefix='multi-trigger-') as folder:
        script_path = Path(folder, f'c{CARD}.lua')
        script_path.write_text(make_script(kind))
        old_official = translate.OFFICIAL
        old_scripts = list(driver.SCRIPTS)
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([CARD, NORMAL])
            descriptions = compare.DESC[CARD]
            assert descriptions == [DESC0, DESC1], (label, descriptions)

            state = dict(summoned=False, chosen=[], chain_prompts=[],
                         native_acts=[], native_resolves=[], native_damage=[], native_moves=[])
            native_duel = driver.Duel

            class WatchedDuel(native_duel):
                def _handle(self, msg):
                    reader = driver.Reader(msg)
                    kind_id = reader.take('B')
                    if kind_id == driver.K['MSG_CHAINING']:
                        code = reader.take('I')
                        reader.take('BBII')
                        reader.take('BBI')
                        desc, _chain_index = reader.take('QI')
                        state['native_acts'].append((code, desc))
                    elif kind_id == driver.K['MSG_CHAIN_SOLVED']:
                        chain_index = reader.take('B')
                        if chain_index in self.chains:
                            code, desc, _negated = self.chains[chain_index]
                            state['native_resolves'].append((code, desc))
                    elif kind_id == driver.K['MSG_DAMAGE']:
                        player, amount = reader.take('BI')
                        state['native_damage'].append((player, amount))
                    elif kind_id == driver.K['MSG_MOVE']:
                        code = reader.take('I')
                        previous = reader.take('BBII')
                        destination = reader.take('BBII')
                        reason = reader.take('I')
                        state['native_moves'].append((code, *previous, *destination, reason))
                    return super()._handle(msg)

            def chooser(dec, indices):
                def find(kind_name, code):
                    return next((i for i in indices if dec.options[i][0] == kind_name
                                 and dec.options[i][1] == code), None)

                effect_options = [i for i in indices if dec.options[i][0] == 'chain'
                                  and dec.options[i][1] == CARD]
                if dec.msg in (driver.K['MSG_SELECT_CHAIN'], driver.K['MSG_SELECT_EFFECTYN']):
                    offered = tuple(dec.options[i][5] for i in effect_options)
                    if offered:
                        state['chain_prompts'].append(offered)
                    if effect_options:
                        if kind in ('mandatory_optional', 'mandatory_cost_target'):
                            desired = DESC1 if DESC1 in offered else DESC0
                        elif len(state['chosen']) < len(preferred_order):
                            desired = (DESC0, DESC1)[preferred_order[len(state['chosen'])]]
                        else:
                            desired = None
                        selected = next((i for i in effect_options
                                         if dec.options[i][5] == desired), None)
                        if selected is None and desired is not None:
                            # Continue the native duel so the final assertion reports
                            # the actual prompt/order when this option was unavailable.
                            selected = effect_options[0]
                        if selected is not None and len(state['chosen']) < 2:
                            state['chosen'].append(dec.options[selected][5])
                            return selected
                        return next((i for i in indices if dec.options[i][0] == 'pass'), indices[0])

                current = compare.LAST_STATE
                if dec.player == 0 and not state['summoned']:
                    selected = find('summon', CARD)
                    if selected is not None:
                        state['summoned'] = True
                        return selected
                if dec.player == 0:
                    direct_attack = next((i for i in indices if dec.options[i][0] == 'attack'
                                          and dec.options[i][1] == CARD and dec.options[i][5]), None)
                    if direct_attack is not None:
                        return direct_attack
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')),
                            indices[0])

            driver.Duel = WatchedDuel
            try:
                result = compare.play(
                    db, bank, cmap, 5900 + case_index,
                    jax.jit(core.step), jax.jit(core.legal), mode='tr', chooser=chooser,
                    given=([[*([NORMAL] * 39), CARD], [NORMAL] * 40], [[], []]))
            finally:
                driver.Duel = native_duel
        finally:
            translate.OFFICIAL = old_official
            driver.SCRIPTS[:] = old_scripts

    assert result['ok'], (label, result.get('err'))
    expected_descs = [(DESC0, DESC1)[i] for i in expected_indices]
    actual_acts = [desc for code, desc in state['native_acts'] if code == CARD]
    actual_resolves = [desc for code, desc in state['native_resolves'] if code == CARD]
    assert actual_acts == expected_descs, (
        label, 'native chain messages', actual_acts, state['chain_prompts'])
    assert actual_resolves == list(reversed(expected_descs)), (
        label, 'native resolution messages', actual_resolves)
    assert all(any(e['code'] == CARD and e['stage'] == 'resolved'
                   and e['description'] == desc and e['count'] for e in result['coverage'])
               for desc in expected_descs), (label, result['coverage'])
    damages = [amount for player, amount in state['native_damage'] if player == 1]
    expected_damages = [111 if i == 0 else 222 for i in reversed(expected_indices)]
    assert damages == expected_damages, (label, damages, expected_damages)
    if kind in ('mandatory_cost_target', 'mandatory_cost_condition'):
        assert any(code == NORMAL and oldloc == driver.K['LOCATION_HAND']
                   and newloc == driver.K['LOCATION_GRAVE']
                   and reason & driver.K['REASON_COST']
                   for code, _oldctrl, oldloc, _oldseq, _oldpos,
                   _newctrl, newloc, _newseq, _newpos, reason in state['native_moves']), (
                       label, 'mandatory discard cost did not move a card from hand to GY',
                       state['native_moves'])
    if label.startswith('optional'):
        assert actual_acts == expected_descs, (label, actual_acts, expected_descs,
                                                state['chain_prompts'])
        assert any(set((DESC0, DESC1)) <= set(prompt) for prompt in state['chain_prompts']), (
            label, 'both source effects never appeared as selectable descriptions',
            state['chain_prompts'])
    print(label, ':', result['decisions'], 'decisions identical;',
          len(actual_acts), 'effects activated/resolved', flush=True)
