"""Differential oracle: GetCode~=id excludes all copies of the handler."""
from pathlib import Path
import shutil
import sys
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'engine'), str(ROOT / 'translate'), str(ROOT / 'oracle')]
import compare
import core
import driver
import translate
from oracle_bank import build

SPELL, MST = 600930059, 5318639
driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'GetCode exclusion probe')
script = '''local s,id=GetID()
function s.filter(c)
 return c:IsSpellTrap() and c:GetCode()~=id
end
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk,chkc)
  if chkc then return chkc:IsControler(tp) and chkc:IsLocation(LOCATION_SZONE) and s.filter(chkc) end
  if chk==0 then return Duel.IsExistingTarget(s.filter,tp,LOCATION_SZONE,0,2,e:GetHandler()) end
  Duel.SelectTarget(tp,s.filter,tp,LOCATION_SZONE,0,2,2,e:GetHandler())
 end)
 e:SetOperation(function(e,tp)
  local g=Duel.GetTargetCards(e)
  Duel.Damage(tp,#g*1234,REASON_EFFECT)
 end)
 c:RegisterEffect(e)
end'''
with tempfile.TemporaryDirectory() as folder:
    old_official = translate.OFFICIAL
    old_scripts = list(driver.SCRIPTS)
    try:
        Path(folder, f'c{SPELL}.lua').write_text(script)
        shutil.copy(Path(translate.OFFICIAL, f'c{MST}.lua'), folder)
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        db, bank, cmap = build([SPELL, MST])
        fired = False
        target_choices = 0
        saw_copies_on_field = False
        observed_lp = set()
        set_turns = set()
        # Three valid cards: choosing two must remain a real decision,
        # otherwise compare.play automatically resolves the only combination.
        set_specs = {1: SPELL, 3: SPELL, 5: MST, 7: MST, 9: MST}

        def chooser(dec, indices):
            global fired, target_choices, saw_copies_on_field, set_turns
            current = compare.LAST_STATE
            turn = int(current.turn)
            observed_lp.add(int(current.lp[0]))
            def find(kind, card):
                return next((i for i in indices if dec.options[i][0] == kind and dec.options[i][1] == card), None)
            choice = None
            if dec.player == 0 and turn in set_specs and turn not in set_turns:
                choice = find('sset', set_specs[turn])
                # The quick windows precede the Main Phase: wait until
                # the Set is actually offered.
                if choice is not None:
                    set_turns.add(turn)
            if turn >= 11 and dec.player == 0 and not fired:
                choice = find('activate', SPELL)
                if choice is not None:
                    fired = True
            if dec.options[0][0] == 'cards' and int(current.pg) == core.PG_TARGET and int(current.ch_src) >= 0 \
                    and int(bank.code[current.cid[current.ch_src]]) == SPELL:
                legal_groups = [dec.options[i][1] for i in indices]
                assert legal_groups and all(len(group) == 2 and all(card[0] == MST for card in group)
                                            for group in legal_groups), legal_groups
                target_choices += 1
                same_code_szone = sum(int(bank.code[current.cid[i]]) == SPELL and int(current.zone[i]) == core.Z_SZONE
                                      for i in range(current.cid.shape[0]))
                assert same_code_szone >= 2, same_code_szone
                saw_copies_on_field = True
                choice = indices[0]
            # Keep the opponent passive so it cannot remove either same-name
            # copy or either valid MST before the target selection.
            if dec.player == 1 and choice is None:
                choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), None)
            if choice is None:
                choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
            return choice

        # Include several copies of the same name and three valid candidates.
        result = compare.play(db, bank, cmap, 3400, jax.jit(core.step), jax.jit(core.legal), mode='tr',
                              chooser=chooser,
                              given=([[SPELL, SPELL, MST, MST] * 10, [MST, SPELL] * 20], [[], []]))
        assert result['ok'], result.get('err')
        assert set_turns == set(set_specs), set_turns
        assert fired and target_choices > 0 and saw_copies_on_field, (fired, target_choices, saw_copies_on_field)
        assert 5532 in observed_lp, observed_lp
        assert any(e['code'] == SPELL and e['stage'] == 'resolved' and e['count']
                   for e in result['coverage']), result['coverage']
        print(f"GetCode~=id: handler copies excluded, two targets among three; {result['decisions']} identical decisions", flush=True)
    finally:
        translate.OFFICIAL = old_official
        driver.SCRIPTS[:] = old_scripts
