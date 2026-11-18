"""Official Deepsea Macrotrema: field→GY, then +500 ATK to face-up WATER allies.

A synthetic targeted spell sends Deepsea to the Graveyard; native messages
and a simultaneous JAX snapshot verify the official effect and a non-WATER control.
"""
from pathlib import Path
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

DEEPSEA, AQUA, NORMAL, SPELL = 13314457, 85639257, 44287299, 600930141
# The probe is a synthetic rules card; it only sends Deepsea to the GY.
driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Dead GetFirst probe')
spell_script = f'''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return Duel.IsExistingTarget(function(tc) return tc:IsCode({DEEPSEA}) end,tp,LOCATION_MZONE,0,1,nil) end
  Duel.SelectTarget(tp,function(tc) return tc:IsCode({DEEPSEA}) end,tp,LOCATION_MZONE,0,1,1,nil)
 end)
 e:SetOperation(function(e,tp)
  local tc=Duel.GetFirstTarget()
  if tc and tc:IsRelateToEffect(e) then Duel.SendtoGrave(tc,REASON_EFFECT) end
 end)
 c:RegisterEffect(e)
end'''

state = {'moves': [], 'snapshots': [], 'summoned': set(), 'spell': False}

class WatchedDuel(driver.Duel):
    def _handle(self, msg):
        reader = driver.Reader(msg)
        kind = reader.take('B')
        if kind == driver.K['MSG_MOVE']:
            code = reader.take('I')
            previous = tuple(reader.take('BBII'))
            current = tuple(reader.take('BBII'))
            state['moves'].append((code, previous, current))
        return super()._handle(msg)

step, legal = jax.jit(core.step), jax.jit(core.legal)
with tempfile.TemporaryDirectory() as folder:
    Path(folder, f'c{SPELL}.lua').write_text(spell_script)
    old_official = translate.OFFICIAL
    shutil.copy(Path(old_official, f'c{DEEPSEA}.lua'), folder)
    translate.OFFICIAL = folder
    driver.SCRIPTS.insert(0, folder)
    native_duel = driver.Duel
    driver.Duel = WatchedDuel
    try:
        db, bank, cmap = build([DEEPSEA, AQUA, NORMAL, SPELL])

        def chooser(dec, indices):
            current = compare.LAST_STATE
            assert current is not None
            snap = {'turn': int(current.turn), 'cards': {}}
            for i in range(current.cid.shape[0]):
                code = int(bank.code[current.cid[i]])
                if code in (DEEPSEA, AQUA, NORMAL):
                    snap['cards'].setdefault(code, []).append(
                        (int(current.zone[i]), int(current.atk_c[i])))
            state['snapshots'].append(snap)

            def find(kind, code=None):
                return next((i for i in indices if dec.options[i][0] == kind
                             and (code is None or dec.options[i][1] == code)), None)

            choice = None
            if dec.options and dec.options[0][0] == 'cards':
                # The unique Deepsea target may be selected automatically.
                choice = next((i for i in indices if any(c[0] == DEEPSEA and c[1] == 0
                                                         for c in dec.options[i][1])), None)
            if choice is None:
                choice = find('chain', DEEPSEA)
            if choice is None and dec.player == 0:
                for code in (DEEPSEA, AQUA, NORMAL):
                    if code not in state['summoned']:
                        choice = find('summon', code)
                        if choice is not None:
                            state['summoned'].add(code)
                            break
                ready = all(any(zone == core.Z_MZONE for zone, _ in snap['cards'].get(code, []))
                            for code in (DEEPSEA, AQUA, NORMAL))
                if choice is None and int(current.turn) >= 5 and ready and not state['spell']:
                    choice = find('activate', SPELL)
                    if choice is not None:
                        state['spell'] = True
            if choice is None:
                choice = next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])
            return choice

        # P0's Normal Summons happen on T1, T3 and T5.
        deck0 = [NORMAL] * 35 + [AQUA, DEEPSEA, SPELL, NORMAL, NORMAL]
        deck1 = [NORMAL] * 40
        result = compare.play(db, bank, cmap, 3400, step, legal, mode='tr', chooser=chooser,
                              given=([deck0, deck1], [[], []]))
    finally:
        driver.Duel = native_duel
        driver.SCRIPTS.remove(folder)
        translate.OFFICIAL = old_official

assert result['ok'], result.get('err')
assert any(e['code'] == SPELL and e['stage'] == 'resolved' and e['count'] for e in result['coverage'])
assert any(e['code'] == DEEPSEA and e['stage'] == 'resolved' and e['count'] for e in result['coverage'])
assert any(code == DEEPSEA and old[1] == driver.K['LOCATION_MZONE']
           and new[1] == driver.K['LOCATION_GRAVE'] for code, old, new in state['moves'])
assert any(snap['cards'].get(DEEPSEA) and any(zone == core.Z_GRAVE for zone, _ in snap['cards'][DEEPSEA])
           and any(zone == core.Z_MZONE and atk == 1700 for zone, atk in snap['cards'].get(AQUA, []))
           and any(zone == core.Z_MZONE and atk == 1100 for zone, atk in snap['cards'].get(NORMAL, []))
           for snap in state['snapshots']), state['snapshots']
print(f"Deepsea Macrotrema: {result['decisions']} decisions identical; Aqua Madoor 1200→1700; Normal Monster unchanged")
