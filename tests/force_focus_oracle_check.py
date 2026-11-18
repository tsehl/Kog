"""Two official Force Focus copies must negate different monsters in one turn.

Synthetic spells only provide the level-six materials and opposing monsters;
the two Xyz summons, detaches and negations use the official card script.
"""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
import driver
import translate
from oracle_bank import build

FOCUS, MATERIAL, TARGET, NORMAL = 64554883, 6631034, 9391354, 44287299
SPAWN_MATERIAL, SPAWN_TARGET = 600930145, 600930146
step, legal = jax.jit(core.step), jax.jit(core.legal)


def spell_script():
    return f'''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(s.target) e:SetOperation(s.operation) c:RegisterEffect(e)
end
function s.filter(c,e,tp)
 return c:IsLevel(6) and c:IsCanBeSpecialSummoned(e,0,tp,false,false)
end
function s.target(e,tp,eg,ep,ev,re,r,rp,chk)
 if chk==0 then return Duel.IsExistingMatchingCard(s.filter,tp,LOCATION_DECK,0,1,nil,e,tp) end
end
function s.operation(e,tp)
 local g=Duel.SelectMatchingCard(tp,s.filter,tp,LOCATION_DECK,0,1,1,nil,e,tp)
 Duel.SpecialSummon(g,0,tp,tp,false,false,POS_FACEUP_ATTACK)
end'''


def main():
    for code in (SPAWN_MATERIAL, SPAWN_TARGET):
        driver.TEST_CARDS[code] = ((0, 0, 2, 0, 0, 0, 0, 0), f'Force Focus setup {code}')
    witnessed = {'xyz': 0, 'first_negated': False, 'both_negated': False}
    sources = set()
    with tempfile.TemporaryDirectory() as folder:
        for code in (SPAWN_MATERIAL, SPAWN_TARGET):
            Path(folder, f'c{code}.lua').write_text(spell_script())
        for code in (FOCUS, TARGET):
            shutil.copy(Path(translate.OFFICIAL, f'c{code}.lua'), folder)
        old_official = translate.OFFICIAL
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([FOCUS, MATERIAL, TARGET, NORMAL, SPAWN_MATERIAL, SPAWN_TARGET])

            def chooser(dec, indices):
                s = compare.LAST_STATE
                negated = sum(int(bank.code[s.cid[i]]) == TARGET and int(s.zone[i]) == core.Z_MZONE
                              and bool(int(s.fl_c[i]) & core.FL_DISABLED) for i in range(s.cid.shape[0]))
                witnessed['first_negated'] |= negated == 1
                witnessed['both_negated'] |= negated == 2
                for i in indices:
                    opt = dec.options[i]
                    if opt[:2] in (('activate', SPAWN_MATERIAL), ('activate', SPAWN_TARGET)):
                        return i
                    if dec.player == 0 and opt[:2] == ('spsummon', FOCUS) and opt[3] == 'extra':
                        witnessed['xyz'] += 1
                        return i
                    if dec.player == 0 and opt[:2] == ('chain', FOCUS) and opt[4] not in sources:
                        # Wait for the first chain to resolve before using the
                        # second copy: exactly one opponent must now be disabled.
                        if not sources or len(sources) == 1 and negated == 1:
                            sources.add(opt[4])
                            return i
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck0 = [MATERIAL] * 4 + [NORMAL] * 32 + [SPAWN_MATERIAL] * 4
            deck1 = [TARGET] * 2 + [NORMAL] * 36 + [SPAWN_TARGET] * 2
            result = compare.play(db, bank, cmap, 3800, step, legal, mode='tr', chooser=chooser, extra_capacity=2,
                                  given=([deck0, deck1], [[FOCUS] * 2, []]))
        finally:
            translate.OFFICIAL = old_official
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert witnessed['xyz'] == 2 and witnessed['first_negated'] and witnessed['both_negated'], witnessed
    assert len(sources) == 2, sources
    for stage in ('summoned', 'activated', 'resolved'):
        assert sum(x['count'] for x in result['coverage'] if x['code'] == FOCUS and x['stage'] == stage) == 2, result['coverage']
    print(f"Force Focus: {result['decisions']} decisions identical; two Xyz summons, two detaches, distinct negated targets")


if __name__ == '__main__':
    main()
