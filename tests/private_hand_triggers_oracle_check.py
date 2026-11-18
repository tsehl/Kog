"""Garmides after a hand-limit discard: alone, or responding to a public trigger."""
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

GARMIDES, NORMAL, PUBLIC, PREP = 43411769, 12482652, 600930155, 600930156
step, legal = jax.jit(core.step), jax.jit(core.legal)


def check(public, via_spell=False):
    driver.TEST_CARDS[PUBLIC] = ((0, 0, 33, 1000, 1000, 4, 1, 1), 'Public graveyard trigger probe')
    driver.TEST_CARDS[PREP] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Hand send probe')
    script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetDescription(aux.Stringid(id,0))
 e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_TRIGGER_O)
 e:SetProperty(EFFECT_FLAG_DELAY) e:SetCode(EVENT_TO_GRAVE)
 e:SetRange(LOCATION_MZONE) e:SetOperation(s.op) c:RegisterEffect(e)
end
function s.op(e,tp) Duel.Recover(tp,123,REASON_EFFECT) end'''
    with tempfile.TemporaryDirectory() as folder:
        old = translate.OFFICIAL
        shutil.copy(Path(old, f'c{GARMIDES}.lua'), folder)
        Path(folder, f'c{PUBLIC}.lua').write_text(script)
        Path(folder, f'c{PREP}.lua').write_text('''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(s.target) e:SetOperation(s.op) c:RegisterEffect(e)
end
function s.target(e,tp,eg,ep,ev,re,r,rp,chk)
 if chk==0 then return Duel.IsExistingMatchingCard(Card.IsType,tp,LOCATION_HAND,0,1,nil,TYPE_NORMAL) end
end
function s.op(e,tp)
 local g=Duel.SelectMatchingCard(tp,Card.IsType,tp,LOCATION_HAND,0,1,1,nil,TYPE_NORMAL)
 Duel.SendtoGrave(g,REASON_EFFECT)
end''')
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([GARMIDES, NORMAL, PUBLIC, PREP])
            seen = {'discard': False, 'public': False, 'private': False, 'summoned': False}
            prepared = False

            def chooser(dec, indices):
                nonlocal prepared
                s = compare.LAST_STATE
                seen['summoned'] |= any(int(bank.code[s.cid[i]]) == GARMIDES and int(s.zone[i]) == core.Z_MZONE
                                        for i in range(s.cid.shape[0]))
                if via_spell and dec.player == 0 and not prepared:
                    choice = next((i for i in indices if dec.options[i][:2] == ('activate', PREP)), None)
                    if choice is not None:
                        prepared = True
                        return choice
                if int(s.dec) == core.D_DISCARD and dec.player == 1:
                    seen['discard'] = True
                    choice = next((i for i in indices if dec.options[i][0] == 'cards'
                                   and all(c[0] == NORMAL for c in dec.options[i][1])), None)
                    if choice is not None:
                        return choice
                if public and dec.player == 0:
                    choice = next((i for i in indices if dec.options[i][:2] == ('summon', PUBLIC)), None)
                    if choice is not None:
                        return choice
                    if seen['discard'] and not seen['public']:
                        choice = next((i for i in indices if dec.options[i][0] in ('chain', 'activate')
                                       and dec.options[i][1] == PUBLIC), None)
                        if choice is not None:
                            seen['public'] = True
                            return choice
                choice = next((i for i in indices if dec.options[i][0] in ('chain', 'activate')
                               and dec.options[i][1] == GARMIDES), None)
                if choice is not None:
                    assert (public and seen['public'] and int(s.n_chain) > 0
                            or via_spell and prepared and not int(s.n_chain)), (seen, dec.options)
                    seen['private'] = True
                    return choice
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck0, deck1 = [NORMAL] * 40, [NORMAL] * 35 + [GARMIDES] + [NORMAL] * 4
            if public:
                deck0[-5] = PUBLIC
            if via_spell:
                deck0[-5:-3] = [GARMIDES, PREP]
                deck1 = [NORMAL] * 40
            result = compare.play(db, bank, cmap, 8117, step, legal, mode='tr', chooser=chooser,
                                  given=([deck0, deck1], [[], []]))
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert seen['discard'], seen
    assert seen['public'] == public and seen['private'] == seen['summoned'] == (public or via_spell), seen
    print(f'Garmides (public={public}, via_spell={via_spell}): {result["decisions"]} identical decisions; {seen}', flush=True)


if __name__ == '__main__':
    check(False)
    check(True)
    check(False, via_spell=True)
