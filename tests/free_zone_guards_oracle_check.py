"""Official Draconnet and Gottoms use newly translated free-zone guards."""
from pathlib import Path
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

DRACONNET, GOTTOMS, NORMAL, SABER, PREP, REPLY = 62706865, 13504844, 12482652, 600930151, 600930152, 600930153
step, legal = jax.jit(core.step), jax.jit(core.legal)


def draconnet(source):
    db, bank, cmap = build([DRACONNET, NORMAL])
    seen = {'summoned': False, 'activated': False, 'selected_source': False, 'defense': False}

    def chooser(dec, indices):
        s = compare.LAST_STATE
        seen['defense'] |= any(int(bank.code[s.cid[i]]) == NORMAL and int(s.ctrl[i]) == 0
                               and int(s.zone[i]) == core.Z_MZONE and int(s.pos[i]) == core.P_FU_DEF
                               for i in range(s.cid.shape[0]))
        if dec.player == 0 and not seen['summoned']:
            choice = next((i for i in indices if dec.options[i][:2] == ('summon', DRACONNET)), None)
            if choice is not None:
                seen['summoned'] = True
                return choice
        choice = next((i for i in indices if dec.options[i][0] in ('activate', 'chain')
                       and dec.options[i][1] == DRACONNET), None)
        if choice is not None:
            seen['activated'] = True
            return choice
        if seen['activated'] and dec.options and dec.options[0][0] == 'cards':
            choice = next((i for i in indices if len(dec.options[i][1]) == 1
                           and dec.options[i][1][0][0] == NORMAL and dec.options[i][1][0][2] == source), None)
            if choice is not None:
                seen['selected_source'] = True
                return choice
        # Accept the optional summon trigger, but otherwise pass every window.
        return next((i for i in indices if dec.options[i][0] in ('yes', 'trigger')),
                    next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0]))

    deck = [NORMAL] * 35 + [DRACONNET] + [NORMAL] * 4
    result = compare.play(db, bank, cmap, 8100, step, legal, mode='tr', chooser=chooser,
                          given=([deck, [NORMAL] * 40], [[], []]))
    assert result['ok'], result.get('err')
    assert all(seen.values()), seen
    print(f"Draconnet from {source}: {result['decisions']} identical decisions; summoned face-up Defense", flush=True)


def gottoms(crowded=False):
    driver.TEST_CARDS[SABER] = ((0, 0x100d, 17, 1000, 1000, 4, 1, 1), 'X-Saber normal probe')
    driver.TEST_CARDS[PREP] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Graveyard preparation probe')
    driver.TEST_CARDS[REPLY] = ((0, 0, 0x10002, 0, 0, 0, 0, 0), 'Response summon probe')
    script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(s.target) e:SetOperation(s.operation) c:RegisterEffect(e)
end
function s.target(e,tp,eg,ep,ev,re,r,rp,chk)
 if chk==0 then return Duel.IsExistingMatchingCard(Card.IsMonster,tp,LOCATION_HAND,0,2,nil) end
end
function s.operation(e,tp)
 local g=Duel.SelectMatchingCard(tp,Card.IsMonster,tp,LOCATION_HAND,0,2,2,nil)
 Duel.SendtoGrave(g,REASON_EFFECT)
end'''
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, f'c{PREP}.lua').write_text(script)
        response = script.replace('LOCATION_HAND', 'LOCATION_DECK').replace(',2,2,nil', ',1,1,nil').replace(',2,nil', ',1,nil')
        response = response.replace('Duel.SendtoGrave(g,REASON_EFFECT)',
                                    'Duel.SpecialSummon(g,0,tp,tp,false,false,POS_FACEUP)')
        Path(folder, f'c{REPLY}.lua').write_text(response)
        old = translate.OFFICIAL
        # Preparation and response are synthetic; official Gottoms remains intact.
        import shutil
        shutil.copy(Path(old, f'c{GOTTOMS}.lua'), folder)
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([GOTTOMS, SABER, PREP, REPLY])
            seen = {'normal': False, 'prepared': False, 'activated': False, 'result': False}
            replied = False

            def chooser(dec, indices):
                nonlocal replied
                s = compare.LAST_STATE
                own = sum(int(s.zone[i]) == core.Z_MZONE and int(s.ctrl[i]) == 0
                          for i in range(s.cid.shape[0]))
                grave = sum(int(s.zone[i]) == core.Z_GRAVE and int(s.ctrl[i]) == 0 and int(bank.code[s.cid[i]]) == SABER
                            for i in range(s.cid.shape[0]))
                if seen['activated'] and not int(s.n_chain):
                    seen['result'] |= own == (4 if crowded else 3) and grave == (2 if crowded else 0)
                if dec.player == 0:
                    if crowded and seen['activated'] and not replied:
                        choice = next((i for i in indices if dec.options[i][:2] == ('chain', REPLY)), None)
                        if choice is not None:
                            replied = True
                            return choice
                    for kind, code, flag in (('sset', GOTTOMS, None), ('summon', SABER, 'normal'),
                                              ('activate', PREP, 'prepared'), ('activate', GOTTOMS, 'activated')):
                        if flag == 'normal' and crowded:
                            if own >= 3:
                                continue
                        elif flag and seen[flag]:
                            continue
                        if code == GOTTOMS and kind == 'activate' and crowded and int(s.turn) < 7:
                            continue
                        choice = next((i for i in indices if dec.options[i][:2] == (kind, code)), None)
                        if choice is not None:
                            if code == GOTTOMS and kind == 'activate':
                                assert own == (3 if crowded else 1) and int(s.turn) >= 3
                            if flag:
                                seen[flag] = True
                            return choice
                if dec.options and dec.options[0][0] == 'cards':
                    choice = next((i for i in indices if len(dec.options[i][1]) == 2
                                   and all(c[0] == SABER for c in dec.options[i][1])), None)
                    if choice is not None:
                        return choice
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck = [SABER] * 35 + [GOTTOMS, PREP] + [SABER] * 3
            if crowded:
                deck[-8] = REPLY
            result = compare.play(db, bank, cmap, 8101, step, legal, mode='tr', chooser=chooser,
                                  given=([deck, [SABER] * 40], [[], []]))
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert all(seen.values()), (seen, replied)
    assert not crowded or replied, 'response did not resolve'
    print(f"Gottoms (crowded={crowded}): {result['decisions']} identical decisions; "
          f"{'both targets remain in Graveyard' if crowded else 'two related targets revived'}", flush=True)


if __name__ == '__main__':
    draconnet('hand')
    draconnet('deck')
    gottoms()
    gottoms(crowded=True)
