"""Special Summon a mixed group, then modify that same Lua Group."""
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

SPELL = 600930147
MONSTERS = (44287299, 85639257, 6631034)
step, legal = jax.jit(core.step), jax.jit(core.legal)
SCRIPT = """local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetTarget(s.target) e:SetOperation(s.operation) c:RegisterEffect(e)
end
function s.filter(c,e,tp)
 return c:IsMonster() and c:IsCanBeSpecialSummoned(e,0,tp,false,false)
end
function s.target(e,tp,eg,ep,ev,re,r,rp,chk)
 if chk==0 then return Duel.IsExistingMatchingCard(s.filter,tp,LOCATION_HAND,0,3,nil,e,tp) end
end
function s.operation(e,tp)
 local g=Duel.SelectMatchingCard(tp,s.filter,tp,LOCATION_HAND,0,3,3,nil,e,tp)
 Duel.SpecialSummon(g,0,tp,tp,false,false,POS_FACEUP)
 for tc in aux.Next(g) do
  local e1=Effect.CreateEffect(e:GetHandler())
  e1:SetType(EFFECT_TYPE_SINGLE) e1:SetCode(EFFECT_UPDATE_ATTACK)
  e1:SetValue(500) e1:SetReset(RESET_EVENT+RESETS_STANDARD)
  tc:RegisterEffect(e1)
 end
end"""

def run(overfill=False):
    driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Group special summon probe')
    witnessed = {'activated': 0, 'group_boosted': False, 'overflow_grave': False}
    limit = 2 if overfill else 1
    first_group = set()
    snapshots = set()
    with tempfile.TemporaryDirectory() as folder:
        script = SCRIPT.replace('LOCATION_HAND', 'LOCATION_DECK') if overfill else SCRIPT
        Path(folder, f'c{SPELL}.lua').write_text(script)
        old = translate.OFFICIAL
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([SPELL, *MONSTERS])

            def chooser(dec, indices):
                s = compare.LAST_STATE
                field = {i for i in range(s.cid.shape[0])
                         if int(s.zone[i]) == core.Z_MZONE and int(s.ctrl[i]) == 0}
                boosted = {i for i in field if int(s.atk_c[i]) == int(bank.atk[s.cid[i]]) + 500}
                if len(field) == 3 and field == boosted and not first_group:
                    first_group.update(field)
                    if not overfill:
                        witnessed['group_boosted'] = True
                if overfill and len(field) == 5:
                    new_group = field - first_group
                    witnessed['group_boosted'] |= len(new_group) == 2 and new_group <= boosted
                    snapshots.add(tuple(sorted((int(bank.code[s.cid[i]]), int(s.atk_c[i])) for i in field)))
                witnessed['overflow_grave'] |= any(
                    int(s.zone[i]) == core.Z_GRAVE and int(bank.code[s.cid[i]]) in MONSTERS
                    and int(s.reason[i]) == core.REASON_RULE for i in range(s.cid.shape[0]))
                if dec.options and dec.options[0][0] == 'cards':
                    choice = next((i for i in indices
                                   if {c[0] for c in dec.options[i][1]} == set(MONSTERS)), None)
                    if choice is not None:
                        return choice
                if dec.player == 0 and witnessed['activated'] < limit:
                    choice = next((i for i in indices if dec.options[i][:2] == ('activate', SPELL)), None)
                    if choice is not None:
                        witnessed['activated'] += 1
                        return choice
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck0 = [MONSTERS[0]] * 35 + [SPELL, *MONSTERS, MONSTERS[0]]
            if overfill:
                deck0 = list(MONSTERS) * 11 + [MONSTERS[0]] * 5 + [SPELL] * 2
            result = compare.play(db, bank, cmap, 3900, step, legal, mode='tr', chooser=chooser,
                                  given=([deck0, [MONSTERS[0]] * 40], [[], []]))
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert witnessed['activated'] == limit and witnessed['group_boosted'], (witnessed, snapshots)
    if overfill:
        assert witnessed['overflow_grave'], witnessed
    print(f"Group special summon ({'overflow' if overfill else 'hand'}): "
          f"{result['decisions']} identical decisions; same group boosted")


def run_extra():
    """A Link placed by the same group cannot yet open a main monster zone."""
    import shutil
    link = 77637979  # LANphorhynchus: bottom-left and bottom-right arrows.
    driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Group special summon probe')
    witnessed = {'activated': False, 'one_link': False}
    with tempfile.TemporaryDirectory() as folder:
        script = SCRIPT.replace('LOCATION_HAND', 'LOCATION_EXTRA').replace(',3,3,', ',2,2,').replace(',3,nil', ',2,nil')
        Path(folder, f'c{SPELL}.lua').write_text(script)
        old = translate.OFFICIAL
        shutil.copy(Path(old, f'c{link}.lua'), folder)
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([SPELL, MONSTERS[0], link])

            def chooser(dec, indices):
                s = compare.LAST_STATE
                links = [i for i in range(s.cid.shape[0]) if int(bank.code[s.cid[i]]) == link]
                field = [i for i in links if int(s.zone[i]) == core.Z_MZONE]
                grave = [i for i in links if int(s.zone[i]) == core.Z_GRAVE]
                witnessed['one_link'] |= (len(field) == len(grave) == 1
                    and int(s.reason[grave[0]]) == core.REASON_RULE
                    and int(s.atk_c[field[0]]) == int(bank.atk[s.cid[field[0]]]) + 500)
                if dec.player == 0 and not witnessed['activated']:
                    choice = next((i for i in indices if dec.options[i][:2] == ('activate', SPELL)), None)
                    if choice is not None:
                        witnessed['activated'] = True
                        return choice
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck0 = [MONSTERS[0]] * 35 + [SPELL] + [MONSTERS[0]] * 4
            result = compare.play(db, bank, cmap, 3901, step, legal, mode='tr', chooser=chooser,
                                  extra_capacity=2,
                                  given=([deck0, [MONSTERS[0]] * 40], [[link] * 2, []]))
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert all(witnessed.values()), witnessed
    print(f"Group special summon (Link): {result['decisions']} identical decisions; "
          "one Link summoned, other sent by rule, pending arrows inactive")


if __name__ == '__main__':
    run()
    run(overfill=True)
    run_extra()
