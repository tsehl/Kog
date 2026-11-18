"""Ordered return guards and counted group summons agree with native Lua."""
from pathlib import Path
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
import driver
import translate
from oracle_bank import build
from group_special_summon_oracle_check import SCRIPT, MONSTERS

SPELL, PSYCHIC, TELEPORT = 600930148, 600930149, 6795211
step, legal = jax.jit(core.step), jax.jit(core.legal)


def group_check(short_circuit=False, overflow=False):
    driver.TEST_CARDS[SPELL] = ((0, 0, 2, 0, 0, 0, 0, 0), 'Counted group probe')
    source = SCRIPT[:SCRIPT.index(' Duel.SpecialSummon(g,')]
    source += (f' if #g{">0" if short_circuit else "==0"} or '
               'Duel.SpecialSummon(g,0,tp,tp,false,false,POS_FACEUP)~=3 then return end\n'
               ' Duel.Recover(tp,123,REASON_EFFECT)\nend')
    if overflow:
        source = source.replace('LOCATION_HAND', 'LOCATION_DECK')
    limit = 2 if overflow else 1
    witnessed = {'activated': 0, 'result': False}
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, f'c{SPELL}.lua').write_text(source)
        old = translate.OFFICIAL
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([SPELL, *MONSTERS])

            def chooser(dec, indices):
                s = compare.LAST_STATE
                field = sum(int(s.zone[i]) == core.Z_MZONE and int(s.ctrl[i]) == 0
                            for i in range(s.cid.shape[0]))
                expected_field = 0 if short_circuit else 5 if overflow else 3
                expected_lp = 8000 if short_circuit else 8123
                if witnessed['activated'] == limit and not int(s.n_chain):
                    witnessed['result'] |= field == expected_field and int(s.lp[0]) == expected_lp
                if dec.options and dec.options[0][0] == 'cards':
                    choice = next((i for i in indices if {c[0] for c in dec.options[i][1]} == set(MONSTERS)), None)
                    if choice is not None:
                        return choice
                if dec.player == 0 and witnessed['activated'] < limit:
                    choice = next((i for i in indices if dec.options[i][:2] == ('activate', SPELL)), None)
                    if choice is not None:
                        witnessed['activated'] += 1
                        return choice
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            deck = [MONSTERS[0]] * 35 + [SPELL, *MONSTERS, MONSTERS[0]]
            if overflow:
                deck = list(MONSTERS) * 11 + [MONSTERS[0]] * 5 + [SPELL] * 2
            result = compare.play(db, bank, cmap, 7900, step, legal, mode='tr', chooser=chooser,
                                  given=([deck, [MONSTERS[0]] * 40], [[], []]))
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
    assert result['ok'], result.get('err')
    assert witnessed['activated'] == limit and witnessed['result'], witnessed
    print(f"Group guard (short_circuit={short_circuit}, overflow={overflow}): "
          f"{result['decisions']} identical decisions", flush=True)


def field_spell_check(source=TELEPORT):
    monster = PSYCHIC if source == TELEPORT else 600930150
    typ = 17 if source == TELEPORT else 0x821
    driver.TEST_CARDS[monster] = ((0, 0, typ, 1000, 1000, 4, 0x100000, 1), 'Hand summon probe')
    db, bank, cmap = build([source, monster, MONSTERS[0]])
    witnessed = {'opponent_summoned': False, 'paid_and_summoned': False}

    def chooser(dec, indices):
        s = compare.LAST_STATE
        field = [(int(s.ctrl[i]), int(bank.code[s.cid[i]])) for i in range(s.cid.shape[0])
                 if int(s.zone[i]) == core.Z_MZONE]
        witnessed['opponent_summoned'] |= (1, MONSTERS[0]) in field
        witnessed['paid_and_summoned'] |= (0, monster) in field and int(s.lp[0]) == (7200 if source == TELEPORT else 8000)
        if dec.player == 1 and not witnessed['opponent_summoned']:
            choice = next((i for i in indices if dec.options[i][:2] == ('summon', MONSTERS[0])), None)
            if choice is not None:
                return choice
        if dec.player == 0 and not witnessed['paid_and_summoned']:
            choice = next((i for i in indices if dec.options[i][:2] == ('activate', source)), None)
            if choice is not None:
                return choice
        if dec.options and dec.options[0][0] == 'cards':
            choice = next((i for i in indices if all(c[0] == monster for c in dec.options[i][1])), None)
            if choice is not None:
                return choice
        return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

    deck = [monster] * 35 + [source] + [monster] * 4
    result = compare.play(db, bank, cmap, 7901, step, legal, mode='tr', chooser=chooser,
                          given=([deck, [MONSTERS[0]] * 40], [[], []]))
    assert result['ok'], result.get('err')
    assert all(witnessed.values()), witnessed
    print(f"Field spell {source}: {result['decisions']} identical decisions; correct LP and summon", flush=True)


if __name__ == '__main__':
    group_check(short_circuit=True)
    group_check()
    group_check(overflow=True)
    field_spell_check()
    # A Gemini-typed effectless fixture isolates the hand filter; no Gemini
    # normal-summon procedure is exercised by this Special Summon scenario.
    with tempfile.TemporaryDirectory() as folder:
        old = translate.OFFICIAL
        shutil.copy(Path(old, 'c57441100.lua'), folder)
        Path(folder, 'c600930150.lua').write_text('local s,id=GetID()\nfunction s.initial_effect(c) end')
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            field_spell_check(57441100)
        finally:
            translate.OFFICIAL = old
            driver.SCRIPTS.remove(folder)
