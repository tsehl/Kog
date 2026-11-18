"""Sonic Bird and Senju search opposite primary types sharing TYPE_RITUAL."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
import driver
from oracle_bank import build

NORMAL, RITUAL_MONSTER, RITUAL_SPELL = 44287299, 5405694, 55761792
SONIC, SENJU = 57617178, 23401839
step, legal = jax.jit(core.step), jax.jit(core.legal)


def check(searcher, expected, excluded, seed):
    db, bank, cmap = build([NORMAL, RITUAL_MONSTER, RITUAL_SPELL, searcher])
    witnessed = {'summoned': False, 'correct_search': False}

    def chooser(dec, indices):
        s = compare.LAST_STATE
        def in_zone(code, zone):
            return any(int(bank.code[s.cid[i]]) == code and int(s.ctrl[i]) == 0
                       and int(s.zone[i]) == zone for i in range(s.cid.shape[0]))
        # Capture both outcomes at the same decision, before later draws can
        # move the opposite-category control out of the Deck.
        if in_zone(searcher, core.Z_MZONE) and in_zone(expected, core.Z_HAND) and in_zone(excluded, core.Z_DECK):
            witnessed['correct_search'] = True
        if dec.player == 0 and not witnessed['summoned']:
            for i in indices:
                if dec.options[i][:2] == ('summon', searcher):
                    witnessed['summoned'] = True
                    return i
        for i in indices:
            if dec.options[i][:2] == ('chain', searcher):
                return i
        return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

    deck = [RITUAL_MONSTER, RITUAL_SPELL] + [NORMAL] * 33 + [NORMAL] * 4 + [searcher]
    result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                          given=([deck, [NORMAL] * 40], [[], []]))
    assert result['ok'], result.get('err')
    assert all(witnessed.values()), witnessed
    assert any(x['code'] == searcher and x['stage'] == 'resolved' and x['count']
               for x in result['coverage']), result['coverage']
    print(f"{searcher}: {result['decisions']} decisions identical; searched {expected}, excluded {excluded}", flush=True)


if __name__ == '__main__':
    check(SONIC, RITUAL_SPELL, RITUAL_MONSTER, 3500)
    check(SENJU, RITUAL_MONSTER, RITUAL_SPELL, 3501)
