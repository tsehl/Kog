"""Tornado destroys one of three opposing main S/T cards, then Escher summons."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
from oracle_bank import build

TORNADO, ESCHER, SET_CARD, NORMAL = 61068510, 24326617, 4542651, 44287299
step, legal = jax.jit(core.step), jax.jit(core.legal)


def main():
    db, bank, cmap = build([TORNADO, ESCHER, SET_CARD, NORMAL])
    witnessed = {'tornado': False, 'escher': False, 'three_before': False, 'two_after': False}

    def chooser(dec, indices):
        s = compare.LAST_STATE
        count = sum(int(s.zone[i]) == core.Z_SZONE and int(s.ctrl[i]) == 1
                    for i in range(s.cid.shape[0]))
        escher_onfield = any(int(bank.code[s.cid[i]]) == ESCHER and int(s.zone[i]) == core.Z_MZONE
                             and int(s.ctrl[i]) == 0 for i in range(s.cid.shape[0]))
        witnessed['two_after'] |= escher_onfield and count == 2
        if dec.player == 1:
            choice = next((i for i in indices if dec.options[i][:2] == ('sset', SET_CARD)), None)
            if choice is not None:
                return choice
        if dec.player == 0:
            if not witnessed['tornado']:
                choice = next((i for i in indices if dec.options[i][:2] == ('activate', TORNADO)), None)
                if choice is not None:
                    assert count == 3, count
                    witnessed['three_before'] = witnessed['tornado'] = True
                    return choice
            if witnessed['tornado'] and not witnessed['escher'] and count == 2:
                choice = next((i for i in indices if dec.options[i][:2] == ('spsummon', ESCHER)), None)
                if choice is not None:
                    witnessed['escher'] = True
                    return choice
        return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

    deck0 = [NORMAL] * 35 + [TORNADO, ESCHER, NORMAL, NORMAL, NORMAL]
    deck1 = [NORMAL] * 35 + [SET_CARD] * 3 + [NORMAL] * 2
    result = compare.play(db, bank, cmap, 3700, step, legal, mode='tr', chooser=chooser,
                          given=([deck0, deck1], [[], []]))
    assert result['ok'], result.get('err')
    assert all(witnessed.values()), witnessed
    for code, stage in ((TORNADO, 'resolved'), (ESCHER, 'summoned')):
        assert any(x['code'] == code and x['stage'] == stage and x['count']
                   for x in result['coverage']), result['coverage']
    print(f"Tornado then Escher: {result['decisions']} decisions identical; opponent S/T count 3→2, Escher Special Summoned")


if __name__ == '__main__':
    main()
