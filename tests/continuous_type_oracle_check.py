"""Official effects distinguish Continuous Spells/Traps in costs and resolution."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
import driver
from oracle_bank import build

NORMAL, SPELL, TRAP = 44287299, 4542651, 85742772  # Yellow Luster Shield / Gravity Bind
PURIFICATION, MALICE, PLANTER = 1669772, 13626450, 1073952
step, legal = jax.jit(core.step), jax.jit(core.legal)


def check(source, removed, preserved, seed):
    db, bank, cmap = build([NORMAL, SPELL, TRAP, source])
    witnessed = {'both_faceup': False, 'activated': False, 'correct_removal': False}

    def chooser(dec, indices):
        s = compare.LAST_STATE
        def in_zone(code, zone, faceup=False):
            return any(int(bank.code[s.cid[i]]) == code and int(s.ctrl[i]) == 0
                       and int(s.zone[i]) == zone and (not faceup or int(s.pos[i]) & core.P_FACEUP)
                       for i in range(s.cid.shape[0]))
        ready = in_zone(SPELL, core.Z_SZONE, True) and in_zone(TRAP, core.Z_SZONE, True)
        witnessed['both_faceup'] |= ready
        if witnessed['activated'] and in_zone(removed, core.Z_GRAVE) and in_zone(preserved, core.Z_SZONE, True):
            witnessed['correct_removal'] = True
        def find(kinds, code):
            return next((i for i in indices if dec.options[i][0] in kinds and dec.options[i][1] == code), None)
        if dec.player == 0:
            if not witnessed['activated'] and ready and int(s.turn) >= 3:
                choice = find(('activate', 'chain'), source)
                if choice is not None:
                    witnessed['activated'] = True
                    return choice
            for code in (SPELL, TRAP):
                choice = find(('activate', 'chain'), code)
                if choice is not None:
                    return choice
            for code in (TRAP, source):
                choice = find(('sset',), code)
                if choice is not None:
                    return choice
        # The sweep's discard cost must use the normal-monster control,
        # keeping both Continuous cards on the field until resolution.
        for i in indices:
            opt = dec.options[i]
            if opt[0] == 'cards' and all(c[0] == NORMAL for c in opt[1]):
                return i
        return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

    deck = [NORMAL] * 35 + [SPELL, TRAP, source, NORMAL, NORMAL]
    result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                          given=([deck, [NORMAL] * 40], [[], []]))
    assert result['ok'], result.get('err')
    assert all(witnessed.values()), witnessed
    assert any(x['code'] == source and x['stage'] == 'resolved' and x['count']
               for x in result['coverage']), result['coverage']
    print(f"{source}: {result['decisions']} decisions identical; removed {removed}, preserved {preserved}", flush=True)


if __name__ == '__main__':
    check(PURIFICATION, SPELL, TRAP, 3600)
    check(MALICE, TRAP, SPELL, 3601)
    check(PLANTER, TRAP, SPELL, 3602)
