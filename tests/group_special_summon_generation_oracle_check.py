"""Exercise actual generated group-summoning cards, including partial success.

Seeds 1, 18 and 83 provide unmodified generated Spells summoning groups
from the hand, Graveyard and Deck. Decks and choices make each effect resolve.
"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'gen', 'oracle')]
import jax
import cardgen
import compare
import core
import driver
import tolua


def run():
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    for seed, spell, count, source, activations in (
            (1, 600000085, 2, core.Z_HAND, 1),
            (18, 600000069, 3, core.Z_GRAVE, 1),
            (83, 600000261, 3, core.Z_DECK, 2)):
        cset = cardgen.Gen(seed).make_set(400, 80)
        card = next(c for c in cset if c['code'] == spell)
        clause, = card['effects'][0]['clauses']
        assert clause['verb'] == 'ss' and clause['k'] == count and clause['f']['me'] == source
        assert clause['sel'] == 'pick'
        normal = next(c['code'] for c in cset
                      if c['ctype'] == core.TYPE_MONSTER | core.TYPE_NORMAL
                      and c['atk'] <= clause['f'].get('atk', (-core.BIG, core.BIG))[1]
                      and (not clause['f'].get('race') or c['race'] & clause['f']['race']))
        bank = cardgen.to_bank(cset)
        cmap = {c['code']: i for i, c in enumerate(cset)}
        db = driver.CardDB()
        db.add(tolua.cdb_rows(cset))
        compare.EFFECT_KINDS.clear()
        compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']]
                                     for c in cset})
        seen = dict(activations=0, first_group=False, full=False, failed=False)

        def chooser(dec, indices):
            s = compare.LAST_STATE
            monsters = bank.code[s.cid] == normal
            field = monsters & (s.zone == core.Z_MZONE) & (s.ctrl == 0)
            seen['first_group'] |= int(field.sum()) == count
            seen['full'] |= int(field.sum()) == 5
            seen['failed'] |= bool((monsters & (s.zone == core.Z_GRAVE)
                                    & (s.reason == core.REASON_RULE)).any())
            if dec.player == 0 and seen['activations'] < activations:
                choice = next((i for i in indices if dec.options[i][:2] == ('activate', spell)), None)
                if choice is not None:
                    seen['activations'] += 1
                    return choice
            if dec.options and dec.options[0][0] == 'cards':
                # Discard monsters (also at the hand limit), keeping the Spell.
                choice = next((i for i in indices if dec.options[i][1]
                               and all(c[0] == normal for c in dec.options[i][1])), None)
                if choice is not None:
                    return choice
            return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

        with tempfile.TemporaryDirectory() as folder:
            tolua.write(cset, folder)
            driver.SCRIPTS.insert(0, folder)
            try:
                main = [normal] * (40 - activations) + [spell] * activations
                result = compare.play(db, bank, cmap, 7600 + seed, step, legal,
                                      mode='gen', chooser=chooser,
                                      given=([main, [normal] * 40], [[], []]))
            finally:
                driver.SCRIPTS.remove(folder)
        assert result['ok'], result.get('err')
        assert seen['activations'] == activations and seen['first_group'], seen
        if activations == 2:
            assert seen['full'] and seen['failed'], seen
        resolved = sum(e['count'] for e in result['coverage']
                       if e['code'] == spell and e['stage'] == 'resolved')
        assert resolved == activations, (resolved, seen)
        print(f'bank {seed}: {result["decisions"]} identical decisions; '
              f'generated k={count}, resolutions={resolved}, overflow={seen["failed"]}', flush=True)


if __name__ == '__main__':
    run()
