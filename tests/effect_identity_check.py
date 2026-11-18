"""Replays in full the five previously inconclusive games, original decks."""
import json
from pathlib import Path
from unittest.mock import patch
import jax
from oracle_bank import build
import compare,core,build_inspector,oracle_bank

cases=json.loads((Path(__file__).parent/'fixtures/ambiguous_effects.json').read_text())
codes=[c for g in cases for decks in (g['decks'],g['extras']) for deck in decks for c in deck]
# This check must keep exercising the inspector, even though the compiler now
# groups equivalent SINGLE clones. For these three cards only, unfold the masks
# into the two original slots (same programs and limits).
# The ordinary campaigns and hand_procedure_oracle_check test the grouped form.
original_translate=oracle_bank.translate.translate
def separate_clones(code):
    effects,notes,proc,descs=original_translate(code)
    if code in (87292536,13492423,27877771):
        expanded=[];descriptions=[]
        mask=(1<<core.EV_SUMMON)|(1<<core.EV_FLIP_SUMMON)|(1<<core.EV_SPSUMMON)
        for e,desc in zip(effects,descs):
            events=e[1]&mask
            if e[0] in (core.K_TRIG_O,core.K_TRIG_F) and events.bit_count()>1:
                for bit in range(3):
                    if events&(1<<bit):
                        expanded.append((e[0],(e[1]&~mask)|(1<<bit),*e[2:]));descriptions.append(desc)
            else:
                expanded.append(e);descriptions.append(desc)
        assert len(expanded)<=core.E
        return expanded,notes,proc,descriptions
    return effects,notes,proc,descs
with patch.object(oracle_bank.translate,'translate',side_effect=separate_clones):
    db,bank,cmap=build(codes)
for g in cases:
    with patch.object(build_inspector,'event',wraps=build_inspector.event) as query:
        r=compare.play(db,bank,cmap,g['seed'],jax.jit(core.step),jax.jit(core.legal),mode='tr',given=(g['decks'],g['extras']))
        assert r['ok'],(g['seed'],r.get('err'))
        assert query.call_count>0 and r['decisions']>g['previous_decisions'],(g['seed'],r['decisions'],query.call_count)
        print(f"seed {g['seed']}: identical, {r['decisions']} decisions, {query.call_count} identities read on the oracle side",flush=True)
