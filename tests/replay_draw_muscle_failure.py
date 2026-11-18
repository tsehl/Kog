"""Exact replay of the campaign divergence, with the same prefix of oracle decisions."""
import json
from pathlib import Path
import jax
from oracle_bank import build
import core,compare
fixture=json.loads((Path(__file__).parent/'fixtures/draw_muscle_182.json').read_text())
codes=sorted({c for d in fixture['decks']+fixture['extras'] for c in d})
db,bank,cmap=build(codes)
result=compare.play(db,bank,cmap,fixture['seed'],jax.jit(core.step),jax.jit(core.legal),mode='tr',
                    given=(fixture['decks'],fixture['extras']),extra_capacity=15)
assert result['ok'],result.get('err')
options=json.loads(json.dumps([t['option'] for t in result['trace']]))
assert options[:len(fixture['oracle_prefix'])]==fixture['oracle_prefix'],'oracle prefix modified'
resolved=sum(e['count'] for e in result['coverage'] if e['code']==41367003 and e['stage']=='resolved')
assert resolved>=2,resolved
print(f"Seed 182: complete game identical, oracle prefix of {len(fixture['oracle_prefix'])} decisions unchanged, {resolved} Draw Muscle resolved",flush=True)
