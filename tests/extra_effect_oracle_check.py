"""Cyber-Stein does not complete the Fusion procedure: Reborn must then reject it."""
import jax
from oracle_bank import build
import core,compare
STEIN,FUSION,REBORN,HOLE,ELF=69015963,8692301,83764718,53129443,44287299
db,bank,cmap=build([STEIN,FUSION,REBORN,HOLE,ELF])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for seed in (2100,2101):
    seen=set();summoned=0
    def chooser(dec,indices):
        global summoned
        state=compare.LAST_STATE
        if state is not None:
            cids=state.cid
            for h in range(len(cids)):
                if int(cids[h])==cmap[FUSION] and int(state.zone[h])==core.Z_MZONE:
                    assert not bool(state.proc[h]),'Summon without a procedure became complete'
                    seen.add('extra summon')
                if int(cids[h])==cmap[FUSION] and int(state.zone[h])==core.Z_GRAVE:
                    seen.add('grave')
        for kind in ('activate','summon','to_ep','pass'):
            for i in indices:
                o=dec.options[i]
                if o[0]!=kind:continue
                if kind=='activate':
                    if o[1]==HOLE and 'extra summon' not in seen:continue
                    if o[1]==REBORN and 'grave' not in seen:continue
                return i
        return indices[0]
    result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',chooser=chooser,extra_capacity=15,
        given=([[STEIN,HOLE,REBORN,ELF]*10,[ELF]*40],[[FUSION],[]]))
    assert result['ok'],(seed,result.get('err'))
    assert seen=={'extra summon','grave'},(seed,seen)
    print(seed,'identical, Extra Deck → field → GY without a complete procedure',flush=True)
