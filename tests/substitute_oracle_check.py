"""Real substitutes against the oracle: a substitute's limit, filters, face-down cards."""
import jax
from oracle_bank import build
import core,compare

POLY, SUB, A, B, BLUE = 24094653,99426834,34460851,44287299,89631139
FUS, ULT, GENERIC = 45231177,23995346,3544583
cases=[('missing name',FUS,[B,SUB]*17+[POLY]+[B,SUB,POLY,B,POLY],True),
       ('two replacements forbidden',FUS,[SUB]*34+[POLY]*6,False),
       ('three materials including a substitute',ULT,[SUB]*31+[POLY]*6+[BLUE]*2+[SUB],True),
       ('generic filter not substitutable',GENERIC,[SUB]*34+[POLY]*6,False),
       ('permission disabled',990000008,[B,SUB]*17+[POLY]+[B,SUB,POLY,B,POLY],False),
       ('IsCode filter not substitutable',990000007,[B,SUB]*17+[POLY]+[B,SUB,POLY,B,POLY],False),
       ('other named slot substitutable',990000007,[A,SUB]*17+[POLY]+[A,SUB,POLY,A,POLY],True),
       ('two substitute cards, a single replacement',990000009,[SUB,POLY]*20,True)]
codes=[A,B,BLUE,SUB,POLY,FUS,ULT,GENERIC,990000007,990000008,990000009]
db,bank,cmap=build(codes,fixtures=(990000007,990000008,990000009))
assert bank.submat.shape[1]==1
for name,fusion,deck,positive in cases:
    count=field_picks=0
    for seed in (200,201):
        set_players=set()
        def chooser(dec,indices):
            for i in indices:
                o=dec.options[i]
                if o[0]=='mset' and o[1]==SUB and dec.player not in set_players:
                    set_players.add(dec.player);return i
            for i in indices:
                o=dec.options[i]
                if o[0]=='activate' and o[1]==POLY:return i
            for i in indices:
                o=dec.options[i]
                if o[0]=='pick' and o[1][0]==SUB and o[1][2]=='mzone':return i
            return indices[0]
        # Policy fully determined by the oracle options.
        r=compare.play(db,bank,cmap,seed,jax.jit(core.step),jax.jit(core.legal),mode='tr',
                       given=([deck,deck],[[fusion],[fusion]]),chooser=chooser)
        assert r['ok'],(name,seed,r.get('err'))
        count+=sum(e['count'] for e in r['coverage'] if e['code']==fusion and e['stage']=='summoned')
        field_picks+=sum(t['option'][0]=='pick' and t['option'][1][0]==SUB and t['option'][1][2]=='mzone' for t in r['trace'])
    assert (count>0)==positive,(name,count)
    if name=='missing name': assert field_picks>0,'substitute on the field not exercised'
    print(f'{name}: 2/2 identical, {count} Fusions, {field_picks} substitute choices on the field',flush=True)
