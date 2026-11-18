"""Official Mispolymerization trigger: Fusion Summon then return Fusion to Extra."""
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
from oracle_bank import build

TRAP = 58392024
TWIN_THUNDER_DRAGON = 54752875
THUNDER_DRAGON = 31786629
POLYMERIZATION = 24094653
NORMAL = 44287299

db, bank, cmap = build([TRAP, TWIN_THUNDER_DRAGON, THUNDER_DRAGON,
                        POLYMERIZATION, NORMAL])
assert db.rows[TRAP][3] & core.TYPE_TRAP
assert db.rows[TWIN_THUNDER_DRAGON][3] & core.TYPE_FUSION

state = dict(trap_set=False, normal_summoned=False, poly_activated=False,
             trap_chain_selected=False, fusion_material_prompts=0,
             native_moves=[], fusion_mzone_mv=None, returned_snapshot=None)
NATIVE_DUEL = driver.Duel


class WatchedDuel(NATIVE_DUEL):
    def _handle(self, msg):
        reader = driver.Reader(msg)
        kind = reader.take('B')
        if kind == driver.K['MSG_MOVE']:
            code = reader.take('I')
            prev_ctrl, prev_zone, prev_seq, prev_pos = reader.take('BBII')
            ctrl, zone, seq, pos = reader.take('BBII')
            state['native_moves'].append((code, prev_ctrl, prev_zone, prev_seq, prev_pos,
                                          ctrl, zone, seq, pos))
        return super()._handle(msg)


def chooser(dec, indices):
    current = compare.LAST_STATE

    # Capture the JAX state once the Trap's native return has completed.
    twin_slots = [i for i in range(current.cid.shape[0])
                  if int(bank.code[current.cid[i]]) == TWIN_THUNDER_DRAGON
                  and int(current.ctrl[i]) == 1]
    if twin_slots:
        i = twin_slots[0]
        if int(current.zone[i]) == core.Z_MZONE:
            state['fusion_mzone_mv'] = int(current.mv[i])
        elif (int(current.zone[i]) == core.Z_EXTRA and state['fusion_mzone_mv'] is not None
              and int(current.mv[i]) > state['fusion_mzone_mv']):
            normal_mzone = any(int(bank.code[current.cid[j]]) == NORMAL
                               and int(current.ctrl[j]) == 1
                               and int(current.zone[j]) == core.Z_MZONE
                               for j in range(current.cid.shape[0]))
            state['returned_snapshot'] = (int(current.zone[i]), int(current.pos[i]),
                                          int(current.mv[i]), normal_mzone)

    def find(kind, code):
        return next((i for i in indices if dec.options[i][0] == kind
                     and dec.options[i][1] == code), None)

    turn = int(current.turn)
    choice = None

    # P0 sets the Trap on turn one.
    if dec.player == 0 and turn == 1 and not state['trap_set']:
        choice = find('sset', TRAP)
        if choice is not None:
            state['trap_set'] = True

    # P1 puts a Normal Monster on the field first, to prove the Trap's
    # operation filter does not send every monster on the field to the Extra.
    if dec.player == 1 and turn == 2 and not state['normal_summoned']:
        choice = find('summon', NORMAL)
        if choice is not None:
            state['normal_summoned'] = True

    if (dec.player == 1 and turn >= 2 and state['normal_summoned']
            and choice is None and not state['poly_activated']):
        choice = find('activate', POLYMERIZATION)
        if choice is not None:
            state['poly_activated'] = True

    if dec.player == 0:
        for i in indices:
            if dec.options[i][0] == 'chain' and dec.options[i][1] == TRAP:
                choice = i
                state['trap_chain_selected'] = True
                break

    is_poly_resolution_choice = (
        dec.options[0][0] == 'cards' and int(current.pg) == core.PG_RESOLVE
        and int(current.ch_src) >= 0
        and int(bank.code[current.cid[current.ch_src]]) == POLYMERIZATION)
    if is_poly_resolution_choice:
        groups = [dec.options[i][1] for i in indices]
        assert groups, ('Fusion material choice has no groups', dec.options)
        assert all(len(group) == 2 and all(card[0] == THUNDER_DRAGON
                   and card[1] == 1 and card[2] == 'hand' for card in group)
                   for group in groups), ('unexpected Fusion material group', groups)
        assert any(int(bank.code[current.cid[j]]) == NORMAL and int(current.ctrl[j]) == 1
                   and int(current.zone[j]) == core.Z_MZONE
                   for j in range(current.cid.shape[0])), ('Normal Monster not on field beside Fusion',)
        state['fusion_material_prompts'] += 1
        choice = indices[0]

    if choice is not None:
        return choice
    return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])


# Pseudo-shuffle adds the final card as the deck top. The opening five for P0
# contain the Trap; P1 opens with Polymerization, both Thunder Dragons, and a
# Level 4 Normal Monster. Only P1 has a Twin-Headed Thunder Dragon in the Extra.
deck0 = [*([NORMAL] * 35), *([NORMAL] * 4), TRAP]
deck1 = [*([NORMAL] * 35), NORMAL, NORMAL, THUNDER_DRAGON, THUNDER_DRAGON, POLYMERIZATION]
assert len(deck0) == len(deck1) == 40

driver.Duel = WatchedDuel
try:
    result = compare.play(db, bank, cmap, 5839, jax.jit(core.step), jax.jit(core.legal),
                          mode='tr', chooser=chooser,
                          given=([deck0, deck1], [[], [TWIN_THUNDER_DRAGON]]),
                          extra_capacity=1)
finally:
    driver.Duel = NATIVE_DUEL

assert result['ok'], result.get('err')
assert (state['trap_set'] and state['normal_summoned'] and state['poly_activated']
        and state['trap_chain_selected']), state
assert state['returned_snapshot'] is not None, state
assert state['returned_snapshot'][0] == core.Z_EXTRA, state
assert state['returned_snapshot'][1] == core.P_FD_DEF, state
assert state['returned_snapshot'][3], ('Normal Monster not retained at native return', state)
for code, stage in ((TRAP, 'activated'), (TRAP, 'resolved'),
                    (POLYMERIZATION, 'resolved'), (TWIN_THUNDER_DRAGON, 'summoned')):
    assert any(e['code'] == code and e['stage'] == stage and e['count']
               for e in result['coverage']), (code, stage, result['coverage'])

fusion_moves = [move for move in state['native_moves'] if move[0] == TWIN_THUNDER_DRAGON]
assert any(m[2] == driver.K['LOCATION_EXTRA'] and m[6] == driver.K['LOCATION_MZONE']
           for m in fusion_moves), ('Fusion never entered MZONE', fusion_moves)
back_to_extra = [m for m in fusion_moves
                 if m[2] == driver.K['LOCATION_MZONE'] and m[6] == driver.K['LOCATION_EXTRA']]
assert back_to_extra, ('native MZONE-to-Extra move missing', fusion_moves)
assert any(m[8] == driver.K['POS_FACEDOWN_DEFENSE'] for m in back_to_extra), back_to_extra
material_moves = [m for m in state['native_moves'] if m[0] == THUNDER_DRAGON
                  and m[2] == driver.K['LOCATION_HAND']
                  and m[6] == driver.K['LOCATION_GRAVE']]
assert len(material_moves) == 2, ('expected both Thunder Dragon Fusion materials to reach GY', material_moves)
assert any(m[0] == NORMAL and m[2] == driver.K['LOCATION_HAND']
           and m[6] == driver.K['LOCATION_MZONE'] for m in state['native_moves']), (
               'Normal Monster was not actually summoned', state['native_moves'])
assert not any(m[0] == NORMAL and m[2] == driver.K['LOCATION_MZONE']
               and m[6] == driver.K['LOCATION_EXTRA'] for m in state['native_moves']), (
                   'non-Fusion Normal Monster was returned to Extra', state['native_moves'])
print('Mispolymerization: Trap set, Twin-Headed Thunder Dragon Fusion Summoned, Trap resolved, '
      'Fusion returned face-down to Extra; Normal Monster stayed out;', result['decisions'],
      'decisions identical', flush=True)
