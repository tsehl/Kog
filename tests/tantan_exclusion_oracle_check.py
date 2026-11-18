"""Oracle scenario: Tantan excludes its own name and Summons a Marmot."""
from pathlib import Path
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import compare
import core
import driver
from oracle_bank import build

TANTAN, MARMOT = 28118128, 75889523

db, bank, cmap = build([TANTAN, MARMOT])
step, legal = jax.jit(core.step), jax.jit(core.legal)
state = dict(set_tantan=False, flip_summon=False, selected=False,
             duplicate_tantan_in_deck=False, marmot_seen=False)


def observe_state():
    current = compare.LAST_STATE
    for slot in range(current.cid.shape[0]):
        code = int(bank.code[current.cid[slot]])
        zone = int(current.zone[slot])
        if code == MARMOT and int(current.ctrl[slot]) == 0 and zone == core.Z_MZONE:
            state['marmot_seen'] = True


def chooser(dec, indices):
    current = compare.LAST_STATE
    observe_state()

    def find(kind, code):
        return next((i for i in indices
                     if dec.options[i][0] == kind and dec.options[i][1] == code), None)

    # Set Tantan on turn 1. Do not assert availability during the early quick windows.
    if dec.player == 0 and int(current.turn) == 1 and not state['set_tantan']:
        choice = find('mset', TANTAN)
        if choice is not None:
            state['set_tantan'] = True
            return choice

    # Flip summon it on turn 3 (repos options are the oracle's flip-summon choices).
    if dec.player == 0 and int(current.turn) >= 3 and state['set_tantan'] and not state['flip_summon']:
        choice = find('repos', TANTAN)
        if choice is not None:
            state['flip_summon'] = True
            return choice

    if dec.options and dec.options[0][0] == 'pos' and state['flip_summon']:
        face_up = next((i for i in indices if dec.options[i] == ('pos', core.P_FU_ATK)), None)
        if face_up is not None:
            return face_up

    # The effect's deck selection must expose only Marmot candidates. There are
    # multiple copies, so compare.play will not auto-answer this decision.
    if dec.options and dec.options[0][0] == 'cards':
        is_tantan_resolution = (
            int(current.pg) == core.PG_RESOLVE
            and int(current.ch_src) >= 0
            and int(bank.code[current.cid[int(current.ch_src)]]) == TANTAN
        )
        if is_tantan_resolution:
            card_choices = [dec.options[i][1] for i in indices]
            assert card_choices, 'Tantan offers no selection candidate'
            assert all(cards and all(card[0] == MARMOT for card in cards)
                       for cards in card_choices), ('candidate other than Marmot', card_choices)
            tantan_left = sum(
                1 for slot in range(current.cid.shape[0])
                if int(current.zone[slot]) == core.Z_DECK
                and int(bank.code[current.cid[slot]]) == TANTAN
            )
            assert tantan_left > 0, 'no Tantan copy left in the Deck to choose from'
            state['duplicate_tantan_in_deck'] = True
            state['selected'] = True
            return indices[0]

    # Opponent and remaining actions stay passive; this also handles the
    # pre-main quick-effect windows without asserting their option contents.
    return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])


# The last element is the top of the deck: alternating puts Tantan in the opening
# hand and leaves several copies of each name to select.
decks = [[TANTAN, MARMOT] * 20, [MARMOT] * 40]
result = compare.play(db, bank, cmap, 28118128, step, legal, mode='tr',
                      given=(decks, [[], []]), chooser=chooser)
assert result['ok'], result.get('err')
coverage = result['coverage']
assert any(e['code'] == TANTAN and e['stage'] == 'activated' and e['count'] > 0
           for e in coverage), 'Tantan activation not covered'
assert any(e['code'] == TANTAN and e['stage'] == 'resolved' and e['count'] > 0
           for e in coverage), 'Tantan resolution not covered'
assert any(e['code'] == MARMOT and e['stage'] == 'summoned' and e['count'] > 0
           for e in coverage), 'Marmot Special Summon not observed'
assert all(state[k] for k in ('set_tantan', 'flip_summon', 'selected',
                               'duplicate_tantan_in_deck', 'marmot_seen')), state
print(f"Tantan/GetCode exclusion: activation, Marmot selection and Summon observed "
      f"({result['decisions']} decisions)", flush=True)
