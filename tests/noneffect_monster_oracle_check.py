"""Native oracle probe for Vishuda: face-up non-effect monster versus two negative gates."""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('tests', 'engine', 'translate', 'oracle')]
import jax
import compare
import core
import driver
import translate
from oracle_bank import build

VISHUDA, MOKEY, EFFECT = 23431858, 12482652, 600930157
step, legal = jax.jit(core.step), jax.jit(core.legal)

# Mokey Mokey (MOKEY) is the real normal monster. EFFECT has TYPE_MONSTER |
# TYPE_EFFECT, but deliberately no registered effects; it isolates card type.
driver.TEST_CARDS[EFFECT] = ((0, 0, 33, 1000, 1000, 4, 1, 1), 'Effect monster type probe')


def check(profile, seed):
    """profile is 'normal', 'facedown', or 'effect'."""
    with tempfile.TemporaryDirectory() as folder:
        old_official = translate.OFFICIAL
        shutil.copy(Path(old_official, f'c{VISHUDA}.lua'), folder)
        Path(folder, f'c{EFFECT}.lua').write_text('local s,id=GetID()\nfunction s.initial_effect(c) end\n')
        translate.OFFICIAL = folder
        driver.SCRIPTS.insert(0, folder)
        try:
            db, bank, cmap = build([VISHUDA, MOKEY, EFFECT])
            observed = {'own_ready': False, 'opponent_target': False,
                        'effect2_offered': False, 'bounced': False,
                        'mzone_before': None, 'hand_before': None,
                        'own_setup_done': False, 'opponent_setup_done': False,
                        'negative_idle_witness': False}
            ally = MOKEY if profile in ('normal', 'facedown') else EFFECT

            def zone_has(s, code, ctrl, zone):
                return any(int(bank.code[s.cid[i]]) == code and int(s.ctrl[i]) == ctrl
                           and int(s.zone[i]) == zone for i in range(s.cid.shape[0]))

            def chooser(dec, indices):
                s = compare.LAST_STATE
                if observed['mzone_before'] is not None:
                    now_mz = sum(int(bank.code[s.cid[i]]) == MOKEY and int(s.ctrl[i]) == 1
                                 and int(s.zone[i]) == core.Z_MZONE for i in range(s.cid.shape[0]))
                    now_hand = sum(int(bank.code[s.cid[i]]) == MOKEY and int(s.ctrl[i]) == 1
                                   and int(s.zone[i]) == core.Z_HAND for i in range(s.cid.shape[0]))
                    if now_mz < observed['mzone_before'] and now_hand > observed['hand_before']:
                        observed['bounced'] = True
                own_ready = zone_has(s, ally, 0, core.Z_MZONE)
                opp_ready = zone_has(s, MOKEY, 1, core.Z_MZONE)
                v_in_hand = zone_has(s, VISHUDA, 0, core.Z_HAND)
                if own_ready and opp_ready and v_in_hand:
                    observed['own_ready'] = True
                    observed['opponent_target'] = True
                    # Effect 2 is the hand/grave bounce. Effect 1 is the
                    # special summon; choosing by index prevents it masking
                    # a missing bounce option.
                    bounce_desc = (VISHUDA << 20) | 1  # aux.Stringid(id, effect-2 index)
                    bounce = next((i for i in indices if dec.options[i][0] in ('chain', 'activate')
                                   and dec.options[i][1] == VISHUDA
                                   and len(dec.options[i]) > 5
                                   and dec.options[i][5] == bounce_desc), None)
                    observed['effect2_offered'] |= bounce is not None
                    if profile == 'normal' and bounce is not None:
                        observed['mzone_before'] = sum(
                            int(bank.code[s.cid[j]]) == MOKEY and int(s.ctrl[j]) == 1
                            and int(s.zone[j]) == core.Z_MZONE for j in range(s.cid.shape[0]))
                        observed['hand_before'] = sum(
                            int(bank.code[s.cid[j]]) == MOKEY and int(s.ctrl[j]) == 1
                            and int(s.zone[j]) == core.Z_HAND for j in range(s.cid.shape[0]))
                        return bounce
                    if profile != 'normal' and dec.player == 0 and int(s.dec) == core.D_IDLE:
                        observed['negative_idle_witness'] = True
                        assert bounce is None, (profile, dec.options)

                if dec.player == 0 and not observed['own_setup_done']:
                    if profile == 'facedown':
                        choice = next((i for i in indices if dec.options[i][:2] == ('mset', MOKEY)), None)
                        if choice is not None:
                            observed['own_setup_done'] = True
                            return choice
                    choice = next((i for i in indices if dec.options[i][:2] == ('summon', ally)), None)
                    if choice is not None:
                        observed['own_setup_done'] = True
                        return choice
                if dec.player == 1 and not observed['opponent_setup_done']:
                    choice = next((i for i in indices if dec.options[i][:2] == ('summon', MOKEY)), None)
                    if choice is not None:
                        observed['opponent_setup_done'] = True
                        return choice
                # The target picker is handled by compare.play's default or
                # chooser fallback after Vishuda has been activated.
                return next((i for i in indices if dec.options[i][0] in ('pass', 'to_ep')), indices[0])

            # The supplied deck order fixes the opening setup. The chooser
            # records readiness before activating the bounce effect.
            p0 = [MOKEY] * 35 + [VISHUDA, ally] + [MOKEY] * 3
            p1 = [MOKEY] * 40
            result = compare.play(db, bank, cmap, seed, step, legal, mode='tr', chooser=chooser,
                                  given=([p0, p1], [[], []]))
        finally:
            translate.OFFICIAL = old_official
            driver.SCRIPTS.remove(folder)

    assert result['ok'], result.get('err')
    assert observed['own_ready'] and observed['opponent_target'], observed
    if profile == 'normal':
        assert observed['effect2_offered'], observed
        assert observed['bounced'], observed
        assert any(x['code'] == VISHUDA and x['stage'] == 'resolved' and x['count']
                   for x in result['coverage']), result['coverage']
    else:
        assert observed['negative_idle_witness'], observed
        assert not observed['effect2_offered'], observed
    print(f"Vishuda {profile}: {result['decisions']} identical decisions; {observed}", flush=True)


if __name__ == '__main__':
    # Native witnesses:
    # normal face-up => effect 2 offered and resolves; face-down normal => absent;
    # face-up effect monster => absent (and also blocks effect 1 summon).
    check('normal', 9111)
    check('facedown', 9112)
    check('effect', 9113)
