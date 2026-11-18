"""Several effects at the same point: initial conditions, costs and SEGOC."""
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'gen', 'train')]
import cards as C
import core


class MultipleTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        event = C.ev(core.EV_SUMMON)
        optional = lambda **kw: C.trigger(event, optional=True, **kw)
        grave = [C.FILTER(0, C.F(me=core.Z_GRAVE)), C.REQ(0, 1)]
        effects = {
            1: [optional(), optional()],
            2: [optional(cost=[C.COND_SELF(prev=core.Z_MZONE | core.Z_SZONE)], cond=1),
                optional(cost=[C.COND_SELF(prev=core.Z_DECK)], cond=1)],
            3: [optional(opt=core.OPT_NAME), optional(opt=core.OPT_NAME)],
            4: [C.trigger(event), optional(target=grave)],
            5: [C.trigger(event), optional(cost=grave, cond=len(grave))],
            6: [optional(opt=core.OPT_NAME), optional(opt=core.OPT_NAMEK)],
        }
        codes = np.arange(1, 7)
        cls.bank = core.make_bank(codes, [1000]*6, [1000]*6, [4]*6,
            **C.assemble(codes, cards=effects, ritual={}, fusion={}, synchro={}, xyz={}, link={}))
        deck = jnp.zeros(40, jnp.int32)
        state = jax.jit(core.new_game)(cls.bank, deck, deck)
        cls.state = state._replace(zone=jnp.full(80, core.Z_DECK, jnp.int32).at[0].set(core.Z_MZONE),
            pos=state.pos.at[0].set(core.P_FU_ATK), seq=state.seq.at[0].set(0),
            pzone=state.pzone.at[0].set(core.Z_DECK), tp=jnp.int32(0))
        cls.evaluate = staticmethod(jax.jit(core.trig_eval))
        cls.collect = staticmethod(jax.jit(lambda bank, s: core.trig_point(bank, core.open_quick(s, 0))))
        cls.push = staticmethod(jax.jit(core.push_link))
        cls.consume = staticmethod(jax.jit(core.use_count))

    def pending(self, code):
        s = self.state._replace(cid=self.state.cid.at[0].set(code - 1))
        s = core.raise_ev(s, core.EV_SUMMON, jnp.arange(80) == 0, 0)
        return self.collect(self.bank, s)

    def chain(self, s, effect):
        # State after paying cost/target: push_link must remove only the chosen effect.
        s = s._replace(ch_src=jnp.int32(0), ch_eff=jnp.int32(effect), ch_p=jnp.int32(0))
        s = self.consume(self.bank, s, jnp.int32(0), jnp.int32(effect), jnp.int32(0))
        return self.push(self.bank, s)

    def test_two_effects_survive_and_can_chain_in_either_order(self):
        for first in (0, 1):
            with self.subTest(first=first):
                s = self.pending(1)
                self.assertEqual(int(s.tg_e[0]), 3)
                self.assertEqual(int(s.tst), -1)
                s = self.evaluate(self.bank, s)
                self.assertEqual((int(s.wk), int(s.tst)), (core.W_TRIG, 2))
                s = self.evaluate(self.bank, self.chain(s, first))
                self.assertEqual(int(s.tg_e[0]), 1 << (1-first))
                self.assertEqual(int(s.tst), 2)
                s = self.evaluate(self.bank, self.chain(s, 1-first))
                self.assertFalse(bool(s.tg.any()))
                self.assertEqual(int(s.wk), core.W_QUICK)
                self.assertEqual(int(s.n_chain), 2)

    def test_first_false_condition_does_not_mask_second(self):
        s = self.evaluate(self.bank, self.pending(2))
        self.assertEqual(int(s.tg_e[0]), 2)
        self.assertEqual(int(s.tst), 2)

    def test_shared_limit_removes_other_effect_after_activation(self):
        s = self.evaluate(self.bank, self.pending(3))
        s = self.evaluate(self.bank, self.chain(s, 0))
        self.assertFalse(bool(s.tg.any()))
        self.assertEqual(int(s.n_chain), 1)

    def test_distinct_limits_allow_both_effects(self):
        s = self.evaluate(self.bank, self.pending(6))
        s = self.evaluate(self.bank, self.chain(s, 0))
        self.assertEqual(int(s.tg_e[0]), 2)
        self.assertEqual(int(s.wk), core.W_TRIG)

    def test_optional_target_is_checked_after_mandatory_cost(self):
        s = self.evaluate(self.bank, self.pending(4))
        self.assertEqual(int(s.tst), 0)
        # Another monster is sent to the GY during the mandatory cost.
        s = s._replace(zone=s.zone.at[1].set(core.Z_GRAVE))
        s = self.evaluate(self.bank, self.chain(s, 0))
        self.assertEqual(int(s.tst), 2)
        self.assertEqual(int(s.tg_e[0]), 2)

    def test_false_initial_condition_cannot_be_enabled_by_mandatory_cost(self):
        s = self.evaluate(self.bank, self.pending(5))
        self.assertEqual(int(s.tst), 0)
        self.assertEqual(int(s.tg_e[0]), 1)
        s = s._replace(zone=s.zone.at[1].set(core.Z_GRAVE))
        s = self.evaluate(self.bank, self.chain(s, 0))
        self.assertFalse(bool(s.tg.any()))
        self.assertEqual(int(s.wk), core.W_QUICK)

    def test_opponent_mandatory_stage_is_not_skipped(self):
        s = self.pending(4)
        s = s._replace(ctrl=s.ctrl.at[0].set(1))
        s = self.evaluate(self.bank, s)
        self.assertEqual((int(s.tst), int(s.dec_player)), (1, 1))

    def test_declining_turn_player_optionals_keeps_opponents_effects(self):
        s = self.state._replace(zone=self.state.zone.at[40].set(core.Z_MZONE),
                                pos=self.state.pos.at[40].set(core.P_FU_ATK))
        s = core.raise_ev(s, core.EV_SUMMON, (jnp.arange(80) == 0) | (jnp.arange(80) == 40), 0)
        s = self.evaluate(self.bank, self.collect(self.bank, s))
        self.assertEqual(int(s.tst), 2)
        s = self.evaluate(self.bank, s._replace(tst=jnp.int32(3), dec=jnp.int32(core.D_TRIG)))
        self.assertEqual((int(s.tst), int(s.dec_player)), (3, 1))
        self.assertEqual(int(s.tg_e[0]), 0)
        self.assertEqual(int(s.tg_e[40]), 3)

    def test_compaction_keeps_condition_cost_boundary(self):
        # FILTER/REQ merge within each prefix, never across it.
        prog = np.zeros((1, core.E, 3, core.P, 4), np.int32)
        prog[0, 0, core.PG_COST, :3] = [C.FILTER(0, 0), C.REQ(0, 1), C.REQ(0, 2)]
        prefix = np.zeros((1, core.E), np.int32)
        prefix[0, 0] = 2
        compact, lengths = core.test_prog(prog, 1, prefix)
        self.assertEqual(int(lengths[0, 0]), 1)
        self.assertEqual(int(compact[0, 0, 1, 0]), core.OP_REQ)
        self.assertEqual((int(compact[0, 0, 0, 3]) >> 4) & 255, 1)

    def test_old_real_bank_loads_with_full_condition_metadata(self):
        import realcards
        bank, *_ = realcards.load()
        self.assertEqual(bank.econd.shape, bank.ekind.shape)

    def test_generator_can_emit_multiple_triggers_with_condition_prefixes(self):
        import cardgen
        found = 0
        for seed in range(4):
            specs = cardgen.Gen(seed).make_set(80, 10)
            for card in specs:
                triggers = [effect for effect in card['effects'] if effect['kind'] == 'trigger']
                if len(triggers) > 1:
                    found += 1
                    for effect in triggers:
                        lowered = cardgen.bricks(effect)
                        self.assertEqual(lowered[7]['cond'], len(cardgen.cond_bricks(effect['conds'])))
        self.assertGreater(found, 0)


if __name__ == '__main__':
    unittest.main()
