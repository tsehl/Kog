"""Private optional triggers join QuickEffect after the public SEGOC."""
from pathlib import Path
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine')]
import cards as C
import core


class PrivateHandTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        event = C.ev(core.EV_TO_GRAVE)
        effects = {1: [C.trigger(event, optional=True, rng=core.Z_HAND)],
                   2: [C.trigger(event, optional=True, rng=core.Z_MZONE)],
                   3: [C.act()], 4: [C.act()],
                   5: [C.trigger(event, optional=True, rng=core.Z_HAND,
                       cost=[C.FILTER(0, C.F(me=core.Z_GRAVE)), C.REQ(0, 1)], cond=2)]}
        codes = np.arange(1, 6)
        cls.bank = core.make_bank(codes, [1000]*5, [1000]*5, [4]*5,
            ctype=[33, 33, core.TYPE_SPELL|core.TYPE_QUICKPLAY, core.TYPE_TRAP|core.TYPE_COUNTER, 33],
            **C.assemble(codes, cards=effects, ritual={}, fusion={}, synchro={}, xyz={}, link={}))
        deck = jnp.zeros(40, jnp.int32)
        s = jax.jit(core.new_game)(cls.bank, deck, deck)
        cls.state = s._replace(zone=jnp.full(80, core.Z_DECK, jnp.int32).at[0].set(core.Z_HAND),
                               pos=s.pos.at[0].set(core.P_HAND), tp=jnp.int32(0))
        cls.evaluate = staticmethod(jax.jit(core.trig_eval))
        cls.candidates = staticmethod(jax.jit(core.act_cand))
        cls.push = staticmethod(jax.jit(core.push_link))
        cls.pass_ = staticmethod(jax.jit(core.pass_priority))

    def event(self, *, public=False, hand_limit=False):
        s = self.state
        if public:
            s = core.with_cid(self.bank, s, s.cid.at[40].set(1))._replace(zone=s.zone.at[40].set(core.Z_MZONE),
                                                                          pos=s.pos.at[40].set(core.P_FU_ATK))
        s = core.raise_ev(s, core.EV_TO_GRAVE, jnp.arange(80) == 1, 0)
        s = core.open_quick(s, 0, core.R_HAND if hand_limit else core.R_IDLE, chained=False)
        return self.evaluate(self.bank, core.trig_point(self.bank, s))

    def test_private_trigger_enters_response_window_not_public_segoc(self):
        s = self.event()
        self.assertEqual(int(s.wk), core.W_QUICK)
        self.assertEqual(int(s.tg_e[0]), 1 << core.E)
        self.assertTrue(bool(self.candidates(self.bank, s)[0, 0]))
        self.assertFalse(bool(self.candidates(self.bank, s._replace(dec_player=jnp.int32(1)))[0, 0]))

    def test_hand_limit_does_not_open_a_private_chain(self):
        s = self.event(hand_limit=True)
        self.assertEqual(int(s.wk), core.W_QUICK)
        self.assertFalse(bool(self.candidates(self.bank, s)[0, 0]))

    def test_public_trigger_opens_chain_then_private_trigger_can_respond(self):
        s = self.event(public=True, hand_limit=True)
        self.assertEqual((int(s.wk), int(s.dec_player)), (core.W_TRIG, 1))
        self.assertFalse(bool(self.candidates(self.bank, s)[0, 0]))
        s = self.push(self.bank, s._replace(ch_src=jnp.int32(40), ch_eff=jnp.int32(0), ch_p=jnp.int32(1)))
        s = self.evaluate(self.bank, s)
        self.assertEqual((int(s.wk), int(s.dec_player)), (core.W_QUICK, 0))
        self.assertTrue(bool(self.candidates(self.bank, s)[0, 0]))
        s = self.push(self.bank, s._replace(ch_src=jnp.int32(0), ch_eff=jnp.int32(0), ch_p=jnp.int32(0)))
        self.assertEqual(int(s.tg_e[0]), 0)
        self.assertFalse(bool(self.candidates(self.bank, s._replace(dec_player=jnp.int32(0)))[0, 0]))

    def test_private_trigger_responds_to_speed_two_but_not_counter_trap(self):
        s = self.event()._replace(n_chain=jnp.int32(1), lk_src=self.state.lk_src.at[0].set(40),
                                 lk_eff=self.state.lk_eff.at[0].set(0))
        for cid, expected in ((2, True), (3, False)):
            self.assertEqual(bool(self.candidates(self.bank, core.with_cid(self.bank, s, s.cid.at[40].set(cid)))[0, 0]), expected)

    def test_false_initial_condition_and_departed_handler_are_excluded(self):
        s = core.with_cid(self.bank, self.state, self.state.cid.at[0].set(4))
        s = core.raise_ev(s, core.EV_TO_GRAVE, jnp.arange(80) == 1, 0)
        s = self.evaluate(self.bank, core.trig_point(self.bank, core.open_quick(s, 0)))
        self.assertEqual(int(s.tg_e[0]), 0)
        s = self.event()._replace(zone=self.state.zone.at[0].set(core.Z_GRAVE))
        self.assertFalse(bool(self.candidates(self.bank, s)[0, 0]))

    def test_unselected_private_trigger_expires_when_window_closes(self):
        s = self.event()._replace(cw_passed=jnp.bool_(True))
        s = self.pass_(self.bank, s)
        self.assertFalse(bool(s.tg.any()))
        self.assertEqual(int(s.tg_e[0]), 0)


if __name__ == '__main__':
    unittest.main()
