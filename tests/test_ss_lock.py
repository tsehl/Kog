"""Turn Special Summon restrictions (OP_SSLOCK) and Special Summons made this turn (ss_t, ST_SST / ST_SSXT)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine')]
import core, cards as C

ALLZ = core.Z_DECK | core.Z_HAND | core.Z_MZONE | core.Z_SZONE | core.Z_GRAVE | core.Z_REMOVED | core.Z_EXTRA
SYN = core.TYPE_MONSTER | core.TYPE_SYNCHRO
XYZ = core.TYPE_MONSTER | core.TYPE_XYZ
NORMAL = core.TYPE_MONSTER | core.TYPE_NORMAL


class SSLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # card 1: resolution = "no more Special Summons from the Extra Deck, except Synchros"
        lock = C.SSLOCK(C.F(me=core.Z_EXTRA), unless=C.F(me=ALLZ, op=ALLZ, typ=core.TYPE_SYNCHRO))
        count = [C.FILTER_NOT(0, C.F(me=ALLZ, op=ALLZ, typ=core.TYPE_SYNCHRO)),
                 C.FILTER(1, C.F(me=ALLZ, op=ALLZ, st=core.ST_SSXT), within=0)]
        eff = [(core.K_IGNITION, 0, 0, core.Z_HAND, [], count, [lock])]
        codes = [1, 2, 3, 4]
        cls.bank = core.make_bank(np.array(codes), [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 6, 4],
                                  ctype=[NORMAL, NORMAL, SYN, XYZ], **C.assemble(codes, cards={1: eff}))
        d = jnp.ones(40, jnp.int32).at[0].set(0).at[1].set(2).at[2].set(3)
        s = jax.jit(core.new_game)(cls.bank, d, d)
        zone = jnp.full(80, core.Z_DECK, jnp.int32).at[0].set(core.Z_HAND).at[1].set(core.Z_EXTRA).at[2].set(core.Z_EXTRA)
        cls.state = s._replace(zone=zone, ctrl=jnp.arange(80, dtype=jnp.int32) // 40)
        cls.refresh = staticmethod(jax.jit(core.refresh))
        cls.mask = staticmethod(jax.jit(core.filter_mask, static_argnames=("bdes",)))

    def row(self):  # row written by the execution of SSLOCK (card 0, effect 0): filter indices of the bank
        op, a, b, c = map(int, self.bank.prog[0, 0, core.PG_RESOLVE, 0])
        self.assertEqual(op, core.OP_SSLOCK)
        return jnp.array([1, 0, b, c, a], jnp.int32)

    def test_lock_with_exception(self):
        s = self.state._replace(ssl=self.state.ssl.at[0].set(self.row()))
        fl = np.asarray(self.refresh(self.bank, s).fl_c & core.FL_NO_SS) != 0
        self.assertFalse(fl[1])   # Synchro from the Extra Deck: exception
        self.assertTrue(fl[2])    # Xyz from the Extra Deck: forbidden
        self.assertFalse(fl[0])   # hand: outside the target
        self.assertFalse(fl[42])  # opponent: not affected
        free = np.asarray(self.refresh(self.bank, self.state).fl_c & core.FL_NO_SS) != 0
        self.assertFalse(free[2])

    def test_summoned_this_turn_virtual_status(self):
        s = self.state
        info = core.summon_info(core.SUMMON_SPECIAL, core.Z_EXTRA, 0)
        s = s._replace(ss_t=s.ss_t.at[2].set(True).at[1].set(True), suminfo=s.suminfo.at[2].set(info).at[1].set(info),
                       zone=s.zone.at[2].set(core.Z_GRAVE))
        f = jnp.array(C.F(me=ALLZ, op=ALLZ, st=core.ST_SSXT), jnp.int32)
        np.testing.assert_array_equal(np.nonzero(np.asarray(self.mask(self.bank, s, f, 0, 0, bdes=False)))[0], [1, 2])
        # the other player Summoned nothing
        self.assertFalse(np.asarray(self.mask(self.bank, s, f, 1, 0, bdes=False)).any())
        hand = core.summon_info(core.SUMMON_SPECIAL, core.Z_HAND, 0)
        s2 = s._replace(suminfo=s.suminfo.at[2].set(hand))
        np.testing.assert_array_equal(np.nonzero(np.asarray(self.mask(self.bank, s2, f, 0, 0, bdes=False)))[0], [1])
        g = jnp.array(C.F(me=ALLZ, op=ALLZ, st=core.ST_SST), jnp.int32)
        np.testing.assert_array_equal(np.nonzero(np.asarray(self.mask(self.bank, s2, g, 0, 0, bdes=False)))[0], [1, 2])

    def test_activation_counter_with_inverted_filter(self):
        # cost: (non-Synchro cards) ∩ (Summoned from the Extra Deck this turn); REQ absent, so the register is read
        s = self.state._replace(dec=jnp.int32(core.D_RUN), ch_src=jnp.int32(0), ch_eff=jnp.int32(0),
                                ch_p=jnp.int32(0), pg=jnp.int32(core.PG_TARGET), pc=jnp.int32(0))
        info = core.summon_info(core.SUMMON_SPECIAL, core.Z_EXTRA, 0)
        s = s._replace(ss_t=s.ss_t.at[2].set(True).at[1].set(True), suminfo=s.suminfo.at[2].set(info).at[1].set(info))
        ex = jax.jit(core.exec_one)
        s = ex(self.bank, ex(self.bank, s))
        np.testing.assert_array_equal(np.nonzero(np.asarray(s.reg[1]))[0], [2])

    def test_turn_reset(self):
        s = self.state._replace(ssl=self.state.ssl.at[0].set(self.row()),
                                ss_t=self.state.ss_t.at[1].set(True))
        s = core.start_turn(s)
        self.assertFalse(bool(s.ssl.any()) or bool(s.ss_t.any()))


if __name__ == "__main__":
    unittest.main()
