"""Return to the top of the Deck: order, owner, redirections and Lua rejections."""
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'translate')]
import core
import translate


class DeckTopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = core.make_bank(np.array([1, 2]), [1000, 1000], [1000, 1000], [4, 4],
                                  ctype=[core.TYPE_MONSTER | core.TYPE_NORMAL,
                                         core.TYPE_MONSTER | core.TYPE_FUSION])
        deck = jnp.zeros(40, jnp.int32)
        cls.state = jax.jit(core.new_game)(cls.bank, deck, deck)
        cls.send = staticmethod(jax.jit(lambda s, mask: core.group_to_owner_zone(
            cls.bank, s, mask, core.Z_DECK, core.REASON_EFFECT)))

    def test_field_order_then_draw_and_owner(self):
        s = self.state
        s = core.compact(s, jnp.arange(s.cid.size) < 3)
        # Cards of different owners under the same control: each one returns to its own
        # Deck; the two cards of p0 follow their field seq.
        s = s._replace(zone=s.zone.at[:3].set(core.Z_MZONE),
                       seq=s.seq.at[:3].set(jnp.array([2, 0, 1])),
                       ctrl=s.ctrl.at[:3].set(0), owner=s.owner.at[:3].set(jnp.array([0, 0, 1])))
        mask = jnp.arange(s.cid.size) < 3
        out = self.send(s, mask)
        self.assertEqual(int(out.ctrl[2]), 1)
        self.assertEqual(int(out.seq[0]), int(out.seq[1]) + 1)
        self.assertTrue(bool(jnp.all(out.zone[:3] == core.Z_DECK)))
        self.assertTrue(bool(jnp.all(out.mv[:3] == s.clk + 1)))
        self.assertTrue(bool(jnp.all(out.reason[:3] == core.REASON_EFFECT)))
        for player in (0, 1):
            seqs = np.asarray(out.seq)[np.asarray((out.zone == core.Z_DECK) & (out.ctrl == player))]
            np.testing.assert_array_equal(np.sort(seqs), np.arange(len(seqs)))
        drawn = jax.jit(core.draw)(out, 0)
        self.assertEqual(int(drawn.zone[0]), core.Z_HAND)
        self.assertEqual(int(drawn.zone[1]), core.Z_DECK)

    def test_extra_redirection_preserves_faceup_pendulum_order(self):
        s = self.state
        s = core.compact(s, jnp.arange(s.cid.size) < 3)
        s = s._replace(cid=s.cid.at[0].set(1), zone=s.zone.at[0].set(core.Z_GRAVE)
                       .at[1:3].set(core.Z_EXTRA),
                       pos=s.pos.at[1].set(core.P_FD_DEF).at[2].set(core.P_FU_DEF),
                       seq=s.seq.at[0].set(0).at[1].set(0).at[2].set(1))
        out = self.send(s, jnp.arange(s.cid.size) == 0)
        self.assertEqual((int(out.zone[0]), int(out.pos[0]), int(out.seq[0])),
                         (core.Z_EXTRA, core.P_FD_DEF, 1))
        self.assertEqual(int(out.seq[2]), 2)

    def test_already_in_deck_extra_and_leave_confirmed_stay(self):
        s = self.state
        s = core.compact(s, (jnp.arange(s.cid.size) == 1) | (jnp.arange(s.cid.size) == 2))
        s = s._replace(cid=s.cid.at[1].set(1), zone=s.zone.at[1].set(core.Z_EXTRA)
                       .at[2].set(core.Z_SZONE), seq=s.seq.at[1:3].set(0),
                       st=s.st.at[2].set(core.ST_LEAVE))
        out = self.send(s, jnp.arange(s.cid.size) < 3)
        for name in ('zone', 'seq', 'mv', 'st'):
            np.testing.assert_array_equal(getattr(out, name), getattr(s, name))

    def compile_script(self, operation):
        script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN)
 e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return Duel.IsExistingTarget(nil,tp,LOCATION_MZONE,0,1,nil) end
  Duel.SelectTarget(tp,nil,tp,LOCATION_MZONE,0,1,1,nil)
 end)
 e:SetOperation(function(e,tp)
  local g=Duel.GetTargetCards(e)
  OPERATION
 end)
 c:RegisterEffect(e)
end'''.replace('OPERATION', operation)
        old = translate.OFFICIAL
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930060.lua').write_text(script)
            translate.OFFICIAL = folder
            try:
                return translate.translate(600930060)
            finally:
                translate.OFFICIAL = old

    def test_top_statement_uses_move(self):
        effects, *_ = self.compile_script('Duel.SendtoDeck(g,nil,SEQ_DECKTOP,REASON_EFFECT)')
        self.assertTrue(any(i[0] == core.OP_MOVE and i[2] == core.Z_DECK for i in effects[0][6]))

    def test_rejects_other_sequences_players_and_return_values(self):
        for operation in ('Duel.SendtoDeck(g,nil,SEQ_DECKBOTTOM,REASON_EFFECT)',
                          'Duel.SendtoDeck(g,nil,SEQ_DECKSHUFFLE,REASON_EFFECT)',
                          'Duel.SendtoDeck(g,tp,SEQ_DECKTOP,REASON_EFFECT)',
                          'Duel.SendtoDeck(g,nil,SEQ_DECKTOP,REASON_EFFECT) Duel.ShuffleDeck(tp)',
                          'local n=Duel.SendtoDeck(g,nil,SEQ_DECKTOP,REASON_EFFECT)',
                          'Duel.SendtoDeck(g,nil,SEQ_DECKTOP,REASON_EFFECT,tp)'):
            with self.subTest(operation=operation), self.assertRaises(translate.Unsupported):
                self.compile_script(operation)

    def test_eleven_official_gains_compile(self):
        for code in (15576074, 30068120, 30068809, 35429292, 47453433,
                     49140998, 53989821, 56605802, 73125233, 78864369, 95204084):
            with self.subTest(code=code):
                translate.translate(code)

    def test_official_followup_shuffle_is_rejected(self):
        with self.assertRaisesRegex(translate.Unsupported, 'explicit shuffle'):
            translate.translate(44424095)

    def test_official_overlapping_triggers_keep_their_conditions(self):
        effects, *_ = translate.translate(6351548)
        self.assertEqual(len(effects), 2)
        self.assertTrue(all(effect[7]['cond'] > 0 for effect in effects))


if __name__ == '__main__':
    unittest.main()
