from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'))
import core,cards,jax
import jax.numpy as jnp
import numpy as np
class SpecialResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank=core.make_bank(np.array([1]),[500],[500],[4],
            **cards.assemble([1],cards={1:[cards.act(resolve=[cards.SPSUMMON(0,core.P_FU_ATK)])]},
                             ritual={},fusion={},synchro={},xyz={},link={}))
        d=jnp.zeros(40,jnp.int32)
        s=jax.jit(core.new_game)(cls.bank,d,d)
        cls.state=s._replace(dec=jnp.int32(core.D_RUN),ch_src=jnp.int32(0),ch_eff=jnp.int32(0),
            ch_p=jnp.int32(0),pg=jnp.int32(core.PG_RESOLVE),pc=jnp.int32(0),
            zone=s.zone.at[5].set(core.Z_GRAVE),reg=s.reg.at[0,5].set(True))
        cls.execute=staticmethod(jax.jit(core.exec_one))
    def test_legal_special_summon_still_opens_placement(self):
        self.assertEqual(int(self.execute(self.bank,self.state).dec),core.D_ZONE)
    def test_a_later_prohibition_cancels_special_summon(self):
        s=self.state._replace(fl_c=self.state.fl_c.at[5].set(core.FL_NO_SS))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.pc),1);self.assertFalse(bool(result.reg[0].any()))
    def test_filled_main_zones_do_not_leave_a_decision_without_actions(self):
        s=self.state._replace(zone=self.state.zone.at[:5].set(core.Z_MZONE),seq=self.state.seq.at[:5].set(jnp.arange(5)))
        self.assertEqual(int(self.execute(self.bank,s).pc),1)
    def test_a_monster_already_on_field_is_not_special_summonable(self):
        s=self.state._replace(zone=self.state.zone.at[5].set(core.Z_MZONE),seq=self.state.seq.at[5].set(0))
        self.assertEqual(int(self.execute(self.bank,s).pc),1)
        mask=core.filter_mask(self.bank,s,jnp.array(cards.F(me=core.Z_MZONE,spsum=1)),0,0)
        self.assertFalse(bool(mask[5]))
    def test_a_failed_attempt_replaces_native_summon_information(self):
        s=self.state._replace(fl_c=self.state.fl_c.at[5].set(core.FL_NO_SS),
            suminfo=self.state.suminfo.at[5].set(core.SUMMON_TRIBUTE))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.suminfo[5]),int(core.summon_info(core.SUMMON_SPECIAL,core.Z_GRAVE,0)))
        self.assertEqual(int(result.zone[5]),core.Z_GRAVE)
    def test_a_space_failure_sends_a_hand_target_to_grave_by_rule(self):
        s=self.state._replace(zone=self.state.zone.at[:5].set(core.Z_MZONE).at[5].set(core.Z_HAND),
            seq=self.state.seq.at[:5].set(jnp.arange(5)).at[5].set(0))
        result=self.execute(self.bank,s)
        self.assertEqual(int(result.zone[5]),core.Z_GRAVE)
        self.assertEqual(int(result.suminfo[5]),int(core.summon_info(core.SUMMON_SPECIAL,core.Z_HAND,0)))
    def test_group_completes_before_emitting_success_and_keeps_all_results(self):
        s=self.state._replace(zone=self.state.zone.at[5:8].set(core.Z_GRAVE),
            seq=self.state.seq.at[5:8].set(jnp.arange(3)),reg=self.state.reg.at[0,5:8].set(True))
        original=np.asarray(s.reg[0]).copy()
        place=jax.jit(core.apply)
        for zone in range(3):
            s=self.execute(self.bank,s)
            self.assertEqual(int(s.dec),core.D_ZONE)
            self.assertFalse(bool(s.ev_g[core.EV_SPSUMMON].any()))
            np.testing.assert_array_equal(s.reg[0],original)
            s=place(self.bank,s,jnp.int32(core.special(80,core.A_ZONE0+zone)))
            s=place(self.bank,s,jnp.int32(core.special(80,core.A_POS_ATK)))
        s=self.execute(self.bank,s)
        self.assertEqual(int(s.pc),1)
        self.assertEqual(int((s.zone[5:8]==core.Z_MZONE).sum()),3)
        np.testing.assert_array_equal(s.reg[0],original)
        np.testing.assert_array_equal(s.ev_g[core.EV_SPSUMMON],original)
        self.assertFalse(bool((s.st & (core.ST_SEL|core.ST_PSTEP|core.ST_SSFAIL)).any()))

    def test_space_shortage_finishes_group_and_returns_only_successful_cards(self):
        s=self.state._replace(zone=self.state.zone.at[:3].set(core.Z_MZONE).at[5:8].set(core.Z_HAND),
            seq=self.state.seq.at[:3].set(jnp.arange(3)).at[5:8].set(jnp.arange(3)),
            reg=self.state.reg.at[0,5:8].set(True))
        place=jax.jit(core.apply)
        for zone in (3,4):
            s=self.execute(self.bank,s)
            self.assertEqual(int(s.dec),core.D_ZONE)
            self.assertTrue(bool((s.zone[5:8]!=core.Z_GRAVE).all()))
            s=place(self.bank,s,jnp.int32(core.special(80,core.A_ZONE0+zone)))
            s=place(self.bank,s,jnp.int32(core.special(80,core.A_POS_ATK)))
        s=self.execute(self.bank,s)
        self.assertEqual(int(s.pc),1)
        self.assertEqual(int(s.reg[0].sum()),2)
        self.assertEqual(int(s.zone[7]),core.Z_GRAVE)
        self.assertEqual(int(s.reason[7]),core.REASON_RULE)
        self.assertFalse(bool(s.ev_g[core.EV_SPSUMMON,7]))

    def test_full_main_zones_do_not_block_a_group_extra_deck_summon(self):
        s=self.state._replace(zone=self.state.zone.at[:5].set(core.Z_MZONE)
            .at[5:8].set(core.Z_HAND).at[8].set(core.Z_EXTRA),
            seq=self.state.seq.at[:5].set(jnp.arange(5)).at[5:8].set(jnp.arange(3)).at[8].set(0),
            reg=self.state.reg.at[0,5:9].set(True))
        # Rejections in the hand are collective; the Extra Deck keeps its placement
        # decision and sends by rule wait for the end of the group.
        s=self.execute(self.bank,s)
        self.assertEqual(int(s.dec),core.D_RUN)
        s=self.execute(self.bank,s)
        self.assertEqual(int(s.dec),core.D_ZONE)
        self.assertEqual(int(s.pend),8)
        self.assertTrue(bool((s.zone[5:8]==core.Z_HAND).all()))
        place=jax.jit(core.apply)
        s=place(self.bank,s,jnp.int32(core.special(80,core.A_ZONE0+5)))
        s=place(self.bank,s,jnp.int32(core.special(80,core.A_POS_ATK)))
        s=self.execute(self.bank,s)
        self.assertEqual(int(s.pc),1)
        self.assertEqual(int(s.zone[8]),core.Z_MZONE)
        self.assertTrue(bool((s.zone[5:8]==core.Z_GRAVE).all()))
        self.assertEqual(int(s.reg[0].sum()),1)
        self.assertTrue(bool(s.reg[0,8]))

    def test_pending_link_has_no_arrows_or_colinks_until_group_completion(self):
        bank=self.bank._replace(ctype=self.bank.ctype.at[0].set(core.TYPE_MONSTER|core.TYPE_LINK),
            lmark=self.bank.lmark.at[0].set(core.LM_BL|core.LM_BR))
        s=core.with_cid(bank,self.state,self.state.cid)._replace(zone=self.state.zone.at[5].set(core.Z_MZONE),
            seq=self.state.seq.at[5].set(5),pos=self.state.pos.at[5].set(core.P_FU_ATK),
            st=self.state.st.at[5].set(core.ST_PSTEP))
        self.assertFalse(bool(core.link_arrows(bank,s,0).any()))
        self.assertEqual(int(core.link_cells(bank,s,0)[0][5]),12)
        s=s._replace(st=s.st & ~core.ST_PSTEP)
        self.assertTrue(bool(core.link_arrows(bank,s,0).any()))
        self.assertEqual(int(core.link_cells(bank,s,0)[0][5]),5)

if __name__=='__main__':unittest.main()
