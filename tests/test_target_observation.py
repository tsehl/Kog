"""Public targets per chain link, independent of the temporary registers."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'train', 'translate')]
import cards, core, model, obs, translate


class TargetObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = core.make_bank(np.array([1]), [1000], [1000], [4])
        deck = jnp.zeros(40, jnp.int32)
        s = jax.jit(core.new_game)(cls.bank, deck, deck)
        assert s.ch_targets.dtype == jnp.bool_ and s.lk_targets.dtype == jnp.bool_
        assert s.ch_targets.nbytes + s.lk_targets.nbytes == s.cid.shape[0] * (core.CH + 1)
        cls.state = s._replace(zone=s.zone.at[:3].set(core.Z_MZONE),
                               seq=s.seq.at[:3].set(jnp.arange(3)),
                               pos=s.pos.at[:3].set(core.P_FU_ATK),
                               n_chain=jnp.int32(2),
                               lk_src=s.lk_src.at[:2].set(jnp.array([0, 1])),
                               lk_targets=s.lk_targets.at[0, 2].set(True).at[1, 2].set(True),
                               lk_tgt_t=s.lk_tgt_t.at[0].set(3).at[1].set(7))
        mask = jnp.zeros(core.NV * s.cid.shape[0] + core.N_SPECIAL, bool)
        cls.observe = staticmethod(jax.jit(lambda s: obs.observe(cls.bank, s, p=0, m=mask)))

    def test_target_membership_is_per_link_and_lost_on_departure(self):
        for moved, expected in ((0, [True, True]), (4, [False, True]), (8, [False, False])):
            s = self.state._replace(mv=self.state.mv.at[2].set(moved))
            view = self.observe(s)
            token = int(jnp.argmax(view.slot == 2))
            np.testing.assert_array_equal(view.targeted[token, :2] > 0, expected)
            self.assertFalse(bool(view.targeted[:, 2:].any()))
        # A return to the field does not recreate the link, even if the register keeps it.
        s = self.state._replace(mv=self.state.mv.at[2].set(9),
                                lk_reg=self.state.lk_reg.at[:, 1, 2].set(True))
        self.assertFalse(bool(self.observe(s).targeted.any()))

    def test_no_hidden_token_and_no_dependency_on_source_relation(self):
        hidden = self.state._replace(zone=self.state.zone.at[2].set(core.Z_DECK),
                                     ctrl=self.state.ctrl.at[2].set(1))
        self.assertFalse(bool(self.observe(hidden).targeted.any()))
        departed_source = self.state._replace(mv=self.state.mv.at[0].set(20))
        view = self.observe(departed_source)
        token = int(jnp.argmax(view.slot == 2))
        self.assertTrue(bool(view.targeted[token, 0]))

    def test_first_announcement_does_not_fall_back_after_departure(self):
        s = self.state._replace(n_chain=jnp.int32(1), uid=jnp.arange(self.state.cid.shape[0]),
                                lk_targets=self.state.lk_targets.at[0, 1].set(True))
        view = self.observe(s)
        first, second = (int(jnp.argmax(view.slot == i)) for i in (1, 2))
        self.assertEqual(int(view.targeted[first, 0]), 2)
        self.assertEqual(int(view.targeted[second, 0]), 1)
        left = self.observe(s._replace(mv=s.mv.at[1].set(4)))
        self.assertEqual(int(left.targeted[first, 0]), 0)
        self.assertEqual(int(left.targeted[second, 0]), 1)

    def test_target_record_survives_register_reuse_and_zero_clock(self):
        # The hand-written Typhon must feed the same observation as the cards
        # coming from the generator and the translator.
        for ins in (cards.CARDS[5318639][0][5][-1], cards.LABEL(1, target=True)):
            bank = self.bank._replace(prog=self.bank.prog.at[0, 0, core.PG_TARGET, 0].set(jnp.array(ins)))
            s = self.state._replace(ch_src=jnp.int32(0), ch_eff=jnp.int32(0),
                                    pg=jnp.int32(core.PG_TARGET), pc=jnp.int32(0),
                                    clk=jnp.int32(0), n_chain=jnp.int32(0),
                                    st=self.state.st.at[2].set(core.ST_SEL),
                                    reg=self.state.reg.at[1, 2].set(True))
            if ins[0] == core.OP_CHOOSE:
                selected = core.sel_finish(bank, s)
            else:
                selected = jax.jit(core.exec_one)(bank, s)
            self.assertEqual(int(selected.ch_targets[2]), 1)
            selected = selected._replace(reg=jnp.zeros_like(selected.reg))
            pushed = jax.jit(core.push_link)(bank, selected)
            self.assertEqual(int(pushed.lk_targets[0, 2]), 1)
            self.assertTrue(bool(self.observe(pushed).targeted.any()))
        ordinary = self.bank._replace(prog=self.bank.prog.at[0, 0, core.PG_TARGET, 0].set(
            jnp.array(cards.CHOOSE(1, 0, 1))))
        self.assertFalse(bool(core.sel_finish(ordinary, s).ch_targets.any()))

    def test_unannounced_relationships_are_not_shown_as_public_targets(self):
        for ins in (cards.CHOOSE(1, 0, 1, tgt=True, public=False),
                    cards.LABEL(1, target=True, public=False)):
            bank = self.bank._replace(prog=self.bank.prog.at[0, 0, core.PG_TARGET, 0].set(jnp.array(ins)))
            s = self.state._replace(ch_src=jnp.int32(0), ch_eff=jnp.int32(0),
                                    pg=jnp.int32(core.PG_TARGET), pc=jnp.int32(0),
                                    clk=jnp.int32(6), ch_tgt_t=jnp.int32(1),
                                    st=self.state.st.at[2].set(core.ST_SEL),
                                    reg=self.state.reg.at[1, 2].set(True))
            result = (core.sel_finish(bank, s) if ins[0] == core.OP_CHOOSE
                      else jax.jit(core.exec_one)(bank, s))
            self.assertEqual(int(result.ch_tgt_t), 6)
            self.assertFalse(bool(result.ch_targets.any()))

    def test_translation_keeps_relationship_separate_from_public_targeting(self):
        module = ast.parse((ROOT / 'tests/target_relation_oracle_check.py').read_text())
        script = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'script')
        code = 600930051
        old = translate.OFFICIAL
        with tempfile.TemporaryDirectory() as folder:
            translate.OFFICIAL = folder
            try:
                for public in (True, False):
                    source = script if public else script.replace('e:SetProperty(EFFECT_FLAG_CARD_TARGET)', '')
                    for label in (False, True):
                        variant = source
                        if label:
                            variant = variant.replace(
                                'Duel.SelectTarget(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,1,1,e:GetHandler())',
                                'Duel.SetTargetCard(Duel.SelectMatchingCard(tp,Card.IsSpellTrap,tp,LOCATION_SZONE,0,1,1,e:GetHandler()))')
                        Path(folder, f'c{code}.lua').write_text(variant)
                        effect = translate.translate(code)[0][0]
                        instructions = effect[5]
                        if label:
                            marker = next(i for i in instructions if i[0] == core.OP_LABEL)
                            self.assertEqual(marker[3], int(not public))
                        else:
                            choice = next(i for i in instructions if i[0] == core.OP_CHOOSE)
                            self.assertTrue(choice[3] & (1 << 10))
                            self.assertEqual(bool(choice[3] & (1 << 11)), not public)
                Path(folder, f'c{code}.lua').write_text(script.replace('e:SetTarget(', 'e:SetCost('))
                with self.assertRaisesRegex(translate.Unsupported, 'SelectTarget outside the targeting'):
                    translate.translate(code)
            finally:
                translate.OFFICIAL = old

    def test_procedure_selection_does_not_require_a_chain_target_property(self):
        # Quickdraw Synchron picks a discard in its procedure, compiled
        # before programs() and hence without a chain link targeting property.
        source = Path(translate.OFFICIAL, 'c20932152.lua').read_text()
        # Quickdraw's other effects remain outside coverage; isolating its
        # procedure makes it possible to check this regression without accepting them.
        source = source[:source.index('\t--\n\tlocal e2')] + 'end\n' + source[source.index('function s.synlimit'):]
        old = translate.OFFICIAL
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c20932152.lua').write_text(source)
            translate.OFFICIAL = folder
            try:
                effects = translate.translate(20932152)[0]
            finally:
                translate.OFFICIAL = old
        procedure = next(effect for effect in effects if effect[0] == core.K_PROC)
        choices = [i for i in procedure[5] if i[0] == core.OP_CHOOSE]
        self.assertTrue(choices)
        self.assertTrue(all(not (i[3] & (3 << 10)) for i in choices))
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c20932152.lua').write_text(source.replace('Duel.SelectMatchingCard', 'Duel.SelectTarget'))
            translate.OFFICIAL = folder
            try:
                with self.assertRaisesRegex(translate.Unsupported, 'SelectTarget outside the targeting'):
                    translate.translate(20932152)
            finally:
                translate.OFFICIAL = old

    def test_previous_dynamic_feature_prefix_has_identical_outputs(self):
        p = model.init(jax.random.PRNGKey(19), d=16, card_layers=0, layers=0)
        old = dict(p, dyn=dict(p['dyn'], w=p['dyn']['w'][:-core.CH]))
        migrated = model.complete(old)
        tokens, mask = model.card_tokens(self.bank)
        embedding = model.encode_cards(p['cards'], tokens, mask)
        view = self.observe(self.state)
        self.assertTrue(bool(view.targeted.any()))
        infer = jax.jit(model.policy)
        for a, b in zip(infer(old, embedding, view), infer(migrated, embedding, view)):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
