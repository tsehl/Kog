"""GetTargetCards: filtered current group, distinct from the original targets."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('engine', 'translate')]
import cards, core, translate


class RelatedTargetGroupTests(unittest.TestCase):
    def test_previous_seventeen_gains_compile_related_filter(self):
        # These were the 17 gains from the first-target relation batch. Keep
        # their catalogue acceptance tied to the explicit related-filter bit.
        codes = (2203790, 4929256, 6343408, 21507589, 25401880, 28357177,
                 29826127, 36280194, 48976825, 49479374, 66149377, 77334267,
                 81191584, 90928333, 93920745, 96947648, 97518132)
        for code in codes:
            with self.subTest(code=code):
                effects, *_ = translate.translate(code)
                related = [ins for effect in effects for program in effect[6:9]
                           if isinstance(program, (list, tuple)) for ins in program
                           if ins[0] == core.OP_FILTER and ins[3] & (1 << 22)]
                self.assertTrue(related)

    def test_relation_filter_rejects_unproven_matching_group(self):
        script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return true end
  Duel.SelectTarget(tp,Card.IsFaceup,tp,LOCATION_MZONE,0,1,2,nil)
 end)
 e:SetOperation(function(e,tp)
  local g=Duel.GetMatchingGroup(Card.IsFaceup,tp,LOCATION_MZONE,0,nil)
  g=g:Filter(Card.IsRelateToEffect,nil,e)
  Duel.Destroy(g,REASON_EFFECT)
 end)
 c:RegisterEffect(e)
end'''
        self.assert_translation_refused(script, 'target group expected')

    def test_relation_filter_rejects_a_different_effect(self):
        script = '''local s,id=GetID()
function s.initial_effect(c)
 local e=Effect.CreateEffect(c)
 e:SetType(EFFECT_TYPE_ACTIVATE) e:SetCode(EVENT_FREE_CHAIN) e:SetProperty(EFFECT_FLAG_CARD_TARGET)
 e:SetTarget(function(e,tp,eg,ep,ev,re,r,rp,chk)
  if chk==0 then return true end
  Duel.SelectTarget(tp,Card.IsFaceup,tp,LOCATION_MZONE,0,1,2,nil)
 end)
 e:SetOperation(function(e,tp,eg,ep,ev,re,r,rp)
  local g=Duel.GetTargetCards(e):Filter(Card.IsRelateToEffect,nil,re)
  Duel.Destroy(g,REASON_EFFECT)
 end)
 c:RegisterEffect(e)
end'''
        self.assert_translation_refused(script, 'relation to another effect')

    def assert_translation_refused(self, script, reason):
        old = translate.OFFICIAL
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c990000001.lua').write_text(script)
            translate.OFFICIAL = folder
            try:
                with self.assertRaisesRegex(translate.Unsupported, reason):
                    translate.translate(990000001)
            finally:
                translate.OFFICIAL = old

    def test_filter_uses_target_clock_and_current_link(self):
        bank = core.make_bank(np.array([1]), [1000], [1000], [4])
        deck = jnp.zeros(40, jnp.int32)
        state = jax.jit(core.new_game)(bank, deck, deck)
        state = state._replace(ch_k=jnp.int32(1), ch_src=jnp.int32(0),
                               reg=state.reg.at[0, :2].set(True),
                               lk_t=state.lk_t.at[1].set(10),
                               lk_tgt_t=state.lk_tgt_t.at[0].set(20).at[1].set(3),
                               mv=state.mv.at[:2].set(jnp.array([1, 5])))
        # Exact copy then relation filtering, even if the ordinary filter does not
        # match the cards: the copy bit must not overwrite the mask.
        ins = jnp.array((core.OP_FILTER, 1, 0, 1 | (1 << 20) | (1 << 22)))
        run = jax.jit(lambda s: core._pure(bank, s, s.reg, jnp.bool_(True),
                                         jnp.int32(0), jnp.int32(0), jnp.int32(0),
                                         s.ch_k, ins)[0])
        result = run(state)
        np.testing.assert_array_equal(result[1, :2], [True, False])
        np.testing.assert_array_equal(result[0], state.reg[0])
        # New query after a leave and return: nothing is recovered.
        returned = state._replace(mv=state.mv.at[0].set(11))
        self.assertFalse(bool(run(returned)[1].any()))

    def test_group_copy_cannot_be_elided_or_shared_with_plain_copy(self):
        effect = (0, 0, 0, 0, [], [], [cards.SELF(0),
                  translate.Translator.vcopy(1, 0, related=True), cards.MOVE(1),
                  translate.Translator.vcopy(2, 0), cards.MOVE(2)])
        instructions = translate.Translator.allocate(effect)[6]
        related = [i for i in instructions if i[0] == core.OP_FILTER and i[3] & (1 << 22)]
        self.assertEqual(len(related), 1)
        self.assertTrue(related[0][3] & (1 << 20))

    def test_official_recycling_batteries_compiles(self):
        effects, *_ = translate.translate(99995595)
        self.assertTrue(any(i[0] == core.OP_FILTER and i[3] & (1 << 22)
                            for i in effects[0][6]))

    def test_first_of_filtered_multiple_group_stays_rejected(self):
        module = ast.parse((ROOT / 'tests/target_relation_oracle_check.py').read_text())
        source = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'script')
        source = source.replace('LOCATION_SZONE,0,1,1,e:GetHandler()',
                                'LOCATION_SZONE,0,2,2,e:GetHandler()')
        source = source.replace('Duel.GetFirstTarget()', 'Duel.GetTargetCards(e):GetFirst()')
        old = translate.OFFICIAL
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c600930051.lua').write_text(source)
            translate.OFFICIAL = folder
            try:
                with self.assertRaisesRegex(translate.Unsupported, 'GetFirst.*multiple'):
                    translate.translate(600930051)
            finally:
                translate.OFFICIAL = old


if __name__ == '__main__':
    unittest.main()
