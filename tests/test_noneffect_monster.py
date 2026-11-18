"""Non-effect monsters reuse positive type and excluded-type filters."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import core
import translate as T


class NonEffectMonsterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = T.Translator(23431858)

    def expression(self, source):
        tree = T.LA.parse(f'function probe(c) return {source} end')
        fl = T.Filt()
        for term in self.translator.conj(tree.body.body[0].body.body[0].values[0]):
            self.translator.filter_term(fl, term, {}, 'c')
        return fl.build(core.Z_MZONE, core.Z_MZONE)

    def test_native_predicate_over_catalogue_and_dynamic_pendulum_spell_type(self):
        fl = self.expression('c:IsNonEffectMonster()')
        for ct in {ct for ct, _ in T.card_metadata().values()} | {core.P_SPELL}:
            expected = bool(ct & core.TYPE_MONSTER) and not bool(ct & T.K['TYPE_EFFECT'])
            actual = bool(ct & fl[core.F_TYPE]) and not bool(ct & fl[core.F_NOTTYPE])
            self.assertEqual(actual, expected, ct)
        self.assertEqual(fl[core.F_POS], 0)  # face-down is allowed by the primitive

    def test_faceup_is_a_separate_condition(self):
        fl = self.expression('c:IsNonEffectMonster() and c:IsFaceup()')
        self.assertEqual(fl[core.F_POS], core.P_FACEUP)
        self.assertEqual(fl[core.F_NODIS], 0)  # a negated effect monster remains an effect monster

    def test_exclusions_are_combined(self):
        fl = self.expression('c:IsNonEffectMonster() and not c:IsType(TYPE_PENDULUM)')
        self.assertEqual(fl[core.F_NOTTYPE], T.K['TYPE_EFFECT'] | core.TYPE_PENDULUM)

    def test_unsupported_forms_stay_refused(self):
        for source in ('c:IsNonEffectMonster(1)', 'not c:IsNonEffectMonster()',
                       'c:IsNonEffectMonster() or c:IsFaceup()'):
            with self.subTest(source=source), self.assertRaises(T.Unsupported):
                self.expression(source)

    def test_official_vishuda_translates(self):
        T.translate(23431858)


if __name__ == '__main__':
    unittest.main()
