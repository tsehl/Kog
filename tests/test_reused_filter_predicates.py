"""Sequence and disabled-state predicates reuse existing filter fields."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import core
import translate as T


class ReusedFilterPredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = T.Translator(61068510)

    def expression(self, source):
        tree = T.LA.parse(f'function probe(c) return {source} end')
        fl = T.Filt()
        for term in self.translator.conj(tree.body.body[0].body.body[0].values[0]):
            self.translator.filter_term(fl, term, {}, 'c')
        return fl

    def test_main_sequences_use_existing_flag(self):
        fl = self.expression('c:GetSequence()<5').build(core.Z_SZONE, 0)
        self.assertEqual(fl[core.F_ZX], core.ZX_MAIN)
        self.assertEqual(fl[core.F_LOC_ME], core.Z_SZONE)

    def test_sequence_filter_preserves_other_constraints(self):
        fl = self.expression('c:IsFaceup() and c:GetSequence()<5')
        self.assertEqual(fl.d['pos'], core.P_FACEUP)
        self.assertEqual(fl.d['zx'], core.ZX_MAIN)

    def test_other_sequence_comparisons_are_refused(self):
        for expr in ('c:GetSequence()<4', 'c:GetSequence()>4', 'c:GetSequence()<6',
                     'c:GetSequence(1)<5', 'c:GetSequence()==5'):
            with self.subTest(expr=expr), self.assertRaises(T.Unsupported):
                self.expression(expr)

    def test_not_disabled_does_not_implicitly_require_faceup(self):
        fl = self.expression('not c:IsDisabled()')
        self.assertEqual(fl.d['nodis'], 1)
        self.assertEqual(fl.d['pos'], 0)

    def test_not_disabled_preserves_type_and_level_constraints(self):
        fl = self.expression('c:IsType(TYPE_EFFECT) and c:IsLevelAbove(5) and not c:IsDisabled()')
        self.assertEqual(fl.d['typ'], T.K['TYPE_EFFECT'])
        self.assertEqual(fl.d['lv'][0], 5)
        self.assertEqual(fl.d['nodis'], 1)

    def test_positive_or_argument_disabled_predicates_are_refused(self):
        for expr in ('c:IsDisabled()', 'not c:IsDisabled(1)'):
            with self.subTest(expr=expr), self.assertRaises(T.Unsupported):
                self.expression(expr)

    def test_three_official_gains_translate(self):
        for code in (61068510, 24326617, 64554883):
            with self.subTest(code=code):
                T.translate(code)


if __name__ == '__main__':
    unittest.main()
