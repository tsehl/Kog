"""Shared subtype bits must distinguish Ritual/Continuous primary categories."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import core
import translate as T

GAINS = (13626450, 1669772, 9391354, 1073952, 57617178, 23401839, 72903645)
PRIMARY = core.TYPE_MONSTER | core.TYPE_SPELL | core.TYPE_TRAP


class SharedCompositeTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = T.Translator(57617178)

    def expression(self, source):
        tree = T.LA.parse(f'function probe(c) return {source} end')
        fl = T.Filt()
        for term in self.translator.conj(tree.body.body[0].body.body[0].values[0]):
            self.translator.filter_term(fl, term, {}, 'c')
        return fl

    def test_pinned_metadata_has_one_primary_type_for_shared_subtypes(self):
        for specialty, _ in T.SHARED_COMPOSITE_TYPES.values():
            for ctype, _ in T.card_metadata().values():
                if ctype & specialty:
                    self.assertIn(ctype & PRIMARY, (1, 2, 4))

    def test_compiled_filter_matches_native_all_bits_over_catalogue_and_pendulum_spell(self):
        types = {ctype for ctype, _ in T.card_metadata().values()} | {core.P_SPELL}
        for method, (specialty, primary) in T.SHARED_COMPOSITE_TYPES.items():
            fl = self.expression(f'c:{method}()').build(core.Z_DECK, core.Z_GRAVE)
            for ctype in types:
                expected = ctype & (specialty | primary) == specialty | primary
                actual = bool(ctype & fl[core.F_TYPE]) and not (ctype & fl[core.F_NOTTYPE])
                self.assertEqual(actual, expected, (method, ctype))

    def test_opposite_primary_categories_with_same_shared_bit_are_excluded(self):
        for method, (specialty, wanted) in T.SHARED_COMPOSITE_TYPES.items():
            fl = self.expression(f'c:{method}()')
            for primary in (core.TYPE_MONSTER, core.TYPE_SPELL, core.TYPE_TRAP):
                ctype = primary | specialty
                self.assertEqual(bool(ctype & fl.d['typ']) and not (ctype & fl.d['not_typ']), primary == wanted)

    def test_arguments_remain_refused(self):
        for method in T.SHARED_COMPOSITE_TYPES:
            with self.subTest(method=method), self.assertRaises(T.Unsupported):
                self.expression(f'c:{method}(1)')

    def test_union_of_shared_composites_is_not_flattened_to_a_subtype_union(self):
        with self.assertRaises(T.Unsupported):
            self.expression('c:IsRitualSpell() or c:IsRitualMonster()')
        with self.assertRaises(T.Unsupported):
            self.expression('c:IsContinuousSpell() or c:IsContinuousTrap()')

    def test_combines_with_faceup_and_existing_type_exclusions(self):
        fl = self.expression('c:IsContinuousSpell() and c:IsFaceup() and not c:IsType(TYPE_FIELD)')
        self.assertEqual(fl.d['pos'], core.P_FACEUP)
        self.assertEqual(fl.d['not_typ'], core.TYPE_MONSTER | core.TYPE_TRAP | T.K['TYPE_FIELD'])

    def test_seven_official_scripts_translate(self):
        for code in GAINS:
            with self.subTest(code=code):
                T.translate(code)


if __name__ == '__main__':
    unittest.main()
