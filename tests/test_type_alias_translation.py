"""Focused pure translation tests for the five composite-type aliases.

Run from the repository with the project's Python environment after the alias
translator patch is integrated. This file intentionally does not initialize JAX.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import core
import translate as T

TYPE_FIELD = 0x80000
COMPOSITES = {
    'IsLinkMonster': (core.TYPE_LINK, core.TYPE_MONSTER),
    'IsFusionMonster': (core.TYPE_FUSION, core.TYPE_MONSTER),
    'IsSynchroMonster': (core.TYPE_SYNCHRO, core.TYPE_MONSTER),
    'IsXyzMonster': (core.TYPE_XYZ, core.TYPE_MONSTER),
    'IsFieldSpell': (TYPE_FIELD, core.TYPE_SPELL),
}
GAIN_CODES = (58392024, 91482773, 97526666)


class CompositeTypeAliasTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = T.Translator(58392024)

    def expression(self, source):
        tree = T.LA.parse(f'function probe(c) return {source} end')
        return tree.body.body[0].body.body[0].values[0]

    def compile_expression(self, source):
        filt = T.Filt()
        node = self.expression(source)
        for term in self.translator.conj(node):
            self.translator.filter_term(filt, term, {}, 'c')
        return filt

    def test_card_metadata_composite_masks_imply_primary_type(self):
        # Native Card.IsCompositeType checks (type & mask) == mask. Since each
        # helper uses primary|specialty, the specialty-only translation is
        # equivalent exactly when this pinned metadata invariant holds.
        types = [ctype for ctype, _level in T.card_metadata().values()]
        self.assertTrue(types)
        for method, (specialty, primary) in COMPOSITES.items():
            with self.subTest(method=method):
                violating = [ctype for ctype in types
                             if ctype & specialty and (ctype & primary) != primary]
                self.assertEqual(violating, [])

    def test_all_three_catalogue_gains_compile(self):
        for code in GAIN_CODES:
            with self.subTest(code=code):
                T.translate(code)

    def test_composite_methods_reject_arguments_through_lua_ast(self):
        for method in COMPOSITES:
            with self.subTest(method=method):
                with self.assertRaises(T.Unsupported):
                    self.compile_expression(f'c:{method}(1)')

    def test_contradictory_primary_types_are_rejected(self):
        # A monster cannot also be a field spell; merge_typ must reject the
        # incompatible type axes instead of silently constructing a filter.
        with self.assertRaises(T.Unsupported):
            self.compile_expression('c:IsMonster() and c:IsFieldSpell()')

    def test_or_between_type_and_position_axes_is_rejected(self):
        with self.assertRaises(T.Unsupported):
            self.compile_expression('c:IsLinkMonster() or c:IsFaceup()')

    def test_positive_specialty_and_primary_conjunction_is_representable(self):
        for method, primary_name in (
            ('IsLinkMonster', 'IsMonster'),
            ('IsFusionMonster', 'IsMonster'),
            ('IsSynchroMonster', 'IsMonster'),
            ('IsXyzMonster', 'IsMonster'),
            ('IsFieldSpell', 'IsSpell'),
        ):
            with self.subTest(method=method):
                filt = self.compile_expression(f'c:{primary_name}() and c:{method}()')
                self.assertNotEqual(filt.d['typ'], 0)


if __name__ == '__main__':
    unittest.main()
