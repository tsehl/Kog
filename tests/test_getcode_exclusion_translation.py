import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine'))
sys.path.insert(0, str(ROOT / 'translate'))
import translate as T

GAIN_CODES = (
    984114, 2407147, 23782705, 23899727, 24040093, 28118128,
    37349495, 40320754, 44250812, 60999392, 61592395, 66378485,
    67111213, 69293721, 72328962, 83446909, 89739383, 90726340,
    92039899, 95027497, 96682430, 97439806, 98780137, 99877698,
)

class GetCodeExclusionTranslationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = T.Translator(11834972)

    def parse_term(self, expression):
        tree = T.LA.parse(f'function probe(c, d) return {expression} end')
        fn = tree.body.body[0]
        return fn.body.body[0].values[0]

    def compile_term(self, expression):
        filt = T.Filt()
        self.tr.filter_term(filt, self.parse_term(expression), {'id': 11834972}, 'c')
        return filt

    def test_positive_static_code_exclusion(self):
        for expression in ('c:GetCode() ~= id', 'c:GetCode() ~= 11834972', 'not c:IsCode(id)'):
            with self.subTest(expression=expression):
                filt = self.compile_term(expression)
                self.assertEqual(filt.d['code'], (-11834972, -11834972))

    def test_rejects_unknown_dynamic_negative_zero_args_original_and_other_card(self):
        invalid = (
            ('c:GetCode() ~= missing', {}),
            ('c:GetCode() ~= dynamic', {'dynamic': None}),
            ('c:GetCode() ~= -3', {}),
            ('c:GetCode() ~= 0', {}),
            ('c:GetCode() ~= 2147483648', {}),
            ('c:GetCode() ~= 4294967296', {}),
            ('c:GetCode() ~= true', {}),
            ('c:GetCode(1) ~= id', {}),
            ('c:GetOriginalCode() ~= id', {}),
            ('d:GetCode() ~= id', {}),
        )
        for expression, env in invalid:
            with self.subTest(expression=expression):
                filt = T.Filt()
                with self.assertRaises(T.Unsupported):
                    self.tr.filter_term(filt, self.parse_term(expression), {'id': 11834972, **env}, 'c')

    def test_rejects_collision_with_positive_iscode(self):
        expr = 'c:GetCode() ~= id and c:IsCode(id)'
        filt = T.Filt()
        with self.assertRaisesRegex(T.Unsupported, 'two name conditions'):
            for term in self.tr.conj(self.parse_term(expr)):
                self.tr.filter_term(filt, term, {'id': 11834972}, 'c')

    def test_all_24_catalogue_gains_translate(self):
        for code in GAIN_CODES:
            with self.subTest(code=code):
                T.translate(code)

if __name__ == '__main__':
    unittest.main()
