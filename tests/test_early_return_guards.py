"""Only representable short-circuit guards enter the translated catalogue."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import translate as T


class EarlyReturnGuardTests(unittest.TestCase):
    def check_guard(self, expression, negate=True):
        tr = T.Translator(6795211)
        prog = T.Prog(tr, 'resolve')
        tree = T.LA.parse(f'function probe() return {expression} end')
        term = tree.body.body[0].body.body[0].values[0]
        env = {'g': T.Reg(100, one=False), 'h': T.Reg(101, one=False)}
        tr.res_cond(term, env, prog, negate=negate)
        return prog.ins

    def test_nested_or_guards_preserve_order(self):
        self.assertEqual(self.check_guard('#g==0 or (#h==0 or #g>3)'),
                         [T.C.REQ(100, 1), T.C.REQ(101, 1), T.C.REQ(100, 3, maximum=True)])

    def test_positive_or_still_requires_branching(self):
        with self.assertRaises(T.Unsupported):
            self.check_guard('#g>0 or #h>0', negate=False)

    def test_negated_conjunction_still_requires_branching(self):
        with self.assertRaises(T.Unsupported):
            self.check_guard('#g==0 and #h==0')

    def test_implicit_reference_cannot_be_inverted(self):
        for expr in ('g or #h==0', '#h==0 or g:IsRelateToEffect(e)'):
            with self.subTest(expr=expr), self.assertRaisesRegex(T.Unsupported, 'implicit reference'):
                self.check_guard(expr)

    def test_nonzero_inequality_stays_rejected(self):
        with self.assertRaisesRegex(T.Unsupported, 'non-zero constant'):
            self.check_guard('#g==2 or #h==0')

    def test_official_gains_translate(self):
        for code in (6795211, 57441100):
            with self.subTest(code=code):
                T.translate(code)


if __name__ == '__main__':
    unittest.main()
