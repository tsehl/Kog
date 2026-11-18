"""Strict upper early returns become integer lower bounds on free zones."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import translate as T


class FreeZoneGuardTests(unittest.TestCase):
    def guard(self, expression, env=None, negate=True):
        tr = T.Translator(62706865)
        node = T.LA.parse(f'function probe() return {expression} end').body.body[0].body.body[0].values[0]
        return tr.loccount(node, env or {}, negate)

    def test_direct_integer_lower_bounds(self):
        for zone in ('LOCATION_MZONE', 'LOCATION_SZONE'):
            for n in (0, 1, 2, 5):
                with self.subTest(zone=zone, n=n):
                    self.assertEqual(self.guard(f'Duel.GetLocationCount(tp,{zone})<{n}', {'tp': T.Player(T.ME)}),
                                     (T.ME, T.K[zone], n))

    def test_local_opponent_count(self):
        value = T.FreeCount(T.Player(T.OPP), T.core.Z_MZONE)
        self.assertEqual(self.guard('ft<2', {'ft': value}), (T.OPP, T.core.Z_MZONE, 2))

    def test_positive_upper_bound_is_still_unrepresentable(self):
        self.assertIsNone(self.guard('ft<2', {'ft': T.FreeCount(T.Player(T.ME), T.core.Z_MZONE)}, negate=False))

    def test_fractional_threshold_is_refused(self):
        with self.assertRaisesRegex(T.Unsupported, 'integer threshold'):
            self.guard('ft<1.5', {'ft': T.FreeCount(T.Player(T.ME), T.core.Z_MZONE)})

    def test_non_free_count_is_not_recognized(self):
        self.assertIsNone(self.guard('ft<2', {'ft': 2}))

    def test_official_gains_translate(self):
        for code in (4072687, 13504844, 35272499, 62706865):
            with self.subTest(code=code):
                T.translate(code)


if __name__ == '__main__':
    unittest.main()
