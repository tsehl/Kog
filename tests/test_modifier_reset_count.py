"""Do not shorten multi-phase temporary modifiers to the current turn."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engine'), str(ROOT / 'translate')]
import translate as T


class ModifierResetCountTests(unittest.TestCase):
    def variant(self, arguments):
        source = Path(T.OFFICIAL, 'c59228631.lua').read_text().replace(
            'SetReset(RESETS_STANDARD_PHASE_END,2)', f'SetReset({arguments})')
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'c59228631.lua').write_text(source)
            with patch.object(T, 'OFFICIAL', folder):
                return T.translate(59228631)

    def test_two_end_phases_stay_refused(self):
        with self.assertRaisesRegex(T.Unsupported, 'reset count'):
            T.translate(59228631)

    def test_zero_one_and_default_have_identical_bytecode(self):
        default = self.variant('RESETS_STANDARD_PHASE_END')
        for count in (0, 1):
            with self.subTest(count=count):
                self.assertEqual(self.variant(f'RESETS_STANDARD_PHASE_END,{count}'), default)

    def test_other_counts_and_extra_arguments_stay_refused(self):
        for args in ('2', '3', '-1', 'true', '1,2'):
            with self.subTest(args=args), self.assertRaisesRegex(T.Unsupported, 'reset count'):
                self.variant(f'RESETS_STANDARD_PHASE_END,{args}')


if __name__ == '__main__':
    unittest.main()
