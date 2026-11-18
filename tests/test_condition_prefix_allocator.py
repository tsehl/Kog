"""Register allocation preserves the SetCondition / cost boundary."""
import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("engine", "translate")]
import cards as C
import core
from translate import Translator


class ConditionPrefixAllocatorTests(unittest.TestCase):
    def test_cost_cse_does_not_move_duplicate_into_condition_prefix(self):
        f = C.F(me=core.Z_GRAVE)
        effect = C.trigger(core.EV_SUMMON,
                           cost=[Translator.vfilter(0, f), C.REQ(0, 1),
                                 Translator.vfilter(1, f), C.REQ(1, 1)],
                           cond=2)

        allocated = Translator.allocate(effect)

        self.assertEqual(allocated[7]["cond"], 2)
        self.assertEqual(len(allocated[4]), 2)
        self.assertEqual(allocated[4][0][0], core.OP_FILTER)
        self.assertEqual(allocated[4][1][0], core.OP_REQ)

    def test_removed_pure_instruction_does_not_include_condition_and_cost(self):
        f = C.F(me=core.Z_GRAVE)
        effect = C.trigger(core.EV_SUMMON,
                           cost=[C.SELF(0), C.COND_TURN(C.ME),
                                 Translator.vfilter(1, f), C.REQ(1, 1)],
                           cond=2)

        allocated = Translator.allocate(effect)

        self.assertEqual(allocated[7]["cond"], 1)
        self.assertEqual(len(allocated[4]), 3)
        self.assertEqual(allocated[4][0][0], core.OP_COND_TURN)
        self.assertEqual(allocated[4][1][0], core.OP_FILTER)
        self.assertEqual(allocated[4][2][0], core.OP_REQ)


if __name__ == "__main__":
    unittest.main()
