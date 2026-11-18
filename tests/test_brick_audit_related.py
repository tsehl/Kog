"""The audit tells related filters from ordinary filters in a generated bank."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("engine", "gen", "train")]
import brick_audit
import cardgen
import core
import model
import numpy as np


class BrickAuditRelatedTests(unittest.TestCase):
    def test_generated_related_filter_is_distinct_from_plain_filter(self):
        spec = dict(code=600000123, ctype=core.TYPE_MONSTER | 0x20, atk=1000, df=1000, lv=4,
                    attr=1, race=1, setc=[], lsc=0, rsc=0, recipe=None)
        spec["effects"] = [dict(kind="ignition", on=0, opt=0, optional=False, rng=0, dmg=0,
                                 conds=[], cost=("discard",), clauses=[
                                     dict(verb="destroy", sel="target", f={"op": core.Z_MZONE},
                                          k=2, v=0, first_original=True)])]
        bank = cardgen.to_bank([spec])
        features = brick_audit.features(bank, 0)

        self.assertIn("cost FILTER ()", features)
        self.assertIn("target FILTER ('target',)", features)
        self.assertIn("resolve FILTER ('within', 'related')", features)
        self.assertEqual(brick_audit.filter_modes(3 | 8 | (1 << 20) | (1 << 22)),
                         ("within", "target", "copy", "related"))
        tokens, mask = model.card_tokens(bank)
        program = np.asarray(bank.prog[0, 0, core.PG_RESOLVE])
        related = np.flatnonzero((program[:, 0] == core.OP_FILTER) & ((program[:, 3] & (1 << 22)) != 0))
        self.assertEqual(len(related), 1)
        instruction = int(related[0])
        self.assertTrue(bool(mask['ins'][0, 0, core.PG_RESOLVE, instruction]))
        # The last sixteen channels are bits 16 to 31 of argument c.
        self.assertEqual(float(tokens['ins'][0, 0, core.PG_RESOLVE, instruction, -10]), 1.0)
        self.assertEqual(float(tokens['ins'][0, 0, core.PG_RESOLVE, instruction, -11]), 0.0)


if __name__ == "__main__":
    unittest.main()
