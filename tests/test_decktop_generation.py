"""Pure-Python coverage of the procedural verb that returns cards to the top of the Deck."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen"))
sys.path.insert(0, str(ROOT / "engine"))
import cardgen
import core


class DeckTopGenerationTest(unittest.TestCase):
    def test_sampled_clause_uses_deck_move_and_excludes_deck_extra_and_spell_trap_sources(self):
        seen = False
        for seed in range(512):
            gen = cardgen.Gen(seed)
            clause = gen.card_clause({"kind": "ignition", "zone": core.Z_MZONE})
            if clause["verb"] != "decktop":
                continue
            seen = True
            source_zones = clause["f"].get("me", 0) | clause["f"].get("op", 0)
            self.assertEqual(source_zones & (core.Z_DECK | core.Z_EXTRA | core.Z_SZONE), 0)
            self.assertNotEqual(clause["f"].get("not_typ", 0) & core.TYPE_EXTRA, 0)
            _, resolve, _ = cardgen.clause_bricks(clause, 0, True)
            self.assertEqual(resolve[-1][0], core.OP_MOVE)
            self.assertEqual(resolve[-1][2], core.Z_DECK)
        self.assertTrue(seen, "the low decktop weight is never drawn on this seed sequence")

    def test_minimum_attack_branch_keeps_extra_deck_exclusion(self):
        gen = cardgen.Gen(0)
        gen.pick = lambda xs, w=None: "decktop" if "decktop" in xs else ("opp_mon" if "opp_mon" in xs else xs[0])
        gen.p = lambda probability: True
        clause = gen.card_clause({"kind": "ignition", "zone": core.Z_MZONE})
        self.assertEqual(clause["verb"], "decktop")
        self.assertEqual(clause["sel"], "min")
        self.assertNotEqual(clause["f"].get("not_typ", 0) & core.TYPE_EXTRA, 0)


if __name__ == "__main__":
    unittest.main()
