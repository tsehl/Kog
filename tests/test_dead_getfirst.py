"""Tests for the strict dead Group:GetFirst loop-prelude recognizer."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "translate")]
import translate
from luaparser import astnodes as A

GAINS = (13314457, 17932494, 34109611, 36560997, 37279508,
         47805931, 74439492, 89547299, 89609515)
NEGATIVES = (15169262, 12247206)


def matches(source):
    tree = translate.LA.parse(source)
    for _, _, _, node, _ in translate._walk(tree):
        if not isinstance(node, A.Block):
            continue
        for i in range(len(node.body) - 1):
            if translate._dead_getfirst_loop_prelude(node.body[i], node.body[i + 1:]):
                return True
    return False


class DeadGetFirstTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(matches("local x=G:GetFirst()\nfor x in aux.Next(G) do print(x) end"))

    def test_rejects_global_assignment(self):
        self.assertFalse(matches("x=G:GetFirst()\nfor x in aux.Next(G) do print(x) end"))

    def test_rejects_arguments(self):
        self.assertFalse(matches("local x=G:GetFirst(1)\nfor x in aux.Next(G) do print(x) end"))
        self.assertFalse(matches("local x=G:GetFirst()\nfor x in aux.Next(G,1) do print(x) end"))

    def test_rejects_same_name_as_group(self):
        self.assertFalse(matches("local x=x:GetFirst()\nfor x in aux.Next(x) do print(x) end"))

    def test_rejects_different_group(self):
        self.assertFalse(matches("local x=G:GetFirst()\nfor x in aux.Next(H) do print(x) end"))

    def test_rejects_extra_loop_targets_or_iterators(self):
        self.assertFalse(matches("local x=G:GetFirst()\nfor x,y in aux.Next(G) do print(x) end"))
        self.assertFalse(matches("local x=G:GetFirst()\nfor x in aux.Next(G), aux.Next(G) do print(x) end"))

    def test_rejects_any_later_read(self):
        self.assertFalse(matches("local x=G:GetFirst()\nfor x in aux.Next(G) do print(x) end\nprint(x)"))

    def test_rejects_later_closure_capture(self):
        src = "local x=G:GetFirst()\nfor x in aux.Next(G) do print(x) end\nlocal f=function() return x end"
        self.assertFalse(matches(src))

    def test_local_scope_ends_with_nested_block(self):
        src = "local x=1\ndo\n local x=G:GetFirst()\n for x in aux.Next(G) do print(x) end\nend\nprint(x)"
        self.assertTrue(matches(src))

    def test_nine_real_scripts_translate(self):
        for code in GAINS:
            with self.subTest(code=code):
                translate.translate(code)

    def test_real_negative_scripts_remain_refused(self):
        for code in NEGATIVES:
            with self.subTest(code=code), self.assertRaises(translate.Unsupported) as ctx:
                translate.translate(code)
            self.assertEqual(str(ctx.exception), "GetFirst on a potentially multiple group not represented")


if __name__ == "__main__":
    unittest.main()
