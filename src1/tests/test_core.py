"""核心数据契约测试。"""

from __future__ import annotations

import unittest

from src1.pmtskill_v2.core.io import load_primitives
from src1.pmtskill_v2.core.models import SkillTopology, TopologyNode


class CoreModelTest(unittest.TestCase):
    def test_primitive_catalog_has_expected_size(self):
        primitives = load_primitives()
        self.assertEqual(len(primitives), 26)
        self.assertEqual(len({item.primitive_id for item in primitives}), 26)

    def test_topological_sort_and_cycle_detection(self):
        topology = SkillTopology.from_sequence(
            ["reason.intent", "action.click", "reason.verify"]
        )
        self.assertEqual(
            topology.primitive_sequence(),
            ("reason.intent", "action.click", "reason.verify"),
        )
        cyclic = SkillTopology(
            (
                TopologyNode("a", "reason.intent", ("b",)),
                TopologyNode("b", "reason.verify", ("a",)),
            )
        )
        with self.assertRaisesRegex(ValueError, "存在环"):
            cyclic.validate()


if __name__ == "__main__":
    unittest.main()

