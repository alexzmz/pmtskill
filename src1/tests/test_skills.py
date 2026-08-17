"""SKVM 导入与技能生命周期测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src1.pmtskill_v2.core.config import MaintenanceConfig
from src1.pmtskill_v2.core.models import ExecutionTrace, TraceEvent, SkillStatus
from src1.pmtskill_v2.skills.importer import import_skvm_skills
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore
from src1.pmtskill_v2.skills.compiler import (
    LLMRawSkillCompiler,
    compile_imported_raw_skills,
)
from src1.pmtskill_v2.core.io import load_primitives


class SkillStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SkillStore(self.root / "skills.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    def test_skvm_import_is_idempotent_and_not_active(self):
        skill_dir = self.root / "skvm" / "android-tap"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Android Tap\ndescription: Click a visible mobile UI element\n---\n"
            "Read the screen, tap the target and verify the result.\n",
            encoding="utf-8",
        )
        first = import_skvm_skills(self.root / "skvm", self.store)
        second = import_skvm_skills(self.root / "skvm", self.store)
        self.assertEqual(first.inserted, 1)
        self.assertEqual(second.skipped, 1)
        skills = self.store.list_skills(kind="raw")
        self.assertEqual(skills[0].status, SkillStatus.IMPORTED)
        self.assertTrue(skills[0].metadata["android_relevant"])

    def test_maintenance_discovers_and_promotes_candidate(self):
        config = MaintenanceConfig(
            minimum_support=2,
            minimum_candidate_trials=10,
            minimum_subsequence_length=2,
            maximum_subsequence_length=2,
            promotion_success_rate=0.7,
        )
        for index in range(2):
            trace = ExecutionTrace.new(
                "goal",
                f"task-{index}",
                True,
                [
                    TraceEvent(
                        0,
                        "model",
                        None,
                        ("action.click", "reason.verify"),
                        True,
                        10,
                    )
                ],
            )
            self.store.append_trace(trace)
        maintainer = SkillMaintainer(self.store, config)
        report = maintainer.run_cycle()
        self.assertEqual(len(report.candidates_created), 1)
        skill_id = report.candidates_created[0]
        for trial in range(10):
            self.store.record_skill_trial(skill_id, "model", trial < 9, 10)
        promoted, rolled_back = maintainer.promote_and_rollback()
        self.assertEqual(promoted, [skill_id])
        self.assertEqual(rolled_back, [])
        self.assertEqual(self.store.get_skill(skill_id).status, SkillStatus.ACTIVE)

    def test_cloud_compiler_gates_raw_skill_before_planning(self):
        skill_dir = self.root / "skvm" / "mobile-helper"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Mobile Helper\ndescription: generic helper\n---\nbody\n",
            encoding="utf-8",
        )
        import_skvm_skills(self.root / "skvm", self.store)

        class FakeClient:
            model_id = "cloud-test"

            def generate(self, prompt, **kwargs):
                return SimpleNamespace(
                    text=(
                        '{"android_relevant":true,'
                        '"primitives":["reason.intent","action.click","reason.verify"],'
                        '"adapted_instruction":"点击目标并校验",'
                        '"reason":"适用于 UI 操作"}'
                    )
                )

        summary = compile_imported_raw_skills(
            self.store,
            LLMRawSkillCompiler(FakeClient(), load_primitives()),
            limit=1,
        )
        self.assertEqual(summary.approved, 1)
        compiled = self.store.list_skills(kind="raw")[0]
        self.assertTrue(compiled.metadata["approved_for_planning"])
        self.assertEqual(
            compiled.topology.primitive_sequence(),
            ("reason.intent", "action.click", "reason.verify"),
        )


if __name__ == "__main__":
    unittest.main()
