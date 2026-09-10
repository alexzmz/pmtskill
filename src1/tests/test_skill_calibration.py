"""模型×技能配对标定、版本隔离和维护策略测试（不启动 GPU/emulator）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src1.pmtskill_v2.calibration import (
    CalibrationOptions,
    SkillCalibrationWorkflow,
    SkillConditionResult,
)
from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import MaintenanceConfig, RoutingConfig
from src1.pmtskill_v2.core.models import (
    ModelProfile,
    SkillRecord,
    SkillStatus,
    SkillTopology,
)
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.skills.identity import model_variant_id, skill_version_id
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore


def _profile(model_id: str, checkpoint: str) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        served_model=model_id,
        base_url="http://127.0.0.1:8002/v1",
        capabilities={"action.click": 0.9, "reason.verify": 0.9},
        metadata={
            "evaluation_checkpoint": checkpoint,
            "base_model_path": "/models/base",
        },
    )


def _skill() -> SkillRecord:
    return SkillRecord(
        skill_id="polished:test:v1",
        name="click-and-verify",
        description="点击后验证",
        kind="polished",
        status=SkillStatus.CANDIDATE,
        level=2,
        topology=SkillTopology.from_sequence(("action.click", "reason.verify")),
        body="点击目标，然后验证界面状态。",
    )


def _episode(task: str, seed: int, success: bool, *, parsed: bool = True):
    return {
        "task_template": task,
        "seed": seed,
        "goal": f"goal-{task}",
        "is_successful": float(success),
        "episode_length": 2,
        "run_time": 0.2,
        "exception_info": None,
        "episode_data": {
            "action_raw_response": [{"response": "x"}],
            "action_output_json": [{"action_type": "click"}] if parsed else [None],
        },
    }


class FakeConditionRunner:
    protocol_id = "skvm-paired-android-world-v1"

    def __init__(self):
        self.no_skill_calls = 0
        self.forced_calls = 0

    def run_no_skill(self, *, profile, tasks, output_dir, **kwargs):
        self.no_skill_calls += 1
        return SkillConditionResult(
            "no-skill",
            output_dir,
            {},
            (
                _episode("TaskA", 1, False),
                _episode("TaskB", 2, True),
            ),
            profile.model_id,
        )

    def run_forced_skill(self, *, profile, skill, tasks, output_dir, **kwargs):
        self.forced_calls += 1
        return SkillConditionResult(
            "forced-skill",
            output_dir,
            {},
            tuple(
                _episode(task, 1 if task == "TaskA" else 2, True)
                for task in (tasks or ("TaskA", "TaskB"))
            ),
            profile.model_id,
            skill.skill_id,
        )


class SkillCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SkillStore(self.root / "skills.sqlite3")
        self.store.initialize()
        self.skill = _skill()
        self.store.upsert_skill(self.skill)

    def tearDown(self):
        self.temporary.cleanup()

    def test_workflow_pairs_conditions_writes_matrix_and_resumes(self):
        runner = FakeConditionRunner()
        profile = _profile("student-a", "/checkpoints/a")
        options = CalibrationOptions(
            output_dir=self.root / "calibration",
            tasks=("TaskA", "TaskB"),
            skill_ids=(self.skill.skill_id,),
        )
        workflow = SkillCalibrationWorkflow(self.store, runner)
        first = workflow.run((profile,), options)

        row = first.summary["model_skill_metrics"][0]
        self.assertEqual(row["valid_trials"], 2)
        self.assertEqual(row["skill_success_rate"], 1.0)
        self.assertEqual(row["baseline_success_rate"], 0.5)
        self.assertEqual(row["success_rate_uplift"], 0.5)
        self.assertTrue(first.matrix_csv.is_file())
        self.assertTrue(first.trials_jsonl.is_file())
        self.assertEqual(runner.no_skill_calls, 1)
        self.assertEqual(runner.forced_calls, 1)

        second = workflow.run((profile,), options)
        self.assertEqual(second.summary["trials_total"], 2)
        self.assertEqual(runner.no_skill_calls, 1)
        self.assertEqual(runner.forced_calls, 1)

    def test_skill_definition_change_does_not_reuse_old_metrics(self):
        profile = _profile("student-a", "/checkpoints/a")
        self.store.start_calibration_run("run", {}, self.root)
        self.store.record_skill_calibration_trial(
            run_id="run",
            pair_id="pair",
            skill=self.skill,
            profile=profile,
            task_name="TaskA",
            task_seed=1,
            baseline_success=False,
            skill_success=True,
        )
        self.assertTrue(
            self.store.calibration_metrics_by_model(self.skill.skill_id)[0]["known"]
        )

        old_version = skill_version_id(self.skill)
        self.skill.body = "这是行为不同的新技能定义。"
        self.store.upsert_skill(self.skill)
        self.assertNotEqual(old_version, skill_version_id(self.skill))
        self.assertEqual(
            self.store.calibration_metrics_by_model(self.skill.skill_id), []
        )

    def test_maintenance_keeps_good_model_pair_and_rejects_bad_pair(self):
        good = _profile("student-good", "/checkpoints/good")
        bad = _profile("student-bad", "/checkpoints/bad")
        self.store.start_calibration_run("run", {}, self.root)
        for index in range(10):
            self.store.record_skill_calibration_trial(
                run_id="run",
                pair_id=f"good-{index}",
                skill=self.skill,
                profile=good,
                task_name="TaskA",
                task_seed=index,
                baseline_success=index < 2,
                skill_success=index < 9,
            )
            self.store.record_skill_calibration_trial(
                run_id="run",
                pair_id=f"bad-{index}",
                skill=self.skill,
                profile=bad,
                task_name="TaskA",
                task_seed=index,
                baseline_success=index < 8,
                skill_success=index < 2,
            )

        maintainer = SkillMaintainer(
            self.store,
            MaintenanceConfig(
                minimum_candidate_trials=10,
                promotion_success_rate=0.7,
                rollback_success_rate=0.45,
                baseline_margin=0.02,
            ),
        )
        promoted, rolled_back = maintainer.promote_and_rollback()
        self.assertEqual(promoted, [self.skill.skill_id])
        self.assertEqual(rolled_back, [])
        active_skill = self.store.get_skill(self.skill.skill_id)
        self.assertEqual(active_skill.status, SkillStatus.ACTIVE)
        self.assertEqual(
            self.store.skill_model_status(active_skill, good)["status"], "active"
        )
        self.assertEqual(
            self.store.skill_model_status(active_skill, bad)["status"], "deprecated"
        )

        topology = SkillTopology.from_sequence(("action.click", "reason.verify"))
        router = DynamicProgrammingRouter(RoutingConfig(), self.store)
        good_plan = router.route("goal", topology, (good,), (active_skill,))
        bad_plan = router.route("goal", topology, (bad,), (active_skill,))
        self.assertEqual(good_plan.steps[0].skill_id, active_skill.skill_id)
        self.assertTrue(all(step.skill_id is None for step in bad_plan.steps))

    def test_cli_exposes_calibration_command(self):
        args = build_parser().parse_args(
            [
                "calibrate-skills",
                "--adapter-paths",
                "adapter-a",
                "--skills",
                self.skill.skill_id,
                "--dry-run",
            ]
        )
        self.assertEqual(args.handler.__name__, "command_calibrate_skills")
        self.assertTrue(args.apply_maintenance)

    def test_model_variant_changes_with_checkpoint_not_service_port(self):
        first = _profile("student", "/checkpoints/epoch-1")
        same = _profile("student", "/checkpoints/epoch-1")
        same.base_url = "http://127.0.0.1:9999/v1"
        newer = _profile("student", "/checkpoints/epoch-2")
        self.assertEqual(model_variant_id(first), model_variant_id(same))
        self.assertNotEqual(model_variant_id(first), model_variant_id(newer))


if __name__ == "__main__":
    unittest.main()
