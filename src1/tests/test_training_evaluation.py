"""训练期 AndroidWorld 评测编排的无 GPU/无 emulator 测试。"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import (
    AndroidWorldConfig,
    MaintenanceConfig,
    OfflineConfig,
    PathConfig,
    ProjectConfig,
    RoutingConfig,
    TrainingEvaluationConfig,
)
from src1.pmtskill_v2.core.models import ModelProfile
from src1.pmtskill_v2.evaluation.deployment import MSSwiftEvaluationDeployment
from src1.pmtskill_v2.evaluation.reporter import EvaluationArtifacts
from src1.pmtskill_v2.offline.trainer import (
    AdapterJob,
    find_latest_adapter_checkpoint,
)
from src1.pmtskill_v2.offline.training_workflow import (
    TrainingEvaluationOptions,
    TrainingEvaluationWorkflow,
    build_epoch_targets,
)


def _config(root: Path) -> ProjectConfig:
    profile = ModelProfile(
        model_id="student-vl",
        served_model=str(root / "student"),
        base_url="http://127.0.0.1:8002/v1",
        capabilities={"action.click": 0.5},
    )
    return ProjectConfig(
        config_path=root / "config.toml",
        paths=PathConfig(
            repo_root=root,
            android_world_root=root / "android_world",
            skvm_skills_root=root / "skills",
            ms_swift_root=root / "ms-swift",
            state_dir=root / "state",
            log_dir=root / "logs",
        ),
        offline=OfflineConfig(
            teacher_model_id="teacher",
            student_model_path=str(root / "student"),
            trajectory_dir=root / "trajectories",
            dataset_dir=root / "dataset",
            output_dir=root / "outputs",
            epochs=2.0,
        ),
        training_evaluation=TrainingEvaluationConfig(
            enabled=True,
            model_id="student-vl",
        ),
        routing=RoutingConfig(),
        maintenance=MaintenanceConfig(),
        android_world=AndroidWorldConfig(),
        models=(profile,),
    )


class _FakeTrainer:
    def __init__(self):
        self.jobs: list[AdapterJob] = []

    def build_command(self, job: AdapterJob) -> list[str]:
        return ["swift", "sft", "--num_train_epochs", str(job.num_train_epochs)]

    def run(self, job: AdapterJob) -> int:
        self.jobs.append(job)
        step = int(float(job.num_train_epochs or 0) * 10)
        checkpoint = job.output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        return 0


class _FakeDeployment:
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.activated: list[Path | None] = []

    @contextlib.contextmanager
    def activate(self, checkpoint: Path | None):
        self.activated.append(checkpoint)
        yield replace(
            self.profile,
            adapter=str(checkpoint) if checkpoint is not None else None,
        )


class _FakeEvaluator:
    def __init__(self):
        self.calls: list[tuple[bool, str | None]] = []

    def run(self, **kwargs):
        profile = kwargs["profile"]
        use_skills = kwargs["use_skills"]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append((use_skills, profile.adapter))
        epoch = 0.0
        if profile.adapter:
            epoch = int(Path(profile.adapter).name.split("-")[-1]) / 10
        sr = 0.20 + epoch * 0.10 + (0.05 if use_skills else 0.0)
        summary = {
            "episodes_evaluated": 2,
            "successes": int(round(sr * 2)),
            "success_rate_micro": sr,
            "success_rate_macro": sr,
            "average_steps": 3.0,
        }
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.md"
        traces_path = output_dir / "traces.jsonl"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        report_path.write_text("report", encoding="utf-8")
        traces_path.write_text("", encoding="utf-8")
        return EvaluationArtifacts(
            output_dir, summary_path, report_path, traces_path, summary
        )


class TrainingEvaluationTest(unittest.TestCase):
    def test_epoch_targets_include_interval_and_fractional_final(self):
        self.assertEqual(build_epoch_targets(3.0, 1), [1.0, 2.0, 3.0])
        self.assertEqual(build_epoch_targets(5.0, 2), [2.0, 4.0, 5.0])
        self.assertEqual(build_epoch_targets(2.5, 1), [1.0, 2.0, 2.5])
        with self.assertRaisesRegex(ValueError, "正整数"):
            build_epoch_targets(2.0, 0)

    def test_latest_checkpoint_ignores_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "v0" / "checkpoint-10"
            complete.mkdir(parents=True)
            (complete / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-99").mkdir()
            self.assertEqual(find_latest_adapter_checkpoint(root), complete.resolve())

    def test_managed_deployment_loads_checkpoint_server_side(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            swift_cli = root / "ms-swift" / "swift" / "cli" / "main.py"
            swift_cli.parent.mkdir(parents=True)
            swift_cli.write_text("# placeholder", encoding="utf-8")
            checkpoint = root / "checkpoint-10"
            deployment = MSSwiftEvaluationDeployment(
                config, config.training_evaluation, config.models[0]
            )

            command = deployment.build_command(checkpoint)
            self.assertIn("--adapters", command)
            self.assertIn(str(checkpoint.resolve()), command)
            profile = deployment.profile(checkpoint)
            self.assertIsNone(profile.adapter)
            self.assertEqual(
                profile.metadata["evaluation_checkpoint"], str(checkpoint.resolve())
            )
            self.assertIn("checkpoint-10", profile.served_model)

    def test_workflow_runs_fair_baselines_each_epoch_and_final_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            trainer = _FakeTrainer()
            deployment = _FakeDeployment(config.models[0])
            evaluator = _FakeEvaluator()
            job = AdapterJob(
                name="all",
                train_dataset=root / "train.jsonl",
                validation_dataset=None,
                output_dir=root / "unused",
            )
            result = TrainingEvaluationWorkflow(
                config,
                object(),  # fake evaluator 不读取技能库
                trainer,
                deployment=deployment,
                evaluator=evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=root / "run",
                    tasks=("TaskA", "TaskB"),
                    every_epochs=1,
                ),
            )

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(trainer.jobs), 2)
            self.assertIsNone(trainer.jobs[0].resume_from_checkpoint)
            self.assertEqual(
                trainer.jobs[1].resume_from_checkpoint.name, "checkpoint-10"
            )
            self.assertEqual(
                [row["label"] for row in result.stages],
                [
                    "baseline_standalone",
                    "baseline_skills",
                    "epoch_001_standalone",
                    "epoch_002_standalone",
                    "final_skills",
                ],
            )
            history = json.loads(result.history_json.read_text(encoding="utf-8"))
            self.assertEqual(history["status"], "completed")
            self.assertEqual(history["manifest"]["tasks"], ["TaskA", "TaskB"])
            self.assertEqual(
                history["summary"]["final_standalone"]["micro_sr"], 0.4
            )
            self.assertIn("最终模型+技能库 SR", result.comparison_markdown.read_text(encoding="utf-8"))
            self.assertTrue(result.history_csv.is_file())

    def test_cli_keeps_evaluation_opt_in(self):
        parser = build_parser()
        plain = parser.parse_args(["train"])
        enabled = parser.parse_args(
            ["train", "--with-evaluation", "--eval-task-count", "24"]
        )
        disabled = parser.parse_args(["train", "--without-evaluation"])
        self.assertIsNone(plain.with_evaluation)
        self.assertTrue(enabled.with_evaluation)
        self.assertEqual(enabled.eval_task_count, 24)
        self.assertFalse(disabled.with_evaluation)


if __name__ == "__main__":
    unittest.main()
