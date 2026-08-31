"""训练期 AndroidWorld 评测编排的无 GPU/无 emulator 测试。"""

from __future__ import annotations

import base64
import contextlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src1.pmtskill_v2.cli import (
    _training_evaluation_output_dir,
    _training_evaluation_settings,
    build_parser,
)
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
    MSSwiftLoraTrainer,
    find_latest_adapter_checkpoint,
    prepare_training_job,
    staged_training_job,
)
from src1.pmtskill_v2.offline.training_workflow import (
    TrainingEvaluationOptions,
    TrainingEvaluationWorkflow,
    build_epoch_plan,
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
            cuda_visible_devices="2",
        ),
        training_evaluation=TrainingEvaluationConfig(
            enabled=True,
            model_id="student-vl",
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            cuda_visible_devices="1",
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


class _FailSecondStageTrainer(_FakeTrainer):
    def run(self, job: AdapterJob) -> int:
        if len(self.jobs) == 1:
            self.jobs.append(job)
            return 7
        return super().run(job)


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
        self.max_steps: list[int] = []

    def run(self, **kwargs):
        profile = kwargs["profile"]
        use_skills = kwargs["use_skills"]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append((use_skills, profile.adapter))
        self.max_steps.append(kwargs["max_steps"])
        epoch = 0.0
        if profile.adapter:
            epoch = int(Path(profile.adapter).name.split("-")[-1]) / 10
        sr = self.score(epoch, use_skills)
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

    def score(self, epoch: float, use_skills: bool) -> float:
        return 0.20 + epoch * 0.10 + (0.05 if use_skills else 0.0)


class _PlateauEvaluator(_FakeEvaluator):
    """baseline 后三个 epoch 都没有超过 1 个百分点的显著提升。"""

    def score(self, epoch: float, use_skills: bool) -> float:
        standalone = {
            0.0: 0.20,
            1.0: 0.205,
            2.0: 0.210,
            3.0: 0.209,
        }.get(epoch, 0.80)
        return standalone + (0.05 if use_skills else 0.0)


class _InterruptedEvaluator(_FakeEvaluator):
    def run(self, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "partial.marker").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic interrupted evaluation")


class TrainingEvaluationTest(unittest.TestCase):
    def test_epoch_targets_include_interval_and_fractional_final(self):
        self.assertEqual(build_epoch_targets(3.0, 1), [1.0, 2.0, 3.0])
        self.assertEqual(build_epoch_targets(5.0, 2), [2.0, 4.0, 5.0])
        self.assertEqual(build_epoch_targets(2.5, 1), [1.0, 2.0, 2.5])
        with self.assertRaisesRegex(ValueError, "正整数"):
            build_epoch_targets(2.0, 0)

    def test_epoch_plan_merges_evaluation_and_checkpoint_intervals(self):
        plan = build_epoch_plan(5.0, 2, 1)
        self.assertEqual([stage.target_epoch for stage in plan], [1, 2, 3, 4, 5])
        self.assertEqual(
            [stage.evaluate for stage in plan], [False, True, False, True, True]
        )
        self.assertTrue(all(stage.retain_checkpoint for stage in plan))

        final_only = build_epoch_plan(2.0, 1, 0)
        self.assertEqual(
            [stage.retain_checkpoint for stage in final_only], [False, True]
        )

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
            max_len_index = command.index("--vllm_max_model_len")
            memory_index = command.index("--vllm_gpu_memory_utilization")
            self.assertEqual(command[max_len_index + 1], "32768")
            self.assertEqual(command[memory_index + 1], "0.9")
            self.assertEqual(command[command.index("--verbose") + 1], "false")
            self.assertEqual(
                deployment.build_environment()["CUDA_VISIBLE_DEVICES"], "1"
            )
            profile = deployment.profile(checkpoint)
            self.assertIsNone(profile.adapter)
            self.assertEqual(
                profile.metadata["evaluation_checkpoint"], str(checkpoint.resolve())
            )
            self.assertIn("checkpoint-10", profile.served_model)

    def test_training_and_evaluation_have_independent_gpu_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            trainer_environment = MSSwiftLoraTrainer(config).build_environment()
            deployment_environment = MSSwiftEvaluationDeployment(
                config, config.training_evaluation, config.models[0]
            ).build_environment()
            self.assertEqual(trainer_environment["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(deployment_environment["CUDA_VISIBLE_DEVICES"], "1")

    def test_staged_command_never_reloads_dataset_args_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            swift_cli = root / "ms-swift" / "swift" / "cli" / "main.py"
            swift_cli.parent.mkdir(parents=True)
            swift_cli.write_text("# placeholder", encoding="utf-8")
            job = staged_training_job(
                AdapterJob("all", root / "train.jsonl", None, root / "unused"),
                output_dir=root / "run" / "training" / "epoch_002",
                target_epoch=2,
                resume_from_checkpoint=root / "checkpoint-10",
            )
            command = MSSwiftLoraTrainer(config).build_command(job)

            self.assertEqual(command[command.index("--load_args") + 1], "false")
            self.assertEqual(
                command[command.index("--load_data_args") + 1], "false"
            )
            self.assertIn(str((root / "train.jsonl").resolve()), command)

    def test_training_snapshot_rebases_old_image_root_and_filters_only_bad_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset_all"
            image_dir = dataset_dir / "images" / "TaskA"
            image_dir.mkdir(parents=True)
            good = image_dir / "good.png"
            try:
                from PIL import Image
            except ModuleNotFoundError:
                good.write_bytes(
                    base64.b64decode(
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
                    )
                )
            else:
                Image.new("RGB", (2, 2), "white").save(good)
            corrupt = image_dir / "corrupt.png"
            corrupt.write_text("not an image", encoding="utf-8")
            old_root = root / "dataset" / "images" / "TaskA"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "<image>\nvalid"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "images": [str(old_root / "good.png")],
                },
                {
                    "messages": [
                        {"role": "user", "content": "<image>\nbad"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "images": [str(old_root / "corrupt.png")],
                },
                {
                    "messages": [
                        {"role": "user", "content": "<image><image>\npartial"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "images": [
                        str(old_root / "good.png"),
                        str(old_root / "missing.png"),
                    ],
                },
            ]
            train = dataset_dir / "train.jsonl"
            train.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            prepared = prepare_training_job(
                AdapterJob("all", train, None, root / "output"),
                configured_dataset_dir=dataset_dir,
                snapshot_dir=root / "run" / "dataset_snapshot",
            )

            snapshot_rows = [
                json.loads(line)
                for line in prepared.job.train_dataset.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(snapshot_rows), 2)
            self.assertEqual(snapshot_rows[0]["images"], [str(good.resolve())])
            self.assertEqual(snapshot_rows[1]["images"], [str(good.resolve())])
            self.assertTrue(
                snapshot_rows[1]["messages"][0]["content"].startswith("<image>\n")
            )
            self.assertEqual(prepared.manifest["train"]["rejected_rows"], 1)
            self.assertEqual(
                prepared.manifest["train"]["rebased_image_references"], 2
            )

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
            job.train_dataset.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "world"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
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
                    training_cuda_visible_devices="2",
                    evaluation_cuda_visible_devices="1",
                    evaluation_max_model_len=32768,
                    evaluation_gpu_memory_utilization=0.9,
                ),
            )

            self.assertEqual(result.return_code, 0)
            self.assertEqual(set(evaluator.max_steps), {30})
            self.assertEqual(len(trainer.jobs), 2)
            self.assertIsNone(trainer.jobs[0].resume_from_checkpoint)
            self.assertEqual(
                trainer.jobs[1].resume_from_checkpoint.name, "checkpoint-10"
            )
            self.assertEqual(trainer.jobs[0].output_dir.name, "epoch_001")
            self.assertEqual(trainer.jobs[1].output_dir.name, "epoch_002")
            self.assertEqual(
                trainer.jobs[0].train_dataset, trainer.jobs[1].train_dataset
            )
            self.assertIn("dataset_snapshot", str(trainer.jobs[0].train_dataset))
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
                history["manifest"]["resource_assignment"],
                {
                    "training_cuda_visible_devices": "2",
                    "evaluation_cuda_visible_devices": "1",
                    "evaluation_max_model_len": 32768,
                    "evaluation_gpu_memory_utilization": 0.9,
                },
            )
            self.assertEqual(
                history["summary"]["final_standalone"]["micro_sr"], 0.4
            )
            self.assertIn("最终模型+技能库 SR", result.comparison_markdown.read_text(encoding="utf-8"))
            self.assertTrue(result.history_csv.is_file())
            self.assertTrue(
                (root / "run" / "evaluations" / "epoch_001" / "standalone").is_dir()
            )
            checkpoints = json.loads(
                result.checkpoints_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["stage"] for row in checkpoints["checkpoints"]],
                ["epoch_001", "epoch_002"],
            )

    def test_sr_early_stopping_uses_three_epoch_patience_and_one_point_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _config(root)
            config = replace(
                base,
                offline=replace(base.offline, epochs=6.0),
            )
            trainer = _FakeTrainer()
            deployment = _FakeDeployment(config.models[0])
            evaluator = _PlateauEvaluator()
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )
            output_dir = root / "run"

            result = TrainingEvaluationWorkflow(
                config,
                object(),
                trainer,
                deployment=deployment,
                evaluator=evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                    full_evaluation=True,
                    early_stopping_enabled=True,
                    early_stopping_patience=3,
                    early_stopping_min_delta=0.01,
                    # 即使完整评测配置为每 2 epoch，早退仍强制逐 epoch probe。
                    every_epochs=2,
                    checkpoint_every_epochs=0,
                ),
            )

            self.assertEqual(result.return_code, 0)
            self.assertTrue(result.early_stopped)
            self.assertEqual(result.stop_epoch, 3.0)
            self.assertEqual(len(trainer.jobs), 3)
            self.assertEqual(
                [job.num_train_epochs for job in trainer.jobs], [1.0, 2.0, 3.0]
            )
            self.assertEqual(result.final_checkpoint.name, "checkpoint-30")
            self.assertTrue(result.final_checkpoint.is_dir())
            self.assertEqual(
                [row["label"] for row in result.stages],
                [
                    "baseline_standalone",
                    "baseline_skills",
                    "epoch_001_standalone",
                    "epoch_002_standalone",
                    "epoch_003_standalone",
                    "final_skills",
                ],
            )
            history = json.loads(result.history_json.read_text(encoding="utf-8"))
            self.assertEqual(history["status"], "early_stopped")
            self.assertTrue(history["early_stopping"]["stopped"])
            self.assertEqual(history["early_stopping"]["stale_epochs"], 3)
            self.assertEqual(len(history["early_stopping"]["observations"]), 4)
            self.assertEqual(
                history["manifest"]["effective_evaluation_every_epochs"], 1
            )
            self.assertTrue(
                next(
                    row
                    for row in history["stages"]
                    if row["label"] == "epoch_003_standalone"
                )["is_final_checkpoint"]
            )

            # 自动续接一个已经早退的运行时不得重新训练或重复评测。
            resumed_trainer = _FakeTrainer()
            resumed_evaluator = _PlateauEvaluator()
            resumed = TrainingEvaluationWorkflow(
                config,
                object(),
                resumed_trainer,
                deployment=_FakeDeployment(config.models[0]),
                evaluator=resumed_evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                    full_evaluation=True,
                    early_stopping_enabled=True,
                    early_stopping_patience=3,
                    early_stopping_min_delta=0.01,
                    every_epochs=2,
                    checkpoint_every_epochs=0,
                ),
            )
            self.assertTrue(resumed.early_stopped)
            self.assertEqual(resumed_trainer.jobs, [])
            self.assertEqual(resumed_evaluator.calls, [])

    def test_without_full_evaluation_still_runs_standalone_early_stop_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _config(root)
            config = replace(base, offline=replace(base.offline, epochs=6.0))
            trainer = _FakeTrainer()
            evaluator = _PlateauEvaluator()
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )

            result = TrainingEvaluationWorkflow(
                config,
                object(),
                trainer,
                deployment=_FakeDeployment(config.models[0]),
                evaluator=evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=root / "probe-only",
                    tasks=("TaskA",),
                    full_evaluation=False,
                    early_stopping_enabled=True,
                ),
            )

            self.assertTrue(result.early_stopped)
            self.assertEqual(result.stop_epoch, 3.0)
            self.assertEqual(len(trainer.jobs), 3)
            self.assertTrue(all(not use_skills for use_skills, _ in evaluator.calls))
            self.assertEqual(
                [row["label"] for row in result.stages],
                [
                    "baseline_standalone",
                    "epoch_001_standalone",
                    "epoch_002_standalone",
                    "epoch_003_standalone",
                ],
            )
            markdown = result.comparison_markdown.read_text(encoding="utf-8")
            self.assertIn("仅早退探测", markdown)
            self.assertNotIn("模型+技能库", markdown)

    def test_workflow_resumes_latest_checkpoint_without_repeating_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )
            output_dir = root / "run"
            first_trainer = _FailSecondStageTrainer()
            first_evaluator = _FakeEvaluator()
            first = TrainingEvaluationWorkflow(
                config,
                object(),
                first_trainer,
                deployment=_FakeDeployment(config.models[0]),
                evaluator=first_evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                    every_epochs=1,
                ),
            )
            self.assertEqual(first.return_code, 7)
            self.assertEqual(len(first_trainer.jobs), 2)
            self.assertEqual(
                [row["label"] for row in first.stages],
                [
                    "baseline_standalone",
                    "baseline_skills",
                    "epoch_001_standalone",
                ],
            )

            resumed_trainer = _FakeTrainer()
            resumed_evaluator = _FakeEvaluator()
            resumed = TrainingEvaluationWorkflow(
                config,
                object(),
                resumed_trainer,
                deployment=_FakeDeployment(config.models[0]),
                evaluator=resumed_evaluator,
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                    every_epochs=1,
                ),
            )

            self.assertEqual(resumed.return_code, 0)
            self.assertTrue(resumed.resumed)
            self.assertEqual(
                resumed.resumed_from_checkpoint.name,
                "checkpoint-10",
            )
            self.assertEqual(len(resumed_trainer.jobs), 1)
            self.assertEqual(resumed_trainer.jobs[0].output_dir.name, "epoch_002")
            self.assertEqual(
                resumed_trainer.jobs[0].resume_from_checkpoint.name,
                "checkpoint-10",
            )
            self.assertEqual(len(resumed_evaluator.calls), 2)
            self.assertEqual(
                [row["label"] for row in resumed.stages],
                [
                    "baseline_standalone",
                    "baseline_skills",
                    "epoch_001_standalone",
                    "epoch_002_standalone",
                    "final_skills",
                ],
            )
            history = json.loads(resumed.history_json.read_text(encoding="utf-8"))
            self.assertEqual(history["status"], "completed")
            self.assertEqual(len(history["resume_events"]), 1)
            self.assertEqual(history["resume_events"][0]["previous_status"], "failed")

    def test_default_output_dir_reuses_latest_run_for_same_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = AdapterJob("all", root / "train.jsonl", None, root / "adapter")
            runs = job.output_dir / "training_runs"
            completed = runs / "older"
            incomplete = runs / "newer"
            completed.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            (completed / "history.json").write_text(
                json.dumps({"status": "completed", "manifest": {"job": "all"}}),
                encoding="utf-8",
            )
            (incomplete / "history.json").write_text(
                json.dumps({"status": "running", "manifest": {"job": "all"}}),
                encoding="utf-8",
            )
            args = build_parser().parse_args(["train"])

            resolved = _training_evaluation_output_dir(job, args)

            self.assertEqual(resolved, incomplete.resolve())

            for child in incomplete.iterdir():
                child.unlink()
            incomplete.rmdir()
            self.assertEqual(
                _training_evaluation_output_dir(job, args), completed.resolve()
            )

    def test_resume_archives_partial_evaluation_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )
            output_dir = root / "run"
            workflow = TrainingEvaluationWorkflow(
                config,
                object(),
                _FakeTrainer(),
                deployment=_FakeDeployment(config.models[0]),
                evaluator=_InterruptedEvaluator(),
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted evaluation"):
                workflow.run(
                    job,
                    TrainingEvaluationOptions(
                        output_dir=output_dir,
                        tasks=("TaskA",),
                    ),
                )

            resumed = TrainingEvaluationWorkflow(
                config,
                object(),
                _FakeTrainer(),
                deployment=_FakeDeployment(config.models[0]),
                evaluator=_FakeEvaluator(),
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                ),
            )

            self.assertEqual(resumed.return_code, 0)
            archived = list(
                (output_dir / "evaluations" / "baseline").glob(
                    "standalone.interrupted_*"
                )
            )
            self.assertEqual(len(archived), 1)
            self.assertTrue((archived[0] / "partial.marker").is_file())

    def test_completed_run_can_extend_total_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )
            output_dir = root / "run"
            TrainingEvaluationWorkflow(
                config,
                object(),
                _FakeTrainer(),
                deployment=_FakeDeployment(config.models[0]),
                evaluator=_FakeEvaluator(),
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                ),
            )
            extended_config = replace(
                config,
                offline=replace(config.offline, epochs=3.0),
            )
            trainer = _FakeTrainer()
            resumed = TrainingEvaluationWorkflow(
                extended_config,
                object(),
                trainer,
                deployment=_FakeDeployment(extended_config.models[0]),
                evaluator=_FakeEvaluator(),
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=output_dir,
                    tasks=("TaskA",),
                ),
            )

            self.assertEqual(len(trainer.jobs), 1)
            self.assertEqual(trainer.jobs[0].num_train_epochs, 3.0)
            self.assertEqual(
                trainer.jobs[0].resume_from_checkpoint.name,
                "checkpoint-20",
            )
            self.assertEqual(
                [row["label"] for row in resumed.stages],
                [
                    "baseline_standalone",
                    "baseline_skills",
                    "epoch_001_standalone",
                    "epoch_002_standalone",
                    "epoch_002_skills",
                    "epoch_003_standalone",
                    "final_skills",
                ],
            )

    def test_checkpoint_interval_zero_keeps_only_final_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            trainer = _FakeTrainer()
            job = AdapterJob("all", root / "train.jsonl", None, root / "unused")
            job.train_dataset.write_text(
                json.dumps({"messages": []}) + "\n", encoding="utf-8"
            )
            result = TrainingEvaluationWorkflow(
                config,
                object(),
                trainer,
                deployment=_FakeDeployment(config.models[0]),
                evaluator=_FakeEvaluator(),
            ).run(
                job,
                TrainingEvaluationOptions(
                    output_dir=root / "run",
                    tasks=("TaskA",),
                    every_epochs=1,
                    checkpoint_every_epochs=0,
                ),
            )

            manifest = json.loads(
                result.checkpoints_manifest.read_text(encoding="utf-8")
            )["checkpoints"]
            self.assertFalse(manifest[0]["retained"])
            self.assertFalse(manifest[0]["exists"])
            self.assertTrue(manifest[0]["removed_after_completion"])
            self.assertTrue(manifest[1]["retained"])
            self.assertTrue(manifest[1]["exists"])
            self.assertFalse(
                (root / "run" / "training" / "epoch_001" / "checkpoint-10").exists()
            )
            self.assertTrue(result.final_checkpoint.is_dir())

    def test_cli_keeps_evaluation_opt_in(self):
        parser = build_parser()
        plain = parser.parse_args(["train"])
        enabled = parser.parse_args(
            [
                "train",
                "--with-evaluation",
                "--eval-task-count",
                "24",
                "--train-cuda-visible-devices",
                "2",
                "--eval-cuda-visible-devices",
                "1",
                "--eval-max-model-len",
                "32768",
                "--eval-max-steps",
                "19",
                "--checkpoint-every-epochs",
                "2",
            ]
        )
        disabled = parser.parse_args(["train", "--without-evaluation"])
        no_early_stop = parser.parse_args(
            [
                "train",
                "--without-evaluation",
                "--no-early-stopping",
                "--early-stopping-patience",
                "4",
                "--early-stopping-min-delta",
                "0.02",
            ]
        )
        self.assertIsNone(plain.with_evaluation)
        self.assertIsNone(plain.eval_task_count)
        self.assertIsNone(plain.eval_max_steps)
        defaults = _training_evaluation_settings(_config(Path(".")), plain)
        self.assertIsNone(defaults.task_count)
        self.assertEqual(defaults.max_steps, 30)
        self.assertTrue(enabled.with_evaluation)
        self.assertEqual(enabled.eval_task_count, 24)
        self.assertEqual(enabled.train_cuda_visible_devices, "2")
        self.assertEqual(enabled.eval_cuda_visible_devices, "1")
        self.assertEqual(enabled.eval_max_model_len, 32768)
        self.assertEqual(enabled.eval_max_steps, 19)
        self.assertEqual(enabled.checkpoint_every_epochs, 2)
        self.assertTrue(enabled.resume)
        self.assertFalse(build_parser().parse_args(["train", "--no-resume"]).resume)
        self.assertFalse(disabled.with_evaluation)
        self.assertIsNone(plain.early_stopping)
        self.assertFalse(no_early_stop.early_stopping)
        self.assertEqual(no_early_stop.early_stopping_patience, 4)
        self.assertEqual(no_early_stop.early_stopping_min_delta, 0.02)


if __name__ == "__main__":
    unittest.main()
