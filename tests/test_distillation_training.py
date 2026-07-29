from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import distillation_training as training  # noqa: E402


TASK_A = "ContactsAddContact"
TASK_B = "MarkorCreateNote"


def make_episode(
    task: str,
    instance_id: int,
    *,
    success: bool = True,
) -> dict:
    return {
        "task_template": task,
        "instance_id": instance_id,
        "seed": 100 + instance_id,
        "outcome": "success" if success else "failure",
        "score": 1.0 if success else 0.0,
        "is_successful": success,
        "steps": [
            {
                "step_number": 0,
                "action_prompt": f"{task} UI prompt {instance_id}",
                "action_output": (
                    "Reason: inspect and act\n"
                    'Action: {"action_type":"click","index":1}'
                ),
                "summary_prompt": f"{task} summary prompt {instance_id}",
                "summary": f"Clicked the target for episode {instance_id}.",
            }
        ],
    }


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class DatasetPreparationTests(unittest.TestCase):
    def test_successful_trajectories_are_split_by_complete_episode(self):
        report = {
            "episodes": [
                make_episode(TASK_A, 0),
                make_episode(TASK_A, 1),
                make_episode(TASK_A, 2, success=False),
                make_episode(TASK_B, 0),
                make_episode(TASK_B, 1),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            manifest = training.prepare_distillation_dataset(
                report_path,
                root / "dataset",
                selected_tasks=[TASK_A, TASK_B],
                teacher_model=training.DEFAULT_TEACHER_MODEL,
                student_model=training.DEFAULT_STUDENT_MODEL,
                validation_ratio=0.5,
                dataset_seed=7,
                include_summaries=True,
                include_failed_trajectories=False,
                require_success_per_task=True,
                deduplicate_samples=True,
            )

            train_rows = read_jsonl(root / "dataset" / "train.jsonl")
            validation_rows = read_jsonl(
                root / "dataset" / "validation.jsonl"
            )
            self.assertEqual(len(train_rows) + len(validation_rows), 8)
            self.assertEqual(manifest["counts"]["included_failed_episodes"], 0)
            self.assertEqual(manifest["split"]["episode_overlap"], [])
            self.assertTrue(manifest["split"]["train_episode_groups"])
            self.assertTrue(manifest["split"]["validation_episode_groups"])

            train_episode_keys = {
                row["metadata"]["episode_key"] for row in train_rows
            }
            validation_episode_keys = {
                row["metadata"]["episode_key"] for row in validation_rows
            }
            self.assertFalse(train_episode_keys & validation_episode_keys)
            self.assertEqual(
                {row["metadata"]["sample_kind"] for row in train_rows},
                {"action", "summary"},
            )
            all_prompts = [
                row["messages"][0]["content"]
                for row in train_rows + validation_rows
            ]
            self.assertNotIn(f"{TASK_A} UI prompt 2", all_prompts)
            self.assertFalse(manifest["distillation"]["direct_gkd_used"])

    def test_missing_success_for_selected_task_is_reported(self):
        report = {"episodes": [make_episode(TASK_A, 0, success=False)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "No successful teacher trajectory"
            ):
                training.prepare_distillation_dataset(
                    report_path,
                    root / "dataset",
                    selected_tasks=[TASK_A],
                    teacher_model="teacher",
                    student_model="student",
                    validation_ratio=0.1,
                    dataset_seed=42,
                    include_summaries=True,
                    include_failed_trajectories=False,
                    require_success_per_task=True,
                    deduplicate_samples=True,
                )


class CommandTests(unittest.TestCase):
    def test_commands_select_tasks_and_use_raw_prompt_lora_sft(self):
        parser = training.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            args = parser.parse_args(
                [
                    "--tasks",
                    TASK_A,
                    TASK_B,
                    "--run_dir",
                    directory,
                    "--no-merge_lora",
                ]
            )
            training.validate_args(args)
            paths = training.resolve_paths(args)
            tasks = training.resolve_selected_tasks(args)
            collect = training.build_teacher_collection_command(
                args, paths, tasks
            )
            sft = training.build_swift_sft_command(
                args, paths, validation_exists=True
            )

        self.assertIn("--include_prompts", collect)
        tasks_index = collect.index("--tasks")
        self.assertEqual(collect[tasks_index + 1 : tasks_index + 3], tasks)
        self.assertEqual(
            collect[collect.index("--model_path") + 1],
            training.DEFAULT_TEACHER_MODEL,
        )
        self.assertEqual(sft[2], "sft")
        self.assertEqual(sft[sft.index("--model") + 1], training.DEFAULT_STUDENT_MODEL)
        self.assertEqual(sft[sft.index("--tuner_type") + 1], "lora")
        self.assertEqual(
            sft[sft.index("--use_chat_template") + 1], "false"
        )
        self.assertEqual(
            sft[sft.index("--truncation_strategy") + 1], "left"
        )
        self.assertIn("--val_dataset", sft)
        self.assertIn("--logging_dir", sft)
        self.assertEqual(
            Path(sft[sft.index("--logging_dir") + 1]),
            paths.tensorboard_dir / "training",
        )
        self.assertNotIn("--save_total_limit", sft)
        self.assertFalse(args.evaluate_after_training)
        self.assertEqual(args.sr_eval_interval_steps, 0)

    def test_latest_numeric_adapter_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 200, 30):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text(
                    "{}", encoding="utf-8"
                )
            self.assertEqual(
                training.find_latest_adapter(root).name, "checkpoint-200"
            )

    def test_periodic_checkpoint_selection_uses_interval_and_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (50, 100, 150, 200, 230):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text(
                    "{}", encoding="utf-8"
                )
            selected = training.select_periodic_checkpoints(
                root,
                interval_steps=100,
                include_final=True,
            )
            self.assertEqual([step for step, _ in selected], [100, 200, 230])

    def test_full_training_enables_final_and_periodic_evaluation_defaults(self):
        parser = training.build_parser()
        args = parser.parse_args(["--tasks", TASK_A, "--save_steps", "30"])
        training.validate_args(args)
        self.assertTrue(args.evaluate_after_training)
        self.assertEqual(args.sr_eval_interval_steps, 60)

    def test_training_state_summary_exposes_loss_trend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-20"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": 20,
                        "epoch": 2.0,
                        "log_history": [
                            {"step": 5, "loss": 1.2},
                            {"step": 10, "eval_loss": 1.0},
                            {"step": 20, "loss": 0.6},
                            {"step": 20, "eval_loss": 0.7},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = training.summarize_training_state(
                root,
                checkpoint,
                output_json=root / "summary.json",
                output_markdown=root / "summary.md",
            )
            self.assertTrue(summary["train"]["loss_decreased"])
            self.assertEqual(summary["train"]["loss_change"], -0.6)
            self.assertEqual(summary["validation"]["best_loss"], 0.7)
            self.assertIn(
                "DECREASED",
                (root / "summary.md").read_text(encoding="utf-8"),
            )

    def test_merged_model_is_reused_only_for_the_same_adapter(self):
        parser = training.build_parser()
        args = parser.parse_args(["--tasks", TASK_A])
        training.validate_args(args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "checkpoint-10"
            adapter.mkdir()
            merged = root / "merged"
            merged.mkdir()
            (merged / "config.json").write_text("{}", encoding="utf-8")
            (merged / ".distillation_merge.json").write_text(
                json.dumps({"adapter_path": str(adapter.resolve())}),
                encoding="utf-8",
            )
            self.assertFalse(
                training.ensure_merged_adapter(
                    args, adapter, merged, label="test merge"
                )
            )
            other = root / "checkpoint-20"
            other.mkdir()
            with self.assertRaisesRegex(
                FileExistsError, "different adapter"
            ):
                training.ensure_merged_adapter(
                    args, other, merged, label="test merge"
                )


class ComparisonTests(unittest.TestCase):
    def test_sr_comparison_uses_task_runner_metric_names(self):
        teacher = {
            "summary": {
                "task_success_rate": 0.75,
                "macro_task_success_rate": 0.7,
            },
            "breakdown": {
                "by_task": [{"name": TASK_A, "task_success_rate": 0.75}]
            },
        }
        student = {
            "summary": {
                "task_success_rate": 0.5,
                "macro_task_success_rate": 0.5,
            },
            "breakdown": {
                "by_task": [{"name": TASK_A, "task_success_rate": 0.5}]
            },
        }
        baseline = {
            "summary": {
                "task_success_rate": 0.25,
                "macro_task_success_rate": 0.25,
            },
            "breakdown": {
                "by_task": [{"name": TASK_A, "task_success_rate": 0.25}]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_path = root / "teacher.json"
            student_path = root / "student.json"
            baseline_path = root / "baseline.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            student_path.write_text(json.dumps(student), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            comparison = training.build_sr_comparison(
                teacher_path,
                student_path,
                baseline_report_path=baseline_path,
                output_json=root / "comparison.json",
                output_markdown=root / "comparison.md",
            )

            self.assertEqual(
                comparison["overall"]["teacher_micro_sr"], 0.75
            )
            self.assertEqual(
                comparison["overall"]["micro_sr_gain_over_base"], 0.25
            )
            self.assertEqual(comparison["overall"]["micro_sr_gap"], -0.25)
            self.assertIn(
                "75.00%",
                (root / "comparison.md").read_text(encoding="utf-8"),
            )

    def test_checkpoint_sr_history_marks_improvement(self):
        baseline = {
            "summary": {
                "task_success_rate": 0.25,
                "macro_task_success_rate": 0.25,
                "planned_episodes": 4,
                "scored_episodes": 4,
                "successful_episodes": 1,
                "error_episodes": 0,
                "evaluation_coverage": 1.0,
            }
        }
        checkpoint = {
            "summary": {
                "task_success_rate": 0.75,
                "macro_task_success_rate": 0.75,
                "planned_episodes": 4,
                "scored_episodes": 4,
                "successful_episodes": 3,
                "error_episodes": 0,
                "evaluation_coverage": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            checkpoint_path = root / "checkpoint.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            checkpoint_path.write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )
            history = training.write_checkpoint_sr_history(
                baseline_path,
                [
                    {
                        "step": 100,
                        "label": "checkpoint-100",
                        "adapter_path": "adapter",
                        "merged_model_path": "model",
                        "report_path": checkpoint_path,
                    }
                ],
                output_json=root / "history.json",
                output_csv=root / "history.csv",
                output_markdown=root / "history.md",
            )
            row = history["checkpoints"][0]
            self.assertEqual(row["micro_sr_gain_over_base"], 0.5)
            self.assertEqual(row["verdict"], "improved")
            self.assertEqual(history["best_checkpoint"]["step"], 100)
            self.assertIn(
                "IMPROVED",
                (root / "history.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "micro_sr_gain_over_base",
                (root / "history.csv").read_text(encoding="utf-8"),
            )

    def test_comparison_lists_fail_to_success_examples(self):
        shared = {
            "task_template": TASK_A,
            "instance_id": 2,
            "seed": 102,
            "goal": "Add a contact",
            "steps": [{"action_output": "Action: click(1)"}],
        }
        teacher = {
            "summary": {
                "task_success_rate": 1.0,
                "macro_task_success_rate": 1.0,
            },
            "episodes": [{**shared, "is_successful": True, "outcome": "success"}],
        }
        baseline = {
            "summary": {
                "task_success_rate": 0.0,
                "macro_task_success_rate": 0.0,
            },
            "episodes": [{**shared, "is_successful": False, "outcome": "failure"}],
        }
        student = {
            "summary": {
                "task_success_rate": 1.0,
                "macro_task_success_rate": 1.0,
            },
            "episodes": [{**shared, "is_successful": True, "outcome": "success"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, report in (
                ("teacher", teacher),
                ("baseline", baseline),
                ("student", student),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths[name] = path
            comparison = training.build_sr_comparison(
                paths["teacher"],
                paths["student"],
                baseline_report_path=paths["baseline"],
                output_json=root / "comparison.json",
                output_markdown=root / "comparison.md",
            )
            self.assertEqual(comparison["overall"]["verdict"], "improved")
            self.assertEqual(
                comparison["episode_changes"]["improved_count"], 1
            )
            self.assertIn(
                "fail to success",
                (root / "comparison.md").read_text(encoding="utf-8"),
            )

    def test_one_page_result_can_fall_back_to_checkpoint_curve(self):
        parser = training.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            args = parser.parse_args(
                [
                    "--tasks",
                    TASK_A,
                    "--run_dir",
                    directory,
                    "--no-merge_lora",
                ]
            )
            training.validate_args(args)
            paths = training.resolve_paths(args)
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            paths.training_summary_json.write_text(
                json.dumps(
                    {
                        "train": {"first_loss": 1.0, "final_loss": 0.5},
                        "validation": {"best_loss": 0.6},
                    }
                ),
                encoding="utf-8",
            )
            paths.checkpoint_sr_json.write_text(
                json.dumps(
                    {
                        "baseline": {"micro_sr": 0.2},
                        "checkpoints": [
                            {
                                "step": 100,
                                "micro_sr": 0.6,
                                "micro_sr_gain_over_base": 0.4,
                                "verdict": "improved",
                            }
                        ],
                        "best_checkpoint": {
                            "step": 100,
                            "micro_sr": 0.6,
                            "micro_sr_gain_over_base": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = training.build_distillation_result(
                paths,
                include_checkpoint_history=True,
                include_final_comparison=False,
            )
            self.assertEqual(result["verdict"], "improved")
            self.assertEqual(
                result["result_source"], "last_periodic_checkpoint"
            )
            self.assertIn(
                "+40.00 pp",
                paths.result_markdown.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
