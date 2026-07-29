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


if __name__ == "__main__":
    unittest.main()
