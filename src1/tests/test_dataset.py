"""多模态蒸馏数据转换测试。"""

from __future__ import annotations

import gzip
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src1.pmtskill_v2.offline.dataset import AndroidWorldDistillationDatasetBuilder


def episode(task: str, color: str) -> dict:
    image = Image.new("RGB", (16, 16), color=color)
    return {
        "goal": f"goal for {task}",
        "task_template": task,
        "is_successful": True,
        "episode_data": {
            "action_prompt": ["choose an action"],
            "action_output": [
                'Reason: target is visible\nAction: {"action_type":"click","index":1}'
            ],
            "raw_screenshot": [image],
            "before_screenshot_with_som": [image.copy()],
        },
    }


class DatasetBuilderTest(unittest.TestCase):
    def test_builds_absolute_two_image_samples_and_splits_by_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            with gzip.open(trajectories / "episodes.pkl.gz", "wb") as handle:
                pickle.dump([episode("TaskA", "red"), episode("TaskB", "blue")], handle)
            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0.5, seed=1
            ).build(trajectories)
            self.assertEqual(result.train_samples, 1)
            self.assertEqual(result.validation_samples, 1)
            train = json.loads(result.train_path.read_text(encoding="utf-8").strip())
            self.assertEqual(train["messages"][0]["content"].count("<image>"), 2)
            self.assertEqual(len(train["images"]), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in train["images"]))
            train_episode = train["metadata"]["episode_id"]
            validation = json.loads(
                result.validation_path.read_text(encoding="utf-8").strip()
            )
            self.assertNotEqual(train_episode, validation["metadata"]["episode_id"])

    def test_skips_non_mapping_episode_data_without_touching_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            invalid_failure = {
                "task_template": "FailedTask",
                "goal": "failed before steps were recorded",
                "is_successful": math.nan,
                "episode_data": math.nan,
            }
            invalid_success = {
                "task_template": "MalformedTask",
                "goal": "malformed checkpoint",
                "is_successful": 1.0,
                "episode_data": 1.0,
            }
            with gzip.open(trajectories / "episodes.pkl.gz", "wb") as handle:
                pickle.dump(
                    [episode("TaskA", "red"), invalid_failure, invalid_success],
                    handle,
                )

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0
            ).build(trajectories)

            self.assertEqual(result.train_samples, 1)
            self.assertEqual(result.accepted_episodes, 1)
            self.assertEqual(result.rejected_episodes, 2)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["rejection_reasons"]["filtered_by_success"], 0)
            self.assertEqual(manifest["rejection_reasons"]["invalid_or_empty"], 2)
            self.assertEqual(manifest["episode_outcomes"]["unknown"], 1)

    def test_accepts_single_episode_checkpoint_with_scalar_step_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            item = episode("ScalarTask", "green")
            item["episode_data"] = {
                key: values[0] for key, values in item["episode_data"].items()
            }
            with gzip.open(trajectories / "single.pkl.gz", "wb") as handle:
                pickle.dump(item, handle)

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0
            ).build(trajectories)

            self.assertEqual(result.train_samples, 1)
            self.assertEqual(result.accepted_episodes, 1)

    def test_accepts_step_major_episode_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            item = episode("StepMajorTask", "yellow")
            fields = item["episode_data"]
            item["episode_data"] = [
                {key: values[0] for key, values in fields.items()}
            ]
            with gzip.open(trajectories / "step-major.pkl.gz", "wb") as handle:
                pickle.dump([item], handle)

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0
            ).build(trajectories)

            self.assertEqual(result.train_samples, 1)
            self.assertEqual(result.accepted_episodes, 1)

    def test_only_nan_failure_reports_clear_empty_dataset_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            with gzip.open(trajectories / "failed.pkl.gz", "wb") as handle:
                pickle.dump(
                    [
                        {
                            "task_template": "FailedTask",
                            "is_successful": math.nan,
                            "episode_data": math.nan,
                        }
                    ],
                    handle,
                )

            with self.assertRaisesRegex(
                ValueError,
                r"episode=1.*filtered_by_success=0.*invalid_or_empty=1",
            ):
                AndroidWorldDistillationDatasetBuilder(
                    root / "dataset", validation_ratio=0
                ).build(trajectories)

    def test_includes_failed_episode_when_step_is_learnable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            item = episode("FailedButLearnable", "purple")
            item["is_successful"] = 0.0
            with gzip.open(trajectories / "failed.pkl.gz", "wb") as handle:
                pickle.dump([item], handle)

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0
            ).build(trajectories)

            self.assertEqual(result.train_samples, 1)
            row = json.loads(result.train_path.read_text(encoding="utf-8"))
            self.assertEqual(row["metadata"]["episode_outcome"], "failed")

    def test_successful_only_remains_available_as_opt_in_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            failed = episode("FailedTask", "purple")
            failed["is_successful"] = False
            with gzip.open(trajectories / "mixed.pkl.gz", "wb") as handle:
                pickle.dump([episode("SuccessTask", "green"), failed], handle)

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", successful_only=True, validation_ratio=0
            ).build(trajectories)

            self.assertEqual(result.train_samples, 1)
            self.assertEqual(result.accepted_episodes, 1)
            self.assertEqual(result.rejected_episodes, 1)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["rejection_reasons"]["filtered_by_success"], 1)

    def test_uses_goal_raw_response_and_one_valid_image_as_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectories = root / "trajectories"
            trajectories.mkdir()
            item = episode("FallbackTask", "orange")
            image = item["episode_data"]["raw_screenshot"][0]
            item["is_successful"] = False
            item["episode_data"] = {
                "action_prompt": [None],
                "action_output": [None],
                "action_raw_response": [
                    'Reason: the first attempt failed\nAction: {"action_type":"back"}'
                ],
                "raw_screenshot": [image],
                "before_screenshot_with_som": [math.nan],
            }
            with gzip.open(trajectories / "fallback.pkl.gz", "wb") as handle:
                pickle.dump([item], handle)

            result = AndroidWorldDistillationDatasetBuilder(
                root / "dataset", validation_ratio=0
            ).build(trajectories)

            row = json.loads(result.train_path.read_text(encoding="utf-8"))
            self.assertEqual(result.train_samples, 1)
            self.assertEqual(len(row["images"]), 1)
            self.assertEqual(row["messages"][0]["content"].count("<image>"), 1)
            self.assertEqual(row["metadata"]["prompt_source"], "goal")
            self.assertEqual(row["metadata"]["target_source"], "action_raw_response")


if __name__ == "__main__":
    unittest.main()
