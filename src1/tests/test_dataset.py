"""多模态蒸馏数据转换测试。"""

from __future__ import annotations

import gzip
import json
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


if __name__ == "__main__":
    unittest.main()
