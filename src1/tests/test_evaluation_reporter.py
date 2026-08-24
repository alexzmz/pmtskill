"""evaluate 汇总对异常数值的健壮性测试。"""

from __future__ import annotations

import math
import unittest

from src1.pmtskill_v2.evaluation.reporter import summarize_episodes


class EvaluationReporterTest(unittest.TestCase):
    def test_nan_reward_is_not_reported_as_success(self):
        summary = summarize_episodes(
            [
                {
                    "task_template": "BrokenTask",
                    "is_successful": math.nan,
                    "episode_length": math.nan,
                    "run_time": math.nan,
                    "exception_info": None,
                    "episode_data": {},
                }
            ],
            [],
        )
        self.assertEqual(summary["successes"], 0)
        self.assertEqual(summary["success_rate_micro"], 0.0)
        self.assertEqual(summary["average_steps"], 0.0)
        self.assertEqual(summary["average_run_time_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
