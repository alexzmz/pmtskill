"""evaluate 汇总对异常数值的健壮性测试。"""

from __future__ import annotations

import math
import unittest

from src1.pmtskill_v2.evaluation.android_world import episodes_to_traces
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

    def test_invalid_episode_data_from_skipped_task_does_not_abort_evaluation(self):
        episode = {
            "task_template": "SimpleSmsResend",
            "is_successful": math.nan,
            "episode_length": math.nan,
            "run_time": math.nan,
            "exception_info": "Invalid element index",
            "episode_data": math.nan,
        }

        traces = episodes_to_traces([episode])
        summary = summarize_episodes([episode], traces)

        self.assertEqual(len(traces), 1)
        self.assertFalse(traces[0].successful)
        self.assertEqual(traces[0].events, ())
        self.assertEqual(traces[0].duration_ms, 0.0)
        self.assertFalse(traces[0].metadata["episode_data_valid"])
        self.assertEqual(summary["episodes_evaluated"], 0)
        self.assertEqual(summary["failure_reasons"], {"exception": 1})

    def test_scalar_and_step_major_trace_fields_are_normalized(self):
        route = {
            "_pmtskill": {
                "model_id": "student",
                "skill_id": "tap",
                "primitive_ids": "action.click",
                "latency_ms": math.nan,
            }
        }
        episode = {
            "task_template": "TaskA",
            "goal": "tap",
            "is_successful": 1.0,
            "run_time": 1.5,
            "episode_data": [
                {
                    "action_raw_response": route,
                    "action_output": '{"action_type":"click"}',
                    "action_output_json": {"action_type": "click"},
                }
            ],
        }

        trace = episodes_to_traces([episode])[0]

        self.assertTrue(trace.successful)
        self.assertEqual(trace.duration_ms, 1500.0)
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(trace.events[0].primitive_ids, ("action.click",))
        self.assertEqual(trace.events[0].latency_ms, 0.0)

    def test_invalid_episode_data_without_exception_is_classified(self):
        episode = {
            "task_template": "BrokenTask",
            "is_successful": 0.0,
            "episode_length": 0,
            "run_time": 1.0,
            "exception_info": None,
            "episode_data": 3.14,
        }

        summary = summarize_episodes([episode], episodes_to_traces([episode]))

        self.assertEqual(summary["episodes_evaluated"], 1)
        self.assertEqual(summary["failure_reasons"], {"invalid_episode_data": 1})


if __name__ == "__main__":
    unittest.main()
