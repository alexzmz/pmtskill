"""collect 最终 TSR/任务/原语汇总测试。"""

from __future__ import annotations

import unittest

from src1.pmtskill_v2.offline.reporting import summarize_collection


class CollectionReportingTest(unittest.TestCase):
    def test_reports_task_and_primitive_success_rates(self):
        episodes = [
            {"task_template": "TaskA", "is_successful": 1.0, "episode_length": 3, "run_time": 2.0, "exception_info": None, "aux_data": None},
            {"task_template": "TaskA", "is_successful": 0.0, "episode_length": 50, "run_time": 4.0, "exception_info": None, "aux_data": {"collector_termination_reason": "max_steps"}},
            {
                "task_template": "TaskB",
                "is_successful": 1.0,
                "episode_length": 2,
                "run_time": 1.0,
                "exception_info": None,
                "aux_data": {
                    "permission_controller_dialogs_dismissed": 1,
                    "permission_controller_model_delegations": 2,
                },
            },
            {"task_template": "TaskB", "is_successful": 0.0, "episode_length": 0, "run_time": 0.1, "exception_info": "emulator error", "aux_data": None},
        ]
        full_episodes = [
            {"is_successful": 1.0, "episode_data": {"action_output": ['Reason: visible\nAction: {"action_type":"click","index":1}']}},
            {"is_successful": 0.0, "episode_data": {"action_output": ['Reason: retry\nAction: {"action_type":"click","index":2}']}},
            {"is_successful": 1.0, "episode_data": {"action_output": ['Reason: launch\nAction: {"action_type":"open_app","app_name":"Clock"}']}},
        ]

        summary = summarize_collection(
            episodes, full_episodes=full_episodes, episode_step_limit=50
        )

        self.assertEqual(summary["episodes_total"], 4)
        self.assertEqual(summary["episodes_evaluated"], 3)
        self.assertEqual(summary["successes"], 2)
        self.assertAlmostEqual(summary["task_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["success_rate_macro"], 0.75)
        self.assertEqual(summary["per_task"]["TaskA"]["episodes_at_step_limit"], 1)
        self.assertEqual(summary["termination_reasons"]["max_steps"], 1)
        self.assertEqual(summary["termination_reasons"]["exception"], 1)
        self.assertEqual(summary["per_primitive"]["action.click"]["trials"], 2)
        self.assertEqual(summary["per_primitive"]["action.click"]["successes"], 1)
        self.assertEqual(summary["per_primitive"]["action.open_app"]["successes"], 1)
        self.assertEqual(summary["permission_controller_dialogs_dismissed"], 1)
        self.assertEqual(summary["permission_controller_model_delegations"], 2)


if __name__ == "__main__":
    unittest.main()
