"""长评测的 M3A step 内存压缩测试。"""

from __future__ import annotations

import types
import unittest

from src1.pmtskill_v2.evaluation.compaction import (
    compact_evaluation_step_data,
    compact_m3a_step_result,
)
from src1.pmtskill_v2.evaluation.android_world import episodes_to_traces


class EvaluationCompactionTest(unittest.TestCase):
    def test_drops_images_ui_trees_and_prompts_but_keeps_trace_fields(self):
        raw = {
            "raw_screenshot": bytearray(1024),
            "before_screenshot_with_som": bytearray(1024),
            "after_screenshot_with_som": bytearray(1024),
            "before_ui_elements": [object()],
            "after_ui_elements": [object()],
            "action_prompt": "large action prompt",
            "summary_prompt": "large summary prompt",
            "action_output": '{"action_type":"click"}',
            "action_output_json": {"action_type": "click"},
            "action_reason": "tap the visible button",
            "action_raw_response": {"_pmtskill": {"model_id": "student"}},
            "summary": "button tapped",
        }

        compact = compact_evaluation_step_data(raw)

        self.assertEqual(compact["action_reason"], "tap the visible button")
        self.assertEqual(compact["summary"], "button tapped")
        self.assertIn("action_raw_response", compact)
        self.assertNotIn("raw_screenshot", compact)
        self.assertNotIn("before_ui_elements", compact)
        self.assertNotIn("action_prompt", compact)

    def test_in_place_compaction_also_releases_agent_history_payload(self):
        step_data = {
            "raw_screenshot": bytearray(4096),
            "action_output": "done",
            "summary": "finished",
        }
        history = [step_data]
        result = types.SimpleNamespace(data=step_data)

        returned = compact_m3a_step_result(result)

        self.assertIs(returned, result)
        self.assertIs(history[0], result.data)
        self.assertEqual(result.data, {"action_output": "done", "summary": "finished"})

    def test_compacted_episode_still_produces_routing_trace(self):
        step = compact_evaluation_step_data(
            {
                "raw_screenshot": bytearray(4096),
                "action_output": "{\"action_type\":\"click\"}",
                "action_output_json": {"action_type": "click"},
                "action_raw_response": {
                    "_pmtskill": {
                        "model_id": "student",
                        "skill_id": "open_app",
                        "primitive_ids": ["tap"],
                        "latency_ms": 12.5,
                    }
                },
            }
        )

        traces = episodes_to_traces(
            [
                {
                    "goal": "Open the app",
                    "task_template": "OpenAppTaskEval",
                    "is_successful": True,
                    "run_time": 1.0,
                    "episode_data": [step],
                }
            ]
        )

        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0].events), 1)
        self.assertEqual(traces[0].events[0].model_id, "student")
        self.assertEqual(traces[0].events[0].skill_id, "open_app")
        self.assertEqual(traces[0].events[0].primitive_ids, ("tap",))


if __name__ == "__main__":
    unittest.main()
