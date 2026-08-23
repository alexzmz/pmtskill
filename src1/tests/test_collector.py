"""离线轨迹采集的 episode 硬上限测试。"""

from __future__ import annotations

import math
import types
import unittest

from src1.pmtskill_v2.offline.collector import (
    HARD_EPISODE_STEP_LIMIT,
    _episode_length,
    _successful_episode,
    enforce_episode_step_limit,
    resolve_episode_step_limit,
)


class CollectorStepLimitTest(unittest.TestCase):
    def test_resolves_legacy_unlimited_and_large_values_to_hard_limit(self):
        self.assertEqual(resolve_episode_step_limit(0), HARD_EPISODE_STEP_LIMIT)
        self.assertEqual(resolve_episode_step_limit(1000), HARD_EPISODE_STEP_LIMIT)
        self.assertEqual(resolve_episode_step_limit(30), 30)
        self.assertEqual(resolve_episode_step_limit(50, override=12), 12)
        self.assertEqual(
            resolve_episode_step_limit(50, override=1000), HARD_EPISODE_STEP_LIMIT
        )
        with self.assertRaisesRegex(ValueError, "正整数"):
            resolve_episode_step_limit(50, override=0)

    def test_runtime_guard_caps_each_call_and_is_restored_after_collection(self):
        def run_episode(goal, agent, max_n_steps=1000):
            del goal, agent
            return max_n_steps

        runner = types.SimpleNamespace(run_episode=run_episode)
        suite_utils = types.SimpleNamespace(episode_runner=runner)
        original = runner.run_episode

        with enforce_episode_step_limit(suite_utils, 50):
            self.assertEqual(runner.run_episode("goal", object(), max_n_steps=1000), 50)
            self.assertEqual(runner.run_episode("goal", object(), max_n_steps=20), 20)
            self.assertEqual(runner.run_episode("goal", object()), 50)

        self.assertIs(runner.run_episode, original)
        self.assertEqual(runner.run_episode("goal", object(), max_n_steps=1000), 1000)

    def test_runtime_guard_marks_episode_that_was_stopped_by_limit(self):
        def run_episode(goal, agent, max_n_steps=1000):
            del goal, agent
            return types.SimpleNamespace(
                done=False,
                step_data={"step_number": list(range(max_n_steps))},
                aux_data=None,
            )

        runner = types.SimpleNamespace(run_episode=run_episode)
        suite_utils = types.SimpleNamespace(episode_runner=runner)
        with enforce_episode_step_limit(suite_utils, 50):
            result = runner.run_episode("goal", object(), max_n_steps=1000)

        self.assertEqual(len(result.step_data["step_number"]), 50)
        self.assertEqual(result.aux_data["collector_termination_reason"], "max_steps")
        self.assertEqual(result.aux_data["collector_step_limit"], 50)

    def test_nan_reward_is_not_counted_as_success(self):
        self.assertFalse(_successful_episode(math.nan))
        self.assertFalse(_successful_episode(0.0))
        self.assertTrue(_successful_episode(1.0))
        self.assertEqual(_episode_length(math.nan), 0)
        self.assertEqual(_episode_length(50), 50)


if __name__ == "__main__":
    unittest.main()
