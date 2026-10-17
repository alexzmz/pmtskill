"""Exercise strict AndroidWorld signatures, not permissive **kwargs mocks."""

from __future__ import annotations

import ast
import contextlib
import math
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.core.config import load_config
from src1.pmtskill_v2.evaluation.android_world import _run_evaluation_suite


class EvaluationSuiteCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).parents[1] / "config.example.toml")
        self.config.android_world.infrastructure_recovery_attempts = 0
        self.config.android_world.permission_controller_recovery_attempts = 0
        self.config.android_world.stop_on_task_success = True
        self.environment = types.SimpleNamespace(steps=0)
        self.agent = types.SimpleNamespace()
        self.progress = mock.Mock()
        self.progress.watch.side_effect = lambda *args: contextlib.nullcontext()

    def harness(self, *, success_after=None, budget=100, original_termination=None):
        environment = self.environment
        task = types.SimpleNamespace(is_successful=lambda env: (
            float(env.steps >= success_after) if success_after is not None else 0.0))
        seen = []

        def run_episode(goal, agent, max_n_steps=10, start_on_home_screen=False,
                        termination_fn=None, print_fn=print):
            environment.steps = 0
            seen.append(max_n_steps)
            for _ in range(max_n_steps):
                environment.steps += 1
                if termination_fn is not None and termination_fn(environment):
                    return types.SimpleNamespace(done=True, step_data={
                        "step_number": list(range(environment.steps))})
            return types.SimpleNamespace(done=False, aux_data=None,
                                         step_data={"step_number": list(range(max_n_steps))})

        runner = types.SimpleNamespace(run_episode=run_episode)
        harness = types.SimpleNamespace(episode_runner=runner)

        def run_task(task, callback, env, demo_mode):
            return callback(task)

        harness._run_task = run_task

        # EXACT parameter set in vendored libs/android_world/suite_utils.py.
        def run(suite, agent, checkpointer=None, demo_mode=False,
                return_full_episode_data=False, process_episodes_fn=None, check_episode_fn=None):
            self.assertIs(checkpointer, self.progress)
            self.assertTrue(return_full_episode_data)

            def callback(task):
                return runner.run_episode("goal", agent, budget, False, original_termination)

            return [harness._run_task(item, callback, environment, demo_mode) for item in suite]

        harness.run = run
        return harness, [task], seen

    def evaluate(self, harness, suite, max_steps=30):
        return _run_evaluation_suite(harness, suite, self.agent, self.environment,
                                     self.config, self.progress, max_steps=max_steps)

    def test_strict_legacy_signature_matches_actual_source_not_mock_kwargs(self):
        source = Path(__file__).parents[2] / "libs/android_world/android_world/suite_utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        actual = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "run")
        harness, _, _ = self.harness()
        import inspect
        self.assertIsNone(actual.args.kwarg)
        self.assertEqual([item.arg for item in actual.args.args], list(inspect.signature(harness.run).parameters))

    def test_legacy_signature_runs_and_hard_limit_is_preserved(self):
        harness, suite, seen = self.harness()
        original_task, original_episode = harness._run_task, harness.episode_runner.run_episode
        result = self.evaluate(harness, suite)
        self.assertEqual(seen, [30])
        self.assertEqual(len(result[0].step_data["step_number"]), 30)
        self.assertIs(harness._run_task, original_task)
        self.assertIs(harness.episode_runner.run_episode, original_episode)
        self.progress.watch.assert_called_once_with(harness, suite)

    def test_legacy_limit_never_increases_smaller_native_task_budget(self):
        harness, suite, seen = self.harness(budget=8)
        self.evaluate(harness, suite)
        self.assertEqual(seen, [8])

    def test_legacy_verifier_stops_immediately_on_success(self):
        harness, suite, _ = self.harness(success_after=2)
        result = self.evaluate(harness, suite)
        self.assertTrue(result[0].done)
        self.assertEqual(self.environment.steps, 2)

    def test_legacy_stop_can_be_disabled_without_removing_step_cap(self):
        self.config.android_world.stop_on_task_success = False
        harness, suite, seen = self.harness(success_after=2)
        result = self.evaluate(harness, suite, max_steps=17)
        self.assertEqual(seen, [17])
        self.assertFalse(result[0].done)
        self.assertEqual(self.environment.steps, 17)

    def test_existing_termination_is_preserved_and_short_circuits_verifier(self):
        harness, suite, _ = self.harness(original_termination=lambda env: env.steps == 1)
        suite[0].is_successful = mock.Mock(side_effect=AssertionError("must not call verifier"))
        result = self.evaluate(harness, suite)
        self.assertTrue(result[0].done)
        self.assertEqual(self.environment.steps, 1)
        suite[0].is_successful.assert_not_called()

    def test_non_success_rewards_do_not_terminate_episode(self):
        for value in (0.0, 0.5, math.nan, math.inf, None):
            with self.subTest(value=value):
                harness, suite, _ = self.harness()
                suite[0].is_successful = lambda env: value
                self.assertFalse(self.evaluate(harness, suite, max_steps=3)[0].done)

    def test_hooks_restore_on_real_errors_without_rerunning_suite(self):
        harness, suite, _ = self.harness()
        original_task, original_episode = harness._run_task, harness.episode_runner.run_episode
        suite[0].is_successful = mock.Mock(side_effect=RuntimeError("real verifier failure"))
        with self.assertRaisesRegex(RuntimeError, "real verifier failure"):
            self.evaluate(harness, suite)
        self.assertEqual(suite[0].is_successful.call_count, 1)
        self.assertIs(harness._run_task, original_task)
        self.assertIs(harness.episode_runner.run_episode, original_episode)

    def test_native_newer_interface_receives_original_options(self):
        captured = {}
        def run(suite, agent, checkpointer, demo_mode, return_full_episode_data,
                max_n_steps_override, stop_on_task_success):
            captured.update(limit=max_n_steps_override, stop=stop_on_task_success,
                            checkpointer=checkpointer, full=return_full_episode_data)
            return ["native-result"]
        # No episode_runner attribute: native options require no legacy hooks.
        harness = types.SimpleNamespace(run=run)
        self.assertEqual(self.evaluate(harness, [], max_steps=17), ["native-result"])
        self.assertEqual(captured, {"limit": 17, "stop": True,
                                    "checkpointer": self.progress, "full": True})

    def test_partial_native_support_still_supplies_missing_feature(self):
        harness, suite, _ = self.harness(success_after=2)
        original = harness.run
        def run(suite, agent, checkpointer, demo_mode, return_full_episode_data,
                max_n_steps_override):
            self.assertEqual(max_n_steps_override, 17)
            return original(suite, agent, checkpointer, demo_mode, return_full_episode_data)
        harness.run = run
        self.assertTrue(self.evaluate(harness, suite, max_steps=17)[0].done)
        self.assertEqual(self.environment.steps, 2)


if __name__ == "__main__":
    unittest.main()
