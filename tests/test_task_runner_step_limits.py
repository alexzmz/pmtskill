from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ANDROID_WORLD_ROOT = REPO_ROOT / "libs" / "android_world" / "android_world"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import task_runner_detail  # noqa: E402


def _load_episode_runner_with_stubs():
    constants = ModuleType("android_world.constants")
    constants.STEP_NUMBER = "step_number"

    base_agent = ModuleType("android_world.agents.base_agent")
    base_agent.EnvironmentInteractingAgent = object
    interface = ModuleType("android_world.env.interface")
    interface.AsyncEnv = object

    android_world = ModuleType("android_world")
    android_world.constants = constants
    agents = ModuleType("android_world.agents")
    agents.base_agent = base_agent
    env = ModuleType("android_world.env")
    env.interface = interface
    termcolor = ModuleType("termcolor")
    termcolor.colored = lambda text, _color: text

    module_name = "_test_android_world_episode_runner"
    path = ANDROID_WORLD_ROOT / "episode_runner.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        module_name: module,
        "android_world": android_world,
        "android_world.constants": constants,
        "android_world.agents": agents,
        "android_world.agents.base_agent": base_agent,
        "android_world.env": env,
        "android_world.env.interface": interface,
        "termcolor": termcolor,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class _FakeAgent:
    def __init__(self, done_after: int | None) -> None:
        self.done_after = done_after
        self.call_count = 0
        self.max_steps = "unset"
        self.env = object()

    def reset(self, _start_on_home_screen: bool) -> None:
        return None

    def set_max_steps(self, value) -> None:
        self.max_steps = value

    def step(self, _goal: str):
        self.call_count += 1
        done = (
            self.done_after is not None
            and self.call_count >= self.done_after
        )
        return SimpleNamespace(done=done, data={"call": self.call_count})


class EpisodeRunnerUnlimitedTest(unittest.TestCase):
    def test_none_budget_runs_until_agent_reports_done(self):
        episode_runner = _load_episode_runner_with_stubs()
        agent = _FakeAgent(done_after=3)
        messages: list[str] = []

        result = episode_runner.run_episode(
            "goal",
            agent,
            max_n_steps=None,
            print_fn=messages.append,
        )

        self.assertTrue(result.done)
        self.assertEqual(agent.max_steps, None)
        self.assertEqual(agent.call_count, 3)
        self.assertEqual(result.step_data["step_number"], [0, 1, 2])
        self.assertFalse(any("Reached max" in message for message in messages))

    def test_unlimited_episode_can_stop_on_environment_success(self):
        episode_runner = _load_episode_runner_with_stubs()
        agent = _FakeAgent(done_after=None)

        result = episode_runner.run_episode(
            "goal",
            agent,
            max_n_steps=None,
            termination_fn=lambda _env: agent.call_count >= 2,
            print_fn=lambda _message: None,
        )

        self.assertTrue(result.done)
        self.assertEqual(agent.call_count, 2)
        self.assertEqual(result.step_data["step_number"], [0, 1])


class SharedRunnerStepLimitTest(unittest.TestCase):
    def test_shared_runner_defaults_to_unlimited_and_checks_success(self):
        args = task_runner_detail.build_parser().parse_args([])
        self.assertEqual(args.max_steps_per_episode, 0)
        self.assertTrue(args.stop_on_task_success)

        config = task_runner_detail._base_config(
            args,
            condition="test",
            android_world_root=ANDROID_WORLD_ROOT.parent,
            skill_info=None,
        )
        self.assertEqual(config["suite"]["max_steps_per_episode"], 0)
        self.assertTrue(config["suite"]["stop_on_task_success"])

    def test_negative_step_limit_is_rejected(self):
        args = task_runner_detail.build_parser().parse_args(
            ["--max_steps_per_episode", "-1"]
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            task_runner_detail.validate_args(args)

    def test_suite_budget_resolver_maps_zero_to_none(self):
        source_path = ANDROID_WORLD_ROOT / "suite_utils.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"_allocate_step_budget", "_resolve_step_budget"}
        ]
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(functions, []), str(source_path), "exec"), namespace)

        resolve = namespace["_resolve_step_budget"]
        self.assertEqual(resolve(1.2, None), 12)
        self.assertIsNone(resolve(1.2, 0))
        self.assertEqual(resolve(1.2, 75), 75)
        with self.assertRaises(ValueError):
            resolve(1.2, -1)


if __name__ == "__main__":
    unittest.main()
