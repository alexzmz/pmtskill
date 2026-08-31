from __future__ import annotations

import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import load_config
from src1.pmtskill_v2.core.models import (
    ModelProfile,
    SkillRecord,
    SkillStatus,
    SkillTopology,
)
from src1.pmtskill_v2.evaluation.adapters import (
    AdapterDeploymentBinding,
    MSSwiftAdapterDeployment,
    resolve_adapter_checkpoint,
)
from src1.pmtskill_v2.evaluation.android_world import (
    AndroidWorldOnlineEvaluator,
    AndroidWorldSimpleSkillEvaluator,
    AndroidWorldStandaloneEvaluator,
    sample_android_world_tasks,
)
from src1.pmtskill_v2.online.executor import SimpleSkillVLWrapper


def _checkpoint(
    path: Path, *, base_model: str = "/models/base", lora_rank: int = 32
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "r": lora_rank}),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")


class _FakeVLClient:
    def __init__(self):
        self.prompts: list[str] = []

    def predict_mm(self, prompt, images):
        self.prompts.append(prompt)
        return "ok", True, {"_pmtskill": {"model_id": "adapter-a"}}


class EvaluationInterfacesTest(unittest.TestCase):
    def test_example_config_defaults_training_evaluation_to_all_tasks(self):
        config = load_config(Path(__file__).parents[1] / "config.example.toml")

        self.assertEqual(config.training_evaluation.tasks, ())
        self.assertIsNone(config.training_evaluation.task_count)
        self.assertEqual(config.training_evaluation.max_steps, 30)

    def test_resolver_selects_latest_run_last_epoch_best_and_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wifi-adapter"
            old = root / "training_runs" / "20260101T010101" / "training"
            _checkpoint(old / "epoch_009" / "best")
            latest = root / "training_runs" / "20260201T010101" / "training"
            _checkpoint(latest / "epoch_001" / "best")
            expected = latest / "epoch_003" / "best"
            _checkpoint(expected)

            resolved = resolve_adapter_checkpoint(root)

            self.assertEqual(resolved.adapter_root, root.resolve())
            self.assertEqual(resolved.run_dir.name, "20260201T010101")
            self.assertEqual(resolved.epoch_dir.name, "epoch_003")
            self.assertEqual(resolved.checkpoint_dir, expected.resolve())
            self.assertEqual(resolved.base_model_path, "/models/base")
            self.assertEqual(resolved.lora_rank, 32)
            self.assertTrue(resolved.optimizer_state.is_file())
            self.assertTrue(resolved.scheduler_state.is_file())
            self.assertEqual(resolved.selection, "latest_run_last_epoch_best")

    def test_resolver_rejects_incomplete_training_checkpoint_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-10"
            _checkpoint(checkpoint)
            (checkpoint / "scheduler.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "scheduler.pt"):
                resolve_adapter_checkpoint(checkpoint)
            resolved = resolve_adapter_checkpoint(
                checkpoint, require_training_state=False
            )
            self.assertEqual(resolved.checkpoint_dir, checkpoint.resolve())

    def test_multi_adapter_deployment_uses_named_ms_swift_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir = root / "first"
            second_dir = root / "second"
            _checkpoint(first_dir)
            _checkpoint(second_dir)
            first = resolve_adapter_checkpoint(first_dir)
            second = resolve_adapter_checkpoint(second_dir)
            config = load_config(Path(__file__).parents[1] / "config.example.toml")
            profile = ModelProfile(
                model_id="template",
                served_model="unused",
                base_url="http://unused/v1",
                capabilities={"action.click": 0.8},
            )
            deployment = MSSwiftAdapterDeployment(
                config,
                config.training_evaluation,
                (
                    AdapterDeploymentBinding("route-a", first, profile),
                    AdapterDeploymentBinding("route-b", second, profile),
                ),
            )

            command = deployment.build_adapter_command()
            adapter_index = command.index("--adapters")
            self.assertEqual(
                command[adapter_index + 1 : adapter_index + 3],
                [
                    f"route-a={first.checkpoint_dir}",
                    f"route-b={second.checkpoint_dir}",
                ],
            )
            self.assertIn("--model", command)
            rank_index = command.index("--vllm_max_lora_rank")
            self.assertEqual(command[rank_index + 1], "32")
            self.assertEqual(
                [item.served_model for item in deployment.profiles()],
                ["route-a", "route-b"],
            )

    def test_simple_skill_wrapper_injects_one_skill_without_dynamic_route(self):
        client = _FakeVLClient()
        skill = SkillRecord(
            skill_id="wifi-on",
            name="Turn on wifi",
            description="enable wifi",
            kind="polished",
            status=SkillStatus.ACTIVE,
            level=2,
            topology=SkillTopology.from_sequence(("action.click",)),
            body="Open settings and enable the Wi-Fi switch.",
        )
        wrapper = SimpleSkillVLWrapper(client, [skill])
        wrapper.set_goal("Turn on wifi")

        _, _, raw = wrapper.predict_mm("choose next action", [])

        self.assertIn("简单技能提示", client.prompts[-1])
        self.assertEqual(raw["_pmtskill"]["skill_id"], "wifi-on")
        self.assertEqual(
            raw["_pmtskill"]["routing_mode"], "simple_keyword_skill"
        )
        wrapper.predict_mm("Summerize the latest step", [])
        self.assertNotIn("简单技能提示", client.prompts[-1])

    def test_cli_exposes_three_distinct_evaluation_interfaces(self):
        parser = build_parser()
        standalone = parser.parse_args(
            ["evaluate-standalone", "--adapter-path", "adapter-a"]
        )
        simple = parser.parse_args(
            ["evaluate-simple-skills", "--adapter-path", "adapter-a"]
        )
        online = parser.parse_args(
            [
                "evaluate-pmtskill",
                "--adapter-paths",
                "adapter-a",
                "adapter-b",
                "--adapter-model-ids",
                "grounding",
                "planning",
                "--routing-switch-weight",
                "0.2",
                "--record-traces",
            ]
        )
        legacy_alias = parser.parse_args(
            ["evaluate", "--adapter-paths", "adapter-a"]
        )

        self.assertEqual(standalone.handler.__name__, "command_evaluate_standalone")
        self.assertEqual(simple.handler.__name__, "command_evaluate_simple_skills")
        self.assertEqual(online.handler.__name__, "command_evaluate_pmtskill")
        self.assertEqual(online.adapter_model_ids, ["grounding", "planning"])
        self.assertEqual(online.routing_switch_weight, 0.2)
        self.assertTrue(online.record_traces)
        self.assertEqual(legacy_alias.handler.__name__, "command_evaluate_pmtskill")
        self.assertIsNone(standalone.tasks)
        self.assertIsNone(standalone.task_count)
        self.assertEqual(standalone.max_steps, 30)
        self.assertEqual(simple.max_steps, 30)
        self.assertEqual(online.max_steps, 30)
        self.assertEqual(legacy_alias.max_steps, 30)

        limited = parser.parse_args(
            [
                "evaluate-standalone",
                "--adapter-path",
                "adapter-a",
                "--task-count",
                "12",
                "--max-steps",
                "17",
            ]
        )
        self.assertEqual(limited.task_count, 12)
        self.assertEqual(limited.max_steps, 17)

    def test_all_evaluator_apis_default_to_thirty_steps(self):
        for evaluator in (
            AndroidWorldStandaloneEvaluator,
            AndroidWorldSimpleSkillEvaluator,
            AndroidWorldOnlineEvaluator,
        ):
            parameter = inspect.signature(evaluator.run).parameters["max_steps"]
            self.assertEqual(parameter.default, 30)

    def test_task_sampler_returns_all_tasks_when_count_is_omitted(self):
        fake_registry = types.SimpleNamespace(
            TaskRegistry=lambda: types.SimpleNamespace(
                get_registry=lambda family: {
                    "TaskC": object(),
                    "TaskA": object(),
                    "TaskB": object(),
                }
            )
        )
        fake_android_world = types.ModuleType("android_world")
        fake_android_world.registry = fake_registry
        with mock.patch(
            "src1.pmtskill_v2.evaluation.android_world.bootstrap_android_world"
        ), mock.patch.dict(sys.modules, {"android_world": fake_android_world}):
            tasks = sample_android_world_tasks(
                mock.Mock(),
                tasks=None,
                task_count=None,
                seed=42,
            )

        self.assertEqual(tasks, ["TaskA", "TaskB", "TaskC"])


if __name__ == "__main__":
    unittest.main()
