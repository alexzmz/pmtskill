"""Configured API standalone reuses M3A without a local deployment or skills."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.cli import build_parser, command_evaluate_standalone
from src1.pmtskill_v2.core.config import AndroidWorldConfig, ProjectConfig, load_config
from src1.pmtskill_v2.core.models import ModelProfile
from src1.pmtskill_v2.inference.vlm import OpenAICompatibleVLClient


class StandaloneAPITest(unittest.TestCase):
    def config(self):
        config = load_config(Path(__file__).parents[1] / "config.example.toml")
        config.android_world = AndroidWorldConfig(console_port=5558, grpc_port=8556)
        config.models = (ModelProfile(
            model_id="teacher-api", served_model="provider-model", base_url="https://provider.invalid/v1",
            capabilities={}, api_key_env="PMTSKILL_TEST_API_KEY",
        ),)
        return config

    def args(self, *extra):
        return build_parser().parse_args([
            "evaluate-standalone", "--api-model-id", "teacher-api",
            "--tasks", "ContactsAddContact", "--output-dir", "unused", *extra,
        ])

    def test_cli_requires_exactly_one_model_source(self):
        parser = build_parser()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            for options in ([], ["--api-model-id", "teacher-api", "--adapter-path", "adapter"]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    parser.parse_args(["evaluate-standalone", *options])
        self.assertIsNone(self.args().adapter_path)
        self.assertIsNone(parser.parse_args(["evaluate-standalone", "--adapter-path", "adapter"]).api_model_id)

    def test_dry_run_needs_no_key_hardware_or_runtime_directories(self):
        config = self.config()
        with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config), mock.patch(
            "src1.pmtskill_v2.cli._adapter_deployment",
        ) as deployment, mock.patch.object(ProjectConfig, "ensure_runtime_dirs") as dirs, mock.patch(
            "src1.pmtskill_v2.evaluation.android_world.AndroidWorldStandaloneEvaluator.run",
        ) as evaluate, mock.patch("src1.pmtskill_v2.cli._print") as output, mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(command_evaluate_standalone(self.args("--dry-run")), 0)
        deployment.assert_not_called()
        dirs.assert_not_called()
        evaluate.assert_not_called()
        payload = output.call_args.args[0]
        self.assertFalse(payload["api_key_available"])
        self.assertFalse(payload["starts_model_server"])
        self.assertFalse(payload["uses_local_model_gpu"])
        self.assertIsNone(payload["deployment_command"])
        self.assertEqual(payload["android_world_connection"], {"console_port": 5558, "grpc_port": 8556})
        self.assertEqual(payload["profiles"][0]["served_model"], "provider-model")

    def test_missing_key_fails_before_emulator_and_does_not_print_a_token(self):
        config = self.config()
        with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config), mock.patch.object(
            ProjectConfig, "ensure_runtime_dirs",
        ) as dirs, mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PMTSKILL_TEST_API_KEY"):
                command_evaluate_standalone(self.args())
        dirs.assert_not_called()

    def test_deployment_only_flags_are_not_silently_ignored(self):
        for flag, value in (("--deploy-port", "8004"), ("--cuda-visible-devices", "2"), ("--infer-backend", "vllm")):
            with self.subTest(flag=flag), mock.patch("src1.pmtskill_v2.cli.load_config", return_value=self.config()):
                with self.assertRaisesRegex(ValueError, "不部署本地模型"):
                    command_evaluate_standalone(self.args(flag, value))

    def test_disabled_and_unknown_models_are_not_replaced_with_a_local_model(self):
        config = self.config()
        for state in ("disabled", "unknown"):
            config.models = ((dataclasses.replace(config.models[0], enabled=False),) if state == "disabled" else ())
            with self.subTest(state=state), mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config):
                with self.assertRaises((ValueError, KeyError)):
                    command_evaluate_standalone(self.args())

    def test_api_uses_common_evaluator_and_reporting_without_deployment(self):
        config = self.config()
        artifacts = types.SimpleNamespace(
            summary={"successes": 1}, summary_json="summary.json", report_markdown="report.md",
            traces_jsonl=None, output_dir=Path("output"),
        )
        with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config), mock.patch.object(
            ProjectConfig, "ensure_runtime_dirs",
        ), mock.patch("src1.pmtskill_v2.cli._adapter_deployment") as deployment, mock.patch(
            "src1.pmtskill_v2.cli._print",
        ) as output, mock.patch.dict(os.environ, {"PMTSKILL_TEST_API_KEY": "test-only-token"}), mock.patch(
            "src1.pmtskill_v2.evaluation.android_world.AndroidWorldStandaloneEvaluator.run", return_value=artifacts,
        ) as evaluate:
            self.assertEqual(command_evaluate_standalone(self.args("--resume", "--no-store-trajectories")), 0)
        deployment.assert_not_called()
        options = evaluate.call_args.kwargs
        self.assertEqual(options["profile"].served_model, "provider-model")
        self.assertEqual(options["profile"].metadata["inference_source"], "configured_api")
        self.assertNotIn("inference_source", config.models[0].metadata)  # TOML profile was not mutated.
        self.assertTrue(options["resume"])
        self.assertFalse(options["store_trajectories"])
        self.assertEqual(options["max_steps"], 30)
        self.assertEqual(options["tasks"], ["ContactsAddContact"])
        result = output.call_args.args[0]
        self.assertEqual(result["adapters"], [])
        self.assertIsNone(result["skill_database"])
        self.assertNotIn("test-only-token", json.dumps(result, default=str))

    def test_api_request_uses_served_model_images_and_env_key(self):
        client = OpenAICompatibleVLClient(self.config().models[0])
        response = {"choices": [{"message": {"content": "Reason: done\nAction: {}"}}]}
        with mock.patch.dict(os.environ, {"PMTSKILL_TEST_API_KEY": "test-only-token"}), mock.patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode()),
        ) as request:
            client.predict_mm("task", [b"test-image"])
        sent = request.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertEqual(sent.full_url, "https://provider.invalid/v1/chat/completions")
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-only-token")
        self.assertEqual(payload["model"], "provider-model")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
