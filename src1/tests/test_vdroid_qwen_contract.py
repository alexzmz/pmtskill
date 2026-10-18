from __future__ import annotations

import ast
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import traceback
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.baseline_vdroid.config import load_vdroid_config
from src1.baseline_vdroid.evaluator import VDroidBaselineEvaluator, VDroidRunOptions
from src1.baseline_vdroid.qwen import StrictFixedClient, validate_qwen_agent
from src1.baseline_vdroid.reporting import write_vdroid_report
from src1.baseline_vdroid.validation import VDroidIntegrationError, require_available_evaluation
from src1.baseline_vdroid.worker import _contain_qwen_harness, run_request
from src1.pmtskill_v2.core.models import ModelProfile
from src1.pmtskill_v2.inference.vlm import GenerationResult


class VDroidQwenContractTest(unittest.TestCase):
    def client(self):
        return StrictFixedClient(ModelProfile.from_dict({
            "model_id": "fixed-control", "served_model": "adapter_control01",
            "base_url": "http://unused/v1",
        }))

    def config(self):
        return load_vdroid_config(Path(__file__).parents[1] / "baseline_vdroid/config.example.toml")

    def test_wrapper_has_upstream_model_identity_and_optional_timing_arguments(self):
        client = self.client()
        self.assertEqual(client.model_name, "adapter_control01")
        validate_qwen_agent(types.SimpleNamespace(llm=client))
        result = GenerationResult("done", 1250, {"choices": []}, client.model_id)
        images = [object()]
        with mock.patch.object(client, "generate", return_value=result) as generate:
            self.assertEqual(client.predict_mm("goal", images), ("done", True, result.raw))
            generate.assert_called_once_with("goal", images, max_tokens=2048)
            self.assertEqual(client.predict("goal", measure_time=True), ("done", 1.25, result.raw))
            self.assertEqual(client.predict_mm("goal", images, True), ("done", 1.25, result.raw))

    def test_preflight_rejects_missing_or_invalid_wrapper_contract(self):
        for client in (object(), types.SimpleNamespace(model_name=""),
                       types.SimpleNamespace(model_name="adapter", predict=lambda: None)):
            with self.subTest(client=client), self.assertRaises(VDroidIntegrationError):
                validate_qwen_agent(types.SimpleNamespace(llm=client))

    def test_api_failure_is_not_returned_as_an_action(self):
        client = self.client()
        with mock.patch.object(client, "generate", side_effect=RuntimeError("server unavailable")):
            with self.assertRaisesRegex(VDroidIntegrationError, "server unavailable"):
                client.predict("goal")

    def test_actual_upstream_search_returns_a_completed_path_with_fixed_client(self):
        # Execute the REAL upstream method without importing CUDA/Ray. This
        # exercises the late llm.model_name access that 116 episodes hit.
        root = Path(__file__).parents[2]
        source = next(root / "libs" / name / "android_world/agents/vdroid.py"
                      for name in ("V-Droid", "v-droid")
                      if (root / "libs" / name / "android_world/agents/vdroid.py").is_file())
        tree = ast.parse(source.read_text(encoding="utf-8"))
        agent_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == "VDroidAgent")
        method = next(node for node in agent_class.body if isinstance(node, ast.FunctionDef)
                      and node.name == "search")
        namespace = {"math": math, "trange": lambda count, **kwargs: range(count)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        node = types.SimpleNamespace(is_terminal=True, depth=1, node_info={"summary": "done"},
                                     state={"screenshot_raw": object()})
        agent = types.SimpleNamespace(
            llm=self.client(), root=node, n_iters=1, iter_idx=0, output_strategy="last_iter",
            _reset_and_construct_root=lambda: None, iterate=lambda *args: [node],
        )
        done, output = namespace["search"](agent)
        self.assertTrue(done)
        self.assertEqual(output, [{"summary": "done", "step_number": 1}])

    def test_swallowed_interface_error_stops_after_first_task_and_restores_cwd(self):
        previous = Path.cwd()
        calls = []
        client = self.client()
        del client.model_name

        def episode():
            # Reproduce the exact error in the attached runtime.log.
            return client.model_name

        module = types.SimpleNamespace(run_episode=episode)

        def upstream_task():
            calls.append(1)
            try:
                module.run_episode()
            except Exception:
                return {"exception_info": traceback.format_exc()}

        module._run_task = upstream_task
        with tempfile.TemporaryDirectory() as directory:
            _contain_qwen_harness(module, Path(directory))
            with self.assertRaisesRegex(VDroidIntegrationError, "model_name"):
                for _ in range(116):
                    module._run_task()
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path.cwd(), previous)

    def test_unrelated_task_exception_is_not_misclassified_as_wrapper_failure(self):
        def episode():
            raise TypeError("invalid task action")

        module = types.SimpleNamespace(run_episode=episode)

        def upstream_task():
            try:
                module.run_episode()
            except Exception:
                return {"exception_info": traceback.format_exc()}

        module._run_task = upstream_task
        with tempfile.TemporaryDirectory() as directory:
            _contain_qwen_harness(module, Path(directory))
            for _ in range(2):
                self.assertIn("invalid task action", module._run_task()["exception_info"])

    def test_real_task_failure_does_not_abort_the_suite(self):
        failed = {"exception_info": None, "is_successful": 0.0}
        module = types.SimpleNamespace(run_episode=lambda: None, _run_task=lambda: failed)
        with tempfile.TemporaryDirectory() as directory:
            _contain_qwen_harness(module, Path(directory))
            self.assertIs(module._run_task(), failed)
            self.assertIs(module._run_task(), failed)
        require_available_evaluation({"evaluation_available": True, "episodes_evaluated": 1,
                                      "episodes_total": 1, "success_rate_micro": 0.0})

    def test_all_invalid_report_is_na_and_has_null_json_sr(self):
        episode = {"task_template": "ContactsAddContact", "exception_info": "AttributeError: model_name"}
        metadata = {"model_id": "fixed-control", "assistant_model": "adapter_control01"}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={}):
            artifacts = write_vdroid_report(directory, [episode], metadata=metadata,
                                            store_trajectories=False)
            markdown = artifacts.report_markdown.read_text(encoding="utf-8")
            summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
            self.assertIn("| Micro SR | N/A |", markdown)
            self.assertIn("| Macro SR | N/A |", markdown)
            self.assertIsNone(summary["success_rate_micro"])
            self.assertIsNone(summary["success_rate_macro"])
            with self.assertRaisesRegex(RuntimeError, "not a measured SR=0"):
                require_available_evaluation(summary)

    def test_parent_rejects_zero_exit_worker_with_no_valid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "worker_result.json").write_text(json.dumps({"summary": {
                "evaluation_available": False, "episodes_evaluated": 0, "episodes_total": 116,
            }}), encoding="utf-8")
            options = dataclasses.replace(VDroidRunOptions.from_config(self.config()), output_dir=target)
            process = mock.Mock(stdout=io.StringIO(), stderr=io.StringIO())
            process.wait.return_value = 0
            with mock.patch("src1.baseline_vdroid.evaluator.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "0 valid episodes / 116"):
                    VDroidBaselineEvaluator(self.config(), options).run()

    def test_worker_keeps_unavailable_report_but_never_writes_success_result(self):
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def score_by_batch(self, *args, **kwargs):
                return []

            def _action_completion(self, node):
                return node

            def search(self):
                return False, []

        class Suite(dict):
            pass

        ray = mock.Mock()
        ray.is_initialized.return_value = False
        suite_utils = mock.Mock()
        suite_utils.create_suite.return_value = Suite(Task=[object()])
        registry = mock.Mock()
        registry.TaskRegistry.return_value.get_registry.return_value = {"Task": object()}
        environment = mock.Mock()
        env_launcher = mock.Mock()
        env_launcher.load_and_setup_env.return_value = environment
        vdroid = types.SimpleNamespace(VDroidAgent=FakeAgent)
        package = types.ModuleType("android_world")
        package.registry, package.suite_utils = registry, suite_utils
        agents = types.ModuleType("android_world.agents")
        agents.vdroid = vdroid
        env = types.ModuleType("android_world.env")
        env.env_launcher = env_launcher
        modules = {"ray": ray, "android_world": package,
                   "android_world.agents": agents, "android_world.env": env}
        progress = mock.MagicMock()
        progress.episodes.return_value = [{"task_template": "Task", "exception_info": "ValueError: invalid"}]
        progress.metadata.return_value = {}
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            request = VDroidBaselineEvaluator(self.config(), VDroidRunOptions()).request(target)
            request["v_droid_root"] = directory
            try:
                with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path", sys.path.copy()), \
                     mock.patch("src1.baseline_vdroid.worker._configure_assistant_environment"), \
                     mock.patch("src1.pmtskill_v2.evaluation.resume.EvaluationProgress", return_value=progress), \
                     mock.patch("src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={}):
                    with self.assertRaisesRegex(RuntimeError, "evaluation unavailable"):
                        run_request(request)
            finally:
                os.chdir(previous)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["evaluation_available"])
            self.assertIsNone(summary["success_rate_micro"])
            self.assertFalse((target / "worker_result.json").exists())
            environment.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
