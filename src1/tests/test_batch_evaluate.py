from __future__ import annotations

import copy
import io
import json
import math
import sqlite3
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from src1 import batch_evaluate as batch
from src1.baseline_skvm.evaluator import MultimodalAdaptiveClient, MultimodalSkillProxy
from src1.baseline_vdroid.qwen import QwenVerifier, yes_no_score
from src1.pmtskill_v2.cli import build_parser as cli_parser
from src1.pmtskill_v2.evaluation.resume import vdroid_resume_specification


class BatchEvaluationTest(unittest.TestCase):
    def args(self, output):
        return batch.build_parser().parse_args([
            "--config", "src1/config.example.toml", "--output-dir", str(output)])

    def test_five_commands_use_first_adapter_except_ports_and_stable_profile_ids(self):
        jobs = batch.build_commands(self.args("/tmp/comparison"))
        self.assertEqual([job["method"] for job in jobs], list(batch.METHODS))
        for job in jobs:
            parsed = cli_parser().parse_args(job["command"][3:])
            self.assertEqual((parsed.combinations, parsed.seed, parsed.max_steps), (2, 42, 30))
            self.assertTrue(parsed.resume)
            self.assertFalse(parsed.store_trajectories)
            self.assertEqual(parsed.cuda_visible_devices, "3")
            if job["method"] == "ports":
                self.assertEqual(len(parsed.adapter_paths), 3)
                self.assertEqual(parsed.adapter_model_ids,
                                 ["adapter_control01", "adapter_crossapp01", "adapter_all02"])
            else:
                self.assertEqual(parsed.adapter_path, batch.DEFAULT_ADAPTERS[0])
                self.assertEqual(parsed.model_id, "adapter_control01")
        self.assertIn("evaluate-vdroid-qwen", jobs[3]["command"])

    def test_preview_does_not_touch_missing_config_checkpoints_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = self.args(output)
            args.config, args.dry_run = "missing-linux-config.toml", True
            with mock.patch("builtins.print"):
                self.assertEqual(batch.run(args), 0)
            self.assertFalse(output.exists())

    def test_task_resource_overrides_apply_to_all_methods(self):
        args = self.args("/tmp/comparison")
        args.tasks = ["ContactsAddContact,ClockTimerEntry"]
        args.max_model_len, args.gpu_memory_utilization = 8192, 0.8
        args.include_candidates = True
        for job in batch.build_commands(args):
            parsed = cli_parser().parse_args(job["command"][3:])
            self.assertEqual(parsed.tasks, ["ContactsAddContact", "ClockTimerEntry"])
            self.assertEqual(parsed.max_model_len, 8192)
            self.assertEqual(parsed.gpu_memory_utilization, 0.8)
            if hasattr(parsed, "include_candidates"):
                self.assertTrue(parsed.include_candidates)
                self.assertFalse(parsed.record_traces)

    def test_sqlite_snapshot_includes_wal_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("source.db", "snapshot.db"))
            connection = sqlite3.connect(source)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE skills (id TEXT)")
                connection.execute("INSERT INTO skills VALUES ('hello')")
                connection.commit()
                batch.snapshot_database(source, target)
                snapshot = sqlite3.connect(target)
                try:
                    self.assertEqual(snapshot.execute("SELECT * FROM skills").fetchall(), [("hello",)])
                    snapshot.execute("DELETE FROM skills")
                    snapshot.commit()
                finally:
                    snapshot.close()
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM skills").fetchone()[0], 1)
            finally:
                connection.close()

    def test_failed_partial_results_are_na_not_fake_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            jobs, state = batch.build_commands(self.args(output)), {}
            for index, job in enumerate(jobs):
                folder = Path(job["output_dir"])
                folder.mkdir()
                summary = {"evaluation_available": True, "episodes_total": 4,
                           "episodes_evaluated": 3 if index == 2 else 4, "successes": index,
                           "success_rate_micro": index / 4, "success_rate_macro": index / 4}
                (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                state[job["method"]] = {"status": "failed" if index == 3 else "completed"}
            batch.write_comparison(output, jobs, state, 4)
            rows = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["micro_sr"], 0)  # genuine SR=0 is valid
            self.assertEqual(rows[1]["delta_micro_sr_pp"], 25)
            self.assertIsNone(rows[2]["micro_sr"])
            self.assertIsNone(rows[3]["micro_sr"])
            self.assertEqual(rows[3]["method"], "vdroid-qwen-adapted")

    def test_skvm_forwards_same_images_and_clears_between_calls(self):
        client = mock.Mock()
        client.predict_mm.return_value = ("action", True, {})
        proxy = MultimodalSkillProxy(client)
        class Adaptive:
            def predict(self, prompt):
                return proxy.predict("adapted " + prompt)
        wrapped = MultimodalAdaptiveClient(Adaptive(), proxy)
        images = [object(), object()]
        wrapped.predict_mm("goal", images)
        self.assertIs(client.predict_mm.call_args.args[1], images)
        wrapped.predict("summary")
        self.assertEqual(client.predict_mm.call_args.args[1], [])
        self.assertEqual(proxy.images, [])

    def test_real_legacy_predict_dispatch_preserves_images(self):
        from src1.baseline_skvm.evaluator import legacy_kernel_module
        legacy = legacy_kernel_module()
        images, client = [object()], mock.Mock()
        client.predict_mm.return_value = ("action", True, {})
        proxy = MultimodalSkillProxy(client)
        with tempfile.TemporaryDirectory() as directory:
            adaptive = legacy.SkVMAdaptiveWrapper(
                proxy, variants=[], target_model="vllm/adapter_control01", mode="inject",
                variant_policy="aot", max_skills=2, max_skill_chars=12000,
                trace_path=Path(directory) / "events.jsonl", trace_include_goal=False)
            with mock.patch.object(adaptive, "_adapt", return_value=("adapted", {
                    "injected_chars": 1, "phase": "action", "intent_labels": [],
                    "ui_state_labels": [], "selected": []})):
                MultimodalAdaptiveClient(adaptive, proxy).predict_mm("goal", images)
            client.predict_mm.assert_called_once_with("adapted", images)

    def test_verifier_requires_real_both_label_probabilities(self):
        first = {"token": "Yes", "logprob": math.log(0.6), "top_logprobs": [
            {"token": "Yes", "logprob": math.log(0.6)},
            {"token": "No", "logprob": math.log(0.3)}]}
        raw = {"choices": [{"logprobs": {"content": [first]}}]}
        self.assertAlmostEqual(yes_no_score(raw), 2 / 3)
        first["top_logprobs"] = [{"token": "Yes", "logprob": -1}]
        with self.assertRaisesRegex(RuntimeError, "不伪造零分"):
            yes_no_score(raw)
        with self.assertRaises(RuntimeError):
            yes_no_score({})

    def test_verifier_batch_preserves_order(self):
        scorer = QwenVerifier({"model_id": "fixed", "base_url": "http://unused/v1"}, workers=2)
        with mock.patch.object(scorer, "_score", side_effect=lambda value: (value, None, [value])):
            self.assertEqual(scorer.predict_scores_batch([3, 1, 2]),
                             [(3, None, [3]), (1, None, [1]), (2, None, [2])])

    def test_verifier_http_payload_targets_fixed_adapter_and_disables_thinking(self):
        response = {"choices": [{"logprobs": {"content": [{"token": "Yes", "logprob": -0.2,
                    "top_logprobs": [{"token": "Yes", "logprob": -0.2},
                                     {"token": "No", "logprob": -2.0}]}]}}]}
        scorer = QwenVerifier({"model_id": "fixed", "served_model": "adapter_control01",
                               "base_url": "http://unused/v1"})
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = scorer._score("proposed action")
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://unused/v1/chat/completions")
        self.assertEqual(payload["model"], "adapter_control01")
        self.assertEqual(payload["structured_outputs_regex"], "Yes|No")
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertGreater(result[2][0], 0.5)

    def test_qwen_actor_factory_never_reserves_gpu_or_initializes_native_verifier(self):
        from src1.baseline_vdroid.qwen import install_qwen_backend, StrictFixedClient
        ray, vdroid = mock.Mock(), mock.Mock()
        ray.is_initialized.return_value = False
        captured = []
        remote = mock.Mock()
        def decorate(**kwargs):
            self.assertEqual(kwargs, {"num_gpus": 0})
            def factory(actor):
                captured.append(actor)
                return remote
            return factory
        ray.remote.side_effect = decorate
        install_qwen_backend(vdroid, ray, {"model_id": "adapter_control01"}, workers=2)
        vdroid.ModelActor.options(num_gpus=1)
        remote.options.assert_called_once_with(num_gpus=0)
        self.assertEqual(captured[0]("ignored-service", "ignored-llama").client.model_id, "adapter_control01")
        self.assertIsInstance(vdroid.infer.Gpt4Wrapper("ignored-gpt"), StrictFixedClient)

    def test_skvm_evaluator_uses_real_adaptation_and_common_reporting_without_hardware(self):
        from src1.baseline_skvm.evaluator import run_skvm, legacy_kernel_module
        from src1.pmtskill_v2.core.config import load_config
        from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology
        legacy = legacy_kernel_module()
        config = load_config("src1/config.example.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli_parser().parse_args(["evaluate-skvm", "--adapter-path", directory,
                                            "--output-dir", str(root / "run"), "--resume",
                                            "--no-store-trajectories"])
            args.skvm_cache_dir = str(root / "cache")
            profile = ModelProfile(model_id="adapter_control01", served_model="adapter_control01",
                                   base_url="http://unused/v1", capabilities={})
            skill = SkillRecord(skill_id="test:contact", name="Contacts", description="add contact",
                                kind="polished", status=SkillStatus.ACTIVE, level=2,
                                topology=SkillTopology.from_sequence(["action.click"]), body="Open Contacts.")
            environment, client, store = mock.Mock(), mock.Mock(), mock.Mock()
            store.database = root / "skills.db"
            client.predict_mm.return_value = ("Reason: open\nAction: {}", True, {"_pmtskill": {"model_id": "adapter_control01"}})
            kernel = mock.Mock(cache_dir=root / "cache", target_model="vllm/adapter_control01")
            def prepare():
                source = next((root / "run/skvm/sources").glob("*/SKILL.md"))
                source = legacy.SkillSource(source.parent.name, "Contacts", source, source.read_text(), "hash")
                return [legacy.SkillVariant(source, "aot", "p1p3", source.path, source.content, "hash")]
            kernel.prepare.side_effect = prepare
            kernel.manifest.return_value = {"variants": [], "skills": []}
            class Agent:
                def __init__(self, env, llm, **kwargs):
                    self.llm = llm
            android = types.ModuleType("android_world")
            android.registry = mock.Mock()
            android.suite_utils = mock.Mock()
            android.suite_utils.create_suite.return_value = types.SimpleNamespace()
            class Suite(dict):
                pass
            android.suite_utils.create_suite.return_value = Suite(ContactsAddContact=[object()])
            agents = types.ModuleType("android_world.agents")
            agents.m3a = types.SimpleNamespace(M3A=Agent)
            env = types.ModuleType("android_world.env")
            env.env_launcher = mock.Mock()
            env.env_launcher.load_and_setup_env.return_value = environment
            images = [object()]
            def suite_run(*positional, **kwargs):
                positional[2].llm.predict_mm("The (overall) user goal/request is: add contact\n", images)
                return [{"task_template": "ContactsAddContact", "is_successful": 1,
                         "episode_length": 1, "run_time": 1, "episode_data": {}, "exception_info": None}]
            with mock.patch.dict(sys.modules, {"android_world": android, "android_world.agents": agents,
                                               "android_world.env": env}), mock.patch(
                    "src1.pmtskill_v2.offline.collector.bootstrap_android_world"), mock.patch(
                    "src1.pmtskill_v2.evaluation.android_world.AndroidWorldSimpleSkillEvaluator._skills",
                    return_value=[skill]), mock.patch.object(legacy, "SkVMKernel", return_value=kernel), mock.patch(
                    "src1.pmtskill_v2.inference.vlm.OpenAICompatibleVLClient", return_value=client), mock.patch(
                    "src1.pmtskill_v2.evaluation.android_world._run_evaluation_suite", side_effect=suite_run), mock.patch.object(
                    legacy.skvm_reporting, "write_skvm_evaluation"), mock.patch(
                    "src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={}):
                artifacts = run_skvm(config, store, profile, args, ["ContactsAddContact"])
            self.assertEqual(artifacts.summary["success_rate_micro"], 1)
            self.assertIs(client.predict_mm.call_args.args[1], images)
            self.assertIn("<skvm-adaptation>", client.predict_mm.call_args.args[0])
            self.assertTrue((root / "run/skvm/skill-catalog.json").is_file())
            self.assertFalse((root / "run/skvm/adaptations.jsonl").exists())
            self.assertEqual(artifacts.summary["metadata"]["evaluation_mode"], "skvm_fixed_adapter_aot")

    def test_qwen_resume_ignores_endpoint_not_weights(self):
        request = {"model": {"model_id": "fixed", "cuda_visible_devices": "3"},
                   "assistant": {"model": "fixed", "base_url": "http://localhost:8005/v1"},
                   "qwen_adapter": {"profile": {"base_url": "http://localhost:8005/v1"},
                                    "identity": {"weights": "hash-a"}}}
        changed = copy.deepcopy(request)
        changed["qwen_adapter"]["profile"]["base_url"] = "http://localhost:8006/v1"
        changed["assistant"]["base_url"] = "http://localhost:8006/v1"
        self.assertEqual(vdroid_resume_specification(request), vdroid_resume_specification(changed))
        changed["qwen_adapter"]["identity"]["weights"] = "hash-b"
        self.assertNotEqual(vdroid_resume_specification(request), vdroid_resume_specification(changed))

    def test_entire_batch_and_second_run_skip_completed_methods_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root / "output")
            args.tasks = ["ContactsAddContact"]
            args.adapter_paths = []
            for name in args.adapter_model_ids:
                adapter = root / name
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "/models/base", "r": 16}), encoding="utf-8")
                (adapter / "adapter_model.safetensors").write_bytes(b"fake-weights")
                args.adapter_paths.append(str(adapter))
            source = root / "source.db"
            connection = sqlite3.connect(source)
            connection.execute("CREATE TABLE skills (id TEXT)")
            connection.commit()
            connection.close()
            args.skill_database = str(source)
            calls = []

            def fake_command(command, log_path):
                calls.append(command)
                if "--dry-run" not in command:
                    output = Path(command[command.index("--output-dir") + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "summary.json").write_text(json.dumps({
                        "evaluation_available": True, "episodes_total": 2,
                        "episodes_evaluated": 2, "successes": 1,
                        "success_rate_micro": 0.5, "success_rate_macro": 0.5,
                    }), encoding="utf-8")
                return 0

            with mock.patch.object(batch, "validate_dependencies"), mock.patch.object(
                batch, "run_command", side_effect=fake_command
            ), mock.patch("builtins.print"):
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(len(calls), 10)  # five preflights + five real commands
                calls.clear()
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(calls, [])
            manifest = json.loads((Path(args.output_dir) / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["specification"]["tasks"], ["ContactsAddContact"])
            self.assertTrue((Path(args.output_dir) / "ports" / "skill_library.sqlite3").is_file())

    def test_new_qwen_evaluator_constructs_request_without_llama_or_teacher(self):
        from src1.baseline_vdroid.qwen import evaluator_for_adapter
        from src1.pmtskill_v2.core.config import load_config
        from src1.pmtskill_v2.core.models import ModelProfile
        config = load_config("src1/config.example.toml")
        with tempfile.TemporaryDirectory() as directory:
            args = cli_parser().parse_args([
                "evaluate-vdroid-qwen", "--adapter-path", directory,
                "--output-dir", directory, "--no-store-trajectories", "--resume"])
            profile = ModelProfile(model_id="adapter_control01", base_url="http://localhost:8005/v1",
                                   served_model="adapter_control01", capabilities={}, metadata={
                                       "base_model_path": "/models/qwen", "evaluation_checkpoint": directory})
            evaluator = evaluator_for_adapter(config, profile, args, ["ContactsAddContact"])
            request = evaluator.request(Path(directory))
            self.assertEqual(request["protocol"], "vdroid-qwen-adapted-yes-no-v1")
            self.assertEqual(request["assistant"]["model"], "adapter_control01")
            self.assertEqual(request["model"]["verifier_adapter"], directory)
            self.assertFalse(request["isolation"]["skill_adaptation"])
            self.assertFalse(request["store_trajectories"])

    def test_kernel_report_reads_common_summary_not_missing_legacy_report(self):
        from src1.baseline_skvm.evaluator import legacy_kernel_module
        legacy = legacy_kernel_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({
                "episodes_total": 4, "episodes_evaluated": 4,
                "success_rate_micro": 0.75, "evaluation_available": True,
            }), encoding="utf-8")
            report = legacy.skvm_reporting.write_skvm_evaluation(
                run_dir=root, cache_dir=root / "cache", target_model="vllm/fixed",
                adapter="bare-agent", variants=[], manifest={},
                runtime_stats={"skvm_adapted_requests": 4})
            self.assertEqual(report["performance_validity"]["scored_episodes"], 4)
            self.assertEqual(report["android_world"]["summary"]["task_success_rate"], 0.75)
            self.assertEqual(report["android_world"]["report_path"], str(root / "summary.json"))

    def test_vdroid_harness_contains_logs_and_stops_infrastructure_failures(self):
        from src1.baseline_vdroid.worker import _contain_qwen_harness
        from src1.pmtskill_v2.evaluation.recovery import AndroidWorldInfrastructureError
        previous = Path.cwd()
        module = types.SimpleNamespace(run_episode=lambda: Path.cwd(),
                                       _run_task=lambda: {"exception_info": "Could not get a11y tree."})
        with tempfile.TemporaryDirectory() as directory:
            _contain_qwen_harness(module, Path(directory))
            self.assertEqual(module.run_episode(), Path(directory))
            self.assertEqual(Path.cwd(), previous)
            with self.assertRaises(AndroidWorldInfrastructureError):
                module._run_task()


if __name__ == "__main__":
    unittest.main()
