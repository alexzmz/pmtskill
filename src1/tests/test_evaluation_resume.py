"""无需真实 AndroidWorld/GPU 的实例恢复与评测接口回归。"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.baseline_vdroid.config import load_vdroid_config
from src1.baseline_vdroid.evaluator import VDroidBaselineEvaluator, VDroidRunOptions, apply_cli_overrides
from src1.baseline_vdroid.worker import run_request
from src1.pmtskill_v2.cli import _evaluation_resume, build_parser
from src1.pmtskill_v2.core.config import load_config
from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology
from src1.pmtskill_v2.evaluation.android_world import (
    AndroidWorldForcedSkillEvaluator,
    AndroidWorldOnlineEvaluator,
    AndroidWorldSimpleSkillEvaluator,
    AndroidWorldStandaloneEvaluator,
)
from src1.pmtskill_v2.evaluation.resume import (
    EvaluationProgress, model_resume_identity, vdroid_resume_specification,
)
from src1.pmtskill_v2.skills.store import SkillStore


def _task(index=0, *, params=None):
    return types.SimpleNamespace(
        name="ContactsAddContact", goal=f"Add contact {index}",
        params=params or {"task_seed": index, "name": f"Contact {index}"},
    )


def _episode(task, *, success=1.0, exception=None):
    return {
        "goal": task.goal, "task_template": task.name,
        "task_seed": task.params["task_seed"], "is_successful": success,
        "episode_data": {"step_number": [0], "action_output": ['{"action_type":"click","index":1}']},
        "exception_info": exception, "episode_length": 1, "run_time": 2.0,
        "aux_data": None,
    }


def _run(suite_utils, progress, suite):
    # 模拟原生 harness 的 metadata-only 恢复限制。
    if progress.load(fields=["task_template", "instance_id"]):
        raise ValueError("Cannot return full episode data when resuming from a checkpoint.")
    episodes = []
    with progress.watch(suite_utils, suite):
        for instances in suite.values():
            for index, task in enumerate(instances):
                result = suite_utils._run_task(task)
                result["instance_id"] = index
                progress.save_episodes([result], f"{task.name}_{index}")
                episodes.append(result)
    return episodes


class EvaluationResumeTest(unittest.TestCase):
    def test_interruption_resumes_only_completed_instances_and_keeps_failed_outcomes(self):
        tasks = [_task(index) for index in range(3)]
        suite = {tasks[0].name: tasks}
        calls = []

        def original(task):
            calls.append(task.goal)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return _episode(task, success=0.0)

        harness = types.SimpleNamespace(_run_task=original)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            initial = EvaluationProgress(target, {"seed": 42}, resume=False)
            with self.assertRaises(KeyboardInterrupt):
                _run(harness, initial, suite)
            self.assertIs(harness._run_task, original)
            recorded = json.loads(initial.progress_path.read_text(encoding="utf-8"))
            self.assertEqual(recorded["completed_count"], 1)
            self.assertFalse(recorded["completed_instances"][0]["successful"])

            calls.clear()
            harness._run_task = lambda task: (calls.append(task.goal) or _episode(task))
            resumed = EvaluationProgress(target, {"seed": 42}, resume=True)
            episodes = _run(harness, resumed, suite)
            self.assertEqual(calls, [tasks[1].goal, tasks[2].goal])
            self.assertEqual([item["instance_id"] for item in episodes], [0, 1, 2])
            self.assertEqual(episodes[0]["is_successful"], 0.0)
            self.assertEqual(resumed.skipped, 1)
            self.assertEqual(resumed.executed, 2)
            self.assertEqual(len(resumed.episodes()), 3)
            self.assertEqual(initial.trace_id(episodes[0]), resumed.trace_id(episodes[0]))

            calls.clear()
            complete = EvaluationProgress(target, {"seed": 42}, resume=True)
            self.assertEqual(len(_run(harness, complete, suite)), 3)
            self.assertEqual(calls, [])
            self.assertEqual(complete.skipped, 3)

    def test_exceptions_nonfinite_rewards_and_infrastructure_flags_are_retried(self):
        for bad in (
            {"exception_info": "Could not get a11y tree."},
            {"is_successful": float("nan")},
            {"episode_data": float("nan")},
            {"aux_data": {"infrastructure_failure_skipped": True}},
        ):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                suite = {"ContactsAddContact": [_task()]}
                initial = EvaluationProgress(Path(directory), {}, resume=False)
                harness = types.SimpleNamespace(_run_task=lambda task: {**_episode(task), **bad})
                failed = _run(harness, initial, suite)[0]
                runner = mock.Mock(side_effect=_episode)
                harness._run_task = runner
                resumed = EvaluationProgress(Path(directory), {}, resume=True)
                retried = _run(harness, resumed, suite)[0]
                self.assertEqual(retried["is_successful"], 1.0)
                self.assertNotEqual(initial.trace_id(failed), resumed.trace_id(retried))
                runner.assert_called_once()
                self.assertEqual(resumed.skipped, 0)

    def test_corrupt_checkpoint_is_not_counted_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            harness = types.SimpleNamespace(_run_task=_episode)
            suite = {"ContactsAddContact": [_task()]}
            initial = EvaluationProgress(target, {}, resume=False)
            _run(harness, initial, suite)
            (initial.checkpoint_dir / "ContactsAddContact_0.pkl.gz").write_bytes(b"partial")
            with self.assertLogs(level="WARNING"):
                resumed = EvaluationProgress(target, {}, resume=True)
            _run(harness, resumed, suite)
            self.assertEqual(resumed.corrupt, 1)
            self.assertEqual(resumed.executed, 1)
            self.assertEqual(resumed.skipped, 0)

    def test_configuration_mismatch_and_nonresume_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            initial = EvaluationProgress(target, {"mode": "standalone", "seed": 42}, resume=False)
            before = initial.manifest_path.read_bytes()
            for spec, resume in (({"mode": "pmtskill", "seed": 42}, True),
                                 ({"mode": "standalone", "seed": 43}, True),
                                 ({"mode": "standalone", "seed": 42}, False)):
                with self.assertRaises(ValueError):
                    EvaluationProgress(target, spec, resume=resume)
                self.assertEqual(initial.manifest_path.read_bytes(), before)

    def test_changed_instance_parameters_are_detected_before_any_new_task_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            suite = {"ContactsAddContact": [_task()]}
            _run(types.SimpleNamespace(_run_task=_episode), EvaluationProgress(target, {}, resume=False), suite)
            suite["ContactsAddContact"][0].params["name"] = "different"
            runner = mock.Mock(side_effect=_episode)
            with self.assertRaisesRegex(ValueError, "参数发生变化"):
                _run(types.SimpleNamespace(_run_task=runner), EvaluationProgress(target, {}, resume=True), suite)
            runner.assert_not_called()

    def test_legacy_checkpoints_without_manifest_are_rejected_without_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            checkpoint = target / "checkpoints" / "old.pkl.gz"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "缺少 evaluation_manifest"):
                EvaluationProgress(target, {}, resume=True)
            self.assertEqual(checkpoint.read_bytes(), b"old")
            self.assertFalse((target / "evaluation_manifest.json").exists())

    def test_same_checkpoint_path_with_changed_weights_has_different_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            weights = target / "adapter_model.safetensors"
            weights.write_bytes(b"weights-1")
            profile = ModelProfile(model_id="student", served_model="student", base_url="http://local/v1",
                                   capabilities={}, metadata={"evaluation_checkpoint": str(target)})
            before = model_resume_identity(profile)
            weights.write_bytes(b"weights-2")
            self.assertNotEqual(before, model_resume_identity(profile))

    def test_failed_atomic_save_does_not_destroy_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = EvaluationProgress(Path(directory), {}, resume=False)
            episode = _run(types.SimpleNamespace(_run_task=_episode), progress,
                           {"ContactsAddContact": [_task()]})[0]
            checkpoint = progress.checkpoint_dir / "ContactsAddContact_0.pkl.gz"
            before = checkpoint.read_bytes()
            with mock.patch("src1.pmtskill_v2.evaluation.resume.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    progress.save_episodes([episode], "ContactsAddContact_0")
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(list(progress.checkpoint_dir.glob("*.tmp")), [])

    def test_vdroid_parent_rejects_mismatch_before_overwriting_request_or_starting_worker(self):
        config = load_vdroid_config(Path(__file__).parents[1] / "baseline_vdroid" / "config.example.toml")
        with tempfile.TemporaryDirectory() as directory:
            options = apply_cli_overrides(VDroidRunOptions.from_config(config),
                                          output_dir=Path(directory), resume=True)
            evaluator = VDroidBaselineEvaluator(config, options)
            EvaluationProgress(Path(directory), vdroid_resume_specification(evaluator.request(Path(directory))), resume=True)
            request_path = Path(directory) / "resolved_request.json"
            request_path.write_text("old request", encoding="utf-8")
            options.seed += 1
            with mock.patch("src1.baseline_vdroid.evaluator.subprocess.Popen") as popen:
                with self.assertRaisesRegex(ValueError, "配置不一致"):
                    evaluator.run()
                popen.assert_not_called()
            self.assertEqual(request_path.read_text(encoding="utf-8"), "old request")

    def test_cli_and_toml_resume_precedence_and_output_requirement(self):
        config = load_config(Path(__file__).parents[1] / "config.example.toml")
        parser = build_parser()
        for command, adapter in (
            ("evaluate-standalone", ["--adapter-path", "a"]),
            ("evaluate-simple-skills", ["--adapter-path", "a"]),
            ("evaluate-pmtskill", ["--adapter-paths", "a"]),
            ("evaluate", ["--adapter-paths", "a"]),
        ):
            with self.subTest(command=command):
                args = parser.parse_args([command, *adapter])
                self.assertIsNone(args.resume)
                self.assertFalse(_evaluation_resume(config, args))
                config.evaluation.resume = True
                with self.assertRaisesRegex(ValueError, "output-dir"):
                    _evaluation_resume(config, args)
                args = parser.parse_args([command, *adapter, "--no-resume"])
                self.assertFalse(_evaluation_resume(config, args))
                args = parser.parse_args([command, *adapter, "--resume", "--output-dir", "fixed"])
                self.assertTrue(_evaluation_resume(config, args))
                config.evaluation.resume = False
        self.assertTrue(parser.parse_args(["calibrate-skills", "--adapter-paths", "a"]).resume)
        self.assertTrue(parser.parse_args(["evaluate-vdroid", "--resume", "--output-dir", "fixed"]).resume)

    def test_vdroid_resume_config_and_validation(self):
        config = load_vdroid_config(Path(__file__).parents[1] / "baseline_vdroid" / "config.example.toml")
        config.evaluation.resume = True
        options = VDroidRunOptions.from_config(config)
        with self.assertRaisesRegex(ValueError, "output-dir"):
            options.validate()
        options = apply_cli_overrides(options, output_dir=Path("fixed"))
        options.validate()


class EvaluationInterfaceResumeTest(unittest.TestCase):
    """完整调用四个 evaluator，验证旧轨迹合并，而不只单测 skip helper。"""

    def test_all_port_evaluators_resume_full_episodes_and_reports(self):
        for mode, store_trajectories in ((mode, storage) for mode in ("standalone", "simple", "forced", "pmtskill")
                                         for storage in (True, False)):
            with self.subTest(mode=mode, store_trajectories=store_trajectories), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = load_config(Path(__file__).parents[1] / "config.example.toml")
                store = SkillStore(root / "skills.sqlite3")
                store.initialize()
                skill = SkillRecord(
                    skill_id="contact", name="Add contact", description="add contacts",
                    body="Open Contacts and add a contact.", kind="polished",
                    status=SkillStatus.ACTIVE, level=3,
                    topology=SkillTopology.from_sequence(["action.click"]),
                )
                store.upsert_skill(skill)
                profile = ModelProfile(model_id="student", served_model="student", base_url="http://local/v1", capabilities={})
                environment = types.SimpleNamespace(close=mock.Mock())
                def recorded_episode(task):
                    result = _episode(task)
                    result["episode_data"]["action_raw_response"] = [{"_pmtskill": {
                        "model_id": "student", "skill_id": "contact", "primitive_ids": ["action.click"],
                        "action_committed": True,
                    }}]
                    result["episode_data"]["action_output_json"] = [{"action_type": "click", "index": 1}]
                    return result

                runner = mock.Mock(side_effect=recorded_episode)
                harness = types.SimpleNamespace(_run_task=runner)

                class Suite(dict):
                    pass

                def create_suite(*args, n_task_combinations, **kwargs):
                    return Suite({"ContactsAddContact": [_task(i) for i in range(n_task_combinations)]})

                def run_suite(suite, agent, checkpointer, **kwargs):
                    self.assertEqual(checkpointer.load(fields=["task_template"]), [])
                    result = []
                    for instances in suite.values():
                        for i, task in enumerate(instances):
                            item = harness._run_task(task)
                            item["instance_id"] = i
                            checkpointer.save_episodes([item], f"{task.name}_{i}")
                            result.append(item)
                    return result

                harness.create_suite = create_suite
                harness.run = run_suite
                android_world = types.ModuleType("android_world")
                android_world.registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(get_registry=lambda family: {}))
                android_world.suite_utils = harness
                agents = types.ModuleType("android_world.agents")

                class M3A:
                    def __init__(self, *args, **kwargs):
                        pass

                agents.m3a = types.SimpleNamespace(M3A=M3A)
                env_module = types.ModuleType("android_world.env")
                env_module.env_launcher = types.SimpleNamespace(load_and_setup_env=lambda **kwargs: environment)
                modules = {"android_world": android_world, "android_world.agents": agents,
                           "android_world.env": env_module}
                common = dict(tasks=["ContactsAddContact"], n_task_combinations=2,
                              output_dir=root / "run", record_traces=True,
                              store_trajectories=store_trajectories)
                if mode == "standalone":
                    evaluator = AndroidWorldStandaloneEvaluator(config)
                    common.pop("record_traces")
                    common["profile"] = profile
                elif mode == "simple":
                    evaluator = AndroidWorldSimpleSkillEvaluator(config, store)
                    common["profile"] = profile
                elif mode == "forced":
                    evaluator = AndroidWorldForcedSkillEvaluator(config, store)
                    common.pop("record_traces")
                    common.update(profile=profile, skill=skill)
                else:
                    evaluator = AndroidWorldOnlineEvaluator(config, store)
                    common["model_profiles"] = [profile]
                with mock.patch.dict(sys.modules, modules), mock.patch(
                    "src1.pmtskill_v2.evaluation.android_world.bootstrap_android_world"
                ), mock.patch(
                    "src1.pmtskill_v2.evaluation.android_world.recover_infrastructure_failures",
                    return_value=contextlib.nullcontext(),
                ), mock.patch(
                    "src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={},
                ):
                    first = evaluator.run(**common)
                    second = evaluator.run(**common, resume=True)
                self.assertEqual(runner.call_count, 2)
                self.assertEqual(len(second.episodes), 2)
                self.assertEqual(first.summary["success_rate_micro"], second.summary["success_rate_micro"])
                for key in ("model_usage", "skill_usage", "average_model_switches", "failure_reasons"):
                    self.assertEqual(first.summary[key], second.summary[key])
                self.assertEqual(second.summary["metadata"]["resume"]["episodes_skipped"], 2)
                self.assertEqual(second.summary["metadata"]["resume"]["episodes_executed"], 0)
                self.assertIn("复用已完成实例 `2`", second.report_markdown.read_text(encoding="utf-8"))
                if mode in {"simple", "pmtskill"} and store_trajectories:
                    self.assertEqual(len(store.list_traces()), 2)
                    self.assertEqual(store.skill_metrics("contact", "student")["trials"], 2)
                if not store_trajectories:
                    self.assertIsNone(second.traces_jsonl)
                    self.assertFalse((root / "run" / "traces.jsonl").exists())
                    self.assertEqual(list((root / "run" / "checkpoints").glob("*.pkl.gz")), [])
                    snapshots = list((root / "run" / "checkpoints").glob("*.json"))
                    self.assertEqual(len(snapshots), 2)
                    for snapshot in snapshots:
                        text = snapshot.read_text(encoding="utf-8")
                        self.assertNotIn("action_output", text)
                        self.assertNotIn("action_raw_response", text)
                    self.assertEqual(store.list_traces(), [])

    def test_vdroid_worker_resumes_compact_checkpoints_and_combined_report(self):
        self._check_vdroid_worker_resume(store_trajectories=True)

    def test_vdroid_worker_resumes_result_only_checkpoints(self):
        self._check_vdroid_worker_resume(store_trajectories=False)

    def _check_vdroid_worker_resume(self, *, store_trajectories):
        config = load_vdroid_config(Path(__file__).parents[1] / "baseline_vdroid" / "config.example.toml")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            options = apply_cli_overrides(VDroidRunOptions.from_config(config),
                                          output_dir=target, combinations=2, resume=True,
                                          store_trajectories=store_trajectories)
            request = VDroidBaselineEvaluator(config, options).request(target)
            environment = types.SimpleNamespace(close=mock.Mock())
            runner = mock.Mock(side_effect=_episode)
            harness = types.SimpleNamespace(_run_task=runner)

            class Suite(dict):
                pass

            harness.create_suite = lambda *args, n_task_combinations, **kwargs: Suite({
                "ContactsAddContact": [_task(i) for i in range(n_task_combinations)]})

            def run_suite(suite, agent, checkpointer, **kwargs):
                self.assertEqual(checkpointer.load(fields=["task_template"]), [])
                for instances in suite.values():
                    for index, task in enumerate(instances):
                        episode = harness._run_task(task)
                        episode["instance_id"] = index
                        checkpointer.save_episodes([episode], f"{task.name}_{index}")

            harness.run = run_suite
            android_world = types.ModuleType("android_world")
            android_world.registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(get_registry=lambda family: {}))
            android_world.suite_utils = harness
            agents = types.ModuleType("android_world.agents")

            class VDroidAgent:
                def __init__(self, *args, **kwargs):
                    pass

            agents.vdroid = types.SimpleNamespace(VDroidAgent=VDroidAgent)
            env_module = types.ModuleType("android_world.env")
            env_module.env_launcher = types.SimpleNamespace(load_and_setup_env=lambda **kwargs: environment,
                                                           verify_api_level=lambda env: None)
            ray = types.SimpleNamespace(is_initialized=lambda: False)
            modules = {"android_world": android_world, "android_world.agents": agents,
                       "android_world.env": env_module, "ray": ray}
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path", list(sys.path)), mock.patch(
                "src1.baseline_vdroid.worker.os.chdir"
            ), mock.patch(
                "src1.baseline_vdroid.worker._configure_assistant_environment"
            ), mock.patch(
                "src1.baseline_vdroid.worker._instrument_vdroid_agent"
            ), mock.patch(
                "src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={},
            ):
                first = run_request(request)
                second = run_request(request)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(second["summary"]["episodes_total"], 2)
            self.assertEqual(first["summary"]["success_rate_micro"], second["summary"]["success_rate_micro"])
            self.assertEqual(second["summary"]["metadata"]["resume"]["episodes_skipped"], 2)
            if not store_trajectories:
                self.assertIsNone(second["traces_jsonl"])
                self.assertFalse((target / "traces.jsonl").exists())
                self.assertEqual(list((target / "checkpoints").glob("*.pkl.gz")), [])
                self.assertEqual(len(list((target / "checkpoints").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
