"""No emulator/GPU: paired representations, frozen parameters, exact counters."""
from __future__ import annotations

import copy
import dataclasses
import io
import json
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.cli import build_parser, command_evaluate_skill_effectiveness
from src1.pmtskill_v2.core.config import ProjectConfig, load_config
from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology, TopologyNode
from src1.pmtskill_v2.evaluation.skill_effectiveness import (
    CONDITIONS, SkillPair, load_pairs, prepare_inputs, run_experiment, write_report,
)
from src1.pmtskill_v2.evaluation.skill_effectiveness_runner import (
    METRICS, EffectivenessWrapper, compact_result, frozen_suite, record_task_metrics,
)
from src1.pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
from src1.pmtskill_v2.skills.store import SkillStore


def make_pair():
    raw = SkillRecord("raw:add", "RAW", "raw original", "raw", SkillStatus.IMPORTED, 1,
                      SkillTopology.from_sequence(["action.click"]), body="RAW_ORIGINAL_GUIDANCE")
    polished = SkillRecord("polished:add", "POLISHED", "polished", "polished", SkillStatus.ACTIVE, 3,
                           SkillTopology.from_sequence(["action.click", "action.type", "action.click"]),
                           body="POLISHED_ONLY_GUIDANCE", metadata={"calibration_tasks": ["ContactsAddContact"]})
    return SkillPair("add-contact", raw, polished, ("ContactsAddContact",))


def profile():
    return ModelProfile("control", "control", "http://unused/v1", {})


def fake_row(condition, *, instance=0, valid=True, success=True):
    return {"pair_id": "add-contact", "condition": condition, "task": "ContactsAddContact",
            "instance_id": instance, "suite_hash": "same", "goal": "same objective", "valid": valid,
            "success": success if valid else None, "interaction_length": 3,
            "model_invocations": 5, "action_calls": 3, "summary_calls": 2,
            "latency_seconds": 9.0, "model_latency_seconds": 6.0}


class Client:
    def __init__(self):
        self.http_request_count = 0
        self.prompts = []

    def predict_mm(self, prompt, images):
        self.http_request_count += 1
        self.prompts.append(prompt)
        return "ok", True, {}


class Suite(dict):
    pass


class Task:
    name = "ContactsAddContact"

    def __init__(self, params):
        self.params = params

    @property
    def goal(self):
        return "Add contact " + self.params["name"]


class SkillEffectivenessTest(unittest.TestCase):
    def config(self):
        config = load_config("src1/config.example.toml")
        config.android_world = dataclasses.replace(config.android_world, stop_on_task_success=False,
                                                   infrastructure_recovery_attempts=0,
                                                   permission_controller_recovery_attempts=0)
        return config

    def store(self, root):
        store = SkillStore(root / "source.sqlite3")
        store.initialize()
        pair = make_pair()
        store.upsert_skill(pair.raw)
        store.upsert_skill(pair.polished)
        return store

    def test_cli_exposes_prepare_and_three_arm_experiment_with_no_trajectory_default(self):
        args = build_parser().parse_args(["evaluate-skill-effectiveness", "--adapter-path", "adapter",
                                          "--pairs-file", "pairs.json", "--output-dir", "out", "--resume"])
        self.assertEqual(args.handler.__name__, "command_evaluate_skill_effectiveness")
        self.assertFalse(args.store_trajectories)
        self.assertTrue(args.resume)
        args = build_parser().parse_args(["prepare-skill-effectiveness", "--output-dir", "out"])
        self.assertEqual(args.handler.__name__, "command_prepare_skill_effectiveness")

    def test_cli_dry_run_does_not_deploy_or_write_source_and_real_run_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store(root)
            path = root / "pairs.json"
            path.write_text(json.dumps({"pairs": [{"pair_id": "pair", "raw_skill_id": "raw:add",
                "polished_skill_id": "polished:add", "equivalence_confirmed": True,
                "tasks": [Task.name]}]}))
            args = build_parser().parse_args(["evaluate-skill-effectiveness", "--adapter-path", "unused",
                "--pairs-file", str(path), "--skill-database", str(store.database),
                "--output-dir", str(root / "out"), "--resume", "--dry-run"])
            deployment = mock.MagicMock()
            deployment.build_adapter_command.return_value = ["swift", "deploy"]
            deployment.activate_adapters.return_value.__enter__.return_value = [profile()]
            before = store.database.read_bytes()
            with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=self.config()), \
                 mock.patch.object(ProjectConfig, "ensure_runtime_dirs") as dirs, \
                 mock.patch("src1.pmtskill_v2.cli._adapter_deployment", return_value=(deployment, [])), \
                 mock.patch("src1.pmtskill_v2.cli._print") as output, \
                 mock.patch("src1.pmtskill_v2.evaluation.skill_effectiveness.run_experiment", return_value={}) as execute:
                self.assertEqual(command_evaluate_skill_effectiveness(args), 0)
                deployment.activate_adapters.assert_not_called()
                execute.assert_not_called()
                dirs.assert_not_called()
                self.assertFalse((root / "out").exists())
                self.assertEqual(output.call_args.args[0]["expected_episodes_per_condition"], 1)
                args.dry_run = False
                self.assertEqual(command_evaluate_skill_effectiveness(args), 0)
                self.assertEqual(execute.call_args.args[1].database, root / "out/inputs/skill_library.sqlite3")
                self.assertTrue(execute.call_args.kwargs["resume"])
                self.assertFalse(execute.call_args.kwargs["store_trajectories"])
            self.assertEqual(store.database.read_bytes(), before)

    def test_prepare_copies_database_and_never_approves_taxonomy_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store(root)
            before = store.database.read_bytes()
            result = prepare_inputs(store.database, root / "out")
            self.assertEqual(store.database.read_bytes(), before)
            self.assertNotEqual(Path(result["skill_database"]), store.database)
            rows = json.loads(Path(result["pairs_template"]).read_text(encoding="utf-8"))["pairs"]
            self.assertFalse(rows[0]["equivalence_confirmed"])
            self.assertFalse(rows[0]["enabled"])
            self.assertEqual(rows[0]["raw_skill_id"], "")
            Path(result["pairs_template"]).write_text("manual changes", encoding="utf-8")
            prepare_inputs(store.database, root / "out")
            self.assertEqual(Path(result["pairs_template"]).read_text(), "manual changes")

    def test_pair_requires_explicit_equivalence_and_preserves_repeated_primitives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store(root)
            path = root / "pairs.json"
            row = {"pair_id": "contact", "raw_skill_id": "raw:add", "polished_skill_id": "polished:add",
                   "tasks": ["ContactsAddContact"]}
            path.write_text(json.dumps({"pairs": [row]}))
            with self.assertRaisesRegex(ValueError, "equivalence_confirmed"):
                load_pairs(store, path)
            row["equivalence_confirmed"] = True
            path.write_text(json.dumps({"pairs": [row]}))
            pair = load_pairs(store, path)[0]
            self.assertEqual(pair.polished.topology.primitive_sequence(), ("action.click", "action.type", "action.click"))
            with self.assertRaisesRegex(ValueError, "没有启用"):
                load_pairs(store, path, tasks=["OtherTask"])
            pair.polished.topology.nodes[1].condition = "if visible"
            store.upsert_skill(pair.polished)
            with self.assertRaisesRegex(ValueError, "不能静默丢弃分支"):
                load_pairs(store, path)

    def test_three_conditions_really_change_guidance_without_leaking_polished_text(self):
        pair = make_pair()
        for condition, marker in (("raw", "RAW_ORIGINAL_GUIDANCE"), ("polished", "POLISHED_ONLY_GUIDANCE"),
                                   ("primitives", "Flat primitive-level implementation")):
            client = Client()
            wrapper = EffectivenessWrapper(profile(), pair, condition, client=client)
            wrapper.set_goal("Add a contact")
            wrapper.begin_step()
            wrapper.predict_mm("action prompt", [])
            wrapper.predict_mm("summary prompt", [])
            self.assertIn(marker, client.prompts[0])
            self.assertNotIn(marker, client.prompts[1])
            if condition == "primitives":
                self.assertNotIn("POLISHED_ONLY_GUIDANCE", client.prompts[0])
                self.assertNotIn("RAW_ORIGINAL_GUIDANCE", client.prompts[0])
                self.assertEqual(wrapper.flat_guidance.count("action.click"), 2)
            self.assertIsNone(wrapper.runtime)
            self.assertEqual(wrapper.metrics["action_calls"], 1)
            self.assertEqual(wrapper.metrics["summary_calls"], 1)
            self.assertEqual(wrapper.metrics["model_invocations"], 2)
            self.assertEqual(wrapper.model_pool.switch_count, 0)

    def test_stepwise_polished_uses_real_ports_guidance(self):
        pair = make_pair()
        pair.polished.metadata.update({"prompt_mode": "stepwise", "compiler": "evidence-hierarchy-v2",
                                       "instructions": ["Fill current name", "Verify live screen"],
                                       "leaf_primitive_ids": ["action.type", "action.click"]})
        client = Client()
        wrapper = EffectivenessWrapper(profile(), pair, "polished", client=client)
        wrapper.set_goal("goal")
        wrapper.begin_step()
        wrapper.predict_mm("action", [])
        self.assertIn("Approximate stage", client.prompts[0])
        self.assertIn("Fill current name", client.prompts[0])

    def test_no_silent_truncation_of_raw_or_primitive_sequence(self):
        pair = make_pair()
        pair.raw.body = "x" * 2000
        with self.assertRaisesRegex(ValueError, "不截断原文"):
            EffectivenessWrapper(profile(), pair, "raw", client=Client(), guidance_max_chars=1000)
        with self.assertRaisesRegex(ValueError, "不能截掉动作"):
            EffectivenessWrapper(profile(), make_pair(), "primitives", client=Client(), guidance_max_chars=10)

    def test_http_invocation_counter_includes_real_retry_attempts(self):
        client = OpenAICompatibleVLClient(profile(), maximum_retries=2)
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "done"}}]}).encode())
        with mock.patch("src1.pmtskill_v2.inference.vlm.urllib.request.urlopen",
                        side_effect=[urllib.error.URLError("retry"), response]), \
             mock.patch("src1.pmtskill_v2.inference.vlm.time.sleep"):
            self.assertEqual(client.generate("goal").text, "done")
        self.assertEqual(client.http_request_count, 2)

    def test_frozen_task_parameters_are_reused_not_regenerated_on_resume(self):
        create = mock.Mock(return_value=Suite(ContactsAddContact=[Task({"name": "folder_TIMESTAMP", "seed": 42})]))
        suite_utils = types.SimpleNamespace(create_suite=create, Suite=Suite)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            args = dict(combinations=1, seed=42, family="android_world", environment=None, path=path)
            first, first_hash = frozen_suite(suite_utils, {Task.name: Task}, [Task.name], **args)
            create.side_effect = AssertionError("must not regenerate timestamp")
            second, second_hash = frozen_suite(suite_utils, {Task.name: Task}, [Task.name], **args)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first[Task.name][0].params, second[Task.name][0].params)
            self.assertEqual(create.call_count, 1)

    def test_report_uses_matched_valid_denominator_and_retains_true_failures(self):
        rows = [fake_row(c, success=c != "raw") for c in CONDITIONS]
        rows += [fake_row(c, instance=1, valid=c != "raw") for c in CONDITIONS]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_report(root, [make_pair()], rows, model_id="control", expected=2)
            self.assertEqual(summary["valid_matched_triplets"], 1)
            self.assertFalse(summary["complete"])
            self.assertEqual([r["success_rate"] for r in summary["table"]], [0, 1, 1])
            self.assertTrue(all(r["latency_seconds"] is None for r in summary["all_three_successful_table"]))
            self.assertTrue((root / "table_skill_effectiveness.tex").is_file())
            self.assertEqual(len((root / "trials.jsonl").read_text().splitlines()), 6)

    def test_mismatched_goals_are_not_reported_as_equivalent(self):
        rows = [fake_row(c) for c in CONDITIONS]
        rows[0]["goal"] = "different objective"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "不一致"):
            write_report(Path(directory), [make_pair()], rows, model_id="control", expected=1)

    def test_changed_experiment_cannot_resume_and_infrastructure_error_leaves_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store(root)
            runner = mock.Mock()
            runner.run.side_effect = lambda pair, condition, **kwargs: [fake_row(condition)]
            result = run_experiment(self.config(), store, profile(), [make_pair()], root / "run", runner=runner)
            self.assertTrue(result["summary"]["complete"])
            with self.assertRaisesRegex(ValueError, "配置发生变化"):
                run_experiment(self.config(), store, profile(), [make_pair()], root / "run",
                               seed=43, resume=True, runner=runner)
            runner.run.side_effect = [ [fake_row("raw")], RuntimeError("emulator unavailable") ]
            with self.assertRaisesRegex(RuntimeError, "emulator unavailable"):
                run_experiment(self.config(), store, profile(), [make_pair()], root / "interrupted", runner=runner)
            summary = json.loads((root / "interrupted/summary.json").read_text())
            self.assertFalse(summary["evaluation_available"])
            self.assertTrue(all(r["success_rate"] is None for r in summary["table"]))

    def test_compact_checkpoint_keeps_counters_without_full_trajectory(self):
        episode = {"goal": "goal", "task_template": Task.name, "is_successful": 0.0,
                   "exception_info": None, "episode_length": 1, "run_time": 2,
                   "episode_data": {"action_output": ["PRIVATE_ACTION"]}, METRICS: {"model_invocations": 3}}
        compact = compact_result(episode)
        self.assertEqual(compact[METRICS]["model_invocations"], 3)
        self.assertEqual(compact["episode_data"], {})
        self.assertNotIn("PRIVATE_ACTION", json.dumps(compact))

    def test_api_failures_are_exception_results_and_can_be_retried(self):
        wrapper = types.SimpleNamespace(metrics={"failed_calls": 1, "model_invocations": 3}, reset=lambda: None)
        harness = types.SimpleNamespace(_run_task=lambda: {"exception_info": None, "is_successful": 0.0})
        original = harness._run_task
        with record_task_metrics(harness, wrapper):
            result = harness._run_task()
        self.assertIs(harness._run_task, original)
        self.assertIn("Model API failure", result["exception_info"])
        self.assertFalse(result["_evaluation_episode_data_valid"])

    def test_real_runner_all_three_arms_and_resume_with_mocked_device_and_http(self):
        """Run real wrapper, meter, frozen-suite, checkpoint and aggregation code."""
        class M3A:
            def __init__(self, env, llm, **kwargs):
                self.env, self.llm = env, llm

            def reset(self, go_home_on_reset=False):
                return None

            def step(self, goal):
                action, _, raw = self.llm.predict_mm("action", [])
                summary, _, summary_raw = self.llm.predict_mm("summerize the latest step", [])
                return types.SimpleNamespace(data={"action_output": action,
                    "action_output_json": {"action_type": "click", "index": 1},
                    "action_raw_response": raw, "summary": summary, "summary_raw_response": summary_raw})

        def original_task(task, agent):
            agent.reset()
            data = [agent.step(task.goal).data for _ in range(2)]
            return {"task_template": task.name, "goal": task.goal, "is_successful": float(agent.llm.condition != "raw"),
                    "exception_info": None, "episode_length": 2, "run_time": 1,
                    "episode_data": {key: [item.get(key) for item in data] for key in data[0]}}

        harness = types.SimpleNamespace(Suite=Suite, _run_task=original_task)
        create = mock.Mock(return_value=Suite(ContactsAddContact=[Task({"name": "saved-timestamp", "seed": 42})]))
        harness.create_suite = create

        def run(suite, agent, *, checkpointer, demo_mode, return_full_episode_data,
                max_n_steps_override, stop_on_task_success):
            rows = []
            for name, tasks in suite.items():
                for i, task in enumerate(tasks):
                    row = harness._run_task(task, agent)
                    row["instance_id"] = i
                    checkpointer.save_episodes([row], f"{name}_{i}")
                    rows.append(row)
            return rows

        harness.run = run
        package = types.ModuleType("android_world")
        package.registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(get_registry=lambda **kwargs: {Task.name: Task}))
        package.suite_utils = harness
        agents = types.ModuleType("android_world.agents")
        agents.m3a = types.SimpleNamespace(M3A=M3A)
        env = types.ModuleType("android_world.env")
        env.env_launcher = object()
        modules = {"android_world": package, "android_world.agents": agents, "android_world.env": env}
        calls = []

        def http(request, **kwargs):
            calls.append(json.loads(request.data))
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": 'Reason: tap\nAction: {"action_type":"click","index":1}'}}]}).encode())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store(root)
            target = root / "run"
            with mock.patch.dict("sys.modules", modules), \
                 mock.patch("src1.pmtskill_v2.evaluation.skill_effectiveness_runner.bootstrap_android_world"), \
                 mock.patch("src1.pmtskill_v2.evaluation.skill_effectiveness_runner.load_android_world_environment", side_effect=lambda *args: mock.Mock()), \
                 mock.patch("src1.pmtskill_v2.inference.vlm.urllib.request.urlopen", side_effect=http), \
                 mock.patch("src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={}):
                result = run_experiment(self.config(), store, profile(), [make_pair()], target)
                self.assertEqual(len(calls), 12)
                self.assertEqual([r["model_invocations"] for r in result["summary"]["table"]], [4, 4, 4])
                calls.clear()
                resumed = run_experiment(self.config(), store, profile(), [make_pair()], target, resume=True)
                self.assertEqual(calls, [])
                self.assertEqual(resumed["summary"]["table"], result["summary"]["table"])
                self.assertEqual(create.call_count, 1)
                self.assertEqual(len(list(target.glob("conditions/*/*/checkpoints/*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
