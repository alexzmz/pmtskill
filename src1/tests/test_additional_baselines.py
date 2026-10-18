"""Model/emulator-free regression for the additional benchmark protocols."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import random
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1 import batch_evaluate as batch
from src1.baselines import echotrail as echo
from src1.baselines.random_skvm import RandomSKVMClient, prepare_model
from src1.baselines.runner import episode_context, run_gui_baseline
from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import load_config
from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology
from src1.pmtskill_v2.evaluation.resume import EvaluationProgress


def profile(name="control"):
    return ModelProfile(model_id=name, served_model=name, base_url="http://localhost/v1", capabilities={})


class Encoder:
    def encode(self, texts):
        return [[1.0, 0.0] if "contact" in text.lower() else [0.0, 1.0] for text in texts]


def document():
    return {"protocol": echo.PROTOCOL, "memories": [
        {"id": "a", "goal": "add contact Alice", "final_intent": "add contact",
         "embedding": [1, 0], "steps": [{"interface_context": "Contacts", "intent": "add", "action": "click Add"}]},
        {"id": "b", "goal": "take a photo", "final_intent": "camera photo",
         "embedding": [0, 1], "steps": [{"interface_context": "Camera", "intent": "photo", "action": "click shutter"}]},
    ]}


class AdditionalBaselinesTest(unittest.TestCase):
    def test_eight_commands_parse_and_resources_match(self):
        args = batch.build_parser().parse_args(["--config", "missing.toml", "--output-dir", "unused",
                                               "--methods", *batch.ALL_METHODS,
                                               "--echotrail-memory", "memory.json", "--embedding-model-path", "/embedding"])
        jobs = batch.build_commands(args)
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            parsed = build_parser().parse_args(job["command"][3:])
            if job["method"] in {"ports", "random-skvm"}:
                self.assertEqual(len(parsed.adapter_paths), 3)
            else:
                self.assertEqual(parsed.adapter_path, batch.DEFAULT_ADAPTERS[0])
            self.assertEqual((parsed.seed, parsed.max_steps, parsed.combinations), (42, 30, 2))
            self.assertTrue(parsed.resume)
            self.assertFalse(parsed.store_trajectories)
        self.assertEqual(build_parser().parse_args(["evaluate-androidworld", "--adapter-path", "a"]).androidworld_agent, "m3a")
        self.assertEqual(build_parser().parse_args(["evaluate-androidworld", "--adapter-path", "a",
                                                   "--androidworld-agent", "t3a"]).androidworld_agent, "t3a")

    def test_random_is_local_uniform_reproducible_and_summary_pinned(self):
        profiles = [profile(name) for name in ("a", "b", "c")]
        clients = {p.model_id: mock.Mock() for p in profiles}
        for name, client in clients.items():
            client.predict_mm.return_value = (name, True, {"_pmtskill": {"skill_id": "real-aot"}})
        before = random.getstate()
        client = RandomSKVMClient(profiles, clients, {}, 42)
        client.start_episode("goal", "stable-instance")
        with self.assertRaises(RuntimeError):
            client.predict("before step")
        sequence = []
        for index in range(40):
            client.begin_step("goal")
            action = client.predict_mm("action", [object()])
            summary = client.predict_mm("summary", [object(), object()])
            self.assertEqual(action[0], summary[0])
            self.assertEqual(action[2]["_pmtskill"]["skill_id"], "real-aot")
            sequence.append(action[0])
        self.assertEqual(set(sequence), {"a", "b", "c"})
        self.assertTrue(any(a == b for a, b in zip(sequence, sequence[1:])))
        client.start_episode("goal", "stable-instance")
        repeated = []
        for _ in sequence:
            client.begin_step("goal")
            repeated.append(client.current)
        self.assertEqual(sequence, repeated)
        self.assertEqual(before, random.getstate())

    def test_random_model_preparation_has_distinct_tcp_aot_cache_identity(self):
        from src1.baseline_skvm.evaluator import legacy_kernel_module
        legacy = legacy_kernel_module()
        config = load_config("src1/config.example.toml")
        args = build_parser().parse_args(["evaluate-random-skvm", "--adapter-paths", "a", "b"])
        skill = SkillRecord(skill_id="one", name="One", description="click", body="Tap visible Add",
                            kind="polished", status=SkillStatus.ACTIVE, level=1,
                            topology=SkillTopology.from_sequence(["action.click"]))
        captured = []
        class Kernel:
            target_model = "test"
            preparation_summary = {"aot_coverage": 1.0}
            def __init__(self, options, **kwargs):
                captured.append((options.skvm_cache_dir, kwargs["served_model"]))
            def prepare(self):
                return []
            def manifest(self, variants):
                return {}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(legacy, "SkVMKernel", Kernel):
            args.skvm_cache_dir = directory
            prepare_model(config, profile("a"), args, [skill], Path(directory) / "a")
            prepare_model(config, profile("b"), args, [skill], Path(directory) / "b")
        self.assertNotEqual(captured[0][0], captured[1][0])
        self.assertEqual([item[1] for item in captured], ["a", "b"])

    def test_episode_context_runs_after_resume_skip_and_restores_hook(self):
        tasks = [types.SimpleNamespace(name="ContactsAddContact", goal=str(i), params={"task_seed": i}) for i in range(2)]
        suite = {"ContactsAddContact": tasks}
        harness = types.SimpleNamespace(_run_task=lambda task: {"goal": task.goal, "task_template": task.name,
                                                               "is_successful": 0.0, "episode_data": {}, "exception_info": None})
        original, client = harness._run_task, mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            progress = EvaluationProgress(Path(directory), {"seed": 42}, resume=False)
            with episode_context(harness, suite, client), progress.watch(harness, suite):
                result = harness._run_task(tasks[0])
                result["instance_id"] = 0
                progress.save_episodes([result], "ContactsAddContact_0")
            client.reset_mock()
            resumed = EvaluationProgress(Path(directory), {"seed": 42}, resume=True)
            with episode_context(harness, suite, client), resumed.watch(harness, suite):
                harness._run_task(tasks[0])
                harness._run_task(tasks[1])
            client.start_episode.assert_called_once()
            self.assertEqual(client.start_episode.call_args.args[0], "1")
            self.assertIs(harness._run_task, original)

    def test_semantic_and_bm25_retrieval_and_exact_goal_holdout(self):
        memory = echo.HybridMemory(document(), Encoder())
        self.assertEqual(memory.retrieve("create contact Bob", alpha=1, k=1)[0]["id"], "a")
        self.assertEqual(memory.retrieve("camera", alpha=0, k=1)[0]["id"], "b")
        self.assertEqual(memory.retrieve("add contact Alice", excluded_goal="add contact Alice")[0]["id"], "b")
        bad = document()
        bad["memories"][0]["embedding"] = [0, 0]
        with self.assertRaises(ValueError):
            echo.HybridMemory(bad, Encoder())

    def test_echo_preserves_images_does_not_inject_into_summary_or_update_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(memory_alpha=0.5, memory_top_k=2, memory_max_chars=1200)
            base = mock.Mock()
            base.predict_mm.return_value = ("action", True, {"_pmtskill": {"model_id": "control"}})
            doc = document()
            client = echo.EchoTrailClient(base, echo.HybridMemory(doc, Encoder()), args, Path(directory) / "stats.json")
            client.begin_step("create contact Bob")
            images = [object()]
            client.predict_mm("CURRENT GOAL", images)
            prompt = base.predict_mm.call_args.args[0]
            self.assertIn("Historical GUI demonstrations", prompt)
            self.assertIn("CURRENT GOAL", prompt)
            self.assertIs(base.predict_mm.call_args.args[1], images)
            client.predict_mm("summarize", [object(), object()])
            self.assertEqual(base.predict_mm.call_args.args[0], "summarize")
            self.assertEqual(client.stats["memory_injected_calls"], 1)
            self.assertEqual(doc, document())

    def test_critic_build_is_real_resumable_readonly_and_seed_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = root / "embedding"
            embedding.mkdir()
            (embedding / "config.json").write_text("{}", encoding="utf-8")
            source = root / "source.db"
            trace = {"trace_id": "one", "goal": "add contact Alice", "task_name": "ContactsAddContact",
                     "metadata": {"collection_context": {"suite_seed": 123}},
                     "events": [{"action": "click Add", "observation": "Contacts", "metadata": {}}]}
            with contextlib.closing(sqlite3.connect(source)) as db:
                with db:
                    db.execute("CREATE TABLE traces(trace_id TEXT,successful INTEGER,trace_json TEXT)")
                    db.execute("INSERT INTO traces VALUES(?,?,?)", ("one", 1, json.dumps(trace)))
            args = argparse.Namespace(output_dir=str(root / "memory"), source_database=str(source),
                                      embedding_model_path=str(embedding), source_seed=123, limit=10,
                                      critic_context_chars=10000, critic_threshold=4, resume=True)
            critic = mock.Mock()
            critic.generate.return_value.text = '{"score":4,"reason":"coherent","final_intent":"add contact"}'
            first = echo.build_memory(args, critic, {"model": "critic"}, encoder=Encoder())
            self.assertEqual(first["accepted"], 1)
            critic.generate.assert_called_once()
            echo.build_memory(args, critic, {"model": "critic"}, encoder=Encoder())
            critic.generate.assert_called_once()  # no duplicate paid calls on resume
            with contextlib.closing(sqlite3.connect(source)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM traces").fetchone()[0], 1)
            args.output_dir, args.source_seed = str(root / "other"), 456
            with self.assertRaisesRegex(ValueError, "seed"):
                echo.build_memory(args, critic, {}, encoder=Encoder())
            args.memory_file, args.seed = first["memory"], 123
            with self.assertRaisesRegex(ValueError, "seed"):
                echo.validate_memory(args)

    def test_invalid_critic_does_not_silently_make_up_a_score(self):
        critic = mock.Mock()
        critic.generate.return_value.text = "success"
        with self.assertRaisesRegex(ValueError, "非法 JSON"):
            echo.assess(critic, {"goal": "test", "steps": []})
        self.assertEqual(critic.generate.call_count, 2)

    def test_prepare_only_freezes_inputs_without_launching_model_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite3"
            with contextlib.closing(sqlite3.connect(source)) as db:
                db.execute("CREATE TABLE skills(id TEXT)")
                db.commit()
            args = batch.build_parser().parse_args([
                "--config", "src1/config.example.toml", "--output-dir", str(root / "compare"),
                "--skill-database", str(source), "--methods", "standalone", "androidworld", "random-skvm",
                "--prepare-only"])
            deployment = types.SimpleNamespace(profiles=lambda: [profile(n) for n in ("a", "b", "c")])
            with mock.patch("src1.pmtskill_v2.cli._adapter_deployment", return_value=(deployment, [])), mock.patch(
                "src1.pmtskill_v2.evaluation.android_world.sample_android_world_tasks", return_value=["ContactsAddContact"]), mock.patch(
                "src1.batch_evaluate.validate_dependencies"), mock.patch("src1.batch_evaluate.run_command", return_value=0) as run:
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(batch.run(args), 0)
                self.assertTrue(all("--dry-run" in call.args[0] for call in run.call_args_list))
            out = root / "compare"
            self.assertTrue((out / "inputs" / "environment.json").exists())
            self.assertTrue((out / "random_skvm" / "skill_library.sqlite3").exists())
            self.assertTrue((out / "commands.sh").exists())
            rows = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(all(r["status"] == "pending" and r["micro_sr"] is None for r in rows))

    def test_curiosity_focus_critic_processing_db_and_permanent_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = root / "encoder"
            embedding.mkdir()
            (embedding / "config.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(output_dir=str(root / "collection"), embedding_model_path=str(embedding),
                                      seed=123, apps=["contacts"], episodes_per_app=2, max_steps=3,
                                      focus_after_steps=1, critic_threshold=4, critic_context_chars=32000, resume=True)
            config = load_config("src1/config.example.toml")
            environment = types.SimpleNamespace(reset=mock.Mock(), execute_action=mock.Mock(), close=mock.Mock(),
                                                 get_state=lambda **kw: types.SimpleNamespace(pixels=object()), logical_screen_size=(100, 200))
            goals = []
            class Agent:
                def __init__(self, *args, **kwargs): self.n = 0
                def reset(self, *args): pass
                def set_max_steps(self, n): pass
                def step(self, goal):
                    self.n += 1
                    goals.append(goal)
                    return types.SimpleNamespace(done=self.n == 2, data={
                        "action_output_json": {"action_type": "click", "index": 1},
                        "before_ui_elements": [], "action_reason": "create contact", "summary": "editor opened"})
            fake_m3a = types.SimpleNamespace(M3A=Agent, _generate_ui_elements_description_list=lambda *a: "Contacts")
            agents, env = types.ModuleType("android_world.agents"), types.ModuleType("android_world.env")
            agents.m3a = fake_m3a
            env.env_launcher = types.SimpleNamespace()
            env.json_action = types.SimpleNamespace(JSONAction=lambda **kw: kw)
            modules = {"android_world": types.ModuleType("android_world"), "android_world.agents": agents, "android_world.env": env}
            client = mock.Mock()
            client.generate.side_effect = lambda prompt, *a, **kw: types.SimpleNamespace(text=(
                '{"goal":"create a contact"}' if prompt.startswith("Based on") else
                '{"score":4,"reason":"coherent workflow","final_intent":"create contact"}'))
            with mock.patch.dict(sys.modules, modules), mock.patch("src1.pmtskill_v2.offline.collector.bootstrap_android_world"), mock.patch(
                "src1.pmtskill_v2.core.environment.load_android_world_environment", return_value=environment), mock.patch(
                "src1.baselines.echotrail.LocalEncoder", return_value=Encoder()), mock.patch(
                "src1.baselines.echotrail.OpenAICompatibleVLClient", return_value=client):
                result = echo.collect_memory(config, profile(), args)
                self.assertEqual(len(goals), 4)
                self.assertIn("Explore contacts", goals[0])
                self.assertIn("Complete this discovered subgoal", goals[1])
                self.assertIn("coherent workflow", goals[2])
                count = client.generate.call_count
                echo.collect_memory(config, profile(), args)
                self.assertEqual(client.generate.call_count, count)
            stored = json.loads(Path(result["memory"]).read_text(encoding="utf-8"))
            self.assertTrue(stored["exploration_reproduced"])
            self.assertEqual(stored["source_kind"], "critic-guided-self-exploration")
            self.assertGreaterEqual(stored["accepted"], 1)
            self.assertNotIn("success_rate", stored)

    def test_native_runner_uses_shared_bounds_ports_resume_and_reports(self):
        for kind in ("m3a", "t3a"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                config = load_config("src1/config.example.toml")
                config.android_world = dataclasses.replace(config.android_world, console_port=5558, grpc_port=8556)
                args = argparse.Namespace(output_dir=directory, combinations=2, seed=42, family="android_world",
                                          max_steps=30, resume=True, store_trajectories=False)
                environment = types.SimpleNamespace(close=mock.Mock())
                base = types.SimpleNamespace(start_episode=mock.Mock(), reset=mock.Mock())
                class Suite(dict): pass
                class Agent:
                    def __init__(self, env, llm, **kwargs):
                        if kind == "t3a":
                            assert "wait_after_action_seconds" not in kwargs
                    def reset(self): pass
                    def step(self, goal):
                        return types.SimpleNamespace(data={"summary": "done", "raw_screenshot": object()})
                tasks = [types.SimpleNamespace(name="ContactsAddContact", goal=str(i), params={"task_seed": i}) for i in range(2)]
                harness = types.SimpleNamespace()
                def run_task(task):
                    return {"goal": task.goal, "task_template": task.name, "task_seed": task.params["task_seed"],
                            "is_successful": 1.0, "exception_info": None, "episode_length": 1, "run_time": 1,
                            "episode_data": {"step_number": [0], "action_output": ['{"action_type":"click","index":1}']}}
                harness._run_task = mock.Mock(side_effect=run_task)
                harness.create_suite = lambda *a, **k: Suite({"ContactsAddContact": tasks})
                def run(suite, agent, *, checkpointer, max_n_steps_override, **kwargs):
                    self.assertEqual(max_n_steps_override, 30)
                    out = []
                    for i, task in enumerate(tasks):
                        item = harness._run_task(task)
                        item["instance_id"] = i
                        checkpointer.save_episodes([item], f"{task.name}_{i}")
                        out.append(item)
                    return out
                harness.run = run
                aw = types.ModuleType("android_world")
                aw.registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(get_registry=lambda **kw: {}))
                aw.suite_utils = harness
                agents = types.ModuleType("android_world.agents")
                agents.m3a, agents.t3a = types.SimpleNamespace(M3A=Agent), types.SimpleNamespace(T3A=Agent)
                env = types.ModuleType("android_world.env")
                env.env_launcher = types.SimpleNamespace(load_and_setup_env=mock.Mock(return_value=environment))
                modules = {"android_world": aw, "android_world.agents": agents, "android_world.env": env}
                with mock.patch.dict(sys.modules, modules), mock.patch("src1.baselines.runner.bootstrap_android_world"), mock.patch(
                    "src1.pmtskill_v2.evaluation.android_world.recover_infrastructure_failures", return_value=contextlib.nullcontext()):
                    for _ in range(2):
                        artifacts = run_gui_baseline(config, [profile()], args, ["ContactsAddContact"], protocol="test-" + kind,
                                                     prepare_client=lambda path: (base, {}), agent_kind=kind)
                    self.assertEqual(harness._run_task.call_count, 2)
                    self.assertEqual(artifacts.summary["episodes_evaluated"], 2)
                    self.assertFalse((Path(directory) / "traces.jsonl").exists())
                    self.assertEqual(env.env_launcher.load_and_setup_env.call_args.kwargs["grpc_port"], 8556)


if __name__ == "__main__":
    unittest.main()
