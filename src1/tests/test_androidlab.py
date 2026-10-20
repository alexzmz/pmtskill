"""Device/API-free tests. Optional official-source contract tests use ANDROIDLAB_REFERENCE."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src1.androidlab import METHODS
from src1.androidlab.agents import AppContractClient
from src1.androidlab.catalog import load_catalog, source_identity
from src1.androidlab.config import AndroidLabConfig
from src1.androidlab.driver import prepare, snapshot_database
from src1.androidlab.environment import (AndroidLabEnvironment, DockerDevice, InfrastructureError,
                                        JudgeError, StepBudgetReached, native_action, xml_forest)
from src1.androidlab.reporting import aggregate
from src1.androidlab.worker import run_episodes
from src1.compare_androidlab import build_commands, build_parser as batch_parser
from src1.pmtskill_v2.cli import build_parser


XML = '<hierarchy rotation="0"><node bounds="[0,0][100,200]" class="android.widget.LinearLayout" package="app"><node bounds="[10,20][50,60]" class="android.widget.EditText" text="Name" clickable="true" enabled="true"/></node></hierarchy>'


def task(name="clock_1"):
    return {"task_id": name, "app": "Clock", "package": "com.google.android.deskclock",
            "instruction": "Set an alarm", "metric_type": "operation", "capability": "unclassified",
            "metric_module": "evaluation.tasks.clock", "adb_query": None}


def options(root):
    return AndroidLabConfig(Path(root), wait_after_action_seconds=0, app_start_wait_seconds=0)


def fake_source(root):
    root = Path(root)
    for relative in ("adb_client.py", "evaluation/task.py", "evaluation/definition.py", "utils_mobile/xml_tool.py"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("# source\n", encoding="utf-8")
    path = root / "evaluation/config/clock.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"APP": "Clock", "package": "com.google.android.deskclock", "tasks": [
        {"task_id": name, "task": name, "metric_type": kind, "metric_func": "evaluation.tasks.clock"}
        for name, kind in (("clock_1", "operation"), ("clock_2", "operations"), ("clock_3", "query_detect"))]}), encoding="utf-8")


def fake_xml_module():
    module = types.ModuleType("utils_mobile.xml_tool")
    class Tree:
        def process(self, xml, **kwargs):
            return json.dumps({"xml": xml})
    module.UIXMLTree = Tree
    return module


class AndroidLabTest(unittest.TestCase):
    def test_all_eight_cli_commands_share_resources_and_correct_adapter_count(self):
        args = batch_parser().parse_args(["--config", "c.toml", "--output-dir", "out", "--dry-run",
                                          "--cuda-visible-devices", "2", "--deploy-port", "8004", "--methods", *METHODS])
        commands = build_commands(args, {"suite_file": "suite.json", "skill_database": "frozen.sqlite3"})
        self.assertEqual(set(commands), set(METHODS))
        for method, command in commands.items():
            parsed = build_parser().parse_args(command[3:])
            self.assertEqual(parsed.method, method)
            self.assertEqual(parsed.family, "android_lab")
            self.assertEqual(len(parsed.adapter_paths), 3 if method in {"ports", "random-skvm"} else 1)
            self.assertEqual(parsed.adapter_paths[0], args.adapter_paths[0])
            self.assertEqual((parsed.seed, parsed.max_steps, parsed.combinations), (42, 30, 1))
            self.assertEqual((parsed.cuda_visible_devices, parsed.deploy_port), ("2", 8004))
            self.assertTrue(parsed.resume)
            self.assertFalse(parsed.store_trajectories)
            self.assertFalse(parsed.record_traces)

    def test_prepare_parses_without_model_argument(self):
        args = build_parser().parse_args(["prepare-androidlab", "--output-dir", "out"])
        self.assertEqual(args.family, "android_lab")
        self.assertEqual(args.method, "standalone")

    def test_api_dry_run_does_not_create_output_or_touch_model_docker(self):
        from src1.androidlab.driver import command_evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_source(root / "lab")
            agent = root / "aw/android_world/agents/m3a.py"
            agent.parent.mkdir(parents=True)
            agent.write_text("# agent", encoding="utf-8")
            config = root / "c.toml"
            config.write_text('[paths]\nandroid_world_root="aw"\n[androidlab]\nroot="lab"\n'
                              '[[models]]\nmodel_id="fixed"\nbase_url="http://127.0.0.1:8002/v1"\n', encoding="utf-8")
            args = build_parser().parse_args(["--config", str(config), "evaluate-androidlab", "--method", "standalone",
                                              "--model-id", "fixed", "--tasks", "clock_1", "--dry-run",
                                              "--output-dir", str(root / "out")])
            with mock.patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}), \
                 mock.patch("src1.androidlab.driver.run_worker") as worker, \
                 mock.patch("src1.androidlab.driver.docker_image_identity") as docker, \
                 contextlib.redirect_stdout(__import__("io").StringIO()):
                self.assertEqual(command_evaluate(args), 0)
                worker.assert_not_called()
                docker.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_config_is_separate_and_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.toml"
            path.write_text('[androidlab]\nroot="source"\ndocker_port=6070\n', encoding="utf-8")
            value = AndroidLabConfig.load(path)
            self.assertEqual(value.root, (Path(directory) / "source").resolve())
            self.assertEqual(value.docker_port, 6070)

    def test_catalog_normalizes_official_typo_and_rejects_aw_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_source(directory)
            yaml = SimpleNamespace(safe_load=json.loads)
            with mock.patch.dict(sys.modules, {"yaml": yaml}):
                values = load_catalog(options(directory))
                self.assertEqual(len(values), 3)
                self.assertEqual(values[1]["metric_type"], "operation")
                self.assertEqual(load_catalog(options(directory), tasks=["clock_2,clock_1"])[0]["task_id"], "clock_2")
                self.assertEqual(load_catalog(options(directory), task_count=2, seed=42),
                                 load_catalog(options(directory), task_count=2, seed=42))
                with self.assertRaisesRegex(ValueError, "Unknown AndroidLab"):
                    load_catalog(options(directory), tasks=["ContactsAddContact"])

    def test_source_identity_changes_with_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_source(directory)
            before = source_identity(directory)
            (Path(directory) / "evaluation/task.py").write_text("# changed", encoding="utf-8")
            self.assertNotEqual(before, source_identity(directory))

    def test_prepare_is_immutable_and_has_no_model_or_docker_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_source(root / "source")
            args = batch_parser().parse_args(["--config", "unused", "--output-dir", str(root / "out"), "--prepare-only"])
            config = SimpleNamespace(paths=SimpleNamespace(database=root / "missing.sqlite3"))
            with mock.patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}), mock.patch("subprocess.run") as run:
                first = prepare(config, options(root / "source"), args)
                self.assertEqual(first, prepare(config, options(root / "source"), args))
                self.assertEqual(first["task_count"], 3)
                self.assertTrue((root / "out/standalone/cli_logs").is_dir())
                self.assertFalse((root / "out/ports").exists())
                args.max_steps += 1
                with self.assertRaisesRegex(ValueError, "differs"):
                    prepare(config, options(root / "source"), args)
                run.assert_not_called()

    def test_sqlite_backup_includes_committed_wal_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite3"
            with contextlib.closing(sqlite3.connect(source)) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE evidence(value)")
                connection.execute("INSERT INTO evidence VALUES (42)")
                connection.commit()
                backup = snapshot_database(source, root / "out/library.sqlite3")
                with contextlib.closing(sqlite3.connect(backup)) as copied:
                    self.assertEqual(copied.execute("SELECT value FROM evidence").fetchone(), (42,))
                connection.execute("INSERT INTO evidence VALUES (99)")
                connection.commit()
                self.assertEqual(snapshot_database(source, backup), backup)
                with contextlib.closing(sqlite3.connect(backup)) as copied:
                    self.assertEqual(copied.execute("SELECT COUNT(*) FROM evidence").fetchone(), (1,))

    def test_xml_preserves_tree_ids_editability_and_leaf_display_order(self):
        forest = xml_forest(XML)
        nodes = forest.windows[0].tree.nodes
        self.assertEqual(nodes[0].child_ids, [1])
        self.assertTrue(nodes[1].is_editable)
        self.assertEqual(nodes[1].bounds_in_screen.left, 10)
        self.assertTrue(nodes[1].HasField("bounds_in_screen"))
        self.assertEqual([n.unique_id for n in nodes if not n.child_ids or n.content_description or n.is_scrollable], [1])

    def test_empty_xml_is_infrastructure_failure(self):
        with self.assertRaises(InfrastructureError):
            xml_forest("<hierarchy/>")

    def test_finish_is_native_schema_not_success(self):
        parsed = native_action({"action_type": "answer", "text": "42"})
        self.assertEqual(parsed, {"operation": "finish", "action": "finish", "kwargs": {"message": "42"}})
        self.assertNotIn("successful", parsed)
        self.assertEqual(native_action({"action_type": "status", "goal_status": "complete"})["action"], "finish")

    def make_environment(self, directory, metric=None):
        device = mock.Mock()
        device.adb.return_value = ""
        environment = AndroidLabEnvironment(device, task(), options(directory), directory,
                                             metric or mock.Mock(judge=mock.Mock(return_value={"complete": False})), 2)
        environment.current = {"index": 0, "xml_text": XML, "image_bytes": None, "window": [100, 200],
                               "current_activity": "app", "command": {}}
        environment.state = SimpleNamespace(pixels=SimpleNamespace(shape=(200, 100, 3)),
                                            ui_elements=[SimpleNamespace(bbox_pixels=SimpleNamespace(center=(30, 40)))])
        environment.app_packages = {"clock": "com.google.android.deskclock"}
        return environment

    def test_last_eligible_judge_result_not_any_success(self):
        metric = mock.Mock()
        metric.judge.side_effect = [{"judge_page": True, "complete": True}, {"judge_page": False},
                                   {"judge_page": True, "complete": False}]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"utils_mobile.xml_tool": fake_xml_module()}):
            env = self.make_environment(directory, metric)
            env.evaluate(native_action({"action_type": "wait"}))
            env.evaluate(native_action({"action_type": "wait"}))
            self.assertTrue(env.final_result["complete"])
            env.evaluate(native_action({"action_type": "wait"}))
            self.assertFalse(env.final_result["complete"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_judge_exception_is_unavailable_not_false(self):
        metric = mock.Mock(judge=mock.Mock(side_effect=RuntimeError("judge offline")))
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"utils_mobile.xml_tool": fake_xml_module()}):
            env = self.make_environment(directory, metric)
            with self.assertRaises(JudgeError):
                env.evaluate(native_action({"action_type": "status"}))
            self.assertIsInstance(env.error, JudgeError)

    def test_answer_judge_transport_errors_are_not_swallowed(self):
        from src1.androidlab.worker import install_answer_judge
        native = types.ModuleType("evaluation.task")
        native.SingleTask = type("SingleTask", (), {})
        evaluation = types.ModuleType("evaluation")
        evaluation.task = native
        client = mock.Mock()
        client.generate.return_value = SimpleNamespace(text="[True]")
        with mock.patch.dict(sys.modules, {"evaluation": evaluation, "evaluation.task": native}), \
             mock.patch("src1.pmtskill_v2.inference.vlm.OpenAICompatibleVLClient", return_value=client):
            install_answer_judge(SimpleNamespace(model_id="judge"))
            metric = native.SingleTask()
            metric.final_ground_truth = "42"
            line = {"parsed_action": {"action": "finish", "kwargs": {"message": "forty two"}}, "target": "count"}
            self.assertTrue(metric.check_answer(line))
            client.generate.side_effect = RuntimeError("API offline")
            with self.assertRaises(JudgeError):
                metric.check_answer(line)
            self.assertIsInstance(metric._ports_answer_judge_error, JudgeError)

    def test_native_task_cannot_hide_answer_judge_api_failure(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"utils_mobile.xml_tool": fake_xml_module()}):
            metric = SimpleNamespace(_ports_answer_judge_error=JudgeError("judge API down"),
                                     judge=lambda *a: {"judge_page": True, "complete": False})
            env = self.make_environment(directory, metric)
            with self.assertRaisesRegex(JudgeError, "judge API down"):
                env.evaluate(native_action({"action_type": "answer", "text": "42"}))

    def test_query_uses_pre_action_snapshot_value(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"utils_mobile.xml_tool": fake_xml_module()}):
            env = self.make_environment(directory)
            env.task["adb_query"] = "adb shell settings get global bluetooth_on"
            env.current["command"] = {env.task["adb_query"]: "0"}
            env.evaluate(native_action({"action_type": "click", "index": 0}))
            line = env.metric.judge.call_args.args[1]
            self.assertEqual(line["command"][env.task["adb_query"]], "0")
            env.device.query.assert_not_called()

    def test_click_uses_own_device_pixels_and_never_host_shell(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"utils_mobile.xml_tool": fake_xml_module()}):
            env = self.make_environment(directory)
            env.execute_action({"action_type": "click", "index": 0})
            env.device.adb.assert_called_once_with("shell", "input", "tap", "30", "40")
            self.assertEqual(env.actions, 1)

    def test_budget_escapes_upstream_exception_handlers(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_environment(directory)
            env.actions = 2
            self.assertFalse(issubclass(StepBudgetReached, Exception))
            with self.assertRaises(StepBudgetReached):
                env.execute_action({"action_type": "click", "index": 0})

    def test_unknown_skill_name_does_not_launch_wrong_aw_application(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_environment(directory)
            with self.assertRaisesRegex(ValueError, "Unknown AndroidLab"):
                env._execute({"action_type": "open_app", "app_name": "file-manager"})
            env.device.adb.assert_not_called()

    def test_text_base64_preserves_untrusted_string_and_requires_keyboard(self):
        import base64
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_environment(directory)
            env.device.adb.side_effect = ["com.android.adbkeyboard/.AdbIME", "Broadcast completed: result=0"]
            text = 'Alice; $(touch /wrong) "中文"'
            env._execute({"action_type": "input_text", "text": text, "clear_text": False})
            args = env.device.adb.call_args.args
            self.assertEqual(base64.b64decode(args[-1]).decode(), text)
            self.assertNotIn(text, args)
            env.device.adb.side_effect = None
            env.device.adb.return_value = "other.IME"
            with self.assertRaisesRegex(InfrastructureError, "ADBKeyboard"):
                env._execute({"action_type": "input_text", "text": "hello"})

    def test_app_contract_shared_and_no_truth_in_prompt(self):
        base = mock.Mock(model_id="control")
        base.predict_mm.return_value = ("Reason: done", True, {"_pmtskill": {"model_id": "control"}})
        wrapped = AppContractClient(base, ["Clock", "Calendar"])
        wrapped.predict_mm("goal only", [])
        prompt = base.predict_mm.call_args.args[0]
        self.assertIn("Clock, Calendar", prompt)
        self.assertIn("a summary request requires a summary", prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertEqual(wrapped.calls, 1)
        base.predict_mm.return_value = ("error", False, None)
        with self.assertRaisesRegex(RuntimeError, "model call failed"):
            wrapped.predict("test")

    def test_resume_skips_real_failure_but_retries_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = [task("clock_1"), task("clock_2")]
            first = mock.Mock(side_effect=[{"successful": False, "steps": 2}, InfrastructureError("offline")])
            summary = run_episodes(jobs, 1, directory, "same", first, resume=True,
                                   continue_on_error=True, metadata={"method": "ports"})
            self.assertIsNone(summary["success_rate"])
            self.assertEqual(summary["valid_episodes"], 1)
            second = mock.Mock(return_value={"successful": True, "steps": 1})
            summary = run_episodes(jobs, 1, directory, "same", second, resume=True,
                                   continue_on_error=False, metadata={"method": "ports"})
            second.assert_called_once()
            self.assertEqual(summary["success_rate"], .5)

    def test_resume_rejects_changed_protocol_and_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_episodes([task()], 1, directory, "old", lambda *a: {"successful": True},
                         resume=True, continue_on_error=False, metadata={"method": "ports"})
            with self.assertRaisesRegex(ValueError, "changed"):
                run_episodes([task()], 1, directory, "new", mock.Mock(),
                             resume=True, continue_on_error=False, metadata={"method": "ports"})
            self.assertTrue((Path(directory) / "report.md").is_file())

    def test_strict_environment_failure_writes_na_report_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InfrastructureError):
                run_episodes([task()], 1, directory, "same", mock.Mock(side_effect=InfrastructureError("down")),
                             resume=True, continue_on_error=False, metadata={"method": "ports"})
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertIsNone(summary["success_rate"])
            self.assertEqual(summary["unavailable_episodes"], 1)

    def test_aggregate_never_treats_missing_as_failure(self):
        summary = aggregate([{**task(), "status": "unavailable", "successful": None}], 1)
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(summary["observed_success_rate"])
        self.assertFalse(summary["evaluation_available"])

    def test_cleanup_only_exact_owned_container(self):
        device = DockerDevice(options("unused"), "sha256:" + "1" * 64)
        device.container_id = "a" * 64
        with mock.patch.object(device, "_docker") as run:
            device.__exit__(None, None, None)
            run.assert_called_once_with("rm", "-f", "a" * 64, timeout=30)
            device.__exit__(None, None, None)
            self.assertEqual(run.call_count, 1)


@unittest.skipUnless(os.environ.get("ANDROIDLAB_REFERENCE") and importlib.util.find_spec("yaml"),
                     "Optional official-source contract test needs ANDROIDLAB_REFERENCE and PyYAML")
class AndroidLabOfficialContractTest(unittest.TestCase):
    def test_actual_official_catalog_is_138_tasks_in_nine_apps(self):
        records = load_catalog(options(os.environ["ANDROIDLAB_REFERENCE"]))
        self.assertEqual(len(records), 138)
        self.assertEqual(len({r["app"] for r in records}), 9)
        self.assertTrue(all(r["capability"] == "unclassified" for r in records))
        for record in records:
            package = Path(os.environ["ANDROIDLAB_REFERENCE"]) / record["metric_module"].replace(".", "/") / "__init__.py"
            tree = ast.parse(package.read_text(encoding="utf-8"))
            mapping = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "function_map" for t in node.targets))
            self.assertIn(record["task_id"], [key.value for key in mapping.keys])

    @unittest.skipUnless(importlib.util.find_spec("xmltodict") and importlib.util.find_spec("lxml"),
                         "Official XML processor needs xmltodict and lxml")
    def test_actual_xml_processor_and_official_contacts_and_setting_predicates(self):
        import copy
        import math
        import re
        import uuid
        import xmltodict
        from collections import deque
        from lxml import etree
        root = Path(os.environ["ANDROIDLAB_REFERENCE"])
        namespace = {"copy": copy, "json": json, "math": math, "re": re, "uuid": uuid,
                     "Dict": dict, "deque": deque, "xmltodict": xmltodict, "etree": etree}
        def definitions(relative, names=None):
            source = ast.parse((root / relative).read_text(encoding="utf-8"))
            selected = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        and (names is None or node.name in names)]
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / relative), "exec"), namespace)
        definitions("utils_mobile/specialCheck.py")
        special = ast.parse((root / "utils_mobile/specialCheck.py").read_text(encoding="utf-8"))
        constants = [node for node in special.body if isinstance(node, ast.Assign)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), "official-special-checks", "exec"), namespace)
        definitions("utils_mobile/xml_tool.py", {"UIXMLTree", "get_words_in_certain_length"})
        definitions("evaluation/utils.py")
        definitions("evaluation/task.py", {"SingleTask"})
        definitions("evaluation/tasks/contacts/contacts.py")
        definitions("evaluation/tasks/setting/setting.py")
        attributes = ('index="0" text="{text}" content-desc="" resource-id="" '
                      'class="android.widget.TextView" package="com.google.android.contacts" '
                      'checkable="false" checked="false" clickable="false" enabled="true" '
                      'focusable="false" focused="false" scrollable="false" long-clickable="false" '
                      'password="false" selected="false" bounds="{bounds}"')
        outer = attributes.format(text="Contacts", bounds="[0,0][100,200]")
        first = attributes.format(text="John", bounds="[10,20][90,50]")
        second = attributes.format(text="1 (234) 567-8", bounds="[10,60][90,100]")
        xml = f'<hierarchy rotation="0"><node {outer}><node {first}/><node {second}/></node></hierarchy>'
        compressed = json.loads(namespace["UIXMLTree"]().process(xml, level=1, str_type="json"))
        metric = namespace["SingleTask_Contacts_1"](SimpleNamespace())
        result = metric.judge(compressed, {"parsed_action": native_action({"action_type": "wait"})})
        self.assertTrue(result["complete"])
        false_tree = json.loads(namespace["UIXMLTree"]().process(xml.replace("John", "Alex"), level=1, str_type="json"))
        self.assertFalse(metric.judge(false_tree, {}).get("complete", False))
        setting = namespace["SingleTask_Setting_0"](SimpleNamespace())
        self.assertTrue(setting.judge({}, {"command": {"adb shell settings get global airplane_mode_on": "1"}})["complete"])
        self.assertFalse(setting.judge({}, {"command": {"adb shell settings get global airplane_mode_on": "0"}})["complete"])


if __name__ == "__main__":
    unittest.main()
