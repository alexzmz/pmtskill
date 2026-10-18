"""Independent emulator console/gRPC bindings without AndroidWorld or GPU."""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.baseline_vdroid.config import VDroidDeviceConfig
from src1.baseline_vdroid.qwen import evaluator_for_adapter
from src1.pmtskill_v2.cli import _dry_evaluation_result, build_parser
from src1.pmtskill_v2.core.config import AndroidWorldConfig, load_config
from src1.pmtskill_v2.core.environment import load_android_world_environment
from src1.pmtskill_v2.core.models import ModelProfile
from src1.pmtskill_v2.evaluation.android_world import _evaluation_progress
from src1.pmtskill_v2.offline.collector import collect_teacher_trajectories
from src1.pmtskill_v2.offline.optimization_collector import (
    AndroidWorldOptimizationTraceCollector,
    OptimizationTraceCollectionOptions,
)


EXAMPLE = Path(__file__).parents[1] / "config.example.toml"


class AndroidWorldConnectionTest(unittest.TestCase):
    def test_only_historical_default_may_omit_grpc_port(self):
        for cls in (AndroidWorldConfig, VDroidDeviceConfig):
            self.assertEqual(cls().grpc_port, 8554)
            for console in (5556, 5558, 5560):
                with self.subTest(cls=cls, console=console):
                    with self.assertRaisesRegex(ValueError, "grpc_port"):
                        cls(console_port=console)

    def test_four_explicit_pairs_and_custom_port_reach_launcher_unchanged(self):
        for console, grpc in ((5554, 8554), (5556, 8555), (5558, 8556), (5560, 8557), (5556, 9100)):
            with self.subTest(console=console, grpc=grpc):
                config = AndroidWorldConfig(console_port=console, grpc_port=grpc, adb_path="custom-adb")
                environment = object()
                launch = mock.Mock(return_value=environment)
                with self.assertLogs(level="INFO") as logs:
                    actual = load_android_world_environment(types.SimpleNamespace(load_and_setup_env=launch), config)
                self.assertIs(actual, environment)
                launch.assert_called_once_with(
                    console_port=console, grpc_port=grpc, emulator_setup=False, adb_path="custom-adb",
                )
                self.assertIn(f"ADB=emulator-{console}", logs.output[0])
                self.assertIn(f"emulator_gRPC=localhost:{grpc}", logs.output[0])

    def test_invalid_ports_are_rejected_before_connecting(self):
        for cls in (AndroidWorldConfig, VDroidDeviceConfig):
            for field in ("console_port", "grpc_port"):
                for value in (0, -1, 65536, True, 5556.0, "8555"):
                    with self.subTest(cls=cls, field=field, value=value):
                        with self.assertRaisesRegex(ValueError, field):
                            cls(**{field: value})

    def test_native_vdroid_device_serial_defaults_to_selected_console(self):
        self.assertEqual(VDroidDeviceConfig(console_port=5556, grpc_port=8555).device_name, "emulator-5556")
        with self.assertRaisesRegex(ValueError, "指向不同模拟器"):
            VDroidDeviceConfig(console_port=5556, grpc_port=8555, device_name="emulator-5554")
        self.assertEqual(VDroidDeviceConfig(device_name="physical-device").device_name, "physical-device")

    def test_toml_reads_explicit_pair_and_rejects_missing_nondefault_grpc(self):
        for snippet, expected in (("", (5554, 8554)), (
            "[android_world]\nconsole_port=5556\ngrpc_port=8555\n", (5556, 8555),
        ), ("[android_world]\nconsole_port=5556\n", None)):
            with self.subTest(snippet=snippet), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(snippet + '\n[[models]]\nmodel_id="test"\n', encoding="utf-8")
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "必须显式配置"):
                        load_config(path)
                else:
                    config = load_config(path)
                    self.assertEqual((config.android_world.console_port, config.android_world.grpc_port), expected)

    def test_old_launcher_without_grpc_support_is_not_silently_used(self):
        called = []

        def old_launcher(console_port, emulator_setup=False, adb_path="adb"):
            called.append(console_port)

        with self.assertRaisesRegex(RuntimeError, "不支持 grpc_port"):
            load_android_world_environment(
                types.SimpleNamespace(load_and_setup_env=old_launcher),
                AndroidWorldConfig(console_port=5556, grpc_port=8555),
            )
        self.assertEqual(called, [])

    def test_new_launcher_explicit_keyword_is_supported(self):
        def launcher(*, console_port, grpc_port, adb_path, emulator_setup):
            return console_port, grpc_port

        self.assertEqual(load_android_world_environment(
            types.SimpleNamespace(load_and_setup_env=launcher),
            AndroidWorldConfig(console_port=5560, grpc_port=8557),
        ), (5560, 8557))

    def test_changed_grpc_port_cannot_resume_previous_device_results(self):
        config = load_config(EXAMPLE)
        config.android_world = AndroidWorldConfig(console_port=5556, grpc_port=8554)
        with tempfile.TemporaryDirectory() as directory:
            options = dict(mode="standalone", profiles=[], tasks=None, combinations=1,
                           seed=42, family="android_world", max_steps=30)
            first = _evaluation_progress(Path(directory), config, resume=False, **options)
            before = first.manifest_path.read_bytes()
            specification = json.loads(before)["specification"]
            self.assertEqual(specification["android_world"]["grpc_port"], 8554)
            config.android_world = dataclasses.replace(config.android_world, grpc_port=8555)
            with self.assertRaisesRegex(ValueError, "android_world"):
                _evaluation_progress(Path(directory), config, resume=True, **options)
            self.assertEqual(first.manifest_path.read_bytes(), before)

    def test_qwen_vdroid_request_inherits_main_config_grpc(self):
        config = load_config(EXAMPLE)
        config.android_world = AndroidWorldConfig(console_port=5556, grpc_port=8555)
        profile = ModelProfile("adapter", "adapter", "http://unused/v1", {}, metadata={
            "base_model_path": "/model", "evaluation_checkpoint": "/model/checkpoint",
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vdroid" / "android_world" / "agents" / "vdroid.py"
            source.parent.mkdir(parents=True)
            source.touch()
            args = build_parser().parse_args([
                "evaluate-vdroid-qwen", "--adapter-path", "unused", "--vdroid-root", str(root / "vdroid"),
                "--output-dir", str(root / "run"), "--no-store-trajectories",
            ])
            request = evaluator_for_adapter(config, profile, args, None).request(root / "run")
            self.assertEqual(request["device"]["console_port"], 5556)
            self.assertEqual(request["device"]["grpc_port"], 8555)
            self.assertEqual(request["device"]["device_name"], "emulator-5556")

    def test_dry_run_shows_device_binding_separately_from_model_port(self):
        config = load_config(EXAMPLE)
        config.android_world = AndroidWorldConfig(console_port=5556, grpc_port=8555)
        deployment = mock.Mock(config=config)
        deployment.build_adapter_command.return_value = ["deploy", "--port", "8003"]
        deployment.profiles.return_value = []
        args = build_parser().parse_args(["evaluate-standalone", "--adapter-path", "unused"])
        with mock.patch("src1.pmtskill_v2.cli._print") as output:
            _dry_evaluation_result("standalone", deployment, [], args, resume=False, store_trajectories=False)
        payload = output.call_args.args[0]
        self.assertEqual(payload["android_world_connection"], {"console_port": 5556, "grpc_port": 8555})
        self.assertEqual(payload["deployment_command"][-1], "8003")

    def test_both_collectors_forward_nondefault_grpc_before_any_task_runs(self):
        config = load_config(EXAMPLE)
        config.android_world = AndroidWorldConfig(console_port=5556, grpc_port=8555)

        class StopAfterBinding(Exception):
            pass

        for kind in ("teacher", "optimization"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                launch = mock.Mock(side_effect=StopAfterBinding)
                android = types.ModuleType("android_world")
                android.registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(
                    get_registry=lambda family: {"ContactsAddContact": object()},
                ))
                android.suite_utils = mock.Mock()
                android.checkpointer = mock.Mock()
                agents = types.ModuleType("android_world.agents")
                agents.m3a = mock.Mock()
                env = types.ModuleType("android_world.env")
                env.env_launcher = types.SimpleNamespace(load_and_setup_env=launch)
                with mock.patch.dict(sys.modules, {
                    "android_world": android, "android_world.agents": agents, "android_world.env": env,
                }), mock.patch("src1.pmtskill_v2.offline.collector.bootstrap_android_world"), mock.patch(
                    "src1.pmtskill_v2.offline.optimization_collector.bootstrap_android_world",
                ), self.assertRaises(StopAfterBinding):
                    if kind == "teacher":
                        collect_teacher_trajectories(config, output_dir=Path(directory))
                    else:
                        collector = AndroidWorldOptimizationTraceCollector(config, mock.Mock(), planner=mock.Mock())
                        collector.run(config.models[0], OptimizationTraceCollectionOptions(
                            source_kind="task", output_dir=Path(directory), tasks=("ContactsAddContact",),
                        ))
                launch.assert_called_once_with(
                    console_port=5556, grpc_port=8555, emulator_setup=False, adb_path="adb",
                )


if __name__ == "__main__":
    unittest.main()
