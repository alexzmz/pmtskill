"""Native backend regression tests: no emulator, Docker, GPU or network required."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src1.androidlab import METHODS
from src1.androidlab.avd import AvdDevice, guest_query, inspect_avd, read_ini, verify_identity
from src1.androidlab.config import AndroidLabConfig
from src1.androidlab.driver import command_inspect_avd, command_evaluate
from src1.androidlab.environment import InfrastructureError
from src1.androidlab.worker import make_device
from src1.compare_androidlab import build_parser as batch_parser, build_commands
from src1.pmtskill_v2.cli import build_parser


def fixture(directory, **overrides):
    root = Path(directory)
    source = root / "templates/Test.avd"
    source.mkdir(parents=True)
    (source / "config.ini").write_text(
        'AvdId=Test\nabi.type=x86_64\nimage.sysdir.1=system-images/android-33/google_apis/x86_64/\n'
        'skin.path=/missing/skin\n', encoding="utf-8")
    (source / "userdata-qemu.img").write_bytes(b"pre-populated benchmark data")
    (source.parent / "Test.ini").write_text('target=android-33\npath=/original/Test.avd\n', encoding="utf-8")
    suffix = ".exe" if os.name == "nt" else ""
    for relative in ("platform-tools/adb" + suffix, "emulator/emulator" + suffix,
                     "system-images/android-33/google_apis/x86_64/system.img"):
        path = root / "sdk" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return AndroidLabConfig(root=root / "lab", sdk_root=root / "sdk", avd_home=source.parent,
                            avd_work_dir=root / "work", avd_name="Test", **overrides)


class NativeAvdTest(unittest.TestCase):
    def inspect(self, options, **kwargs):
        with mock.patch("src1.androidlab.avd.platform.machine", return_value="x86_64"):
            return inspect_avd(options, **kwargs)

    def test_named_commands_fix_method_and_allow_single_task(self):
        parser = build_parser()
        for method in METHODS:
            args = parser.parse_args([f"evaluate-androidlab-{method}", "--model-id", "control",
                                      "--tasks", "clock_1", "--output-dir", "out"])
            self.assertEqual(args.method, method)
            self.assertEqual(args.tasks, ["clock_1"])
            self.assertEqual(args.family, "android_lab")

    def test_comparison_defaults_to_standalone_only(self):
        args = batch_parser().parse_args(["--config", "unused", "--output-dir", "out"])
        commands = build_commands(args, {"suite_file": "suite.json", "skill_database": None})
        self.assertEqual(list(commands), ["standalone"])

    def test_inspect_is_read_only_and_hashes_real_template(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            with mock.patch("subprocess.run") as run:
                identity = self.inspect(options, hash_images=True)
                verify_identity(identity)
                self.assertEqual(identity["architecture"], "x86_64")
                self.assertTrue(identity["content_identity"])
                self.assertFalse(options.avd_work_dir.exists())
                run.assert_not_called()

    def test_read_only_inspect_command_never_starts_device(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            args = build_parser().parse_args(["inspect-androidlab-avd"])
            with mock.patch("src1.androidlab.driver.AndroidLabConfig.load", return_value=options), \
                 mock.patch("src1.androidlab.avd.platform.machine", return_value="x86_64"), \
                 mock.patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(command_inspect_avd(args), 0)
                self.assertFalse(json.loads(output.getvalue())["emulator_started"])
                run.assert_not_called()

    def test_arm_image_on_x86_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            config = options.avd_home / "Test.avd/config.ini"
            config.write_text(config.read_text().replace("x86_64", "arm64-v8a"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                self.inspect(options)

    def test_blank_avd_or_wrong_android_version_rejected(self):
        for invalid in ("empty", "API"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                options = fixture(directory)
                source = options.avd_home / "Test.avd"
                if invalid == "empty":
                    (source / "userdata-qemu.img").write_bytes(b"")
                else:
                    config = source / "config.ini"
                    config.write_text(config.read_text().replace("android-33", "android-34"), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.inspect(options)

    def test_active_template_and_unsafe_work_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            lock = options.avd_home / "Test.avd/multiinstance.lock"
            lock.write_text("active", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lock"):
                self.inspect(options)
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "inside"):
                self.inspect(dataclasses.replace(options, avd_work_dir=options.avd_home / "Test.avd/work"))

    def test_inventory_and_content_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            identity = self.inspect(options)
            extra = options.avd_home / "Test.avd/unexpected.img"
            extra.write_bytes(b"new")
            with self.assertRaisesRegex(InfrastructureError, "inventory changed"):
                verify_identity(identity)
            extra.unlink()
            (options.avd_home / "Test.avd/userdata-qemu.img").write_bytes(b"changed")
            with self.assertRaisesRegex(InfrastructureError, "template changed"):
                verify_identity(identity)

    def test_clone_rewrites_only_disposable_copy_and_never_wipes(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            source = options.avd_home / "Test.avd/config.ini"
            before = source.read_bytes()
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            try:
                device._clone()
                work = device.workspace
                clone = work / (device.name + ".avd")
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual((clone / "userdata-qemu.img").read_bytes(), b"pre-populated benchmark data")
                self.assertEqual(read_ini(clone / "config.ini")["AvdId"], device.name)
                command = device.build_command()
                self.assertIn("-no-snapshot", command)
                self.assertIn("-no-snapshot-save", command)
                self.assertNotIn("-wipe-data", command)
                self.assertEqual(command[command.index("-datadir") + 1], str(clone))
                self.assertEqual(device._environment()["ANDROID_AVD_HOME"], str(work))
            finally:
                device.__exit__()
            self.assertFalse(work.exists())
            self.assertTrue(source.is_file())

    def test_missing_explicit_snapshot_fails_not_cold_boot_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory, avd_snapshot="baseline")
            with self.assertRaisesRegex(FileNotFoundError, "snapshot missing"):
                self.inspect(options)

    def test_failed_explicit_snapshot_restore_cleans_owned_device(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory, avd_snapshot="baseline")
            (options.avd_home / "Test.avd/snapshots/baseline").mkdir(parents=True)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            with mock.patch("src1.androidlab.avd.socket.socket") as socket, \
                 mock.patch.object(device, "_wait_ready"), \
                 mock.patch.object(device, "adb", return_value="KO: snapshot load failed"), \
                 mock.patch("src1.androidlab.avd.subprocess.Popen"), mock.patch.object(device, "__exit__") as cleanup:
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                with self.assertRaisesRegex(InfrastructureError, "refusing cold-boot fallback"):
                    device.__enter__()
                cleanup.assert_called_once()
            if device.log:
                device.log.close()

    def test_occupied_port_is_not_attached_to_or_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            with mock.patch("src1.androidlab.avd.socket.socket") as socket, mock.patch.object(device, "_clone") as clone:
                socket.return_value.__enter__.return_value.connect_ex.return_value = 0
                with self.assertRaisesRegex(InfrastructureError, "occupied"):
                    device.__enter__()
                clone.assert_not_called()
                self.assertIsNone(device.process)

    def test_adb_serial_and_owned_avd_name_checked_for_every_action(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            device.process = mock.Mock(poll=mock.Mock(return_value=None))
            with mock.patch("src1.androidlab.avd.subprocess.run") as run:
                run.side_effect = [subprocess.CompletedProcess([], 0, (device.name + "\nOK\n").encode(), b""),
                                   subprocess.CompletedProcess([], 0, b"\x89PNG\x00", b"")]
                self.assertEqual(device.adb("exec-out", "screencap", "-p", binary=True), b"\x89PNG\x00")
                for call in run.call_args_list:
                    self.assertEqual(call.args[0][1:3], ["-s", "emulator-5562"])
                run.side_effect = None
                run.return_value = subprocess.CompletedProcess([], 0, b"unrelated-AVD\nOK", b"")
                with self.assertRaisesRegex(InfrastructureError, "not owned"):
                    device.adb("shell", "input", "tap", "1", "2")
                run.return_value = subprocess.CompletedProcess([], 0, b"", b"")
                with self.assertRaises(InfrastructureError):
                    device._verify_binding()

    def test_cleanup_checks_workspace_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            device._clone()
            workspace = device.workspace
            (workspace / ".pmtskill-owned.json").write_text('{"nonce":"not-ours"}', encoding="utf-8")
            with mock.patch("src1.androidlab.avd.shutil.rmtree") as delete:
                with self.assertRaisesRegex(InfrastructureError, "ownership"):
                    device.__exit__()
                delete.assert_not_called()
                self.assertTrue(workspace.is_dir())

    def test_only_read_only_official_queries_are_allowed(self):
        self.assertEqual(guest_query('adb shell "getprop persist.sys.timezone"'), "getprop persist.sys.timezone")
        self.assertEqual(guest_query("adb shell settings list system | grep screen_brightness"),
                         "settings list system | grep screen_brightness")
        for command in ("adb shell settings put global airplane_mode_on 1", "adb shell getprop; touch bad",
                        "adb shell getprop | sh", "adb shell $(rm wrong)", "adb shell pm clear app"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                guest_query(command)

    def test_native_factory_never_inspects_or_creates_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            identity = self.inspect(options)
            with mock.patch("src1.androidlab.worker.DockerDevice") as docker:
                device = make_device(options, {"device_identity": identity}, "logs")
                self.assertIsInstance(device, AvdDevice)
                docker.assert_not_called()
            with self.assertRaisesRegex(ValueError, "backend changed"):
                make_device(options, {"device_identity": {"backend": "docker"}}, "logs")

    def test_native_standalone_orchestration_passes_only_selected_task_without_docker(self):
        from src1.tests.test_androidlab import fake_source
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            root = Path(directory)
            fake_source(options.root)
            agent = root / "aw/android_world/agents/m3a.py"
            agent.parent.mkdir(parents=True)
            agent.write_text("# agent", encoding="utf-8")
            config = root / "c.toml"
            config.write_text('[paths]\nandroid_world_root="aw"\n[androidlab]\nroot="lab"\n'
                              'sdk_root="sdk"\navd_home="templates"\navd_name="Test"\navd_work_dir="work"\n'
                              '[[models]]\nmodel_id="control"\nbase_url="http://127.0.0.1:8002/v1"\n', encoding="utf-8")
            target = root / "out"
            args = build_parser().parse_args(["--config", str(config), "evaluate-androidlab-standalone",
                                              "--model-id", "control", "--tasks", "clock_1",
                                              "--output-dir", str(target)])
            with mock.patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}), \
                 mock.patch("src1.androidlab.avd.platform.machine", return_value="x86_64"), \
                 mock.patch("src1.androidlab.driver.run_worker", return_value=0) as worker, \
                 mock.patch("src1.androidlab.driver.docker_image_identity") as docker, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(command_evaluate(args), 0)
                docker.assert_not_called()
                self.assertEqual(worker.call_count, 2)
                self.assertTrue(worker.call_args_list[0].kwargs["check"])
            request = json.loads((target / "worker_request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["device_identity"]["backend"], "avd")
            self.assertTrue(request["device_identity"]["content_identity"])
            self.assertEqual([r["task_id"] for r in request["tasks"]], ["clock_1"])
            self.assertEqual(request["metadata"]["model_ids"], ["control"])
            self.assertIsNone(request["skill_database"])

    def test_adbd_reconnect_waits_before_read_only_binding_check(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            events = []
            with mock.patch.object(device, "_adb_raw", side_effect=lambda *a, **kw: events.append(a) or "ready"), \
                 mock.patch.object(device, "_verify_binding", side_effect=lambda: events.append("binding")):
                self.assertEqual(device.adb("wait-for-device"), "ready")
                self.assertEqual(events, [("wait-for-device",), "binding"])

    def test_cleanup_stops_only_owned_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            options = fixture(directory)
            device = AvdDevice(options, self.inspect(options), Path(directory) / "logs")
            process = mock.Mock(pid=43210, poll=mock.Mock(return_value=None))
            device.process = process
            with mock.patch("src1.androidlab.avd.os.name", "posix"), \
                 mock.patch("src1.androidlab.avd.os.killpg", create=True) as stop:
                device.__exit__()
                self.assertEqual(stop.call_args.args[0], 43210)
                self.assertEqual(stop.call_count, 1)
                process.wait.assert_called_once_with(timeout=20)

    def test_config_defaults_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.toml"
            path.write_text('[androidlab]\nsdk_root="sdk"\navd_home="avd"\n', encoding="utf-8")
            options = AndroidLabConfig.load(path)
            self.assertEqual(options.backend, "avd")
            self.assertEqual(options.sdk_root, (path.parent / "sdk").resolve())
            for entry in ('console_port=5555', 'backend="unknown"', 'avd_name="../x"',
                          'avd_name=3', 'avd_snapshot="../x"', 'clone_timeout_seconds=0', 'avd_headless=1'):
                path.write_text('[androidlab]\n' + entry + '\n', encoding="utf-8")
                with self.subTest(entry=entry), self.assertRaises(ValueError):
                    AndroidLabConfig.load(path)


if __name__ == "__main__":
    unittest.main()
