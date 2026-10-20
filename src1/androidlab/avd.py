"""Native AVD backend: immutable template -> disposable copy -> owned emulator.

Never attach to or stop an existing host emulator. Every ADB call names and
verifies the newly started serial. No Docker, host shell, wipe-data, snapshot
save, global kill-server, or edits to the user's downloaded AVD are used.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .catalog import file_digest
from .environment import DockerDevice, InfrastructureError
from ..pmtskill_v2.evaluation.resume import fingerprint


def read_ini(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith(("#", ";")):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _ignored(relative, snapshot):
    return (any(part.endswith(".lock") for part in relative.parts)
            or relative.name in {"hardware-qemu.ini", "emulator-user.ini", "quickbootChoice.ini"}
            or (not snapshot and ("snapshots" in relative.parts or relative.name == "snapshots.img")))


def inspect_avd(options, *, hash_images=False):
    """Read-only validation; never start or reset an emulator during preflight."""
    if options.sdk_root is None or options.avd_home is None:
        raise ValueError("Native AndroidLab needs [androidlab].sdk_root and avd_home; see ANDROIDLAB_EVALUATION.md")
    sdk, home = options.sdk_root.resolve(), options.avd_home.resolve()
    directory, descriptor = home / (options.avd_name + ".avd"), home / (options.avd_name + ".ini")
    if directory.is_symlink() or descriptor.is_symlink():
        raise ValueError("AVD template/descriptor must not be symlinks")
    for path in (directory / "config.ini", descriptor):
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded AndroidLab AVD is incomplete: {path}")
    config, entry = read_ini(directory / "config.ini"), read_ini(descriptor)
    suffix = ".exe" if os.name == "nt" else ""
    adb, emulator = sdk / "platform-tools" / ("adb" + suffix), sdk / "emulator" / ("emulator" + suffix)
    for path in (adb, emulator):
        if not path.is_file():
            raise FileNotFoundError(f"Android SDK executable missing: {path}")
    architecture = config.get("abi.type", "")
    image_path = config.get("image.sysdir.1", "").replace("\\", "/")
    parts = [part for part in image_path.split("/") if part]
    if not architecture and parts:
        architecture = parts[-1]
    host = platform.machine().lower()
    expected = "x86_64" if host in {"amd64", "x86_64"} else "arm64-v8a" if host in {"arm64", "aarch64"} else None
    if not expected or architecture != expected:
        raise ValueError(f"AVD architecture mismatch: host={host}, AVD={architecture}; "
                         "the official macOS ARM64 AVD is not a Linux x86_64 AVD")
    api = next((part for part in parts if re.fullmatch(r"android-\d+", part)), entry.get("target", ""))
    if api != "android-33":
        raise ValueError(f"Expected official AndroidLab API 33, found {api!r}; do not use an AndroidWorld AVD")
    variant = parts[parts.index(api) + 1] if api in parts and parts.index(api) + 1 < len(parts) else "google_apis"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", variant):
        raise ValueError("Invalid Android system image variant")
    system = sdk / "system-images" / api / variant / architecture
    if not system.is_dir() or not (system / "system.img").is_file():
        raise FileNotFoundError(f"Matching Android system image missing: {system}; "
                                f"install system-images;{api};{variant};{architecture}")
    userdata = directory / "userdata-qemu.img"
    if not userdata.is_file() or userdata.stat().st_size == 0:
        raise ValueError("Pre-populated userdata-qemu.img missing/empty; a blank AVD is not the benchmark")
    if options.avd_snapshot and not (directory / "snapshots" / options.avd_snapshot).is_dir():
        raise FileNotFoundError(f"Requested baseline snapshot missing: {directory / 'snapshots' / options.avd_snapshot}")
    if options.avd_work_dir and options.avd_work_dir.resolve().is_relative_to(directory.resolve()):
        raise ValueError("avd_work_dir must not be inside the immutable AVD template")
    records = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.endswith(".lock") for part in relative.parts):
            raise ValueError(f"Template contains an emulator lock: {path}; verify it is not in use before cleanup")
        if path.is_symlink():
            raise ValueError(f"AVD template symlink is not supported: {path}")
        if not path.is_file() or _ignored(relative, options.avd_snapshot):
            continue
        stamp = path.stat()
        record = {"path": relative.as_posix(), "size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}
        if hash_images:
            if stamp.st_size > 100 * 1024 * 1024:
                print(f"[AndroidLab] hashing AVD baseline {relative} ({stamp.st_size / 1024**3:.2f} GiB)", flush=True)
            record["sha256"] = file_digest(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
                raise InfrastructureError(f"AVD template changed while hashing: {path}")
        records.append(record)
    if config.get("hw.sdCard.path") or config.get("sdcard.path"):
        if not (directory / "sdcard.img").is_file():
            raise ValueError("AVD references an external/missing SD card; include its original sdcard.img in the template")
    runtime_files = [descriptor, adb, emulator, *sorted(path for path in system.rglob("*") if path.is_file())]
    runtime = []
    for path in runtime_files:
        if path.is_symlink():
            raise ValueError(f"SDK/system image symlink is not supported: {path}")
        stamp = path.stat()
        record = {"path": str(path), "size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}
        if hash_images:
            record["sha256"] = file_digest(path)
            if (path.stat().st_size, path.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
                raise InfrastructureError(f"Android SDK changed while hashing: {path}")
        runtime.append(record)
    return {"backend": "avd", "template": str(directory), "descriptor": str(descriptor),
            "system_image": str(system), "architecture": architecture, "api": api,
            "adb": str(adb), "emulator": str(emulator), "files": records, "runtime_files": runtime,
            "snapshot": options.avd_snapshot, "reset": "disposable-template-copy-per-episode",
            "content_identity": fingerprint({"files": records, "runtime": runtime}) if hash_images else None}


def verify_identity(identity):
    """Cheap stat checks before/after each clone; initial run hashes file contents."""
    source = Path(identity["template"])
    if source.is_symlink():
        raise InfrastructureError("AVD template was replaced with a symlink")
    inventory = set()
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_symlink() or any(part.endswith(".lock") for part in relative.parts):
            raise InfrastructureError(f"AVD template became active or contains a symlink: {path}")
        if path.is_file() and not _ignored(relative, identity.get("snapshot")):
            inventory.add(relative.as_posix())
    if inventory != {record["path"] for record in identity["files"]}:
        raise InfrastructureError("Immutable AVD template file inventory changed")
    for record in identity["files"]:
        path = source / record["path"]
        if path.is_symlink() or not path.is_file():
            raise InfrastructureError(f"Immutable AVD template changed: {path}")
        stamp = path.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise InfrastructureError(f"Immutable AVD template changed: {path}; use a new experiment directory")
    for record in identity["runtime_files"]:
        path = Path(record["path"])
        if path.is_symlink() or not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise InfrastructureError(f"Android SDK/AVD descriptor changed: {path}")


def guest_query(command):
    """Translate official read-only adb queries/pipelines into a guest shell, not a host shell."""
    if not command.startswith("adb shell ") or any(token in command for token in ("\n", "\r", "`", "$", ";", "&", ">", "<")):
        raise ValueError("Only official read-only adb shell queries are supported")
    parts = shlex.split(command[len("adb shell "):])
    if len(parts) == 1:
        parts = shlex.split(parts[0])  # Official setting_14 wraps the whole command in quotes.
    stages = [[]]
    for part in parts:
        if part == "|":
            stages.append([])
        else:
            stages[-1].append(part)
    first = stages[0]
    allowed = bool(first) and (first[0] == "getprop" or first[:2] in (["settings", "get"], ["settings", "list"])
                              or first[:3] == ["pm", "list", "packages"])
    if not allowed or len(stages) > 2 or any(not stage or stage[0] != "grep" for stage in stages[1:]):
        raise ValueError(f"Unsupported official read-only adb query: {command}")
    return " | ".join(shlex.join(stage) for stage in stages)


class AvdDevice:
    # These two operations are transport-independent and contain only official
    # app initialization/XML collection; they never invoke Docker themselves.
    initialize_task = DockerDevice.initialize_task
    xml = DockerDevice.xml

    def __init__(self, options, identity, log_dir):
        self.options, self.identity, self.log_dir = options, identity, Path(log_dir)
        self.serial = f"emulator-{options.console_port}"
        self.name = "PMTAndroidLab_" + uuid.uuid4().hex[:16]
        self.nonce = uuid.uuid4().hex
        self.workspace, self.process, self.log = None, None, None

    def _environment(self):
        result = os.environ.copy()
        result["ANDROID_HOME"] = str(self.options.sdk_root)
        result["ANDROID_SDK_ROOT"] = str(self.options.sdk_root)
        result["ANDROID_AVD_HOME"] = str(self.workspace)
        return result

    def _copy_file(self, source, target):
        if platform.system() == "Linux" and shutil.which("cp"):
            subprocess.run(["cp", "--reflink=auto", "--sparse=always", "--", str(source), str(target)],
                           check=True, capture_output=True, timeout=self.options.clone_timeout_seconds)
            shutil.copystat(source, target)
            return str(target)
        return shutil.copy2(source, target)

    def _clone(self):
        verify_identity(self.identity)
        root = self.options.avd_work_dir
        if root is None:
            raise ValueError("Native AVD needs avd_work_dir on a filesystem with sufficient free space")
        root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix="pmtskill-androidlab-", dir=root)).resolve()
        (self.workspace / ".pmtskill-owned.json").write_text(json.dumps({"nonce": self.nonce}), encoding="utf-8")
        source = Path(self.identity["template"])
        target = self.workspace / (self.name + ".avd")
        def ignore(directory, names):
            return [name for name in names if _ignored((Path(directory) / name).relative_to(source), self.options.avd_snapshot)]
        print(f"[AndroidLab] preparing disposable AVD copy: {self.name} (template is not modified)", flush=True)
        shutil.copytree(source, target, copy_function=self._copy_file, ignore=ignore)
        verify_identity(self.identity)
        config = read_ini(target / "config.ini")
        replacements = {str(source): str(target), "[ANDROID_AVD_HOME]": str(self.workspace),
                        "[ANDROID_SDK_HOME]": str(self.options.sdk_root)}
        old_path = read_ini(self.identity["descriptor"]).get("path")
        if old_path:
            replacements[old_path] = str(target)
        for key, value in list(config.items()):
            for old, new in replacements.items():
                value = value.replace(old, new)
            # Official Linux exports can retain the original machine's SDK path.
            marker = f"system-images/{self.identity['api']}/"
            normalized = value.replace("\\", "/")
            if marker in normalized:
                relative = normalized[normalized.index(marker):]
                value = str(self.options.sdk_root / relative)
            config[key] = value
        config.update(AvdId=self.name, **{"avd.ini.displayname": self.name,
                      "image.sysdir.1": self.identity["system_image"],
                      "disk.dataPartition.path": str(target / "userdata-qemu.img"),
                      "disk.cachePartition.path": str(target / "cache.img")})
        if "disk.encryptionKeyPartition.path" in config:
            config["disk.encryptionKeyPartition.path"] = str(target / "encryptionkey.img")
        for key in ("hw.sdCard.path", "sdcard.path"):
            if key in config:
                config[key] = str(target / "sdcard.img")
        if "skin.path" in config and not Path(config["skin.path"]).exists():
            config.pop("skin.path")
            config["showDeviceFrame"] = "no"
        (target / "config.ini").write_text("\n".join(f"{key}={value}" for key, value in config.items()) + "\n", encoding="utf-8")
        (self.workspace / (self.name + ".ini")).write_text(
            f"avd.ini.encoding=UTF-8\npath={target}\npath.rel=avd/{self.name}.avd\ntarget=android-33\n", encoding="utf-8")

    def build_command(self):
        data = self.workspace / (self.name + ".avd")
        command = [self.identity["emulator"], "-avd", self.name, "-port", str(self.options.console_port),
                   "-sysdir", self.identity["system_image"], "-datadir", str(data),
                   "-data", str(data / "userdata-qemu.img"), "-no-snapshot-save", "-no-audio",
                   "-no-snapshot-load" if self.options.avd_snapshot else "-no-snapshot",
                   "-gpu", self.options.avd_gpu_mode]
        if self.options.avd_headless:
            command.append("-no-window")
        return command

    def __enter__(self):
        for port in (self.options.console_port, self.options.console_port + 1):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise InfrastructureError(f"Native AVD port {port} is occupied; refusing to touch an existing emulator")
        try:
            self._clone()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / (self.name + ".log")
            self.log = log_path.open("wb")
            self.process = subprocess.Popen(self.build_command(), env=self._environment(),
                                            stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=os.name != "nt")
            self._wait_ready()
            if self.options.avd_snapshot:
                output = self.adb("emu", "avd", "snapshot", "load", self.options.avd_snapshot)
                if "OK" not in output or re.search(r"\b(KO|error|failed)\b", output, flags=re.I):
                    raise InfrastructureError(f"Baseline snapshot restore failed; refusing cold-boot fallback: {output}")
                self._wait_ready()
            print(f"[AndroidLab] native AVD ready: {self.serial}; log={log_path}", flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _adb_raw(self, *args, binary=False):
        try:
            result = subprocess.run([self.identity["adb"], "-s", self.serial, *map(str, args)],
                                    capture_output=True, timeout=self.options.command_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError(f"Native AndroidLab ADB unavailable ({self.serial}): {exc}") from exc
        if result.returncode:
            raise InfrastructureError(result.stderr.decode("utf-8", errors="replace")[-1500:])
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace").strip()

    def _verify_binding(self):
        if self.process is None or self.process.poll() is not None:
            raise InfrastructureError("Owned AndroidLab emulator exited; refusing any further ADB operation")
        lines = self._adb_raw("emu", "avd", "name").splitlines()
        name = lines[0].strip() if lines else ""
        if name != self.name:
            raise InfrastructureError(f"ADB serial {self.serial} belongs to {name!r}, not owned AVD {self.name!r}")

    def _wait_ready(self):
        deadline = time.monotonic() + self.options.startup_timeout_seconds
        last = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise InfrastructureError(f"Native emulator exited (code={self.process.returncode}); see {self.log_dir}")
            try:
                self._verify_binding()
                booted = self._adb_raw("shell", "getprop", "sys.boot_completed")
                if booted == "1":
                    api = self._adb_raw("shell", "getprop", "ro.build.version.sdk")
                    abi = self._adb_raw("shell", "getprop", "ro.product.cpu.abi")
                    if api != "33" or abi != self.identity["architecture"]:
                        raise ValueError(f"Booted AVD is not the selected AndroidLab system: API={api}, ABI={abi}")
                    return
            except InfrastructureError as exc:
                last = exc
            time.sleep(0.5)
        raise InfrastructureError(f"Native AndroidLab boot/restore timed out: {last}; see {self.log_dir}")

    def adb(self, *args, binary=False):
        # adbd restarts after `adb root`. Waiting is read-only and must be
        # allowed before the console-name check can succeed again.
        if args == ("wait-for-device",):
            output = self._adb_raw(*args, binary=binary)
            self._verify_binding()
            return output
        self._verify_binding()
        return self._adb_raw(*args, binary=binary)

    def query(self, trusted_command):
        return self.adb("shell", guest_query(trusted_command))

    def __exit__(self, *exc):
        if self.process is not None:
            process = self.process
            self.process = None
            if process.poll() is None:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.workspace is not None:
            workspace = self.workspace
            self.workspace = None
            root = self.options.avd_work_dir.resolve()
            if (workspace.is_symlink() or not workspace.resolve().is_relative_to(root)
                    or workspace.resolve() == root or not workspace.name.startswith("pmtskill-androidlab-")):
                raise InfrastructureError(f"Refusing unsafe AVD workspace cleanup: {workspace}")
            marker = json.loads((workspace / ".pmtskill-owned.json").read_text(encoding="utf-8"))
            if marker.get("nonce") != self.nonce:
                raise InfrastructureError("AVD workspace ownership marker changed; refusing cleanup")
            shutil.rmtree(workspace)
