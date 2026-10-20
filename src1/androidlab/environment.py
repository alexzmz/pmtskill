"""Bounded official-snapshot Docker lifecycle and an AndroidWorld agent bridge.

No host ADB device is touched. Only the newly created, labelled container is
removed. Model output is parsed as JSON actions, never executed as Python/shell.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace


class InfrastructureError(RuntimeError):
    pass


class JudgeError(RuntimeError):
    pass


class StepBudgetReached(BaseException):
    """Escape upstream broad Exception handlers without fabricating an action."""


def docker_image_identity(image):
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                            check=True, capture_output=True, text=True, timeout=30)
    value = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise InfrastructureError("Cannot identify the AndroidLab Docker image")
    return value


class DockerDevice:
    """A fresh official image per task; timeout and cleanup are always bounded."""

    adb_binary = "/root/.android/platform-tools/adb"

    def __init__(self, options, image_id):
        self.options, self.image_id = options, image_id
        self.container_id, self.serial = None, None

    def _docker(self, *args, binary=False, timeout=None):
        try:
            result = subprocess.run(["docker", *map(str, args)], capture_output=True,
                                    timeout=timeout or self.options.command_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError(f"AndroidLab Docker operation unavailable: {exc}") from exc
        if result.returncode:
            raise InfrastructureError(result.stderr.decode("utf-8", errors="replace")[-2000:])
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace").strip()

    def _http(self, path, data, timeout):
        request = urllib.request.Request(f"http://127.0.0.1:{self.options.docker_port}/{path}",
                                         data=json.dumps(data).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        if result.get("error") or result.get("result") == "ERROR":
            raise InfrastructureError(f"AndroidLab device service error: {result}")
        return result

    def __enter__(self):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.options.docker_port)) == 0:
                raise InfrastructureError(f"AndroidLab Docker port {self.options.docker_port} is already occupied")
        try:
            self.container_id = self._docker(
                "run", "-i", "-t", "-d", "--privileged", "--label", "pmtskill.androidlab=true",
                "-p", f"127.0.0.1:{self.options.docker_port}:6060", self.image_id)
            if not re.fullmatch(r"[0-9a-f]{64}", self.container_id):
                self.container_id = None
                raise InfrastructureError("Docker did not return a valid owned container ID")
            self._docker("cp", self.options.root / "adb_client.py", f"{self.container_id}:/adb_client.py")
            self._docker("exec", "-d", self.container_id, "/usr/local/bin/python", "/adb_client.py")
            deadline = time.monotonic() + self.options.startup_timeout_seconds
            while True:
                try:
                    self._http("execute", {"command": "adb version"}, timeout=3)
                    break
                except (OSError, ValueError, InfrastructureError):
                    if time.monotonic() >= deadline:
                        raise InfrastructureError("AndroidLab adb_client startup timed out")
                    time.sleep(0.5)
            response = self._http("start", {"avd_name": self.options.avd_name},
                                  timeout=max(1, deadline - time.monotonic()))
            self.serial = response.get("device")
            if not self.serial or not re.fullmatch(r"emulator-\d+", self.serial):
                raise InfrastructureError(f"AndroidLab emulator did not boot: {response}")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self.container_id:
            owned = self.container_id
            self.container_id = None
            self._docker("rm", "-f", owned, timeout=30)

    def adb(self, *args, binary=False):
        return self._docker("exec", self.container_id, self.adb_binary, "-s", self.serial,
                            *args, binary=binary)

    def query(self, trusted_command):
        # Only official frozen task metadata enters this method, never LLM text.
        if not trusted_command.startswith("adb ") or "\n" in trusted_command:
            raise InfrastructureError("Invalid official adb_query")
        command = "adb -s " + self.serial + trusted_command[3:]
        return self._http("execute", {"command": command},
                          timeout=self.options.command_timeout_seconds)["result"]

    def initialize_task(self, task):
        self.adb("root")
        self.adb("wait-for-device")
        self.adb("emu", "geo", "fix", "-122.156", "37.438")
        if "map.me" not in task["app"].lower():
            self.adb("shell", 'date "2024-05-10 12:00:00"')
        installed = self.adb("shell", "pm", "path", task["package"])
        if not installed.startswith("package:"):
            raise InfrastructureError(f"Official AndroidLab app missing: {task['package']}")
        # Match official AutoTest's in_app initialization, without any task answer.
        output = self.adb("shell", "monkey", "-p", task["package"],
                          "-c", "android.intent.category.LAUNCHER", "1")
        if "No activities" in output or "Error" in output:
            raise InfrastructureError(f"Could not launch AndroidLab app: {output}")
        time.sleep(self.options.app_start_wait_seconds)

    def xml(self, use_accessibility):
        if use_accessibility:
            path = "/sdcard/Android/data/com.example.android.xml_parser/files/ui.xml"
            # Same freshness criterion as official check_ac_survive; never reuse a stale file.
            modified = int(self.adb("shell", "stat", "-c", "%Y", path))
            current = int(self.adb("shell", "date", "+%s"))
            if abs(current - modified) > 10:
                raise InfrastructureError("AndroidLab accessibility XML is stale (>10s)")
        else:
            path = "/sdcard/pmtskill-androidlab-ui.xml"
            self.adb("shell", "uiautomator", "dump", path)
        value = self.adb("exec-out", "cat", path)
        ET.fromstring(value)  # Validate before offering a possibly empty observation.
        return value


class AccessibilityNode(SimpleNamespace):
    def HasField(self, name):  # V-Droid's tree formatter uses the protobuf interface.
        return getattr(self, name, None) is not None


def xml_forest(xml):
    """Keep ancestry + pre-order IDs so V-Droid and SoM share display indices."""
    root = ET.fromstring(xml)
    elements = [element for element in root.iter() if "bounds" in element.attrib]
    if not elements:
        raise InfrastructureError("AndroidLab XML has no bounded accessibility nodes")
    ids = {id(element): index for index, element in enumerate(elements)}
    nodes = []
    for element in elements:
        values = list(map(int, re.findall(r"-?\d+", element.get("bounds", ""))))
        if len(values) != 4:
            raise InfrastructureError("Invalid AndroidLab accessibility bounds")
        x1, y1, x2, y2 = values
        attributes = dict(text=element.get("text", ""), content_description=element.get("content-desc", ""),
                          class_name=element.get("class", ""), hint_text=element.get("hint", ""),
                          package_name=element.get("package", ""),
                          view_id_resource_name=element.get("resource-id", ""),
                          unique_id=ids[id(element)],
                          child_ids=[ids[id(child)] for child in element if id(child) in ids],
                          bounds_in_screen=SimpleNamespace(left=x1, right=x2, top=y1, bottom=y2),
                          is_visible_to_user=element.get("visible-to-user", "true") == "true",
                          is_editable="EditText" in element.get("class", ""))
        for source in ("checked", "checkable", "clickable", "enabled", "focused", "focusable",
                       "long-clickable", "scrollable", "selected"):
            attributes["is_" + source.replace("-", "_")] = element.get(source, "false") == "true"
        nodes.append(AccessibilityNode(**attributes))
    return SimpleNamespace(windows=[SimpleNamespace(id=0, tree=SimpleNamespace(nodes=nodes))])


def native_action(action):
    """Only adapt the action schema; finish is never interpreted as success."""
    value = action.as_dict() if hasattr(action, "as_dict") else dict(action or {})
    kind = value.get("action_type")
    if kind in {"answer", "status"}:
        return {"operation": "finish", "action": "finish", "kwargs": {"message": value.get("text", "")}}
    names = {"click": "Tap", "double_tap": "Tap", "long_press": "Long Press", "input_text": "Type", "clear_text": "Type",
             "navigate_home": "Home", "navigate_back": "Back", "keyboard_enter": "Enter",
             "open_app": "Launch", "scroll": "Swipe", "swipe": "Swipe", "wait": "Wait"}
    return {"operation": "do", "action": names.get(kind, "Invalid"), "kwargs": value}


class AndroidLabEnvironment:
    """Duck-typed AsyncEnv; AndroidLab alone owns initial state and task truth."""

    def __init__(self, device, task, options, work_dir, metric, max_steps, *, retain=False):
        self.device, self.task, self.options = device, task, options
        self.work_dir, self.metric, self.max_steps = Path(work_dir), metric, max_steps
        self.retain = retain
        self.current = None
        self.state = None
        self.snapshot_index = 0
        self.actions = 0
        self.error = None
        self.final_result = {"complete": False}
        self.judge_calls = 0
        self.judge_time_ms = 0.0
        self.last_action = {"operation": "do", "action": "Wait", "kwargs": {}}
        self._committed = False
        self.use_accessibility = False
        if any(name in task["app"].lower() for name in ("pimusic", "map.me")):
            # Follow official selection: accessibility service when alive, UIAutomator otherwise.
            try:
                device.xml(True)
                self.use_accessibility = True
            except Exception:
                self.use_accessibility = False

    @property
    def goal(self):
        return f"You should use {self.task['app']} to complete the following task: {self.task['instruction']}"

    def begin_step(self):
        self._committed = False

    def _guard(self, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            self.error = InfrastructureError(str(exc))
            raise self.error from exc

    def get_state(self, wait_to_stabilize=False):
        if wait_to_stabilize:
            time.sleep(self.options.wait_after_action_seconds)
        return self._guard(self._capture)

    def _capture(self):
        import numpy as np
        from PIL import Image
        from android_world.env import interface, representation_utils
        xml = self.device.xml(self.use_accessibility)
        forest = xml_forest(xml)
        pixels = np.asarray(Image.open(io.BytesIO(self.device.adb("exec-out", "screencap", "-p", binary=True))).convert("RGB"))
        height, width = pixels.shape[:2]
        rotation = int(ET.fromstring(xml).get("rotation", "0"))
        # A silently rotated bounding-box transform can invalidate an entire run.
        if rotation != 0:
            raise InfrastructureError("This AndroidLab bridge currently requires the official portrait snapshot")
        elements = representation_utils.forest_to_ui_elements(forest, screen_size=(width, height))
        self.state = interface.State(pixels=pixels, forest=forest, ui_elements=elements)
        self.current = {"xml_text": xml, "image_bytes": pixels, "index": self.snapshot_index,
                        "window": [width, height], "current_activity": self.task["package"],
                        "command": ({self.task["adb_query"]: self.device.query(self.task["adb_query"])}
                                    if self.task.get("adb_query") else {})}
        self.snapshot_index += 1
        return self.state

    @property
    def logical_screen_size(self):
        if self.state is None:
            self.get_state()
        return self.state.pixels.shape[1], self.state.pixels.shape[0]

    @property
    def physical_frame_boundary(self):
        width, height = self.logical_screen_size
        return 0, 0, width, height

    @property
    def orientation(self):
        return 0

    @property
    def controller(self):
        return self

    def reset(self, go_home=False):
        if go_home:
            raise InfrastructureError("AndroidLab tasks must start in_app, not on the AndroidWorld home screen")
        return self.get_state()

    def hide_automation_ui(self):
        pass

    def execute_action(self, action):
        if self.actions >= self.max_steps:
            raise StepBudgetReached()
        try:
            self._execute(action)
        except InfrastructureError as exc:
            self.error = exc
            raise
        self.actions += 1
        self._committed = True
        self.last_action = native_action(action)
        self.evaluate(self.last_action)

    def _execute(self, action):
        value = action.as_dict() if hasattr(action, "as_dict") else dict(action)
        kind, index = value.get("action_type"), value.get("index")
        center = None
        if index is not None:
            if not 0 <= int(index) < len(self.state.ui_elements):
                raise ValueError("UI element index out of bounds")
            center = self.state.ui_elements[int(index)].bbox_pixels.center
        elif value.get("x") is not None and value.get("y") is not None:
            center = (value["x"], value["y"])
        width, height = self.logical_screen_size
        def adb(*args):
            return self.device.adb(*map(str, args))
        if kind in {"click", "double_tap", "long_press", "input_text", "clear_text"} and center:
            x, y = map(int, center)
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("Click coordinates outside screen")
            if kind == "long_press":
                adb("shell", "input", "swipe", x, y, x, y, 1000)
            else:
                adb("shell", "input", "tap", x, y)
                if kind == "double_tap":
                    adb("shell", "input", "tap", x, y)
        elif kind in {"click", "double_tap", "long_press"}:
            raise ValueError("Click requires an index or pixel coordinates")
        if kind in {"input_text", "clear_text"}:
            text = value.get("text") if kind == "input_text" else ""
            if text is None:
                raise ValueError("input_text requires text")
            if "com.android.adbkeyboard" not in adb("shell", "settings", "get", "secure", "default_input_method"):
                raise InfrastructureError("Official AndroidLab snapshot has no active ADBKeyboard; input would be discarded")
            # Same ADBKeyboard transport as AndroidLab, no shell interpolation of model text.
            if kind == "clear_text" or value.get("clear_text", True):
                adb("shell", "input", "keyevent", "KEYCODE_MOVE_END")
                adb("shell", "input", "keyevent", *(["67"] * 100))
            if text:
                encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
                output = adb("shell", "am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", encoded)
                if "Broadcast completed" not in output:
                    raise InfrastructureError("AndroidLab ADBKeyboard input failed")
        elif kind in {"navigate_home", "navigate_back", "keyboard_enter"}:
            adb("shell", "input", "keyevent", {"navigate_home": "3", "navigate_back": "4", "keyboard_enter": "66"}[kind])
        elif kind in {"scroll", "swipe"}:
            direction = value.get("direction")
            x, y = center or (width / 2, height / 2)
            dx, dy = {"down": (0, -height * .35), "up": (0, height * .35),
                      "right": (-width * .35, 0), "left": (width * .35, 0)}[direction]
            if kind == "swipe":
                dx, dy = -dx, -dy
            clamp = lambda v, maximum: int(max(1, min(maximum - 2, v)))
            adb("shell", "input", "swipe", clamp(x - dx / 2, width), clamp(y - dy / 2, height),
                clamp(x + dx / 2, width), clamp(y + dy / 2, height), 400)
        elif kind == "open_app":
            packages = self.app_packages
            name = value.get("app_name", "").replace(" ", "").casefold()
            package = packages.get(name)
            if package is None:
                raise ValueError(f"Unknown AndroidLab app: {value.get('app_name')}; use {sorted(packages)}")
            adb("shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1")
        elif kind == "wait":
            time.sleep(1)
        elif kind not in {"click", "double_tap", "long_press", "input_text", "answer", "status"}:
            raise ValueError(f"Unsupported AndroidLab action: {kind}")

    def evaluate(self, parsed):
        from utils_mobile.xml_tool import UIXMLTree
        if self.current is None:
            self.get_state()
        current = self.current
        line = {"trace_id": self.task["task_id"], "index": current["index"], "target": self.goal,
                "parsed_action": parsed, "window": current["window"],
                "current_activity": current["current_activity"], "xml": "in-memory.xml", "ac_xml": None,
                "image": "in-memory.png", "prompt": "** XML **"}
        if self.task.get("adb_query"):
            line["command"] = current["command"]
        started = time.perf_counter()
        try:
            compressed = json.loads(UIXMLTree().process(current["xml_text"], level=1, str_type="json"))
            result = self.metric.judge(compressed, line)
            judge_error = getattr(self.metric, "_ports_answer_judge_error", None)
            if isinstance(judge_error, JudgeError):
                raise judge_error
            if not isinstance(result, dict):
                raise TypeError("Official predicate did not return a dict")
            if result.get("judge_page", True):
                if not isinstance(result.get("complete"), bool):
                    raise TypeError("Official predicate did not return Boolean complete")
                self.final_result = result
            self.judge_calls += 1
        except Exception as exc:
            self.error = JudgeError(f"AndroidLab official predicate failed: {exc}")
            raise self.error from exc
        finally:
            self.judge_time_ms += (time.perf_counter() - started) * 1000
        if self.retain:
            from PIL import Image
            self.work_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{current['index']:05d}"
            xml_path = self.work_dir / (stem + ".xml")
            image_path = self.work_dir / (stem + ".png")
            xml_path.write_text(current["xml_text"], encoding="utf-8")
            Image.fromarray(current["image_bytes"]).save(image_path)
            line.update(xml=str(xml_path), image=str(image_path))
            with (self.work_dir / "native_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    def finalize_step(self, data):
        if self.error:
            raise self.error
        action = data.get("action_output_json")
        if not self._committed:
            self.last_action = native_action(action) if action is not None else {"operation": "do", "action": "Invalid", "kwargs": {}}
            self.evaluate(self.last_action)

    def finish(self):
        if self.error:
            raise self.error
        self.get_state()
        self.evaluate(self.last_action)
        if not self.judge_calls:
            raise JudgeError("No official observations were judged")
        return self.final_result["complete"]


@contextlib.contextmanager
def bridge_adb_helpers():
    """T3A calls these helpers via controller rather than AsyncEnv properties."""
    from android_world.env import adb_utils
    originals = {}
    for name, attribute in (("get_physical_frame_boundary", "physical_frame_boundary"),
                            ("get_orientation", "orientation"), ("get_logical_screen_size", "logical_screen_size")):
        original = getattr(adb_utils, name)
        originals[name] = original
        def bridged(env, *args, _original=original, _attribute=attribute, **kwargs):
            return getattr(env, _attribute) if isinstance(env, AndroidLabEnvironment) else _original(env, *args, **kwargs)
        setattr(adb_utils, name, bridged)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(adb_utils, name, original)
