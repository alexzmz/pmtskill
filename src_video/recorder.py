"""Collect real emulator frames and timestamped agent observations."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
import time
from pathlib import Path


class EpisodeRecorder:
    def __init__(self, output: Path, *, adb: str, serial: str, capture_fps: float,
                 command_timeout: float):
        self.output = output
        self.adb = adb
        self.serial = serial
        self.interval = 1 / capture_fps
        self.command_timeout = command_timeout
        self.episodes: list[Path] = []
        self.errors: list[str] = []
        self.active = None

    def _adb(self, *args):
        result = subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True,
                                timeout=self.command_timeout)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise RuntimeError(f"ADB {self.serial} {args[0]} failed ({result.returncode}): {detail}")
        return result.stdout

    def preflight(self):
        if self._adb("get-state").strip() != b"device":
            raise RuntimeError(f"{self.serial} is not ready")
        self._screenshot()

    def _screenshot(self):
        from PIL import Image

        data = self._adb("exec-out", "screencap", "-p")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise RuntimeError("ADB did not return a PNG screenshot")
            image.verify()
        return data

    def begin(self, task_id):
        if self.active is not None:
            raise RuntimeError("AndroidWorld recording sessions must not overlap")
        directory = self.output / f"episode_{len(self.episodes) + 1:03d}"
        (directory / "frames").mkdir(parents=True)
        (directory / "observations").mkdir()
        state = {
            "directory": directory, "task_id": str(task_id), "start": time.monotonic(),
            "stop": threading.Event(), "frames": [], "steps": [], "controls": [],
            "sample_failures": 0, "last_sample_error": None, "last_capture": None,
            "observation_errors": [], "finished_at": None,
        }
        self.active = state
        self.episodes.append(directory)
        self._capture(state)
        state["thread"] = threading.Thread(target=self._sample, args=(state,), daemon=True,
                                           name="androidworld-video-capture")
        state["thread"].start()

    def _capture(self, state, *, final=False):
        timestamp = time.monotonic() - state["start"]
        try:
            data = self._screenshot()
            digest = hashlib.sha256(data).digest()
            previous = state["last_capture"]
            if previous is not None and previous[0] == digest:
                relative = previous[1]
            else:
                relative = f"frames/{len(state['frames']):07d}.png"
                (state["directory"] / relative).write_bytes(data)
                state["last_capture"] = (digest, relative)
            state["frames"].append({"t": timestamp, "path": relative, "source": "adb",
                                    "final_capture": final})
        except Exception as exc:
            state["sample_failures"] += 1
            state["last_sample_error"] = f"{type(exc).__name__}: {exc}"

    def _sample(self, state):
        delay = self.interval
        while not state["stop"].wait(delay):
            started = time.monotonic()
            self._capture(state)
            delay = max(0, self.interval - (time.monotonic() - started))

    def observe(self, step, pixels, boxes):
        self.observe_at(step, pixels, boxes, time.monotonic())

    def observe_at(self, step, pixels, boxes, captured_at):
        """Use the exact model observation; labels are AndroidWorld SoM indices."""
        from PIL import Image

        state = self.active
        if state is None or state["stop"].is_set():
            return
        timestamp = max(0.0, captured_at - state["start"])
        try:
            image = Image.fromarray(pixels).convert("RGB")
            width, height = image.size
            relative = f"observations/{len(state['controls']):07d}.png"
            image.save(state["directory"] / relative, compress_level=1)
            absolute_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box["bounds"]
                absolute_boxes.append({
                    "bounds": [x1 * width, y1 * height, x2 * width, y2 * height],
                    "label": str(box["label"]),
                })
            state["controls"].append({"t": timestamp, "path": relative,
                                      "size": [width, height], "boxes": absolute_boxes})
            if step is not None:
                event = {"t": timestamp, "path": relative, "step": int(step)}
                state["steps"].append(event)
        except Exception as exc:
            state["observation_errors"].append(f"{type(exc).__name__}: {exc}")

    def stop_capture(self):
        state = self.active
        if state is None or state["stop"].is_set():
            return
        state["finished_at"] = time.monotonic() - state["start"]
        state["stop"].set()
        if "thread" in state:
            state["thread"].join()
        # The hook calls this before benchmark teardown. Capture the terminal UI
        # even when the periodic sampler's latest frame precedes the last action.
        self._capture(state, final=True)

    def end(self, success, error=None):
        state = self.active
        if state is None:
            return
        self.stop_capture()
        self.active = None
        adb_frames = state["frames"]
        # Exact raw trajectory screenshots are interleaved on the continuous
        # timeline, with source provenance, rather than labeling a lookalike frame.
        frames = sorted([*adb_frames, *(dict(s, source="trajectory") for s in state["steps"])],
                        key=lambda row: row["t"])
        duration = max(state["finished_at"] or 0, frames[-1]["t"] if frames else 0)
        metadata = {
            "task_id": state["task_id"], "success": success, "error": str(error) if error else None,
            "duration": duration, "episode_result_time": state["finished_at"],
            "serial": self.serial, "adb_frame_count": len(adb_frames),
            "sample_failures": state["sample_failures"], "last_sample_error": state["last_sample_error"],
            "observation_errors": state["observation_errors"], "frames": frames,
            "steps": state["steps"], "controls": state["controls"],
        }
        write_json(state["directory"] / "capture.json", metadata)
        if not adb_frames:
            self.errors.append(f"{state['directory']}: no continuous ADB screenshots captured")
        elif not adb_frames[-1].get("final_capture"):
            self.errors.append(f"{state['directory']}: terminal screenshot unavailable")
        if success is None:
            self.errors.append(f"{state['directory']}: no benchmark verdict; inspect capture.json and cli_logs")
        if state["observation_errors"]:
            self.errors.append(f"{state['directory']}: could not save some agent observations")
        if not state["controls"] and success is not None:
            self.errors.append(f"{state['directory']}: no agent observations captured")


def write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
