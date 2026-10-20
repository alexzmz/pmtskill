"""Separate AndroidLab configuration: never reuse AndroidWorld AVD/gRPC state."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from ..pmtskill_v2.core.config import tomllib


@dataclasses.dataclass(frozen=True)
class AndroidLabConfig:
    root: Path
    backend: str = "avd"
    sdk_root: Path | None = None
    avd_home: Path | None = None
    avd_work_dir: Path | None = None
    console_port: int = 5562
    avd_snapshot: str | None = None
    avd_gpu_mode: str = "off"
    avd_headless: bool = True
    clone_timeout_seconds: float = 600.0
    docker_image: str = "android_eval:latest"
    avd_name: str = "Pixel_7_Pro_API_33"
    docker_port: int = 6060
    startup_timeout_seconds: float = 240.0
    command_timeout_seconds: float = 45.0
    episode_timeout_seconds: float = 1800.0
    wait_after_action_seconds: float = 1.0
    app_start_wait_seconds: float = 15.0
    judge_model_id: str | None = None
    capability_map: Path | None = None

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        with path.open("rb") as handle:
            raw = dict(tomllib.load(handle).get("androidlab", {}))
        def resolve(value):
            item = Path(value).expanduser()
            return (path.parent / item).resolve() if not item.is_absolute() else item.resolve()
        raw["root"] = resolve(raw.get("root", "../libs/Android-Lab"))
        raw.setdefault("avd_work_dir", "./runtime/androidlab_avd_work")
        for key in ("capability_map", "sdk_root", "avd_home", "avd_work_dir"):
            if raw.get(key):
                raw[key] = resolve(raw[key])
        result = cls(**raw)
        if result.backend not in {"avd", "docker"}:
            raise ValueError("androidlab.backend must be avd or docker")
        if (isinstance(result.console_port, bool) or not isinstance(result.console_port, int)
                or not 1024 <= result.console_port <= 65534 or result.console_port % 2):
            raise ValueError("androidlab.console_port must be an even integer in [1024,65534]")
        if not isinstance(result.avd_headless, bool):
            raise ValueError("androidlab.avd_headless must be Boolean")
        import re
        if not isinstance(result.avd_name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", result.avd_name) or result.avd_name in {".", ".."}:
            raise ValueError("Invalid AndroidLab AVD name")
        if result.avd_snapshot is not None and (not isinstance(result.avd_snapshot, str)
                                               or not re.fullmatch(r"[A-Za-z0-9_-]+", result.avd_snapshot)):
            raise ValueError("avd_snapshot must be an explicit simple snapshot name, or omitted for cold clone boot")
        if result.avd_gpu_mode not in {"off", "auto", "host", "software", "swiftshader_indirect", "swiftshader"}:
            raise ValueError("Unsupported avd_gpu_mode")
        if isinstance(result.docker_port, bool) or not isinstance(result.docker_port, int) or not 1 <= result.docker_port <= 65535:
            raise ValueError("androidlab.docker_port must be in [1,65535]")
        for key in ("startup_timeout_seconds", "command_timeout_seconds", "episode_timeout_seconds", "clone_timeout_seconds",
                    "wait_after_action_seconds", "app_start_wait_seconds"):
            value = getattr(result, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or ("timeout" in key and value <= 0):
                raise ValueError(f"androidlab.{key} must be finite and positive (waits may be zero)")
        for value in (result.docker_image, result.avd_name):
            if not isinstance(value, str) or not value or value.startswith("-") or "\n" in value:
                raise ValueError("Invalid AndroidLab image/AVD name")
        return result

    def validate_source(self):
        for relative in ("adb_client.py", "evaluation/config", "evaluation/task.py",
                         "evaluation/definition.py", "utils_mobile/xml_tool.py"):
            if not (self.root / relative).exists():
                raise FileNotFoundError(
                    f"AndroidLab source missing: {self.root / relative}; see src1/ANDROIDLAB_EVALUATION.md")

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in dataclasses.asdict(self).items()}
