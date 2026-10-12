"""Standalone TOML configuration for the V-Droid baseline."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib  # type: ignore[no-redef]


@dataclasses.dataclass(slots=True)
class VDroidPathConfig:
    repo_root: Path
    v_droid_root: Path
    output_dir: Path
    log_dir: Path


@dataclasses.dataclass(slots=True)
class VDroidDeviceConfig:
    adb_path: str = "adb"
    console_port: int = 5554
    grpc_port: int = 8554
    device_name: str = "emulator-5554"
    emulator_setup: bool = False
    freeze_datetime: bool = True
    verify_api_level: bool = True
    wait_after_action_seconds: float = 1.0


@dataclasses.dataclass(slots=True)
class VDroidModelConfig:
    """The fixed verifier used for every task and every action decision."""

    model_id: str = "vdroid-8b-0323"
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    # Upstream V-Droid prepends ``V-Droid/`` and therefore expects the short
    # Hugging Face repository name by default.
    verifier_adapter: str = "V-Droid-8B-0323"
    num_actors: int = 1
    search_iterations: int = 1
    summary_mode: str = "llm"
    cuda_visible_devices: str | None = None


@dataclasses.dataclass(slots=True)
class VDroidAssistantConfig:
    """Fixed LLM used only for action completion and working memory."""

    service: str = "openai"
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_version: str | None = None


@dataclasses.dataclass(slots=True)
class VDroidEvaluationConfig:
    family: str = "android_world"
    tasks: tuple[str, ...] = ()
    task_count: int | None = None
    combinations: int = 1
    seed: int = 42
    max_steps: int = 30
    fixed_task_seed: bool = False
    store_screenshots: bool = False
    resume: bool = False
    store_trajectories: bool = True


@dataclasses.dataclass(slots=True)
class VDroidBaselineConfig:
    config_path: Path
    paths: VDroidPathConfig
    device: VDroidDeviceConfig
    model: VDroidModelConfig
    assistant: VDroidAssistantConfig
    evaluation: VDroidEvaluationConfig

    def ensure_output_dirs(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, *, require_source: bool = True) -> None:
        if require_source and not (
            self.paths.v_droid_root / "android_world" / "agents" / "vdroid.py"
        ).is_file():
            raise FileNotFoundError(
                "V-Droid source not found under "
                f"{self.paths.v_droid_root}; expected android_world/agents/vdroid.py"
            )
        if self.device.console_port <= 0:
            raise ValueError("android_world.console_port must be positive")
        if self.device.grpc_port <= 0:
            raise ValueError("android_world.grpc_port must be positive")
        if self.device.wait_after_action_seconds < 0:
            raise ValueError(
                "android_world.wait_after_action_seconds cannot be negative"
            )
        if self.model.num_actors <= 0:
            raise ValueError("vdroid.num_actors must be positive")
        if self.model.search_iterations <= 0:
            raise ValueError("vdroid.search_iterations must be positive")
        if self.model.summary_mode not in {"llm", "rule", "skip"}:
            raise ValueError("vdroid.summary_mode must be llm, rule, or skip")
        if self.assistant.service not in {"openai", "trapi", "azure_openai"}:
            raise ValueError(
                "assistant.service must be openai, trapi, or azure_openai"
            )
        if self.evaluation.task_count is not None and self.evaluation.task_count <= 0:
            raise ValueError("evaluation.task_count must be positive")
        if self.evaluation.combinations <= 0:
            raise ValueError("evaluation.combinations must be positive")
        if self.evaluation.max_steps <= 0:
            raise ValueError("evaluation.max_steps must be positive")
        if not isinstance(self.evaluation.resume, bool):
            raise ValueError("evaluation.resume must be a boolean")
        if not isinstance(self.evaluation.store_trajectories, bool):
            raise ValueError("evaluation.store_trajectories must be a boolean")


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"[{name}] must be a TOML table")
    return dict(value)


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_vdroid_config(path: str | os.PathLike[str]) -> VDroidBaselineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    base = config_path.parent

    path_raw = _section(raw, "paths")
    paths = VDroidPathConfig(
        repo_root=_resolve_path(base, path_raw.get("repo_root", "../..")),
        v_droid_root=_resolve_path(
            base, path_raw.get("v_droid_root", "../../libs/V-Droid")
        ),
        output_dir=_resolve_path(
            base, path_raw.get("output_dir", "../runtime/vdroid_evaluations")
        ),
        log_dir=_resolve_path(base, path_raw.get("log_dir", "../runtime/logs")),
    )

    device = VDroidDeviceConfig(**_section(raw, "android_world"))
    model = VDroidModelConfig(**_section(raw, "vdroid"))
    assistant = VDroidAssistantConfig(**_section(raw, "assistant"))
    evaluation_raw = _section(raw, "evaluation")
    evaluation = VDroidEvaluationConfig(
        family=str(evaluation_raw.get("family", "android_world")),
        tasks=tuple(str(item) for item in evaluation_raw.get("tasks", ())),
        task_count=(
            int(evaluation_raw["task_count"])
            if evaluation_raw.get("task_count") is not None
            else None
        ),
        combinations=int(evaluation_raw.get("combinations", 1)),
        seed=int(evaluation_raw.get("seed", 42)),
        max_steps=int(evaluation_raw.get("max_steps", 30)),
        fixed_task_seed=bool(evaluation_raw.get("fixed_task_seed", False)),
        store_screenshots=bool(evaluation_raw.get("store_screenshots", False)),
        resume=evaluation_raw.get("resume", False),
        store_trajectories=evaluation_raw.get("store_trajectories", True),
    )
    config = VDroidBaselineConfig(
        config_path=config_path,
        paths=paths,
        device=device,
        model=model,
        assistant=assistant,
        evaluation=evaluation,
    )
    config.validate()
    return config

