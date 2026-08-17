"""TOML 配置读取与路径解析。

所有本地路径都相对于配置文件所在目录解析；模型密钥只写环境变量名，
不把 token 落盘。
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 可安装 tomli 作为兼容。
    import tomli as tomllib  # type: ignore[no-redef]

from .models import ModelProfile


@dataclasses.dataclass(slots=True)
class PathConfig:
    repo_root: Path
    android_world_root: Path
    skvm_skills_root: Path
    ms_swift_root: Path
    state_dir: Path

    @property
    def database(self) -> Path:
        return self.state_dir / "skill_library.sqlite3"


@dataclasses.dataclass(slots=True)
class OfflineConfig:
    teacher_model_id: str
    student_model_path: str
    trajectory_dir: Path
    dataset_dir: Path
    output_dir: Path
    successful_only: bool = True
    validation_ratio: float = 0.05
    max_pixels: int = 1003520
    freeze_vit: bool = False
    epochs: float = 2.0
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32


@dataclasses.dataclass(slots=True)
class RoutingConfig:
    success_weight: float = 1.0
    latency_weight: float = 0.0001
    switch_weight: float = 0.003
    polished_bonus: float = 0.05
    degradation_weight: float = 0.02
    minimum_capability: float = 0.05
    maximum_candidates_per_position: int = 32


@dataclasses.dataclass(slots=True)
class MaintenanceConfig:
    minimum_support: int = 5
    minimum_candidate_trials: int = 10
    minimum_subsequence_length: int = 2
    maximum_subsequence_length: int = 5
    promotion_success_rate: float = 0.70
    rollback_success_rate: float = 0.45
    baseline_margin: float = 0.02
    # 配置后 backend 会逐批让云模型把 SKVM raw skill 编译成 Android 原语拓扑。
    raw_skill_compiler_model_id: str | None = None
    raw_skill_compile_batch_size: int = 8


@dataclasses.dataclass(slots=True)
class AndroidWorldConfig:
    adb_path: str = "adb"
    console_port: int = 5554
    emulator_setup: bool = False
    task_registry: str = "android_world.task_evals.registry.TaskRegistry"
    max_steps: int = 0
    stop_on_task_success: bool = True
    wait_after_action_seconds: float = 1.0


@dataclasses.dataclass(slots=True)
class ProjectConfig:
    """整套离线/在线系统配置。"""

    config_path: Path
    paths: PathConfig
    offline: OfflineConfig
    routing: RoutingConfig
    maintenance: MaintenanceConfig
    android_world: AndroidWorldConfig
    models: tuple[ModelProfile, ...]

    def model(self, model_id: str) -> ModelProfile:
        for profile in self.models:
            if profile.model_id == model_id:
                return profile
        raise KeyError(f"配置中不存在模型: {model_id}")

    def ensure_runtime_dirs(self) -> None:
        """仅创建 src1 的运行目录，不触碰旧 src。"""

        for path in (
            self.paths.state_dir,
            self.offline.trajectory_dir,
            self.offline.dataset_dir,
            self.offline.output_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_path(base: Path, raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置段 [{key}] 必须是表")
    return dict(value)


def load_config(path: str | os.PathLike[str]) -> ProjectConfig:
    """读取 TOML，并把所有相对路径转换成绝对路径。"""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    base = config_path.parent

    path_raw = _section(raw, "paths")
    paths = PathConfig(
        repo_root=_resolve_path(base, path_raw.get("repo_root", "..")),
        android_world_root=_resolve_path(
            base, path_raw.get("android_world_root", "../libs/android_world")
        ),
        skvm_skills_root=_resolve_path(
            base, path_raw.get("skvm_skills_root", "../libs/skvm/skvm-data/skills")
        ),
        ms_swift_root=_resolve_path(
            base, path_raw.get("ms_swift_root", "../libs/ms-swift")
        ),
        state_dir=_resolve_path(base, path_raw.get("state_dir", "./runtime")),
    )

    offline_raw = _section(raw, "offline")
    offline = OfflineConfig(
        teacher_model_id=str(offline_raw.get("teacher_model_id", "teacher-vl")),
        student_model_path=str(offline_raw.get("student_model_path", "")),
        trajectory_dir=_resolve_path(
            base, offline_raw.get("trajectory_dir", "./runtime/trajectories")
        ),
        dataset_dir=_resolve_path(
            base, offline_raw.get("dataset_dir", "./runtime/dataset")
        ),
        output_dir=_resolve_path(
            base, offline_raw.get("output_dir", "./runtime/checkpoints")
        ),
        successful_only=bool(offline_raw.get("successful_only", True)),
        validation_ratio=float(offline_raw.get("validation_ratio", 0.05)),
        max_pixels=int(offline_raw.get("max_pixels", 1003520)),
        freeze_vit=bool(offline_raw.get("freeze_vit", False)),
        epochs=float(offline_raw.get("epochs", 2.0)),
        learning_rate=float(offline_raw.get("learning_rate", 1e-4)),
        lora_rank=int(offline_raw.get("lora_rank", 16)),
        lora_alpha=int(offline_raw.get("lora_alpha", 32)),
    )
    routing = RoutingConfig(**_section(raw, "routing"))
    maintenance = MaintenanceConfig(**_section(raw, "maintenance"))
    android_world = AndroidWorldConfig(**_section(raw, "android_world"))

    model_items = raw.get("models", [])
    if not isinstance(model_items, list):
        raise ValueError("[[models]] 必须是表数组")
    models = tuple(ModelProfile.from_dict(item) for item in model_items)
    if not models:
        raise ValueError("至少需要配置一个 [[models]]")

    return ProjectConfig(
        config_path=config_path,
        paths=paths,
        offline=offline,
        routing=routing,
        maintenance=maintenance,
        android_world=android_world,
        models=models,
    )
