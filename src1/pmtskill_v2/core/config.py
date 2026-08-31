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
    log_dir: Path
    # 可独立指向一份固定技能库；None 时保持旧行为，使用 state_dir 下的数据库。
    skill_library_db: Path | None = None

    @property
    def database(self) -> Path:
        return self.skill_library_db or (self.state_dir / "skill_library.sqlite3")


@dataclasses.dataclass(slots=True)
class OfflineConfig:
    teacher_model_id: str
    student_model_path: str
    trajectory_dir: Path
    dataset_dir: Path
    output_dir: Path
    successful_only: bool = False
    validation_ratio: float = 0.05
    max_pixels: int = 1003520
    freeze_vit: bool = False
    epochs: float = 2.0
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    # 物理 GPU 编号，例如 "2" 或 "2,3"；子进程内部会重新映射为 cuda:0,1。
    cuda_visible_devices: str | None = None


@dataclasses.dataclass(slots=True)
class TrainingEvaluationConfig:
    """训练前、中、后的 AndroidWorld 小样本评测配置。

    ``enabled`` 控制是否生成完整的模型/技能库评测报告。SR 早退独立配置且默认
    开启，因此普通 ``train`` 也会在固定小样本上做最低限度的 baseline/逐 epoch
    standalone probe；显式关闭早退后才退化为一次不访问 emulator 的纯 SFT。
    """

    enabled: bool = False
    model_id: str | None = None
    family: str = "android_world"
    tasks: tuple[str, ...] = ()
    task_count: int = 30
    combinations: int = 1
    seed: int = 42
    every_epochs: int = 1
    # 连续 patience 个完整 epoch 的 standalone Micro SR 没有比历史最佳值高出
    # min_delta（0.01 = 1 个百分点）时停止继续训练。
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.01
    # 0 仅保留最终 checkpoint；N>0 每 N 个 epoch 永久保留一次。
    checkpoint_every_epochs: int = 1
    include_candidate_skills: bool = False
    deploy_host: str = "127.0.0.1"
    deploy_port: int = 8002
    infer_backend: str = "vllm"
    # 单独约束 vLLM 的 KV cache 上下文，避免读取模型原始 262K 配置后 OOM。
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    # 与训练进程独立的物理 GPU 编号，例如 "1"。
    cuda_visible_devices: str | None = None
    startup_timeout_seconds: float = 600.0
    startup_poll_seconds: float = 2.0
    max_new_tokens: int = 2048
    deploy_extra_args: tuple[str, ...] = ()


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
    max_steps: int = 50
    stop_on_task_success: bool = True
    wait_after_action_seconds: float = 1.0
    # a11y/ADB 等基础设施故障后的恢复与当前 task 重试次数；0 表示关闭。
    infrastructure_recovery_attempts: int = 1
    # 硬恢复只重启 Android emulator guest，不会重启宿主机。
    recovery_timeout_seconds: float = 180.0
    recovery_poll_seconds: float = 2.0


@dataclasses.dataclass(slots=True)
class ProjectConfig:
    """整套离线/在线系统配置。"""

    config_path: Path
    paths: PathConfig
    offline: OfflineConfig
    training_evaluation: TrainingEvaluationConfig
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
            self.paths.log_dir,
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
    state_dir = _resolve_path(base, path_raw.get("state_dir", "./runtime"))
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
        state_dir=state_dir,
        log_dir=_resolve_path(base, path_raw.get("log_dir", state_dir / "logs")),
        skill_library_db=_resolve_path(
            base,
            path_raw.get("skill_library_db", state_dir / "skill_library.sqlite3"),
        ),
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
        successful_only=bool(offline_raw.get("successful_only", False)),
        validation_ratio=float(offline_raw.get("validation_ratio", 0.05)),
        max_pixels=int(offline_raw.get("max_pixels", 1003520)),
        freeze_vit=bool(offline_raw.get("freeze_vit", False)),
        epochs=float(offline_raw.get("epochs", 2.0)),
        learning_rate=float(offline_raw.get("learning_rate", 1e-4)),
        lora_rank=int(offline_raw.get("lora_rank", 16)),
        lora_alpha=int(offline_raw.get("lora_alpha", 32)),
        cuda_visible_devices=(
            str(offline_raw["cuda_visible_devices"])
            if offline_raw.get("cuda_visible_devices") is not None
            else None
        ),
    )
    training_evaluation_raw = _section(raw, "training_evaluation")
    training_evaluation = TrainingEvaluationConfig(
        enabled=bool(training_evaluation_raw.get("enabled", False)),
        model_id=(
            str(training_evaluation_raw["model_id"])
            if training_evaluation_raw.get("model_id")
            else None
        ),
        family=str(training_evaluation_raw.get("family", "android_world")),
        tasks=tuple(str(item) for item in training_evaluation_raw.get("tasks", ())),
        task_count=int(training_evaluation_raw.get("task_count", 30)),
        combinations=int(training_evaluation_raw.get("combinations", 1)),
        seed=int(training_evaluation_raw.get("seed", 42)),
        every_epochs=int(training_evaluation_raw.get("every_epochs", 1)),
        early_stopping_enabled=bool(
            training_evaluation_raw.get("early_stopping_enabled", True)
        ),
        early_stopping_patience=int(
            training_evaluation_raw.get("early_stopping_patience", 3)
        ),
        early_stopping_min_delta=float(
            training_evaluation_raw.get("early_stopping_min_delta", 0.01)
        ),
        checkpoint_every_epochs=int(
            training_evaluation_raw.get("checkpoint_every_epochs", 1)
        ),
        include_candidate_skills=bool(
            training_evaluation_raw.get("include_candidate_skills", False)
        ),
        deploy_host=str(
            training_evaluation_raw.get("deploy_host", "127.0.0.1")
        ),
        deploy_port=int(training_evaluation_raw.get("deploy_port", 8002)),
        infer_backend=str(training_evaluation_raw.get("infer_backend", "vllm")),
        max_model_len=int(training_evaluation_raw.get("max_model_len", 32768)),
        gpu_memory_utilization=float(
            training_evaluation_raw.get("gpu_memory_utilization", 0.90)
        ),
        cuda_visible_devices=(
            str(training_evaluation_raw["cuda_visible_devices"])
            if training_evaluation_raw.get("cuda_visible_devices") is not None
            else None
        ),
        startup_timeout_seconds=float(
            training_evaluation_raw.get("startup_timeout_seconds", 600.0)
        ),
        startup_poll_seconds=float(
            training_evaluation_raw.get("startup_poll_seconds", 2.0)
        ),
        max_new_tokens=int(training_evaluation_raw.get("max_new_tokens", 2048)),
        deploy_extra_args=tuple(
            str(item) for item in training_evaluation_raw.get("deploy_extra_args", ())
        ),
    )
    if training_evaluation.task_count <= 0:
        raise ValueError("training_evaluation.task_count 必须是正整数")
    if training_evaluation.combinations <= 0:
        raise ValueError("training_evaluation.combinations 必须是正整数")
    if training_evaluation.every_epochs <= 0:
        raise ValueError("training_evaluation.every_epochs 必须是正整数")
    if training_evaluation.early_stopping_patience <= 0:
        raise ValueError(
            "training_evaluation.early_stopping_patience 必须是正整数"
        )
    if not 0 <= training_evaluation.early_stopping_min_delta <= 1:
        raise ValueError(
            "training_evaluation.early_stopping_min_delta 必须在 [0, 1]"
        )
    if training_evaluation.checkpoint_every_epochs < 0:
        raise ValueError(
            "training_evaluation.checkpoint_every_epochs 必须是非负整数"
        )
    if not 1 <= training_evaluation.deploy_port <= 65535:
        raise ValueError("training_evaluation.deploy_port 必须在 [1, 65535]")
    if training_evaluation.max_model_len <= 0:
        raise ValueError("training_evaluation.max_model_len 必须是正整数")
    if not 0 < training_evaluation.gpu_memory_utilization <= 1:
        raise ValueError(
            "training_evaluation.gpu_memory_utilization 必须在 (0, 1]"
        )
    if training_evaluation.startup_timeout_seconds <= 0:
        raise ValueError("training_evaluation.startup_timeout_seconds 必须为正数")
    if training_evaluation.startup_poll_seconds <= 0:
        raise ValueError("training_evaluation.startup_poll_seconds 必须为正数")
    if training_evaluation.max_new_tokens <= 0:
        raise ValueError("training_evaluation.max_new_tokens 必须是正整数")
    routing = RoutingConfig(**_section(raw, "routing"))
    maintenance = MaintenanceConfig(**_section(raw, "maintenance"))
    android_world = AndroidWorldConfig(**_section(raw, "android_world"))
    if android_world.max_steps <= 0:
        raise ValueError("android_world.max_steps 必须是正整数")
    if android_world.infrastructure_recovery_attempts < 0:
        raise ValueError(
            "android_world.infrastructure_recovery_attempts 必须是非负整数"
        )
    if android_world.recovery_timeout_seconds <= 0:
        raise ValueError("android_world.recovery_timeout_seconds 必须为正数")
    if android_world.recovery_poll_seconds <= 0:
        raise ValueError("android_world.recovery_poll_seconds 必须为正数")

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
        training_evaluation=training_evaluation,
        routing=routing,
        maintenance=maintenance,
        android_world=android_world,
        models=models,
    )
