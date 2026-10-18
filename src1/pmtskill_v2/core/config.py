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
    """训练前、中、后的 AndroidWorld 评测配置。

    ``enabled`` 控制是否生成完整的模型/技能库评测报告。SR 早退独立配置且默认
    开启，因此普通 ``train`` 也会在固定小样本上做最低限度的 baseline/逐 epoch
    standalone probe；显式关闭早退后才退化为一次不访问 emulator 的纯 SFT。
    """

    enabled: bool = False
    model_id: str | None = None
    family: str = "android_world"
    tasks: tuple[str, ...] = ()
    # None 表示评测 family 中的全部任务；正整数才表示固定抽样。
    task_count: int | None = None
    combinations: int = 1
    seed: int = 42
    max_steps: int = 30
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
    latency_weight: float = 0.0
    switch_weight: float = 0.0
    polished_bonus: float = 0.05
    degradation_weight: float = 0.02
    minimum_capability: float = 0.05
    maximum_candidates_per_position: int = 32
    # 离线估计仅在缺少实测证据时作为先验，不覆盖已观测的模型—技能状态。
    use_skill_estimates: bool = True
    # Success is the primary objective, not a term latency can outweigh.
    objective: str = "success_first"
    runtime_rerouting: bool = True
    runtime_context_weight: float = 0.25
    stall_threshold: int = 3
    stall_model_discount: float = 0.35

    def __post_init__(self):
        import math
        if self.objective not in {"success_first", "weighted"}:
            raise ValueError("routing.objective must be success_first or weighted")
        for name in ("success_weight", "latency_weight", "switch_weight", "polished_bonus", "degradation_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"routing.{name} must be finite and nonnegative")
        if self.objective == "success_first" and self.success_weight <= 0:
            raise ValueError("success_first requires positive success_weight")
        if not 0 <= self.runtime_context_weight <= 1 or not 0 < self.stall_model_discount < 1 or self.stall_threshold < 2:
            raise ValueError("Invalid routing runtime_context_weight/stall_model_discount/stall_threshold")


@dataclasses.dataclass(slots=True)
class EvidenceMiningConfig:
    """分层经验准入；这些阈值不是学生模型的标定门槛。"""

    l1_min_successes: int = 4
    l1_min_tasks: int = 2
    l1_min_success_rate: float = 0.75
    l1_min_wilson_lower: float = 0.40
    l2_min_successes: int = 3
    l2_min_tasks: int = 2
    l2_min_success_rate: float = 0.85
    l2_min_wilson_lower: float = 0.40
    l3_min_successes: int = 2
    l3_min_success_rate: float = 0.85
    l3_min_wilson_lower: float = 0.30
    l3_min_goal_variants: int = 2
    l3_min_shared_fraction: float = 0.60
    subflow_min_actions: int = 2
    subflow_max_actions: int = 4
    max_active_l2: int = 60
    prompt_max_chars: int = 1200


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
    # success_once 是覆盖优先实验库，不代表任何学生模型已经通过配对标定。
    admission_policy: str = "calibrated"
    build_hierarchy: bool = False
    hierarchy_block_size: int = 3
    mining: EvidenceMiningConfig = dataclasses.field(default_factory=EvidenceMiningConfig)


def resolve_android_world_grpc_port(console_port: int, grpc_port: int | None) -> int:
    """Validate independent emulator ports; never guess a second device's gRPC.

    Only the historical 5554/8554 default is implicit. Other console ports must
    name the actual ``emulator -grpc`` port, which Android does not derive from
    ``-port`` (and which is unrelated to the model server's deploy port).
    """
    if (
        isinstance(console_port, bool)
        or not isinstance(console_port, int)
        or not 1 <= console_port <= 65535
    ):
        raise ValueError("android_world.console_port 必须是 [1, 65535] 内的整数")
    if grpc_port is None:
        if console_port != 5554:
            raise ValueError(
                f"android_world.console_port={console_port} 时必须显式配置 "
                "android_world.grpc_port，填写模拟器启动命令的 -grpc 端口；"
                "它不能由 console_port 推断，也不是模型的 deploy_port"
            )
        grpc_port = 8554
    if (
        isinstance(grpc_port, bool)
        or not isinstance(grpc_port, int)
        or not 1 <= grpc_port <= 65535
    ):
        raise ValueError("android_world.grpc_port 必须是 [1, 65535] 内的整数")
    return grpc_port


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
    # Android 运行时权限弹窗由系统 permission controller 承载，它不会造成
    # a11y/ADB 异常，却会遮住被测 App。正数启用专项处理：明确的 Allow 自动
    # 点击并继续当前 episode，不明确的按钮写入日志并交给模型；0 表示关闭。
    permission_controller_recovery_attempts: int = 3
    # 点击权限按钮后留给系统关闭/切换下一层权限弹窗的等待时间。
    permission_controller_settle_seconds: float = 0.5
    # 硬恢复只重启 Android emulator guest，不会重启宿主机。
    recovery_timeout_seconds: float = 180.0
    recovery_poll_seconds: float = 2.0
    # 独立于 console_port；非默认设备必须填写 emulator 的真实 -grpc 端口。
    grpc_port: int | None = None

    def __post_init__(self) -> None:
        self.grpc_port = resolve_android_world_grpc_port(self.console_port, self.grpc_port)


@dataclasses.dataclass(slots=True)
class EvaluationConfig:
    """独立 evaluate 命令的恢复策略；不会隐式改变训练期评测。"""

    resume: bool = False
    store_trajectories: bool = True
    # Opt-in for ALL methods sharing this TOML, not an implicit PORTS advantage.
    fixed_step_budget: bool = False


@dataclasses.dataclass(slots=True)
class CalibrationConfig:
    mode: str = "androidworld"
    offline_scorer: str = "prior"
    max_samples_per_skill: int = 8
    max_training_samples: int = 2000


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
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = dataclasses.field(default_factory=CalibrationConfig)

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
        task_count=(
            int(training_evaluation_raw["task_count"])
            if training_evaluation_raw.get("task_count") is not None
            else None
        ),
        combinations=int(training_evaluation_raw.get("combinations", 1)),
        seed=int(training_evaluation_raw.get("seed", 42)),
        max_steps=int(training_evaluation_raw.get("max_steps", 30)),
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
    if (
        training_evaluation.task_count is not None
        and training_evaluation.task_count <= 0
    ):
        raise ValueError("training_evaluation.task_count 必须是正整数")
    if training_evaluation.combinations <= 0:
        raise ValueError("training_evaluation.combinations 必须是正整数")
    if training_evaluation.max_steps <= 0:
        raise ValueError("training_evaluation.max_steps 必须是正整数")
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
    maintenance_raw = _section(raw, "maintenance")
    mining_raw = maintenance_raw.pop("mining", {})
    if not isinstance(mining_raw, dict):
        raise ValueError("[maintenance.mining] 必须是表")
    mining = EvidenceMiningConfig(**mining_raw)
    maintenance = MaintenanceConfig(**maintenance_raw, mining=mining)
    if maintenance.admission_policy not in {"calibrated", "success_once", "evidence_based"}:
        raise ValueError("maintenance.admission_policy 必须为 calibrated / success_once / evidence_based")
    for name, value in dataclasses.asdict(mining).items():
        if name.endswith(("success_rate", "wilson_lower", "shared_fraction")):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"maintenance.mining.{name} 必须在 [0,1]")
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"maintenance.mining.{name} 必须为正整数")
    if mining.subflow_min_actions < 2 or mining.subflow_max_actions < mining.subflow_min_actions:
        raise ValueError("子流程长度需要 2 <= subflow_min_actions <= subflow_max_actions")
    if mining.prompt_max_chars < 400:
        raise ValueError("maintenance.mining.prompt_max_chars 必须 >= 400")
    if maintenance.hierarchy_block_size < 2:
        raise ValueError("maintenance.hierarchy_block_size 必须 >= 2")
    if maintenance.minimum_support <= 0:
        raise ValueError("maintenance.minimum_support 必须是正整数")
    if maintenance.minimum_candidate_trials <= 0:
        raise ValueError("maintenance.minimum_candidate_trials 必须是正整数")
    if not 0 <= maintenance.promotion_success_rate <= 1:
        raise ValueError("maintenance.promotion_success_rate 必须在 [0, 1]")
    if not 0 <= maintenance.rollback_success_rate <= 1:
        raise ValueError("maintenance.rollback_success_rate 必须在 [0, 1]")
    if not 0 <= maintenance.baseline_margin <= 1:
        raise ValueError("maintenance.baseline_margin 必须在 [0, 1]")
    android_world = AndroidWorldConfig(**_section(raw, "android_world"))
    if android_world.max_steps <= 0:
        raise ValueError("android_world.max_steps 必须是正整数")
    if android_world.infrastructure_recovery_attempts < 0:
        raise ValueError(
            "android_world.infrastructure_recovery_attempts 必须是非负整数"
        )
    if android_world.permission_controller_recovery_attempts < 0:
        raise ValueError(
            "android_world.permission_controller_recovery_attempts 必须是非负整数"
        )
    if android_world.permission_controller_settle_seconds < 0:
        raise ValueError(
            "android_world.permission_controller_settle_seconds 不能为负数"
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
    evaluation_raw = _section(raw, "evaluation")
    resume = evaluation_raw.get("resume", False)
    if not isinstance(resume, bool):
        raise ValueError("evaluation.resume 必须是布尔值")
    store_trajectories = evaluation_raw.get("store_trajectories", True)
    if not isinstance(store_trajectories, bool):
        raise ValueError("evaluation.store_trajectories 必须是布尔值")
    if not isinstance(evaluation_raw.get("fixed_step_budget", False), bool):
        raise ValueError("evaluation.fixed_step_budget 必须是布尔值")
    calibration = CalibrationConfig(**_section(raw, "calibration"))
    if calibration.mode not in {"androidworld", "offline"}:
        raise ValueError("calibration.mode 必须是 androidworld 或 offline")
    if calibration.offline_scorer not in {"prior", "teacher-forcing"}:
        raise ValueError("calibration.offline_scorer 必须是 prior 或 teacher-forcing")
    if calibration.max_samples_per_skill <= 0 or calibration.max_training_samples <= 0:
        raise ValueError("calibration 样本上限必须大于 0")
    if not isinstance(routing.use_skill_estimates, bool):
        raise ValueError("routing.use_skill_estimates 必须是布尔值")

    return ProjectConfig(
        config_path=config_path,
        paths=paths,
        offline=offline,
        training_evaluation=training_evaluation,
        routing=routing,
        maintenance=maintenance,
        android_world=android_world,
        models=models,
        evaluation=EvaluationConfig(resume=resume, store_trajectories=store_trajectories,
                                    fixed_step_budget=evaluation_raw.get("fixed_step_budget", False)),
        calibration=calibration,
    )
