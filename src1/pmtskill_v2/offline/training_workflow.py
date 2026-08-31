"""LoRA 训练与 AndroidWorld SR 评测的可选闭环编排。"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import math
import shutil
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..core.config import ProjectConfig
from ..core.io import write_json_atomic
from ..core.models import ModelProfile
from ..evaluation.deployment import MSSwiftEvaluationDeployment
from ..evaluation.reporter import EvaluationArtifacts
from ..skills.store import SkillStore
from .trainer import (
    AdapterJob,
    MSSwiftLoraTrainer,
    find_latest_adapter_checkpoint,
    load_prepared_training_job,
    prepare_training_job,
    staged_training_job,
    validate_staged_training_args,
)


def build_epoch_targets(total_epochs: float, every_epochs: int) -> list[float]:
    """生成累计训练目标，例如 ``3, 2`` 得到 ``[2, 3]``。

    整数节点是周期评测点；若总 epoch 带小数，最后的部分 epoch 仍会训练并做
    最终评测，但不会伪装成一个完整 epoch。
    """

    if not math.isfinite(total_epochs) or total_epochs <= 0:
        raise ValueError("offline.epochs 必须是正数")
    if every_epochs <= 0:
        raise ValueError("eval_every_epochs 必须是正整数")
    targets: list[float] = []
    current = float(every_epochs)
    while current < total_epochs - 1e-9:
        targets.append(current)
        current += every_epochs
    if not targets or not math.isclose(targets[-1], total_epochs):
        targets.append(float(total_epochs))
    return targets


def epoch_label(value: float) -> str:
    """把 epoch 数转换成稳定、可按字典序排序的目录标签。"""

    if float(value).is_integer():
        return f"epoch_{int(value):03d}"
    whole = int(value)
    fraction = int(round((value - whole) * 1000))
    return f"epoch_{whole:03d}_{fraction:03d}"


@dataclass(frozen=True, slots=True)
class EpochStage:
    """一个需要停下训练的累计 epoch 节点及其用途。"""

    target_epoch: float
    evaluate: bool
    retain_checkpoint: bool
    final: bool


def build_epoch_plan(
    total_epochs: float,
    evaluation_every_epochs: int,
    checkpoint_every_epochs: int,
) -> list[EpochStage]:
    """合并评测与持久 checkpoint 节点；0 表示仅保留最终 checkpoint。"""

    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs 必须是非负整数")
    evaluation_targets = build_epoch_targets(total_epochs, evaluation_every_epochs)
    checkpoint_targets = (
        [float(total_epochs)]
        if checkpoint_every_epochs == 0
        else build_epoch_targets(total_epochs, checkpoint_every_epochs)
    )
    targets = sorted(set(evaluation_targets + checkpoint_targets))

    def contains(values: Sequence[float], target: float) -> bool:
        return any(math.isclose(value, target, abs_tol=1e-9) for value in values)

    return [
        EpochStage(
            target_epoch=target,
            evaluate=contains(evaluation_targets, target),
            retain_checkpoint=contains(checkpoint_targets, target),
            final=math.isclose(target, total_epochs, abs_tol=1e-9),
        )
        for target in targets
    ]


@dataclass(slots=True)
class SuccessRateEarlyStopping:
    """基于固定 AndroidWorld 子集 standalone Micro SR 的早退状态机。

    baseline 作为第一个观测值。只有当前 SR 严格高于历史有效最佳值
    ``min_delta`` 以上才清零计数；因此恰好提升 0.01 在阈值为 0.01 时仍视为
    “没有超过 1 个百分点”。状态完全可序列化，续训时可由 history 中的评测行重建。
    """

    patience: int = 3
    min_delta: float = 0.01
    best_sr: float | None = None
    best_epoch: float | None = None
    stale_epochs: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early_stopping_patience 必须是正整数")
        if not 0 <= self.min_delta <= 1:
            raise ValueError("early_stopping_min_delta 必须在 [0, 1]")

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience

    def observe(
        self,
        *,
        epoch: float,
        micro_sr: float,
        checkpoint: str | None,
    ) -> dict[str, Any]:
        """记录一个可比评测点并返回本次早退判定。"""

        score = float(micro_sr)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"AndroidWorld Micro SR 无效: {micro_sr!r}")
        previous_best = self.best_sr
        improved = previous_best is None or score > (
            previous_best + self.min_delta + 1e-12
        )
        if improved:
            self.best_sr = score
            self.best_epoch = float(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        observation = {
            "epoch": float(epoch),
            "micro_sr": score,
            "checkpoint": checkpoint,
            "previous_best_sr": previous_best,
            "significant_improvement": improved,
            "best_sr": self.best_sr,
            "best_epoch": self.best_epoch,
            "stale_epochs": self.stale_epochs,
            "should_stop": self.should_stop,
        }
        self.observations.append(observation)
        return observation

    def to_dict(
        self, *, enabled: bool, stopped: bool, stop_epoch: float | None
    ) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "metric": "standalone_micro_sr",
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_sr": self.best_sr,
            "best_epoch": self.best_epoch,
            "stale_epochs": self.stale_epochs,
            "stopped": stopped,
            "stop_epoch": stop_epoch,
            "observations": list(self.observations),
        }


def resolve_student_profile(
    config: ProjectConfig, model_id: str | None
) -> ModelProfile:
    """解析训练评测的学生画像；旧配置未声明时也提供安全回退。"""

    if model_id:
        return config.model(model_id)
    student_path = str(Path(config.offline.student_model_path).expanduser())
    for profile in config.models:
        if str(Path(profile.served_model).expanduser()) == student_path:
            return profile
    for profile in config.models:
        if "student" in profile.model_id.lower():
            return profile
    return ModelProfile(
        model_id="student-training-eval",
        served_model=config.offline.student_model_path,
        base_url="http://127.0.0.1:8002/v1",
        capabilities={},
    )


@dataclass(slots=True)
class TrainingEvaluationOptions:
    """一次训练评测实验的全部已解析参数。"""

    output_dir: Path
    tasks: tuple[str, ...]
    family: str = "android_world"
    combinations: int = 1
    seed: int = 42
    max_steps: int = 30
    every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    include_candidate_skills: bool = False
    # False 表示仅运行早退必需的 standalone probes，不评测技能库。
    full_evaluation: bool = True
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.01
    training_cuda_visible_devices: str | None = None
    evaluation_cuda_visible_devices: str | None = None
    evaluation_max_model_len: int | None = None
    evaluation_gpu_memory_utilization: float | None = None
    resume: bool = True


@dataclass(slots=True)
class TrainingEvaluationResult:
    """CLI 可直接序列化的训练评测结果。"""

    return_code: int
    early_stopped: bool
    stop_epoch: float | None
    best_sr: float | None
    resumed: bool
    resumed_from_checkpoint: Path | None
    output_dir: Path
    training_output_dir: Path
    final_checkpoint: Path | None
    history_json: Path
    history_csv: Path
    comparison_markdown: Path
    dataset_manifest: Path
    checkpoints_manifest: Path
    stages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_code": self.return_code,
            "early_stopped": self.early_stopped,
            "stop_epoch": self.stop_epoch,
            "best_sr": self.best_sr,
            "resumed": self.resumed,
            "resumed_from_checkpoint": (
                str(self.resumed_from_checkpoint)
                if self.resumed_from_checkpoint
                else None
            ),
            "output_dir": str(self.output_dir),
            "training_output_dir": str(self.training_output_dir),
            "final_checkpoint": (
                str(self.final_checkpoint) if self.final_checkpoint else None
            ),
            "history_json": str(self.history_json),
            "history_csv": str(self.history_csv),
            "comparison_markdown": str(self.comparison_markdown),
            "dataset_manifest": str(self.dataset_manifest),
            "checkpoints_manifest": str(self.checkpoints_manifest),
            "stages": self.stages,
        }


class EvaluationDeployment(Protocol):
    """模型部署替换点，便于后续接入常驻 vLLM 或远端调度器。"""

    def activate(
        self, checkpoint: Path | None
    ) -> AbstractContextManager[ModelProfile]: ...


class StageEvaluator(Protocol):
    """单阶段 AndroidWorld 评测替换点。"""

    def run(
        self,
        *,
        profile: ModelProfile,
        use_skills: bool,
        tasks: Sequence[str],
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        include_candidate_skills: bool,
        output_dir: Path,
    ) -> EvaluationArtifacts: ...


class AndroidWorldTrainingStageEvaluator:
    """把裸模型 evaluator 与动态技能 evaluator 统一成一个阶段接口。"""

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def run(
        self,
        *,
        profile: ModelProfile,
        use_skills: bool,
        tasks: Sequence[str],
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        include_candidate_skills: bool,
        output_dir: Path,
    ) -> EvaluationArtifacts:
        # 延迟导入，普通 train 和单元测试无需安装 AndroidWorld 的运行依赖。
        from ..evaluation.android_world import (
            AndroidWorldOnlineEvaluator,
            AndroidWorldStandaloneEvaluator,
        )

        if use_skills:
            return AndroidWorldOnlineEvaluator(self.config, self.store).run(
                tasks=tasks,
                n_task_combinations=combinations,
                seed=seed,
                family=family,
                max_steps=max_steps,
                include_candidate_skills=include_candidate_skills,
                output_dir=output_dir,
                model_profiles=(profile,),
                # 训练评测数据不写回技能库，避免测试集反向污染路由统计。
                record_traces=False,
            )
        return AndroidWorldStandaloneEvaluator(self.config).run(
            profile=profile,
            tasks=tasks,
            n_task_combinations=combinations,
            seed=seed,
            family=family,
            max_steps=max_steps,
            output_dir=output_dir,
        )


class TrainingEvaluationRecorder:
    """增量维护 JSON/CSV/Markdown，异常退出时也保留已完成阶段。"""

    def __init__(
        self,
        output_dir: Path,
        manifest: dict[str, Any],
        *,
        resume_state: dict[str, Any] | None = None,
    ):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_json = self.output_dir / "history.json"
        self.history_csv = self.output_dir / "history.csv"
        self.comparison_markdown = self.output_dir / "comparison.md"
        self.manifest_path = self.output_dir / "sample_manifest.json"
        self.command_path = self.output_dir / "training_stage_commands.json"
        self.checkpoints_path = self.output_dir / "checkpoints.json"
        now = dt.datetime.now().astimezone().isoformat()
        if resume_state is None:
            self.state = {
                "schema_version": 4,
                "created_at": now,
                "status": "running",
                "manifest": manifest,
                "stages": [],
                "training_commands": [],
                "checkpoints": [],
                "early_stopping": None,
                "resume_events": [],
                "error": None,
            }
        else:
            self.state = resume_state
            self.state["schema_version"] = max(
                4, int(self.state.get("schema_version", 1))
            )
            self.state["manifest"] = manifest
            self.state.setdefault("stages", [])
            self.state.setdefault("training_commands", [])
            self.state.setdefault("checkpoints", [])
            self.state.setdefault("early_stopping", None)
            self.state.setdefault("resume_events", []).append(
                {
                    "resumed_at": now,
                    "previous_status": self.state.get("status", "unknown"),
                }
            )
            self.state["status"] = "running"
            self.state["error"] = None
            self.state.pop("finished_at", None)
        write_json_atomic(self.manifest_path, manifest)
        self.flush()

    def evaluation_recorded(self, label: str) -> bool:
        return any(row.get("label") == label for row in self.state["stages"])

    def checkpoint_for_epoch(self, epoch: float) -> Path | None:
        for row in reversed(self.state["checkpoints"]):
            try:
                same_epoch = math.isclose(float(row.get("epoch")), epoch)
            except (TypeError, ValueError):
                same_epoch = False
            if not same_epoch:
                continue
            checkpoint = Path(str(row.get("checkpoint", ""))).expanduser()
            if checkpoint.is_dir() and (checkpoint / "adapter_config.json").is_file():
                return checkpoint.resolve()
        return None

    def checkpoint_recorded(self, epoch: float) -> bool:
        for row in self.state["checkpoints"]:
            try:
                if math.isclose(float(row.get("epoch")), epoch):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def record_command(
        self,
        *,
        target_epoch: float,
        stage_output_dir: Path,
        evaluate: bool,
        retain_checkpoint: bool,
        resume: Path | None,
        command: Sequence[str],
    ) -> None:
        self.state["training_commands"].append(
            {
                "stage": epoch_label(target_epoch),
                "target_epoch": target_epoch,
                "stage_output_dir": str(stage_output_dir),
                "evaluate_after_stage": evaluate,
                "retain_checkpoint": retain_checkpoint,
                "resume_from_checkpoint": str(resume) if resume else None,
                "command": list(command),
            }
        )
        self.flush()

    def record_checkpoint(
        self,
        *,
        epoch: float,
        checkpoint: Path,
        stage_output_dir: Path,
        retained: bool,
    ) -> dict[str, Any]:
        row = {
            "stage": epoch_label(epoch),
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "stage_output_dir": str(stage_output_dir),
            "retained": retained,
            "exists": checkpoint.exists(),
            "removed_after_completion": False,
        }
        self.state["checkpoints"].append(row)
        self.flush()
        return row

    def mark_checkpoint_removed(self, row: dict[str, Any]) -> None:
        row["exists"] = False
        row["removed_after_completion"] = True
        self.flush()

    def mark_checkpoint_retained(self, row: dict[str, Any]) -> None:
        """早退点即新的训练终点，必须保留其 adapter checkpoint。"""

        row["retained"] = True
        row["exists"] = True
        self.flush()

    def mark_evaluation_final(self, row: dict[str, Any]) -> None:
        """把计划外早退 epoch 标记为本次运行的最终 checkpoint。"""

        row["is_final_checkpoint"] = True
        self.flush()

    def record_early_stopping(self, state: dict[str, Any]) -> None:
        """持久化完整早退判定，使中断和续训仍可审计。"""

        self.state["early_stopping"] = state
        self.flush()

    def record_evaluation(
        self,
        *,
        label: str,
        mode: str,
        epoch: float,
        checkpoint: Path | None,
        artifacts: EvaluationArtifacts,
        final_checkpoint: bool = False,
    ) -> dict[str, Any]:
        summary = artifacts.summary
        row = {
            "label": label,
            "stage": "baseline" if epoch <= 0 else epoch_label(epoch),
            "mode": mode,
            "epoch": epoch,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "is_final_checkpoint": final_checkpoint,
            "episodes": summary.get("episodes_evaluated", 0),
            "successes": summary.get("successes", 0),
            "micro_sr": summary.get("success_rate_micro", 0.0),
            "macro_sr": summary.get("success_rate_macro", 0.0),
            "average_steps": summary.get("average_steps", 0.0),
            "summary_json": str(artifacts.summary_json),
            "report_markdown": str(artifacts.report_markdown),
            "traces_jsonl": str(artifacts.traces_jsonl),
            "artifact_dir": str(artifacts.output_dir),
        }
        self.state["stages"].append(row)
        self.flush()
        return row

    def finish(self, status: str, *, error: str | None = None) -> None:
        self.state["status"] = status
        self.state["finished_at"] = dt.datetime.now().astimezone().isoformat()
        self.state["error"] = error
        self.flush()

    def _derived(self) -> dict[str, Any]:
        rows = self.state["stages"]
        baseline = next(
            (
                row
                for row in rows
                if row["label"] == "baseline_standalone"
            ),
            None,
        )
        baseline_skills = next(
            (row for row in rows if row["label"] == "baseline_skills"), None
        )
        checkpoint_rows = [
            row for row in rows if row["mode"] == "standalone" and row["epoch"] > 0
        ]
        final_standalone = checkpoint_rows[-1] if checkpoint_rows else None
        final_skills = next(
            (row for row in reversed(rows) if row["label"] == "final_skills"),
            None,
        )
        baseline_sr = float(baseline["micro_sr"]) if baseline else None
        if baseline_sr is not None:
            for row in rows:
                row["gain_over_baseline"] = float(row["micro_sr"]) - baseline_sr
        best = max(checkpoint_rows, key=lambda row: float(row["micro_sr"]), default=None)
        return {
            "baseline_standalone": baseline,
            "baseline_skills": baseline_skills,
            "final_standalone": final_standalone,
            "final_skills": final_skills,
            "best_standalone_checkpoint": best,
        }

    def _write_csv(self) -> None:
        columns = (
            "label",
            "stage",
            "mode",
            "epoch",
            "is_final_checkpoint",
            "episodes",
            "successes",
            "micro_sr",
            "macro_sr",
            "gain_over_baseline",
            "average_steps",
            "checkpoint",
            "summary_json",
            "artifact_dir",
        )
        temporary = self.history_csv.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {key: row.get(key) for key in columns}
                for row in self.state["stages"]
            )
        temporary.replace(self.history_csv)

    def _write_markdown(self, derived: dict[str, Any]) -> None:
        manifest = self.state["manifest"]
        lines = [
            "# LoRA 训练 × AndroidWorld SR 对比",
            "",
            f"- 状态：**{self.state['status']}**",
            f"- 运行模式：**{'完整评测' if manifest.get('full_evaluation') else '仅早退探测'}**",
            f"- 固定任务数：**{len(manifest['tasks'])}**",
            f"- 每任务组合数：**{manifest['combinations']}**",
            f"- 随机种子：**{manifest['seed']}**",
            f"- 每个评测 episode 步数上限：**{manifest.get('evaluation_max_steps', 30)}**",
            "- 所有行使用同一任务列表与 seed，训练评测轨迹不会写回技能库。",
        ]
        early = self.state.get("early_stopping")
        if isinstance(early, dict) and early.get("enabled"):
            lines.extend(
                (
                    f"- 早退：patience={early.get('patience')}，"
                    f"min_delta={float(early.get('min_delta', 0)):.2%}，"
                    f"连续未显著提升={early.get('stale_epochs', 0)}",
                    f"- 是否触发早退：**{'是' if early.get('stopped') else '否'}**"
                    + (
                        f"（epoch {float(early['stop_epoch']):g}）"
                        if early.get("stop_epoch") is not None
                        else ""
                    ),
                )
            )
        lines.extend(
            (
                "",
                "| 阶段 | 模式 | Epoch | 成功/有效 | Micro SR | Macro SR | 相对裸基座 |",
            "|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for row in self.state["stages"]:
            gain = row.get("gain_over_baseline")
            gain_text = "—" if gain is None else f"{float(gain):+.2%}"
            mode = "模型+技能库" if row["mode"] == "skills" else "裸模型"
            lines.append(
                f"| {row['label']} | {mode} | {row['epoch']:g} | "
                f"{row['successes']}/{row['episodes']} | "
                f"{float(row['micro_sr']):.2%} | "
                f"{float(row['macro_sr']):.2%} | {gain_text} |"
            )
        best = derived["best_standalone_checkpoint"]
        if best:
            lines.extend(
                (
                    "",
                    "## 最佳裸模型 checkpoint",
                    "",
                    f"- 阶段：`{best['label']}`",
                    f"- Micro SR：**{float(best['micro_sr']):.2%}**",
                    f"- checkpoint：`{best['checkpoint']}`",
                )
            )
        final_standalone = derived["final_standalone"]
        final_skills = derived["final_skills"]
        if final_standalone and final_skills:
            lift = float(final_skills["micro_sr"]) - float(
                final_standalone["micro_sr"]
            )
            lines.extend(
                (
                    "",
                    "## 最终效果",
                    "",
                    f"- 最终裸模型 SR：**{float(final_standalone['micro_sr']):.2%}**",
                    f"- 最终模型+技能库 SR：**{float(final_skills['micro_sr']):.2%}**",
                    f"- 技能库增益：**{lift:+.2%}**",
                )
            )
        if self.state.get("error"):
            lines.extend(("", "## 失败信息", "", f"```text\n{self.state['error']}\n```"))
        self.comparison_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def flush(self) -> None:
        derived = self._derived()
        serializable = dict(self.state)
        serializable["summary"] = derived
        write_json_atomic(self.history_json, serializable)
        write_json_atomic(
            self.command_path, {"commands": self.state["training_commands"]}
        )
        write_json_atomic(
            self.checkpoints_path,
            {
                "schema_version": 1,
                "checkpoint_every_epochs": self.state["manifest"].get(
                    "checkpoint_every_epochs"
                ),
                "checkpoints": self.state["checkpoints"],
            },
        )
        self._write_csv()
        self._write_markdown(derived)


def _remove_transient_checkpoint(checkpoint: Path) -> None:
    """成功完成全部阶段后，精确移除未请求永久保留的中间 checkpoint。"""

    resolved = checkpoint.resolve()
    if checkpoint.parent.is_dir():
        for child in checkpoint.parent.iterdir():
            if not child.is_symlink():
                continue
            try:
                points_to_checkpoint = child.resolve() == resolved
            except OSError:
                points_to_checkpoint = False
            if points_to_checkpoint:
                child.unlink()
    if checkpoint.is_dir():
        shutil.rmtree(checkpoint)


def _load_resume_state(output_dir: Path) -> dict[str, Any] | None:
    history_path = output_dir / "history.json"
    if not history_path.is_file():
        return None
    try:
        state = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"续训 history.json 不是合法 JSON: {history_path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"续训 history.json 必须是 JSON object: {history_path}")
    return state


def _validate_resume_manifest(
    existing: dict[str, Any], current: dict[str, Any]
) -> None:
    """防止把不同数据/任务规划误接到同一个训练运行。"""

    immutable_fields = (
        "job",
        "family",
        "tasks",
        "combinations",
        "seed",
        "every_epochs",
        "effective_evaluation_every_epochs",
        "checkpoint_every_epochs",
        "include_candidate_skills",
        "full_evaluation",
        "early_stopping_enabled",
        "early_stopping_patience",
        "early_stopping_min_delta",
    )
    mismatched = [
        field
        for field in immutable_fields
        if existing.get(field) != current.get(field)
    ]
    if (
        "evaluation_max_steps" in existing
        and existing.get("evaluation_max_steps")
        != current.get("evaluation_max_steps")
    ):
        mismatched.append("evaluation_max_steps")
    if mismatched:
        raise ValueError(
            "续训配置与原运行不一致: "
            + ", ".join(mismatched)
            + "；请恢复原配置或使用 --no-resume 创建新运行"
        )
    try:
        previous_total = float(existing.get("total_epochs", 0))
        current_total = float(current.get("total_epochs", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("续训 manifest 的 total_epochs 无效") from exc
    if current_total + 1e-9 < previous_total:
        raise ValueError(
            f"续训总 epoch 不能从 {previous_total:g} 降到 {current_total:g}"
        )


def _archive_interrupted_evaluation(output_dir: Path) -> Path | None:
    """保留未写入 history 的残缺评测，然后给重试提供空目录。"""

    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        return None
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = output_dir.with_name(f"{output_dir.name}.interrupted_{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = output_dir.with_name(
            f"{output_dir.name}.interrupted_{stamp}_{suffix}"
        )
        suffix += 1
    output_dir.rename(candidate)
    logging.warning("保留中断评测目录并重新执行该阶段: %s", candidate)
    return candidate


class TrainingEvaluationWorkflow:
    """执行基线、分段训练、逐 checkpoint 与最终技能评测。"""

    def __init__(
        self,
        config: ProjectConfig,
        store: SkillStore,
        trainer: MSSwiftLoraTrainer,
        deployment: EvaluationDeployment | None = None,
        evaluator: StageEvaluator | None = None,
    ):
        self.config = config
        self.store = store
        self.trainer = trainer
        if deployment is None:
            profile = resolve_student_profile(
                config, config.training_evaluation.model_id
            )
            deployment = MSSwiftEvaluationDeployment(
                config, config.training_evaluation, profile
            )
        self.deployment = deployment
        self.evaluator = evaluator or AndroidWorldTrainingStageEvaluator(config, store)

    def _evaluate(
        self,
        *,
        recorder: TrainingEvaluationRecorder,
        profile: ModelProfile,
        options: TrainingEvaluationOptions,
        label: str,
        use_skills: bool,
        epoch: float,
        checkpoint: Path | None,
        final_checkpoint: bool = False,
    ) -> dict[str, Any]:
        mode = "skills" if use_skills else "standalone"
        stage = "baseline" if epoch <= 0 else epoch_label(epoch)
        artifacts = self.evaluator.run(
            profile=profile,
            use_skills=use_skills,
            tasks=options.tasks,
            combinations=options.combinations,
            seed=options.seed,
            family=options.family,
            max_steps=options.max_steps,
            include_candidate_skills=options.include_candidate_skills,
            output_dir=options.output_dir / "evaluations" / stage / mode,
        )
        return recorder.record_evaluation(
            label=label,
            mode=mode,
            epoch=epoch,
            checkpoint=checkpoint,
            artifacts=artifacts,
            final_checkpoint=final_checkpoint,
        )

    def run(
        self, job: AdapterJob, options: TrainingEvaluationOptions
    ) -> TrainingEvaluationResult:
        if not options.tasks:
            raise ValueError("训练评测任务列表不能为空")
        if options.combinations <= 0:
            raise ValueError("训练评测 combinations 必须是正整数")
        if options.max_steps <= 0:
            raise ValueError("训练评测 max_steps 必须是正整数")
        if options.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience 必须是正整数")
        if not 0 <= options.early_stopping_min_delta <= 1:
            raise ValueError("early_stopping_min_delta 必须在 [0, 1]")
        output_has_files = options.output_dir.exists() and any(
            options.output_dir.iterdir()
        )
        if output_has_files and not options.resume:
            raise FileExistsError(
                "训练评测输出目录必须为空，避免混入旧 checkpoint/报告: "
                f"{options.output_dir}"
            )
        resume_state = (
            _load_resume_state(options.output_dir) if output_has_files else None
        )
        previous_status = (
            str(resume_state.get("status", "")) if resume_state is not None else ""
        )
        previous_early = (
            resume_state.get("early_stopping")
            if isinstance(resume_state, dict)
            else None
        )
        if output_has_files and resume_state is None:
            raise FileExistsError(
                "训练评测输出目录非空但缺少 history.json，无法安全判断已完成阶段: "
                f"{options.output_dir}"
            )
        validate_staged_training_args(job)
        # SR 早退必须逐 epoch 获得可比观测；完整报告的用户配置可以更稀疏，
        # 但启用早退后 standalone probe 会自动收紧到每个 epoch 一次。
        effective_every_epochs = (
            1 if options.early_stopping_enabled else options.every_epochs
        )
        plan = build_epoch_plan(
            self.config.offline.epochs,
            effective_every_epochs,
            options.checkpoint_every_epochs,
        )
        training_output = options.output_dir / "training"
        snapshot_dir = options.output_dir / "dataset_snapshot"
        if resume_state is not None:
            prepared = load_prepared_training_job(job, snapshot_dir=snapshot_dir)
        else:
            prepared = prepare_training_job(
                job,
                configured_dataset_dir=self.config.offline.dataset_dir,
                snapshot_dir=snapshot_dir,
            )
        job = prepared.job
        manifest = {
            "schema_version": 4,
            "job": job.name,
            "family": options.family,
            "tasks": list(options.tasks),
            "task_count": len(options.tasks),
            "combinations": options.combinations,
            "planned_episodes_per_stage": len(options.tasks)
            * options.combinations,
            "seed": options.seed,
            "evaluation_max_steps": options.max_steps,
            "every_epochs": options.every_epochs,
            "effective_evaluation_every_epochs": effective_every_epochs,
            "checkpoint_every_epochs": options.checkpoint_every_epochs,
            "total_epochs": self.config.offline.epochs,
            "epoch_plan": [asdict(stage) for stage in plan],
            "dataset_snapshot_manifest": str(prepared.manifest_path),
            "dataset": prepared.manifest,
            "include_candidate_skills": options.include_candidate_skills,
            "full_evaluation": options.full_evaluation,
            "early_stopping_enabled": options.early_stopping_enabled,
            "early_stopping_patience": options.early_stopping_patience,
            "early_stopping_min_delta": options.early_stopping_min_delta,
            "output_layout": {
                "training": "training/epoch_XXX/",
                "evaluations": "evaluations/epoch_XXX/{standalone,skills}/",
                "baseline": "evaluations/baseline/{standalone,skills}/",
            },
            "resource_assignment": {
                "training_cuda_visible_devices": (
                    options.training_cuda_visible_devices
                ),
                "evaluation_cuda_visible_devices": (
                    options.evaluation_cuda_visible_devices
                ),
                "evaluation_max_model_len": options.evaluation_max_model_len,
                "evaluation_gpu_memory_utilization": (
                    options.evaluation_gpu_memory_utilization
                ),
            },
        }
        if resume_state is not None:
            existing_manifest = resume_state.get("manifest", {})
            if not isinstance(existing_manifest, dict):
                raise ValueError("续训 history.json 缺少合法 manifest")
            _validate_resume_manifest(existing_manifest, manifest)
            if (
                previous_status != "early_stopped"
                and float(manifest["total_epochs"])
                > float(existing_manifest.get("total_epochs", 0)) + 1e-9
            ):
                # 原来的“最终技能评测”现在只是一个中间 epoch 记录，腾出稳定标签给新终点。
                for row in resume_state.get("stages", []):
                    if not isinstance(row, dict):
                        continue
                    if row.get("label") == "final_skills":
                        row["label"] = f"{row.get('stage', 'previous')}_skills"
                    row["is_final_checkpoint"] = False
            manifest["dataset_snapshot_manifest"] = existing_manifest.get(
                "dataset_snapshot_manifest", manifest["dataset_snapshot_manifest"]
            )
            manifest["dataset"] = existing_manifest.get("dataset", manifest["dataset"])
        recorder = TrainingEvaluationRecorder(
            options.output_dir,
            manifest,
            resume_state=resume_state,
        )
        monitor = SuccessRateEarlyStopping(
            patience=options.early_stopping_patience,
            min_delta=options.early_stopping_min_delta,
        )
        # history 中只有 standalone 行参与早退；技能库 SR 不能与裸模型曲线混用。
        for row in recorder.state["stages"]:
            if not isinstance(row, dict) or row.get("mode") != "standalone":
                continue
            label = str(row.get("label", ""))
            if label != "baseline_standalone" and float(row.get("epoch", 0)) <= 0:
                continue
            monitor.observe(
                epoch=float(row.get("epoch", 0)),
                micro_sr=float(row.get("micro_sr", 0)),
                checkpoint=(
                    str(row["checkpoint"]) if row.get("checkpoint") else None
                ),
            )

        early_stopped = bool(
            options.early_stopping_enabled
            and (
                previous_status == "early_stopped"
                or (
                    monitor.should_stop
                    and monitor.observations
                    and float(monitor.observations[-1]["epoch"])
                    < self.config.offline.epochs - 1e-9
                )
            )
        )
        stop_epoch: float | None = None
        if early_stopped:
            if (
                isinstance(previous_early, dict)
                and previous_early.get("stop_epoch") is not None
            ):
                stop_epoch = float(previous_early["stop_epoch"])
            elif monitor.observations:
                stop_epoch = float(monitor.observations[-1]["epoch"])
        recorder.record_early_stopping(
            monitor.to_dict(
                enabled=options.early_stopping_enabled,
                stopped=early_stopped,
                stop_epoch=stop_epoch,
            )
        )

        checkpoint: Path | None = None
        discovered_checkpoint: Path | None = None
        try:
            discovered_checkpoint = find_latest_adapter_checkpoint(training_output)
        except FileNotFoundError:
            pass
        if discovered_checkpoint is not None:
            checkpoint = discovered_checkpoint
        resumed_from_checkpoint = discovered_checkpoint if resume_state else None
        if resume_state is not None:
            logging.info(
                "继续训练运行: output_dir=%s latest_checkpoint=%s completed_evaluations=%d",
                options.output_dir,
                resumed_from_checkpoint,
                len(recorder.state["stages"]),
            )
        transient_checkpoints: list[tuple[Path, dict[str, Any]]] = []
        for row in recorder.state["checkpoints"]:
            if row.get("retained", True) or not row.get("exists", True):
                continue
            candidate = Path(str(row.get("checkpoint", ""))).expanduser()
            if candidate.is_dir():
                transient_checkpoints.append((candidate.resolve(), row))
        return_code = 0
        try:
            missing_baselines = [
                ("baseline_standalone", False, "standalone"),
            ]
            if options.full_evaluation:
                missing_baselines.append(("baseline_skills", True, "skills"))
            missing_baselines = [
                item
                for item in missing_baselines
                if not recorder.evaluation_recorded(item[0])
            ]
            if missing_baselines and not early_stopped:
                # 同一次部署补齐尚未完成的基线；已写入 history 的阶段不重复跑。
                with self.deployment.activate(None) as profile:
                    for label, use_skills, mode in missing_baselines:
                        _archive_interrupted_evaluation(
                            options.output_dir / "evaluations" / "baseline" / mode
                        )
                        row = self._evaluate(
                            recorder=recorder,
                            profile=profile,
                            options=options,
                            label=label,
                            use_skills=use_skills,
                            epoch=0.0,
                            checkpoint=None,
                        )
                        if options.early_stopping_enabled and not use_skills:
                            monitor.observe(
                                epoch=0.0,
                                micro_sr=float(row["micro_sr"]),
                                checkpoint=None,
                            )
                            recorder.record_early_stopping(
                                monitor.to_dict(
                                    enabled=True,
                                    stopped=False,
                                    stop_epoch=None,
                                )
                            )

            for stage in plan:
                if early_stopped:
                    break
                target_epoch = stage.target_epoch
                stage_output = training_output / epoch_label(target_epoch)
                recorded_checkpoint = recorder.checkpoint_for_epoch(target_epoch)
                checkpoint_row: dict[str, Any] | None = None
                if recorder.checkpoint_recorded(target_epoch):
                    # checkpoint 可能已按保留策略删除；后续阶段会使用仍存在的最新项。
                    if recorded_checkpoint is not None:
                        checkpoint = recorded_checkpoint
                    logging.info("跳过已完成训练阶段: %s", epoch_label(target_epoch))
                else:
                    resume_checkpoint = checkpoint
                    if discovered_checkpoint is not None:
                        try:
                            belongs_to_current_stage = (
                                discovered_checkpoint.is_relative_to(stage_output)
                            )
                        except AttributeError:  # pragma: no cover - Python < 3.9
                            belongs_to_current_stage = stage_output.resolve() in (
                                discovered_checkpoint.resolve().parents
                            )
                        if belongs_to_current_stage:
                            resume_checkpoint = discovered_checkpoint
                    stage_job = staged_training_job(
                        job,
                        output_dir=stage_output,
                        target_epoch=target_epoch,
                        resume_from_checkpoint=resume_checkpoint,
                    )
                    recorder.record_command(
                        target_epoch=target_epoch,
                        stage_output_dir=stage_output,
                        evaluate=stage.evaluate,
                        retain_checkpoint=stage.retain_checkpoint,
                        resume=resume_checkpoint,
                        command=self.trainer.build_command(stage_job),
                    )
                    return_code = self.trainer.run(stage_job)
                    if return_code != 0:
                        recorder.finish(
                            "failed",
                            error=(
                                f"ms-swift 训练到 epoch {target_epoch:g} 失败，"
                                f"return_code={return_code}"
                            ),
                        )
                        break
                    checkpoint = find_latest_adapter_checkpoint(stage_output)
                    discovered_checkpoint = checkpoint
                    checkpoint_row = recorder.record_checkpoint(
                        epoch=target_epoch,
                        checkpoint=checkpoint,
                        stage_output_dir=stage_output,
                        retained=stage.retain_checkpoint,
                    )
                    if not stage.retain_checkpoint:
                        transient_checkpoints.append((checkpoint, checkpoint_row))
                if stage.evaluate:
                    standalone_label = f"{epoch_label(target_epoch)}_standalone"
                    needs_standalone = not recorder.evaluation_recorded(
                        standalone_label
                    )
                    needs_final_skills = (
                        options.full_evaluation
                        and stage.final
                        and not recorder.evaluation_recorded("final_skills")
                    )
                    if needs_standalone or needs_final_skills:
                        if checkpoint is None:
                            raise FileNotFoundError(
                                f"epoch {target_epoch:g} 已记录但没有可用 checkpoint"
                            )
                        with self.deployment.activate(checkpoint) as profile:
                            standalone_row: dict[str, Any] | None = None
                            if needs_standalone:
                                _archive_interrupted_evaluation(
                                    options.output_dir
                                    / "evaluations"
                                    / epoch_label(target_epoch)
                                    / "standalone"
                                )
                                standalone_row = self._evaluate(
                                    recorder=recorder,
                                    profile=profile,
                                    options=options,
                                    label=standalone_label,
                                    use_skills=False,
                                    epoch=target_epoch,
                                    checkpoint=checkpoint,
                                    final_checkpoint=stage.final,
                                )
                                if options.early_stopping_enabled:
                                    monitor.observe(
                                        epoch=target_epoch,
                                        micro_sr=float(standalone_row["micro_sr"]),
                                        checkpoint=str(checkpoint),
                                    )
                                    # 到计划终点自然结束，不把它误报成“提前”停止。
                                    early_stopped = bool(
                                        monitor.should_stop and not stage.final
                                    )
                                    if early_stopped:
                                        stop_epoch = target_epoch
                                        recorder.mark_evaluation_final(standalone_row)
                                        if checkpoint_row is None:
                                            checkpoint_row = next(
                                                (
                                                    row
                                                    for row in reversed(
                                                        recorder.state["checkpoints"]
                                                    )
                                                    if math.isclose(
                                                        float(row.get("epoch", -1)),
                                                        target_epoch,
                                                        abs_tol=1e-9,
                                                    )
                                                ),
                                                None,
                                            )
                                        if checkpoint_row is not None:
                                            recorder.mark_checkpoint_retained(
                                                checkpoint_row
                                            )
                                    recorder.record_early_stopping(
                                        monitor.to_dict(
                                            enabled=True,
                                            stopped=early_stopped,
                                            stop_epoch=stop_epoch,
                                        )
                                    )
                                    if (
                                        early_stopped
                                        and options.full_evaluation
                                        and not recorder.evaluation_recorded(
                                            "final_skills"
                                        )
                                    ):
                                        needs_final_skills = True
                            if needs_final_skills:
                                _archive_interrupted_evaluation(
                                    options.output_dir
                                    / "evaluations"
                                    / epoch_label(target_epoch)
                                    / "skills"
                                )
                                # 最终 checkpoint 再补一次技能库总评。
                                self._evaluate(
                                    recorder=recorder,
                                    profile=profile,
                                    options=options,
                                    label="final_skills",
                                    use_skills=True,
                                    epoch=target_epoch,
                                    checkpoint=checkpoint,
                                    final_checkpoint=True,
                                )
            # 进程若在“评测已写入、早退状态尚未收尾”之间中断，续训会从 history
            # 重建 monitor；这里补齐最终技能评测而不继续训练。
            if (
                return_code == 0
                and early_stopped
                and options.full_evaluation
                and checkpoint is not None
                and not recorder.evaluation_recorded("final_skills")
            ):
                final_epoch = stop_epoch or float(
                    monitor.observations[-1]["epoch"]
                )
                final_row = next(
                    (
                        row
                        for row in reversed(recorder.state["stages"])
                        if row.get("mode") == "standalone"
                        and math.isclose(
                            float(row.get("epoch", -1)), final_epoch, abs_tol=1e-9
                        )
                    ),
                    None,
                )
                if final_row is not None:
                    recorder.mark_evaluation_final(final_row)
                with self.deployment.activate(checkpoint) as profile:
                    _archive_interrupted_evaluation(
                        options.output_dir
                        / "evaluations"
                        / epoch_label(final_epoch)
                        / "skills"
                    )
                    self._evaluate(
                        recorder=recorder,
                        profile=profile,
                        options=options,
                        label="final_skills",
                        use_skills=True,
                        epoch=final_epoch,
                        checkpoint=checkpoint,
                        final_checkpoint=True,
                    )
            if return_code == 0:
                for transient, checkpoint_row in transient_checkpoints:
                    if (
                        early_stopped
                        and checkpoint is not None
                        and transient.resolve() == checkpoint.resolve()
                    ):
                        recorder.mark_checkpoint_retained(checkpoint_row)
                        continue
                    if transient.is_dir():
                        _remove_transient_checkpoint(transient)
                    if checkpoint_row.get("exists", True):
                        recorder.mark_checkpoint_removed(checkpoint_row)
                recorder.finish("early_stopped" if early_stopped else "completed")
        except Exception as exc:
            recorder.finish(
                "failed",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            raise

        return TrainingEvaluationResult(
            return_code=return_code,
            early_stopped=early_stopped,
            stop_epoch=stop_epoch,
            best_sr=monitor.best_sr,
            resumed=resume_state is not None,
            resumed_from_checkpoint=resumed_from_checkpoint,
            output_dir=options.output_dir,
            training_output_dir=training_output,
            final_checkpoint=checkpoint,
            history_json=recorder.history_json,
            history_csv=recorder.history_csv,
            comparison_markdown=recorder.comparison_markdown,
            dataset_manifest=prepared.manifest_path,
            checkpoints_manifest=recorder.checkpoints_path,
            stages=list(recorder.state["stages"]),
        )
