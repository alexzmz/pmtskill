"""LoRA 训练与 AndroidWorld SR 评测的可选闭环编排。"""

from __future__ import annotations

import csv
import datetime as dt
import math
import shutil
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
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
    every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    include_candidate_skills: bool = False
    training_cuda_visible_devices: str | None = None
    evaluation_cuda_visible_devices: str | None = None
    evaluation_max_model_len: int | None = None
    evaluation_gpu_memory_utilization: float | None = None


@dataclass(slots=True)
class TrainingEvaluationResult:
    """CLI 可直接序列化的训练评测结果。"""

    return_code: int
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
            output_dir=output_dir,
        )


class TrainingEvaluationRecorder:
    """增量维护 JSON/CSV/Markdown，异常退出时也保留已完成阶段。"""

    def __init__(self, output_dir: Path, manifest: dict[str, Any]):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_json = self.output_dir / "history.json"
        self.history_csv = self.output_dir / "history.csv"
        self.comparison_markdown = self.output_dir / "comparison.md"
        self.manifest_path = self.output_dir / "sample_manifest.json"
        self.command_path = self.output_dir / "training_stage_commands.json"
        self.checkpoints_path = self.output_dir / "checkpoints.json"
        self.state: dict[str, Any] = {
            "schema_version": 2,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "status": "running",
            "manifest": manifest,
            "stages": [],
            "training_commands": [],
            "checkpoints": [],
            "error": None,
        }
        write_json_atomic(self.manifest_path, manifest)
        self.flush()

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
            f"- 固定任务数：**{len(manifest['tasks'])}**",
            f"- 每任务组合数：**{manifest['combinations']}**",
            f"- 随机种子：**{manifest['seed']}**",
            "- 所有行使用同一任务列表与 seed，训练评测轨迹不会写回技能库。",
            "",
            "| 阶段 | 模式 | Epoch | 成功/有效 | Micro SR | Macro SR | 相对裸基座 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
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
        if options.output_dir.exists() and any(options.output_dir.iterdir()):
            raise FileExistsError(
                "训练评测输出目录必须为空，避免混入旧 checkpoint/报告: "
                f"{options.output_dir}"
            )
        validate_staged_training_args(job)
        plan = build_epoch_plan(
            self.config.offline.epochs,
            options.every_epochs,
            options.checkpoint_every_epochs,
        )
        training_output = options.output_dir / "training"
        prepared = prepare_training_job(
            job,
            configured_dataset_dir=self.config.offline.dataset_dir,
            snapshot_dir=options.output_dir / "dataset_snapshot",
        )
        job = prepared.job
        manifest = {
            "schema_version": 2,
            "job": job.name,
            "family": options.family,
            "tasks": list(options.tasks),
            "task_count": len(options.tasks),
            "combinations": options.combinations,
            "planned_episodes_per_stage": len(options.tasks)
            * options.combinations,
            "seed": options.seed,
            "every_epochs": options.every_epochs,
            "checkpoint_every_epochs": options.checkpoint_every_epochs,
            "total_epochs": self.config.offline.epochs,
            "epoch_plan": [asdict(stage) for stage in plan],
            "dataset_snapshot_manifest": str(prepared.manifest_path),
            "dataset": prepared.manifest,
            "include_candidate_skills": options.include_candidate_skills,
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
        recorder = TrainingEvaluationRecorder(options.output_dir, manifest)
        checkpoint: Path | None = None
        transient_checkpoints: list[tuple[Path, dict[str, Any]]] = []
        return_code = 0
        try:
            # 基座只部署一次，连续跑裸模型与“同模型+技能库”，避免重复加载权重。
            with self.deployment.activate(None) as profile:
                self._evaluate(
                    recorder=recorder,
                    profile=profile,
                    options=options,
                    label="baseline_standalone",
                    use_skills=False,
                    epoch=0.0,
                    checkpoint=None,
                )
                self._evaluate(
                    recorder=recorder,
                    profile=profile,
                    options=options,
                    label="baseline_skills",
                    use_skills=True,
                    epoch=0.0,
                    checkpoint=None,
                )

            for stage in plan:
                target_epoch = stage.target_epoch
                stage_output = training_output / epoch_label(target_epoch)
                stage_job = staged_training_job(
                    job,
                    output_dir=stage_output,
                    target_epoch=target_epoch,
                    resume_from_checkpoint=checkpoint,
                )
                recorder.record_command(
                    target_epoch=target_epoch,
                    stage_output_dir=stage_output,
                    evaluate=stage.evaluate,
                    retain_checkpoint=stage.retain_checkpoint,
                    resume=checkpoint,
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
                checkpoint_row = recorder.record_checkpoint(
                    epoch=target_epoch,
                    checkpoint=checkpoint,
                    stage_output_dir=stage_output,
                    retained=stage.retain_checkpoint,
                )
                if not stage.retain_checkpoint:
                    transient_checkpoints.append((checkpoint, checkpoint_row))
                if stage.evaluate:
                    with self.deployment.activate(checkpoint) as profile:
                        self._evaluate(
                            recorder=recorder,
                            profile=profile,
                            options=options,
                            label=f"{epoch_label(target_epoch)}_standalone",
                            use_skills=False,
                            epoch=target_epoch,
                            checkpoint=checkpoint,
                            final_checkpoint=stage.final,
                        )
                        # 中间 epoch 只测裸模型；最终 checkpoint 再补一次技能库总评。
                        if stage.final:
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
            if return_code == 0:
                for transient, checkpoint_row in transient_checkpoints:
                    _remove_transient_checkpoint(transient)
                    recorder.mark_checkpoint_removed(checkpoint_row)
                recorder.finish("completed")
        except Exception as exc:
            recorder.finish(
                "failed",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            raise

        return TrainingEvaluationResult(
            return_code=return_code,
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
