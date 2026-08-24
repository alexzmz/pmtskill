"""LoRA 训练与 AndroidWorld SR 评测的可选闭环编排。"""

from __future__ import annotations

import csv
import datetime as dt
import math
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass
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
    include_candidate_skills: bool = False


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
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "status": "running",
            "manifest": manifest,
            "stages": [],
            "training_commands": [],
            "error": None,
        }
        write_json_atomic(self.manifest_path, manifest)
        self.flush()

    def record_command(
        self, *, target_epoch: float, resume: Path | None, command: Sequence[str]
    ) -> None:
        self.state["training_commands"].append(
            {
                "target_epoch": target_epoch,
                "resume_from_checkpoint": str(resume) if resume else None,
                "command": list(command),
            }
        )
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
        self._write_csv()
        self._write_markdown(derived)


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
        artifacts = self.evaluator.run(
            profile=profile,
            use_skills=use_skills,
            tasks=options.tasks,
            combinations=options.combinations,
            seed=options.seed,
            family=options.family,
            include_candidate_skills=options.include_candidate_skills,
            output_dir=options.output_dir / "evaluations" / label,
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
        targets = build_epoch_targets(
            self.config.offline.epochs, options.every_epochs
        )
        training_output = options.output_dir / "training"
        manifest = {
            "schema_version": 1,
            "job": job.name,
            "family": options.family,
            "tasks": list(options.tasks),
            "task_count": len(options.tasks),
            "combinations": options.combinations,
            "planned_episodes_per_stage": len(options.tasks)
            * options.combinations,
            "seed": options.seed,
            "every_epochs": options.every_epochs,
            "total_epochs": self.config.offline.epochs,
            "epoch_targets": targets,
            "include_candidate_skills": options.include_candidate_skills,
        }
        recorder = TrainingEvaluationRecorder(options.output_dir, manifest)
        checkpoint: Path | None = None
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

            for index, target_epoch in enumerate(targets):
                stage_job = staged_training_job(
                    job,
                    output_dir=training_output,
                    target_epoch=target_epoch,
                    resume_from_checkpoint=checkpoint,
                )
                recorder.record_command(
                    target_epoch=target_epoch,
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
                checkpoint = find_latest_adapter_checkpoint(training_output)
                is_final = index == len(targets) - 1
                with self.deployment.activate(checkpoint) as profile:
                    self._evaluate(
                        recorder=recorder,
                        profile=profile,
                        options=options,
                        label=f"{epoch_label(target_epoch)}_standalone",
                        use_skills=False,
                        epoch=target_epoch,
                        checkpoint=checkpoint,
                        final_checkpoint=is_final,
                    )
                    # 中间 epoch 只测裸模型；最终 checkpoint 再补一次技能库总评。
                    if is_final:
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
            stages=list(recorder.state["stages"]),
        )
