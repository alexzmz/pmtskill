"""模型×技能的无 LLM 配对标定编排、持久化和报告。

核心协议借鉴 SKVM bench：同一个 task 分别运行 ``no-skill`` 和
``forced-skill`` condition，再比较 score、步骤和耗时。区别在于这里的 score 来自
AndroidWorld task evaluator，而不是让另一个大模型评价输出。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core.io import write_json_atomic, write_jsonl
from ..core.models import ModelProfile, SkillRecord, SkillStatus
from ..evaluation.reporter import (
    episode_step_values,
    finite_float_value,
    normalize_episode_data,
    successful_episode_value,
)
from ..skills.identity import model_variant_id, skill_version_id
from ..skills.importer import relevant_raw_skills
from ..skills.store import SkillStore
from .runner import (
    AndroidWorldSkillConditionRunner,
    SkillCalibrationConditionRunner,
    SkillConditionResult,
)


PROTOCOL_ID = "skvm-paired-android-world-v1"


@dataclass(slots=True)
class CalibrationOptions:
    """一次标定的固定范围；相同 run 恢复时这些字段必须完全一致。"""

    output_dir: Path
    tasks: tuple[str, ...] | None = None
    family: str = "android_world"
    combinations: int = 1
    seed: int = 42
    max_steps: int = 30
    skill_ids: tuple[str, ...] = ()
    skill_kind: str = "polished"
    statuses: tuple[str, ...] = (
        SkillStatus.CANDIDATE.value,
        SkillStatus.ACTIVE.value,
    )
    task_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_ID,
            "output_dir": str(self.output_dir.resolve()),
            "tasks": list(self.tasks) if self.tasks is not None else None,
            "family": self.family,
            "combinations": self.combinations,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "skill_ids": list(self.skill_ids),
            "skill_kind": self.skill_kind,
            "statuses": list(self.statuses),
            "task_map": {key: list(value) for key, value in self.task_map.items()},
        }


@dataclass(slots=True)
class CalibrationArtifacts:
    output_dir: Path
    summary_json: Path
    report_markdown: Path
    matrix_csv: Path
    trials_jsonl: Path
    summary: dict[str, Any]


def _slug(value: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "unknown"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{suffix}"


def _json_safe(value: Any) -> Any:
    """把 datetime/numpy scalar 等 episode 边缘类型转成可恢复 JSON。"""

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _episode_field(episode: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in episode:
            return episode[name]
    return default


def _episode_key(episode: Mapping[str, Any], index: int) -> tuple[str, str]:
    task = str(
        _episode_field(episode, "task_template", "task_name", default="unknown")
    )
    seed = _episode_field(episode, "seed")
    instance = _episode_field(episode, "instance_id", default=index)
    # AndroidWorld 为每个 task combination 生成稳定 seed；旧 episode 没有 seed 时
    # 回退 instance_id，仍可与相同顺序的 condition 配对。
    identity = str(seed if seed is not None else f"instance:{instance}")
    return task, identity


def _episode_parse_rate(episode: Mapping[str, Any]) -> float:
    data = normalize_episode_data(_episode_field(episode, "episode_data", default={}))
    raw = episode_step_values(data.get("action_raw_response"))
    parsed = episode_step_values(data.get("action_output_json"))
    denominator = max(len(raw), len(parsed))
    if denominator == 0:
        return 0.0
    valid = sum(isinstance(value, Mapping) for value in parsed)
    return min(1.0, valid / denominator)


def _condition_cache_path(output_dir: Path) -> Path:
    return output_dir / "condition-result.json"


def _save_condition(result: SkillConditionResult) -> None:
    write_json_atomic(
        _condition_cache_path(result.output_dir),
        _json_safe(
            {
                "condition": result.condition,
                "model_id": result.model_id,
                "skill_id": result.skill_id,
                "summary": result.summary,
                "episodes": list(result.episodes),
            }
        ),
    )


def _load_condition(output_dir: Path) -> SkillConditionResult | None:
    path = _condition_cache_path(output_dir)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return SkillConditionResult(
        condition=str(value["condition"]),
        output_dir=output_dir,
        summary=dict(value.get("summary", {})),
        episodes=tuple(value.get("episodes", ())),
        model_id=str(value["model_id"]),
        skill_id=value.get("skill_id"),
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    rows = summary.get("model_skill_metrics", [])
    lines = [
        "# PMT-Skill 模型×技能标定报告",
        "",
        "本报告采用 SKVM-style `no-skill` / `forced-skill` 配对 condition；",
        "成功标签由 AndroidWorld task evaluator 产生，未使用额外 LLM judge。",
        "",
        "## 总览",
        "",
        f"- 标定协议：`{summary.get('protocol')}`",
        f"- run ID：`{summary.get('run_id')}`",
        f"- 模型版本数：{summary.get('model_variant_count', 0)}",
        f"- 技能版本数：{summary.get('skill_version_count', 0)}",
        f"- 有效/全部配对 trial：{summary.get('valid_trials', 0)} / "
        f"{summary.get('trials_total', 0)}",
        f"- 未找到适用任务的技能对：{len(summary.get('skipped_pairs', []))}",
        "",
        "## 模型–技能矩阵",
        "",
        "| 模型 | 技能 | 有效试验 | 裸模型 SR | 技能 SR | SR uplift | Wilson 下界 | W/L/T |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['skill_id']} | {row['valid_trials']} | "
            f"{row['baseline_success_rate']:.2%} | "
            f"{row['skill_success_rate']:.2%} | "
            f"{row['success_rate_uplift']:+.2%} | "
            f"{row['skill_success_wilson_lower']:.2%} | "
            f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} |"
        )
    if not rows:
        lines.append("| （暂无有效标定） | - | 0 | - | - | - | - | - |")
    lines.extend(
        (
            "",
            "## 使用说明",
            "",
            "- `技能 SR` 表示强制注入该技能后的端到端任务成功率。",
            "- `SR uplift` 是技能 SR 减去同模型、同任务实例的裸模型 SR。",
            "- `Wilson 下界` 用于避免少量样本的偶然 100% 直接触发技能晋升。",
            "- 基础设施异常的配对会保留在 trials.jsonl，但不进入 SR 分母。",
            "- 标定任务必须与最终论文测试实例隔离，避免路由器使用测试集反馈。",
            "",
        )
    )
    return "\n".join(lines)


class SkillCalibrationWorkflow:
    """运行、恢复并汇总 model variant × skill version 配对标定。"""

    def __init__(
        self,
        store: SkillStore,
        runner: SkillCalibrationConditionRunner | None = None,
        *,
        config=None,
    ):
        if runner is None:
            if config is None:
                raise ValueError("未提供 runner 时必须提供 ProjectConfig")
            runner = AndroidWorldSkillConditionRunner(config, store)
        self.store = store
        self.runner = runner

    def _skills(self, options: CalibrationOptions) -> list[SkillRecord]:
        if options.skill_ids:
            result = []
            missing = []
            for skill_id in options.skill_ids:
                skill = self.store.get_skill(skill_id)
                if skill is None:
                    missing.append(skill_id)
                else:
                    result.append(skill)
            if missing:
                raise ValueError("技能库中不存在: " + ", ".join(missing))
            return result

        kinds = ("raw", "polished") if options.skill_kind == "all" else (
            options.skill_kind,
        )
        result: list[SkillRecord] = []
        allowed_statuses = set(options.statuses)
        for kind in kinds:
            candidates = self.store.list_skills(kind=kind)
            if kind == "raw":
                result.extend(relevant_raw_skills(candidates))
            else:
                result.extend(
                    skill
                    for skill in candidates
                    if skill.status.value in allowed_statuses
                )
        if not result:
            raise ValueError("当前筛选条件下没有可标定技能")
        return result

    @staticmethod
    def _tasks_for_skill(
        skill: SkillRecord, options: CalibrationOptions
    ) -> tuple[tuple[str, ...] | None, str]:
        mapped = options.task_map.get(skill.skill_id)
        source = "task-map"
        if mapped is None:
            raw = skill.metadata.get("calibration_tasks")
            mapped = (
                tuple(str(item) for item in raw)
                if isinstance(raw, (list, tuple))
                else None
            )
            source = "skill-metadata" if mapped is not None else "global-scope"
        if mapped is None:
            return options.tasks, source
        if options.tasks is None:
            return tuple(dict.fromkeys(mapped)), source
        allowed = set(options.tasks)
        return tuple(item for item in dict.fromkeys(mapped) if item in allowed), source

    def _run_or_load(
        self,
        output_dir: Path,
        *,
        resume: bool,
        execute: Callable[[], SkillConditionResult],
    ) -> SkillConditionResult:
        cached = _load_condition(output_dir) if resume else None
        if cached is not None:
            return cached
        result = execute()
        _save_condition(result)
        return result

    def _record_pairs(
        self,
        *,
        run_id: str,
        profile: ModelProfile,
        skill: SkillRecord,
        baseline: SkillConditionResult,
        forced: SkillConditionResult,
        allowed_tasks: Sequence[str] | None,
    ) -> int:
        allowed = set(allowed_tasks) if allowed_tasks is not None else None
        baseline_by_key = {
            _episode_key(episode, index): episode
            for index, episode in enumerate(baseline.episodes)
            if allowed is None or _episode_key(episode, index)[0] in allowed
        }
        forced_by_key = {
            _episode_key(episode, index): episode
            for index, episode in enumerate(forced.episodes)
        }
        keys = sorted(set(baseline_by_key) | set(forced_by_key))
        for task_name, task_seed in keys:
            baseline_episode = baseline_by_key.get((task_name, task_seed))
            skill_episode = forced_by_key.get((task_name, task_seed))
            baseline_exception = (
                _episode_field(baseline_episode, "exception_info")
                if baseline_episode is not None
                else "missing baseline episode"
            )
            skill_exception = (
                _episode_field(skill_episode, "exception_info")
                if skill_episode is not None
                else "missing forced-skill episode"
            )
            valid = (
                baseline_episode is not None
                and skill_episode is not None
                and not baseline_exception
                and not skill_exception
            )
            stable = (
                f"{run_id}|{model_variant_id(profile)}|{skill_version_id(skill)}|"
                f"{task_name}|{task_seed}"
            )
            pair_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
            self.store.record_skill_calibration_trial(
                run_id=run_id,
                pair_id=pair_id,
                skill=skill,
                profile=profile,
                task_name=task_name,
                task_seed=task_seed,
                baseline_success=(
                    successful_episode_value(
                        _episode_field(
                            baseline_episode or {}, "is_successful", default=False
                        )
                    )
                ),
                skill_success=(
                    successful_episode_value(
                        _episode_field(
                            skill_episode or {}, "is_successful", default=False
                        )
                    )
                ),
                baseline_steps=int(
                    finite_float_value(
                        _episode_field(
                            baseline_episode or {}, "episode_length", default=0
                        )
                    )
                ),
                skill_steps=int(
                    finite_float_value(
                        _episode_field(
                            skill_episode or {}, "episode_length", default=0
                        )
                    )
                ),
                baseline_run_time_ms=1000
                * finite_float_value(
                    _episode_field(baseline_episode or {}, "run_time", default=0)
                ),
                skill_run_time_ms=1000
                * finite_float_value(
                    _episode_field(skill_episode or {}, "run_time", default=0)
                ),
                baseline_parse_rate=_episode_parse_rate(baseline_episode or {}),
                skill_parse_rate=_episode_parse_rate(skill_episode or {}),
                valid=valid,
                detail={
                    "baseline_exception": baseline_exception,
                    "skill_exception": skill_exception,
                    "baseline_goal": _episode_field(
                        baseline_episode or {}, "goal", default=""
                    ),
                    "skill_goal": _episode_field(
                        skill_episode or {}, "goal", default=""
                    ),
                    "baseline_condition_dir": str(baseline.output_dir),
                    "skill_condition_dir": str(forced.output_dir),
                },
            )
        return len(keys)

    def run(
        self,
        profiles: Sequence[ModelProfile],
        options: CalibrationOptions,
    ) -> CalibrationArtifacts:
        if not profiles:
            raise ValueError("至少需要一个模型/adapter 参与标定")
        if options.combinations <= 0:
            raise ValueError("标定 combinations 必须是正整数")
        if options.max_steps <= 0:
            raise ValueError("标定 max_steps 必须是正整数")

        target = options.output_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "calibration_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if options.resume and manifest_path.is_file()
            else None
        )
        current_config = options.to_dict()
        current_config["model_variants"] = [
            model_variant_id(profile) for profile in profiles
        ]
        if manifest is not None:
            if manifest.get("config") != current_config:
                raise ValueError(
                    "标定输出目录中的已有配置与本次参数不同；请换目录或使用 --no-resume"
                )
            run_id = str(manifest["run_id"])
        else:
            if any(target.iterdir()):
                reason = (
                    "目录非空但缺少 calibration_manifest.json，不能安全恢复"
                    if options.resume
                    else "--no-resume 要求 --output-dir 为空或不存在"
                )
                raise ValueError(reason)
            run_id = uuid.uuid4().hex
            write_json_atomic(
                manifest_path,
                {"run_id": run_id, "protocol": PROTOCOL_ID, "config": current_config},
            )

        self.store.start_calibration_run(
            run_id, current_config, target, protocol=PROTOCOL_ID
        )
        skills = self._skills(options)
        skipped_pairs: list[dict[str, Any]] = []

        try:
            for profile in profiles:
                self.store.register_model_variant(profile)
                model_dir = target / "conditions" / _slug(profile.model_id)
                baseline_dir = model_dir / "no-skill"
                baseline = self._run_or_load(
                    baseline_dir,
                    resume=options.resume,
                    execute=lambda profile=profile, baseline_dir=baseline_dir: (
                        self.runner.run_no_skill(
                            profile=profile,
                            tasks=options.tasks,
                            combinations=options.combinations,
                            seed=options.seed,
                            family=options.family,
                            max_steps=options.max_steps,
                            output_dir=baseline_dir,
                        )
                    ),
                )
                for skill in skills:
                    skill_tasks, scope_source = self._tasks_for_skill(skill, options)
                    if skill_tasks == ():
                        skipped_pairs.append(
                            {
                                "model_id": profile.model_id,
                                "model_variant_id": model_variant_id(profile),
                                "skill_id": skill.skill_id,
                                "skill_version_id": skill_version_id(skill),
                                "reason": "no_applicable_tasks_in_selected_scope",
                                "scope_source": scope_source,
                            }
                        )
                        continue
                    forced_dir = model_dir / "forced-skill" / _slug(skill.skill_id)
                    forced = self._run_or_load(
                        forced_dir,
                        resume=options.resume,
                        execute=lambda profile=profile, skill=skill,
                        skill_tasks=skill_tasks, forced_dir=forced_dir: (
                            self.runner.run_forced_skill(
                                profile=profile,
                                skill=skill,
                                tasks=skill_tasks,
                                combinations=options.combinations,
                                seed=options.seed,
                                family=options.family,
                                max_steps=options.max_steps,
                                output_dir=forced_dir,
                            )
                        ),
                    )
                    self._record_pairs(
                        run_id=run_id,
                        profile=profile,
                        skill=skill,
                        baseline=baseline,
                        forced=forced,
                        allowed_tasks=skill_tasks,
                    )

            metrics = []
            for profile in profiles:
                variant = model_variant_id(profile)
                for skill in skills:
                    row = self.store.calibration_metrics(
                        skill.skill_id,
                        model_id=profile.model_id,
                        model_variant_id_value=variant,
                        run_id=run_id,
                    )
                    if row["trials_total"]:
                        metrics.append(row)
            trials = self.store.list_calibration_trials(run_id=run_id)
            summary = {
                "protocol": PROTOCOL_ID,
                "label_source": "android_world_task_evaluator",
                "uses_external_llm_judge": False,
                "run_id": run_id,
                "model_variant_count": len(profiles),
                "skill_version_count": len(skills),
                "trials_total": len(trials),
                "valid_trials": sum(bool(row["valid"]) for row in trials),
                "invalid_trials": sum(not bool(row["valid"]) for row in trials),
                "model_skill_metrics": metrics,
                "skipped_pairs": skipped_pairs,
                "config": current_config,
            }
            summary_path = target / "summary.json"
            trials_path = target / "trials.jsonl"
            matrix_path = target / "model_skill_matrix.csv"
            report_path = target / "report.md"
            write_json_atomic(summary_path, _json_safe(summary))
            write_jsonl(trials_path, (_json_safe(row) for row in trials))
            columns = (
                "model_id",
                "model_variant_id",
                "skill_id",
                "skill_version_id",
                "valid_trials",
                "skill_success_rate",
                "baseline_success_rate",
                "success_rate_uplift",
                "skill_success_wilson_lower",
                "paired_wins",
                "paired_losses",
                "paired_ties",
                "average_skill_steps",
                "average_baseline_steps",
                "average_skill_run_time_ms",
                "average_baseline_run_time_ms",
                "average_skill_parse_rate",
                "average_baseline_parse_rate",
            )
            with matrix_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(metrics)
            report_path.write_text(_markdown(summary), encoding="utf-8")
            self.store.finish_calibration_run(
                run_id, status="completed", summary=summary
            )
            return CalibrationArtifacts(
                target,
                summary_path,
                report_path,
                matrix_path,
                trials_path,
                summary,
            )
        except Exception as exc:
            self.store.finish_calibration_run(
                run_id,
                status="failed",
                summary={
                    "protocol": PROTOCOL_ID,
                    "run_id": run_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
