"""不连接 AndroidWorld 的模型—技能适配估计；分数不是 SR。

prior: 原语先验 + 当前 adapter 实际训练快照覆盖 + 技能结构复杂度。
teacher-forcing: 另在同一批已有视觉参考样本上比较有/无技能的平均 log likelihood。
估计独立入库，不生成 calibration trial，也不自动改变技能生命周期。
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import json
import logging
import math
import random
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..core.io import read_jsonl, write_json_atomic
from ..core.models import ModelProfile, SkillRecord
from ..evaluation.adapters import AdapterDeploymentBinding
from ..evaluation.capabilities import CAPABILITY_LABELS, CAPABILITY_ORDER, TASK_TO_CAPABILITY
from ..evaluation.resume import fingerprint, model_resume_identity, skills_resume_identity
from ..skills.identity import model_variant_id, skill_version_id
from ..skills.store import SkillStore
from .workflow import CalibrationOptions, SkillCalibrationWorkflow


LOGGER = logging.getLogger(__name__)
PROTOCOL = "pmtskill-offline-estimate-v1"


@dataclasses.dataclass(slots=True)
class OfflineOptions:
    output_dir: Path
    scorer: str = "prior"
    dataset: Path | None = None
    max_samples_per_skill: int = 8
    max_training_samples: int = 2000
    seed: int = 42
    resume: bool = True
    base_model_path: str | None = None
    scorer_settings: dict[str, Any] = dataclasses.field(default_factory=dict)


class ReferenceScorer(Protocol):
    def score(self, profile: ModelProfile, sample: Mapping[str, Any], skill: SkillRecord | None) -> float:
        """返回参考 assistant 响应的平均 token log probability（越大越好）。"""
        ...


def _training_path(binding: AdapterDeploymentBinding) -> Path | None:
    run = binding.checkpoint.run_dir
    path = run / "dataset_snapshot" / "train.jsonl" if run else None
    return path if path and path.is_file() else None


def _sample_key(row: Mapping[str, Any]) -> str:
    meta = row.get("metadata", {})
    # 按 episode 排除训练重叠，比只排除同一步更保守。
    if meta.get("episode_id"):
        return fingerprint([meta.get("task_name"), meta["episode_id"]])
    return fingerprint([row.get("messages"), row.get("images")])


def _training_evidence(path: Path | None, limit: int, seed: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    rng = random.Random(seed)
    count = 0
    if path:
        for count, row in enumerate(read_jsonl(path), 1):
            seen_keys.add(_sample_key(row))
            if len(selected) < limit:
                selected.append(row)
            else:
                index = rng.randrange(count)
                if index < limit:
                    selected[index] = row
    primitives: collections.Counter[str] = collections.Counter()
    tasks: collections.Counter[str] = collections.Counter()
    categories: collections.Counter[str] = collections.Counter()
    for row in selected:
        meta = row.get("metadata", {})
        primitives.update(set(meta.get("primitives", ())))
        task = str(meta.get("task_name", ""))
        if task:
            tasks[task] += 1
        if category := TASK_TO_CAPABILITY.get(task):
            categories[category] += 1
    return {"path": str(path) if path else None, "rows_seen": count, "rows_sampled": len(selected),
            "primitives": dict(primitives), "tasks": dict(tasks), "categories": dict(categories),
            "episode_keys": seen_keys,
            "fingerprint": fingerprint([row.get("metadata", {}) for row in selected])}


def _references(dataset: Path, excluded: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    rejected: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for row in read_jsonl(dataset):
        meta = row.get("metadata", {})
        if _sample_key(row) in excluded:
            rejected["training_episode_overlap"] += 1
            continue
        if meta.get("episode_outcome") not in {"successful", "success"}:
            rejected["not_verified_successful_reference"] += 1
            continue
        messages = row.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
            rejected["missing_reference_response"] += 1
            continue
        images = row.get("images", [])
        if not images:
            rejected["missing_visual_input"] += 1
            continue
        paths = [(Path(value) if Path(value).is_absolute() else dataset.parent / value).resolve()
                 for value in images]
        if not all(path.is_file() for path in paths):
            rejected["missing_image_file"] += 1
            continue
        normalized = {**row, "images": [str(path) for path in paths]}
        identity = fingerprint([messages, normalized["images"]])
        if identity in seen:
            rejected["duplicate_reference"] += 1
            continue
        seen.add(identity)
        normalized["reference_id"] = identity
        rows.append(normalized)
    return rows, dict(rejected)


def _prior(skill: SkillRecord, profile: ModelProfile, evidence: Mapping[str, Any], tasks: Sequence[str]) -> dict[str, Any]:
    primitives = tuple(dict.fromkeys(skill.metadata.get("leaf_primitive_ids") or skill.topology.primitive_sequence()))
    inherited = str(profile.metadata.get("capability_prior_source", "")).startswith("inherited-template:")
    known = [profile.capability(p) for p in primitives if p in profile.capabilities and not inherited]
    values = [profile.capability(p) if not inherited else 0.5 for p in primitives]
    prior = math.exp(sum(math.log(value) for value in values) / len(values)) if values else 0.5
    components = [(0.55, prior)]
    actionable = [p for p in primitives if p.startswith("action.") or p == "control.finish"]
    counts = evidence["primitives"]
    coverage = (sum(min(1.0, counts.get(p, 0) / 5) for p in actionable) / len(actionable)
                if actionable and counts else None)
    if coverage is not None:
        components.append((0.25, coverage))
    task_coverage = (sum(min(1.0, evidence["tasks"].get(task, 0) / 5) for task in tasks) / len(tasks)
                     if tasks and evidence["tasks"] else None)
    if task_coverage is not None:
        components.append((0.20, task_coverage))
    branches = sum(len(node.depends_on) > 1 or bool(node.condition) for node in skill.topology.nodes)
    penalty = 1 / (1 + 0.025 * max(0, len(skill.topology.nodes) - 1) + 0.10 * branches)
    score = sum(weight * value for weight, value in components) / sum(weight for weight, _ in components) * penalty
    return {"estimated_support_score": min(0.999999, max(0.000001, score)),
            "primitive_prior_score": prior, "known_primitive_count": len(known),
            "action_training_coverage": coverage, "task_training_coverage": task_coverage,
            "complexity_penalty": penalty, "training_rows_sampled": evidence["rows_sampled"],
            "training_snapshot": evidence["path"], "categories": sorted({TASK_TO_CAPABILITY[t] for t in tasks if t in TASK_TO_CAPABILITY}),
            "uncertainty": "unmeasured", "score_is_success_rate": False,
            "warnings": (["no_model_specific_primitive_profile"] if not known else [])
                        + (["no_training_snapshot"] if not evidence["rows_sampled"] else [])}


def run_offline_calibration(
    store: SkillStore, bindings: Sequence[AdapterDeploymentBinding], selection: CalibrationOptions,
    options: OfflineOptions, *, scorer: ReferenceScorer | None = None,
) -> dict[str, Any]:
    if options.scorer not in {"prior", "teacher-forcing"}:
        raise ValueError("未知的 offline scorer")
    if options.max_samples_per_skill <= 0 or options.max_training_samples <= 0:
        raise ValueError("离线样本预算必须大于 0")
    if options.scorer == "teacher-forcing" and options.dataset is None:
        raise ValueError("teacher-forcing 必须指定 --offline-dataset（保留截图的成功参考 JSONL）")
    if options.scorer == "teacher-forcing" and scorer is None:
        raise ValueError("teacher-forcing 未提供 ReferenceScorer")
    # 复用筛选规则，但此处从不调用 condition runner / AndroidWorld。
    skills = SkillCalibrationWorkflow(store, runner=object())._skills(selection)
    evidence = {binding.model_id: _training_evidence(_training_path(binding), options.max_training_samples, options.seed)
                for binding in bindings}
    excluded = set().union(*(item["episode_keys"] for item in evidence.values()))
    references, rejected = (_references(options.dataset, excluded) if options.scorer == "teacher-forcing" else ([], {}))
    if options.scorer == "teacher-forcing" and not references:
        raise ValueError(f"没有可用的独立视觉参考样本: {rejected}；不能静默退化为文本或虚构成功率")
    if options.scorer == "teacher-forcing" and not any(
        item.get("metadata", {}).get("task_name") in (SkillCalibrationWorkflow._tasks_for_skill(skill, selection)[0] or ())
        for skill in skills for item in references
    ):
        raise ValueError("参考样本与所选技能的任务映射没有交集；请提供 --skill-task-map 或匹配的参考集")
    profiles = [binding.profile(base_model_path=options.base_model_path or binding.checkpoint.base_model_path or "")
                for binding in bindings]
    image_state = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in sorted({path for row in references for path in row["images"]})}
    identities = {profile.model_id: model_resume_identity(profile) for profile in profiles}
    manifest_config = {
        "protocol": PROTOCOL, "scorer": options.scorer, "seed": options.seed,
        "max_samples_per_skill": options.max_samples_per_skill,
        "max_training_samples": options.max_training_samples,
        "scorer_settings": options.scorer_settings,
        "selection": {key: value for key, value in selection.to_dict().items() if key not in {"output_dir", "protocol", "combinations", "max_steps"}},
        "skills": [skill_version_id(skill) for skill in skills],
        "skill_definitions_and_metadata": skills_resume_identity(skills),
        "models": [identities[profile.model_id] for profile in profiles],
        "training": {key: item["fingerprint"] for key, item in evidence.items()},
        "references": fingerprint([references, image_state]),
    }
    target = options.output_dir.resolve()
    manifest_path = target / "offline_manifest.json"
    cache_path = target / "estimates.json"
    pair_cache_dir = target / "pairs"
    cached: dict[str, Any] = {}
    if manifest_path.is_file() and options.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != manifest_config:
            raise ValueError("离线估计续跑配置/权重/训练或参考数据变化，请使用新的 --output-dir")
        run_id = previous["run_id"]
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        for path in pair_cache_dir.glob("*.json"):
            try:
                cached[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                cached.pop(path.stem, None)
                LOGGER.warning("离线组合缓存损坏，将重新估计: %s", path, exc_info=True)
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("离线估计要求新目录，或使用 --resume 恢复同一 offline_manifest")
        run_id = uuid.uuid4().hex
        write_json_atomic(manifest_path, {"run_id": run_id, "config": manifest_config})
    rows = []
    baseline_cache: dict[tuple[str, str], float] = {}
    for skill in skills:
        store.register_skill_version(skill)
    for profile in profiles:
        store.register_model_variant(profile)
    for profile in profiles:
        for skill in skills:
            tasks, _ = SkillCalibrationWorkflow._tasks_for_skill(skill, selection)
            if tasks == ():
                continue
            applicable = tuple(tasks or ())
            key = fingerprint([model_variant_id(profile), skill_version_id(skill)])
            if key in cached:
                row = cached[key]
            else:
                row = {"model_id": profile.model_id, "model_variant_id": model_variant_id(profile),
                       "skill_id": skill.skill_id, "skill_version_id": skill_version_id(skill),
                       "method": options.scorer, "source": "estimated", "run_id": run_id,
                       "model_weight_fingerprint": fingerprint(identities[profile.model_id]),
                       **_prior(skill, profile, evidence[profile.model_id], applicable),
                       "reference_samples": 0, "likelihood_gain_nats_per_token": None}
                if options.scorer == "teacher-forcing":
                    # 相同技能对所有 adapter 使用同一固定参考子集；不按 adapter 成绩挑选。
                    choices = [item for item in references if applicable and item.get("metadata", {}).get("task_name") in applicable]
                    choices.sort(key=lambda item: fingerprint([options.seed, skill.skill_id, item["reference_id"]]))
                    choices = choices[:options.max_samples_per_skill]
                    gains = []
                    for sample in choices:
                        baseline_key = (model_variant_id(profile), sample["reference_id"])
                        if baseline_key not in baseline_cache:
                            baseline_cache[baseline_key] = float(scorer.score(profile, sample, None))
                        with_skill = float(scorer.score(profile, sample, skill))
                        base = baseline_cache[baseline_key]
                        if not math.isfinite(base) or not math.isfinite(with_skill):
                            raise ValueError("离线 scorer 返回非有限 log probability；拒绝保存虚构分数")
                        gains.append(with_skill - base)
                    if gains:
                        gain = sum(gains) / len(gains)
                        affinity = 1 / (1 + math.exp(-max(-30, min(30, gain))))
                        row["estimated_support_score"] = 0.5 * row["estimated_support_score"] + 0.5 * affinity
                        row.update(reference_samples=len(gains), likelihood_gain_nats_per_token=gain)
                    else:
                        row["requested_method"] = "teacher-forcing"
                        row["method"] = "prior"
                        row["warnings"].append("no_matching_reference_tasks_prior_only")
                cached[key] = row
                write_json_atomic(pair_cache_dir / f"{key}.json", row)
            store.set_skill_model_estimate(skill, profile, row, register_versions=False)
            rows.append(row)
            LOGGER.info("OFFLINE_SKILL_ESTIMATE model=%s skill=%s score=%.4f source=%s",
                        profile.model_id, skill.skill_id, row["estimated_support_score"], row["method"])
    categories = {
        category: {"label": CAPABILITY_LABELS[category], "score_type": "estimated_support_not_sr",
                   "model_scores": {profile.model_id: _category_score(rows, profile.model_id, category) for profile in profiles}}
        for category in CAPABILITY_ORDER
    }
    summary = {"protocol": PROTOCOL, "mode": "offline", "scorer": options.scorer, "run_id": run_id,
               "score_is_success_rate": False, "androidworld_executed": False,
               "model_variant_count": len(profiles), "skill_version_count": len(skills),
               "estimated_pairs": len(rows), "observed_trials_created": 0,
               "reference_rejections": rejected, "capability_estimates": categories,
               "model_skill_estimates": rows}
    write_json_atomic(cache_path, cached)
    write_json_atomic(target / "summary.json", summary)
    columns = ["model_id", "skill_id", "method", "estimated_support_score", "primitive_prior_score",
               "action_training_coverage", "task_training_coverage", "reference_samples", "likelihood_gain_nats_per_token"]
    with (target / "model_skill_estimates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 离线模型—技能适配估计", "", "本报告不包含真实 SR，也不代表技能已通过标定；不自动晋升 active。", "",
             "## 三类适配分数（不是 SR）", "", "| 模型 | Reasoning | Coordination | Manipulation |", "|---|---:|---:|---:|"]
    for profile in profiles:
        values = [categories[c]["model_scores"][profile.model_id] for c in CAPABILITY_ORDER]
        lines.append(f"| {profile.model_id} | " + " | ".join("N/A" if v is None else f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 模型—技能估计", "", "| 模型 | 技能 | 方法 | 适配分数 | 参考样本 | ΔlogP/token |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        gain = row["likelihood_gain_nats_per_token"]
        lines.append(f"| {row['model_id']} | {row['skill_id']} | {row['method']} | {row['estimated_support_score']:.4f} | {row['reference_samples']} | "
                     + ("N/A" if gain is None else f"{gain:+.4f}") + " |")
    lines += ["", "先验为启发式排序分数；缺少模型画像/训练快照时不会编造差异。",
              "teacher-forcing 只衡量已有成功视觉参考响应的拟合增益，不验证自由执行或偏离后的恢复。",
              "训练 episode 与参考 episode 的重叠按所有已找到的 adapter 训练快照的并集排除；参考集仍应与最终测试集隔离。",
              "完整的缺失证据/降级原因见 summary.json 的 warnings。", ""]
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"calibration_mode": "offline", "summary": summary, "summary_json": target / "summary.json",
            "report_markdown": target / "report.md", "model_skill_estimates_csv": target / "model_skill_estimates.csv"}


def _category_score(rows: Sequence[Mapping[str, Any]], model_id: str, category: str) -> float | None:
    values = [float(row["estimated_support_score"]) for row in rows if row["model_id"] == model_id and category in row["categories"]]
    return sum(values) / len(values) if values else None
