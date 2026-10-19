"""Paired raw / flat-primitive / polished representation ablation for PORTS.

No model selection, retrieval, maintenance, or model-skill calibration is run.
Raw/polished equivalence is an explicit experimental annotation, not inferred
from a shared app/category or a taxonomy-only related_raw edge.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import sqlite3
import statistics
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.io import write_json_atomic, write_jsonl
from ..core.models import ModelProfile, SkillRecord
from ..skills.identity import skill_definition_hash
from ..skills.store import SkillStore
from .resume import fingerprint, model_resume_identity

PROTOCOL = "ports-skill-effectiveness-v2"
CONDITIONS = ("raw", "primitives", "polished")


def validate_experiment_protocol(target: Path) -> None:
    """Fail before GPU deployment; v1 prompts must not mix with v2 app hints."""
    manifest = target / "experiment_manifest.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))["specification"].get("protocol")
        if previous != PROTOCOL:
            raise ValueError(f"技能有效性实验协议已升级为 {PROTOCOL}（旧目录：{previous}）；"
                             "三组新增相同应用名提示，不能混合旧结果。请保留旧目录并使用新的 --output-dir，例如 skill_effectiveness_control_v2")


def slug(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass
class SkillPair:
    pair_id: str
    raw: SkillRecord
    polished: SkillRecord
    tasks: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, "raw_skill_id": self.raw.skill_id,
                "polished_skill_id": self.polished.skill_id, "tasks": list(self.tasks),
                "raw_definition": skill_definition_hash(self.raw),
                "polished_definition": skill_definition_hash(self.polished),
                "raw_metadata": fingerprint(self.raw.metadata),
                "polished_metadata": fingerprint(self.polished.metadata),
                "primitive_topology": [n.to_dict() for n in self.polished.topology.topological_nodes()]}


def load_pairs(store: SkillStore, path: Path, tasks: Sequence[str] | None = None) -> list[SkillPair]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = value.get("pairs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("pairs-file 必须包含非空 pairs 数组；先运行 prepare-skill-effectiveness")
    pairs, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("每个 pair 必须是 JSON object")
        if row.get("enabled", True) is False:
            continue
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in seen:
            raise ValueError("pair_id 必须非空且唯一")
        seen.add(pair_id)
        if row.get("equivalence_confirmed") is not True:
            raise ValueError(f"{pair_id}: 请人工确认 raw/polished 目标等价后设置 equivalence_confirmed=true")
        raw = store.get_skill(str(row.get("raw_skill_id", "")))
        polished = store.get_skill(str(row.get("polished_skill_id", "")))
        if not raw or raw.kind != "raw" or not polished or polished.kind != "polished":
            raise ValueError(f"{pair_id}: 必须明确指定存在的 raw_skill_id 和 polished_skill_id")
        names = row.get("tasks")
        if not isinstance(names, list) or not names or any(not isinstance(t, str) or not t.strip() for t in names):
            raise ValueError(f"{pair_id}: tasks 必须是非空 AndroidWorld 任务名数组")
        names = tuple(dict.fromkeys(names))
        if tasks is not None:
            names = tuple(t for t in names if t in tasks)
        if not names:
            continue
        nodes = polished.topology.topological_nodes()
        if not nodes:
            raise ValueError(f"{pair_id}: polished skill 没有原语实现")
        previous = ()
        for node in nodes:
            if node.condition or node.depends_on != previous:
                raise ValueError(f"{pair_id}: 当前 primitive-sequence 实验仅支持线性、无条件拓扑，不能静默丢弃分支")
            previous = (node.node_id,)
        pairs.append(SkillPair(pair_id, raw, polished, names))
    if not pairs:
        raise ValueError("没有启用且匹配 tasks 的技能配对")
    return pairs


def prepare_inputs(source: Path, target: Path) -> dict[str, Any]:
    """Freeze a SQLite backup, and export a deliberately unapproved pair template."""
    source, target = source.resolve(), target.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"技能库不存在: {source}")
    inputs = target / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    snapshot = inputs / "skill_library.sqlite3"
    if not snapshot.exists():
        try:
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(snapshot)) as dst:
                src.backup(dst)
        except BaseException:
            # A partial backup must not be mistaken for a complete snapshot.
            snapshot.unlink(missing_ok=True)
            raise
    store = SkillStore(snapshot)
    store.initialize()
    raw = store.list_skills(kind="raw")
    polished = store.list_skills(kind="polished")
    catalog = inputs / "skills_catalog.json"
    if not catalog.exists():
        write_json_atomic(catalog, {"raw": [s.to_dict() for s in raw],
                                    "polished": [s.to_dict() for s in polished]})
    template = inputs / "pairs.example.json"
    if not template.exists():
        rows = []
        for skill in sorted(polished, key=lambda s: s.skill_id):
            if skill.level != 3 or skill.status.value != "active":
                continue
            direct = [sid for sid in skill.metadata.get("source_skill_ids", [])
                      if any(s.skill_id == sid for s in raw)]
            related = [sid for sid in skill.metadata.get("related_raw_skill_ids", [])
                       if any(s.skill_id == sid for s in raw)]
            tasks = skill.metadata.get("calibration_tasks") or skill.metadata.get("task_names") or []
            rows.append({"pair_id": skill.skill_id, "enabled": False,
                         "equivalence_confirmed": False, "raw_skill_id": direct[0] if len(direct) == 1 else "",
                         "polished_skill_id": skill.skill_id, "tasks": tasks,
                         "provenance_raw_candidates": direct, "taxonomy_only_raw_candidates": related})
        write_json_atomic(template, {"protocol": PROTOCOL,
                                    "instructions": "复制为 pairs.json；选择目标等价的 raw/polished，填写任务，人工确认后启用。分类关联不是等价证明。",
                                    "pairs": rows})
    return {"skill_database": str(snapshot), "catalog": str(catalog), "pairs_template": str(template),
            "raw_count": len(raw), "polished_count": len(polished),
            "note": "已有快照/模板保留不覆盖；输入库的技能状态与标定数据不会被此实验修改"}


def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def write_report(target: Path, pairs: Sequence[SkillPair], rows: list[dict[str, Any]],
                 *, model_id: str, expected: int) -> dict[str, Any]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["pair_id"], row["task"], row["instance_id"])
        if row["condition"] in grouped.setdefault(key, {}):
            raise ValueError(f"重复的条件结果: {key}, {row['condition']}")
        grouped[key][row["condition"]] = row
    matched = [items for items in grouped.values()
               if all(c in items and items[c].get("valid") is True for c in CONDITIONS)]
    for items in matched:
        if len({r.get("suite_hash") for r in items.values()}) != 1 or len({r.get("goal") for r in items.values()}) != 1:
            raise ValueError("三组任务参数/goal 不一致，不能形成等价配对")
    successful = [items for items in matched if all(items[c]["success"] for c in CONDITIONS)]

    def aggregate(groups):
        result = []
        for condition in CONDITIONS:
            values = [items[condition] for items in groups]
            result.append({"condition": condition, "episodes": len(values),
                           "success_rate": _mean(values, "success"),
                           "interaction_length": _mean(values, "interaction_length"),
                           "model_invocations": _mean(values, "model_invocations"),
                           "action_calls": _mean(values, "action_calls"),
                           "summary_calls": _mean(values, "summary_calls"),
                           "latency_seconds": _mean(values, "latency_seconds"),
                           "model_latency_seconds": _mean(values, "model_latency_seconds")})
        return result

    table = aggregate(matched)
    summary = {"protocol": PROTOCOL, "model_id": model_id, "expected_triplets": expected,
               "valid_matched_triplets": len(matched), "evaluation_available": bool(matched),
               "complete": len(matched) == expected, "table": table,
               "all_three_successful_table": aggregate(successful),
               "excluded_triplets": len(grouped) - len(matched),
               "unobserved_triplets": expected - len(grouped),
               "per_pair": {pair.pair_id: aggregate([items for items in matched
                                                       if items["raw"]["pair_id"] == pair.pair_id]) for pair in pairs},
               "metric_scope": "matched valid whole-task episodes; not isolated fragment SR; fixed single model",
               "latency_scope": "sum of M3A step wall times incl. GUI actions and model calls; excludes deployment/task initialization",
               "invocation_scope": "attempted HTTP requests, including retries; action and summary calls also reported separately"}
    write_json_atomic(target / "summary.json", summary)
    write_jsonl(target / "trials.jsonl", rows)
    for name, values in (("comparison.csv", table), ("trials.csv", rows)):
        with (target / name).open("w", encoding="utf-8-sig", newline="") as handle:
            fields = list(dict.fromkeys(k for row in values for k in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)

    def number(value):
        return "N/A" if value is None else f"{value:.2f}"

    lines = ["# Polished Skill Effectiveness (PORTS fixed-model ablation)", "",
             f"- Model: `{model_id}`; valid matched triplets: {len(matched)}/{expected}.",
             "- 三组共享任务参数；仅比较完整、有效的三元配对。异常不作为 SR=0。",
             "- Interaction length = M3A 决策步数；Model invocations = HTTP 请求尝试数（含总结和重试）。",
             "- Latency = M3A step 执行墙钟时间，含模型/设备交互，不含部署和任务初始化。",
             "- 固定模型/技能表示，不进行自动检索、模型切换或标定更新；不是完整 PORTS 路由实验。", "",
             "| Representation | Episodes | SR (%) | Interaction length | Model invocations | Latency (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Representation & SR (\%) & Interactions & Model calls & Latency (s) \\", r"\midrule"]
    for row in table:
        sr = "N/A" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}"
        values = [sr, number(row["interaction_length"]), number(row["model_invocations"]), number(row["latency_seconds"])]
        lines.append(f"| {row['condition']} | {row['episodes']} | {sr} | " + " | ".join(values[1:]) + " |")
        tex.append(row["condition"] + " & " + " & ".join(values) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    lines += ["", "另见 summary.json 中 all_three_successful_table（相同且三组均成功的实例上的效率对比）。",
              "trials.jsonl/csv 保留异常、未配对和逐实例数值；不将宏技能注入次数误作模型调用节省。"]
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (target / "table_skill_effectiveness.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return summary


def run_experiment(config, store: SkillStore, profile: ModelProfile, pairs: Sequence[SkillPair],
                   target: Path, *, combinations=1, seed=42, max_steps=30, family="android_world",
                   resume=False, store_trajectories=False, guidance_max_chars=12000, runner=None) -> dict[str, Any]:
    if min(combinations, max_steps, guidance_max_chars) <= 0:
        raise ValueError("combinations / max_steps / guidance_max_chars 必须为正数")
    if not pairs:
        raise ValueError("至少需要一个明确等价的技能配对")
    target = target.resolve()
    validate_experiment_protocol(target)
    specification = {"protocol": PROTOCOL, "model": model_resume_identity(profile),
                     "pairs": [p.identity() for p in pairs], "combinations": combinations,
                     "seed": seed, "max_steps": max_steps, "family": family,
                     "android_world": dataclasses.asdict(config.android_world),
                     "harness_root": str(config.paths.android_world_root.resolve()),
                     "guidance_max_chars": guidance_max_chars,
                     "store_trajectories": store_trajectories}
    specification = json.loads(json.dumps(specification, default=str))
    manifest = target / "experiment_manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding="utf-8"))["specification"] != specification:
            raise ValueError("技能有效性实验配置发生变化或未指定 --resume；请使用新的 output-dir")
    elif any((target / name).exists() for name in ("conditions", "suites", "summary.json")):
        raise ValueError("已有实验结果缺少 experiment_manifest.json，请使用新的 output-dir")
    target.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest, {"specification": specification})
    if runner is None:
        from .skill_effectiveness_runner import AndroidWorldEffectivenessRunner
        runner = AndroidWorldEffectivenessRunner(config, profile, store, target,
                                               guidance_max_chars=guidance_max_chars)
    rows = []
    expected = sum(len(pair.tasks) * combinations for pair in pairs)
    try:
        for index, pair in enumerate(pairs):
            # Rotate condition order across pairs; never pick it based on outcomes.
            order = CONDITIONS[(seed + index) % 3:] + CONDITIONS[:(seed + index) % 3]
            for condition in order:
                values = runner.run(pair, condition, combinations=combinations, seed=seed,
                                    max_steps=max_steps, family=family, resume=resume,
                                    store_trajectories=store_trajectories)
                rows.extend(values)
                write_report(target, pairs, rows, model_id=profile.model_id, expected=expected)
    finally:
        summary = write_report(target, pairs, rows, model_id=profile.model_id, expected=expected)
    if not summary["complete"]:
        raise RuntimeError("技能有效性实验不完整；请检查 report.md / trials.jsonl；异常结果不计为真实失败，可 --resume 重试")
    return {"summary": summary, "summary_json": str(target / "summary.json"),
            "report_markdown": str(target / "report.md"), "comparison_csv": str(target / "comparison.csv"),
            "latex_table": str(target / "table_skill_effectiveness.tex")}
