"""从成功示范构建 L1 操作 / L2 子流程 / L3 任务技能，不伪造标定结果。

L0 原语仍保存在 primitive_catalog。技能 ID 包含语义指令，而不只包含原语序列：
点击 Save 与点击 Delete 即使拓扑相同也不能合并成同一个技能。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from ..core.config import MaintenanceConfig
from ..core.io import load_primitives, write_json_atomic, write_jsonl
from ..core.models import ExecutionTrace, SkillRecord, SkillStatus, SkillTopology, TraceEvent
from .identity import skill_definition_hash
from .store import SkillStore, classify_trajectory_quality


COMPILER_ID = "successful-trajectory-hierarchy-v1"


def _action(event: TraceEvent) -> dict[str, Any]:
    match = re.search(r"Action:\s*(\{.*\})", event.action or "", re.DOTALL)
    try:
        value = json.loads(match.group(1) if match else (event.action or "{}"))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _values(trace: ExecutionTrace) -> list[str]:
    # 只收集任务输入，不将 observation / teacher 给出的答案写成可重放的常量。
    values = set(re.findall(r"['\"]([^'\"]+)['\"]", trace.goal))
    for event in trace.events:
        action = _action(event)
        if action.get("action_type") in {"input_text", "type", "type_text"}:
            if isinstance(action.get("text"), str):
                values.add(action["text"])
    return sorted((value for value in values if len(value.strip()) >= 2), key=lambda s: (-len(s), s))


def parameterize(text: str, values: Sequence[str] = ()) -> str:
    """删除示范实例的输入值；在线以当前 goal 的参数为准，不回放坐标/索引。"""

    for value in values:
        text = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", "<goal_value>", text, flags=re.I)
    text = re.sub(r"\b\d+(?:[-:/.,h]\d+)*\b", "<goal_value>", text)
    return re.sub(r"\s+", " ", text).strip()


def _events(trace: ExecutionTrace, known: set[str]) -> list[tuple[tuple[str, ...], str]]:
    result = []
    values = _values(trace)
    for event in trace.events:
        if event.metadata.get("action_committed", True) is False:
            continue
        primitives = tuple(event.primitive_ids)
        if not primitives or any(p not in known for p in primitives):
            continue
        if not any(p.startswith("action.") or p == "control.finish" for p in primitives):
            continue
        unit = event.metadata.get("planned_unit")
        instruction = str(unit.get("instruction") or "") if isinstance(unit, dict) else ""
        if not instruction:
            action = _action(event)
            if action.get("action_type") == "open_app" and action.get("app_name"):
                instruction = f"Open {action['app_name']}."
            else:
                instruction = "Inspect the current screen and perform " + ", ".join(primitives) + " for the current goal."
        result.append((primitives, parameterize(instruction, values)))
    return result


def _body(instructions: list[str], role: str) -> str:
    header = (
        f"Successful AndroidWorld {role} workflow (teacher evidence, not student calibration).\n"
        "Use the CURRENT goal for all <goal_value> fields, names, dates, quantities and answers. "
        "Never replay demonstration text, coordinates or UI indices. Inspect the current screen, "
        "skip already-completed steps, and adapt to dialogs. Perform ONE AndroidWorld action per call; "
        "retain Reason/Action format. Verify the current goal before declaring success.\n"
    )
    # 在线 wrapper 注入上限为 3000 字；保留每一步，避免长流程尾部被截断。
    budget = max(24, (2850 - len(header) - 7 * len(instructions)) // max(1, len(instructions)))
    return header + "\n".join(f"{i}. {instruction[:budget]}" for i, instruction in enumerate(instructions, 1))


def build_hierarchical_library(store: SkillStore, config: MaintenanceConfig) -> tuple[list[str], dict[str, Any]]:
    traces = store.list_traces()
    known = {p.primitive_id for p in load_primitives()}
    records: dict[str, SkillRecord] = {}
    evidence: defaultdict[str, set[str]] = defaultdict(set)
    edges: defaultdict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"traces": set(), "spans": set()})
    eligible: list[ExecutionTrace] = []
    skipped: list[str] = []
    policy = config.admission_policy

    def add(role: str, units: list[tuple[tuple[str, ...], str]], trace: ExecutionTrace,
            children: list[dict[str, Any]] | None = None) -> str:
        payload = {"role": role, "units": units, "task": trace.task_name if role == "task" else None}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        skill_id = f"polished:hierarchy:{role}:{digest}:v1"
        sequence = tuple(p for primitives, _ in units for p in primitives)
        goal_template = parameterize(trace.goal, _values(trace))
        if skill_id not in records:
            level = {"operation": 1, "subflow": 2, "task": 3}[role]
            instructions = [instruction for _, instruction in units]
            records[skill_id] = SkillRecord(
                skill_id=skill_id,
                name=f"{trace.task_name} workflow" if role == "task" else f"L{level} {instructions[0][:80]}",
                description=f"Successful {role}: " + (goal_template if role == "task" else " / ".join(instructions))[:400],
                kind="polished", status=SkillStatus.CANDIDATE, level=level,
                topology=SkillTopology.from_sequence(sequence, topology_id=skill_id),
                fallback_topology=SkillTopology.from_sequence(sequence),
                body=_body(instructions, role),
                metadata={
                    "compiler": COMPILER_ID, "hierarchy_role": role,
                    "operation_count": len(units), "instructions": instructions,
                    "operation_primitive_lengths": [len(primitives) for primitives, _ in units],
                    "components": children or [], "leaf_primitive_ids": list(sequence),
                    "task_names": [], "calibration_tasks": [], "goal_templates": [],
                    "source_trace_ids": [], "source_skill_ids": [], "teacher_model_ids": [],
                    "admission_policy": policy, "student_calibrated": False,
                    "evidence_scope": "successful_episode_not_independent_component_trial",
                },
            )
        skill = records[skill_id]
        evidence[skill_id].add(trace.trace_id)
        for key, items in (
            ("task_names", [trace.task_name]), ("calibration_tasks", [trace.task_name]),
            ("goal_templates", [goal_template] if role == "task" else []),
            ("source_trace_ids", [trace.trace_id]),
            ("teacher_model_ids", sorted({event.model_id for event in trace.events})),
        ):
            skill.metadata[key] = sorted(set(skill.metadata[key]) | set(items))
        source = trace.metadata.get("collection_source", {})
        source_id = source.get("skill_id") if isinstance(source, dict) else None
        if source_id and store.get_skill(str(source_id)):
            skill.metadata["source_skill_ids"] = sorted(set(skill.metadata["source_skill_ids"]) | {str(source_id)})
            edge = edges[(skill_id, str(source_id), "derived_from_input_skill")]
            edge["traces"].add(trace.trace_id)
        for child in children or []:
            edge = edges[(skill_id, child["skill_id"], "composes_polished")]
            edge["traces"].add(trace.trace_id)
            edge["spans"].add((child["start"], child["end"]))
        return skill_id

    for trace in sorted(traces, key=lambda t: t.trace_id):
        quality = classify_trajectory_quality(trace)
        if not trace.successful or quality["status"] == "rejected":
            continue
        if policy == "calibrated" and quality["status"] != "excellent":
            continue
        units = _events(trace, known)
        # 不悄悄用一条缺失/未知原语的路径作为完整任务成功示范。
        executable = [event for event in trace.events if event.metadata.get("action_committed", True) is not False
                      and any(p.startswith("action.") or p == "control.finish" for p in event.primitive_ids)]
        if not units or len(units) != len(executable):
            skipped.append(trace.trace_id)
            continue
        eligible.append(trace)
        operation_ids = [add("operation", [unit], trace) for unit in units]
        blocks: list[dict[str, Any]] = []
        for start in range(0, len(units), config.hierarchy_block_size):
            end = min(len(units), start + config.hierarchy_block_size)
            children = [{"skill_id": operation_ids[i], "start": i - start, "end": i - start + 1} for i in range(start, end)]
            child_id = add("subflow", units[start:end], trace, children) if end - start > 1 else operation_ids[start]
            blocks.append({"skill_id": child_id, "start": start, "end": end})
        add("task", units, trace, blocks)

    threshold = 1 if policy == "success_once" else config.minimum_support
    admitted = {skill_id for skill_id in records if len(evidence[skill_id]) >= threshold}
    created: list[str] = []
    for skill_id in sorted(admitted):
        skill = records[skill_id]
        skill.metadata["support"] = len(evidence[skill_id])
        skill.metadata["components"] = [c for c in skill.metadata["components"] if c["skill_id"] in admitted]
        existing = store.get_skill(skill_id)
        if existing:
            skill.created_at = existing.created_at
            skill.status = existing.status
            skill.version = existing.version + int(skill_definition_hash(existing) != skill_definition_hash(skill))
        # 已经配对标定过（包括 deprecated）的技能不被覆盖优先策略强行重新激活。
        if policy == "success_once" and not store.has_any_skill_model_status(skill_id):
            skill.status = SkillStatus.ACTIVE
        if (not existing or skill_definition_hash(existing) != skill_definition_hash(skill)
                or existing.metadata != skill.metadata or existing.status != skill.status):
            if store.upsert_skill(skill):
                created.append(skill_id)
    existing_edges = {(row["parent_skill_id"], row["child_skill_id"], row["relation_type"]): row
                      for row in store.list_skill_relations()}
    for (parent, child, relation), detail in sorted(edges.items()):
        if parent not in admitted or (relation == "composes_polished" and child not in admitted):
            continue
        if relation == "derived_from_input_skill" and store.get_skill(child).kind == "raw":
            relation = "derived_from_raw"
        edge_detail = {"compiler": COMPILER_ID, "operation_spans": [list(span) for span in sorted(detail["spans"])],
                       "evidence_scope": "successful_episode_composition"}
        previous = existing_edges.get((parent, child, relation))
        if previous and previous["evidence_count"] == len(detail["traces"]) and previous["detail"] == edge_detail:
            continue
        store.upsert_skill_relation(parent, child, relation, evidence_count=len(detail["traces"]), detail=edge_detail)

    skills = [records[skill_id] for skill_id in admitted]
    observed = sorted({trace.task_name for trace in traces})
    covered = sorted({task for skill in skills if skill.level == 3 for task in skill.metadata["task_names"]})
    quality_counts = Counter(classify_trajectory_quality(trace)["status"] for trace in traces)
    report = {
        "admission_policy": policy, "student_calibrated": False,
        "traces_total": len(traces), "quality_counts": dict(sorted(quality_counts.items())),
        "successful_traces": sum(trace.successful for trace in traces), "eligible_successful_traces": len(eligible),
        "skipped_successful_trace_ids": skipped, "raw_skills": len(store.list_skills(kind="raw")),
        "primitive_count": len(known), "polished_skills": len(skills),
        "levels": {f"L{level}": sum(skill.level == level for skill in skills) for level in (1, 2, 3)},
        "observed_task_count": len(observed), "covered_task_count": len(covered),
        "coverage_scope": "tasks_present_in_input_database_not_entire_androidworld_registry",
        "covered_tasks": covered, "uncovered_tasks": sorted(set(observed) - set(covered)),
        "warning": "Active means admitted from teacher success, NOT measured student SR or calibrated compatibility.",
    }
    store.log_maintenance_event("hierarchical_library_built", None, report)
    return created, report


def export_hierarchical_library(store: SkillStore, directory: Path, report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    skills = [skill for skill in store.list_skills(kind="polished") if skill.metadata.get("compiler") == COMPILER_ID]
    write_json_atomic(directory / "coverage.json", report)
    write_jsonl(directory / "skills.jsonl", (skill.to_dict() for skill in skills))
    write_json_atomic(directory / "hierarchy.json", {
        "primitives": [primitive.to_dict() for primitive in store.list_primitive_catalog()],
        "skills": [{"skill_id": skill.skill_id, "name": skill.name, "level": skill.level,
                    "components": skill.metadata["components"], "primitive_ids": skill.metadata["leaf_primitive_ids"]} for skill in skills],
        "relations": store.list_skill_relations(),
    })
    rows = ["# AndroidWorld 覆盖优先技能库", "", f"- 数据库：`{store.database.name}`",
        f"- 准入策略：`{report['admission_policy']}`；active 不代表学生模型已经标定。",
        f"- 轨迹：{report['traces_total']}；可用成功示范：{report['eligible_successful_traces']}",
        f"- 成功覆盖：{report['covered_task_count']} / {report['observed_task_count']} 个已采集任务（不是全部 registry）。",
        f"- L0 原语：{report['primitive_count']}；L1/L2/L3：{report['levels']}", "",
        "## 在线测评", "", "将生成的 sqlite3 上传到评测机器，沿用你的模型/adapter 配置：", "", "```bash",
        "python -m src1 --config src1/config.local_all.toml evaluate-pmtskill \\",
        "  --adapter-paths /your/adapter/run --skill-database /your/skill_library.sqlite3 \\",
        "  --combinations 2 --seed 123 --max-steps 25", "```", "",
        "建议用未参与采集的参数/seed 测试；本库含采集任务的成功示范，不应宣称零样本泛化。",
        "原 raw 保留；只有明确的 collection_source.skill_id 才创建 derived_from_raw，不伪造来源关系。", "",
        "## 已覆盖任务", "", "| Task | L3 variants | Successful demonstrations |", "|---|---:|---:|"]
    for task in report["covered_tasks"]:
        variants = [skill for skill in skills if skill.level == 3 and task in skill.metadata["task_names"]]
        trace_ids = {trace_id for skill in variants for trace_id in skill.metadata["source_trace_ids"]}
        rows.append(f"| {task} | {len(variants)} | {len(trace_ids)} |")
    rows.extend(["", "## 缺少成功示范的任务", "", *(f"- {task}" for task in report["uncovered_tasks"])])
    (directory / "coverage.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_skill_database(source: Path, target: Path) -> None:
    """SQLite backup 保留输入库；拒绝覆盖输出和忽略活跃 WAL。"""

    source, target = source.resolve(), target.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"技能库不存在: {source}")
    if source == target or target.exists():
        raise ValueError("输出数据库必须是新的路径，不能覆盖已有数据库")
    uri = source.as_uri() + "?mode=ro"
    try:
        input_db = sqlite3.connect(uri, uri=True)
        input_db.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError:
        # 某些只读附件目录无法创建 WAL 的 -shm。只有无 WAL 的静态副本才可 immutable。
        if "input_db" in locals():
            input_db.close()
        wal = Path(str(source) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("输入库有未合并 WAL；请关闭写入并通过 SQLite backup 导出后重试")
        input_db = sqlite3.connect(uri + "&immutable=1", uri=True)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(target)) as output_db:
            input_db.backup(output_db)
    finally:
        input_db.close()
