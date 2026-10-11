"""生成库的离线完整性/规划/短提示检查；不是 AndroidWorld 成功率评测。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from ..core.config import RoutingConfig
from ..core.io import write_json_atomic
from ..core.models import ModelProfile, SkillStatus
from ..inference.model_pool import ModelPool
from ..online.executor import RoutedVLWrapper
from ..online.planner import KeywordSkillPlanner, PlannerPipeline, PrimitiveTopologyGenerator, scope_hierarchical_skills
from ..online.router import DynamicProgrammingRouter
from .evidence_miner import COMPILER_ID
from .store import SkillStore


def _payloads(path: Path, table: str, key: str, payload: str) -> dict[str, str]:
    wal = Path(str(path.resolve()) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"关闭数据库写入并合并 WAL 后再做静态审计: {path}")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        return dict(db.execute(f"SELECT {key}, {payload} FROM {table}"))


def audit_library(source: Path, library: Path) -> dict[str, Any]:
    source_skills = _payloads(source, "skills", "skill_id", "record_json")
    source_traces = _payloads(source, "traces", "trace_id", "trace_json")
    target_skills = _payloads(library, "skills", "skill_id", "record_json")
    target_traces = _payloads(library, "traces", "trace_id", "trace_json")
    errors = []
    raw_ids = [key for key, value in source_skills.items() if json.loads(value)["kind"] == "raw"]
    raw_preserved = all(target_skills.get(key) == source_skills[key] for key in raw_ids)
    traces_preserved = all(target_traces.get(key) == value for key, value in source_traces.items())
    if not raw_preserved or not traces_preserved:
        errors.append("source raw/trace payload changed or missing")
    with closing(sqlite3.connect(library)) as db:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_errors = db.execute("PRAGMA foreign_key_check").fetchall()
    if integrity != "ok" or foreign_key_errors:
        errors.append("SQLite integrity/foreign keys failed")
    store = SkillStore(library)
    skills = [s for s in store.list_skills(kind="polished") if s.metadata.get("compiler") == COMPILER_ID]
    by_id = {s.skill_id: s for s in skills}
    def flatten(skill_id: str, seen: set[str]) -> tuple[str, ...]:
        if skill_id in seen:
            raise ValueError(f"hierarchy cycle: {skill_id}")
        skill = by_id[skill_id]
        components = skill.metadata["components"]
        if not components:
            return skill.topology.primitive_sequence()
        sequence = []
        offset = 0
        for child in components:
            if child["start"] != offset or child["end"] <= offset:
                raise ValueError(f"invalid component span: {skill_id}")
            child_skill = by_id[child["skill_id"]]
            if child["end"] - child["start"] != child_skill.metadata["operation_count"]:
                raise ValueError(f"incorrect child operation span: {skill_id}")
            sequence.extend(flatten(child["skill_id"], seen | {skill_id}))
            offset = child["end"]
        if offset != skill.metadata["operation_count"]:
            raise ValueError(f"incomplete components: {skill_id}")
        return tuple(sequence)
    for skill in skills:
        try:
            if flatten(skill.skill_id, set()) != skill.topology.primitive_sequence():
                errors.append("hierarchy flatten mismatch: " + skill.skill_id)
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
    prior_edges = [r for r in store.list_skill_relations() if r["relation_type"] == "related_raw"]
    if any(r["evidence_count"] != 0 for r in prior_edges):
        errors.append("raw taxonomy incorrectly counted as execution evidence")
    metrics = [store.skill_metrics(s.skill_id)["trials"] for s in skills]
    pipeline = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator())
    profile = ModelProfile("audit-only-student", "audit-only-student", "http://unused/v1", {})
    router = DynamicProgrammingRouter(RoutingConfig(), store)
    traces_by_id = {t.trace_id: t for t in store.list_traces()}
    retrieval_cases = []
    route_count = 0
    max_guidance = 0
    class FakeClient:
        def predict_mm(self, prompt, images):
            return "Reason: audit only\nAction: {}", True, {}
    pool = ModelPool([profile], client_factory=lambda _: FakeClient())
    for root in (s for s in skills if s.level == 3 and s.status == SkillStatus.ACTIVE):
        goals = sorted({traces_by_id[i].goal for i in root.metadata["source_trace_ids"] if traces_by_id[i].successful})
        plan = None
        for goal in goals:
            decomposition, topology = pipeline.plan(goal, [], skills)
            selected = by_id.get(decomposition.task_skill_id or "")
            retrieval_cases.append({"expected_task": root.metadata["task_names"][0], "selected_skill_id": decomposition.task_skill_id,
                                    "selected_tasks": selected.metadata["task_names"] if selected else [],
                                    "exact_task_match": bool(selected and root.metadata["task_names"][0] in selected.metadata["task_names"])})
            if not selected:
                errors.append("active task goal not retrievable: " + root.skill_id)
            if selected and plan is None:
                scoped = scope_hierarchical_skills(skills, selected.skill_id)
                plan = router.route(goal, topology, [profile], scoped)
                plan.metadata["task_skill_id"] = selected.skill_id
                if not any(step.is_polished for step in plan.steps):
                    errors.append("selected task has no polished route: " + root.skill_id)
                route_count += 1
                wrapper = RoutedVLWrapper(pool, skills)
                wrapper.set_plan(plan)
                for index in (0, selected.metadata["operation_count"] - 1):
                    wrapper.action_call_index = index
                    wrapper.predict_mm("AndroidWorld audit prompt", [])
                    chars = wrapper.call_log[-1]["guidance_chars"]
                    max_guidance = max(max_guidance, chars)
                    if chars > selected.metadata["prompt_max_chars"]:
                        errors.append("stepwise guidance exceeds configured budget")
    return {
        "passed": not errors, "errors": errors,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "sqlite_quick_check": integrity, "foreign_key_error_count": len(foreign_key_errors),
        "raw_records_preserved_exactly": raw_preserved, "raw_count": len(raw_ids),
        "trace_payloads_preserved_exactly": traces_preserved, "trace_count": len(source_traces),
        "hierarchy_skill_count_checked": len(skills), "zero_execution_evidence_raw_prior_edges": len(prior_edges),
        "new_skill_online_trials": sum(metrics), "skill_model_status_rows": len(store.list_skill_model_statuses()),
        "retrieval_case_count": len(retrieval_cases),
        "retrieved_case_count": sum(bool(r["selected_skill_id"]) for r in retrieval_cases),
        "exact_task_match_count": sum(r["exact_task_match"] for r in retrieval_cases),
        "retrieval_aliases_or_mismatches": [r for r in retrieval_cases if not r["exact_task_match"]],
        "router_smoke_checks": route_count, "maximum_injected_guidance_chars": max_guidance,
        "status_counts": dict(Counter(f"L{s.level}:{s.status.value}" for s in skills)),
        "real_androidworld_sr_measured": False,
        "scope": "offline structure and fake-client prompt smoke checks; no emulator, no model execution, no SR claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_library(args.source, args.library)
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
