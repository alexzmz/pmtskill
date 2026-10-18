"""Read both released (flat) and registry-based SkVM compilation plans.

Normalization is in memory only. In particular, a legacy plan has no pass-run
telemetry: do not invent successful passRuns to make validation pass.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping


def normalize_compilation_plan(plan):
    if not isinstance(plan, Mapping):
        return {}
    result = copy.deepcopy(dict(plan))
    if "artifacts" in result or "passRuns" in result:
        return result  # Never disguise an incomplete modern plan as legacy.
    if not isinstance(result.get("scr"), Mapping):
        return result
    artifacts = {"scr": result["scr"], "gaps": result.get("gaps", [])}
    pass3 = result.get("pass3")
    if isinstance(pass3, Mapping) and isinstance(pass3.get("dag"), Mapping):
        artifacts["dag"] = pass3["dag"]
    result["artifacts"] = artifacts
    result["_plan_schema"] = "legacy-flat-v1"
    return result


def annotation_errors(plan, *, skill_id, model, harness, passes=(1,)):
    """Validate evidence, not merely the SKILL.md pre-copied by the compiler."""
    errors = []
    artifacts = plan.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    scr = artifacts.get("scr")
    purposes = scr.get("purposes") if isinstance(scr, Mapping) else None
    if 1 in passes and (not isinstance(purposes, list) or not purposes
                        or any(not isinstance(p, Mapping) or not p.get("id") for p in purposes)):
        errors.append("no valid SCR purposes")
    for field, expected in (("skillName", skill_id), ("model", model), ("harness", harness)):
        if field in plan and plan[field] != expected:
            errors.append(f"{field} mismatch")
    if plan.get("guardPassed") is not True:
        errors.append("guard not passed")
    if plan.get("_plan_schema") == "legacy-flat-v1":
        # Flat plans are written only after the compiler finishes. A copied
        # SKILL.md, an empty plan, or an arbitrary SCR alone is not completion.
        for field in ("skillName", "model", "harness", "compiledAt"):
            if not isinstance(plan.get(field), str) or not plan[field].strip():
                errors.append(f"legacy plan missing {field}")
        if not isinstance(plan.get("gaps"), list):
            errors.append("legacy plan missing gaps")
        if 3 in passes:
            dag = artifacts.get("dag")
            if not isinstance(dag, Mapping) or not all(isinstance(dag.get(k), list) for k in ("steps", "parallelism")):
                errors.append("legacy plan missing pass3 DAG")
    else:
        runs = plan.get("passRuns")
        for number in passes:
            name = {1: "rewrite-skill", 2: "bind-env", 3: "extract-parallelism"}[number]
            run = runs.get(name) if isinstance(runs, Mapping) else None
            if not isinstance(run, Mapping) or run.get("status") != "ok":
                errors.append(f"{name} pass not successful")
    return errors


def referenced_primitives(plan):
    """Primitives whose corrected TCP entries invalidate this cached variant."""
    artifacts = plan.get("artifacts") or {}
    result = {gap.get("primitiveId") for gap in artifacts.get("gaps", [])
              if isinstance(gap, Mapping)}
    scr = artifacts.get("scr") or {}
    for purpose in scr.get("purposes", []):
        if not isinstance(purpose, Mapping):
            continue
        paths = [purpose.get("currentPath", {}), *purpose.get("alternativePaths", [])]
        for path in paths:
            if isinstance(path, Mapping):
                result.update(p.get("id") for p in path.get("primitives", []) if isinstance(p, Mapping))
    return result - {None}
