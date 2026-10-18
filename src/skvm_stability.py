"""Failure-isolated SKVM preparation for the fixed-adapter baseline.

Bad artifacts never become valid ones. Each skill uses a validated cached AOT,
then bounded compilation attempts, then an explicitly unadapted original.
The resulting selection is frozen before the first AndroidWorld episode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

try:
    from .skvm_artifacts import annotation_errors
except ImportError:  # Direct legacy runner imports src/ modules by basename.
    from skvm_artifacts import annotation_errors


SKVM_STABILITY_POLICY = "valid-aot-or-original-v1"


class SkVMCommandError(RuntimeError):
    def __init__(self, message, *, recoverable=False):
        super().__init__(message)
        self.recoverable = recoverable


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_episodes(root):
    checkpoints = root / "checkpoints"
    return (root / "summary.json").is_file() or (root / "report.json").is_file() or (
        checkpoints.is_dir() and any(checkpoints.rglob("*.json"))
    ) or (checkpoints.is_dir() and any(checkpoints.rglob("*.pkl.gz")))


def prepare_stable(kernel, variant_type, read_json, write_json):
    """Do not swallow configuration, filesystem, provider or profile failures."""
    root = kernel.run_dir / "skvm"
    record_path = root / "preparation.json"
    signature = {"policy": SKVM_STABILITY_POLICY, "target_model": kernel.target_model,
                 "compiler_model": kernel.compiler_model,
                 "passes": [list(p) for p in kernel.pass_sets],
                 "skills": [[s.skill_id, s.sha256] for s in kernel.skills]}
    mode = kernel.args.skvm_prepare
    retries = getattr(kernel.args, "skvm_compile_retries", 1)
    if not 0 <= retries <= 3:
        raise ValueError("--skvm-compile-retries must be between 0 and 3")
    if record_path.is_file() and mode != "force":
        record = read_json(record_path)
        if record.get("signature") != signature:
            raise ValueError("Frozen SkVM preparation does not match this experiment; use a new output-dir")
        variants = []
        skills = {s.skill_id: s for s in kernel.skills}
        for item in record["selections"]:
            skill = skills[item["skill_id"]]
            content = Path(item["path"]).read_text(encoding="utf-8")
            if _hash(content) != item["sha256"]:
                raise ValueError("Frozen SkVM skill snapshot was modified; refusing mixed resume")
            plan = read_json(Path(item["plan_path"])) if item.get("plan_path") else {}
            if _hash_json(plan) != item["plan_sha256"]:
                raise ValueError("Frozen SkVM plan snapshot was modified; refusing mixed resume")
            if item["source"] == "aot":
                errors = annotation_errors(plan, skill_id=skill.skill_id, model=kernel.target_model,
                                           harness="bare-agent", passes=tuple(item["passes"]))
                if errors:
                    raise ValueError("Frozen SkVM AOT is invalid: " + "; ".join(errors))
                variants.append(variant_type(skill, "original", "original", skill.path,
                                              skill.content, skill.sha256))
            elif item["sha256"] != skill.sha256:
                raise ValueError("Frozen original fallback does not match the input skill")
            variants.append(variant_type(skill, item["source"], item["tag"], Path(item["path"]),
                                          content, item["sha256"], plan=plan,
                                          fallback_reason=item.get("fallback_reason")))
        if {i["skill_id"] for i in record["selections"]} != set(skills):
            raise ValueError("Frozen SkVM preparation has incomplete skill coverage")
        snapshot = Path(record["profile_snapshot"])
        if _hash_json(read_json(snapshot)) != record["profile_sha256"]:
            raise ValueError("Frozen SkVM TCP snapshot was modified; refusing mixed resume")
        kernel.preparation_summary = record
        print("SkVM stability: reusing frozen preparation; no recompilation", flush=True)
        return variants
    if _has_episodes(kernel.run_dir):
        raise ValueError("Cannot rebuild SkVM skill choices after episodes exist; use a new output-dir")

    repair_path = kernel.cache_dir / "aot-profile-repair.json"
    if repair_path.is_file():
        kernel._repair_primitives.update(read_json(repair_path).get("pending_primitives", []))
    bad = kernel._profile_context_failures()
    if bad and mode == "reuse":
        raise RuntimeError("Cached TCP contains context/infrastructure errors; use --skvm-prepare auto")
    profile_path = kernel._profile_path()
    profile = read_json(profile_path) if profile_path.is_file() else None
    needs_profile = bad or mode == "force" or not profile or profile.get("isPartial")
    if needs_profile:
        if mode == "reuse":
            raise RuntimeError("SKVM baseline requires a complete TCP; use --skvm-prepare auto")
        kernel._configure_local_route()
        if bad and mode != "force":
            kernel._repair_primitives.update(bad)
            write_json(repair_path, {"pending_primitives": sorted(kernel._repair_primitives)})
            command = kernel._profile_command(sorted(bad), force=True)
        else:
            command = kernel._profile_command(getattr(kernel.args, "skvm_profile_primitives", None),
                                               force=mode == "force")
        kernel._run(command, "baseline model profiling")
        profile = read_json(profile_path)
    if (not isinstance(profile, dict) or not profile.get("capabilities") or profile.get("isPartial")
            or kernel._profile_context_failures()
            or (kernel._bridge and kernel._bridge.context_errors)):
        raise RuntimeError("SkVM TCP is incomplete or context-tainted; evaluation unavailable, not SR=0")

    passes_order = sorted(set(kernel.pass_sets),
                          key=lambda p: (p == (1, 3), p == (1,), len(p)), reverse=True)
    if not passes_order:
        raise ValueError("SKVM requires at least one AOT pass set")
    preferred = passes_order[0]
    chosen = {}
    statuses = {}
    attempts = {}
    command_errors = []

    def inspect(skill, passes):
        plan, errors = kernel._cached_plan(skill, passes)
        if not errors and kernel._variant_needs_profile_repair(skill, passes, plan):
            errors = ["AOT depends on a repaired TCP; stale profile fingerprint"]
        statuses[(skill.skill_id, passes)] = {
            "skill_id": skill.skill_id, "variant": _tag(passes), "errors": errors,
            "guard_passed": plan.get("guardPassed"),
            "guard_violations": plan.get("guardViolations", []),
        }
        return plan, errors

    # A valid alternative in the cache is enough. Never spend another full
    # compilation round just to upgrade a usable p1 into a failing p1p3.
    for skill in kernel.skills:
        for passes in passes_order:
            plan, errors = inspect(skill, passes)
            if not errors and mode != "force":
                chosen[skill.skill_id] = (passes, plan, "cache")
                break
    if mode != "reuse" and len(chosen) < len(kernel.skills):
        kernel._configure_local_route()
        if kernel._bridge:
            kernel._bridge.stage = "compile"
        for passes in passes_order:
            for _ in range(retries + 1):
                pending = [s for s in kernel.skills if s.skill_id not in chosen]
                if not pending:
                    break
                paths = [str(s.path) for s in pending]
                if any("," in p for p in paths):
                    raise ValueError("SkVM multi-skill CLI cannot accept a path containing a comma")
                command = ["aot-compile", f"--skill={','.join(paths)}", f"--model={kernel.target_model}",
                           "--adapter=bare-agent", f"--pass={','.join(map(str, passes))}",
                           f"--compiler-model={kernel.compiler_model}",
                           f"--concurrency={kernel.args.skvm_compile_concurrency}",
                           f"--timeout-ms={kernel.args.skvm_compile_timeout_ms}"]
                for skill in pending:
                    key = (skill.skill_id, passes)
                    attempts[key] = attempts.get(key, 0) + 1
                try:
                    kernel._compile_variants(pending, passes, command)
                except SkVMCommandError as exc:
                    if not exc.recoverable:
                        raise  # Invalid flags, missing providers, auth/outage are not bad skills.
                    command_errors.append(str(exc))
                for skill in pending:
                    plan, errors = inspect(skill, passes)
                    if not errors:
                        chosen[skill.skill_id] = (passes, plan, "compiled")

    variants, selections = [], []
    snapshot_root = root / "prepared-skills" / uuid.uuid4().hex
    for skill in kernel.skills:
        reason = None
        source, tag, passes, plan = "original", "original", (), {}
        content = skill.content
        if skill.skill_id in chosen:
            passes, plan, origin = chosen[skill.skill_id]
            source, tag = "aot", _tag(passes)
            content = (kernel._variant_dir(skill.skill_id, passes) / "SKILL.md").read_text(encoding="utf-8")
            if passes != preferred:
                reason = "preferred " + _tag(preferred) + " unavailable: " + "; ".join(
                    statuses[(skill.skill_id, preferred)]["errors"])
            variants.append(variant_type(skill, "original", "original", skill.path,
                                          skill.content, skill.sha256))
        else:
            origin = "fallback"
            reason = "no validated AOT: " + "; ".join(
                _tag(p) + ": " + "; ".join(inspect(skill, p)[1]) for p in passes_order)
        folder = snapshot_root / skill.skill_id / tag
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "SKILL.md"
        path.write_text(content, encoding="utf-8", newline="\n")
        plan_path = folder / "compilation-plan.json" if source == "aot" else None
        if plan_path:
            write_json(plan_path, plan)
        variant = variant_type(skill, source, tag, path, content, _hash(content), plan=plan,
                               fallback_reason=reason)
        variants.append(variant)
        selections.append({"skill_id": skill.skill_id, "source": source, "tag": tag,
                           "passes": list(passes), "sha256": variant.sha256,
                           "path": str(path), "plan_path": str(plan_path) if plan_path else None,
                           "plan_sha256": _hash_json(plan), "fallback_reason": reason, "origin": origin})
    for key, status in statuses.items():
        status["compile_attempts"] = attempts.get(key, 0)
    with_aot = sum(i["source"] == "aot" for i in selections)
    profile_snapshot = snapshot_root / "tcp.json"
    write_json(profile_snapshot, profile)
    summary = {"policy": SKVM_STABILITY_POLICY, "signature": signature,
               "profile_snapshot": str(profile_snapshot), "profile_sha256": _hash_json(profile),
               "skills_total": len(kernel.skills), "skills_with_aot": with_aot,
               "aot_coverage": with_aot / len(kernel.skills) if kernel.skills else 0,
               "original_fallback_skills": len(kernel.skills) - with_aot,
               "alternate_aot_skills": sum(i["source"] == "aot" and bool(i["fallback_reason"]) for i in selections),
               "compile_attempts": sum(attempts.values()),
               "guard_failed_variants": sum(s["guard_passed"] is False for s in statuses.values()),
               "inspected_variants": len(statuses),
               "validated_variants": sum(not s["errors"] for s in statuses.values()),
               "failed_attempted_variants": sum(bool(s["errors"]) and s["compile_attempts"] > 0 for s in statuses.values()),
               "unavailable_variants": sum(bool(s["errors"]) for s in statuses.values()),
               "command_errors": command_errors, "variants": list(statuses.values()), "selections": selections}
    write_json(record_path, summary)  # Commit choice set only after all snapshots exist.
    kernel.preparation_summary = summary
    if kernel._repair_primitives:
        write_json(repair_path, {"pending_primitives": []})
        kernel._repair_primitives.clear()
    print(f"SkVM stability: {with_aot}/{len(kernel.skills)} skills use AOT; "
          f"{summary['alternate_aot_skills']} alternatives, {summary['original_fallback_skills']} original fallbacks", flush=True)
    return variants


def _tag(passes):
    return "".join("p" + str(p) for p in passes)


def _hash_json(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
