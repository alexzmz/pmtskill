"""Self-contained SkVM evaluation artifacts for the Android World runner."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


LEVEL_VALUES = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _safe_model_name(model: str) -> str:
    return model.replace("/", "--").replace(":", "_")


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    ).strip("-._") or "artifact"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _primitive_category(primitive_id: str) -> str:
    prefix = primitive_id.split(".", 1)[0]
    return {
        "gen": "generation",
        "reason": "reasoning",
        "tool": "tool_use",
        "follow": "instruction_following",
    }.get(prefix, "other")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _profile_path(
    cache_dir: Path, adapter: str, target_model: str
) -> Path:
    return (
        cache_dir
        / "profiles"
        / adapter
        / _safe_model_name(target_model)
        / "latest.json"
    )


def _capability_evaluation(
    tcp: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tcp is None:
        return (
            {
                "status": "missing",
                "capability_count": 0,
                "mean_level_value": None,
                "normalized_capability_score": None,
                "level_distribution": {},
                "by_category": [],
            },
            [],
        )

    details = {
        str(item.get("primitiveId")): item
        for item in tcp.get("details") or []
        if isinstance(item, Mapping) and item.get("primitiveId")
    }
    rows: list[dict[str, Any]] = []
    distribution = {level: 0 for level in LEVEL_VALUES}
    category_values: dict[str, list[float]] = {}
    total_passed = 0
    total_attempts = 0

    for primitive_id, level_value_raw in sorted(
        (tcp.get("capabilities") or {}).items()
    ):
        level = str(level_value_raw)
        numeric = LEVEL_VALUES.get(level, 0)
        distribution[level if level in distribution else "L0"] += 1
        category = _primitive_category(str(primitive_id))
        category_values.setdefault(category, []).append(float(numeric))
        detail = details.get(str(primitive_id), {})
        level_results = detail.get("levelResults") or []
        passed = sum(
            int(item.get("passCount") or 0)
            for item in level_results
            if isinstance(item, Mapping)
        )
        attempts = sum(
            int(item.get("totalCount") or 0)
            for item in level_results
            if isinstance(item, Mapping)
        )
        total_passed += passed
        total_attempts += attempts
        rows.append(
            {
                "primitive_id": primitive_id,
                "category": category,
                "level": level,
                "level_value": numeric,
                "normalized_value": round(numeric / 3, 4),
                "profile_passed_instances": passed,
                "profile_total_instances": attempts,
                "profile_pass_rate": (
                    round(passed / attempts, 4) if attempts else None
                ),
                "calibration_note": detail.get("calibrationNote"),
            }
        )

    numeric_values = [float(row["level_value"]) for row in rows]
    by_category = []
    for category, values in sorted(category_values.items()):
        mean_value = _mean(values)
        by_category.append(
            {
                "category": category,
                "primitive_count": len(values),
                "mean_level_value": _round(mean_value),
                "normalized_capability_score": _round(
                    mean_value / 3 if mean_value is not None else None
                ),
            }
        )
    mean_value = _mean(numeric_values)
    summary = {
        "status": "partial" if tcp.get("isPartial") else "complete",
        "model": tcp.get("model"),
        "harness": tcp.get("harness"),
        "profiled_at": tcp.get("profiledAt"),
        "capability_count": len(rows),
        "mean_level_value": _round(mean_value),
        "normalized_capability_score": _round(
            mean_value / 3 if mean_value is not None else None
        ),
        "level_distribution": distribution,
        "microbenchmark_passed_instances": total_passed,
        "microbenchmark_total_instances": total_attempts,
        "microbenchmark_pass_rate": (
            round(total_passed / total_attempts, 4)
            if total_attempts
            else None
        ),
        "by_category": by_category,
        "profile_cost": tcp.get("cost") or {},
    }
    return summary, rows


def _snapshot_variant(
    run_dir: Path, variant: Any
) -> Path:
    destination = (
        run_dir
        / "skvm"
        / "artifacts"
        / _slug(variant.skill.skill_id)
        / f"{_slug(variant.source)}-{_slug(variant.tag)}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(variant.path, destination / "SKILL.md")
    if variant.source == "aot":
        for name in (
            "compilation-plan.json",
            "meta.json",
            "workflow-dag.md",
            "env-setup.sh",
        ):
            source = variant.path.parent / name
            if source.is_file():
                shutil.copy2(source, destination / name)
    return destination


def _variant_evaluation(
    run_dir: Path, variants: Sequence[Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    variant_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    purpose_rows: list[dict[str, Any]] = []
    for variant in variants:
        snapshot = _snapshot_variant(run_dir, variant)
        plan = variant.plan or {}
        artifacts = plan.get("artifacts") or {}
        gaps = artifacts.get("gaps") or []
        scr = artifacts.get("scr") or {}
        purposes = scr.get("purposes") or []
        pass_runs = plan.get("passRuns") or {}
        variant_rows.append(
            {
                "skill_id": variant.skill.skill_id,
                "source": variant.source,
                "variant": variant.tag,
                "changed_from_original": (
                    variant.sha256 != variant.skill.sha256
                ),
                "sha256": variant.sha256,
                "guard_passed": plan.get("guardPassed"),
                "guard_violations": json.dumps(
                    plan.get("guardViolations") or [], ensure_ascii=False
                ),
                "purpose_count": len(purposes),
                "gap_count": len(gaps),
                "successful_pass_count": sum(
                    1
                    for item in pass_runs.values()
                    if isinstance(item, Mapping)
                    and item.get("status") == "ok"
                ),
                "failed_pass_count": sum(
                    1
                    for item in pass_runs.values()
                    if isinstance(item, Mapping)
                    and item.get("status") == "failed"
                ),
                "proposal_id": variant.proposal_id,
                "source_path": str(variant.path),
                "snapshot_path": str(snapshot),
            }
        )
        for gap in gaps:
            if not isinstance(gap, Mapping):
                continue
            gap_rows.append(
                {
                    "skill_id": variant.skill.skill_id,
                    "variant_source": variant.source,
                    "variant": variant.tag,
                    "purpose_id": gap.get("purposeId"),
                    "primitive_id": gap.get("primitiveId"),
                    "required_level": gap.get("requiredLevel"),
                    "model_level": gap.get("modelLevel"),
                    "required_value": LEVEL_VALUES.get(
                        str(gap.get("requiredLevel")), 0
                    ),
                    "model_value": LEVEL_VALUES.get(
                        str(gap.get("modelLevel")), 0
                    ),
                    "gap_size": (
                        LEVEL_VALUES.get(str(gap.get("requiredLevel")), 0)
                        - LEVEL_VALUES.get(str(gap.get("modelLevel")), 0)
                    ),
                    "gap_type": gap.get("gapType"),
                }
            )
        for purpose in purposes:
            if not isinstance(purpose, Mapping):
                continue
            current_path = purpose.get("currentPath") or {}
            primitives = (
                current_path.get("primitives")
                if isinstance(current_path, Mapping)
                else []
            )
            purpose_rows.append(
                {
                    "skill_id": variant.skill.skill_id,
                    "variant_source": variant.source,
                    "variant": variant.tag,
                    "purpose_id": purpose.get("id"),
                    "description": purpose.get("description"),
                    "required_primitives": json.dumps(
                        primitives or [], ensure_ascii=False
                    ),
                }
            )
    return variant_rows, gap_rows, purpose_rows


def _performance_validity(
    android_report: Mapping[str, Any] | None,
    runtime_stats: Mapping[str, Any],
    capability_summary: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    summary = (android_report or {}).get("summary") or {}
    attempted = int(summary.get("attempted_episodes") or 0)
    errors = int(summary.get("error_episodes") or 0)
    scored = int(summary.get("scored_episodes") or 0)
    adapted = int(runtime_stats.get("skvm_adapted_requests") or 0)
    if attempted == 0:
        reasons.append("No Android World episode was attempted.")
    if scored == 0:
        reasons.append("No episode received an Android World score.")
    if attempted and errors / attempted > 0.2:
        reasons.append(
            f"Infrastructure/evaluation error rate is {errors / attempted:.1%}, "
            "above the 20% validity threshold."
        )
    if adapted == 0:
        reasons.append(
            "No model request received online SkVM skill adaptation."
        )
    if capability_summary.get("status") == "missing":
        reasons.append("The target model TCP capability profile is missing.")
    return {
        "valid_for_skvm_effect_comparison": not reasons,
        "invalid_reasons": reasons,
        "attempted_episodes": attempted,
        "scored_episodes": scored,
        "error_episodes": errors,
        "infrastructure_error_rate": (
            round(errors / attempted, 4) if attempted else None
        ),
        "adapted_model_requests": adapted,
    }


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    capability = report["model_capability_evaluation"]
    validity = report["performance_validity"]
    android = report.get("android_world") or {}
    summary = android.get("summary") or {}
    lines = [
        "# SkVM Evaluation Report",
        "",
        f"- Target model: `{report['target_model']}`",
        f"- Harness: `{report['adapter']}`",
        f"- Generated at: `{report['generated_at']}`",
        (
            "- Valid for effect comparison: "
            f"**{'yes' if validity['valid_for_skvm_effect_comparison'] else 'no'}**"
        ),
        "",
    ]
    if validity["invalid_reasons"]:
        lines.extend(["## Validity warnings", ""])
        lines.extend(f"- {reason}" for reason in validity["invalid_reasons"])
        lines.append("")

    lines.extend(
        [
            "## Target capability profile (TCP)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Status | {capability.get('status')} |",
            f"| Profiled primitives | {capability.get('capability_count')} |",
            f"| Mean capability level (0-3) | {capability.get('mean_level_value')} |",
            (
                "| Normalized capability score | "
                f"{_format_percent(capability.get('normalized_capability_score'))} |"
            ),
            (
                "| Microbenchmark pass rate | "
                f"{_format_percent(capability.get('microbenchmark_pass_rate'))} |"
            ),
            "",
            "### Capability by category",
            "",
            "| Category | Primitives | Mean level | Normalized score |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in capability.get("by_category") or []:
        lines.append(
            f"| {item['category']} | {item['primitive_count']} | "
            f"{item['mean_level_value']} | "
            f"{_format_percent(item['normalized_capability_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Skill compilation",
            "",
            "| Skill | Source | Variant | Changed | Guard | Purposes | Gaps |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["skill_compilation"]["variants"]:
        lines.append(
            f"| {item['skill_id']} | {item['source']} | {item['variant']} | "
            f"{item['changed_from_original']} | {item['guard_passed']} | "
            f"{item['purpose_count']} | {item['gap_count']} |"
        )

    runtime = report["runtime_adaptation"]
    lines.extend(
        [
            "",
            "## Runtime adaptation",
            "",
            f"- Adapted requests: `{runtime.get('skvm_adapted_requests', 0)}`",
            (
                "- Variant sources: `"
                f"{json.dumps(runtime.get('skvm_variant_source_counts') or {}, ensure_ascii=False, sort_keys=True)}`"
            ),
            (
                "- Variant tags: `"
                f"{json.dumps(runtime.get('skvm_variant_tag_counts') or {}, ensure_ascii=False, sort_keys=True)}`"
            ),
            (
                "- Android environment recovery: `"
                f"{json.dumps(runtime.get('android_environment_events') or {}, ensure_ascii=False, sort_keys=True)}`"
            ),
            "",
            "## Android World outcome",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Attempted episodes | {summary.get('attempted_episodes', 0)} |",
            f"| Scored episodes | {summary.get('scored_episodes', 0)} |",
            f"| Error episodes | {summary.get('error_episodes', 0)} |",
            (
                "| Task success rate | "
                f"{_format_percent(summary.get('task_success_rate'))} |"
            ),
            "",
            "Detailed machine-readable data and the full TCP snapshot are stored "
            "next to this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_skvm_evaluation(
    *,
    run_dir: Path,
    cache_dir: Path,
    target_model: str,
    adapter: str,
    variants: Sequence[Any],
    manifest: Mapping[str, Any],
    runtime_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Write JSON/Markdown/CSV summaries plus immutable artifact snapshots."""
    profile_source = _profile_path(cache_dir, adapter, target_model)
    tcp: Mapping[str, Any] | None = None
    profile_snapshot: Path | None = None
    if profile_source.is_file():
        loaded = _read_json(profile_source)
        if isinstance(loaded, Mapping):
            tcp = loaded
            profile_snapshot = run_dir / "skvm" / "tcp-profile.json"
            _write_json(profile_snapshot, tcp)

    capability_summary, capability_rows = _capability_evaluation(tcp)
    variant_rows, gap_rows, purpose_rows = _variant_evaluation(
        run_dir, variants
    )
    android_report_path = run_dir / "report.json"
    android_report: Mapping[str, Any] | None = None
    if android_report_path.is_file():
        loaded_android = _read_json(android_report_path)
        if isinstance(loaded_android, Mapping):
            android_report = loaded_android

    validity = _performance_validity(
        android_report, runtime_stats, capability_summary
    )
    report = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_model": target_model,
        "adapter": adapter,
        "performance_validity": validity,
        "model_capability_evaluation": capability_summary
        | {
            "profile_source_path": (
                str(profile_source) if profile_source.is_file() else None
            ),
            "profile_snapshot_path": (
                str(profile_snapshot) if profile_snapshot else None
            ),
            "capabilities": capability_rows,
        },
        "skill_compilation": {
            "skill_count": len(manifest.get("skills") or []),
            "variant_count": len(variant_rows),
            "gap_count": len(gap_rows),
            "purpose_count": len(purpose_rows),
            "variants": variant_rows,
            "gaps": gap_rows,
            "purposes": purpose_rows,
        },
        "runtime_adaptation": dict(runtime_stats),
        "android_world": {
            "report_path": (
                str(android_report_path)
                if android_report_path.is_file()
                else None
            ),
            "run": (android_report or {}).get("run"),
            "summary": (android_report or {}).get("summary"),
        },
        "artifact_files": {
            "capabilities_csv": str(
                run_dir / "skvm" / "capabilities.csv"
            ),
            "variants_csv": str(run_dir / "skvm" / "variants.csv"),
            "gaps_csv": str(run_dir / "skvm" / "capability-gaps.csv"),
            "purposes_csv": str(run_dir / "skvm" / "scr-purposes.csv"),
        },
    }

    _write_csv(run_dir / "skvm" / "capabilities.csv", capability_rows)
    _write_csv(run_dir / "skvm" / "variants.csv", variant_rows)
    _write_csv(run_dir / "skvm" / "capability-gaps.csv", gap_rows)
    _write_csv(run_dir / "skvm" / "scr-purposes.csv", purpose_rows)
    _write_json(run_dir / "skvm_report.json", report)
    _write_text(run_dir / "skvm_report.md", _render_markdown(report))
    return report
