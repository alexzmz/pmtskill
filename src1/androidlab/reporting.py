"""Report unavailable evaluation separately from genuine unsuccessful tasks."""

from __future__ import annotations

import collections
import csv
from pathlib import Path

from ..pmtskill_v2.core.io import write_json_atomic


def aggregate(rows, expected):
    valid = [r for r in rows if r.get("status") == "completed"]
    successes = sum(r["successful"] is True for r in valid)
    available = len(valid) == expected and expected > 0
    return {"expected_episodes": expected, "valid_episodes": len(valid),
            "unavailable_episodes": expected - len(valid), "successful_episodes": successes,
            "evaluation_available": available,
            "success_rate": successes / expected if available else None,
            "observed_success_rate": successes / len(valid) if valid else None,
            "mean_steps": sum(r.get("steps", 0) for r in valid) / len(valid) if valid else None,
            "mean_latency_ms": sum(r.get("duration_ms", 0) for r in valid) / len(valid) if valid else None,
            "model_switches": sum(r.get("switch_count", 0) for r in valid)}


def write_report(target, rows, expected, metadata):
    target = Path(target)
    total = aggregate(rows, expected)
    by_app, by_capability = collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        by_app[row["app"]].append(row)
        by_capability[row.get("capability", "unclassified")].append(row)
    summary = {**total, "metadata": metadata,
               "by_app": {key: aggregate(values, len(values)) for key, values in sorted(by_app.items())},
               "by_capability": {key: aggregate(values, len(values)) for key, values in sorted(by_capability.items())}}
    write_json_atomic(target / "summary.json", summary)
    fields = ("task_id", "repeat", "app", "capability", "status", "successful", "steps", "decision_steps",
              "duration_ms", "switch_count", "model_calls", "error")
    with (target / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = [f"# {metadata['method']} — AndroidLab", "",
             f"- Complete evaluation available: {total['evaluation_available']}",
             f"- SR: {percent(total['success_rate'])}",
             f"- Valid / expected: {total['valid_episodes']} / {expected}",
             f"- Observed SR (partial, not full benchmark SR): {percent(total['observed_success_rate'])}",
             "", "All methods use the official AndroidLab task predicates. Agent-declared completion is not success.",
             "Exceptions/timeouts/judge failures are unavailable, not zero-score task outcomes.",
             "Only SR is reported as an AndroidLab benchmark metric; Sub-SR/RRR/ROR are not reimplemented.",
             "", "| App | Valid / expected | SR |", "|---|---:|---:|"]
    for name, stats in summary["by_app"].items():
        lines.append(f"| {name} | {stats['valid_episodes']} / {stats['expected_episodes']} | {percent(stats['success_rate'])} |")
    device_protocol = (f"Disposable native AVD copy per episode; baseline snapshot={metadata.get('avd_snapshot') or 'none (cold boot from pre-populated userdata)'}."
                       if metadata.get("backend") == "avd" else "Fresh official Docker container per episode.")
    lines += ["", "## Protocol", "", f"- {device_protocol} No AndroidWorld setup/reset.",
              "- M3A/T3A/V-Droid observations/actions are bridged from AndroidLab XML and screenshots.",
              "- Last eligible official judge result is retained; no success-oracle early stopping.",
              "- Final post-action state is included explicitly (bridge extension to upstream trace collection).",
              "- Repeats reuse fixed task definitions, NOT new randomized task parameters.",
              "- Execution latency excludes GPU deployment, device initialization, judge time and resource teardown; policy initialization/planning are included.",
              "- Steps count committed JSON GUI actions (not low-level taps, summary calls or finish); M3A decision_steps are stored separately.",
              "- AndroidWorld-trained adapters/skills, if supplied, are cross-benchmark transfer; not AndroidLab calibration."]
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
