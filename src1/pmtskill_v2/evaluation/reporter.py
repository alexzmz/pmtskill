"""AndroidWorld 可读评测报告。"""

from __future__ import annotations

import collections
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.io import write_json_atomic, write_jsonl
from ..core.models import ExecutionTrace


@dataclass(slots=True)
class EvaluationArtifacts:
    output_dir: Path
    summary_json: Path
    report_markdown: Path
    traces_jsonl: Path
    summary: dict[str, Any]


def successful_episode_value(value: Any) -> bool:
    """只把有限且大于 0.5 的 reward 计为成功，避免 ``bool(NaN)`` 误报。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.5


def finite_float_value(value: Any) -> float:
    """把 NaN、无穷值和非数值统一收敛为 0。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def episode_data_is_usable(value: Any) -> bool:
    """判断 episode_data 是否属于已知的 mapping 或 step-major 结构。"""

    return isinstance(value, Mapping) or (
        isinstance(value, (list, tuple))
        and all(isinstance(step, Mapping) for step in value)
    )


def normalize_episode_data(value: Any) -> Mapping[str, Any]:
    """兼容 dict-of-steps、step-major list；异常标量（含 NaN）视为空。"""

    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (list, tuple)) or not value:
        return {}
    if not all(isinstance(step, Mapping) for step in value):
        return {}
    keys = {
        key
        for step in value
        for key in step
        if isinstance(key, str)
    }
    return {key: [step.get(key) for step in value] for key in keys}


def episode_step_values(value: Any) -> list[Any]:
    """AndroidWorld 的单步字段有时是标量，有时是 list/tuple。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _episode_field(episode: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in episode:
            return episode[key]
    return default


def summarize_episodes(
    episodes: Iterable[dict[str, Any]], traces: Iterable[ExecutionTrace]
) -> dict[str, Any]:
    """计算 micro/macro SR、任务粒度统计和在线路由开销。"""

    episode_list = list(episodes)
    trace_list = list(traces)
    valid = [episode for episode in episode_list if not _episode_field(episode, "exception_info")]
    successes = sum(
        successful_episode_value(_episode_field(item, "is_successful", default=False))
        for item in valid
    )
    by_task: dict[str, list[bool]] = collections.defaultdict(list)
    failure_reasons: collections.Counter[str] = collections.Counter()
    step_counts: list[int] = []
    run_times: list[float] = []
    for episode in episode_list:
        task = str(_episode_field(episode, "task_template", "task_name", default="unknown"))
        if _episode_field(episode, "exception_info"):
            failure_reasons["exception"] += 1
            continue
        successful = successful_episode_value(
            _episode_field(episode, "is_successful", default=False)
        )
        by_task[task].append(successful)
        step_counts.append(
            int(finite_float_value(_episode_field(episode, "episode_length", default=0)))
        )
        run_times.append(
            finite_float_value(_episode_field(episode, "run_time", default=0.0))
        )
        if not successful:
            raw_episode_data = _episode_field(episode, "episode_data", default={})
            episode_data = normalize_episode_data(raw_episode_data)
            outputs = episode_step_values(episode_data.get("action_output"))
            if raw_episode_data is not None and not episode_data_is_usable(
                raw_episode_data
            ):
                failure_reasons["invalid_episode_data"] += 1
                continue
            if outputs and "infeasible" in str(outputs[-1]).lower():
                failure_reasons["agent_infeasible"] += 1
            elif outputs and "status" not in str(outputs[-1]).lower():
                failure_reasons["environment_not_satisfied"] += 1
            else:
                failure_reasons["agent_finished_but_failed"] += 1

    task_rows = {
        task: {
            "successes": sum(results),
            "episodes": len(results),
            "success_rate": sum(results) / len(results) if results else 0.0,
        }
        for task, results in sorted(by_task.items())
    }
    model_usage: collections.Counter[str] = collections.Counter()
    skill_usage: collections.Counter[str] = collections.Counter()
    switch_counts: list[int] = []
    for trace in trace_list:
        previous: str | None = None
        switches = 0
        for event in trace.events:
            model_usage[event.model_id] += 1
            if event.skill_id:
                skill_usage[event.skill_id] += 1
            if previous is not None and previous != event.model_id:
                switches += 1
            previous = event.model_id
        switch_counts.append(switches)

    per_task_rates = [row["success_rate"] for row in task_rows.values()]
    return {
        "episodes_total": len(episode_list),
        "episodes_evaluated": len(valid),
        "successes": successes,
        "success_rate_micro": successes / len(valid) if valid else 0.0,
        "success_rate_macro": statistics.fmean(per_task_rates) if per_task_rates else 0.0,
        "task_count": len(task_rows),
        "average_steps": statistics.fmean(step_counts) if step_counts else 0.0,
        "average_run_time_seconds": statistics.fmean(run_times) if run_times else 0.0,
        "average_model_switches": statistics.fmean(switch_counts) if switch_counts else 0.0,
        "model_usage": dict(model_usage.most_common()),
        "skill_usage": dict(skill_usage.most_common()),
        "failure_reasons": dict(failure_reasons.most_common()),
        "per_task": task_rows,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PMT-Skill AndroidWorld 评测报告",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 有效 episode | {summary['episodes_evaluated']} / {summary['episodes_total']} |",
        f"| 成功数 | {summary['successes']} |",
        f"| Micro SR | {summary['success_rate_micro']:.2%} |",
        f"| Macro SR | {summary['success_rate_macro']:.2%} |",
        f"| 平均步数 | {summary['average_steps']:.2f} |",
        f"| 平均运行时间 | {summary['average_run_time_seconds']:.2f}s |",
        f"| 平均模型/adapter 切换 | {summary['average_model_switches']:.2f} |",
        "",
        "## 每任务结果",
        "",
        "| 任务 | 成功/总数 | SR |",
        "|---|---:|---:|",
    ]
    for task, row in summary["per_task"].items():
        lines.append(
            f"| {task} | {row['successes']}/{row['episodes']} | {row['success_rate']:.2%} |"
        )
    lines.extend(("", "## 模型调用", "", "| 模型/adapter | 次数 |", "|---|---:|"))
    for model, count in summary["model_usage"].items():
        lines.append(f"| {model} | {count} |")
    lines.extend(("", "## 技能调用", "", "| polished/raw skill | 次数 |", "|---|---:|"))
    if summary["skill_usage"]:
        for skill, count in summary["skill_usage"].items():
            lines.append(f"| {skill} | {count} |")
    else:
        lines.append("| （仅原语 fallback） | 0 |")
    lines.extend(("", "## 失败分类", "", "| 原因 | 次数 |", "|---|---:|"))
    for reason, count in summary["failure_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    output_dir: str | Path,
    episodes: list[dict[str, Any]],
    traces: list[ExecutionTrace],
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluationArtifacts:
    """同时写机器可读 JSON/JSONL 与人工可读 Markdown。"""

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    summary = summarize_episodes(episodes, traces)
    summary["metadata"] = metadata or {}
    summary_path = target / "summary.json"
    markdown_path = target / "report.md"
    traces_path = target / "traces.jsonl"
    write_json_atomic(summary_path, summary)
    write_jsonl(traces_path, (trace.to_dict() for trace in traces))
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return EvaluationArtifacts(target, summary_path, markdown_path, traces_path, summary)
