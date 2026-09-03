"""教师轨迹采集的最终统计。

AndroidWorld 在每个 episode 后会打印累计 DataFrame；这里把同一批信息转换成
稳定字典，让 CLI 的 ``result.json``/``result.md`` 在任务结束后仍可查阅。
"""

from __future__ import annotations

import collections
import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from .dataset import infer_action_primitives


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _successful(value: Any) -> bool:
    return _finite_float(value, default=0.0) > 0.5


def _episode_data(value: Any) -> dict[str, list[Any]]:
    """兼容 AndroidWorld 的 dict-of-lists 和旧版 list-of-dicts。"""

    if isinstance(value, Mapping):
        normalized: dict[str, list[Any]] = {}
        for key, item in value.items():
            if isinstance(item, (list, tuple)):
                normalized[str(key)] = list(item)
            elif item is not None:
                normalized[str(key)] = [item]
        return normalized
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        keys = {str(key) for item in value for key in item}
        return {key: [item.get(key) for item in value] for key in keys}
    return {}


def _primitive_metrics(episodes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 episode 最终结果为动作原语做保守信用分配。

    一个 episode 成功时其中被调用的 primitive 记成功，否则记失败。这和在线
    backend 的保守信用分配一致，虽然不能证明某个局部动作独立正确，但不会因
    “动作可解析”就高估 primitive 能力。
    """

    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for episode in episodes:
        success = _successful(episode.get("is_successful", False))
        data = _episode_data(episode.get("episode_data", {}))
        outputs = data.get("action_output", [])
        if not outputs:
            outputs = data.get("action_raw_response", [])
        for output in outputs:
            if output is None:
                continue
            for primitive in infer_action_primitives(str(output)):
                counts[primitive][0] += int(success)
                counts[primitive][1] += 1
    return {
        primitive: {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials if trials else 0.0,
        }
        for primitive, (successes, trials) in sorted(counts.items())
    }


def summarize_collection(
    episodes: Iterable[dict[str, Any]],
    *,
    full_episodes: Iterable[dict[str, Any]] = (),
    episode_step_limit: int,
) -> dict[str, Any]:
    """生成 collect 最终汇总，包括 TSR、每任务和每原语统计。"""

    rows = list(episodes)
    detailed_rows = list(full_episodes)
    per_task_values: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    termination_reasons: collections.Counter[str] = collections.Counter()
    permission_restarts = 0
    permission_dialogs_dismissed = 0
    permission_model_delegations = 0
    valid_rows: list[dict[str, Any]] = []
    for episode in rows:
        task = str(episode.get("task_template") or episode.get("task_name") or "unknown")
        per_task_values[task].append(episode)
        exception = episode.get("exception_info")
        if not exception:
            valid_rows.append(episode)
        aux_data = episode.get("aux_data")
        if isinstance(aux_data, Mapping):
            reason = aux_data.get("collector_termination_reason")
            if reason:
                termination_reasons[str(reason)] += 1
            permission_restarts += int(
                _finite_float(aux_data.get("permission_controller_restarts"))
            )
            permission_dialogs_dismissed += int(
                _finite_float(
                    aux_data.get("permission_controller_dialogs_dismissed")
                )
            )
            permission_model_delegations += int(
                _finite_float(
                    aux_data.get("permission_controller_model_delegations")
                )
            )
        if exception:
            termination_reasons["exception"] += 1

    per_task: dict[str, dict[str, Any]] = {}
    for task, task_rows in sorted(per_task_values.items()):
        evaluated = [item for item in task_rows if not item.get("exception_info")]
        successes = sum(_successful(item.get("is_successful")) for item in evaluated)
        step_values = [
            _finite_float(item.get("episode_length"))
            for item in evaluated
            if _finite_float(item.get("episode_length")) > 0
        ]
        runtime_values = [_finite_float(item.get("run_time")) for item in evaluated]
        per_task[task] = {
            "successes": successes,
            "episodes": len(evaluated),
            "attempted_episodes": len(task_rows),
            "exceptions": len(task_rows) - len(evaluated),
            "success_rate": successes / len(evaluated) if evaluated else 0.0,
            "average_steps": statistics.fmean(step_values) if step_values else 0.0,
            "total_runtime_seconds": sum(runtime_values),
            "episodes_at_step_limit": sum(
                _finite_float(item.get("episode_length")) >= episode_step_limit
                for item in evaluated
            ),
        }

    successes = sum(_successful(item.get("is_successful")) for item in valid_rows)
    task_rates = [value["success_rate"] for value in per_task.values() if value["episodes"]]
    step_values = [
        _finite_float(item.get("episode_length"))
        for item in valid_rows
        if _finite_float(item.get("episode_length")) > 0
    ]
    runtime_values = [_finite_float(item.get("run_time")) for item in valid_rows]
    return {
        "episodes_total": len(rows),
        "episodes_evaluated": len(valid_rows),
        "successes": successes,
        "task_success_rate": successes / len(valid_rows) if valid_rows else 0.0,
        "success_rate_micro": successes / len(valid_rows) if valid_rows else 0.0,
        "success_rate_macro": statistics.fmean(task_rates) if task_rates else 0.0,
        "task_count": len(per_task),
        "average_steps": statistics.fmean(step_values) if step_values else 0.0,
        "total_runtime_seconds": sum(runtime_values),
        "episode_step_limit": episode_step_limit,
        "episodes_at_step_limit": sum(
            _finite_float(item.get("episode_length")) >= episode_step_limit
            for item in valid_rows
        ),
        "permission_controller_restarts": permission_restarts,
        "permission_controller_dialogs_dismissed": permission_dialogs_dismissed,
        "permission_controller_model_delegations": permission_model_delegations,
        "termination_reasons": dict(termination_reasons.most_common()),
        "primitive_credit_assignment": "primitive call + final episode success",
        "per_task": per_task,
        "per_primitive": _primitive_metrics(detailed_rows),
    }
