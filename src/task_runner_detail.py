"""Run a reproducible Android World evaluation with detailed reports.

Unlike ``local_minimal_task_runner.py``, this runner evaluates a deterministic
suite of task instances, uses Android World's official episode runner and
success signals, saves resumable checkpoints, and produces human- and
machine-readable reports.

Example:

    python src/task_runner_detail.py \
      --model_path /models/Qwen \
      --tasks ContactsAddContact ClockCreateTimer

Omit ``--tasks`` to run the complete ``android_world`` family.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANDROID_WORLD_ROOT = REPO_ROOT / "libs" / "android_world"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "android_world"
REPORT_SCHEMA_VERSION = "1.0"


def _bootstrap_import_paths() -> Path:
    """Make the vendored Android World and local wrappers importable."""
    configured_root = os.environ.get("ANDROID_WORLD_ROOT")
    android_world_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else DEFAULT_ANDROID_WORLD_ROOT
    )
    for path in (android_world_root, REPO_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return android_world_root


def _find_adb() -> str:
    configured = os.environ.get("ADB_PATH")
    if configured:
        return configured

    candidates = [
        Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb",
        Path.home() / "Android" / "Sdk" / "platform-tools" / "adb",
        Path("/home/zmz/Workspace/gui/Android/Sdk/platform-tools/adb"),
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Android"
            / "Sdk"
            / "platform-tools"
            / "adb.exe"
        )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            # A sandboxed process may be denied access to an otherwise common
            # SDK location. Fall through to the next candidate/normal PATH.
            continue
    return "adb"


def build_parser(
    description: str | None = None,
    *,
    add_help: bool = True,
    include_model_args: bool = True,
) -> argparse.ArgumentParser:
    """Build the shared Android World command-line interface."""
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    runtime = parser.add_argument_group("Android runtime")
    runtime.add_argument("--adb_path", default=_find_adb(), help="Path to adb.")
    runtime.add_argument(
        "--console_port",
        type=int,
        default=5554,
        help="Console port of the running Android emulator.",
    )
    runtime.add_argument(
        "--grpc_port",
        type=int,
        default=8554,
        help="gRPC port exposed by the running Android emulator.",
    )
    runtime.add_argument(
        "--perform_emulator_setup",
        action="store_true",
        help="Perform the one-time Android World emulator setup.",
    )
    runtime.add_argument(
        "--transition_pause",
        type=float,
        default=None,
        help=(
            "Seconds to wait after actions. The default lets Android World "
            "dynamically wait for a stable screen."
        ),
    )
    runtime.add_argument(
        "--max_consecutive_infrastructure_errors",
        type=int,
        default=3,
        help=(
            "Abort after this many consecutive emulator/a11y/model-backend "
            "errors instead of recording the same invalid result for the "
            "entire suite. Set zero to disable."
        ),
    )

    suite = parser.add_argument_group("Evaluation suite")
    suite.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=(
            "Task template names separated by spaces or commas. Omit this "
            "option to evaluate the complete android_world family."
        ),
    )
    suite.add_argument(
        "--n_task_combinations",
        type=int,
        default=1,
        help="Number of seeded instances to run per task template.",
    )
    suite.add_argument(
        "--task_random_seed",
        type=int,
        default=30,
        help="Seed used by Android World to generate task parameters.",
    )
    suite.add_argument(
        "--fixed_task_seed",
        action="store_true",
        help="Give repeated combinations of one template identical parameters.",
    )
    suite.add_argument(
        "--list_tasks",
        action="store_true",
        help="List available Android World task names and exit.",
    )

    if include_model_args:
        model = parser.add_argument_group("Local vLLM model")
        model.add_argument(
            "--model_path",
            default=os.environ.get(
                "LOCAL_MODEL_PATH", "/home/zmz/Workspace/models/qwen3.5-4b"
            ),
            help="Local Hugging Face model directory or vLLM model identifier.",
        )
        model.add_argument("--tensor_parallel_size", type=int, default=1)
        model.add_argument("--gpu_memory_utilization", type=float, default=0.9)
        model.add_argument("--max_model_len", type=int, default=None)
        model.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help=(
                "Sampling temperature. Zero is recommended for reproducible "
                "evals."
            ),
        )
        model.add_argument("--top_p", type=float, default=0.95)
        model.add_argument("--max_tokens", type=int, default=512)

    output = parser.add_argument_group("Reports and checkpoints")
    output.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory used when creating a new timestamped run.",
    )
    output.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help=(
            "Exact run directory. Reusing a compatible directory resumes from "
            "its per-episode checkpoints."
        ),
    )
    output.add_argument(
        "--include_prompts",
        action="store_true",
        help="Include full action and summary prompts in report.json.",
    )
    output.add_argument(
        "--markdown_step_char_limit",
        type=int,
        default=500,
        help="Maximum characters per action/summary in the Markdown report.",
    )
    return parser


def normalize_tasks(raw_tasks: Sequence[str] | None) -> list[str] | None:
    """Accept both whitespace- and comma-separated task names."""
    if not raw_tasks:
        return None
    tasks: list[str] = []
    for item in raw_tasks:
        tasks.extend(part.strip() for part in item.split(",") if part.strip())
    return list(dict.fromkeys(tasks)) or None


def validate_args(args: argparse.Namespace) -> None:
    if args.n_task_combinations < 1:
        raise ValueError("--n_task_combinations must be at least 1.")
    if (
        hasattr(args, "tensor_parallel_size")
        and args.tensor_parallel_size < 1
    ):
        raise ValueError("--tensor_parallel_size must be at least 1.")
    if (
        hasattr(args, "gpu_memory_utilization")
        and not 0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("--gpu_memory_utilization must be in (0, 1].")
    if (
        hasattr(args, "max_model_len")
        and args.max_model_len is not None
        and args.max_model_len < 1
    ):
        raise ValueError("--max_model_len must be positive.")
    if args.max_tokens < 1:
        raise ValueError("--max_tokens must be positive.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1].")
    if args.transition_pause is not None and args.transition_pause < 0:
        raise ValueError("--transition_pause cannot be negative.")
    if not 1 <= args.grpc_port <= 65535:
        raise ValueError("--grpc_port must be between 1 and 65535.")
    if args.max_consecutive_infrastructure_errors < 0:
        raise ValueError(
            "--max_consecutive_infrastructure_errors cannot be negative."
        )
    if args.markdown_step_char_limit < 80:
        raise ValueError("--markdown_step_char_limit must be at least 80.")


def create_vllm(args: argparse.Namespace) -> Any:
    """Instantiate the baseline local model wrapper."""
    from vllm_wrapper import VLLMWrapper

    return VLLMWrapper(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        raise_on_error=True,
    )


def _slug(text: str, limit: int = 60) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return (value or "model")[:limit]


def _resolve_run_dir(args: argparse.Namespace, condition: str) -> Path:
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    model_reference = (
        getattr(args, "model_path", None)
        or getattr(args, "deepseek_model", None)
        or condition
    )
    model_name = Path(str(model_reference).rstrip("/\\")).name
    return (
        args.output_dir.expanduser().resolve()
        / f"{_slug(condition)}_{_slug(model_name)}_{timestamp}"
    )


def _config_signature(config: Mapping[str, Any]) -> str:
    comparable = {
        key: config.get(key)
        for key in (
            "condition",
            "model",
            "suite",
            "android_runtime",
            "skill",
        )
    }
    encoded = json.dumps(
        comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    """Convert common numpy/datetime/path values without importing numpy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.fmean(cleaned) if cleaned else None


def _safe_median(values: Iterable[float | int | None]) -> float | None:
    cleaned = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(cleaned) if cleaned else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _load_task_metadata(android_world_root: Path) -> dict[str, dict[str, Any]]:
    metadata_path = android_world_root / "android_world" / "task_metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row["task_name"]): {
            "difficulty": row.get("difficulty"),
            "optimal_steps": row.get("optimal_steps"),
            "tags": row.get("tags") or [],
        }
        for row in rows
        if isinstance(row, dict) and row.get("task_name")
    }


def _step_at(step_data: Mapping[str, Any], key: str, index: int) -> Any:
    values = step_data.get(key)
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return None


def _extract_reason_and_action(output: Any) -> tuple[str | None, str | None]:
    if not isinstance(output, str):
        return None, None
    match = re.search(
        r"Reason:\s*(.*?)\s*Action:\s*(\{.*\})",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _extract_steps(
    episode_data: Any, *, include_prompts: bool
) -> list[dict[str, Any]]:
    if not isinstance(episode_data, Mapping):
        return []
    lengths = [
        len(value)
        for value in episode_data.values()
        if isinstance(value, (list, tuple))
    ]
    step_count = max(lengths, default=0)
    steps: list[dict[str, Any]] = []
    for index in range(step_count):
        action_output = _step_at(episode_data, "action_output", index)
        reason, action = _extract_reason_and_action(action_output)
        step = {
            "step_number": _json_value(
                _step_at(episode_data, "step_number", index)
            )
            if _step_at(episode_data, "step_number", index) is not None
            else index,
            "reason": reason,
            "action": action,
            "action_output": _json_value(action_output),
            "summary": _json_value(
                _step_at(episode_data, "summary", index)
            ),
        }
        if include_prompts:
            step["action_prompt"] = _json_value(
                _step_at(episode_data, "action_prompt", index)
            )
            step["summary_prompt"] = _json_value(
                _step_at(episode_data, "summary_prompt", index)
            )
        steps.append(step)
    return steps


def _extract_episode(
    raw: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    include_prompts: bool,
) -> dict[str, Any]:
    task_name = str(raw.get("task_template") or "unknown")
    exception = raw.get("exception_info")
    score = _finite_number(raw.get("is_successful"))
    has_error = exception is not None
    if has_error:
        outcome = "error"
        score = None
    elif score is None:
        outcome = "unscored"
    elif score > 0.5:
        outcome = "success"
    else:
        outcome = "failure"

    task_metadata = metadata.get(task_name, {})
    return {
        "task_template": task_name,
        "instance_id": _json_value(raw.get("instance_id")),
        "seed": _json_value(raw.get("seed")),
        "goal": _json_value(raw.get("goal")),
        "outcome": outcome,
        "score": _round(score),
        "is_successful": bool(score is not None and score > 0.5),
        "episode_length": _round(_finite_number(raw.get("episode_length"))),
        "runtime_s": _round(_finite_number(raw.get("run_time")), 3),
        "difficulty": task_metadata.get("difficulty"),
        "optimal_steps": _json_value(task_metadata.get("optimal_steps")),
        "tags": list(task_metadata.get("tags") or []),
        "exception": _json_value(exception),
        "aux_data": _json_value(raw.get("aux_data")),
        "steps": _extract_steps(
            raw.get("episode_data"), include_prompts=include_prompts
        ),
    }


def _aggregate_group(
    name: str, episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    scored = [row for row in episodes if row.get("score") is not None]
    scores = [float(row["score"]) for row in scored]
    successes = sum(float(row["score"]) > 0.5 for row in scored)
    errors = sum(row.get("outcome") == "error" for row in episodes)
    return {
        "name": name,
        "attempted_episodes": len(episodes),
        "scored_episodes": len(scored),
        "successful_episodes": successes,
        "failed_episodes": len(scored) - successes,
        "error_episodes": errors,
        "task_success_rate": _round(_safe_mean(scores)),
        "mean_episode_length": _round(
            _safe_mean(row.get("episode_length") for row in scored), 2
        ),
        "mean_runtime_s": _round(
            _safe_mean(row.get("runtime_s") for row in episodes), 2
        ),
        "total_runtime_s": _round(
            sum(
                float(row["runtime_s"])
                for row in episodes
                if row.get("runtime_s") is not None
            ),
            2,
        ),
    }


def build_report(
    raw_episodes: Sequence[Mapping[str, Any]],
    *,
    run_config: Mapping[str, Any],
    android_world_root: Path,
    planned_episodes: int,
    run_status: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    runner_error: str | None,
    inference_stats: Mapping[str, Any] | None,
    include_prompts: bool,
) -> dict[str, Any]:
    metadata = _load_task_metadata(android_world_root)
    episodes = [
        _extract_episode(
            raw, metadata, include_prompts=include_prompts
        )
        for raw in raw_episodes
    ]
    episodes.sort(
        key=lambda row: (
            str(row["task_template"]),
            -1 if row["instance_id"] is None else int(row["instance_id"]),
        )
    )

    by_task: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_task.setdefault(episode["task_template"], []).append(episode)
    task_metrics = []
    for task_name, task_episodes in sorted(by_task.items()):
        row = _aggregate_group(task_name, task_episodes)
        row["difficulty"] = task_episodes[0].get("difficulty")
        row["tags"] = task_episodes[0].get("tags") or []
        row["optimal_steps"] = task_episodes[0].get("optimal_steps")
        task_metrics.append(row)

    scored = [row for row in episodes if row["score"] is not None]
    scores = [float(row["score"]) for row in scored]
    successful = sum(score > 0.5 for score in scores)
    errors = sum(row["outcome"] == "error" for row in episodes)
    macro_scores = [
        float(row["task_success_rate"])
        for row in task_metrics
        if row["task_success_rate"] is not None
    ]

    summary = {
        "planned_episodes": planned_episodes,
        "attempted_episodes": len(episodes),
        "scored_episodes": len(scored),
        "successful_episodes": successful,
        "failed_episodes": len(scored) - successful,
        "error_episodes": errors,
        # Android World tasks are binary; mean score is therefore the primary
        # task success rate. Evaluation errors are excluded, matching the
        # official process_episodes() aggregation.
        "task_success_rate": _round(_safe_mean(scores)),
        "macro_task_success_rate": _round(_safe_mean(macro_scores)),
        "evaluation_coverage": _round(
            len(scored) / planned_episodes if planned_episodes else None
        ),
        "error_rate": _round(
            errors / len(episodes) if episodes else None
        ),
        "mean_episode_length": _round(
            _safe_mean(row["episode_length"] for row in scored), 2
        ),
        "median_episode_length": _round(
            _safe_median(row["episode_length"] for row in scored), 2
        ),
        "total_episode_runtime_s": _round(
            sum(
                float(row["runtime_s"])
                for row in episodes
                if row["runtime_s"] is not None
            ),
            2,
        ),
    }

    difficulty_groups: dict[str, list[dict[str, Any]]] = {}
    tag_groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        difficulty_groups.setdefault(
            str(episode.get("difficulty") or "unknown"), []
        ).append(episode)
        tags = episode.get("tags") or ["untagged"]
        for tag in tags:
            tag_groups.setdefault(str(tag), []).append(episode)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_clock_runtime_s": round(
                (finished_at - started_at).total_seconds(), 3
            ),
            "runner_error": runner_error,
            "config": _json_value(run_config),
        },
        "metric_definition": {
            "task_success_rate": (
                "Mean Android World is_successful score across episodes that "
                "finished evaluation without an exception."
            ),
            "macro_task_success_rate": (
                "Unweighted mean of each task template's success rate."
            ),
            "evaluation_coverage": (
                "Scored episodes divided by the number of planned episodes."
            ),
            "success_threshold": "is_successful > 0.5",
            "error_policy": (
                "Infrastructure/evaluator exceptions are counted separately "
                "and excluded from the success-rate denominator."
            ),
        },
        "summary": summary,
        "inference": _json_value(dict(inference_stats or {})),
        "breakdown": {
            "by_task": task_metrics,
            "by_difficulty": [
                _aggregate_group(name, rows)
                for name, rows in sorted(difficulty_groups.items())
            ],
            "by_tag": [
                _aggregate_group(name, rows)
                for name, rows in sorted(tag_groups.items())
            ],
        },
        "episodes": episodes,
    }


def _percent(value: Any) -> str:
    number = _finite_number(value)
    return "N/A" if number is None else f"{number * 100:.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    number = _finite_number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _table_text(value: Any) -> str:
    return str(value if value is not None else "N/A").replace("|", "\\|")


def _compact(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(report: Mapping[str, Any], step_char_limit: int) -> str:
    run = report["run"]
    config = run["config"]
    summary = report["summary"]
    condition = config.get("condition", "unknown")
    model = config.get("model", {}).get("model_path", "unknown")

    lines = [
        "# Android World Evaluation Report",
        "",
        f"- Status: **{run['status']}**",
        f"- Condition: `{condition}`",
        f"- Model: `{model}`",
        f"- Started: `{run['started_at']}`",
        f"- Finished: `{run['finished_at']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Task success rate | **{_percent(summary['task_success_rate'])}** |",
        (
            "| Macro task success rate | "
            f"**{_percent(summary['macro_task_success_rate'])}** |"
        ),
        f"| Planned episodes | {summary['planned_episodes']} |",
        f"| Attempted episodes | {summary['attempted_episodes']} |",
        f"| Scored episodes | {summary['scored_episodes']} |",
        f"| Successful episodes | {summary['successful_episodes']} |",
        f"| Failed episodes | {summary['failed_episodes']} |",
        f"| Evaluation errors | {summary['error_episodes']} |",
        f"| Evaluation coverage | {_percent(summary['evaluation_coverage'])} |",
        (
            "| Mean episode length | "
            f"{_number(summary['mean_episode_length'])} steps |"
        ),
        (
            "| Total episode runtime | "
            f"{_number(summary['total_episode_runtime_s'])} s |"
        ),
        "",
        (
            "The primary success rate is the mean of Android World's official "
            "`is_successful` signal over scored episodes. Evaluation errors "
            "are shown separately and excluded from that denominator."
        ),
    ]

    skill = config.get("skill")
    if skill:
        lines.extend(
            [
                "",
                "## SkVM skill condition",
                "",
                f"- Mode: `{skill.get('mode')}`",
                f"- Skill: `{skill.get('name')}`",
                f"- Source: `{skill.get('path')}`",
                f"- SHA-256: `{skill.get('sha256')}`",
            ]
        )
        if skill.get("delivery") == "skvm-kernel-adaptive":
            lines.extend(
                [
                    f"- SkVM target: `{skill.get('target_model')}`",
                    f"- Compiler model: `{skill.get('compiler_model')}`",
                    f"- Prepare mode: `{skill.get('prepare_mode')}`",
                    (
                        "- Catalog: "
                        f"{len(skill.get('skills') or [])} skill(s), "
                        f"{len(skill.get('variants') or [])} variant(s)"
                    ),
                    f"- Adaptation trace: `{skill.get('adaptation_trace')}`",
                ]
            )

    inference = report.get("inference") or {}
    if inference:
        lines.extend(
            [
                "",
                "## Inference",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Model requests | {inference.get('request_count', 'N/A')} |",
                f"| Model-call errors | {inference.get('error_count', 'N/A')} |",
                f"| Prompt tokens | {inference.get('prompt_tokens', 'N/A')} |",
                (
                    "| Generated tokens | "
                    f"{inference.get('generated_tokens', 'N/A')} |"
                ),
                (
                    "| Mean model latency | "
                    f"{_number(inference.get('mean_latency_s'), 3)} s |"
                ),
            ]
        )
        if inference.get("skill_load_requests") is not None:
            lines.append(
                "| SkVM skill load requests | "
                f"{inference.get('skill_load_requests')} |"
            )
        optional_inference_metrics = (
            ("API attempts", "api_attempt_count", None),
            ("API retries", "retry_count", None),
            ("Reasoning tokens", "reasoning_tokens", None),
            ("Prompt cache-hit tokens", "prompt_cache_hit_tokens", None),
            ("Prompt cache-miss tokens", "prompt_cache_miss_tokens", None),
            ("Prompt cache-hit rate", "prompt_cache_hit_rate", "percent"),
            ("VL prompt profile", "prompt_profile", None),
            ("VL bbox coordinate mode", "bbox_coordinate_mode", None),
            ("VL action requests", "action_request_count", None),
            ("VL summary requests", "summary_request_count", None),
            ("Screenshots sent", "image_count", None),
            ("Screenshots resized", "resized_image_count", None),
            ("Actions normalized", "converted_action_count", None),
            ("Native M3A actions", "native_action_count", None),
            ("Actions left unparsed", "unparsed_action_count", None),
        )
        for label, key, value_format in optional_inference_metrics:
            if inference.get(key) is None:
                continue
            value = (
                _percent(inference[key])
                if value_format == "percent"
                else inference[key]
            )
            lines.append(f"| {label} | {value} |")
        if inference.get("skvm_adapted_requests") is not None:
            lines.extend(
                [
                    (
                        "| SkVM-adapted requests | "
                        f"{inference.get('skvm_adapted_requests')} |"
                    ),
                    (
                        "| SkVM variant sources | "
                        f"{_table_text(json.dumps(inference.get('skvm_variant_source_counts') or {}, ensure_ascii=False, sort_keys=True))} |"
                    ),
                    (
                        "| SkVM variant tags | "
                        f"{_table_text(json.dumps(inference.get('skvm_variant_tag_counts') or {}, ensure_ascii=False, sort_keys=True))} |"
                    ),
                    (
                        "| SkVM UI-state labels | "
                        f"{_table_text(json.dumps(inference.get('skvm_ui_state_counts') or {}, ensure_ascii=False, sort_keys=True))} |"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "## Results by difficulty",
            "",
            "| Difficulty | Scored | Successes | Errors | Success rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["breakdown"]["by_difficulty"]:
        lines.append(
            f"| {_table_text(row['name'])} | {row['scored_episodes']} | "
            f"{row['successful_episodes']} | {row['error_episodes']} | "
            f"{_percent(row['task_success_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Results by task",
            "",
            (
                "| Task template | Difficulty | Scored | Successes | Errors | "
                "Success rate | Mean steps | Runtime (s) |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["breakdown"]["by_task"]:
        lines.append(
            f"| {_table_text(row['name'])} | "
            f"{_table_text(row.get('difficulty'))} | "
            f"{row['scored_episodes']} | {row['successful_episodes']} | "
            f"{row['error_episodes']} | "
            f"{_percent(row['task_success_rate'])} | "
            f"{_number(row['mean_episode_length'])} | "
            f"{_number(row['total_runtime_s'])} |"
        )

    lines.extend(["", "## Episode details", ""])
    for episode in report["episodes"]:
        icon = {
            "success": "PASS",
            "failure": "FAIL",
            "error": "ERROR",
            "unscored": "UNSCORED",
        }[episode["outcome"]]
        lines.extend(
            [
                (
                    f"### [{icon}] {episode['task_template']} "
                    f"#{episode['instance_id']}"
                ),
                "",
                f"- Goal: {_compact(episode['goal'], 1000)}",
                (
                    f"- Score: {_number(episode['score'], 4)}; "
                    f"steps: {_number(episode['episode_length'], 0)}; "
                    f"runtime: {_number(episode['runtime_s'], 2)} s"
                ),
            ]
        )
        if episode.get("exception"):
            lines.append(
                f"- Exception: `{_compact(episode['exception'], 1000)}`"
            )
        steps = episode.get("steps") or []
        if steps:
            lines.extend(["", "| Step | Action | Reason / summary |", "|---:|---|---|"])
            for step in steps:
                action = _compact(
                    step.get("action") or step.get("action_output"),
                    step_char_limit,
                )
                explanation = _compact(
                    step.get("reason") or step.get("summary"),
                    step_char_limit,
                )
                lines.append(
                    f"| {step['step_number']} | {_table_text(action)} | "
                    f"{_table_text(explanation)} |"
                )
        lines.append("")

    if run.get("runner_error"):
        lines.extend(
            [
                "## Runner error",
                "",
                "```text",
                str(run["runner_error"]).rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report_files(report: Mapping[str, Any], run_dir: Path) -> None:
    """Write one primary report plus convenient CSV sidecars."""
    _write_json(run_dir / "report.json", report)
    _write_text(
        run_dir / "report.md",
        render_markdown(
            report,
            int(
                report["run"]["config"]["reporting"][
                    "markdown_step_char_limit"
                ]
            ),
        ),
    )

    episode_rows = []
    for episode in report["episodes"]:
        episode_rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in episode.items()
                if key != "steps"
            }
        )
    episode_columns = [
        "task_template",
        "instance_id",
        "seed",
        "goal",
        "outcome",
        "score",
        "is_successful",
        "episode_length",
        "runtime_s",
        "difficulty",
        "optimal_steps",
        "tags",
        "exception",
        "aux_data",
    ]
    _write_csv(run_dir / "episodes.csv", episode_rows, episode_columns)

    task_columns = [
        "name",
        "difficulty",
        "optimal_steps",
        "tags",
        "attempted_episodes",
        "scored_episodes",
        "successful_episodes",
        "failed_episodes",
        "error_episodes",
        "task_success_rate",
        "mean_episode_length",
        "mean_runtime_s",
        "total_runtime_s",
    ]
    task_rows = []
    for row in report["breakdown"]["by_task"]:
        task_rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
        )
    _write_csv(run_dir / "tasks.csv", task_rows, task_columns)


def _base_config(
    args: argparse.Namespace,
    *,
    condition: str,
    android_world_root: Path,
    skill_info: Mapping[str, Any] | None,
    model_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model_info is None:
        model_config: dict[str, Any] = {
            "model_path": str(args.model_path),
            "backend": "vllm",
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
    else:
        model_config = dict(model_info)
    model_config["inference_error_policy"] = "mark_episode_as_error"

    return {
        "condition": condition,
        "runner": str(Path(sys.argv[0]).resolve()),
        "android_world_root": str(android_world_root),
        "model": _json_value(model_config),
        "suite": {
            "family": "android_world",
            "tasks": normalize_tasks(args.tasks),
            "n_task_combinations": args.n_task_combinations,
            "task_random_seed": args.task_random_seed,
            "fixed_task_seed": args.fixed_task_seed,
        },
        "android_runtime": {
            "adb_path": str(args.adb_path),
            "console_port": args.console_port,
            "grpc_port": args.grpc_port,
            "perform_emulator_setup": args.perform_emulator_setup,
            "transition_pause": args.transition_pause,
            "max_consecutive_infrastructure_errors": (
                args.max_consecutive_infrastructure_errors
            ),
        },
        "reporting": {
            "include_prompts": args.include_prompts,
            "markdown_step_char_limit": args.markdown_step_char_limit,
        },
        "skill": _json_value(dict(skill_info)) if skill_info else None,
    }


def _ensure_resume_compatible(run_dir: Path, config: Mapping[str, Any]) -> None:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    with config_path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    if existing.get("config_signature") != _config_signature(config):
        raise ValueError(
            f"{run_dir} contains checkpoints created with a different model, "
            "suite, runtime, or skill configuration. Use a new --run_dir."
        )


def list_available_tasks() -> int:
    android_world_root = _bootstrap_import_paths()
    task_metadata = _load_task_metadata(android_world_root)
    if not task_metadata:
        raise RuntimeError(
            "Could not read Android World task_metadata.json. "
            "Check ANDROID_WORLD_ROOT."
        )
    for name in sorted(task_metadata):
        print(name)
    print(f"\nTotal: {len(task_metadata)} task templates")
    return 0


_INFRASTRUCTURE_ERROR_MARKERS = (
    "Could not get a11y tree",
    "vLLM server inference failed",
    "vLLM inference failed",
    "DeepSeek API inference failed",
    "failed to connect to the emulator",
    "grpc_status:14",
    "StatusCode.UNAVAILABLE",
)


class _FailFastCheckpointer:
    """Persist an episode, then abort on repeated infrastructure failures."""

    def __init__(self, delegate: Any, limit: int) -> None:
        self.delegate = delegate
        self.limit = limit
        self.consecutive_errors = 0

    def save_episodes(
        self, task_episodes: list[dict[str, Any]], task_name: str
    ) -> None:
        # Save first so the final diagnostic report includes the episode that
        # tripped the threshold.
        self.delegate.save_episodes(task_episodes, task_name)
        for episode in task_episodes:
            exception = str(episode.get("exception_info") or "")
            is_infrastructure = any(
                marker.lower() in exception.lower()
                for marker in _INFRASTRUCTURE_ERROR_MARKERS
            )
            self.consecutive_errors = (
                self.consecutive_errors + 1 if is_infrastructure else 0
            )
        if self.limit and self.consecutive_errors >= self.limit:
            raise RuntimeError(
                "Aborting Android World after "
                f"{self.consecutive_errors} consecutive infrastructure "
                "errors. The last episode was saved. Fix the emulator/a11y/"
                "model backend and resume the same --run_dir."
            )

    def load(self, fields: list[str] | None = None) -> list[dict[str, Any]]:
        return self.delegate.load(fields)


def run_evaluation(
    args: argparse.Namespace,
    *,
    condition: str,
    model_factory: Callable[[argparse.Namespace], Any] = create_vllm,
    skill_info: Mapping[str, Any] | None = None,
    agent_factory: Callable[[Any, Any, Any], Any] | None = None,
    model_info: Mapping[str, Any] | None = None,
    backend_label: str | None = None,
    agent_name: str | None = None,
) -> int:
    """Execute a suite and always materialize reports from its checkpoints."""
    validate_args(args)
    args.tasks = normalize_tasks(args.tasks)
    android_world_root = _bootstrap_import_paths()

    from android_world import checkpointer as checkpointer_lib
    from android_world import registry
    from android_world import suite_utils
    from android_world.agents import t3a
    from android_world.env import env_launcher

    run_dir = _resolve_run_dir(args, condition)
    checkpoint_dir = run_dir / "checkpoints"
    config = _base_config(
        args,
        condition=condition,
        android_world_root=android_world_root,
        skill_info=skill_info,
        model_info=model_info,
    )
    config["config_signature"] = _config_signature(config)
    _ensure_resume_compatible(run_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", config)

    started_at = dt.datetime.now(dt.timezone.utc).astimezone()
    run_status = "initializing"
    runner_error: str | None = None
    env = None
    llm = None
    planned_episodes = 0
    caught_error: BaseException | None = None
    incremental_checkpointer = checkpointer_lib.IncrementalCheckpointer(
        str(checkpoint_dir)
    )
    checkpointer: Any = incremental_checkpointer
    if args.max_consecutive_infrastructure_errors:
        checkpointer = _FailFastCheckpointer(
            incremental_checkpointer,
            args.max_consecutive_infrastructure_errors,
        )

    print(f"Run directory: {run_dir}", flush=True)
    try:
        print("Loading Android environment...", flush=True)
        env = env_launcher.load_and_setup_env(
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            emulator_setup=args.perform_emulator_setup,
            adb_path=args.adb_path,
        )

        task_registry = registry.TaskRegistry()
        family = task_registry.get_registry(
            task_registry.ANDROID_WORLD_FAMILY
        )
        suite = suite_utils.create_suite(
            family,
            n_task_combinations=args.n_task_combinations,
            seed=args.task_random_seed,
            tasks=args.tasks,
            use_identical_params=args.fixed_task_seed,
            env=env,
        )
        suite.suite_family = task_registry.ANDROID_WORLD_FAMILY
        planned_episodes = sum(len(instances) for instances in suite.values())
        print(
            f"Selected {len(suite)} task templates / "
            f"{planned_episodes} episodes.",
            flush=True,
        )

        model_config = config["model"]
        model_identifier = (
            model_config.get("model_path")
            or model_config.get("model")
            or model_config.get("model_id")
            or "unknown"
        )
        resolved_backend_label = backend_label or (
            "vLLM"
            if model_config.get("backend") == "vllm"
            else str(model_config.get("backend") or "model")
        )
        print(
            f"Initializing {resolved_backend_label} backend: "
            f"{model_identifier}",
            flush=True,
        )
        llm = model_factory(args)
        resolved_agent_name = agent_name or (
            f"t3a_{_slug(str(model_config.get('backend') or 'model'))}_"
            f"{condition}"
        )
        if agent_factory is None:
            agent = t3a.T3A(env, llm, name=resolved_agent_name)
        else:
            agent = agent_factory(env, llm, t3a)
            agent.name = resolved_agent_name
        agent.transition_pause = args.transition_pause

        run_status = "running"
        suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer,
            demo_mode=False,
            return_full_episode_data=False,
        )
        run_status = "completed"
    except KeyboardInterrupt as exc:
        run_status = "interrupted"
        runner_error = "Interrupted by user."
        caught_error = exc
    except BaseException as exc:  # Save a diagnostic report before re-raising.
        run_status = "failed"
        runner_error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        caught_error = exc
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # pylint: disable=broad-exception-caught
                traceback.print_exc()

        raw_episodes = checkpointer.load()
        inference_stats = (
            llm.get_stats()
            if llm is not None and hasattr(llm, "get_stats")
            else {}
        )
        finished_at = dt.datetime.now(dt.timezone.utc).astimezone()
        report = build_report(
            raw_episodes,
            run_config=config,
            android_world_root=android_world_root,
            planned_episodes=planned_episodes,
            run_status=run_status,
            started_at=started_at,
            finished_at=finished_at,
            runner_error=runner_error,
            inference_stats=inference_stats,
            include_prompts=args.include_prompts,
        )
        write_report_files(report, run_dir)
        print(f"Readable report: {run_dir / 'report.md'}", flush=True)
        print(f"Structured report: {run_dir / 'report.json'}", flush=True)

    if caught_error is not None:
        raise caught_error
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        return list_available_tasks()
    return run_evaluation(args, condition="baseline")


if __name__ == "__main__":
    raise SystemExit(main())
