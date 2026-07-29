"""AndroidWorld trajectory distillation with ms-swift LoRA.

This module implements *cross-tokenizer behaviour distillation*:

1. Run a teacher model in the real AndroidWorld environment.
2. Keep successful trajectories and turn each T3A action/summary call into a
   supervised example.
3. Split by complete episode (never by adjacent steps) and train a LoRA adapter
   for the student with the vendored ``libs/ms-swift``.
4. Optionally merge LoRA and compare teacher/base/distilled AndroidWorld SR on
   the same held-out task instances.

The configured GLM-4.1 teacher and Qwen3.5 student do not share a tokenizer or
vocabulary. ms-swift's token-level GKD requires aligned token ids, so applying
GKD directly to this pair would compare unrelated vocabulary positions. The
offline teacher-trajectory method used here is the compatible sequence-level
distillation route.

Typical full run::

    python src/distillation_training.py \
        --tasks ContactsAddContact MarkorCreateNote \
        --teacher_rollouts_per_task 8 \
        --evaluate_after_training

Useful staged runs::

    python src/distillation_training.py --stage collect --tasks ContactsAddContact
    python src/distillation_training.py --stage prepare --tasks ContactsAddContact \
        --run_dir results/distillation/<run>
    python src/distillation_training.py --stage train --tasks ContactsAddContact \
        --run_dir results/distillation/<run>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ANDROID_WORLD_ROOT = REPO_ROOT / "libs" / "android_world"
MS_SWIFT_ROOT = REPO_ROOT / "libs" / "ms-swift"
TASK_RUNNER = SRC_ROOT / "task_runner_detail.py"
SWIFT_CLI_ENTRYPOINT = MS_SWIFT_ROOT / "swift" / "cli" / "main.py"

DEFAULT_TEACHER_MODEL = "/home/zmz/Workspace/models/glm4.1-9b"
DEFAULT_STUDENT_MODEL = "/home/zmz/Workspace/models/qwen3.5-0.8b"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "distillation"

DISTILLATION_METHOD = "cross_tokenizer_offline_androidworld_trajectory_distillation"
DIRECT_GKD_DISABLED_REASON = (
    "GLM-4.1 and Qwen3.5 use different tokenizers/vocabularies. The vendored "
    "ms-swift GKD implementation requires teacher and student response token "
    "ids to have the same meaning, so direct logit/token GKD is invalid for "
    "this model pair."
)


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved output locations for one distillation run."""

    run_dir: Path
    teacher_rollouts: Path
    dataset_dir: Path
    training_output: Path
    merged_model: Path
    teacher_evaluation: Path
    student_baseline_evaluation: Path
    student_evaluation: Path

    @property
    def train_dataset(self) -> Path:
        return self.dataset_dir / "train.jsonl"

    @property
    def validation_dataset(self) -> Path:
        return self.dataset_dir / "validation.jsonl"

    @property
    def dataset_manifest(self) -> Path:
        return self.dataset_dir / "dataset_manifest.json"

    @property
    def pipeline_manifest(self) -> Path:
        return self.run_dir / "distillation_manifest.json"

    @property
    def comparison_json(self) -> Path:
        return self.run_dir / "sr_comparison.json"

    @property
    def comparison_markdown(self) -> Path:
        return self.run_dir / "sr_comparison.md"


def _find_adb() -> str:
    """Use the same common Android SDK locations as the evaluation runner."""

    candidates = [
        Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools" / "adb",
        Path(os.environ.get("ANDROID_SDK_ROOT", ""))
        / "platform-tools"
        / "adb",
    ]
    if os.name == "nt":
        candidates = [path.with_suffix(".exe") for path in candidates]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return str(candidate)
    return "adb"


def build_parser() -> argparse.ArgumentParser:
    """Build the complete collection, preparation, training and eval CLI."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pipeline = parser.add_argument_group("Pipeline")
    pipeline.add_argument(
        "--stage",
        choices=("all", "collect", "prepare", "train"),
        default="all",
        help="Run the full pipeline or one resumable stage.",
    )
    pipeline.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Stable run directory. A timestamped directory is created by default.",
    )
    pipeline.add_argument(
        "--teacher_run_dir",
        type=Path,
        default=None,
        help="Reuse an existing task_runner_detail teacher run.",
    )
    pipeline.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Prepared train/validation JSONL directory.",
    )
    pipeline.add_argument(
        "--training_output_dir",
        type=Path,
        default=None,
        help="ms-swift LoRA checkpoint directory.",
    )
    pipeline.add_argument(
        "--merged_model_dir",
        type=Path,
        default=None,
        help="Directory for the merged student model.",
    )
    pipeline.add_argument(
        "--dry_run",
        action="store_true",
        help="Write a manifest and print commands without launching models.",
    )

    tasks = parser.add_argument_group("AndroidWorld tasks and trajectories")
    selection = tasks.add_mutually_exclusive_group()
    selection.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help="Task names separated by spaces or commas.",
    )
    selection.add_argument(
        "--all_tasks",
        action="store_true",
        help="Explicitly select every AndroidWorld task template.",
    )
    tasks.add_argument(
        "--list_tasks",
        action="store_true",
        help="List locally available AndroidWorld tasks and exit.",
    )
    tasks.add_argument("--teacher_rollouts_per_task", type=int, default=8)
    tasks.add_argument("--task_random_seed", type=int, default=30)
    tasks.add_argument(
        "--fixed_task_seed",
        action="store_true",
        help="Use identical task parameters for repeated teacher rollouts.",
    )
    tasks.add_argument("--validation_ratio", type=float, default=0.1)
    tasks.add_argument("--dataset_seed", type=int, default=42)
    tasks.add_argument(
        "--include_summaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also distil the T3A post-action summary calls.",
    )
    tasks.add_argument(
        "--include_failed_trajectories",
        action="store_true",
        help="Also train on failed teacher episodes (normally undesirable).",
    )
    tasks.add_argument(
        "--require_success_per_task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preparation if a selected task has no successful teacher episode.",
    )
    tasks.add_argument(
        "--deduplicate_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove exact duplicate prompt/response pairs.",
    )

    android = parser.add_argument_group("Android runtime")
    android.add_argument("--adb_path", default=_find_adb())
    android.add_argument("--console_port", type=int, default=5554)
    android.add_argument("--grpc_port", type=int, default=8554)
    android.add_argument("--perform_emulator_setup", action="store_true")
    android.add_argument("--transition_pause", type=float, default=None)
    android.add_argument(
        "--max_consecutive_infrastructure_errors", type=int, default=3
    )

    teacher = parser.add_argument_group("Teacher vLLM")
    teacher.add_argument("--teacher_model", default=DEFAULT_TEACHER_MODEL)
    teacher.add_argument("--teacher_tensor_parallel_size", type=int, default=1)
    teacher.add_argument(
        "--teacher_gpu_memory_utilization", type=float, default=0.9
    )
    teacher.add_argument("--teacher_max_model_len", type=int, default=None)
    teacher.add_argument("--teacher_temperature", type=float, default=0.0)
    teacher.add_argument("--teacher_top_p", type=float, default=0.95)
    teacher.add_argument("--teacher_max_tokens", type=int, default=512)

    training = parser.add_argument_group("ms-swift student LoRA")
    training.add_argument("--student_model", default=DEFAULT_STUDENT_MODEL)
    training.add_argument("--torch_dtype", default="bfloat16")
    training.add_argument("--num_train_epochs", type=float, default=3.0)
    training.add_argument("--learning_rate", type=float, default=1e-4)
    training.add_argument("--per_device_train_batch_size", type=int, default=1)
    training.add_argument("--per_device_eval_batch_size", type=int, default=1)
    training.add_argument("--gradient_accumulation_steps", type=int, default=8)
    training.add_argument("--max_length", type=int, default=8192)
    training.add_argument("--lora_rank", type=int, default=16)
    training.add_argument("--lora_alpha", type=int, default=32)
    training.add_argument("--lora_dropout", type=float, default=0.05)
    training.add_argument(
        "--target_modules", nargs="+", default=["all-linear"]
    )
    training.add_argument("--warmup_ratio", type=float, default=0.05)
    training.add_argument("--save_steps", type=int, default=50)
    training.add_argument("--eval_steps", type=int, default=50)
    training.add_argument("--logging_steps", type=int, default=5)
    training.add_argument("--save_total_limit", type=int, default=2)
    training.add_argument("--dataset_num_proc", type=int, default=4)
    training.add_argument("--dataloader_num_workers", type=int, default=4)
    training.add_argument(
        "--report_to", nargs="+", default=["tensorboard"]
    )
    training.add_argument("--deepspeed", default=None)
    training.add_argument("--resume_from_checkpoint", type=Path, default=None)
    training.add_argument("--cuda_visible_devices", default=None)
    training.add_argument("--nproc_per_node", type=int, default=None)
    training.add_argument(
        "--packing", action=argparse.BooleanOptionalAction, default=False
    )
    training.add_argument(
        "--merge_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge the last LoRA checkpoint into a standalone student model.",
    )

    evaluation = parser.add_argument_group("Held-out SR comparison")
    evaluation.add_argument(
        "--evaluate_after_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate teacher, base student and merged student on identical "
            "held-out instances."
        ),
    )
    evaluation.add_argument("--evaluation_rollouts_per_task", type=int, default=3)
    evaluation.add_argument("--evaluation_seed", type=int, default=130)
    evaluation.add_argument(
        "--student_eval_tensor_parallel_size", type=int, default=1
    )
    evaluation.add_argument(
        "--student_eval_gpu_memory_utilization", type=float, default=0.9
    )
    evaluation.add_argument("--student_eval_max_model_len", type=int, default=None)

    return parser


def normalize_tasks(raw_tasks: Sequence[str] | None) -> list[str] | None:
    """Accept both whitespace-separated and comma-separated task names."""

    if not raw_tasks:
        return None
    result: list[str] = []
    for item in raw_tasks:
        result.extend(part.strip() for part in item.split(",") if part.strip())
    return list(dict.fromkeys(result)) or None


def available_tasks() -> list[str]:
    """Read task names without importing AndroidWorld and its dependencies."""

    metadata_path = ANDROID_WORLD_ROOT / "android_world" / "task_metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read AndroidWorld task metadata: {metadata_path}"
        ) from exc
    return sorted(
        {
            str(row["task_name"])
            for row in rows
            if isinstance(row, Mapping) and row.get("task_name")
        }
    )


def resolve_selected_tasks(args: argparse.Namespace) -> list[str]:
    """Resolve and validate the explicit task selection."""

    known = available_tasks()
    selected = known if args.all_tasks else normalize_tasks(args.tasks)
    if not selected:
        raise ValueError(
            "Select tasks with --tasks TASK [TASK ...], or explicitly use "
            "--all_tasks. Full-suite collection is never selected implicitly."
        )
    unknown = sorted(set(selected) - set(known))
    if unknown:
        raise ValueError(
            "Unknown AndroidWorld task(s): "
            + ", ".join(unknown)
            + ". Use --list_tasks to inspect valid names."
        )
    return selected


def validate_args(args: argparse.Namespace) -> None:
    """Validate values before launching any expensive process."""

    positive_ints = {
        "--teacher_rollouts_per_task": args.teacher_rollouts_per_task,
        "--teacher_tensor_parallel_size": args.teacher_tensor_parallel_size,
        "--teacher_max_tokens": args.teacher_max_tokens,
        "--per_device_train_batch_size": args.per_device_train_batch_size,
        "--per_device_eval_batch_size": args.per_device_eval_batch_size,
        "--gradient_accumulation_steps": args.gradient_accumulation_steps,
        "--max_length": args.max_length,
        "--lora_rank": args.lora_rank,
        "--lora_alpha": args.lora_alpha,
        "--save_steps": args.save_steps,
        "--eval_steps": args.eval_steps,
        "--logging_steps": args.logging_steps,
        "--save_total_limit": args.save_total_limit,
        "--evaluation_rollouts_per_task": args.evaluation_rollouts_per_task,
        "--student_eval_tensor_parallel_size": (
            args.student_eval_tensor_parallel_size
        ),
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be at least 1.")
    if args.dataset_num_proc < 0 or args.dataloader_num_workers < 0:
        raise ValueError("Dataset/dataloader worker counts cannot be negative.")
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation_ratio must be in [0, 1).")
    if args.num_train_epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("Epoch count and learning rate must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout must be in [0, 1).")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup_ratio must be in [0, 1).")
    for name, value in (
        ("--teacher_gpu_memory_utilization", args.teacher_gpu_memory_utilization),
        (
            "--student_eval_gpu_memory_utilization",
            args.student_eval_gpu_memory_utilization,
        ),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1].")
    if args.evaluate_after_training and not args.merge_lora:
        raise ValueError(
            "--evaluate_after_training requires --merge_lora because "
            "task_runner_detail expects a standalone model directory."
        )
    if args.stage not in {"all", "train"} and args.evaluate_after_training:
        raise ValueError(
            "--evaluate_after_training is only valid with --stage all or train."
        )


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    """Create deterministic child paths beneath the selected run directory."""

    if args.run_dir is None:
        stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_OUTPUT_ROOT / f"glm_to_qwen_androidworld_{stamp}"
    else:
        run_dir = args.run_dir
    run_dir = run_dir.expanduser().resolve()
    return PipelinePaths(
        run_dir=run_dir,
        teacher_rollouts=(
            args.teacher_run_dir.expanduser().resolve()
            if args.teacher_run_dir
            else run_dir / "teacher_rollouts"
        ),
        dataset_dir=(
            args.dataset_dir.expanduser().resolve()
            if args.dataset_dir
            else run_dir / "dataset"
        ),
        training_output=(
            args.training_output_dir.expanduser().resolve()
            if args.training_output_dir
            else run_dir / "lora"
        ),
        merged_model=(
            args.merged_model_dir.expanduser().resolve()
            if args.merged_model_dir
            else run_dir / "merged_model"
        ),
        teacher_evaluation=run_dir / "evaluation" / "teacher",
        student_baseline_evaluation=(
            run_dir / "evaluation" / "student_baseline"
        ),
        student_evaluation=run_dir / "evaluation" / "student_distilled",
    )


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend((name, str(value)))


def build_teacher_collection_command(
    args: argparse.Namespace, paths: PipelinePaths, tasks: Sequence[str]
) -> list[str]:
    """Build the isolated teacher rollout process command."""

    command = [
        sys.executable,
        str(TASK_RUNNER),
        "--model_path",
        str(args.teacher_model),
        "--tasks",
        *tasks,
        "--n_task_combinations",
        str(args.teacher_rollouts_per_task),
        "--task_random_seed",
        str(args.task_random_seed),
        "--run_dir",
        str(paths.teacher_rollouts),
        "--include_prompts",
        "--adb_path",
        str(args.adb_path),
        "--console_port",
        str(args.console_port),
        "--grpc_port",
        str(args.grpc_port),
        "--max_consecutive_infrastructure_errors",
        str(args.max_consecutive_infrastructure_errors),
        "--tensor_parallel_size",
        str(args.teacher_tensor_parallel_size),
        "--gpu_memory_utilization",
        str(args.teacher_gpu_memory_utilization),
        "--temperature",
        str(args.teacher_temperature),
        "--top_p",
        str(args.teacher_top_p),
        "--max_tokens",
        str(args.teacher_max_tokens),
    ]
    _append_option(command, "--transition_pause", args.transition_pause)
    _append_option(command, "--max_model_len", args.teacher_max_model_len)
    if args.perform_emulator_setup:
        command.append("--perform_emulator_setup")
    if args.fixed_task_seed:
        command.append("--fixed_task_seed")
    return command


def build_swift_sft_command(
    args: argparse.Namespace,
    paths: PipelinePaths,
    *,
    validation_exists: bool,
) -> list[str]:
    """Build ms-swift LoRA SFT for raw T3A prompts and teacher completions."""

    command = [
        sys.executable,
        str(SWIFT_CLI_ENTRYPOINT),
        "sft",
        "--model",
        str(args.student_model),
        "--tuner_type",
        "lora",
        "--dataset",
        str(paths.train_dataset),
        "--split_dataset_ratio",
        "0",
        "--use_chat_template",
        "false",
        "--add_non_thinking_prefix",
        "false",
        "--torch_dtype",
        str(args.torch_dtype),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        *args.target_modules,
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_length",
        str(args.max_length),
        "--truncation_strategy",
        "left",
        "--packing",
        str(args.packing).lower(),
        "--output_dir",
        str(paths.training_output),
        "--add_version",
        "false",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--logging_steps",
        str(args.logging_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--dataset_num_proc",
        str(args.dataset_num_proc),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--report_to",
        *args.report_to,
        "--seed",
        str(args.dataset_seed),
    ]
    if validation_exists:
        command.extend(
            (
                "--val_dataset",
                str(paths.validation_dataset),
                "--eval_strategy",
                "steps",
                "--eval_steps",
                str(args.eval_steps),
            )
        )
    else:
        command.extend(("--eval_strategy", "no"))
    _append_option(command, "--deepspeed", args.deepspeed)
    _append_option(
        command, "--resume_from_checkpoint", args.resume_from_checkpoint
    )
    return command


def build_merge_command(
    adapter_path: Path, merged_model_path: Path
) -> list[str]:
    """Build the ms-swift adapter export/merge command."""

    return [
        sys.executable,
        str(SWIFT_CLI_ENTRYPOINT),
        "export",
        "--adapters",
        str(adapter_path),
        "--merge_lora",
        "true",
        "--output_dir",
        str(merged_model_path),
    ]


def build_evaluation_command(
    args: argparse.Namespace,
    tasks: Sequence[str],
    *,
    model_path: str | Path,
    run_dir: Path,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> list[str]:
    """Build a deterministic held-out AndroidWorld evaluation command."""

    command = [
        sys.executable,
        str(TASK_RUNNER),
        "--model_path",
        str(model_path),
        "--tasks",
        *tasks,
        "--n_task_combinations",
        str(args.evaluation_rollouts_per_task),
        "--task_random_seed",
        str(args.evaluation_seed),
        "--run_dir",
        str(run_dir),
        "--adb_path",
        str(args.adb_path),
        "--console_port",
        str(args.console_port),
        "--grpc_port",
        str(args.grpc_port),
        "--max_consecutive_infrastructure_errors",
        str(args.max_consecutive_infrastructure_errors),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--temperature",
        "0.0",
        "--top_p",
        str(args.teacher_top_p),
        "--max_tokens",
        str(args.teacher_max_tokens),
    ]
    _append_option(command, "--transition_pause", args.transition_pause)
    _append_option(command, "--max_model_len", max_model_len)
    return command


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_from_step(
    *,
    prompt: Any,
    response: Any,
    sample_kind: str,
    episode: Mapping[str, Any],
    episode_key: str,
    step_number: Any,
    teacher_model: str,
) -> dict[str, Any] | None:
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if not isinstance(response, str) or not response.strip():
        return None
    sample_hash = _sha256_json(
        {
            "prompt": prompt,
            "response": response,
            "kind": sample_kind,
        }
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "sample_id": sample_hash,
            "distillation_method": DISTILLATION_METHOD,
            "teacher_model": teacher_model,
            "task_template": episode.get("task_template"),
            "episode_key": episode_key,
            "instance_id": episode.get("instance_id"),
            "task_seed": episode.get("seed"),
            "teacher_success": bool(episode.get("is_successful")),
            "teacher_score": episode.get("score"),
            "sample_kind": sample_kind,
            "step_number": step_number,
        },
    }


def _episode_key(episode: Mapping[str, Any], fallback_index: int) -> str:
    return (
        f"{episode.get('task_template', 'unknown')}:"
        f"{episode.get('instance_id', fallback_index)}:"
        f"{episode.get('seed', 'none')}"
    )


def _validation_count(group_count: int, ratio: float) -> int:
    if ratio <= 0 or group_count < 2:
        return 0
    count = max(1, round(group_count * ratio))
    return min(count, group_count - 1)


def prepare_distillation_dataset(
    report_path: Path,
    dataset_dir: Path,
    *,
    selected_tasks: Sequence[str],
    teacher_model: str,
    student_model: str,
    validation_ratio: float,
    dataset_seed: int,
    include_summaries: bool,
    include_failed_trajectories: bool,
    require_success_per_task: bool,
    deduplicate_samples: bool,
) -> dict[str, Any]:
    """Convert a detailed teacher report into leakage-safe ms-swift JSONL."""

    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read teacher report: {report_path}") from exc
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{report_path} does not contain an episodes list.")
    run_info = report.get("run")
    run_config = run_info.get("config") if isinstance(run_info, Mapping) else None
    model_config = (
        run_config.get("model") if isinstance(run_config, Mapping) else None
    )
    reported_teacher = (
        model_config.get("model_path")
        if isinstance(model_config, Mapping)
        else None
    )
    if reported_teacher is not None and str(reported_teacher) != str(teacher_model):
        raise ValueError(
            "Teacher report model does not match --teacher_model: "
            f"{reported_teacher!s} != {teacher_model!s}. Pass the report's "
            "model path explicitly or collect a new teacher run."
        )

    selected_set = set(selected_tasks)
    episodes_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        task_name = str(episode.get("task_template") or "")
        if task_name in selected_set:
            episodes_by_task[task_name].append(episode)

    missing_report_tasks = [
        task for task in selected_tasks if not episodes_by_task.get(task)
    ]
    if missing_report_tasks:
        raise ValueError(
            "Teacher report has no episodes for selected task(s): "
            + ", ".join(missing_report_tasks)
        )

    successful_counts = {
        task: sum(bool(row.get("is_successful")) for row in rows)
        for task, rows in episodes_by_task.items()
    }
    missing_success = [
        task for task in selected_tasks if successful_counts.get(task, 0) == 0
    ]
    if require_success_per_task and missing_success:
        raise ValueError(
            "No successful teacher trajectory for: "
            + ", ".join(missing_success)
            + ". Increase --teacher_rollouts_per_task or explicitly use "
            "--no-require_success_per_task. Failed trajectories remain excluded "
            "unless --include_failed_trajectories is also set."
        )

    grouped_samples: dict[str, dict[str, Any]] = {}
    seen_pairs: set[str] = set()
    prompt_responses: dict[str, set[str]] = defaultdict(set)
    skipped_duplicates = 0
    skipped_incomplete = 0
    included_failed_episodes = 0

    for task_name in selected_tasks:
        for episode_index, episode in enumerate(episodes_by_task[task_name]):
            is_successful = bool(episode.get("is_successful"))
            if not is_successful and not include_failed_trajectories:
                continue
            if not is_successful:
                included_failed_episodes += 1
            key = _episode_key(episode, episode_index)
            samples: list[dict[str, Any]] = []
            steps = episode.get("steps")
            if not isinstance(steps, list):
                continue
            for default_step, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    continue
                step_number = step.get("step_number", default_step)
                candidates = [
                    (
                        step.get("action_prompt"),
                        step.get("action_output"),
                        "action",
                    )
                ]
                if include_summaries:
                    candidates.append(
                        (
                            step.get("summary_prompt"),
                            step.get("summary"),
                            "summary",
                        )
                    )
                for prompt, response, sample_kind in candidates:
                    sample = _sample_from_step(
                        prompt=prompt,
                        response=response,
                        sample_kind=sample_kind,
                        episode=episode,
                        episode_key=key,
                        step_number=step_number,
                        teacher_model=teacher_model,
                    )
                    if sample is None:
                        skipped_incomplete += 1
                        continue
                    messages = sample["messages"]
                    prompt_text = messages[0]["content"]
                    response_text = messages[1]["content"]
                    pair_hash = _sha256_json([prompt_text, response_text])
                    prompt_hash = hashlib.sha256(
                        prompt_text.encode("utf-8")
                    ).hexdigest()
                    prompt_responses[prompt_hash].add(
                        hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                    )
                    if deduplicate_samples and pair_hash in seen_pairs:
                        skipped_duplicates += 1
                        continue
                    seen_pairs.add(pair_hash)
                    samples.append(sample)
            if samples:
                grouped_samples[key] = {
                    "task": task_name,
                    "samples": samples,
                    "teacher_success": is_successful,
                }

    if not grouped_samples:
        raise ValueError(
            "No trainable action/summary prompt pairs were found. The teacher "
            "report must have been generated with --include_prompts."
        )

    groups_by_task: dict[str, list[str]] = defaultdict(list)
    for key, group in grouped_samples.items():
        groups_by_task[group["task"]].append(key)
    missing_trainable_tasks = [
        task for task in selected_tasks if not groups_by_task.get(task)
    ]
    if require_success_per_task and missing_trainable_tasks:
        raise ValueError(
            "Selected task(s) have no complete trainable prompt/response pairs: "
            + ", ".join(missing_trainable_tasks)
            + ". Regenerate the teacher report with --include_prompts and "
            "collect at least one successful trajectory for every task."
        )

    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    for task_name in selected_tasks:
        keys = sorted(groups_by_task.get(task_name, []))
        stable_seed = int(
            hashlib.sha256(
                f"{dataset_seed}:{task_name}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        task_rng = random.Random(stable_seed)
        task_rng.shuffle(keys)
        val_count = _validation_count(len(keys), validation_ratio)
        validation_groups.update(keys[:val_count])
        train_groups.update(keys[val_count:])

    train_rows = [
        sample
        for key in sorted(train_groups)
        for sample in grouped_samples[key]["samples"]
    ]
    validation_rows = [
        sample
        for key in sorted(validation_groups)
        for sample in grouped_samples[key]["samples"]
    ]
    random.Random(dataset_seed).shuffle(train_rows)
    random.Random(dataset_seed + 1).shuffle(validation_rows)

    if not train_rows:
        raise ValueError(
            "The split produced an empty training dataset. Reduce "
            "--validation_ratio or collect more teacher episodes."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    if validation_rows:
        _write_jsonl(validation_path, validation_rows)
    elif validation_path.exists():
        validation_path.unlink()

    def count_samples(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
        return Counter(
            str(row["metadata"]["sample_kind"])
            for row in rows
            if isinstance(row.get("metadata"), Mapping)
        )

    per_task: dict[str, Any] = {}
    for task_name in selected_tasks:
        task_train = {
            key for key in train_groups if grouped_samples[key]["task"] == task_name
        }
        task_val = {
            key
            for key in validation_groups
            if grouped_samples[key]["task"] == task_name
        }
        task_samples = [
            sample
            for key in task_train | task_val
            for sample in grouped_samples[key]["samples"]
        ]
        kinds = count_samples(task_samples)
        per_task[task_name] = {
            "report_episodes": len(episodes_by_task[task_name]),
            "successful_teacher_episodes": successful_counts[task_name],
            "included_episode_groups": len(task_train | task_val),
            "train_episode_groups": len(task_train),
            "validation_episode_groups": len(task_val),
            "action_samples": kinds["action"],
            "summary_samples": kinds["summary"],
        }

    train_kinds = count_samples(train_rows)
    validation_kinds = count_samples(validation_rows)
    conflicts = sum(
        len(responses) > 1 for responses in prompt_responses.values()
    )
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "distillation": {
            "method": DISTILLATION_METHOD,
            "teacher_model": teacher_model,
            "student_model": student_model,
            "direct_gkd_used": False,
            "direct_gkd_disabled_reason": DIRECT_GKD_DISABLED_REASON,
            "training_objective": (
                "Supervised next-response imitation of successful AndroidWorld "
                "T3A teacher action and optional summary calls."
            ),
        },
        "source": {
            "teacher_report": str(report_path.resolve()),
            "teacher_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "selected_tasks": list(selected_tasks),
            "include_failed_trajectories": include_failed_trajectories,
            "include_summaries": include_summaries,
        },
        "split": {
            "unit": "episode",
            "validation_ratio": validation_ratio,
            "dataset_seed": dataset_seed,
            "train_episode_groups": sorted(train_groups),
            "validation_episode_groups": sorted(validation_groups),
            "episode_overlap": sorted(train_groups & validation_groups),
        },
        "counts": {
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "train_action_samples": train_kinds["action"],
            "train_summary_samples": train_kinds["summary"],
            "validation_action_samples": validation_kinds["action"],
            "validation_summary_samples": validation_kinds["summary"],
            "skipped_exact_duplicates": skipped_duplicates,
            "skipped_incomplete_pairs": skipped_incomplete,
            "prompts_with_conflicting_teacher_responses": conflicts,
            "included_failed_episodes": included_failed_episodes,
        },
        "tasks": per_task,
        "files": {
            "train": str(train_path.resolve()),
            "validation": (
                str(validation_path.resolve()) if validation_rows else None
            ),
        },
    }
    _write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def _command_string(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _swift_environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(MS_SWIFT_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(MS_SWIFT_ROOT), old_pythonpath))
    )
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.nproc_per_node:
        env["NPROC_PER_NODE"] = str(args.nproc_per_node)
    return env


def run_command(
    command: Sequence[str],
    *,
    label: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run one visible, fail-fast pipeline subprocess."""

    print(f"\n[{label}]\n{_command_string(command)}", flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=str(REPO_ROOT),
        env=dict(env) if env is not None else None,
        check=True,
    )


def find_latest_adapter(training_output: Path) -> Path:
    """Find the last numeric ms-swift LoRA checkpoint, with root fallback."""

    candidates: list[tuple[int, Path]] = []
    if (training_output / "adapter_config.json").is_file():
        candidates.append((-1, training_output))
    if training_output.is_dir():
        for child in training_output.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", child.name)
            if (
                match
                and child.is_dir()
                and (child / "adapter_config.json").is_file()
            ):
                candidates.append((int(match.group(1)), child))
    if not candidates:
        raise RuntimeError(
            f"No LoRA adapter checkpoint found beneath {training_output}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _read_report(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read evaluation report: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation report is not an object: {path}")
    return value


def _task_metric_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    breakdown = report.get("breakdown")
    rows = breakdown.get("by_task", []) if isinstance(breakdown, Mapping) else []
    return {
        str(row["name"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("name")
    }


def build_sr_comparison(
    teacher_report_path: Path,
    student_report_path: Path,
    *,
    baseline_report_path: Path | None = None,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Create a readable base-vs-distilled-vs-teacher held-out SR report."""

    teacher = _read_report(teacher_report_path)
    student = _read_report(student_report_path)
    baseline = (
        _read_report(baseline_report_path)
        if baseline_report_path is not None
        else None
    )
    teacher_summary = teacher.get("summary", {})
    student_summary = student.get("summary", {})
    baseline_summary = baseline.get("summary", {}) if baseline else {}
    teacher_tasks = _task_metric_map(teacher)
    student_tasks = _task_metric_map(student)
    baseline_tasks = _task_metric_map(baseline) if baseline else {}

    def number(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    rows = []
    for task_name in sorted(
        set(teacher_tasks) | set(student_tasks) | set(baseline_tasks)
    ):
        teacher_sr = number(teacher_tasks.get(task_name, {}).get("task_success_rate"))
        student_sr = number(student_tasks.get(task_name, {}).get("task_success_rate"))
        baseline_sr = number(
            baseline_tasks.get(task_name, {}).get("task_success_rate")
        )
        rows.append(
            {
                "task": task_name,
                "teacher_sr": teacher_sr,
                "base_student_sr": baseline_sr,
                "student_sr": student_sr,
                "student_minus_base": (
                    round(student_sr - baseline_sr, 4)
                    if baseline_sr is not None and student_sr is not None
                    else None
                ),
                "student_minus_teacher": (
                    round(student_sr - teacher_sr, 4)
                    if teacher_sr is not None and student_sr is not None
                    else None
                ),
            }
        )

    # task_runner_detail names the episode-weighted (micro) metric
    # ``task_success_rate`` and separately exposes macro_task_success_rate.
    teacher_micro = number(teacher_summary.get("task_success_rate"))
    student_micro = number(student_summary.get("task_success_rate"))
    baseline_micro = number(baseline_summary.get("task_success_rate"))
    teacher_macro = number(teacher_summary.get("macro_task_success_rate"))
    student_macro = number(student_summary.get("macro_task_success_rate"))
    baseline_macro = number(
        baseline_summary.get("macro_task_success_rate")
    )
    comparison = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "fairness": (
            "Teacher, base student and distilled student were evaluated with "
            "the same selected tasks, task seed, rollout count and "
            "deterministic decoding."
        ),
        "teacher_report": str(teacher_report_path.resolve()),
        "base_student_report": (
            str(baseline_report_path.resolve())
            if baseline_report_path is not None
            else None
        ),
        "student_report": str(student_report_path.resolve()),
        "overall": {
            "teacher_micro_sr": teacher_micro,
            "base_student_micro_sr": baseline_micro,
            "student_micro_sr": student_micro,
            "micro_sr_gain_over_base": (
                round(student_micro - baseline_micro, 4)
                if baseline_micro is not None and student_micro is not None
                else None
            ),
            "micro_sr_gap": (
                round(student_micro - teacher_micro, 4)
                if teacher_micro is not None and student_micro is not None
                else None
            ),
            "teacher_macro_sr": teacher_macro,
            "base_student_macro_sr": baseline_macro,
            "student_macro_sr": student_macro,
            "macro_sr_gain_over_base": (
                round(student_macro - baseline_macro, 4)
                if baseline_macro is not None and student_macro is not None
                else None
            ),
            "macro_sr_gap": (
                round(student_macro - teacher_macro, 4)
                if teacher_macro is not None and student_macro is not None
                else None
            ),
        },
        "by_task": rows,
    }
    _write_json(output_json, comparison)

    def percentage(value: Any) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.2f}%"

    markdown = [
        "# AndroidWorld Distillation SR Comparison",
        "",
        (
            "Teacher, base student, and distilled student use identical "
            "held-out tasks, generated task instances, rollout counts, and "
            "deterministic decoding."
        ),
        "",
        "## Overall",
        "",
        (
            "| Metric | Teacher | Base student | Distilled student | "
            "Distilled - Base | Distilled - Teacher |"
        ),
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Micro SR | {percentage(teacher_micro)} | "
            f"{percentage(baseline_micro)} | "
            f"{percentage(student_micro)} | "
            f"{percentage(comparison['overall']['micro_sr_gain_over_base'])} | "
            f"{percentage(comparison['overall']['micro_sr_gap'])} |"
        ),
        (
            f"| Macro task SR | {percentage(teacher_macro)} | "
            f"{percentage(baseline_macro)} | "
            f"{percentage(student_macro)} | "
            f"{percentage(comparison['overall']['macro_sr_gain_over_base'])} | "
            f"{percentage(comparison['overall']['macro_sr_gap'])} |"
        ),
        "",
        "## Per task",
        "",
        (
            "| Task | Teacher SR | Base student SR | Distilled student SR | "
            "Distilled - Base | Distilled - Teacher |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['task']} | {percentage(row['teacher_sr'])} | "
            f"{percentage(row['base_student_sr'])} | "
            f"{percentage(row['student_sr'])} | "
            f"{percentage(row['student_minus_base'])} | "
            f"{percentage(row['student_minus_teacher'])} |"
        )
    _atomic_write_text(output_markdown, "\n".join(markdown) + "\n")
    return comparison


def _base_manifest(
    args: argparse.Namespace,
    paths: PipelinePaths,
    tasks: Sequence[str],
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "planned",
        "stage": args.stage,
        "distillation": {
            "method": DISTILLATION_METHOD,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "direct_gkd_used": False,
            "direct_gkd_disabled_reason": DIRECT_GKD_DISABLED_REASON,
        },
        "tasks": list(tasks),
        "paths": _json_safe(paths.__dict__),
        "arguments": _json_safe(vars(args)),
        "commands": _json_safe(commands),
    }


def _print_dry_run(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    print(f"Dry-run manifest: {manifest_path}")
    commands = manifest.get("commands", {})
    for label, command in commands.items():
        if command:
            print(f"\n[{label}]\n{_command_string(command)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        names = available_tasks()
        print("\n".join(names))
        print(f"\nTotal: {len(names)} task templates")
        return 0

    try:
        validate_args(args)
        tasks = resolve_selected_tasks(args)
    except ValueError as exc:
        parser.error(str(exc))

    paths = resolve_paths(args)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    teacher_command = build_teacher_collection_command(args, paths, tasks)
    planned_training_command = build_swift_sft_command(
        args, paths, validation_exists=True
    )
    planned_merge_command = build_merge_command(
        Path("<latest-lora-checkpoint>"), paths.merged_model
    )
    teacher_eval_command = build_evaluation_command(
        args,
        tasks,
        model_path=args.teacher_model,
        run_dir=paths.teacher_evaluation,
        tensor_parallel_size=args.teacher_tensor_parallel_size,
        gpu_memory_utilization=args.teacher_gpu_memory_utilization,
        max_model_len=args.teacher_max_model_len,
    )
    student_baseline_eval_command = build_evaluation_command(
        args,
        tasks,
        model_path=args.student_model,
        run_dir=paths.student_baseline_evaluation,
        tensor_parallel_size=args.student_eval_tensor_parallel_size,
        gpu_memory_utilization=args.student_eval_gpu_memory_utilization,
        max_model_len=args.student_eval_max_model_len,
    )
    student_eval_command = build_evaluation_command(
        args,
        tasks,
        model_path=paths.merged_model,
        run_dir=paths.student_evaluation,
        tensor_parallel_size=args.student_eval_tensor_parallel_size,
        gpu_memory_utilization=args.student_eval_gpu_memory_utilization,
        max_model_len=args.student_eval_max_model_len,
    )
    commands = {
        "collect_teacher": teacher_command,
        "train_lora": planned_training_command,
        "merge_lora": planned_merge_command if args.merge_lora else None,
        "evaluate_teacher": (
            teacher_eval_command if args.evaluate_after_training else None
        ),
        "evaluate_base_student": (
            student_baseline_eval_command
            if args.evaluate_after_training
            else None
        ),
        "evaluate_student": (
            student_eval_command if args.evaluate_after_training else None
        ),
    }
    manifest = _base_manifest(args, paths, tasks, commands)
    _write_json(paths.pipeline_manifest, manifest)

    if args.dry_run:
        manifest["status"] = "dry_run"
        _write_json(paths.pipeline_manifest, manifest)
        _print_dry_run(manifest, paths.pipeline_manifest)
        return 0

    try:
        if args.stage in {"all", "collect"}:
            if not TASK_RUNNER.is_file():
                raise FileNotFoundError(f"Missing task runner: {TASK_RUNNER}")
            manifest["status"] = "collecting_teacher_trajectories"
            _write_json(paths.pipeline_manifest, manifest)
            run_command(teacher_command, label="teacher AndroidWorld collection")

        dataset_manifest: Mapping[str, Any] | None = None
        if args.stage in {"all", "prepare"}:
            manifest["status"] = "preparing_dataset"
            _write_json(paths.pipeline_manifest, manifest)
            dataset_manifest = prepare_distillation_dataset(
                paths.teacher_rollouts / "report.json",
                paths.dataset_dir,
                selected_tasks=tasks,
                teacher_model=str(args.teacher_model),
                student_model=str(args.student_model),
                validation_ratio=args.validation_ratio,
                dataset_seed=args.dataset_seed,
                include_summaries=args.include_summaries,
                include_failed_trajectories=args.include_failed_trajectories,
                require_success_per_task=args.require_success_per_task,
                deduplicate_samples=args.deduplicate_samples,
            )
            manifest["dataset"] = dataset_manifest
            _write_json(paths.pipeline_manifest, manifest)

        if args.stage in {"all", "train"}:
            if not SWIFT_CLI_ENTRYPOINT.is_file():
                raise FileNotFoundError(
                    f"Missing vendored ms-swift CLI entrypoint: "
                    f"{SWIFT_CLI_ENTRYPOINT}"
                )
            if not paths.train_dataset.is_file():
                raise FileNotFoundError(
                    f"Prepared dataset not found: {paths.train_dataset}. "
                    "Run --stage prepare first or set --dataset_dir."
                )
            validation_exists = (
                paths.validation_dataset.is_file()
                and paths.validation_dataset.stat().st_size > 0
            )
            training_command = build_swift_sft_command(
                args, paths, validation_exists=validation_exists
            )
            manifest["commands"]["train_lora"] = training_command
            manifest["status"] = "training_lora"
            _write_json(paths.pipeline_manifest, manifest)
            run_command(
                training_command,
                label="ms-swift student LoRA training",
                env=_swift_environment(args),
            )

            adapter_path = find_latest_adapter(paths.training_output)
            manifest["latest_adapter"] = str(adapter_path.resolve())

            if args.merge_lora:
                if paths.merged_model.exists():
                    raise FileExistsError(
                        f"Merged model path already exists: "
                        f"{paths.merged_model}. ms-swift export refuses to "
                        "overwrite it; choose --merged_model_dir or remove the "
                        "stale export explicitly."
                    )
                merge_command = build_merge_command(
                    adapter_path, paths.merged_model
                )
                manifest["commands"]["merge_lora"] = merge_command
                manifest["status"] = "merging_lora"
                _write_json(paths.pipeline_manifest, manifest)
                run_command(
                    merge_command,
                    label="merge LoRA into standalone student",
                    env=_swift_environment(args),
                )

            if args.evaluate_after_training:
                manifest["status"] = "evaluating_held_out_sr"
                _write_json(paths.pipeline_manifest, manifest)
                run_command(
                    teacher_eval_command,
                    label="held-out teacher AndroidWorld evaluation",
                )
                run_command(
                    student_baseline_eval_command,
                    label="held-out base student AndroidWorld evaluation",
                )
                run_command(
                    student_eval_command,
                    label="held-out distilled student AndroidWorld evaluation",
                )
                comparison = build_sr_comparison(
                    paths.teacher_evaluation / "report.json",
                    paths.student_evaluation / "report.json",
                    baseline_report_path=(
                        paths.student_baseline_evaluation / "report.json"
                    ),
                    output_json=paths.comparison_json,
                    output_markdown=paths.comparison_markdown,
                )
                manifest["sr_comparison"] = comparison

        manifest["status"] = "completed"
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
        _write_json(paths.pipeline_manifest, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
        manifest["error"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        _write_json(paths.pipeline_manifest, manifest)
        raise

    print(f"\nDistillation manifest: {paths.pipeline_manifest}")
    if paths.dataset_manifest.is_file():
        print(f"Dataset manifest: {paths.dataset_manifest}")
    if paths.comparison_markdown.is_file():
        print(f"Readable SR comparison: {paths.comparison_markdown}")
    elif args.merge_lora and args.stage in {"all", "train"}:
        print(f"Merged student model: {paths.merged_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
