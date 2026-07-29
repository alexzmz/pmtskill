"""AndroidWorld trajectory distillation with ms-swift LoRA.

This module implements *cross-tokenizer behaviour distillation*:

1. Run a teacher model in the real AndroidWorld environment.
2. Keep successful trajectories and turn each T3A action/summary call into a
   supervised example.
3. Split by complete episode (never by adjacent steps) and train a LoRA adapter
   for the student with the vendored ``libs/ms-swift``.
4. Merge LoRA and compare teacher/base/distilled AndroidWorld SR on the same
   held-out task instances.
5. Evaluate retained checkpoints at a configurable step interval, write a
   human-readable improvement curve, and mirror loss/SR/pipeline state into a
   persistent TensorBoard server.

The configured GLM-4.1 teacher and Qwen3.5 student do not share a tokenizer or
vocabulary. ms-swift's token-level GKD requires aligned token ids, so applying
GKD directly to this pair would compare unrelated vocabulary positions. The
offline teacher-trajectory method used here is the compatible sequence-level
distillation route.

Typical full run::

    python src/distillation_training.py \
        --tasks ContactsAddContact MarkorCreateNote \
        --teacher_rollouts_per_task 8 \
        --sr_eval_interval_steps 100

The default TensorBoard URL is ``http://127.0.0.1:6006``. Periodic SR is
evaluated after the training process exits so vLLM does not contend with the
trainer for the same GPU; each point still corresponds to its saved checkpoint.

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
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
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
    periodic_evaluations: Path
    tensorboard_dir: Path

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

    @property
    def training_summary_json(self) -> Path:
        return self.run_dir / "training_summary.json"

    @property
    def training_summary_markdown(self) -> Path:
        return self.run_dir / "training_summary.md"

    @property
    def checkpoint_sr_json(self) -> Path:
        return self.run_dir / "checkpoint_sr_history.json"

    @property
    def checkpoint_sr_csv(self) -> Path:
        return self.run_dir / "checkpoint_sr_history.csv"

    @property
    def checkpoint_sr_markdown(self) -> Path:
        return self.run_dir / "checkpoint_sr_history.md"

    @property
    def pipeline_events(self) -> Path:
        return self.tensorboard_dir / "pipeline" / "metrics.jsonl"

    @property
    def result_json(self) -> Path:
        return self.run_dir / "distillation_result.json"

    @property
    def result_markdown(self) -> Path:
        return self.run_dir / "distillation_result.md"


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
    training.add_argument(
        "--save_total_limit",
        type=int,
        default=0,
        help=(
            "Maximum retained checkpoints; 0 keeps all checkpoints so periodic "
            "SR evaluation can reconstruct the complete curve."
        ),
    )
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
        default=None,
        help=(
            "Evaluate teacher, base student and merged student on identical "
            "held-out instances. Defaults to enabled for train/all when LoRA "
            "merging is enabled."
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

    monitoring = parser.add_argument_group(
        "Periodic AndroidWorld SR and TensorBoard"
    )
    monitoring.add_argument(
        "--sr_eval_interval_steps",
        type=int,
        default=None,
        help=(
            "Evaluate retained LoRA checkpoints whose step is a multiple of "
            "this value; 0 disables. Defaults to 100 for train/all with LoRA "
            "merging (or twice --save_steps when that value changes), "
            "otherwise 0. The last checkpoint is always selectable."
        ),
    )
    monitoring.add_argument(
        "--sr_eval_rollouts_per_task",
        type=int,
        default=1,
        help="Low-cost AndroidWorld rollouts per task for each checkpoint.",
    )
    monitoring.add_argument(
        "--sr_eval_tasks",
        nargs="+",
        metavar="TASK",
        default=None,
        help=(
            "Optional task subset for the checkpoint SR curve. Defaults to all "
            "tasks selected for this distillation run."
        ),
    )
    monitoring.add_argument(
        "--sr_eval_include_final_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the last checkpoint even when it is off the interval.",
    )
    monitoring.add_argument(
        "--launch_tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start or reuse a persistent TensorBoard server for training loss, "
            "dataset, pipeline, and AndroidWorld SR metrics."
        ),
    )
    monitoring.add_argument("--tensorboard_host", default="127.0.0.1")
    monitoring.add_argument("--tensorboard_port", type=int, default=6006)
    monitoring.add_argument(
        "--tensorboard_auto_port",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the next free port when the requested port serves another run.",
    )
    monitoring.add_argument(
        "--tensorboard_log_dir",
        type=Path,
        default=None,
        help="TensorBoard root. Defaults to <run_dir>/tensorboard.",
    )
    monitoring.add_argument(
        "--tensorboard_strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of warning if TensorBoard cannot be launched.",
    )

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


def resolve_sr_tasks(
    args: argparse.Namespace, selected_tasks: Sequence[str]
) -> list[str]:
    """Resolve the low-cost periodic SR subset and keep comparisons fair."""

    requested = normalize_tasks(args.sr_eval_tasks)
    if not requested:
        return list(selected_tasks)
    outside_run = sorted(set(requested) - set(selected_tasks))
    if outside_run:
        raise ValueError(
            "--sr_eval_tasks must be a subset of --tasks: "
            + ", ".join(outside_run)
        )
    return requested


def validate_args(args: argparse.Namespace) -> None:
    """Validate values before launching any expensive process."""

    if args.evaluate_after_training is None:
        args.evaluate_after_training = (
            args.stage in {"all", "train"} and args.merge_lora
        )
    if args.sr_eval_interval_steps is None:
        args.sr_eval_interval_steps = (
            args.save_steps * 2
            if args.stage in {"all", "train"} and args.merge_lora
            else 0
        )

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
        "--evaluation_rollouts_per_task": args.evaluation_rollouts_per_task,
        "--sr_eval_rollouts_per_task": args.sr_eval_rollouts_per_task,
        "--student_eval_tensor_parallel_size": (
            args.student_eval_tensor_parallel_size
        ),
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be at least 1.")
    if args.dataset_num_proc < 0 or args.dataloader_num_workers < 0:
        raise ValueError("Dataset/dataloader worker counts cannot be negative.")
    if args.save_total_limit < 0:
        raise ValueError("--save_total_limit cannot be negative.")
    if args.sr_eval_interval_steps < 0:
        raise ValueError("--sr_eval_interval_steps cannot be negative.")
    if (
        args.sr_eval_interval_steps
        and args.sr_eval_interval_steps % args.save_steps
    ):
        raise ValueError(
            "--sr_eval_interval_steps must be a multiple of --save_steps "
            "because AndroidWorld SR is evaluated from saved checkpoints."
        )
    if not 1 <= args.tensorboard_port <= 65535:
        raise ValueError("--tensorboard_port must be in [1, 65535].")
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
    if args.stage not in {"all", "train"} and args.sr_eval_interval_steps:
        raise ValueError(
            "--sr_eval_interval_steps is only valid with --stage all or train."
        )


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    """Create deterministic child paths beneath the selected run directory."""

    if args.run_dir is None:
        stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_OUTPUT_ROOT / f"glm_to_qwen_androidworld_{stamp}"
    else:
        run_dir = args.run_dir
    run_dir = run_dir.expanduser().resolve()
    tensorboard_dir = (
        args.tensorboard_log_dir.expanduser().resolve()
        if args.tensorboard_log_dir
        else run_dir / "tensorboard"
    )
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
        periodic_evaluations=run_dir / "evaluation" / "checkpoints",
        tensorboard_dir=tensorboard_dir,
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
        "--logging_dir",
        str(paths.tensorboard_dir / "training"),
        "--run_name",
        paths.run_dir.name,
        "--add_version",
        "false",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
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
    if args.save_total_limit:
        command.extend(("--save_total_limit", str(args.save_total_limit)))
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


def ensure_merged_adapter(
    args: argparse.Namespace,
    adapter_path: Path,
    merged_model_path: Path,
    *,
    label: str,
) -> bool:
    """Export an adapter once and reject stale/unverifiable merged models."""

    source_metadata = merged_model_path / ".distillation_merge.json"
    expected_adapter = str(adapter_path.resolve())
    if (merged_model_path / "config.json").is_file():
        try:
            metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                "A merged model already exists, but its source checkpoint "
                f"cannot be verified: {merged_model_path}. Choose a new output "
                "directory or remove the stale export explicitly."
            ) from exc
        if (
            isinstance(metadata, Mapping)
            and metadata.get("adapter_path") == expected_adapter
        ):
            print(
                f"[{label}] reusing verified merged model: {merged_model_path}",
                flush=True,
            )
            return False
        raise FileExistsError(
            "The existing merged model was exported from a different adapter: "
            f"{merged_model_path}"
        )
    if merged_model_path.exists():
        raise FileExistsError(
            "An incomplete merged model path exists and cannot be safely "
            f"overwritten: {merged_model_path}"
        )

    run_command(
        build_merge_command(adapter_path, merged_model_path),
        label=label,
        env=_swift_environment(args),
    )
    if not (merged_model_path / "config.json").is_file():
        raise RuntimeError(
            "ms-swift export returned successfully but did not create a "
            f"loadable merged model at {merged_model_path}."
        )
    _write_json(
        source_metadata,
        {
            "schema_version": 1,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "adapter_path": expected_adapter,
            "merged_model_path": str(merged_model_path.resolve()),
        },
    )
    return True


def build_evaluation_command(
    args: argparse.Namespace,
    tasks: Sequence[str],
    *,
    model_path: str | Path,
    run_dir: Path,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    rollouts_per_task: int | None = None,
    evaluation_seed: int | None = None,
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
        str(
            args.evaluation_rollouts_per_task
            if rollouts_per_task is None
            else rollouts_per_task
        ),
        "--task_random_seed",
        str(args.evaluation_seed if evaluation_seed is None else evaluation_seed),
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


@dataclass(frozen=True)
class TensorBoardServer:
    """Metadata for a launched or already-running TensorBoard server."""

    url: str
    log_dir: Path
    pid: int | None
    started: bool
    log_file: Path | None


class DistillationMetrics:
    """Durable pipeline metrics mirrored to TensorBoard when available."""

    def __init__(self, events_path: Path, *, tensorboard_enabled: bool) -> None:
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self.tensorboard_error: str | None = None
        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(
                    log_dir=str(self.events_path.parent),
                    flush_secs=5,
                )
            except (ImportError, OSError, RuntimeError) as exc:
                self.tensorboard_error = f"{type(exc).__name__}: {exc}"
                print(
                    "WARNING: custom distillation metrics will remain in "
                    f"{self.events_path}, but TensorBoard event writing is "
                    f"unavailable: {self.tensorboard_error}",
                    file=sys.stderr,
                    flush=True,
                )

    def _append_event(self, event: Mapping[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def scalar(self, tag: str, value: Any, step: int = 0) -> None:
        if not isinstance(value, (int, float)):
            return
        numeric = float(value)
        event = {
            "time": dt.datetime.now().astimezone().isoformat(),
            "type": "scalar",
            "tag": tag,
            "step": int(step),
            "value": numeric,
        }
        self._append_event(event)
        if self._writer is not None:
            self._writer.add_scalar(tag, numeric, int(step))
            self._writer.flush()

    def text(self, tag: str, value: str, step: int = 0) -> None:
        event = {
            "time": dt.datetime.now().astimezone().isoformat(),
            "type": "text",
            "tag": tag,
            "step": int(step),
            "value": value,
        }
        self._append_event(event)
        if self._writer is not None:
            self._writer.add_text(tag, value, int(step))
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _tensorboard_connect_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection(
            (_tensorboard_connect_host(host), port), timeout=0.25
        ):
            return True
    except OSError:
        return False


def _active_tensorboard_logdir(url: str) -> Path | None:
    try:
        with urllib.request.urlopen(
            f"{url}/data/logdir", timeout=1.0
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None
    raw = payload.get("logdir") if isinstance(payload, Mapping) else None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return None


def launch_tensorboard(
    log_dir: Path,
    *,
    host: str,
    port: int,
    auto_port: bool = True,
) -> TensorBoardServer:
    """Launch a detached TensorBoard, or reuse the configured listening port."""

    log_dir.mkdir(parents=True, exist_ok=True)
    display_host = "localhost" if host in {"0.0.0.0", "::", "[::]"} else host
    url = f"http://{display_host}:{port}"
    if _port_is_open(host, port):
        active_log_dir = _active_tensorboard_logdir(url)
        if active_log_dir != log_dir.resolve():
            occupied_description = (
                f"TensorBoard for {active_log_dir}"
                if active_log_dir is not None
                else "another or unverifiable service"
            )
            if not auto_port:
                raise RuntimeError(
                    f"Port {port} is already in use by "
                    f"{occupied_description}. Choose another "
                    "--tensorboard_port."
                )
            requested_port = port
            for candidate_port in range(port + 1, min(port + 51, 65536)):
                if not _port_is_open(host, candidate_port):
                    port = candidate_port
                    url = f"http://{display_host}:{port}"
                    break
            else:
                raise RuntimeError(
                    "Could not find a free TensorBoard port in "
                    f"[{requested_port}, {min(requested_port + 50, 65535)}]."
                )
            print(
                f"TensorBoard port {requested_port} is used by "
                f"{occupied_description}; using {port}.",
                flush=True,
            )
        else:
            print(
                f"TensorBoard port is already active; reusing {url}",
                flush=True,
            )
            return TensorBoardServer(
                url=url,
                log_dir=log_dir,
                pid=None,
                started=False,
                log_file=None,
            )

    log_file = log_dir / "tensorboard_server.log"
    tensorboard_executable = shutil.which("tensorboard")
    command = [
        *(
            [tensorboard_executable]
            if tensorboard_executable
            else [sys.executable, "-m", "tensorboard.main"]
        ),
        "--logdir",
        str(log_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    popen_options: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "start_new_session": os.name != "nt",
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    with log_file.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            **popen_options,
        )
    time.sleep(1.0)
    return_code = process.poll()
    if return_code is not None:
        try:
            details = log_file.read_text(encoding="utf-8")[-4000:]
        except OSError:
            details = ""
        raise RuntimeError(
            "TensorBoard exited during startup with code "
            f"{return_code}. Log: {log_file}\n{details}"
        )
    print(
        f"TensorBoard: {url} (PID {process.pid}, logs: {log_dir})",
        flush=True,
    )
    return TensorBoardServer(
        url=url,
        log_dir=log_dir,
        pid=process.pid,
        started=True,
        log_file=log_file,
    )


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


def find_adapter_checkpoints(training_output: Path) -> list[tuple[int, Path]]:
    """Return all numeric, loadable LoRA checkpoints in ascending order."""

    checkpoints: list[tuple[int, Path]] = []
    if not training_output.is_dir():
        return checkpoints
    for child in training_output.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if (
            match
            and child.is_dir()
            and (child / "adapter_config.json").is_file()
        ):
            checkpoints.append((int(match.group(1)), child))
    return sorted(checkpoints, key=lambda item: item[0])


def select_periodic_checkpoints(
    training_output: Path,
    *,
    interval_steps: int,
    include_final: bool,
) -> list[tuple[int, Path]]:
    """Select a stable checkpoint SR curve without evaluating duplicates."""

    checkpoints = find_adapter_checkpoints(training_output)
    if not checkpoints:
        adapter = find_latest_adapter(training_output)
        return [(0, adapter)] if include_final else []
    selected = [
        item for item in checkpoints if interval_steps and item[0] % interval_steps == 0
    ]
    if include_final and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return selected


def _trainer_state_path(training_output: Path, adapter_path: Path) -> Path:
    candidates = (
        training_output / "trainer_state.json",
        adapter_path / "trainer_state.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No trainer_state.json found in the training output or latest "
        f"checkpoint: {training_output}"
    )


def summarize_training_state(
    training_output: Path,
    adapter_path: Path,
    *,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Summarize the loss curve in an immediately readable artifact."""

    state_path = _trainer_state_path(training_output, adapter_path)
    state = _read_report(state_path)
    log_history = state.get("log_history")
    rows = log_history if isinstance(log_history, list) else []

    train_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("loss"), (int, float))
    ]
    eval_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("eval_loss"), (int, float))
    ]

    def value(row: Mapping[str, Any] | None, key: str) -> float | None:
        raw = row.get(key) if row else None
        return float(raw) if isinstance(raw, (int, float)) else None

    first_loss = value(train_rows[0] if train_rows else None, "loss")
    final_loss = value(train_rows[-1] if train_rows else None, "loss")
    first_eval_loss = value(eval_rows[0] if eval_rows else None, "eval_loss")
    final_eval_loss = value(eval_rows[-1] if eval_rows else None, "eval_loss")
    best_eval_row = (
        min(eval_rows, key=lambda row: float(row["eval_loss"]))
        if eval_rows
        else None
    )
    best_eval_loss = value(best_eval_row, "eval_loss")
    global_step_raw = state.get("global_step")
    global_step = (
        int(global_step_raw)
        if isinstance(global_step_raw, (int, float))
        else None
    )
    loss_change = (
        round(final_loss - first_loss, 6)
        if first_loss is not None and final_loss is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "trainer_state": str(state_path.resolve()),
        "latest_adapter": str(adapter_path.resolve()),
        "global_step": global_step,
        "epoch": state.get("epoch"),
        "train": {
            "logged_points": len(train_rows),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_change": loss_change,
            "loss_decreased": (
                final_loss < first_loss
                if first_loss is not None and final_loss is not None
                else None
            ),
        },
        "validation": {
            "logged_points": len(eval_rows),
            "first_loss": first_eval_loss,
            "final_loss": final_eval_loss,
            "best_loss": best_eval_loss,
            "best_step": (
                best_eval_row.get("step") if best_eval_row is not None else None
            ),
        },
    }
    _write_json(output_json, summary)

    def metric(value_to_format: Any) -> str:
        return (
            "N/A"
            if value_to_format is None
            else f"{float(value_to_format):.6f}"
        )

    direction = (
        "DECREASED (expected)"
        if summary["train"]["loss_decreased"] is True
        else (
            "INCREASED"
            if summary["train"]["loss_decreased"] is False
            else "N/A"
        )
    )
    markdown = [
        "# Distillation Training Summary",
        "",
        f"- Global step: **{global_step if global_step is not None else 'N/A'}**",
        f"- Latest adapter: `{adapter_path.resolve()}`",
        f"- Training loss trend: **{direction}**",
        "",
        "| Metric | First | Final | Best |",
        "|---|---:|---:|---:|",
        (
            f"| Training loss | {metric(first_loss)} | "
            f"{metric(final_loss)} | N/A |"
        ),
        (
            f"| Validation loss | {metric(first_eval_loss)} | "
            f"{metric(final_eval_loss)} | {metric(best_eval_loss)} |"
        ),
        "",
        (
            "Loss only measures teacher-response imitation. AndroidWorld SR "
            "below is the primary behavioural outcome."
        ),
    ]
    _atomic_write_text(output_markdown, "\n".join(markdown) + "\n")
    return summary


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


def _report_summary_metrics(report_path: Path) -> dict[str, Any]:
    report = _read_report(report_path)
    summary = report.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}

    def number(key: str) -> float | None:
        value = summary.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "micro_sr": number("task_success_rate"),
        "macro_sr": number("macro_task_success_rate"),
        "planned_episodes": summary.get("planned_episodes"),
        "scored_episodes": summary.get("scored_episodes"),
        "successful_episodes": summary.get("successful_episodes"),
        "error_episodes": summary.get("error_episodes"),
        "evaluation_coverage": number("evaluation_coverage"),
    }


def evaluation_report_is_compatible(
    report_path: Path,
    *,
    model_path: str | Path,
    tasks: Sequence[str],
    rollouts_per_task: int,
    evaluation_seed: int,
) -> bool:
    """Return False when absent; reject stale reports with a clear reason."""

    if not report_path.is_file():
        return False
    report = _read_report(report_path)
    run = report.get("run")
    config = run.get("config") if isinstance(run, Mapping) else None
    model = config.get("model") if isinstance(config, Mapping) else None
    suite = config.get("suite") if isinstance(config, Mapping) else None
    if not isinstance(model, Mapping) or not isinstance(suite, Mapping):
        raise ValueError(
            f"Existing evaluation report has no compatible run config: {report_path}"
        )
    expected = {
        "model_path": str(model_path),
        "tasks": list(tasks),
        "n_task_combinations": int(rollouts_per_task),
        "task_random_seed": int(evaluation_seed),
    }
    actual = {
        "model_path": str(model.get("model_path")),
        "tasks": list(suite.get("tasks") or []),
        "n_task_combinations": suite.get("n_task_combinations"),
        "task_random_seed": suite.get("task_random_seed"),
    }
    if actual != expected:
        raise ValueError(
            "Existing evaluation report was produced with a different model "
            f"or suite configuration: {report_path}\n"
            f"expected={expected}\nactual={actual}"
        )
    return run.get("status") == "completed"


def _improvement_label(gain: float | None) -> str:
    if gain is None:
        return "unknown"
    if gain > 1e-12:
        return "improved"
    if gain < -1e-12:
        return "regressed"
    return "unchanged"


def write_checkpoint_sr_history(
    baseline_report_path: Path,
    checkpoint_results: Sequence[Mapping[str, Any]],
    *,
    output_json: Path,
    output_csv: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Persist an incremental base-to-checkpoint AndroidWorld SR curve."""

    baseline = _report_summary_metrics(baseline_report_path)
    baseline_micro = baseline["micro_sr"]
    rows: list[dict[str, Any]] = []
    for raw in sorted(
        checkpoint_results, key=lambda item: int(item.get("step", 0))
    ):
        report_path = Path(str(raw["report_path"]))
        metrics = _report_summary_metrics(report_path)
        micro = metrics["micro_sr"]
        gain = (
            round(micro - baseline_micro, 6)
            if micro is not None and baseline_micro is not None
            else None
        )
        rows.append(
            {
                "step": int(raw.get("step", 0)),
                "label": str(raw.get("label") or f"step-{raw.get('step', 0)}"),
                "adapter_path": raw.get("adapter_path"),
                "merged_model_path": raw.get("merged_model_path"),
                "report_path": str(report_path.resolve()),
                **metrics,
                "micro_sr_gain_over_base": gain,
                "verdict": _improvement_label(gain),
            }
        )

    history = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "metric": "AndroidWorld task_success_rate",
        "baseline_report": str(baseline_report_path.resolve()),
        "baseline": baseline,
        "checkpoints": rows,
        "best_checkpoint": (
            max(
                (row for row in rows if row["micro_sr"] is not None),
                key=lambda row: float(row["micro_sr"]),
                default=None,
            )
        ),
    }
    _write_json(output_json, history)

    csv_columns = [
        "step",
        "label",
        "micro_sr",
        "macro_sr",
        "micro_sr_gain_over_base",
        "verdict",
        "planned_episodes",
        "scored_episodes",
        "successful_episodes",
        "error_episodes",
        "evaluation_coverage",
        "adapter_path",
        "merged_model_path",
        "report_path",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in csv_columns} for row in rows
        )
    temporary_csv.replace(output_csv)

    def percentage(value: Any) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.2f}%"

    markdown = [
        "# Periodic AndroidWorld SR",
        "",
        (
            f"- Base student SR: **{percentage(baseline_micro)}** "
            f"({baseline.get('successful_episodes', 'N/A')}/"
            f"{baseline.get('scored_episodes', 'N/A')} successful)"
        ),
        "- Each row uses the same tasks, seed, and rollout count as the base row.",
        "",
        "| Step | Student SR | Gain vs base | Macro SR | Verdict | Scored | Errors |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        icon = {
            "improved": "IMPROVED",
            "unchanged": "UNCHANGED",
            "regressed": "REGRESSED",
        }.get(row["verdict"], "UNKNOWN")
        markdown.append(
            f"| {row['step']} | {percentage(row['micro_sr'])} | "
            f"{percentage(row['micro_sr_gain_over_base'])} | "
            f"{percentage(row['macro_sr'])} | **{icon}** | "
            f"{row['scored_episodes']} | {row['error_episodes']} |"
        )
    if history["best_checkpoint"] is not None:
        best = history["best_checkpoint"]
        markdown.extend(
            [
                "",
                (
                    f"Best checkpoint: **step {best['step']}**, SR "
                    f"**{percentage(best['micro_sr'])}**, gain "
                    f"**{percentage(best['micro_sr_gain_over_base'])}**."
                ),
            ]
        )
    _atomic_write_text(output_markdown, "\n".join(markdown) + "\n")
    return history


def evaluate_periodic_checkpoint(
    args: argparse.Namespace,
    paths: PipelinePaths,
    tasks: Sequence[str],
    *,
    step: int,
    adapter_path: Path,
) -> dict[str, Any]:
    """Merge one LoRA checkpoint and evaluate its real AndroidWorld SR."""

    checkpoint_root = paths.periodic_evaluations / f"step-{step:08d}"
    merged_model = checkpoint_root / "merged_model"
    evaluation_dir = checkpoint_root / "androidworld"
    report_path = evaluation_dir / "report.json"

    report_exists = evaluation_report_is_compatible(
        report_path,
        model_path=merged_model,
        tasks=tasks,
        rollouts_per_task=args.sr_eval_rollouts_per_task,
        evaluation_seed=args.evaluation_seed,
    )
    if not report_exists:
        ensure_merged_adapter(
            args,
            adapter_path,
            merged_model,
            label=f"merge checkpoint step {step}",
        )
        command = build_evaluation_command(
            args,
            tasks,
            model_path=merged_model,
            run_dir=evaluation_dir,
            tensor_parallel_size=args.student_eval_tensor_parallel_size,
            gpu_memory_utilization=args.student_eval_gpu_memory_utilization,
            max_model_len=args.student_eval_max_model_len,
            rollouts_per_task=args.sr_eval_rollouts_per_task,
            evaluation_seed=args.evaluation_seed,
        )
        run_command(
            command,
            label=f"AndroidWorld checkpoint SR at step {step}",
        )

    # Validate now so corrupt/partial reports fail before entering the history.
    _report_summary_metrics(report_path)
    return {
        "step": step,
        "label": f"checkpoint-{step}",
        "adapter_path": str(adapter_path.resolve()),
        "merged_model_path": str(merged_model.resolve()),
        "report_path": str(report_path.resolve()),
    }


def _episode_change_examples(
    baseline: Mapping[str, Any] | None,
    student: Mapping[str, Any],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    """Find matched instances where distillation changed task outcome."""

    def episode_map(
        report: Mapping[str, Any] | None,
    ) -> dict[tuple[str, str, str], Mapping[str, Any]]:
        if report is None:
            return {}
        raw_episodes = report.get("episodes")
        episodes = raw_episodes if isinstance(raw_episodes, list) else []
        result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            key = (
                str(episode.get("task_template")),
                str(episode.get("instance_id")),
                str(episode.get("seed")),
            )
            result[key] = episode
        return result

    def actions(episode: Mapping[str, Any]) -> list[str]:
        raw_steps = episode.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        result: list[str] = []
        for step in steps[:5]:
            if not isinstance(step, Mapping):
                continue
            action = step.get("action") or step.get("action_output")
            if isinstance(action, str) and action.strip():
                compact = " ".join(action.split())
                result.append(compact[:500])
        return result

    base_map = episode_map(baseline)
    student_map = episode_map(student)
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    matched = 0
    for key in sorted(set(base_map) & set(student_map)):
        base_episode = base_map[key]
        student_episode = student_map[key]
        matched += 1
        base_success = bool(base_episode.get("is_successful"))
        student_success = bool(student_episode.get("is_successful"))
        if base_success == student_success:
            continue
        row = {
            "task": key[0],
            "instance_id": base_episode.get("instance_id"),
            "seed": base_episode.get("seed"),
            "goal": base_episode.get("goal") or student_episode.get("goal"),
            "base_outcome": base_episode.get("outcome"),
            "student_outcome": student_episode.get("outcome"),
            "base_actions": actions(base_episode),
            "student_actions": actions(student_episode),
        }
        (improved if student_success else regressed).append(row)
    return {
        "matched_episodes": matched,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "net_improved_instances": len(improved) - len(regressed),
        "improved_examples": improved[:limit],
        "regressed_examples": regressed[:limit],
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
    micro_gain = (
        round(student_micro - baseline_micro, 4)
        if baseline_micro is not None and student_micro is not None
        else None
    )
    macro_gain = (
        round(student_macro - baseline_macro, 4)
        if baseline_macro is not None and student_macro is not None
        else None
    )
    episode_changes = _episode_change_examples(baseline, student)
    comparison = {
        "schema_version": 2,
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
            "micro_sr_gain_over_base": micro_gain,
            "verdict": _improvement_label(micro_gain),
            "micro_sr_gap": (
                round(student_micro - teacher_micro, 4)
                if teacher_micro is not None and student_micro is not None
                else None
            ),
            "teacher_macro_sr": teacher_macro,
            "base_student_macro_sr": baseline_macro,
            "student_macro_sr": student_macro,
            "macro_sr_gain_over_base": macro_gain,
            "macro_sr_gap": (
                round(student_macro - teacher_macro, 4)
                if teacher_macro is not None and student_macro is not None
                else None
            ),
        },
        "by_task": rows,
        "episode_changes": episode_changes,
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
            f"Verdict: **{str(comparison['overall']['verdict']).upper()}** "
            f"(micro SR gain {percentage(micro_gain)})."
        ),
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
    markdown.extend(
        [
            "",
            "## Matched episode changes",
            "",
            (
                f"- Matched episodes: {episode_changes['matched_episodes']}; "
                f"fail to success: **{episode_changes['improved_count']}**; "
                f"success to fail: **{episode_changes['regressed_count']}**; "
                f"net: **{episode_changes['net_improved_instances']:+d}**."
            ),
        ]
    )
    for heading, key in (
        ("Improved examples", "improved_examples"),
        ("Regressed examples", "regressed_examples"),
    ):
        examples = episode_changes[key]
        if not examples:
            continue
        markdown.extend(["", f"### {heading}", ""])
        for example in examples:
            goal = " ".join(str(example.get("goal") or "N/A").split())[:500]
            markdown.extend(
                [
                    (
                        f"- **{example['task']} #{example['instance_id']}** "
                        f"(seed {example['seed']}): {goal}"
                    ),
                    (
                        f"  - Base: `{example['base_outcome']}`; actions: "
                        f"{json.dumps(example['base_actions'], ensure_ascii=False)}"
                    ),
                    (
                        f"  - Distilled: `{example['student_outcome']}`; actions: "
                        f"{json.dumps(example['student_actions'], ensure_ascii=False)}"
                    ),
                ]
            )
    _atomic_write_text(output_markdown, "\n".join(markdown) + "\n")
    return comparison


def print_distillation_outcome(comparison: Mapping[str, Any]) -> None:
    """Print the one-screen result a user needs at the end of a run."""

    overall = comparison.get("overall")
    overall = overall if isinstance(overall, Mapping) else {}
    changes = comparison.get("episode_changes")
    changes = changes if isinstance(changes, Mapping) else {}

    def percentage(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        return f"{float(value) * 100:.2f}%"

    def percentage_points(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        return f"{float(value) * 100:+.2f} pp"

    verdict = str(overall.get("verdict") or "unknown").upper()
    print("\n=== Distillation outcome ===", flush=True)
    print(
        f"{verdict}: base SR "
        f"{percentage(overall.get('base_student_micro_sr'))} -> distilled SR "
        f"{percentage(overall.get('student_micro_sr'))} "
        f"({percentage_points(overall.get('micro_sr_gain_over_base'))})",
        flush=True,
    )
    print(
        "Matched instance changes: "
        f"{changes.get('improved_count', 0)} fail->success, "
        f"{changes.get('regressed_count', 0)} success->fail.",
        flush=True,
    )


def build_distillation_result(
    paths: PipelinePaths,
    *,
    include_checkpoint_history: bool = True,
    include_final_comparison: bool = True,
) -> dict[str, Any]:
    """Combine training and behavioural evidence into one hand-off artifact."""

    training = (
        _read_report(paths.training_summary_json)
        if paths.training_summary_json.is_file()
        else None
    )
    checkpoint_history = (
        _read_report(paths.checkpoint_sr_json)
        if include_checkpoint_history and paths.checkpoint_sr_json.is_file()
        else None
    )
    comparison = (
        _read_report(paths.comparison_json)
        if include_final_comparison and paths.comparison_json.is_file()
        else None
    )

    comparison_overall = (
        comparison.get("overall")
        if isinstance(comparison, Mapping)
        and isinstance(comparison.get("overall"), Mapping)
        else {}
    )
    checkpoint_rows = (
        checkpoint_history.get("checkpoints")
        if isinstance(checkpoint_history, Mapping)
        and isinstance(checkpoint_history.get("checkpoints"), list)
        else []
    )
    last_checkpoint = (
        max(
            (row for row in checkpoint_rows if isinstance(row, Mapping)),
            key=lambda row: int(row.get("step", 0)),
            default=None,
        )
        if checkpoint_rows
        else None
    )
    if comparison_overall:
        base_sr = comparison_overall.get("base_student_micro_sr")
        student_sr = comparison_overall.get("student_micro_sr")
        gain = comparison_overall.get("micro_sr_gain_over_base")
        verdict = comparison_overall.get("verdict")
        result_source = "final_held_out_evaluation"
    elif last_checkpoint is not None:
        baseline = checkpoint_history.get("baseline")
        baseline = baseline if isinstance(baseline, Mapping) else {}
        base_sr = baseline.get("micro_sr")
        student_sr = last_checkpoint.get("micro_sr")
        gain = last_checkpoint.get("micro_sr_gain_over_base")
        verdict = last_checkpoint.get("verdict")
        result_source = "last_periodic_checkpoint"
    else:
        base_sr = student_sr = gain = None
        verdict = "not_evaluated"
        result_source = "training_loss_only"

    result = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "verdict": verdict,
        "result_source": result_source,
        "base_student_micro_sr": base_sr,
        "distilled_student_micro_sr": student_sr,
        "micro_sr_gain_over_base": gain,
        "teacher_micro_sr": comparison_overall.get("teacher_micro_sr"),
        "training": training,
        "best_periodic_checkpoint": (
            checkpoint_history.get("best_checkpoint")
            if isinstance(checkpoint_history, Mapping)
            else None
        ),
        "last_periodic_checkpoint": last_checkpoint,
        "episode_changes": (
            comparison.get("episode_changes")
            if isinstance(comparison, Mapping)
            else None
        ),
        "artifacts": {
            "training_summary": (
                str(paths.training_summary_markdown)
                if paths.training_summary_markdown.is_file()
                else None
            ),
            "checkpoint_sr_history": (
                str(paths.checkpoint_sr_markdown)
                if (
                    include_checkpoint_history
                    and paths.checkpoint_sr_markdown.is_file()
                )
                else None
            ),
            "final_sr_comparison": (
                str(paths.comparison_markdown)
                if (
                    include_final_comparison
                    and paths.comparison_markdown.is_file()
                )
                else None
            ),
            "tensorboard_log_dir": str(paths.tensorboard_dir),
            "pipeline_manifest": str(paths.pipeline_manifest),
        },
    }
    _write_json(paths.result_json, result)

    def percentage(value: Any) -> str:
        return (
            "N/A"
            if not isinstance(value, (int, float))
            else f"{float(value) * 100:.2f}%"
        )

    def points(value: Any) -> str:
        return (
            "N/A"
            if not isinstance(value, (int, float))
            else f"{float(value) * 100:+.2f} pp"
        )

    train = (
        training.get("train")
        if isinstance(training, Mapping)
        and isinstance(training.get("train"), Mapping)
        else {}
    )
    validation = (
        training.get("validation")
        if isinstance(training, Mapping)
        and isinstance(training.get("validation"), Mapping)
        else {}
    )
    markdown = [
        "# Distillation Result",
        "",
        f"## {str(verdict).upper()}",
        "",
        (
            f"Base student SR **{percentage(base_sr)}** -> distilled student "
            f"SR **{percentage(student_sr)}** "
            f"(**{points(gain)}**)."
        ),
        "",
        f"- Result source: `{result_source}`",
        (
            f"- Teacher SR: {percentage(comparison_overall.get('teacher_micro_sr'))}"
        ),
        (
            f"- Training loss: {train.get('first_loss', 'N/A')} -> "
            f"{train.get('final_loss', 'N/A')}"
        ),
        f"- Best validation loss: {validation.get('best_loss', 'N/A')}",
    ]
    changes = result.get("episode_changes")
    if isinstance(changes, Mapping):
        markdown.extend(
            [
                (
                    "- Matched instances: "
                    f"{changes.get('improved_count', 0)} fail->success, "
                    f"{changes.get('regressed_count', 0)} success->fail"
                )
            ]
        )
    best = result.get("best_periodic_checkpoint")
    if isinstance(best, Mapping):
        markdown.extend(
            [
                "",
                "## Best periodic checkpoint",
                "",
                (
                    f"Step **{best.get('step')}**: SR "
                    f"**{percentage(best.get('micro_sr'))}**, gain "
                    f"**{points(best.get('micro_sr_gain_over_base'))}**."
                ),
            ]
        )
    markdown.extend(
        [
            "",
            "## Detailed artifacts",
            "",
            f"- Training: `{result['artifacts']['training_summary']}`",
            f"- Checkpoint SR: `{result['artifacts']['checkpoint_sr_history']}`",
            f"- Final SR: `{result['artifacts']['final_sr_comparison']}`",
            f"- TensorBoard logs: `{result['artifacts']['tensorboard_log_dir']}`",
        ]
    )
    _atomic_write_text(paths.result_markdown, "\n".join(markdown) + "\n")
    return result


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
        sr_tasks = resolve_sr_tasks(args, tasks)
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
    periodic_baseline_eval_command = build_evaluation_command(
        args,
        sr_tasks,
        model_path=args.student_model,
        run_dir=paths.periodic_evaluations / "baseline",
        tensor_parallel_size=args.student_eval_tensor_parallel_size,
        gpu_memory_utilization=args.student_eval_gpu_memory_utilization,
        max_model_len=args.student_eval_max_model_len,
        rollouts_per_task=args.sr_eval_rollouts_per_task,
        evaluation_seed=args.evaluation_seed,
    )
    commands = {
        "collect_teacher": (
            teacher_command if args.stage in {"all", "collect"} else None
        ),
        "train_lora": (
            planned_training_command
            if args.stage in {"all", "train"}
            else None
        ),
        "merge_lora": (
            planned_merge_command
            if args.stage in {"all", "train"} and args.merge_lora
            else None
        ),
        "evaluate_periodic_base_student": (
            periodic_baseline_eval_command
            if args.sr_eval_interval_steps
            else None
        ),
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

    metrics: DistillationMetrics | None = None
    try:
        tensorboard_server: TensorBoardServer | None = None
        if args.launch_tensorboard:
            try:
                tensorboard_server = launch_tensorboard(
                    paths.tensorboard_dir,
                    host=args.tensorboard_host,
                    port=args.tensorboard_port,
                    auto_port=args.tensorboard_auto_port,
                )
            except (OSError, RuntimeError) as exc:
                manifest["tensorboard_error"] = f"{type(exc).__name__}: {exc}"
                _write_json(paths.pipeline_manifest, manifest)
                if args.tensorboard_strict:
                    raise
                print(
                    f"WARNING: TensorBoard could not be launched: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        metrics = DistillationMetrics(
            paths.pipeline_events,
            tensorboard_enabled=(
                args.launch_tensorboard or "tensorboard" in args.report_to
            ),
        )
        manifest["monitoring"] = {
            "tensorboard": (
                _json_safe(tensorboard_server.__dict__)
                if tensorboard_server is not None
                else None
            ),
            "tensorboard_log_dir": str(paths.tensorboard_dir),
            "pipeline_events": str(paths.pipeline_events),
            "sr_eval_interval_steps": args.sr_eval_interval_steps,
            "sr_eval_tasks": sr_tasks,
            "sr_eval_rollouts_per_task": args.sr_eval_rollouts_per_task,
            "checkpoint_evaluation_timing": (
                "after_training_to_avoid_training_vllm_gpu_contention"
            ),
        }
        metrics.text("pipeline/status", "started", 0)
        _write_json(paths.pipeline_manifest, manifest)

        if args.stage in {"all", "collect"}:
            if not TASK_RUNNER.is_file():
                raise FileNotFoundError(f"Missing task runner: {TASK_RUNNER}")
            manifest["status"] = "collecting_teacher_trajectories"
            metrics.text("pipeline/status", manifest["status"], 1)
            _write_json(paths.pipeline_manifest, manifest)
            run_command(teacher_command, label="teacher AndroidWorld collection")
            teacher_collection_metrics = _report_summary_metrics(
                paths.teacher_rollouts / "report.json"
            )
            metrics.scalar(
                "distillation/teacher_collection_micro_sr",
                teacher_collection_metrics.get("micro_sr"),
                0,
            )
            metrics.scalar(
                "distillation/teacher_collection_successful_episodes",
                teacher_collection_metrics.get("successful_episodes"),
                0,
            )

        dataset_manifest: Mapping[str, Any] | None = None
        if args.stage in {"all", "prepare"}:
            manifest["status"] = "preparing_dataset"
            metrics.text("pipeline/status", manifest["status"], 2)
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
        elif paths.dataset_manifest.is_file():
            dataset_manifest = _read_report(paths.dataset_manifest)

        if dataset_manifest is not None:
            counts = dataset_manifest.get("counts")
            if isinstance(counts, Mapping):
                metrics.scalar(
                    "dataset/train_samples", counts.get("train_samples"), 0
                )
                metrics.scalar(
                    "dataset/validation_samples",
                    counts.get("validation_samples"),
                    0,
                )
                metrics.scalar(
                    "dataset/train_action_samples",
                    counts.get("train_action_samples"),
                    0,
                )
                metrics.scalar(
                    "dataset/train_summary_samples",
                    counts.get("train_summary_samples"),
                    0,
                )

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
            metrics.text("pipeline/status", manifest["status"], 3)
            _write_json(paths.pipeline_manifest, manifest)
            run_command(
                training_command,
                label="ms-swift student LoRA training",
                env=_swift_environment(args),
            )

            adapter_path = find_latest_adapter(paths.training_output)
            manifest["latest_adapter"] = str(adapter_path.resolve())
            global_step = 0
            try:
                training_summary = summarize_training_state(
                    paths.training_output,
                    adapter_path,
                    output_json=paths.training_summary_json,
                    output_markdown=paths.training_summary_markdown,
                )
                manifest["training_summary"] = training_summary
                global_step = int(training_summary.get("global_step") or 0)
                train_summary = training_summary.get("train")
                validation_summary = training_summary.get("validation")
                if isinstance(train_summary, Mapping):
                    metrics.scalar(
                        "distillation/train_first_loss",
                        train_summary.get("first_loss"),
                        0,
                    )
                    metrics.scalar(
                        "distillation/train_final_loss",
                        train_summary.get("final_loss"),
                        global_step,
                    )
                if isinstance(validation_summary, Mapping):
                    metrics.scalar(
                        "distillation/validation_best_loss",
                        validation_summary.get("best_loss"),
                        global_step,
                    )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                manifest["training_summary_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(
                    f"WARNING: could not summarize trainer_state.json: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if not global_step:
                match = re.fullmatch(r"checkpoint-(\d+)", adapter_path.name)
                global_step = int(match.group(1)) if match else 0
            _write_json(paths.pipeline_manifest, manifest)

            if args.sr_eval_interval_steps:
                manifest["status"] = "evaluating_periodic_checkpoint_sr"
                metrics.text("pipeline/status", manifest["status"], 4)
                _write_json(paths.pipeline_manifest, manifest)
                periodic_baseline_report = (
                    paths.periodic_evaluations / "baseline" / "report.json"
                )
                periodic_baseline_exists = evaluation_report_is_compatible(
                    periodic_baseline_report,
                    model_path=args.student_model,
                    tasks=sr_tasks,
                    rollouts_per_task=args.sr_eval_rollouts_per_task,
                    evaluation_seed=args.evaluation_seed,
                )
                if not periodic_baseline_exists:
                    run_command(
                        periodic_baseline_eval_command,
                        label="periodic SR base-student baseline",
                    )
                baseline_metrics = _report_summary_metrics(
                    periodic_baseline_report
                )
                metrics.scalar(
                    "androidworld/checkpoint_curve_micro_sr",
                    baseline_metrics.get("micro_sr"),
                    0,
                )
                metrics.scalar(
                    "androidworld/checkpoint_curve_macro_sr",
                    baseline_metrics.get("macro_sr"),
                    0,
                )

                selected_checkpoints = select_periodic_checkpoints(
                    paths.training_output,
                    interval_steps=args.sr_eval_interval_steps,
                    include_final=args.sr_eval_include_final_checkpoint,
                )
                manifest["periodic_checkpoint_inventory"] = {
                    "available": [
                        {
                            "step": step,
                            "adapter_path": str(path.resolve()),
                        }
                        for step, path in find_adapter_checkpoints(
                            paths.training_output
                        )
                    ],
                    "selected_steps": [
                        step for step, _ in selected_checkpoints
                    ],
                }
                checkpoint_results: list[dict[str, Any]] = []
                for checkpoint_step, checkpoint_path in selected_checkpoints:
                    result = evaluate_periodic_checkpoint(
                        args,
                        paths,
                        sr_tasks,
                        step=checkpoint_step,
                        adapter_path=checkpoint_path,
                    )
                    checkpoint_results.append(result)
                    history = write_checkpoint_sr_history(
                        periodic_baseline_report,
                        checkpoint_results,
                        output_json=paths.checkpoint_sr_json,
                        output_csv=paths.checkpoint_sr_csv,
                        output_markdown=paths.checkpoint_sr_markdown,
                    )
                    current = next(
                        row
                        for row in history["checkpoints"]
                        if row["step"] == checkpoint_step
                    )
                    current_sr = current.get("micro_sr")
                    current_gain = current.get("micro_sr_gain_over_base")
                    current_sr_text = (
                        "N/A"
                        if current_sr is None
                        else f"{float(current_sr) * 100:.2f}%"
                    )
                    current_gain_text = (
                        "N/A"
                        if current_gain is None
                        else f"{float(current_gain) * 100:+.2f} pp"
                    )
                    print(
                        f"[checkpoint {checkpoint_step}] "
                        f"{str(current.get('verdict')).upper()}: SR "
                        f"{current_sr_text}, gain vs base {current_gain_text}",
                        flush=True,
                    )
                    metrics.scalar(
                        "androidworld/checkpoint_curve_micro_sr",
                        current.get("micro_sr"),
                        checkpoint_step,
                    )
                    metrics.scalar(
                        "androidworld/checkpoint_curve_macro_sr",
                        current.get("macro_sr"),
                        checkpoint_step,
                    )
                    metrics.scalar(
                        "androidworld/checkpoint_curve_gain_over_base",
                        current.get("micro_sr_gain_over_base"),
                        checkpoint_step,
                    )
                    manifest["checkpoint_sr_history"] = history
                    _write_json(paths.pipeline_manifest, manifest)

            if args.merge_lora:
                merge_command = build_merge_command(
                    adapter_path, paths.merged_model
                )
                manifest["commands"]["merge_lora"] = merge_command
                manifest["status"] = "merging_lora"
                metrics.text("pipeline/status", manifest["status"], 5)
                _write_json(paths.pipeline_manifest, manifest)
                ensure_merged_adapter(
                    args,
                    adapter_path,
                    paths.merged_model,
                    label="merge LoRA into standalone student",
                )

            if args.evaluate_after_training:
                manifest["status"] = "evaluating_held_out_sr"
                metrics.text("pipeline/status", manifest["status"], 6)
                _write_json(paths.pipeline_manifest, manifest)
                base_report = paths.student_baseline_evaluation / "report.json"
                student_report = paths.student_evaluation / "report.json"
                teacher_report = paths.teacher_evaluation / "report.json"
                if not evaluation_report_is_compatible(
                    base_report,
                    model_path=args.student_model,
                    tasks=tasks,
                    rollouts_per_task=args.evaluation_rollouts_per_task,
                    evaluation_seed=args.evaluation_seed,
                ):
                    run_command(
                        student_baseline_eval_command,
                        label="held-out base student AndroidWorld evaluation",
                    )
                if not evaluation_report_is_compatible(
                    student_report,
                    model_path=paths.merged_model,
                    tasks=tasks,
                    rollouts_per_task=args.evaluation_rollouts_per_task,
                    evaluation_seed=args.evaluation_seed,
                ):
                    run_command(
                        student_eval_command,
                        label="held-out distilled student AndroidWorld evaluation",
                    )
                if not evaluation_report_is_compatible(
                    teacher_report,
                    model_path=args.teacher_model,
                    tasks=tasks,
                    rollouts_per_task=args.evaluation_rollouts_per_task,
                    evaluation_seed=args.evaluation_seed,
                ):
                    run_command(
                        teacher_eval_command,
                        label="held-out teacher AndroidWorld evaluation",
                    )
                comparison = build_sr_comparison(
                    teacher_report,
                    student_report,
                    baseline_report_path=base_report,
                    output_json=paths.comparison_json,
                    output_markdown=paths.comparison_markdown,
                )
                manifest["sr_comparison"] = comparison
                overall = comparison.get("overall")
                if isinstance(overall, Mapping):
                    metrics.scalar(
                        "androidworld/final_base_micro_sr",
                        overall.get("base_student_micro_sr"),
                        0,
                    )
                    metrics.scalar(
                        "androidworld/final_distilled_micro_sr",
                        overall.get("student_micro_sr"),
                        global_step,
                    )
                    metrics.scalar(
                        "androidworld/final_teacher_micro_sr",
                        overall.get("teacher_micro_sr"),
                        global_step,
                    )
                    metrics.scalar(
                        "androidworld/final_gain_over_base",
                        overall.get("micro_sr_gain_over_base"),
                        global_step,
                    )
                print_distillation_outcome(comparison)

        if args.stage in {"all", "train"}:
            manifest["distillation_result"] = build_distillation_result(
                paths,
                include_checkpoint_history=bool(
                    args.sr_eval_interval_steps
                ),
                include_final_comparison=args.evaluate_after_training,
            )

        manifest["status"] = "completed"
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
        metrics.text("pipeline/status", manifest["status"], 7)
        _write_json(paths.pipeline_manifest, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
        manifest["error"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        _write_json(paths.pipeline_manifest, manifest)
        raise
    finally:
        if metrics is not None:
            metrics.close()

    print(f"\nDistillation manifest: {paths.pipeline_manifest}")
    if paths.result_markdown.is_file():
        print(f"One-page distillation result: {paths.result_markdown}")
    if paths.dataset_manifest.is_file():
        print(f"Dataset manifest: {paths.dataset_manifest}")
    if paths.comparison_markdown.is_file():
        print(f"Readable SR comparison: {paths.comparison_markdown}")
    elif args.merge_lora and args.stage in {"all", "train"}:
        print(f"Merged student model: {paths.merged_model}")
    if paths.training_summary_markdown.is_file():
        print(f"Training summary: {paths.training_summary_markdown}")
    if paths.checkpoint_sr_markdown.is_file():
        print(f"Checkpoint SR history: {paths.checkpoint_sr_markdown}")
    if tensorboard_server is not None:
        print(f"TensorBoard: {tensorboard_server.url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
