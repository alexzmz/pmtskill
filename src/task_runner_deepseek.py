"""Run Android World with the hosted DeepSeek Chat Completions API.

This runner deliberately reuses ``task_runner_detail.py`` for task generation,
official Android World scoring, checkpoints, fail-fast infrastructure handling,
and JSON/Markdown/CSV reports. Only the model backend is replaced.

Example:

    export DEEPSEEK_API_KEY=...
    python src/task_runner_deepseek.py \
      --deepseek_model deepseek-v4-flash \
      --tasks ContactsAddContact ClockCreateTimer
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Mapping, Sequence

import task_runner_detail


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def build_parser() -> argparse.ArgumentParser:
    parser = task_runner_detail.build_parser(
        description=__doc__,
        include_model_args=False,
    )
    model = parser.add_argument_group("DeepSeek API model")
    model.add_argument(
        "--deepseek_model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        help="DeepSeek Chat Completions model ID.",
    )
    model.add_argument(
        "--deepseek_api_key",
        default=None,
        help=(
            "DeepSeek API key. Prefer the DEEPSEEK_API_KEY environment "
            "variable so the secret is not exposed in the process list."
        ),
    )
    model.add_argument(
        "--deepseek_base_url",
        default=os.environ.get(
            "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
        ),
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    model.add_argument(
        "--deepseek_thinking",
        choices=("auto", "enabled", "disabled"),
        default="disabled",
        help=(
            "Thinking mode. 'auto' omits the API field; disabled most closely "
            "matches the former deepseek-chat non-thinking behavior."
        ),
    )
    model.add_argument(
        "--deepseek_timeout_s",
        type=float,
        default=180.0,
        help="Timeout for one HTTP attempt.",
    )
    model.add_argument(
        "--deepseek_max_retries",
        type=int,
        default=3,
        help="Retries for timeouts, rate limits, and transient server errors.",
    )
    model.add_argument(
        "--deepseek_retry_base_s",
        type=float,
        default=1.0,
        help="Initial exponential-backoff delay.",
    )
    model.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Zero is recommended for reproducible Android World evaluation.",
    )
    model.add_argument("--top_p", type=float, default=0.95)
    model.add_argument("--max_tokens", type=int, default=512)
    return parser


def _api_key(args: argparse.Namespace) -> str:
    return str(
        args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
    ).strip()


def validate_deepseek_args(args: argparse.Namespace) -> None:
    task_runner_detail.validate_args(args)
    if not str(args.deepseek_model).strip():
        raise ValueError("--deepseek_model cannot be empty.")
    if not str(args.deepseek_base_url).strip():
        raise ValueError("--deepseek_base_url cannot be empty.")
    if not _api_key(args):
        raise ValueError(
            "DeepSeek API key is required. Set DEEPSEEK_API_KEY or pass "
            "--deepseek_api_key."
        )
    if args.deepseek_timeout_s <= 0:
        raise ValueError("--deepseek_timeout_s must be positive.")
    if args.deepseek_max_retries < 0:
        raise ValueError("--deepseek_max_retries cannot be negative.")
    if args.deepseek_retry_base_s < 0:
        raise ValueError("--deepseek_retry_base_s cannot be negative.")


def create_deepseek(args: argparse.Namespace) -> Any:
    from deepseek_wrapper import DeepSeekWrapper

    return DeepSeekWrapper(
        model=str(args.deepseek_model),
        api_key=_api_key(args),
        base_url=str(args.deepseek_base_url),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_s=args.deepseek_timeout_s,
        max_retries=args.deepseek_max_retries,
        retry_base_s=args.deepseek_retry_base_s,
        thinking=args.deepseek_thinking,
        raise_on_error=True,
    )


def _model_info(args: argparse.Namespace) -> Mapping[str, Any]:
    # Never include the API key in run_config.json or the resume signature.
    return {
        "model": str(args.deepseek_model),
        "backend": "deepseek-api",
        "base_url": str(args.deepseek_base_url).rstrip("/"),
        "thinking": args.deepseek_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "request_timeout_s": args.deepseek_timeout_s,
        "max_retries": args.deepseek_max_retries,
        "retry_base_s": args.deepseek_retry_base_s,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        return task_runner_detail.list_available_tasks()
    validate_deepseek_args(args)
    return task_runner_detail.run_evaluation(
        args,
        condition="deepseek_api",
        model_factory=create_deepseek,
        model_info=_model_info(args),
        backend_label="DeepSeek API",
        agent_name="t3a_deepseek_api",
    )


if __name__ == "__main__":
    raise SystemExit(main())
