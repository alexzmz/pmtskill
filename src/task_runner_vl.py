"""Evaluate a bare vision-language model on Android World.

This runner uses Android World's M3A agent, supplies both the raw screenshot
and the screen-of-marks screenshot to vLLM, applies the model's own Hugging
Face chat template, and writes the same detailed/resumable reports as
``task_runner_detail.py``.

Examples:

    python src/task_runner_vl.py \
      --model_path /home/zmz/Workspace/models/glm4.1-vl-9b-base \
      --tasks ContactsAddContact ClockCreateTimer

    python src/task_runner_vl.py \
      --model_path /home/zmz/Workspace/models/mobilerl-9b \
      --prompt_profile mobilerl \
      --tasks ContactsAddContact

For Qwen2.5-VL and general instruction-tuned VL checkpoints, leave
``--prompt_profile auto`` (it resolves to the stock M3A action protocol).
MobileRL-named checkpoints are automatically assigned the MobileRL profile.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import task_runner_detail
from vllm_vl_wrapper import VLLMMultimodalWrapper, resolve_prompt_profile


DEFAULT_VL_MODEL = os.environ.get(
    "LOCAL_VL_MODEL_PATH",
    "/home/zmz/Workspace/models/glm4.1-vl-9b-base",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the VL runner CLI while retaining the shared suite/report flags."""
    parser = task_runner_detail.build_parser(description=__doc__)
    parser.set_defaults(model_path=DEFAULT_VL_MODEL, max_tokens=1024)

    vision = parser.add_argument_group("Vision-language inference")
    vision.add_argument(
        "--prompt_profile",
        choices=("auto", "m3a", "mobilerl"),
        default="auto",
        help=(
            "Action protocol expected by the checkpoint. Auto selects "
            "MobileRL when 'mobilerl' appears in the model path and M3A "
            "otherwise."
        ),
    )
    vision.add_argument(
        "--bbox_coordinate_mode",
        choices=("auto", "normalized_1000", "pixels"),
        default="auto",
        help=(
            "How MobileRL bounding-box coordinates map to AndroidWorld "
            "screen pixels."
        ),
    )
    vision.add_argument(
        "--max_images_per_prompt",
        type=int,
        default=2,
        help="Maximum screenshots accepted by one M3A model request.",
    )
    vision.add_argument(
        "--image_max_pixels",
        type=int,
        default=500_000,
        help=(
            "Downscale each screenshot to at most this many pixels before "
            "inference; use zero to keep the original resolution."
        ),
    )
    vision.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype, for example auto, bfloat16, or float16.",
    )
    vision.add_argument(
        "--max_num_seqs",
        type=int,
        default=1,
        help="vLLM sequence concurrency (one is memory-safe for VL eval).",
    )
    vision.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable CUDA graph execution in vLLM. Enabled by default for "
            "GLM-4.1V compatibility; use --no-enforce_eager if desired."
        ),
    )
    vision.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Per-request vLLM sampling seed.",
    )
    vision.add_argument(
        "--wait_after_action_seconds",
        type=float,
        default=2.0,
        help="M3A wait after executing an Android action.",
    )
    vision.add_argument(
        "--include_ui_bboxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add pixel and normalized 0..1000 bounds from Android "
            "accessibility data to M3A's UI-element descriptions. This is "
            "important for bbox-trained MobileRL checkpoints."
        ),
    )
    vision.add_argument(
        "--system_prompt_file",
        type=Path,
        default=None,
        help=(
            "Optional UTF-8 system prompt override. Usually the built-in "
            "M3A/MobileRL prompts should be retained."
        ),
    )
    vision.add_argument(
        "--normalize_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize JSON-only/fenced answers and translate MobileRL "
            "do(action=...) outputs into M3A JSON."
        ),
    )
    return parser


def validate_vl_args(args: argparse.Namespace) -> None:
    """Validate VL-specific settings before allocating model/GPU memory."""
    task_runner_detail.validate_args(args)
    if args.max_images_per_prompt < 1:
        raise ValueError("--max_images_per_prompt must be at least 1.")
    if args.image_max_pixels < 0:
        raise ValueError("--image_max_pixels cannot be negative.")
    if args.image_max_pixels and args.image_max_pixels < 1_024:
        raise ValueError(
            "--image_max_pixels must be zero or at least 1024."
        )
    if args.max_num_seqs < 1:
        raise ValueError("--max_num_seqs must be at least 1.")
    if args.wait_after_action_seconds < 0:
        raise ValueError("--wait_after_action_seconds cannot be negative.")
    if args.system_prompt_file is not None:
        prompt_path = args.system_prompt_file.expanduser()
        if not prompt_path.is_file():
            raise ValueError(
                f"--system_prompt_file does not exist: {prompt_path}"
            )


def _load_system_prompt(path: Path | None) -> str | None:
    if path is None:
        return None
    content = path.expanduser().read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("--system_prompt_file cannot be empty.")
    return content


def create_vl_model(args: argparse.Namespace) -> VLLMMultimodalWrapper:
    """Instantiate the in-process multimodal vLLM adapter."""
    return VLLMMultimodalWrapper(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        sampling_seed=args.sampling_seed,
        max_images_per_prompt=args.max_images_per_prompt,
        image_max_pixels=args.image_max_pixels,
        prompt_profile=args.prompt_profile,
        bbox_coordinate_mode=args.bbox_coordinate_mode,
        system_prompt=_load_system_prompt(args.system_prompt_file),
        normalize_actions=args.normalize_actions,
        raise_on_error=True,
    )


def _install_bbox_prompt_adapter(m3a_module: Any) -> None:
    """Augment M3A UI descriptions with MobileRL-compatible XML bounds."""
    if getattr(m3a_module, "_vl_bbox_prompt_installed", False):
        return

    def generate_ui_elements_description_list(
        ui_elements: list[Any],
        screen_width_height_px: tuple[int, int],
    ) -> str:
        lines: list[str] = []
        for index, element in enumerate(ui_elements):
            if not m3a_module.m3a_utils.validate_ui_element(
                element,
                screen_width_height_px,
            ):
                continue
            payload: dict[str, Any] = {"index": index}
            for field in (
                "text",
                "content_description",
                "hint_text",
                "tooltip",
                "class_name",
                "package_name",
                "resource_name",
                "resource_id",
            ):
                value = getattr(element, field, None)
                if value:
                    payload[field] = value
            for field in (
                "is_clickable",
                "is_long_clickable",
                "is_editable",
                "is_scrollable",
                "is_focusable",
                "is_focused",
                "is_selected",
                "is_checked",
                "is_enabled",
                "is_visible",
            ):
                value = getattr(element, field, None)
                if value is not None:
                    payload[field] = bool(value)

            bbox_pixels = getattr(element, "bbox_pixels", None)
            if bbox_pixels is not None:
                payload["bbox_pixels"] = [
                    round(bbox_pixels.x_min),
                    round(bbox_pixels.y_min),
                    round(bbox_pixels.x_max),
                    round(bbox_pixels.y_max),
                ]
            bbox = getattr(element, "bbox", None)
            if bbox is not None:
                payload["bbox_1000"] = [
                    round(bbox.x_min * 1000),
                    round(bbox.y_min * 1000),
                    round(bbox.x_max * 1000),
                    round(bbox.y_max * 1000),
                ]
            lines.append(
                f"UI element {index}: "
                + json.dumps(payload, ensure_ascii=False)
            )
        return "\n".join(lines) + ("\n" if lines else "")

    # M3A.step resolves this helper from its module on every turn. The patch
    # is confined to this runner process and leaves the vendored library
    # untouched on disk.
    m3a_module._generate_ui_elements_description_list = (  # noqa: SLF001
        generate_ui_elements_description_list
    )
    m3a_module._vl_bbox_prompt_installed = True


def create_m3a_agent(
    env: Any,
    llm: Any,
    _unused_t3a_module: Any,
    *,
    wait_after_action_seconds: float,
    include_ui_bboxes: bool,
) -> Any:
    """Create Android World's screenshot-aware M3A agent."""
    from android_world.agents import m3a

    if include_ui_bboxes:
        _install_bbox_prompt_adapter(m3a)
    return m3a.M3A(
        env,
        llm,
        name="m3a_vllm_vl",
        wait_after_action_seconds=wait_after_action_seconds,
    )


def _model_info(args: argparse.Namespace) -> dict[str, Any]:
    resolved_profile = resolve_prompt_profile(
        args.model_path,
        args.prompt_profile,
    )
    resolved_bbox_mode = (
        "normalized_1000"
        if args.bbox_coordinate_mode == "auto"
        and resolved_profile == "mobilerl"
        else args.bbox_coordinate_mode
    )
    return {
        "model_path": str(args.model_path),
        "backend": "vllm-multimodal",
        "agent": "android_world.m3a.M3A",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "sampling_seed": args.sampling_seed,
        "max_images_per_prompt": args.max_images_per_prompt,
        "image_max_pixels": args.image_max_pixels,
        "prompt_profile_requested": args.prompt_profile,
        "prompt_profile": resolved_profile,
        "bbox_coordinate_mode_requested": args.bbox_coordinate_mode,
        "bbox_coordinate_mode": resolved_bbox_mode,
        "normalize_actions": args.normalize_actions,
        "system_prompt_file": (
            str(args.system_prompt_file.expanduser().resolve())
            if args.system_prompt_file
            else None
        ),
        "wait_after_action_seconds": args.wait_after_action_seconds,
        "include_ui_bboxes": args.include_ui_bboxes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        return task_runner_detail.list_available_tasks()
    try:
        validate_vl_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    resolved_profile = resolve_prompt_profile(
        args.model_path,
        args.prompt_profile,
    )
    return task_runner_detail.run_evaluation(
        args,
        condition="vl_baseline",
        model_factory=create_vl_model,
        agent_factory=lambda env, llm, t3a_module: create_m3a_agent(
            env,
            llm,
            t3a_module,
            wait_after_action_seconds=args.wait_after_action_seconds,
            include_ui_bboxes=args.include_ui_bboxes,
        ),
        model_info=_model_info(args),
        backend_label="vLLM multimodal",
        agent_name=f"m3a_vllm_vl_{resolved_profile}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
