"""Protocol provenance prevents reusing episodes produced by a broken TCP."""

from __future__ import annotations

import json
from pathlib import Path

from src.skvm_protocol import SKVM_TOOL_PROTOCOL, SKVM_CONTEXT_POLICY
from src.skvm_stability import SKVM_STABILITY_POLICY

SKVM_EVALUATION_PROTOCOL = SKVM_TOOL_PROTOCOL + "-" + SKVM_CONTEXT_POLICY + "-" + SKVM_STABILITY_POLICY


def has_current_protocol(folder: Path, failure_policy="fallback") -> bool:
    marker = folder / "skvm" / "kernel-protocol.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        return (value.get("protocol") == SKVM_TOOL_PROTOCOL and value.get("context_policy") == SKVM_CONTEXT_POLICY
                and value.get("preparation_policy") == (SKVM_STABILITY_POLICY if failure_policy == "fallback" else "strict"))
    except (OSError, ValueError, AttributeError):
        return False


def has_episode_artifacts(folder: Path) -> bool:
    checkpoints = folder / "checkpoints"
    return (folder / "summary.json").exists() or (
        checkpoints.is_dir() and any(checkpoints.rglob("*.json"))
    ) or (checkpoints.is_dir() and any(checkpoints.rglob("*.pkl.gz")))


def output_for_protocol(folder: Path, failure_policy="fallback") -> Path:
    """Preserve old results; only SKVM moves to a new nested experiment."""
    if has_episode_artifacts(folder) and not has_current_protocol(folder, failure_policy):
        return folder / (SKVM_EVALUATION_PROTOCOL if failure_policy == "fallback" else SKVM_EVALUATION_PROTOCOL + "-strict")
    return folder


def validate_protocol_resume(folder: Path, failure_policy="fallback") -> None:
    if has_episode_artifacts(folder) and not has_current_protocol(folder, failure_policy):
        raise ValueError(
            "此 SKVM 目录已有旧工具/上下文协议产生的 episode，不能与修复后的画像混跑；"
            "请使用新的 --output-dir（batch_evaluate 会自动创建协议隔离子目录）。")
