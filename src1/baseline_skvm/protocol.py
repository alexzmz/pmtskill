"""Protocol provenance prevents reusing episodes produced by a broken TCP."""

from __future__ import annotations

import json
from pathlib import Path

from src.skvm_protocol import SKVM_TOOL_PROTOCOL, SKVM_CONTEXT_POLICY

SKVM_EVALUATION_PROTOCOL = SKVM_TOOL_PROTOCOL + "-" + SKVM_CONTEXT_POLICY


def has_current_protocol(folder: Path) -> bool:
    marker = folder / "skvm" / "kernel-protocol.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        return value.get("protocol") == SKVM_TOOL_PROTOCOL and value.get("context_policy") == SKVM_CONTEXT_POLICY
    except (OSError, ValueError, AttributeError):
        return False


def has_episode_artifacts(folder: Path) -> bool:
    checkpoints = folder / "checkpoints"
    return (folder / "summary.json").exists() or (
        checkpoints.is_dir() and any(checkpoints.rglob("*.json"))
    ) or (checkpoints.is_dir() and any(checkpoints.rglob("*.pkl.gz")))


def output_for_protocol(folder: Path) -> Path:
    """Preserve old results; only SKVM moves to a new nested experiment."""
    if has_episode_artifacts(folder) and not has_current_protocol(folder):
        return folder / SKVM_EVALUATION_PROTOCOL
    return folder


def validate_protocol_resume(folder: Path) -> None:
    if has_episode_artifacts(folder) and not has_current_protocol(folder):
        raise ValueError(
            "此 SKVM 目录已有旧工具/上下文协议产生的 episode，不能与修复后的画像混跑；"
            "请使用新的 --output-dir（batch_evaluate 会自动创建协议隔离子目录）。")
