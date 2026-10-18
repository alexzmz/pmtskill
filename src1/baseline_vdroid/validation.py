"""Lightweight guards shared by the parent CLI and the isolated worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class VDroidIntegrationError(RuntimeError):
    """The backend/upstream interface is broken, not a failed GUI task."""


def require_available_evaluation(summary: Any) -> None:
    """An exited worker with no valid episodes is not a successful evaluation."""
    if isinstance(summary, Mapping):
        evaluated = summary.get("episodes_evaluated", 0)
        if (summary.get("evaluation_available") is True
                and isinstance(evaluated, (int, float)) and evaluated > 0):
            return
        total = summary.get("episodes_total", "unknown")
    else:
        evaluated, total = 0, "unknown"
    raise RuntimeError(
        f"V-Droid evaluation unavailable: {evaluated} valid episodes / {total}; "
        "this is not a measured SR=0. See report.md and CLI runtime.log/errors.log."
    )
