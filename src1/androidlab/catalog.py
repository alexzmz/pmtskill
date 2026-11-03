"""Freeze official task IDs, instructions and evaluator versions, not AW tasks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from ..pmtskill_v2.evaluation.resume import fingerprint


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root):
    root = Path(root)
    files = [p for directory in ("evaluation", "utils_mobile", "templates")
             for p in (root / directory).rglob("*") if p.suffix in {".py", ".yaml", ".json"}]
    files.append(root / "adb_client.py")
    return fingerprint([(p.relative_to(root).as_posix(), file_digest(p)) for p in sorted(files)])


def load_catalog(options, *, tasks=None, task_count=None, seed=42):
    options.validate_source()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 AndroidLab 官方任务需要 PyYAML：pip install pyyaml") from exc
    records, seen = [], set()
    categories = json.loads(options.capability_map.read_text(encoding="utf-8")) if options.capability_map else {}
    for path in sorted((options.root / "evaluation" / "config").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not data.get("APP") or not re.fullmatch(r"[\w.]+", data.get("package", "")):
            raise ValueError(f"Invalid AndroidLab app config: {path}")
        for task in data.get("tasks", []):
            task_id = task["task_id"]
            if task_id in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
                raise ValueError(f"Duplicate/unsafe task_id: {task_id}")
            seen.add(task_id)
            module = task.get("metric_func", "")
            if not re.fullmatch(r"evaluation\.tasks\.[A-Za-z0-9_]+", module):
                raise ValueError(f"Unsupported official metric module: {module}")
            if task.get("metric_type") not in {"operation", "operations", "query_detect"}:
                raise ValueError(f"Unknown official metric_type: {task.get('metric_type')}")
            capability = categories.get(task_id, "unclassified")
            if capability not in {"reasoning", "coordination", "manipulation", "unclassified"}:
                raise ValueError(f"Invalid capability for {task_id}: {capability}")
            records.append({"task_id": task_id, "app": data["APP"], "package": data["package"],
                            "instruction": str(task["task"]).strip(), "metric_type": task["metric_type"].replace("operations", "operation"),
                            "metric_module": module, "adb_query": task.get("adb_query"),
                            "category": task.get("category"), "capability": capability,
                            "config_file": path.relative_to(options.root).as_posix()})
    if not records:
        raise ValueError("AndroidLab task catalog is empty")
    selected = list(dict.fromkeys(part.strip() for value in tasks or []
                                  for part in value.split(",") if part.strip()))
    if selected:
        missing = set(selected) - seen
        if missing:
            raise ValueError(f"Unknown AndroidLab task IDs (not AndroidWorld names): {sorted(missing)}")
        by_id = {r["task_id"]: r for r in records}
        records = [by_id[key] for key in selected]
    elif task_count is not None:
        if task_count <= 0:
            raise ValueError("--task-count must be positive")
        records = sorted(random.Random(seed).sample(records, min(task_count, len(records))),
                         key=lambda r: r["task_id"])
    return records
