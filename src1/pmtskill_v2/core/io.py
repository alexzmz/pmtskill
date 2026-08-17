"""小型、原子化的 JSON/JSONL 读写工具。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import PrimitiveSpec


def write_json_atomic(path: Path, value: Any) -> None:
    """先写同目录临时文件再替换，避免进程中断留下半个报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """原子写 JSONL，返回样本数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL，忽略空行并在错误中包含行号。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc


def load_primitives(path: Path | None = None) -> list[PrimitiveSpec]:
    """读取 PPT 约定的 26 个可配置原语。"""

    resource = path or Path(__file__).resolve().parents[2] / "resources" / "primitives.json"
    with resource.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    primitives = [PrimitiveSpec.from_dict(item) for item in values]
    ids = [item.primitive_id for item in primitives]
    if len(ids) != len(set(ids)):
        raise ValueError(f"原语定义存在重复 ID: {resource}")
    return primitives
