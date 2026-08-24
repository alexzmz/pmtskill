"""CLI 单次运行的统一日志与最终结果归档。

设计目标：

1. 每次 CLI 子命令都拥有独立目录，避免多次实验互相覆盖；
2. ``runtime.log`` 完整保留 print、warning、error 和 traceback；
3. ``result.json`` 面向程序读取，``result.md`` 面向实验结束后人工查看；
4. 即使命令异常，仍尽量写出状态、错误类型和 traceback。

本模块位于 CLI 最外层，不侵入 collect/evaluate 的算法实现，因此以后增加新的
训练器、路由器或子命令时也会自动获得同样的日志行为。
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime as dt
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import traceback as traceback_module
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from .io import write_json_atomic


_ACTIVE_RUN: contextvars.ContextVar["CommandRunLogger | None"] = (
    contextvars.ContextVar("pmtskill_active_run", default=None)
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _jsonable(value: Any) -> Any:
    """把 Path、dataclass、tuple 等对象转换成稳定的 JSON 值。"""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


def _slug(value: str, maximum: int = 96) -> str:
    """生成跨 Windows/Linux 都安全的短目录名。"""

    normalized = re.sub(r"[^a-zA-Z0-9_.+-]+", "-", value.strip()).strip("-._")
    return (normalized or "run")[:maximum]


class _TeeStream:
    """把 stdout/stderr 同时写到原终端和一个或多个日志文件。"""

    def __init__(self, console: TextIO, *files: TextIO):
        self.console = console
        self.files = files
        self._lock = threading.RLock()

    @property
    def encoding(self) -> str:
        return getattr(self.console, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str | None:
        return getattr(self.console, "errors", None)

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.console.fileno()

    def write(self, text: str) -> int:
        with self._lock:
            written = self.console.write(text)
            self.console.flush()
            clean = _ANSI_ESCAPE.sub("", text)
            for handle in self.files:
                handle.write(clean)
                handle.flush()
            return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        with self._lock:
            self.console.flush()
            for handle in self.files:
                handle.flush()

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        """兼容少数会读取 stdout.name/buffer 等属性的第三方库。"""

        return getattr(self.console, name)


@dataclasses.dataclass(slots=True)
class RunLogArtifacts:
    """一次 CLI 调用固定产生的日志文件。"""

    run_dir: Path
    runtime_log: Path
    errors_log: Path
    run_json: Path
    result_json: Path
    result_markdown: Path

    def to_dict(self) -> dict[str, str]:
        return {
            field.name: str(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


def _markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| 指标 | 数值 |", "|---|---:|"]
    for key, value in rows:
        if isinstance(value, float):
            rendered = f"{value:.6g}"
        elif isinstance(value, (list, tuple)):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        rendered = rendered.replace("|", "\\|")
        lines.append(f"| {key} | {rendered} |")
    return lines


def _render_result_markdown(envelope: dict[str, Any]) -> str:
    """将通用结果转换成可快速浏览的 Markdown；完整数据仍保存在 JSON。"""

    run = envelope["run"]
    result = envelope.get("result")
    lines = [
        f"# PMT-Skill 运行结果：{run['command']}",
        "",
        "## 运行状态",
        "",
        *_markdown_table(
            [
                ("状态", run["status"]),
                ("退出码", run["exit_code"]),
                ("开始时间", run["started_at"]),
                ("结束时间", run["finished_at"]),
                ("耗时（秒）", run["duration_seconds"]),
                ("运行目录", run["run_dir"]),
            ]
        ),
    ]
    if envelope.get("error"):
        error = envelope["error"]
        lines.extend(
            [
                "",
                "## 错误",
                "",
                f"- 类型：`{error.get('type', 'UnknownError')}`",
                f"- 信息：{error.get('message', '')}",
                f"- 完整 traceback：`{run['errors_log']}`",
            ]
        )

    if isinstance(result, dict):
        # collect/evaluate 都会把核心指标放在 summary；其他命令直接展示顶层标量。
        summary = result.get("summary")
        metric_source = summary if isinstance(summary, dict) else result
        simple_rows = [
            (str(key), value)
            for key, value in metric_source.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        ]
        if simple_rows:
            lines.extend(["", "## 最终指标", "", *_markdown_table(simple_rows)])

        per_task = metric_source.get("per_task")
        if isinstance(per_task, dict) and per_task:
            lines.extend([
                "",
                "## 每任务结果",
                "",
                "| 任务 | 成功/总数 | SR | 平均步数 |",
                "|---|---:|---:|---:|",
            ])
            for task, row in sorted(per_task.items()):
                if not isinstance(row, dict):
                    continue
                successes = row.get("successes", 0)
                episodes = row.get("episodes", row.get("trials", 0))
                rate = float(row.get("success_rate", 0.0) or 0.0)
                average_steps = float(row.get("average_steps", 0.0) or 0.0)
                lines.append(
                    f"| {task} | {successes}/{episodes} | {rate:.2%} | {average_steps:.2f} |"
                )

        primitive_rows = metric_source.get("per_primitive") or metric_source.get(
            "primitive_metrics"
        )
        if isinstance(primitive_rows, dict) and primitive_rows:
            lines.extend(
                ["", "## 每原语结果", "", "| 原语 | 成功/调用 | SR |", "|---|---:|---:|"]
            )
            for primitive, row in sorted(primitive_rows.items()):
                if not isinstance(row, dict):
                    continue
                successes = int(row.get("successes", 0) or 0)
                trials = int(row.get("trials", row.get("calls", 0)) or 0)
                rate = float(row.get("success_rate", 0.0) or 0.0)
                lines.append(f"| {primitive} | {successes}/{trials} | {rate:.2%} |")

        artifact_rows = [
            (str(key), value)
            for key, value in result.items()
            if isinstance(value, str)
            and any(token in key for token in ("path", "dir", "json", "markdown", "log"))
        ]
        if artifact_rows:
            lines.extend(["", "## 产物", "", *_markdown_table(artifact_rows)])
    elif result is not None:
        lines.extend(["", "## 结果", "", f"`{result}`"])

    lines.extend(
        [
            "",
            "## 日志文件",
            "",
            f"- 运行时完整输出：`{run['runtime_log']}`",
            f"- warning/error/traceback：`{run['errors_log']}`",
            f"- 机器可读最终结果：`{run['result_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


class CommandRunLogger:
    """管理一次 CLI 命令从启动到退出的全部日志。"""

    def __init__(
        self,
        log_root: str | Path,
        *,
        command: str,
        label: str = "",
        argv: list[str] | None = None,
        arguments: Mapping[str, Any] | None = None,
        log_level: str = "INFO",
    ):
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        suffix = f"_{_slug(label, 56)}" if label else ""
        run_name = f"{stamp}_{_slug(command, 32)}{suffix}_{uuid.uuid4().hex[:8]}"
        run_dir = Path(log_root).expanduser().resolve() / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        self.artifacts = RunLogArtifacts(
            run_dir=run_dir,
            runtime_log=run_dir / "runtime.log",
            errors_log=run_dir / "errors.log",
            run_json=run_dir / "run.json",
            result_json=run_dir / "result.json",
            result_markdown=run_dir / "result.md",
        )
        self.command = command
        self.argv = list(argv or ())
        self.arguments = {
            str(key): _jsonable(value)
            for key, value in dict(arguments or {}).items()
            if key != "handler"
        }
        self.log_level = log_level.upper()
        self.started_wall = time.time()
        self.started_perf = time.perf_counter()
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.last_result: Any = None
        self._token: contextvars.Token | None = None
        self._runtime_handle: TextIO | None = None
        self._errors_handle: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._root_handlers: list[logging.Handler] = []
        self._root_level = logging.NOTSET
        self._capture_warnings = False
        self._write_run_metadata(status="running", exit_code=None)

    def _run_metadata(
        self,
        *,
        status: str,
        exit_code: int | None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": self.artifacts.run_dir.name,
            "command": self.command,
            "status": status,
            "exit_code": exit_code,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "run_dir": str(self.artifacts.run_dir),
            "runtime_log": str(self.artifacts.runtime_log),
            "errors_log": str(self.artifacts.errors_log),
            "result_json": str(self.artifacts.result_json),
            "result_markdown": str(self.artifacts.result_markdown),
            "argv": self.argv,
            "arguments": self.arguments,
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        }

    def _write_run_metadata(
        self,
        *,
        status: str,
        exit_code: int | None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        metadata = self._run_metadata(
            status=status,
            exit_code=exit_code,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        write_json_atomic(self.artifacts.run_json, metadata)
        return metadata

    @contextlib.contextmanager
    def capture(self) -> Iterator["CommandRunLogger"]:
        """重定向 Python 输出并配置 root logger，退出时恢复原始进程状态。"""

        self._runtime_handle = self.artifacts.runtime_log.open(
            "a", encoding="utf-8", buffering=1
        )
        self._errors_handle = self.artifacts.errors_log.open(
            "a", encoding="utf-8", buffering=1
        )
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._runtime_handle)  # type: ignore[assignment]
        sys.stderr = _TeeStream(  # type: ignore[assignment]
            self._stderr, self._runtime_handle, self._errors_handle
        )

        root = logging.getLogger()
        self._root_handlers = list(root.handlers)
        self._root_level = root.level
        for handler in self._root_handlers:
            root.removeHandler(handler)
        level = getattr(logging, self.log_level, logging.INFO)
        root.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(self._stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        runtime_handler = logging.FileHandler(
            self.artifacts.runtime_log, encoding="utf-8"
        )
        runtime_handler.setLevel(level)
        runtime_handler.setFormatter(formatter)
        errors_handler = logging.FileHandler(
            self.artifacts.errors_log, encoding="utf-8"
        )
        errors_handler.setLevel(logging.WARNING)
        errors_handler.setFormatter(formatter)
        for handler in (console_handler, runtime_handler, errors_handler):
            root.addHandler(handler)
        logging.captureWarnings(True)
        self._capture_warnings = True
        self._token = _ACTIVE_RUN.set(self)
        try:
            print(f"[PMT-Skill] 本次运行日志目录：{self.artifacts.run_dir}")
            logging.info("CLI run started: command=%s", self.command)
            yield self
        finally:
            if self._token is not None:
                _ACTIVE_RUN.reset(self._token)
                self._token = None
            if self._capture_warnings:
                logging.captureWarnings(False)
                self._capture_warnings = False
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.flush()
                handler.close()
            for handler in self._root_handlers:
                root.addHandler(handler)
            root.setLevel(self._root_level)
            sys.stdout, sys.stderr = self._stdout, self._stderr  # type: ignore[assignment]
            self._runtime_handle.close()
            self._errors_handle.close()

    def record_result(self, value: Any) -> None:
        """记录命令最近一次结构化输出；maintain watch 会保留最后一个周期。"""

        self.last_result = _jsonable(value)

    def finalize(
        self,
        exit_code: int,
        *,
        error: BaseException | None = None,
        traceback_text: str | None = None,
    ) -> dict[str, Any]:
        """无论成功、失败或中断都写最终 JSON/Markdown。"""

        finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        duration = round(time.perf_counter() - self.started_perf, 6)
        status = (
            "success"
            if exit_code == 0
            else "interrupted"
            if exit_code == 130
            else "failed"
        )
        run = self._write_run_metadata(
            status=status,
            exit_code=exit_code,
            finished_at=finished_at,
            duration_seconds=duration,
        )
        error_value = None
        if error is not None:
            error_value = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback_text
                or "".join(
                    traceback_module.format_exception(type(error), error, error.__traceback__)
                ),
            }
        envelope = {
            "schema_version": 1,
            "run": run,
            "result": self.last_result,
            "error": error_value,
        }
        write_json_atomic(self.artifacts.result_json, envelope)
        self.artifacts.result_markdown.write_text(
            _render_result_markdown(envelope), encoding="utf-8"
        )
        logging.info(
            "CLI run finished: status=%s exit_code=%s duration=%.3fs",
            status,
            exit_code,
            duration,
        )
        print(f"[PMT-Skill] 最终结果：{self.artifacts.result_markdown}")
        return envelope


def active_run_logger() -> CommandRunLogger | None:
    """返回当前 CLI 的日志会话；非 CLI 调用时为 ``None``。"""

    return _ACTIVE_RUN.get()
