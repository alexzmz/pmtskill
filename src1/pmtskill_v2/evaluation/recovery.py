"""AndroidWorld 评测期间的设备健康检查与有限自愈。"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import time
from collections.abc import Iterator
from typing import Any

from ..core.config import AndroidWorldConfig


class AndroidWorldInfrastructureError(RuntimeError):
    """模拟器或 AndroidEnv 基础设施已失效，当前评测不能继续。"""


_INFRASTRUCTURE_FAILURE_MARKERS = (
    "could not get a11y tree",
    "adbcontrollererror",
    "adb command failed",
    "device offline",
    "no devices/emulators found",
    "could not find adb device",
    "failed to connect to emulator",
    "emulator is not running",
)

_CONNECTION_FAILURE_MARKERS = (
    "statuscode.unavailable",
    "_inactiverpcerror",
    "connection refused",
    "connection reset",
)

_ANDROID_CONNECTION_CONTEXT = (
    "android_env",
    "androidenv",
    "accessibility",
    "a11y",
    "emulator",
)


def is_infrastructure_failure(episode: Any) -> bool:
    """判断 AndroidWorld 的 failed episode 是否属于设备基础设施故障。"""

    if not isinstance(episode, dict):
        return False
    exception = episode.get("exception_info")
    if not exception:
        return False
    normalized = str(exception).casefold()
    if any(marker in normalized for marker in _INFRASTRUCTURE_FAILURE_MARKERS):
        return True
    return (
        any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS)
        and any(context in normalized for context in _ANDROID_CONNECTION_CONTEXT)
    )


def _adb_command(
    config: AndroidWorldConfig,
    *arguments: str,
    timeout: float = 30.0,
) -> str:
    command = [
        config.adb_path,
        "-P",
        "5037",
        "-s",
        f"emulator-{config.console_port}",
        *arguments,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise RuntimeError(f"ADB command failed ({completed.returncode}): {detail}")
    return completed.stdout.strip()


def _restore_networking(config: AndroidWorldConfig) -> None:
    """恢复飞行模式相关设置；broadcast 可补足只改 settings 不生效的问题。"""

    commands = (
        ("shell", "settings", "put", "global", "airplane_mode_on", "0"),
        (
            "shell",
            "am",
            "broadcast",
            "-a",
            "android.intent.action.AIRPLANE_MODE",
            "--ez",
            "state",
            "false",
        ),
        ("shell", "svc", "wifi", "enable"),
        ("shell", "input", "keyevent", "KEYCODE_HOME"),
    )
    for command in commands:
        try:
            _adb_command(config, *command)
        except Exception as exc:  # 单项失败后仍允许完整重启接管恢复。
            logging.warning("AndroidWorld 状态恢复命令失败 (%s): %s", command, exc)


def _probe_environment(environment: Any) -> None:
    controller = getattr(environment, "controller", None)
    if controller is None or not callable(getattr(controller, "get_ui_elements", None)):
        raise RuntimeError("AndroidWorld environment 没有可用的 UI 健康检查接口")
    controller.get_ui_elements()


def _refresh_environment(environment: Any) -> None:
    controller = getattr(environment, "controller", None)
    refresh = getattr(controller, "refresh_env", None)
    if not callable(refresh):
        raise RuntimeError("AndroidWorld controller 不支持 refresh_env")
    refresh()
    _probe_environment(environment)


def _wait_for_boot(config: AndroidWorldConfig) -> None:
    deadline = time.monotonic() + config.recovery_timeout_seconds
    last_error = "device did not respond"
    while time.monotonic() < deadline:
        try:
            state = _adb_command(
                config,
                "get-state",
                timeout=min(15.0, config.recovery_timeout_seconds),
            )
            booted = _adb_command(
                config,
                "shell",
                "getprop",
                "sys.boot_completed",
                timeout=min(15.0, config.recovery_timeout_seconds),
            )
            if state.strip() == "device" and booted.strip() == "1":
                return
            last_error = f"state={state!r}, sys.boot_completed={booted!r}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(config.recovery_poll_seconds)
    raise TimeoutError(
        "Android emulator did not finish rebooting within "
        f"{config.recovery_timeout_seconds:g}s ({last_error})"
    )


def recover_android_world_environment(
    environment: Any,
    config: AndroidWorldConfig,
) -> None:
    """先做网络/转发软恢复，失败后重启 Android guest 并重新连接。"""

    logging.warning("AndroidWorld 环境失去 UI 可观测性，开始状态恢复。")
    try:
        if _adb_command(config, "get-state") == "device":
            _restore_networking(config)
            _refresh_environment(environment)
            logging.warning("AndroidWorld 环境已通过软恢复重新可用。")
            return
    except Exception as exc:
        logging.warning("AndroidWorld 软恢复失败，将重启 emulator guest: %s", exc)

    _adb_command(config, "reboot")
    _wait_for_boot(config)
    _restore_networking(config)
    _refresh_environment(environment)
    logging.warning("AndroidWorld emulator guest 重启后已恢复。")


def ensure_valid_evaluation_episodes(
    episodes: Any,
    *,
    expected_episodes: int,
) -> None:
    """拒绝把全异常结果误写成 SR=0 的有效评测阶段。"""

    if expected_episodes <= 0:
        return
    rows = list(episodes or ())
    if any(not row.get("exception_info") for row in rows if isinstance(row, dict)):
        return
    first_error = next(
        (
            str(row.get("exception_info"))
            for row in rows
            if isinstance(row, dict) and row.get("exception_info")
        ),
        "no episode result was returned",
    )
    first_line = (
        first_error.strip().splitlines()[-1]
        if first_error.strip()
        else first_error
    )
    raise AndroidWorldInfrastructureError(
        "AndroidWorld 评测未产生任何有效 episode，拒绝记录伪造的 SR=0；"
        f"首个异常: {first_line}"
    )


@contextlib.contextmanager
def recover_infrastructure_failures(
    suite_utils: Any,
    environment: Any,
    config: AndroidWorldConfig,
) -> Iterator[None]:
    """在 suite 的每个任务前探活，并在基础设施异常后恢复、重试当前任务。"""

    attempts = config.infrastructure_recovery_attempts
    original_run_task = getattr(suite_utils, "_run_task", None)
    if attempts <= 0 or not callable(original_run_task):
        yield
        return

    def recover_or_raise(task_name: str, reason: BaseException | str) -> None:
        latest: BaseException | None = None
        for attempt in range(1, attempts + 1):
            logging.error(
                "AndroidWorld task %s 遇到基础设施故障；恢复尝试 %d/%d。原因: %s",
                task_name,
                attempt,
                attempts,
                reason,
            )
            try:
                recover_android_world_environment(environment, config)
                return
            except BaseException as exc:  # 需保留 KeyboardInterrupt/SystemExit 语义。
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                latest = exc
                logging.exception("AndroidWorld 环境恢复尝试失败")
        raise AndroidWorldInfrastructureError(
            f"AndroidWorld 环境在 {attempts} 次恢复后仍不可用；"
            f"停止评测以避免后续任务全部被记为 0 episode。task={task_name}"
        ) from latest

    def guarded_run_task(*args: Any, **kwargs: Any) -> Any:
        task = args[0] if args else kwargs.get("task")
        task_name = str(getattr(task, "name", "unknown"))

        try:
            _probe_environment(environment)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            recover_or_raise(task_name, exc)

        result = original_run_task(*args, **kwargs)
        retries = 0
        while is_infrastructure_failure(result) and retries < attempts:
            recover_or_raise(task_name, str(result.get("exception_info", "")))
            retries += 1
            logging.warning(
                "AndroidWorld 环境已恢复，重新执行当前 task: %s (%d/%d)",
                task_name,
                retries,
                attempts,
            )
            result = original_run_task(*args, **kwargs)

        if is_infrastructure_failure(result):
            raise AndroidWorldInfrastructureError(
                f"AndroidWorld task {task_name} 在恢复并重试后仍发生基础设施故障；"
                "停止当前评测，避免继续空跑。"
            )
        return result

    suite_utils._run_task = guarded_run_task
    try:
        yield
    finally:
        suite_utils._run_task = original_run_task
