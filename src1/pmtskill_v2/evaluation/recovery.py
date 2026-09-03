"""AndroidWorld 评测期间的设备健康检查与有限自愈。"""

from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import time
from collections.abc import Iterator
from typing import Any

from ..core.config import AndroidWorldConfig


class AndroidWorldInfrastructureError(RuntimeError):
    """模拟器或 AndroidEnv 基础设施已失效，当前评测不能继续。"""


class AndroidWorldPermissionControllerInterruption(RuntimeError):
    """系统权限弹窗打断了 episode；该次尝试必须丢弃并从头执行。"""


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

_PERMISSION_CONTROLLER_PACKAGES = (
    "com.android.permissioncontroller",
    "com.google.android.permissioncontroller",
)
_PERMISSION_INTERRUPTION_MARKER = "pmtskill_permission_controller_interruption"
_PERMISSION_DISMISSED_PATTERN = re.compile(r"dismissed=(\d+)")
_ALLOW_RESOURCE_PRIORITIES = (
    "permission_allow_button",
    "permission_allow_foreground_only_button",
    "permission_allow_always_button",
    "permission_allow_one_time_button",
)
_ALLOW_TEXT_PRIORITIES = (
    "allow",
    "while using the app",
    "only this time",
)
_MAX_PERMISSION_DIALOGS_PER_CHECK = 8


def _exception_text(episode: Any) -> str:
    if not isinstance(episode, dict):
        return ""
    exception = episode.get("exception_info")
    return str(exception).casefold() if exception else ""


def is_permission_controller_interruption(episode: Any) -> bool:
    """判断失败结果是否由本模块主动终止的权限弹窗 episode 产生。"""

    return _PERMISSION_INTERRUPTION_MARKER in _exception_text(episode)


def is_infrastructure_failure(episode: Any) -> bool:
    """判断 AndroidWorld 的 failed episode 是否属于设备基础设施故障。"""

    normalized = _exception_text(episode)
    if not normalized:
        return False
    if _PERMISSION_INTERRUPTION_MARKER in normalized:
        return True
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


def _permission_package(value: Any) -> bool:
    normalized = str(value or "").casefold()
    return normalized in _PERMISSION_CONTROLLER_PACKAGES or normalized.endswith(
        ".permissioncontroller"
    )


def _element_resource_id(element: Any) -> str:
    return str(
        getattr(element, "resource_name", None)
        or getattr(element, "resource_id", None)
        or ""
    ).casefold()


def _element_text(element: Any) -> str:
    return str(
        getattr(element, "text", None)
        or getattr(element, "content_description", None)
        or ""
    ).strip().casefold()


def _permission_allow_button(elements: list[Any]) -> Any | None:
    """只选择明确的 Allow 按钮，绝不误点 Don't allow。"""

    permission_elements = [
        element
        for element in elements
        if _permission_package(getattr(element, "package_name", None))
    ]
    if not permission_elements:
        return None
    for marker in _ALLOW_RESOURCE_PRIORITIES:
        for element in permission_elements:
            if marker in _element_resource_id(element):
                return element
    for label in _ALLOW_TEXT_PRIORITIES:
        for element in permission_elements:
            if _element_text(element) == label:
                return element
    raise AndroidWorldInfrastructureError(
        "检测到 Android permission controller，但没有找到可安全点击的 Allow 按钮；"
        "为避免误点 Don't allow，已停止自动操作。"
    )


def _element_center(element: Any) -> tuple[int, int]:
    bounds = getattr(element, "bbox_pixels", None) or getattr(element, "bbox", None)
    if bounds is None:
        raise AndroidWorldInfrastructureError(
            "检测到 Android 权限 Allow 按钮，但 accessibility tree 没有按钮坐标。"
        )
    try:
        x = round((float(bounds.x_min) + float(bounds.x_max)) / 2)
        y = round((float(bounds.y_min) + float(bounds.y_max)) / 2)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AndroidWorldInfrastructureError("Android 权限按钮坐标无效") from exc
    return x, y


def dismiss_permission_controller_dialogs(
    environment: Any,
    config: AndroidWorldConfig,
) -> int:
    """关闭当前可见的 Android 运行时权限弹窗，并返回点击次数。

    某些 App 会连续请求联系人、电话等多项权限，因此一次检查最多连续处理 8 层。
    只有 permission controller 中资源 ID/文案明确为 Allow 的按钮会被点击。
    """

    controller = getattr(environment, "controller", None)
    if controller is None or not callable(getattr(controller, "get_ui_elements", None)):
        raise RuntimeError("AndroidWorld environment 没有可用的 UI 权限检查接口")
    dismissed = 0
    for _ in range(_MAX_PERMISSION_DIALOGS_PER_CHECK):
        elements = list(controller.get_ui_elements() or ())
        button = _permission_allow_button(elements)
        if button is None:
            return dismissed
        x, y = _element_center(button)
        logging.warning(
            "检测到 Android permission controller，自动点击 Allow (%d, %d)。",
            x,
            y,
        )
        _adb_command(config, "shell", "input", "tap", str(x), str(y))
        dismissed += 1
        if config.permission_controller_settle_seconds:
            time.sleep(config.permission_controller_settle_seconds)
    remaining = list(controller.get_ui_elements() or ())
    if any(
        _permission_package(getattr(element, "package_name", None))
        for element in remaining
    ):
        raise AndroidWorldInfrastructureError(
            "连续处理 Android 权限弹窗达到安全上限，仍检测到 permission controller。"
        )
    return dismissed


def _permission_dismissal_count(episode: Any) -> int:
    match = _PERMISSION_DISMISSED_PATTERN.search(_exception_text(episode))
    return int(match.group(1)) if match else 0


def _annotate_permission_recovery(
    result: Any,
    *,
    restarts: int,
    dialogs_dismissed: int,
) -> None:
    if not isinstance(result, dict) or (restarts <= 0 and dialogs_dismissed <= 0):
        return
    aux_data = result.get("aux_data")
    aux_data = dict(aux_data) if isinstance(aux_data, dict) else {}
    aux_data["permission_controller_restarts"] = restarts
    aux_data["permission_controller_dialogs_dismissed"] = dialogs_dismissed
    result["aux_data"] = aux_data


def _reset_after_permission_interruption(
    task: Any,
    environment: Any,
    config: AndroidWorldConfig,
) -> None:
    """尽力清理被中断任务，然后回到桌面，让下一次 initialize_task 从头开始。"""

    tear_down = getattr(task, "tear_down", None)
    if callable(tear_down):
        try:
            tear_down(environment)
        except Exception:
            logging.warning("权限弹窗中断后的 task tear_down 失败。", exc_info=True)
    _adb_command(config, "shell", "input", "keyevent", "KEYCODE_HOME")
    _refresh_environment(environment)


@contextlib.contextmanager
def _watch_permission_controller(
    suite_utils: Any,
    environment: Any,
    config: AndroidWorldConfig,
) -> Iterator[None]:
    """在模型 step 前后监控权限弹窗；命中时终止并丢弃整个临时 episode。"""

    runner_module = getattr(suite_utils, "episode_runner", None)
    original_run_episode = getattr(runner_module, "run_episode", None)
    if (
        config.permission_controller_recovery_attempts <= 0
        or not callable(original_run_episode)
    ):
        yield
        return

    class PermissionWatchingAgent:
        def __init__(self, wrapped: Any):
            self._wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

        def step(self, goal: str) -> Any:
            before = dismiss_permission_controller_dialogs(environment, config)
            if before:
                raise AndroidWorldPermissionControllerInterruption(
                    f"{_PERMISSION_INTERRUPTION_MARKER}: stage=before_step dismissed={before}"
                )
            result = self._wrapped.step(goal)
            after = dismiss_permission_controller_dialogs(environment, config)
            if after:
                raise AndroidWorldPermissionControllerInterruption(
                    f"{_PERMISSION_INTERRUPTION_MARKER}: stage=after_step dismissed={after}"
                )
            return result

    def watched_run_episode(*args: Any, **kwargs: Any) -> Any:
        dismissed = dismiss_permission_controller_dialogs(environment, config)
        if dismissed:
            raise AndroidWorldPermissionControllerInterruption(
                f"{_PERMISSION_INTERRUPTION_MARKER}: stage=episode_start dismissed={dismissed}"
            )
        positional = list(args)
        if "agent" in kwargs:
            kwargs["agent"] = PermissionWatchingAgent(kwargs["agent"])
        elif len(positional) >= 2:
            positional[1] = PermissionWatchingAgent(positional[1])
        else:
            raise TypeError("run_episode 调用缺少 agent 参数")
        return original_run_episode(*positional, **kwargs)

    runner_module.run_episode = watched_run_episode
    try:
        yield
    finally:
        runner_module.run_episode = original_run_episode


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
    permission_attempts = config.permission_controller_recovery_attempts
    original_run_task = getattr(suite_utils, "_run_task", None)
    if (attempts <= 0 and permission_attempts <= 0) or not callable(original_run_task):
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
        permission_restarts = 0
        dialogs_dismissed = 0

        if permission_attempts > 0:
            preflight_dismissed = dismiss_permission_controller_dialogs(
                environment, config
            )
            if preflight_dismissed:
                dialogs_dismissed += preflight_dismissed
                logging.warning(
                    "task %s 开始前清除了 %d 个遗留权限弹窗；随后从干净状态初始化。",
                    task_name,
                    preflight_dismissed,
                )
                _adb_command(
                    config, "shell", "input", "keyevent", "KEYCODE_HOME"
                )
                _refresh_environment(environment)

        try:
            _probe_environment(environment)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            recover_or_raise(task_name, exc)

        result = original_run_task(*args, **kwargs)
        retries = 0
        while is_infrastructure_failure(result):
            if is_permission_controller_interruption(result):
                if permission_restarts >= permission_attempts:
                    raise AndroidWorldInfrastructureError(
                        f"AndroidWorld task {task_name} 连续 {permission_attempts} 次被 "
                        "permission controller 中断；停止评测，避免把异常记为任务失败。"
                    )
                permission_restarts += 1
                dialogs_dismissed += max(1, _permission_dismissal_count(result))
                logging.warning(
                    "权限弹窗已关闭；丢弃受干扰 episode，并从第 0 step 重跑 task %s "
                    "(%d/%d)。",
                    task_name,
                    permission_restarts,
                    permission_attempts,
                )
                _reset_after_permission_interruption(task, environment, config)
                result = original_run_task(*args, **kwargs)
                continue
            if retries >= attempts:
                break
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
        _annotate_permission_recovery(
            result,
            restarts=permission_restarts,
            dialogs_dismissed=dialogs_dismissed,
        )
        return result

    suite_utils._run_task = guarded_run_task
    try:
        with _watch_permission_controller(suite_utils, environment, config):
            yield
    finally:
        suite_utils._run_task = original_run_task
