"""AndroidWorld 评测期间的设备健康检查与有限自愈。"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any

from ..core.config import AndroidWorldConfig


class AndroidWorldInfrastructureError(RuntimeError):
    """模拟器或 AndroidEnv 基础设施已失效，当前评测不能继续。"""


class AndroidWorldPermissionControllerInterruption(RuntimeError):
    """系统权限弹窗打断了 episode；该次尝试必须丢弃并从头执行。"""


_INFRASTRUCTURE_FAILURE_MARKERS = (
    "could not get a11y tree",
    "device offline",
    "device unauthorized",
    "no devices/emulators found",
    "could not find adb device",
    "failed to connect to emulator",
    "emulator is not running",
    "device not found",
    "transport endpoint is not connected",
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
_UIAUTOMATOR_DUMP_PATH = "/sdcard/pmtskill_window.xml"
_BOUNDS_PATTERN = re.compile(
    r"\[(?P<x_min>-?\d+),(?P<y_min>-?\d+)\]"
    r"\[(?P<x_max>-?\d+),(?P<y_max>-?\d+)\]"
)


@dataclasses.dataclass(frozen=True, slots=True)
class _Bounds:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


@dataclasses.dataclass(frozen=True, slots=True)
class _DumpElement:
    text: str | None
    content_description: str | None
    class_name: str | None
    bbox_pixels: _Bounds | None
    is_clickable: bool | None
    is_enabled: bool | None
    package_name: str | None
    resource_name: str | None = None
    resource_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _PermissionHandlingResult:
    detected: bool = False
    dismissed: int = 0
    delegated_to_model: bool = False
    model_guidance: str | None = None


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
    # Accessibility forwarder 使用 resource_name，uiautomator dump 使用
    # resource_id。两者可能同时存在且其中一个不完整，因此不能用 or 丢弃后者。
    return " ".join(
        str(value)
        for value in (
            getattr(element, "resource_name", None),
            getattr(element, "resource_id", None),
        )
        if value
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
    return None


def _permission_elements(elements: list[Any]) -> list[Any]:
    return [
        element
        for element in elements
        if _permission_package(getattr(element, "package_name", None))
    ]


def _parse_bounds(value: str | None) -> _Bounds | None:
    match = _BOUNDS_PATTERN.fullmatch(str(value or "").strip())
    if not match:
        return None
    values = {name: int(raw) for name, raw in match.groupdict().items()}
    return _Bounds(
        x_min=values["x_min"],
        x_max=values["x_max"],
        y_min=values["y_min"],
        y_max=values["y_max"],
    )


def _uiautomator_permission_elements(
    config: AndroidWorldConfig,
) -> list[_DumpElement]:
    """使用 uiautomator 获取权限弹窗的完整节点，补足 a11y 漏掉的按钮。"""

    try:
        _adb_command(
            config,
            "shell",
            "uiautomator",
            "dump",
            _UIAUTOMATOR_DUMP_PATH,
        )
        raw_xml = _adb_command(
            config,
            "shell",
            "cat",
            _UIAUTOMATOR_DUMP_PATH,
        )
        xml_start = raw_xml.find("<?xml")
        if xml_start < 0:
            xml_start = raw_xml.find("<hierarchy")
        if xml_start < 0:
            raise ValueError("uiautomator 输出中没有 XML hierarchy")
        root = ET.fromstring(raw_xml[xml_start:])
    except Exception:
        # UI dump 是补充观测，失败不应让训练或评测退出。
        logging.warning(
            "读取 Android permission controller 的 uiautomator dump 失败；"
            "继续使用 accessibility tree。",
            exc_info=True,
        )
        return []

    elements: list[_DumpElement] = []
    for node in root.iter("node"):
        package_name = node.attrib.get("package")
        if not _permission_package(package_name):
            continue
        elements.append(
            _DumpElement(
                text=node.attrib.get("text") or None,
                content_description=node.attrib.get("content-desc") or None,
                class_name=node.attrib.get("class") or None,
                bbox_pixels=_parse_bounds(node.attrib.get("bounds")),
                is_clickable=node.attrib.get("clickable") == "true",
                is_enabled=node.attrib.get("enabled") != "false",
                package_name=package_name,
                resource_id=node.attrib.get("resource-id") or None,
            )
        )
    return elements


def _is_actionable_permission_element(element: Any) -> bool:
    resource_id = _element_resource_id(element)
    class_name = str(getattr(element, "class_name", None) or "").casefold()
    return bool(
        getattr(element, "is_clickable", None) is True
        or "button" in resource_id
        or class_name.endswith("button")
    )


def _element_bounds_text(element: Any) -> str:
    bounds = getattr(element, "bbox_pixels", None) or getattr(element, "bbox", None)
    if bounds is None:
        return "unknown"
    try:
        return (
            f"[{round(float(bounds.x_min))},{round(float(bounds.y_min))}]"
            f"[{round(float(bounds.x_max))},{round(float(bounds.y_max))}]"
        )
    except (AttributeError, TypeError, ValueError):
        return "invalid"


def _permission_action_descriptions(elements: list[Any]) -> list[str]:
    descriptions: list[str] = []
    for index, element in enumerate(
        item for item in elements if _is_actionable_permission_element(item)
    ):
        center = None
        try:
            center = _element_center(element)
        except AndroidWorldInfrastructureError:
            pass
        descriptions.append(
            f"candidate[{index}] text={getattr(element, 'text', None)!r}, "
            f"content_desc={getattr(element, 'content_description', None)!r}, "
            f"resource_id={_element_resource_id(element)!r}, "
            f"class={getattr(element, 'class_name', None)!r}, "
            f"clickable={getattr(element, 'is_clickable', None)!r}, "
            f"enabled={getattr(element, 'is_enabled', None)!r}, "
            f"bounds={_element_bounds_text(element)}, center={center!r}"
        )
    return descriptions


def _log_permission_dialog(elements: list[Any], *, source: str) -> None:
    messages = list(
        dict.fromkeys(
            text
            for element in elements
            if (text := _element_text(element))
            and not _is_actionable_permission_element(element)
        )
    )
    actions = _permission_action_descriptions(elements)
    logging.warning(
        "检测到 Android permission controller（source=%s）。\n"
        "弹窗文本: %s\n可交互控件:\n%s",
        source,
        messages or ["<none>"],
        "\n".join(actions) if actions else "<none>",
    )


def _permission_model_guidance(elements: list[Any]) -> str:
    actions = _permission_action_descriptions(elements)
    choices = "\n".join(actions) if actions else "没有结构化按钮，请结合截图判断。"
    return (
        "Android 系统权限弹窗正在遮挡目标应用，自动处理器无法确定安全的 "
        "Allow 选项。请根据任务和截图从以下控件中选择最合适的一项，优先授予"
        "目标应用完成任务所必需的权限；不要因此将任务标记为完成或 infeasible。"
        "若 UI element 没有索引，可以使用控件 center 给出的像素坐标执行 "
        '{"action_type":"click","x":<x>,"y":<y>}。如果选择后应用无法继续，'
        "请回到桌面，重新打开目标应用后继续。\n候选控件:\n"
        f"{choices}"
    )


def _permission_dialog_signature(elements: list[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{_element_text(element)}|{_element_resource_id(element)}|"
            f"{_element_bounds_text(element)}"
            for element in elements
        )
    )


def _inspect_permission_dialog(
    environment: Any,
    config: AndroidWorldConfig,
) -> tuple[list[Any], str]:
    controller = getattr(environment, "controller", None)
    if controller is None or not callable(getattr(controller, "get_ui_elements", None)):
        raise RuntimeError("AndroidWorld environment 没有可用的 UI 权限检查接口")
    accessibility_elements = list(controller.get_ui_elements() or ())
    permission_elements = _permission_elements(accessibility_elements)
    if not permission_elements:
        return [], "accessibility"

    # 如果 a11y 已给出明确 Allow，直接使用；否则复现人工排障中可靠的
    # `uiautomator dump` 路径，获取 permission controller 的完整按钮集合。
    accessibility_allow = _permission_allow_button(permission_elements)
    if accessibility_allow is not None:
        try:
            _element_center(accessibility_allow)
            return permission_elements, "accessibility"
        except AndroidWorldInfrastructureError:
            pass
    dumped = _uiautomator_permission_elements(config)
    return (dumped, "uiautomator") if dumped else (permission_elements, "accessibility")


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


def _handle_permission_controller_dialogs(
    environment: Any,
    config: AndroidWorldConfig,
) -> _PermissionHandlingResult:
    """自动同意明确权限；不明确的选项留给模型，不抛出致命异常。"""

    dismissed = 0
    seen_dialogs: set[tuple[str, ...]] = set()
    for _ in range(_MAX_PERMISSION_DIALOGS_PER_CHECK):
        elements, source = _inspect_permission_dialog(environment, config)
        if not elements:
            return _PermissionHandlingResult(
                detected=dismissed > 0,
                dismissed=dismissed,
            )
        _log_permission_dialog(elements, source=source)
        signature = _permission_dialog_signature(elements)
        if signature in seen_dialogs:
            logging.warning(
                "自动点击后权限弹窗内容没有变化；停止重复点击同一坐标，"
                "保留界面交给模型继续处理。"
            )
            return _PermissionHandlingResult(
                detected=True,
                dismissed=dismissed,
                delegated_to_model=True,
                model_guidance=_permission_model_guidance(elements),
            )
        seen_dialogs.add(signature)
        button = _permission_allow_button(elements)
        if button is None:
            guidance = _permission_model_guidance(elements)
            logging.warning(
                "权限弹窗没有可明确识别的 Allow 按钮；不终止当前任务，"
                "将候选控件交给模型判断。"
            )
            return _PermissionHandlingResult(
                detected=True,
                dismissed=dismissed,
                delegated_to_model=True,
                model_guidance=guidance,
            )
        try:
            x, y = _element_center(button)
            logging.warning(
                "自动点击 Android permission Allow: text=%r, resource_id=%r, "
                "center=(%d, %d)。",
                getattr(button, "text", None),
                _element_resource_id(button),
                x,
                y,
            )
            _adb_command(config, "shell", "input", "tap", str(x), str(y))
        except Exception:
            logging.warning(
                "Android 权限 Allow 按钮自动点击失败；不终止当前任务，"
                "改由模型根据当前界面处理。",
                exc_info=True,
            )
            return _PermissionHandlingResult(
                detected=True,
                dismissed=dismissed,
                delegated_to_model=True,
                model_guidance=_permission_model_guidance(elements),
            )
        dismissed += 1
        if config.permission_controller_settle_seconds:
            time.sleep(config.permission_controller_settle_seconds)

    remaining, source = _inspect_permission_dialog(environment, config)
    if remaining:
        _log_permission_dialog(remaining, source=source)
        logging.warning(
            "连续自动处理权限弹窗达到 %d 层；不终止任务，剩余界面交给模型。",
            _MAX_PERMISSION_DIALOGS_PER_CHECK,
        )
        return _PermissionHandlingResult(
            detected=True,
            dismissed=dismissed,
            delegated_to_model=True,
            model_guidance=_permission_model_guidance(remaining),
        )
    return _PermissionHandlingResult(detected=True, dismissed=dismissed)


def dismiss_permission_controller_dialogs(
    environment: Any,
    config: AndroidWorldConfig,
) -> int:
    """关闭可明确同意的 Android 权限弹窗，并返回自动点击次数。

    某些 App 会连续请求联系人、电话等多项权限，因此一次检查最多连续处理 8 层。
    只有 resource ID/文案明确为 Allow 的按钮会被点击；其余选项留给模型判断。
    任何权限识别或点击问题都不会从这里终止整场训练/评测。
    """

    return _handle_permission_controller_dialogs(environment, config).dismissed


def _permission_dismissal_count(episode: Any) -> int:
    match = _PERMISSION_DISMISSED_PATTERN.search(_exception_text(episode))
    return int(match.group(1)) if match else 0


def _annotate_permission_recovery(
    result: Any,
    *,
    restarts: int,
    dialogs_dismissed: int,
    model_delegations: int = 0,
) -> None:
    if not isinstance(result, dict) or not any(
        value > 0 for value in (restarts, dialogs_dismissed, model_delegations)
    ):
        return
    aux_data = result.get("aux_data")
    aux_data = dict(aux_data) if isinstance(aux_data, dict) else {}
    aux_data["permission_controller_restarts"] = restarts
    aux_data["permission_controller_dialogs_dismissed"] = dialogs_dismissed
    aux_data["permission_controller_model_delegations"] = model_delegations
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
    activity: dict[str, int],
) -> Iterator[None]:
    """在模型 step 前后处理权限弹窗，不中断当前 episode。"""

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
            before = _handle_permission_controller_dialogs(environment, config)
            activity["dismissed"] += before.dismissed
            activity["model_delegations"] += int(before.delegated_to_model)

            # M3A 原生支持 additional_guidelines。临时注入 uiautomator 提取的
            # 按钮文字/坐标，让模型在 a11y 漏掉按钮节点时仍能二选一。
            has_guidelines = hasattr(self._wrapped, "additional_guidelines")
            original_guidelines = (
                getattr(self._wrapped, "additional_guidelines", None)
                if has_guidelines
                else None
            )
            if before.model_guidance and has_guidelines:
                guidelines = list(original_guidelines or ())
                guidelines.append(before.model_guidance)
                self._wrapped.additional_guidelines = guidelines
            try:
                result = self._wrapped.step(goal)
            finally:
                if before.model_guidance and has_guidelines:
                    self._wrapped.additional_guidelines = original_guidelines

            after = _handle_permission_controller_dialogs(environment, config)
            activity["dismissed"] += after.dismissed
            activity["model_delegations"] += int(after.delegated_to_model)
            return result

    def watched_run_episode(*args: Any, **kwargs: Any) -> Any:
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
    """拒绝把基础设施全挂误写成 SR=0，同时允许任务级异常被报告。"""

    if expected_episodes <= 0:
        return
    rows = list(episodes or ())
    if not rows:
        raise AndroidWorldInfrastructureError(
            "AndroidWorld 评测未返回任何 episode，拒绝记录伪造的 SR=0。"
        )
    if any(not row.get("exception_info") for row in rows if isinstance(row, dict)):
        return
    if not any(is_infrastructure_failure(row) for row in rows):
        logging.warning(
            "AndroidWorld 本次返回的 %d 个 episode 均为任务级异常；"
            "保留评测结果并继续，不将其误判为模拟器基础设施故障。",
            len(rows),
        )
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
    permission_activity = {"dismissed": 0, "model_delegations": 0}
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
        dismissed_at_start = permission_activity["dismissed"]
        delegations_at_start = permission_activity["model_delegations"]

        if permission_attempts > 0:
            try:
                preflight = _handle_permission_controller_dialogs(
                    environment, config
                )
            except Exception:
                logging.warning(
                    "task %s 开始前的权限弹窗检查失败；不直接终止，"
                    "继续执行环境探活。",
                    task_name,
                    exc_info=True,
                )
                preflight = _PermissionHandlingResult()
            permission_activity["dismissed"] += preflight.dismissed
            permission_activity["model_delegations"] += int(
                preflight.delegated_to_model
            )
            if preflight.detected:
                logging.warning(
                    "task %s 开始前检测到遗留权限弹窗（自动关闭=%d，交给模型=%s）；"
                    "先回到桌面，再从干净状态初始化。",
                    task_name,
                    preflight.dismissed,
                    preflight.delegated_to_model,
                )
                try:
                    _adb_command(
                        config, "shell", "input", "keyevent", "KEYCODE_HOME"
                    )
                    _refresh_environment(environment)
                except Exception:
                    logging.warning(
                        "权限弹窗预处理后回桌面失败；继续执行并交由后续环境探活处理。",
                        exc_info=True,
                    )

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
                    logging.error(
                        "AndroidWorld task %s 的旧式 permission interruption 已达到 "
                        "%d 次；保留当前失败 episode 并继续后续任务，不终止整场评测。",
                        task_name,
                        permission_attempts,
                    )
                    break
                permission_restarts += 1
                permission_activity["dismissed"] += max(
                    1, _permission_dismissal_count(result)
                )
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

        if is_infrastructure_failure(result) and not is_permission_controller_interruption(
            result
        ):
            raise AndroidWorldInfrastructureError(
                f"AndroidWorld task {task_name} 在恢复并重试后仍发生基础设施故障；"
                "停止当前评测，避免继续空跑。"
            )
        _annotate_permission_recovery(
            result,
            restarts=permission_restarts,
            dialogs_dismissed=(
                permission_activity["dismissed"] - dismissed_at_start
            ),
            model_delegations=(
                permission_activity["model_delegations"] - delegations_at_start
            ),
        )
        return result

    suite_utils._run_task = guarded_run_task
    try:
        with _watch_permission_controller(
            suite_utils, environment, config, permission_activity
        ):
            yield
    finally:
        suite_utils._run_task = original_run_task
