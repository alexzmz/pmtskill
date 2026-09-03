"""AndroidWorld 基础设施恢复与空评测熔断测试。"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from src1.pmtskill_v2.core.config import AndroidWorldConfig
from src1.pmtskill_v2.evaluation.recovery import (
    AndroidWorldInfrastructureError,
    dismiss_permission_controller_dialogs,
    ensure_valid_evaluation_episodes,
    is_infrastructure_failure,
    is_permission_controller_interruption,
    recover_android_world_environment,
    recover_infrastructure_failures,
)


class _Controller:
    def __init__(self) -> None:
        self.refreshes = 0
        self.probes = 0

    def refresh_env(self) -> None:
        self.refreshes += 1

    def get_ui_elements(self) -> list[object]:
        self.probes += 1
        return []


class AndroidWorldRecoveryTest(unittest.TestCase):
    def test_permission_dialog_clicks_allow_by_resource_id(self):
        state = {"visible": True}
        bounds = types.SimpleNamespace(
            x_min=133, x_max=947, y_min=1231, y_max=1378
        )

        class PermissionController(_Controller):
            def get_ui_elements(self) -> list[object]:
                self.probes += 1
                if not state["visible"]:
                    return []
                return [
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_message",
                        resource_id=None,
                        text="Allow SMS Messenger to access your contacts?",
                        content_description=None,
                        bbox_pixels=None,
                        bbox=None,
                    ),
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_allow_button",
                        resource_id=None,
                        text="Allow",
                        content_description=None,
                        bbox_pixels=bounds,
                        bbox=None,
                    ),
                ]

        controller = PermissionController()
        environment = types.SimpleNamespace(controller=controller)
        config = AndroidWorldConfig(permission_controller_settle_seconds=0)

        def adb_result(_config, *arguments, **kwargs):
            del _config, kwargs
            if arguments[:3] == ("shell", "input", "tap"):
                state["visible"] = False
            return ""

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery._adb_command",
            side_effect=adb_result,
        ) as adb:
            dismissed = dismiss_permission_controller_dialogs(
                environment, config
            )

        self.assertEqual(dismissed, 1)
        self.assertIn(
            ("shell", "input", "tap", "540", "1304"),
            [call.args[1:] for call in adb.call_args_list],
        )

    def test_permission_dialog_falls_back_to_uiautomator_dump(self):
        state = {"visible": True}

        class IncompletePermissionController(_Controller):
            def get_ui_elements(self) -> list[object]:
                self.probes += 1
                if not state["visible"]:
                    return []
                # 复现真实问题：a11y forwarder 只返回说明文字，漏掉按钮。
                return [
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_message",
                        resource_id=None,
                        text="Allow SMS Messenger to access your contacts?",
                        content_description=None,
                        class_name="android.widget.TextView",
                        is_clickable=False,
                        is_enabled=True,
                        bbox_pixels=None,
                        bbox=None,
                    )
                ]

        xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node text="Allow SMS Messenger to access your contacts?"
    resource-id="com.android.permissioncontroller:id/permission_message"
    class="android.widget.TextView"
    package="com.google.android.permissioncontroller"
    clickable="false" enabled="true" bounds="[133,1030][947,1163]" />
  <node text="Allow"
    resource-id="com.android.permissioncontroller:id/permission_allow_button"
    class="android.widget.Button"
    package="com.google.android.permissioncontroller"
    clickable="true" enabled="true" bounds="[133,1231][947,1378]" />
  <node text="Don’t allow"
    resource-id="com.android.permissioncontroller:id/permission_deny_button"
    class="android.widget.Button"
    package="com.google.android.permissioncontroller"
    clickable="true" enabled="true" bounds="[133,1388][947,1535]" />
</hierarchy>"""
        environment = types.SimpleNamespace(
            controller=IncompletePermissionController()
        )
        config = AndroidWorldConfig(permission_controller_settle_seconds=0)

        def adb_result(_config, *arguments, **kwargs):
            del _config, kwargs
            if arguments[:3] == ("shell", "cat", "/sdcard/pmtskill_window.xml"):
                return xml
            if arguments[:3] == ("shell", "input", "tap"):
                state["visible"] = False
            return ""

        with self.assertLogs(level="WARNING") as logs:
            with mock.patch(
                "src1.pmtskill_v2.evaluation.recovery._adb_command",
                side_effect=adb_result,
            ) as adb:
                dismissed = dismiss_permission_controller_dialogs(
                    environment, config
                )

        self.assertEqual(dismissed, 1)
        self.assertIn("可交互控件", "\n".join(logs.output))
        self.assertIn("permission_allow_button", "\n".join(logs.output))
        self.assertIn(
            ("shell", "input", "tap", "540", "1304"),
            [call.args[1:] for call in adb.call_args_list],
        )

    def test_unknown_permission_choices_are_logged_and_not_fatal(self):
        bounds = types.SimpleNamespace(
            x_min=100, x_max=400, y_min=1000, y_max=1150
        )
        controller = _Controller()
        controller.get_ui_elements = mock.Mock(
            return_value=[
                types.SimpleNamespace(
                    package_name="com.google.android.permissioncontroller",
                    resource_name="permission_cancel_button",
                    resource_id=None,
                    text="Cancel",
                    content_description=None,
                    class_name="android.widget.Button",
                    is_clickable=True,
                    is_enabled=True,
                    bbox_pixels=bounds,
                    bbox=None,
                ),
                types.SimpleNamespace(
                    package_name="com.google.android.permissioncontroller",
                    resource_name="permission_settings_button",
                    resource_id=None,
                    text="Settings",
                    content_description=None,
                    class_name="android.widget.Button",
                    is_clickable=True,
                    is_enabled=True,
                    bbox_pixels=bounds,
                    bbox=None,
                ),
            ]
        )
        environment = types.SimpleNamespace(controller=controller)
        config = AndroidWorldConfig(permission_controller_settle_seconds=0)

        with self.assertLogs(level="WARNING") as logs:
            with mock.patch(
                "src1.pmtskill_v2.evaluation.recovery._adb_command",
                return_value="",
            ):
                dismissed = dismiss_permission_controller_dialogs(
                    environment, config
                )

        output = "\n".join(logs.output)
        self.assertEqual(dismissed, 0)
        self.assertIn("Cancel", output)
        self.assertIn("Settings", output)
        self.assertIn("交给模型判断", output)

    def test_permission_dialog_is_handled_without_restarting_episode(self):
        state = {"visible": False}
        bounds = types.SimpleNamespace(
            x_min=133, x_max=947, y_min=1231, y_max=1378
        )

        class PermissionController(_Controller):
            def get_ui_elements(self) -> list[object]:
                self.probes += 1
                if not state["visible"]:
                    return []
                return [
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_allow_button",
                        resource_id=None,
                        text="Allow",
                        content_description=None,
                        bbox_pixels=bounds,
                        bbox=None,
                    )
                ]

        class Agent:
            def __init__(self, env):
                self.env = env
                self.calls = 0

            def step(self, goal):
                del goal
                self.calls += 1
                if self.calls == 1:
                    state["visible"] = True
                return types.SimpleNamespace(done=False, data={})

        controller = PermissionController()
        environment = types.SimpleNamespace(controller=controller)
        agent = Agent(environment)
        task = types.SimpleNamespace(
            name="SmsTask",
            tear_down=mock.Mock(),
        )

        def run_episode(*, goal, agent):
            agent.step(goal)
            return {
                "exception_info": None,
                "is_successful": 1.0,
                "episode_length": 1,
                "aux_data": {},
            }

        episode_runner = types.SimpleNamespace(run_episode=run_episode)
        suite_utils = types.SimpleNamespace(episode_runner=episode_runner)

        def invoke_episode(_task):
            return suite_utils.episode_runner.run_episode(
                goal="send a message", agent=agent
            )

        def run_task(_task, runner, _environment, _demo_mode):
            try:
                return runner(_task)
            except Exception as exc:  # 模拟 AndroidWorld _run_task 的异常封装。
                return {"exception_info": str(exc)}

        suite_utils._run_task = run_task
        config = AndroidWorldConfig(
            permission_controller_recovery_attempts=2,
            permission_controller_settle_seconds=0,
        )

        def adb_result(_config, *arguments, **kwargs):
            del _config, kwargs
            if arguments[:3] == ("shell", "input", "tap"):
                state["visible"] = False
            return ""

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery._adb_command",
            side_effect=adb_result,
        ):
            with recover_infrastructure_failures(
                suite_utils, environment, config
            ):
                result = suite_utils._run_task(
                    task, invoke_episode, environment, False
                )

        self.assertEqual(agent.calls, 1)
        self.assertEqual(result["episode_length"], 1)
        self.assertEqual(result["aux_data"]["permission_controller_restarts"], 0)
        self.assertEqual(
            result["aux_data"]["permission_controller_dialogs_dismissed"], 1
        )
        task.tear_down.assert_not_called()
        self.assertFalse(
            is_permission_controller_interruption(result)
        )

    def test_unknown_permission_choices_are_added_to_model_guidelines(self):
        state = {"visible": False}
        bounds = types.SimpleNamespace(
            x_min=100, x_max=400, y_min=1000, y_max=1150
        )

        class PermissionController(_Controller):
            def get_ui_elements(self) -> list[object]:
                self.probes += 1
                if not state["visible"]:
                    return []
                return [
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_cancel_button",
                        resource_id=None,
                        text="Cancel",
                        content_description=None,
                        class_name="android.widget.Button",
                        is_clickable=True,
                        is_enabled=True,
                        bbox_pixels=bounds,
                        bbox=None,
                    ),
                    types.SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name="permission_settings_button",
                        resource_id=None,
                        text="Settings",
                        content_description=None,
                        class_name="android.widget.Button",
                        is_clickable=True,
                        is_enabled=True,
                        bbox_pixels=bounds,
                        bbox=None,
                    ),
                ]

        class Agent:
            def __init__(self):
                self.additional_guidelines = ["existing"]
                self.guidelines_seen = None

            def step(self, goal):
                del goal
                self.guidelines_seen = list(self.additional_guidelines)
                return types.SimpleNamespace(done=False, data={})

        controller = PermissionController()
        environment = types.SimpleNamespace(controller=controller)
        agent = Agent()
        task = types.SimpleNamespace(name="PermissionChoiceTask")

        def run_episode(*, goal, agent):
            agent.step(goal)
            return {
                "exception_info": None,
                "is_successful": 0.0,
                "episode_length": 1,
                "aux_data": {},
            }

        suite_utils = types.SimpleNamespace(
            episode_runner=types.SimpleNamespace(run_episode=run_episode)
        )

        def invoke_episode(_task):
            state["visible"] = True
            return suite_utils.episode_runner.run_episode(
                goal="continue the task", agent=agent
            )

        suite_utils._run_task = (
            lambda task, runner, environment, demo_mode: runner(task)
        )
        config = AndroidWorldConfig(
            permission_controller_recovery_attempts=2,
            permission_controller_settle_seconds=0,
        )

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery._adb_command",
            return_value="",
        ):
            with recover_infrastructure_failures(
                suite_utils, environment, config
            ):
                result = suite_utils._run_task(
                    task, invoke_episode, environment, False
                )

        self.assertEqual(agent.additional_guidelines, ["existing"])
        self.assertIsNotNone(agent.guidelines_seen)
        guidance = "\n".join(agent.guidelines_seen)
        self.assertIn("Cancel", guidance)
        self.assertIn("Settings", guidance)
        self.assertIn('"action_type":"click"', guidance)
        self.assertEqual(
            result["aux_data"]["permission_controller_model_delegations"], 2
        )

    def test_only_device_failures_are_classified_as_infrastructure(self):
        self.assertTrue(
            is_infrastructure_failure(
                {"exception_info": "RuntimeError: Could not get a11y tree."}
            )
        )
        self.assertFalse(
            is_infrastructure_failure(
                {"exception_info": "ValueError: Invalid element index 99"}
            )
        )
        self.assertFalse(
            is_infrastructure_failure(
                {"exception_info": "httpx.ConnectError: Connection refused"}
            )
        )
        self.assertTrue(
            is_infrastructure_failure(
                {
                    "exception_info": (
                        "android_env grpc._channel._InactiveRpcError: "
                        "StatusCode.UNAVAILABLE"
                    )
                }
            )
        )

    def test_failed_a11y_task_is_recovered_and_retried(self):
        calls = 0

        def run_task(*args, **kwargs):
            nonlocal calls
            del args, kwargs
            calls += 1
            if calls == 1:
                return {"exception_info": "Could not get a11y tree."}
            return {"exception_info": None, "is_successful": 1.0}

        suite_utils = types.SimpleNamespace(_run_task=run_task)
        environment = types.SimpleNamespace(controller=_Controller())
        task = types.SimpleNamespace(name="OpenAppTaskEval")
        original = suite_utils._run_task

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery.recover_android_world_environment"
        ) as recover:
            with recover_infrastructure_failures(
                suite_utils, environment, AndroidWorldConfig()
            ):
                result = suite_utils._run_task(task, None, environment, False)

        self.assertIsNone(result["exception_info"])
        self.assertEqual(calls, 2)
        recover.assert_called_once()
        self.assertIs(suite_utils._run_task, original)

    def test_repeated_a11y_failure_stops_instead_of_empty_loop(self):
        suite_utils = types.SimpleNamespace(
            _run_task=lambda *args, **kwargs: {
                "exception_info": "Could not get a11y tree."
            }
        )
        environment = types.SimpleNamespace(controller=_Controller())
        task = types.SimpleNamespace(name="BrokenTask")

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery.recover_android_world_environment"
        ):
            with self.assertRaisesRegex(
                AndroidWorldInfrastructureError, "停止当前评测"
            ):
                with recover_infrastructure_failures(
                    suite_utils, environment, AndroidWorldConfig()
                ):
                    suite_utils._run_task(task, None, environment, False)

    def test_soft_recovery_restores_network_and_refreshes_controller(self):
        controller = _Controller()
        environment = types.SimpleNamespace(controller=controller)
        config = AndroidWorldConfig()

        def adb_result(_config, *arguments, **kwargs):
            del _config, kwargs
            return "device" if arguments == ("get-state",) else ""

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery._adb_command",
            side_effect=adb_result,
        ) as adb:
            recover_android_world_environment(environment, config)

        issued = [call.args[1:] for call in adb.call_args_list]
        self.assertIn(("get-state",), issued)
        self.assertIn(
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
            issued,
        )
        self.assertNotIn(("reboot",), issued)
        self.assertEqual(controller.refreshes, 1)

    def test_failed_soft_recovery_reboots_guest_and_waits_for_boot(self):
        class FlakyController(_Controller):
            def get_ui_elements(self) -> list[object]:
                self.probes += 1
                if self.refreshes < 2:
                    raise RuntimeError("Could not get a11y tree.")
                return []

        controller = FlakyController()
        environment = types.SimpleNamespace(controller=controller)
        config = AndroidWorldConfig()

        def adb_result(_config, *arguments, **kwargs):
            del _config, kwargs
            if arguments == ("get-state",):
                return "device"
            if arguments == ("shell", "getprop", "sys.boot_completed"):
                return "1"
            return ""

        with mock.patch(
            "src1.pmtskill_v2.evaluation.recovery._adb_command",
            side_effect=adb_result,
        ) as adb:
            recover_android_world_environment(environment, config)

        issued = [call.args[1:] for call in adb.call_args_list]
        self.assertIn(("reboot",), issued)
        self.assertIn(("shell", "getprop", "sys.boot_completed"), issued)
        self.assertEqual(controller.refreshes, 2)

    def test_zero_valid_episodes_is_not_accepted_as_real_score(self):
        with self.assertRaisesRegex(
            AndroidWorldInfrastructureError, "未产生任何有效 episode"
        ):
            ensure_valid_evaluation_episodes(
                [{"exception_info": "Could not get a11y tree."}],
                expected_episodes=1,
            )


if __name__ == "__main__":
    unittest.main()
