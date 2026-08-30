"""AndroidWorld 基础设施恢复与空评测熔断测试。"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from src1.pmtskill_v2.core.config import AndroidWorldConfig
from src1.pmtskill_v2.evaluation.recovery import (
    AndroidWorldInfrastructureError,
    ensure_valid_evaluation_episodes,
    is_infrastructure_failure,
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
