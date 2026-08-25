"""CLI 运行日志和最终结果归档测试。"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from src1.pmtskill_v2.core.run_records import CommandRunLogger


class CommandRunLoggerTest(unittest.TestCase):
    def test_captures_runtime_errors_and_readable_final_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = CommandRunLogger(
                Path(temporary) / "logs",
                command="collect",
                label="TaskA",
                argv=["collect", "--tasks", "TaskA"],
                arguments={"tasks": ["TaskA"], "handler": lambda: None},
            )
            with session.capture():
                print("辅助 print 输出")
                logging.info("运行信息")
                logging.warning("测试 warning")
                print("测试 stderr", file=sys.stderr)
                session.record_result(
                    {
                        "summary": {
                            "episodes_total": 2,
                            "successes": 1,
                            "task_success_rate": 0.5,
                            "per_task": {
                                "TaskA": {
                                    "successes": 1,
                                    "episodes": 2,
                                    "success_rate": 0.5,
                                    "average_steps": 3,
                                }
                            },
                            "per_primitive": {
                                "action.click": {
                                    "successes": 1,
                                    "trials": 2,
                                    "success_rate": 0.5,
                                }
                            },
                        }
                    }
                )
                session.finalize(0)

            runtime = session.artifacts.runtime_log.read_text(encoding="utf-8")
            errors = session.artifacts.errors_log.read_text(encoding="utf-8")
            markdown = session.artifacts.result_markdown.read_text(encoding="utf-8")
            result = json.loads(session.artifacts.result_json.read_text(encoding="utf-8"))
            metadata = json.loads(session.artifacts.run_json.read_text(encoding="utf-8"))

            self.assertIn("辅助 print 输出", runtime)
            self.assertIn("运行信息", runtime)
            self.assertIn("测试 warning", runtime)
            self.assertIn("测试 stderr", runtime)
            self.assertIn("测试 warning", errors)
            self.assertIn("测试 stderr", errors)
            self.assertIn("每任务结果", markdown)
            self.assertIn("每原语结果", markdown)
            self.assertEqual(result["run"]["status"], "success")
            self.assertEqual(result["result"]["summary"]["successes"], 1)
            self.assertEqual(metadata["command"], "collect")
            self.assertNotIn("handler", metadata["arguments"])

    def test_failed_run_records_traceback_in_final_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = CommandRunLogger(temporary, command="train")
            caught = None
            traceback_text = "Traceback: synthetic failure"
            with session.capture():
                try:
                    raise RuntimeError("training failed")
                except RuntimeError as exc:
                    caught = exc
                    print(traceback_text, file=sys.stderr)
                session.finalize(1, error=caught, traceback_text=traceback_text)

            result = json.loads(session.artifacts.result_json.read_text(encoding="utf-8"))
            self.assertEqual(result["run"]["status"], "failed")
            self.assertEqual(result["error"]["type"], "RuntimeError")
            self.assertIn("synthetic failure", result["error"]["traceback"])
            self.assertIn(
                "synthetic failure",
                session.artifacts.errors_log.read_text(encoding="utf-8"),
            )

    def test_persistent_logs_and_json_redact_credentials(self):
        fake_secret = "sk-test-1234567890abcdef1234567890"
        with tempfile.TemporaryDirectory() as temporary:
            session = CommandRunLogger(
                temporary,
                command="evaluate",
                argv=["evaluate", f"OPENAI_API_KEY={fake_secret}"],
                arguments={"api_key": fake_secret},
            )
            with session.capture():
                print({"OPENAI_API_KEY": fake_secret})
                logging.error("PASSWORD=%s", fake_secret)
                session.record_result({"access_token": fake_secret})
                session.finalize(
                    1,
                    error=RuntimeError(f"request failed with {fake_secret}"),
                )

            persisted = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    session.artifacts.runtime_log,
                    session.artifacts.errors_log,
                    session.artifacts.run_json,
                    session.artifacts.result_json,
                    session.artifacts.result_markdown,
                )
            )
            self.assertNotIn(fake_secret, persisted)
            self.assertIn("[REDACTED]", persisted)


if __name__ == "__main__":
    unittest.main()
