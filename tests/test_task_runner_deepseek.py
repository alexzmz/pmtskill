from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib import error as urllib_error


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepseek_wrapper import DeepSeekWrapper  # noqa: E402
import task_runner_deepseek as runner  # noqa: E402
import task_runner_detail  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def success_response():
    return {
        "id": "response-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": (
                        'Reason: done\nAction: {"action_type":"status",'
                        ' "goal_status":"complete"}'
                    ),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 60,
            "prompt_cache_miss_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


class DeepSeekWrapperTests(unittest.TestCase):
    def test_returns_android_world_tuple_and_records_usage(self):
        wrapper = DeepSeekWrapper(
            model="deepseek-v4-flash",
            api_key="test-secret",
            thinking="disabled",
            max_retries=0,
        )
        with mock.patch(
            "deepseek_wrapper.urllib_request.urlopen",
            return_value=FakeResponse(success_response()),
        ) as urlopen:
            text, is_safe, raw = wrapper.predict("Android action prompt")

        self.assertTrue(text.startswith("Reason: done"))
        self.assertIsNone(is_safe)
        self.assertEqual(raw["id"], "response-1")
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(
            request.headers["Authorization"], "Bearer test-secret"
        )
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "Android action prompt"}],
        )
        self.assertEqual(payload["thinking"], {"type": "disabled"})

        stats = wrapper.get_stats()
        self.assertEqual(stats["request_count"], 1)
        self.assertEqual(stats["api_attempt_count"], 1)
        self.assertEqual(stats["error_count"], 0)
        self.assertEqual(stats["prompt_tokens"], 100)
        self.assertEqual(stats["generated_tokens"], 20)
        self.assertEqual(stats["reasoning_tokens"], 7)
        self.assertEqual(stats["prompt_cache_hit_rate"], 0.6)
        self.assertEqual(stats["finish_reason_counts"], {"stop": 1})

    def test_retries_transient_http_error_once(self):
        http_error = urllib_error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            503,
            "Service Unavailable",
            {"Retry-After": "0"},
            io.BytesIO(b'{"error":{"message":"busy"}}'),
        )
        wrapper = DeepSeekWrapper(
            api_key="test-secret",
            max_retries=2,
            retry_base_s=0,
        )
        with mock.patch(
            "deepseek_wrapper.urllib_request.urlopen",
            side_effect=[http_error, FakeResponse(success_response())],
        ) as urlopen:
            text, _, _ = wrapper.predict("prompt")

        self.assertIn("Action:", text)
        self.assertEqual(urlopen.call_count, 2)
        stats = wrapper.get_stats()
        self.assertEqual(stats["request_count"], 1)
        self.assertEqual(stats["api_attempt_count"], 2)
        self.assertEqual(stats["retry_count"], 1)
        self.assertEqual(stats["error_count"], 0)

    def test_non_retryable_auth_error_is_exposed_as_inference_error(self):
        http_error = urllib_error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"invalid key"}}'),
        )
        wrapper = DeepSeekWrapper(
            api_key="bad-secret",
            max_retries=3,
        )
        with mock.patch(
            "deepseek_wrapper.urllib_request.urlopen",
            side_effect=http_error,
        ) as urlopen:
            with self.assertRaisesRegex(
                RuntimeError, "DeepSeek API inference failed: HTTP 401"
            ):
                wrapper.predict("prompt")

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(wrapper.get_stats()["error_count"], 1)


class DeepSeekRunnerTests(unittest.TestCase):
    def test_deepseek_config_has_no_vllm_fields_or_api_key(self):
        args = runner.build_parser().parse_args(
            [
                "--deepseek_api_key",
                "do-not-store",
                "--deepseek_model",
                "deepseek-v4-pro",
            ]
        )
        runner.validate_deepseek_args(args)
        model_info = runner._model_info(args)
        config = task_runner_detail._base_config(
            args,
            condition="deepseek_api",
            android_world_root=REPO_ROOT / "libs" / "android_world",
            skill_info=None,
            model_info=model_info,
        )

        self.assertEqual(config["model"]["backend"], "deepseek-api")
        self.assertEqual(config["model"]["model"], "deepseek-v4-pro")
        self.assertNotIn("api_key", json.dumps(config))
        self.assertNotIn("tensor_parallel_size", config["model"])
        self.assertNotIn("model_path", config["model"])

        with tempfile.TemporaryDirectory() as temporary:
            args.output_dir = Path(temporary)
            run_dir = task_runner_detail._resolve_run_dir(
                args, "deepseek_api"
            )
            self.assertIn("deepseek-v4-pro", run_dir.name)

    def test_main_delegates_to_shared_detail_evaluator(self):
        with mock.patch(
            "task_runner_deepseek.task_runner_detail.run_evaluation",
            return_value=0,
        ) as evaluate:
            result = runner.main(
                [
                    "--deepseek_api_key",
                    "test-secret",
                    "--tasks",
                    "ContactsAddContact",
                ]
            )

        self.assertEqual(result, 0)
        kwargs = evaluate.call_args.kwargs
        self.assertEqual(kwargs["condition"], "deepseek_api")
        self.assertEqual(kwargs["backend_label"], "DeepSeek API")
        self.assertEqual(
            kwargs["model_info"]["backend"], "deepseek-api"
        )


if __name__ == "__main__":
    unittest.main()
