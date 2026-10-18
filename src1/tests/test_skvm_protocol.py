from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib import error, request

from src1.baseline_skvm.evaluator import legacy_kernel_module


runner = legacy_kernel_module()
from skvm_protocol import SKVMProtocolBridge, normalize_tool_finish_reasons  # noqa: E402


def tool_response(reason="stop", name="read_file", arguments=None):
    return {"choices": [{"finish_reason": reason, "message": {
        "role": "assistant", "content": "<tool_call>...</tool_call>",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments or {"path": "input.txt"})}}],
    }}], "usage": {"prompt_tokens": 652, "completion_tokens": 73}}


class SKVMToolProtocolTest(unittest.TestCase):
    def test_valid_tool_stop_null_and_missing_finish_are_normalized_without_mutation(self):
        for finish in ("stop", None, "missing"):
            with self.subTest(finish=finish):
                payload = tool_response(finish)
                if finish == "missing":
                    del payload["choices"][0]["finish_reason"]
                before = copy.deepcopy(payload)
                normalized, count = normalize_tool_finish_reasons(payload)
                self.assertEqual(count, 1)
                self.assertEqual(normalized["choices"][0]["finish_reason"], "tool_calls")
                self.assertEqual(normalized["choices"][0]["message"], before["choices"][0]["message"])
                self.assertEqual(normalized["usage"], before["usage"])
                self.assertEqual(payload, before)

    def test_truncation_filtering_and_native_tool_finish_are_preserved(self):
        for finish in ("length", "content_filter", "tool_calls", "unknown"):
            payload = tool_response(finish)
            self.assertEqual(normalize_tool_finish_reasons(payload), (payload, 0))

    def test_text_markup_is_not_treated_as_executable_tools(self):
        payload = tool_response()
        del payload["choices"][0]["message"]["tool_calls"]
        self.assertEqual(normalize_tool_finish_reasons(payload), (payload, 0))

    def test_invalid_arguments_names_and_empty_calls_are_not_normalized(self):
        for arguments in ("{", "null", "[]", '{"x":NaN}', None):
            payload = tool_response()
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
            self.assertEqual(normalize_tool_finish_reasons(payload), (payload, 0))
        for calls in ([], None, [{"function": {"name": "", "arguments": "{}"}}]):
            payload = tool_response()
            payload["choices"][0]["message"]["tool_calls"] = calls
            self.assertEqual(normalize_tool_finish_reasons(payload), (payload, 0))

    def test_http_bridge_forwards_requests_tool_results_and_http_errors(self):
        seen = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Authorization"), data))
                if data["model"] == "unauthorized":
                    body, status = b'{"error":"bad key"}', 401
                else:
                    payload = (tool_response() if len(data["messages"]) == 1 else
                               {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]})
                    body, status = json.dumps(payload).encode(), 200
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            with SKVMProtocolBridge(f"http://127.0.0.1:{upstream.server_port}/v1") as bridge:
                def send(model="adapter_control01", messages=None):
                    data = {"model": model, "messages": messages or [{"role": "user", "content": "make script"}]}
                    req = request.Request(bridge.base_url + "/chat/completions", json.dumps(data).encode(),
                                          {"Content-Type": "application/json", "Authorization": "Bearer test"})
                    with request.urlopen(req) as result:
                        return json.load(result)
                first = send()
                self.assertEqual(first["choices"][0]["finish_reason"], "tool_calls")
                messages = [{"role": "user", "content": "make script"},
                            {"role": "assistant", "tool_calls": first["choices"][0]["message"]["tool_calls"]},
                            {"role": "tool", "tool_call_id": "call-1", "content": "a:80"}]
                self.assertEqual(send(messages=messages)["choices"][0]["message"]["content"], "done")
                self.assertEqual(seen[1], ("/v1/chat/completions", "Bearer test",
                                          {"model": "adapter_control01", "messages": messages}))
                with self.assertRaises(error.HTTPError) as caught:
                    send(model="unauthorized")
                self.assertEqual(caught.exception.code, 401)
                self.assertEqual(bridge.summary()["normalized_tool_choices"], 1)
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5)

    def test_kernel_routes_through_bridge_and_config_restores_real_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(skvm_cache_dir=root / "cache", skvm_target_model=None,
                                   skvm_compiler_model="target", skvm_aot_pass_sets=["1"],
                                   skvm_prepare="auto", skvm_command="/existing/skvm")
            kernel = runner.SkVMKernel(args, skills=[], run_dir=root / "run",
                                       vllm_base_url="http://127.0.0.1:8002/v1", served_model="adapter_control01")
            old_profile = root / "cache/profiles/bad.json"
            old_profile.parent.mkdir(parents=True)
            old_profile.write_text('{"level":"L0"}', encoding="utf-8")
            def prepare():
                self.assertNotEqual(kernel.vllm_base_url, "http://127.0.0.1:8002/v1")
                kernel._configure_local_route()
                return ["variants"]
            with mock.patch.object(kernel, "_prepare", side_effect=prepare):
                self.assertEqual(kernel.prepare(), ["variants"])
            self.assertEqual(kernel.vllm_base_url, "http://127.0.0.1:8002/v1")
            config = json.loads((kernel.cache_dir / "skvm.config.json").read_text())
            self.assertEqual(config["providers"]["routes"][0]["baseUrl"], kernel.vllm_base_url)
            self.assertEqual(kernel.cache_dir, (root / "cache" / runner.SKVM_TOOL_PROTOCOL).resolve())
            self.assertTrue(old_profile.is_file())
            self.assertTrue((root / "run/skvm/protocol-compat.json").is_file())

    def test_old_formal_results_are_preserved_but_never_skipped_as_current(self):
        from src1.baseline_skvm.protocol import output_for_protocol, validate_protocol_resume
        from src1.batch_evaluate import read_summary
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "skvm"
            folder.mkdir()
            summary = {"evaluation_available": True, "episodes_total": 2, "episodes_evaluated": 2,
                       "metadata": {"evaluation_mode": "skvm_fixed_adapter_aot"}}
            path = folder / "summary.json"
            original = json.dumps(summary)
            path.write_text(original, encoding="utf-8")
            self.assertFalse(read_summary({"method": "skvm", "output_dir": str(folder)}, 2)[1])
            self.assertEqual(output_for_protocol(folder), folder / runner.SKVM_TOOL_PROTOCOL)
            with self.assertRaisesRegex(ValueError, "不能与修复后的画像混跑"):
                validate_protocol_resume(folder)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_new_partial_episodes_can_resume_under_same_protocol(self):
        from src1.baseline_skvm.protocol import output_for_protocol, validate_protocol_resume
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "checkpoints").mkdir()
            (folder / "checkpoints" / "task.json").write_text("{}", encoding="utf-8")
            marker = folder / "skvm" / "kernel-protocol.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({"protocol": runner.SKVM_TOOL_PROTOCOL}), encoding="utf-8")
            self.assertEqual(output_for_protocol(folder), folder)
            validate_protocol_resume(folder)

    def test_prepare_errors_restore_route_without_turning_failure_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(skvm_cache_dir=Path(directory), skvm_target_model=None,
                                   skvm_compiler_model="target", skvm_aot_pass_sets=["1"], skvm_prepare="auto")
            kernel = runner.SkVMKernel(args, skills=[], run_dir=Path(directory) / "run",
                                       vllm_base_url="http://127.0.0.1:8002/v1", served_model="fixed")
            def fail():
                kernel._configure_local_route()
                raise RuntimeError("original profiler failure")
            with mock.patch.object(kernel, "_prepare", side_effect=fail), self.assertRaisesRegex(RuntimeError, "original profiler failure"):
                kernel.prepare()
            self.assertEqual(kernel.vllm_base_url, "http://127.0.0.1:8002/v1")


if __name__ == "__main__":
    unittest.main()
