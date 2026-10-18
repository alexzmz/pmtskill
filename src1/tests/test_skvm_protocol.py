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
from skvm_protocol import SKVMProtocolBridge, budget_request, normalize_tool_finish_reasons  # noqa: E402


def tool_response(reason="stop", name="read_file", arguments=None):
    return {"choices": [{"finish_reason": reason, "message": {
        "role": "assistant", "content": "<tool_call>...</tool_call>",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments or {"path": "input.txt"})}}],
    }}], "usage": {"prompt_tokens": 652, "completion_tokens": 73}}


class SKVMToolProtocolTest(unittest.TestCase):
    def test_context_budget_preserves_instructions_tools_and_ids(self):
        payload = {"model": "test", "max_tokens": 32768, "messages": [
            {"role": "system", "content": "Never weaken the task contract."},
            {"role": "user", "content": "Read all 100000 lines and compute the correct answer."},
            {"role": "assistant", "tool_calls": [{"id": "read", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"input.txt"}'}}]},
            {"role": "tool", "tool_call_id": "read", "content": "row:123\n" * 100000},
        ]}
        before = copy.deepcopy(payload)
        bounded, excerpts, limits = budget_request(payload, context_window=32768)
        self.assertEqual(payload, before)
        self.assertEqual(bounded["messages"][:3], before["messages"][:3])
        self.assertEqual(bounded["messages"][3]["tool_call_id"], "read")
        self.assertLessEqual(len(bounded["messages"][3]["content"]), 12000)
        self.assertIn("full data remains on disk", bounded["messages"][3]["content"])
        self.assertEqual((excerpts, limits, bounded["max_tokens"]), (1, 1, 8192))

    def test_compiler_only_excerpts_profiling_bundle_not_skill_gap_or_other_files(self):
        prefix = "# Compilation Task\n\n## Current SKILL.md\n\n```markdown\nDo not omit validation.\n```\n\n## Bundled Files\n"
        suffix = "\n### helper.py\n\n```python\nprint('required')\n```\n\n## Capability Gaps\nKeep all checks."
        evidence = "\n### _profiling/tool.exec/L1/instance-0.jsonl\n\n```jsonl\n" + "x" * 100000 + "\n```"
        payload = {"messages": [{"role": "user", "content": prefix + evidence + suffix}], "max_tokens": 32768}
        bounded, count, _ = budget_request(payload, context_window=32768, compiler=True)
        content = bounded["messages"][0]["content"]
        self.assertEqual(count, 1)
        self.assertTrue(content.startswith(prefix))
        self.assertTrue(content.endswith(suffix))
        self.assertLess(len(content), 14000)
        unchanged, count, _ = budget_request(payload, context_window=32768, compiler=False)
        self.assertEqual(unchanged["messages"], payload["messages"])
        self.assertEqual(count, 0)

    def test_tool_batch_shares_evidence_budget(self):
        payload = {"messages": [{"role": "tool", "tool_call_id": str(i), "content": "x" * 8000}
                                for i in range(10)]}
        bounded, count, _ = budget_request(payload, context_window=32768)
        self.assertEqual(count, 10)
        self.assertLessEqual(sum(len(m["content"]) for m in bounded["messages"]), 12000)

    def test_compiler_preserves_raw_skill_even_if_it_contains_profiling_headings(self):
        skill = "### _profiling/example.jsonl\n\n```jsonl\n" + "required" * 5000 + "\n```"
        prompt = ("# Compilation Task\n\n## Current SKILL.md\n\n```markdown\n" + skill + "\n```\n\n"
                  "## Bundled Files\n\n### _profiling/real.jsonl\n\n```jsonl\n" + "evidence" * 5000 + "\n```\n\n"
                  "## Capability Gaps\nPreserve checks.")
        bounded, count, _ = budget_request({"messages": [{"role": "user", "content": prompt}]},
                                           context_window=32768, compiler=True, protected_texts=[skill])
        self.assertIn(skill, bounded["messages"][0]["content"])
        self.assertEqual(count, 1)

    def test_http_context_retry_is_bounded_and_unrecoverable_context_is_explicit(self):
        seen = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(data)
                if data["model"] == "other-error":
                    status, payload = 400, {"message": "invalid tool schema"}
                elif data["model"] == "too-large" or len(data["messages"][-1]["content"]) > 7000:
                    status, payload = 400, {"message": "max_tokens must be at least 1, got -169"}
                else:
                    status, payload = 200, tool_response()
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            with SKVMProtocolBridge(f"http://127.0.0.1:{upstream.server_port}/v1") as bridge:
                def send(model):
                    payload = {"model": model, "max_tokens": 32768, "messages": [
                        {"role": "user", "content": "Preserve this task"},
                        {"role": "tool", "tool_call_id": "read", "content": "x" * 100000}]}
                    req = request.Request(bridge.base_url + "/chat/completions", json.dumps(payload).encode(),
                                          {"Content-Type": "application/json"})
                    return request.urlopen(req)
                with send("works") as result:
                    self.assertEqual(json.load(result)["choices"][0]["finish_reason"], "tool_calls")
                self.assertEqual(len(seen), 2)
                self.assertEqual([s["max_tokens"] for s in seen], [8192, 4096])
                self.assertTrue(all(s["messages"][0]["content"] == "Preserve this task" for s in seen))
                self.assertEqual(bridge.summary()["context_retries"], 1)
                with self.assertRaises(error.HTTPError) as caught:
                    send("too-large")
                self.assertEqual(caught.exception.code, 400)
                self.assertEqual(json.loads(caught.exception.read())["error"]["code"], "SKVM_CONTEXT_WINDOW_EXCEEDED")
                self.assertEqual(len(seen), 5)  # Exactly three attempts, no unbounded retry.
                self.assertEqual(bridge.summary()["unresolved_context_errors"], 1)
                with self.assertRaises(error.HTTPError) as caught:
                    send("other-error")
                self.assertEqual(json.loads(caught.exception.read())["message"], "invalid tool schema")
                self.assertEqual(len(seen), 6)  # Unrelated HTTP400 must not be retried.
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5)

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
        from src1.baseline_skvm.protocol import output_for_protocol, validate_protocol_resume, SKVM_EVALUATION_PROTOCOL
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
            self.assertEqual(output_for_protocol(folder), folder / SKVM_EVALUATION_PROTOCOL)
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
            from src.skvm_protocol import SKVM_CONTEXT_POLICY
            marker.write_text(json.dumps({"protocol": runner.SKVM_TOOL_PROTOCOL,
                                           "context_policy": SKVM_CONTEXT_POLICY}), encoding="utf-8")
            self.assertEqual(output_for_protocol(folder), folder)
            validate_protocol_resume(folder)

    def test_tool_fixed_but_context_old_formal_results_are_also_isolated(self):
        from src1.baseline_skvm.protocol import output_for_protocol, validate_protocol_resume, SKVM_EVALUATION_PROTOCOL
        from src1.batch_evaluate import read_summary
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "skvm").mkdir()
            (folder / "skvm/kernel-protocol.json").write_text(
                json.dumps({"protocol": runner.SKVM_TOOL_PROTOCOL}), encoding="utf-8")
            (folder / "summary.json").write_text(json.dumps({
                "evaluation_available": True, "episodes_total": 2, "episodes_evaluated": 2,
                "metadata": {"skvm_tool_protocol": runner.SKVM_TOOL_PROTOCOL},
            }), encoding="utf-8")
            self.assertEqual(output_for_protocol(folder), folder / SKVM_EVALUATION_PROTOCOL)
            self.assertFalse(read_summary({"method": "skvm", "output_dir": str(folder)}, 2)[1])
            with self.assertRaisesRegex(ValueError, "不能与修复后的画像混跑"):
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
