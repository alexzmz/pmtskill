"""Narrow OpenAI tool-call compatibility bridge for existing SKVM binaries.

ms-swift may return a fully parsed tool_calls array with finish_reason='stop'.
Older SKVM binaries treat that as end_turn and never dispatch the tools. This
loopback-only bridge normalizes that one inconsistency, without interpreting
textual tool markup, changing arguments, or converting truncations to success.
"""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


SKVM_TOOL_PROTOCOL = "openai-tool-calls-v2"


def _reject_nonfinite(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def normalize_tool_finish_reasons(payload):
    """Return (payload, changed choices); do not modify the input response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return payload, 0
    result = copy.deepcopy(payload)
    changed = 0
    for choice in result["choices"]:
        if not isinstance(choice, dict) or choice.get("finish_reason") not in (None, "stop"):
            continue
        message = choice.get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or not calls:
            continue
        valid = True
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                    or not function["name"].strip()):
                valid = False
                break
            try:
                arguments = json.loads(function["arguments"], parse_constant=_reject_nonfinite)
                if not isinstance(arguments, dict):
                    valid = False
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                break
        if valid:
            choice["finish_reason"] = "tool_calls"
            changed += 1
    return (result if changed else payload), changed


class SKVMProtocolBridge:
    def __init__(self, upstream_base_url: str, timeout_seconds: float = 360):
        self.upstream = upstream_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.normalized_choices = 0
        self.requests = 0
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self.base_url = ""

    def __enter__(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                # Do not log Authorization headers or request/model content.
                pass

            def respond(self, status, body, content_type="application/json", retry_after=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if retry_after is not None:
                    self.send_header("Retry-After", retry_after)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def forward(self):
                route = self.path.split("?", 1)[0]
                if route not in {"/v1/chat/completions", "/v1/models"}:
                    self.respond(404, b'{"error":"unsupported bridge route"}')
                    return
                data = None
                if self.command == "POST":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 64 * 1024 * 1024:
                        self.respond(413, b'{"error":"invalid bridge request size"}')
                        return
                    data = self.rfile.read(length)
                    try:
                        if json.loads(data).get("stream"):
                            self.respond(400, b'{"error":"SKVM bridge requires non-streaming requests"}')
                            return
                    except (ValueError, AttributeError):
                        self.respond(400, b'{"error":"invalid JSON request"}')
                        return
                headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                target = bridge.upstream + route.removeprefix("/v1")
                upstream_request = request.Request(target, data=data, headers=headers, method=self.command)
                with bridge._lock:
                    bridge.requests += 1
                try:
                    with request.urlopen(upstream_request, timeout=bridge.timeout_seconds) as upstream:
                        body = upstream.read()
                        status = upstream.status
                        content_type = upstream.headers.get("Content-Type", "application/json")
                    normalized = 0
                    if route == "/v1/chat/completions":
                        try:
                            payload, normalized = normalize_tool_finish_reasons(json.loads(body))
                            if normalized:
                                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        except (ValueError, UnicodeError):
                            pass  # Preserve invalid upstream responses for SKVM's error handling.
                    with bridge._lock:
                        bridge.normalized_choices += normalized
                    self.respond(status, body, content_type)
                except error.HTTPError as exc:
                    self.respond(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"),
                                 exc.headers.get("Retry-After"))
                except (error.URLError, TimeoutError, OSError):
                    self.respond(502, b'{"error":"SKVM bridge upstream request failed"}')

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def summary(self):
        return {"protocol": SKVM_TOOL_PROTOCOL, "requests": self.requests,
                "normalized_tool_choices": self.normalized_choices}
