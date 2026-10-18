"""OpenAI tool-call compatibility and evidence budgets for SKVM binaries.

ms-swift may return a fully parsed tool_calls array with finish_reason='stop'.
Older SKVM binaries treat that as end_turn and never dispatch the tools. This
loopback-only bridge normalizes that one inconsistency, without interpreting
textual tool markup, changing arguments, or converting truncations to success.
Preparation requests also get bounded optional evidence/output budgets, with
explicit, bounded context-error retries. The online AndroidWorld agent bypasses
this bridge entirely.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


SKVM_TOOL_PROTOCOL = "openai-tool-calls-v2"
SKVM_CONTEXT_POLICY = "bounded-tool-evidence-v1"


def is_context_error(value):
    text = str(value).lower()
    return ("max_tokens must be at least 1, got -" in text
            or "skvm_context_window_exceeded" in text
            or "maximum context length" in text
            or "context_length_exceeded" in text)


def _clip_evidence(text, limit):
    if len(text) <= limit:
        return text
    marker = "\n[SKVM context budget: evidence excerpted; full data remains on disk. Use tools to process the file or request a smaller excerpt.]\n"
    available = max(0, limit - len(marker))
    head = available * 3 // 4
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def budget_request(payload, *, context_window, compiler=False, attempt=0, protected_texts=()):
    """Bound optional evidence, NEVER task/system text or tool-call arguments.

    This is a character-based preventive bound, not a tokenizer. If the backend
    still rejects the context, the bridge retries with smaller evidence at most
    twice and otherwise returns an explicit infrastructure error.
    """
    result = copy.deepcopy(payload)
    output_limit = max(256, min(8192, context_window // 4) // (2 ** attempt))
    output_changes = 0
    for key in ("max_tokens", "max_completion_tokens"):
        if key in result and isinstance(result[key], int) and result[key] > output_limit:
            result[key] = output_limit
            output_changes += 1
    messages = result.get("messages", [])
    tools = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"
             and isinstance(m.get("content"), str)]
    # Shared budget prevents a batch of individually small tool results from
    # overflowing the window. All IDs/messages are retained in their order.
    evidence_limit = max(512, min(12000, context_window // 2) // (2 ** attempt))
    per_tool = max(256, evidence_limit // max(1, len(tools)))
    excerpts = 0
    for message in tools:
        clipped = _clip_evidence(message["content"], per_tool)
        excerpts += clipped != message["content"]
        message["content"] = clipped
    if compiler:
        # Only the compiler's explicitly labelled _profiling/ bundle snippets
        # are auxiliary. Preserve Current SKILL.md, gap requirements and every
        # other bundled file byte-for-byte, including untrusted headings inside
        # fenced file content (the boundary requires a closed markdown fence).
        pattern = re.compile(r"(?m)(^### _profiling/[^\n]*\n\s*```[^\n]*\n)(.*?)(\n```(?=\s*(?:\n### |\n## |\Z)))", re.DOTALL)
        matches = []
        for message in messages:
            if (isinstance(message, dict) and message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                    and "# Compilation Task" in message["content"]
                    and "## Current SKILL.md" in message["content"]):
                matches.append((message, list(pattern.finditer(message["content"]))))
        count = sum(len(items) for _, items in matches)
        per_file = max(256, evidence_limit // max(1, count))
        for message, _ in matches:
            protected_ranges = []
            for text in protected_texts:
                if not text:
                    continue
                start = message["content"].find(text)
                if start >= 0:
                    protected_ranges.append((start, start + len(text)))
            def replace(match):
                nonlocal excerpts
                if any(start < match.end() and end > match.start() for start, end in protected_ranges):
                    return match[0]
                clipped = _clip_evidence(match[2], per_file)
                excerpts += clipped != match[2]
                return match[1] + clipped + match[3]
            message["content"] = pattern.sub(replace, message["content"])
    return result, excerpts, output_changes


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
    def __init__(self, upstream_base_url: str, timeout_seconds: float = 360,
                 context_window: int = 32768, protected_texts=()):
        if context_window < 1024:
            raise ValueError("SKVM context window must be at least 1024 tokens")
        self.upstream = upstream_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.normalized_choices = 0
        self.requests = 0
        self.context_window = context_window
        self.protected_texts = tuple(protected_texts)
        self.stage = "profile"
        self.evidence_excerpts = 0
        self.output_limits = 0
        self.context_retries = 0
        self.context_errors = 0
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
                payload = None
                if self.command == "POST":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 64 * 1024 * 1024:
                        self.respond(413, b'{"error":"invalid bridge request size"}')
                        return
                    data = self.rfile.read(length)
                    try:
                        payload = json.loads(data)
                        if not isinstance(payload, dict):
                            raise ValueError("Expected an object")
                        if payload.get("stream"):
                            self.respond(400, b'{"error":"SKVM bridge requires non-streaming requests"}')
                            return
                    except (ValueError, AttributeError):
                        self.respond(400, b'{"error":"invalid JSON request"}')
                        return
                headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                target = bridge.upstream + route.removeprefix("/v1")
                with bridge._lock:
                    bridge.requests += 1
                try:
                    for attempt in range(3):
                        if payload is not None and route == "/v1/chat/completions":
                            bounded, excerpts, limits = budget_request(
                                payload, context_window=bridge.context_window,
                                compiler=bridge.stage == "compile", attempt=attempt,
                                protected_texts=bridge.protected_texts)
                            data = json.dumps(bounded, ensure_ascii=False).encode("utf-8")
                            with bridge._lock:
                                bridge.evidence_excerpts += excerpts
                                bridge.output_limits += limits
                        upstream_request = request.Request(target, data=data, headers=headers, method=self.command)
                        try:
                            with request.urlopen(upstream_request, timeout=bridge.timeout_seconds) as upstream:
                                body = upstream.read()
                                status = upstream.status
                                content_type = upstream.headers.get("Content-Type", "application/json")
                            break
                        except error.HTTPError as exc:
                            body = exc.read()
                            if exc.code == 400 and is_context_error(body):
                                if attempt < 2:
                                    with bridge._lock:
                                        bridge.context_retries += 1
                                    continue
                                with bridge._lock:
                                    bridge.context_errors += 1
                                body = json.dumps({"error": {
                                    "code": "SKVM_CONTEXT_WINDOW_EXCEEDED",
                                    "message": "Protected task instructions/tool arguments do not fit the context budget; not a model capability failure.",
                                    "context_window": bridge.context_window,
                                    "upstream_error": body.decode("utf-8", errors="replace"),
                                }}).encode("utf-8")
                            self.respond(exc.code, body, exc.headers.get("Content-Type", "application/json"),
                                         exc.headers.get("Retry-After"))
                            return
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
                "normalized_tool_choices": self.normalized_choices,
                "context_policy": SKVM_CONTEXT_POLICY, "context_window": self.context_window,
                "evidence_excerpts": self.evidence_excerpts, "output_limits": self.output_limits,
                "context_retries": self.context_retries, "unresolved_context_errors": self.context_errors}
