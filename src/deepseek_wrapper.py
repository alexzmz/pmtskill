"""DeepSeek Chat Completions wrapper for Android World's text-only agents."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


_RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}


class DeepSeekWrapper:
    """Call the DeepSeek API through Android World's ``LlmWrapper`` contract."""

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        retry_base_s: float = 1.0,
        thinking: str = "disabled",
        raise_on_error: bool = True,
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY or pass "
                "--deepseek_api_key."
            )
        if thinking not in {"auto", "enabled", "disabled"}:
            raise ValueError(
                "thinking must be one of: auto, enabled, disabled."
            )

        normalized = str(base_url).strip().rstrip("/")
        if not normalized:
            raise ValueError("DeepSeek base URL cannot be empty.")
        if normalized.endswith("/chat/completions"):
            self.endpoint = normalized
            self.base_url = normalized[: -len("/chat/completions")]
        else:
            self.base_url = normalized
            self.endpoint = f"{normalized}/chat/completions"

        self.model = model
        self.api_key = api_key.strip()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        self.thinking = thinking
        self.raise_on_error = raise_on_error

        self._request_count = 0
        self._api_attempt_count = 0
        self._retry_count = 0
        self._error_count = 0
        self._prompt_tokens = 0
        self._generated_tokens = 0
        self._reasoning_tokens = 0
        self._cache_hit_tokens = 0
        self._cache_miss_tokens = 0
        self._total_latency_s = 0.0
        self._last_errors: list[str] = []
        self._finish_reason_counts: dict[str, int] = {}

    def _payload(self, text_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": text_prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking != "auto":
            payload["thinking"] = {"type": self.thinking}
        return payload

    def _post(self, text_prompt: str) -> dict[str, Any]:
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(self._payload(text_prompt)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "pmtskill-android-world/1.0",
            },
            method="POST",
        )
        with urllib_request.urlopen(
            request, timeout=self.timeout_s
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("DeepSeek response is not a JSON object.")
        return payload

    @staticmethod
    def _http_error_text(exc: urllib_error.HTTPError) -> str:
        text = f"HTTP {exc.code}: {exc.reason}"
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except OSError:
            body = ""
        return f"{text}: {body[:2000]}" if body else text

    def _retry_delay(
        self, attempt: int, exc: BaseException
    ) -> float:
        if isinstance(exc, urllib_error.HTTPError):
            retry_after = (
                exc.headers.get("Retry-After") if exc.headers else None
            )
            if retry_after:
                try:
                    return min(max(float(retry_after), 0.0), 60.0)
                except ValueError:
                    pass
        return min(self.retry_base_s * (2**attempt), 30.0)

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, urllib_error.HTTPError):
            return exc.code in _RETRYABLE_HTTP_STATUS
        return isinstance(
            exc,
            (
                TimeoutError,
                urllib_error.URLError,
                ConnectionError,
                OSError,
            ),
        )

    @staticmethod
    def _content(response: Mapping[str, Any]) -> tuple[str, str]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("DeepSeek response contains no choices.")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("DeepSeek response choice is malformed.")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("DeepSeek response contains no assistant message.")
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, Mapping)
            ).strip()
        else:
            text = str(content or "").strip()
        if not text:
            raise ValueError("DeepSeek response content is empty.")
        return text, str(first.get("finish_reason") or "unknown")

    def _record_usage(
        self, response: Mapping[str, Any], finish_reason: str
    ) -> None:
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._generated_tokens += int(usage.get("completion_tokens") or 0)
        self._cache_hit_tokens += int(
            usage.get("prompt_cache_hit_tokens") or 0
        )
        self._cache_miss_tokens += int(
            usage.get("prompt_cache_miss_tokens") or 0
        )
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, Mapping):
            self._reasoning_tokens += int(
                completion_details.get("reasoning_tokens") or 0
            )
        self._finish_reason_counts[finish_reason] = (
            self._finish_reason_counts.get(finish_reason, 0) + 1
        )

    def predict(
        self, text_prompt: str
    ) -> tuple[str, Optional[bool], Any]:
        """Generate one response and return Android World's expected tuple."""
        started_at = time.perf_counter()
        self._request_count += 1
        last_error: BaseException | None = None
        try:
            for attempt in range(self.max_retries + 1):
                self._api_attempt_count += 1
                try:
                    response = self._post(text_prompt)
                    text, finish_reason = self._content(response)
                    self._record_usage(response, finish_reason)
                    return text, None, response
                except (
                    OSError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    urllib_error.HTTPError,
                    urllib_error.URLError,
                ) as exc:
                    last_error = exc
                    if attempt >= self.max_retries or not self._retryable(exc):
                        break
                    self._retry_count += 1
                    delay = self._retry_delay(attempt, exc)
                    if delay:
                        time.sleep(delay)

            assert last_error is not None
            if isinstance(last_error, urllib_error.HTTPError):
                error_text = self._http_error_text(last_error)
            else:
                error_text = f"{type(last_error).__name__}: {last_error}"
            self._error_count += 1
            self._last_errors.append(error_text)
            self._last_errors = self._last_errors[-10:]
            if self.raise_on_error:
                raise RuntimeError(
                    f"DeepSeek API inference failed: {error_text}"
                ) from last_error
            return (
                f"DeepSeek API inference failed: {error_text}",
                False,
                {"error": error_text},
            )
        finally:
            self._total_latency_s += time.perf_counter() - started_at

    def predict_mm(
        self, text_prompt: str, images: Optional[list[Any]] = None
    ) -> tuple[str, Optional[bool], Any]:
        """Text-only fallback required by some Android World agents."""
        del images
        return self.predict(text_prompt)

    def get_stats(self) -> dict[str, Any]:
        """Return JSON-serializable aggregate API and token statistics."""
        mean_latency = (
            self._total_latency_s / self._request_count
            if self._request_count
            else 0.0
        )
        cache_total = self._cache_hit_tokens + self._cache_miss_tokens
        return {
            "backend": "deepseek-api",
            "base_url": self.base_url,
            "model": self.model,
            "thinking": self.thinking,
            "request_count": self._request_count,
            "api_attempt_count": self._api_attempt_count,
            "retry_count": self._retry_count,
            "error_count": self._error_count,
            "prompt_tokens": self._prompt_tokens,
            "generated_tokens": self._generated_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "prompt_cache_hit_tokens": self._cache_hit_tokens,
            "prompt_cache_miss_tokens": self._cache_miss_tokens,
            "prompt_cache_hit_rate": (
                round(self._cache_hit_tokens / cache_total, 4)
                if cache_total
                else None
            ),
            "total_latency_s": round(self._total_latency_s, 3),
            "mean_latency_s": round(mean_latency, 3),
            "finish_reason_counts": dict(self._finish_reason_counts),
            "last_errors": list(self._last_errors),
        }
