"""vLLM wrapper for Android World's text-only agents."""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


class VLLMWrapper:
    """Wrap a local vLLM model with Android World's ``LlmWrapper`` contract.

    The wrapper intentionally keeps the same plain-prompt behaviour as the
    original minimal runner.  It also records lightweight inference statistics
    so a benchmark report can distinguish model-call failures from task
    failures.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 512,
        raise_on_error: bool = False,
    ):
        # Keep vLLM optional at module-import time.  The SKVM runner talks to a
        # vLLM OpenAI-compatible server and should not import/load a second
        # in-process engine merely by importing this module.
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.raise_on_error = raise_on_error

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self._request_count = 0
        self._error_count = 0
        self._prompt_tokens = 0
        self._generated_tokens = 0
        self._total_latency_s = 0.0
        self._last_errors: list[str] = []

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        """Generate one response and return Android World's expected tuple."""
        started_at = time.perf_counter()
        self._request_count += 1
        try:
            outputs = self.llm.generate([text_prompt], self.sampling_params)
            request_output = outputs[0]
            candidate = request_output.outputs[0]
            generated_text = candidate.text.strip()

            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            generated_token_ids = getattr(candidate, "token_ids", None)
            if prompt_token_ids is not None:
                self._prompt_tokens += len(prompt_token_ids)
            if generated_token_ids is not None:
                self._generated_tokens += len(generated_token_ids)

            # T3A checks that raw_response is truthy before parsing the text.
            return generated_text, None, outputs
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._error_count += 1
            error_text = f"{type(exc).__name__}: {exc}"
            self._last_errors.append(error_text)
            self._last_errors = self._last_errors[-10:]
            if self.raise_on_error:
                raise RuntimeError(
                    f"vLLM inference failed: {exc}"
                ) from exc
            return (
                f"vLLM inference failed: {exc}",
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
        """Return JSON-serializable aggregate inference statistics."""
        average_latency = (
            self._total_latency_s / self._request_count
            if self._request_count
            else 0.0
        )
        return {
            "backend": "vllm-in-process",
            "request_count": self._request_count,
            "error_count": self._error_count,
            "prompt_tokens": self._prompt_tokens,
            "generated_tokens": self._generated_tokens,
            "total_latency_s": round(self._total_latency_s, 3),
            "mean_latency_s": round(average_latency, 3),
            "last_errors": list(self._last_errors),
        }


class VLLMOpenAIWrapper:
    """Use a vLLM OpenAI-compatible server through Android World's contract.

    The SKVM runner deliberately shares this server with SkVM's profiler and
    compiler.  Consequently the TCP profile, AOT variant, and Android World
    actions all target the same loaded model.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout_s: float = 300.0,
        raise_on_error: bool = False,
        use_chat_completions: bool = False,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        self.base_url = normalized
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.raise_on_error = raise_on_error
        self.use_chat_completions = use_chat_completions

        self._request_count = 0
        self._error_count = 0
        self._prompt_tokens = 0
        self._generated_tokens = 0
        self._total_latency_s = 0.0
        self._last_errors: list[str] = []

    def _post_completion(self, text_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_chat_completions:
            payload["messages"] = [{"role": "user", "content": text_prompt}]
            endpoint = "chat/completions"
        else:
            # Match VLLMWrapper's raw LLM.generate(prompt) semantics so a
            # baseline-vs-SKVM comparison does not accidentally include a
            # chat-template change as another experimental variable.
            payload["prompt"] = text_prompt
            endpoint = "completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        started_at = time.perf_counter()
        self._request_count += 1
        try:
            response = self._post_completion(text_prompt)
            choices = response.get("choices") or []
            first_choice = choices[0] if choices else {}
            if self.use_chat_completions:
                message = first_choice.get("message") or {}
                content = message.get("content")
            else:
                content = first_choice.get("text")
            if isinstance(content, list):
                generated_text = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                ).strip()
            else:
                generated_text = str(content or "").strip()

            usage = response.get("usage") or {}
            self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self._generated_tokens += int(usage.get("completion_tokens") or 0)
            return generated_text, None, response
        except (
            OSError,
            TimeoutError,
            ValueError,
            KeyError,
            urllib_error.HTTPError,
        ) as exc:
            self._error_count += 1
            error_text = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib_error.HTTPError):
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                    error_text += f": {body[:1000]}"
                except OSError:
                    pass
            self._last_errors.append(error_text)
            self._last_errors = self._last_errors[-10:]
            if self.raise_on_error:
                raise RuntimeError(
                    f"vLLM server inference failed: {error_text}"
                ) from exc
            return (
                f"vLLM server inference failed: {error_text}",
                False,
                {"error": error_text},
            )
        finally:
            self._total_latency_s += time.perf_counter() - started_at

    def predict_mm(
        self, text_prompt: str, images: Optional[list[Any]] = None
    ) -> tuple[str, Optional[bool], Any]:
        del images
        return self.predict(text_prompt)

    def get_stats(self) -> dict[str, Any]:
        average_latency = (
            self._total_latency_s / self._request_count
            if self._request_count
            else 0.0
        )
        return {
            "backend": "vllm-openai-server",
            "base_url": self.base_url,
            "served_model": self.model,
            "api_mode": (
                "chat/completions"
                if self.use_chat_completions
                else "completions"
            ),
            "request_count": self._request_count,
            "error_count": self._error_count,
            "prompt_tokens": self._prompt_tokens,
            "generated_tokens": self._generated_tokens,
            "total_latency_s": round(self._total_latency_s, 3),
            "mean_latency_s": round(average_latency, 3),
            "last_errors": list(self._last_errors),
        }
