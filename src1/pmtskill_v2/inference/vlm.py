"""OpenAI-compatible VL 服务客户端。

vLLM、SGLang 和多数本地推理服务都能暴露 ``/chat/completions``，因此这里
不绑定具体模型家族。图像会被编码成 data URL，适用于 Qwen-VL、GLM-VL、
MobileRL 等视觉语言模型。
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..core.models import ModelProfile


class VLModelClient(Protocol):
    """规划器和执行器依赖的最小 VL 推理接口。"""

    model_id: str

    def generate(
        self,
        prompt: str,
        images: Sequence[Any] = (),
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> "GenerationResult": ...


@dataclass(slots=True)
class GenerationResult:
    text: str
    latency_ms: float
    raw: dict[str, Any]
    model_id: str


def _image_bytes(image: Any) -> bytes:
    """把 numpy/PIL/路径/字节统一转换为 JPEG 或原始图像字节。"""

    if isinstance(image, bytes):
        return image
    if isinstance(image, (str, os.PathLike)):
        with open(image, "rb") as handle:
            return handle.read()
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()
    except Exception as exc:  # 延迟导入，让 doctor/路由测试不强制依赖 Pillow。
        raise TypeError(f"不支持的图像类型: {type(image)!r}") from exc


def image_data_url(image: Any) -> str:
    encoded = base64.b64encode(_image_bytes(image)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class OpenAICompatibleVLClient:
    """无第三方 SDK 的 OpenAI-compatible HTTP 客户端。"""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        timeout_seconds: float = 180.0,
        maximum_retries: int = 3,
    ):
        self.profile = profile
        self.model_id = profile.model_id
        self.timeout_seconds = timeout_seconds
        self.maximum_retries = max(1, maximum_retries)

    @property
    def endpoint(self) -> str:
        return self.profile.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key_env:
            token = os.environ.get(self.profile.api_key_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def generate(
        self,
        prompt: str,
        images: Sequence[Any] = (),
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """发送一次多模态请求，并对临时网络/服务错误指数退避。"""

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            *(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image)},
                }
                for image in images
            ),
        ]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": self.profile.served_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 某些服务用 LoRA 名称作为 served model；其余 adapter 信息仅供审计。
        if self.profile.adapter:
            payload["metadata"] = {"adapter": self.profile.adapter}

        last_error: Exception | None = None
        for attempt in range(self.maximum_retries):
            started = time.perf_counter()
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    text = "".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
                else:
                    text = str(content)
                return GenerationResult(
                    text=text.strip(),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    raw=raw,
                    model_id=self.model_id,
                )
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.maximum_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"模型 {self.model_id} 请求失败，endpoint={self.endpoint}: {last_error}"
        )

    def predict_mm(
        self, text_prompt: str, images: list[Any]
    ) -> tuple[str, bool | None, dict[str, Any] | None]:
        """兼容 AndroidWorld ``MultimodalLlmWrapper`` 的三元组接口。"""

        try:
            print("predict_mm called, images =", len(images))
            result = self.generate(text_prompt, images, max_tokens=2048)
            raw = dict(result.raw)
            raw["_pmtskill"] = {
                "model_id": result.model_id,
                "latency_ms": result.latency_ms,
            }
            return result.text, True, raw
        except Exception as exc:  # M3A 会用空 raw 识别模型调用失败。
            return f"Error calling VL model: {exc}", None, None

    def predict(
        self, text_prompt: str
    ) -> tuple[str, bool | None, dict[str, Any] | None]:
        return self.predict_mm(text_prompt, [])
