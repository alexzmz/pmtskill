"""在线模型/adapter 池。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from ..core.models import ModelProfile
from .vlm import OpenAICompatibleVLClient, VLModelClient


class ModelPool:
    """按 ``model_id`` 懒加载客户端并跟踪切换开销。

    当多个 LoRA 由同一 vLLM 服务托管时，切换只是改变 served model；如果后续
    使用自定义热加载服务，可通过 ``on_switch`` 回调执行真正的 adapter 加载。
    """

    def __init__(
        self,
        profiles: list[ModelProfile] | tuple[ModelProfile, ...],
        *,
        client_factory: Callable[[ModelProfile], VLModelClient] = OpenAICompatibleVLClient,
        on_switch: Callable[[ModelProfile | None, ModelProfile], None] | None = None,
    ):
        self.profiles = {profile.model_id: profile for profile in profiles if profile.enabled}
        if not self.profiles:
            raise ValueError("模型池没有启用的模型")
        self.client_factory = client_factory
        self.on_switch = on_switch
        self._clients: dict[str, VLModelClient] = {}
        self.current_model_id: str | None = None
        self.switch_count = 0
        self.switch_time_ms = 0.0

    def profile(self, model_id: str) -> ModelProfile:
        try:
            return self.profiles[model_id]
        except KeyError as exc:
            raise KeyError(f"模型池不存在或未启用: {model_id}") from exc

    def client(self, model_id: str) -> VLModelClient:
        """返回对应客户端，并在必要时触发 adapter/model 切换。"""

        target = self.profile(model_id)
        if self.current_model_id != model_id:
            previous = (
                self.profiles.get(self.current_model_id) if self.current_model_id else None
            )
            started = time.perf_counter()
            if self.on_switch:
                self.on_switch(previous, target)
            self.switch_time_ms += (time.perf_counter() - started) * 1000
            if self.current_model_id is not None:
                self.switch_count += 1
            self.current_model_id = model_id
        if model_id not in self._clients:
            self._clients[model_id] = self.client_factory(target)
        return self._clients[model_id]

    @contextmanager
    def use(self, model_id: str) -> Iterator[VLModelClient]:
        """上下文形式便于执行器将单步调用绑定到选中的模型。"""

        yield self.client(model_id)

    def reset_counters(self) -> None:
        self.current_model_id = None
        self.switch_count = 0
        self.switch_time_ms = 0.0

