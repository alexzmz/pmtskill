"""可选 Transformers/PEFT teacher-forcing 后端；不生成动作，不连接模拟器。

参考官方多模态 apply_chat_template 和 PEFT 多 adapter 接口：
https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
https://huggingface.co/docs/peft/main/en/package_reference/peft_model
所有重量级依赖均延迟导入，prior 模式完全不需要 torch/transformers。
"""

from __future__ import annotations

import gc
import hashlib
from typing import Any, Mapping

from ..core.models import ModelProfile, SkillRecord
from ..skills.identity import model_variant_id


def visual_messages(sample: Mapping[str, Any], skill: SkillRecord | None) -> list[dict[str, Any]]:
    """只支持已明确还原图像的 ms-swift JSONL；绝不把缺图样本伪装成视觉评估。"""
    images = list(sample.get("images", ()))
    if not images:
        raise ValueError("teacher-forcing 需要真实视觉输入")
    messages = []
    inserted = False
    for item in sample["messages"]:
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("离线 teacher-forcing 当前要求 messages.content 为字符串的 ms-swift JSONL")
        if item["role"] == "user":
            content = content.replace("<image>", "")
        if item["role"] == "user" and not inserted:
            guidance = (f"\n\n技能标定约束（SKVM forced-skill condition）：\n"
                        f"- 必须使用技能：{skill.name}\n"
                        f"- 覆盖原语：{', '.join(skill.topology.primitive_sequence())}\n"
                        f"- 技能说明：{skill.body[:3000] or skill.description}\n"
                        "请根据当前真实界面执行技能，但仍严格遵循 AndroidWorld 要求的 Reason/Action 输出格式；"
                        "界面不满足前置条件时应安全停止或恢复。" if skill else "")
            content = [{"type": "image", "url": path} for path in images] + [{"type": "text", "text": content + guidance}]
            inserted = True
        messages.append({"role": item["role"], "content": content})
    if not inserted or messages[-1]["role"] != "assistant":
        raise ValueError("参考样本必须有 user 输入及最后一条 assistant 参考响应")
    return messages


class TransformersReferenceScorer:
    def __init__(self, *, max_length: int = 32768, max_pixels: int = 1003520, trust_remote_code: bool = False):
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None
        self.base_path = None
        self.adapters: set[str] = set()

    def _activate(self, profile: ModelProfile) -> None:
        base = profile.metadata.get("base_model_path")
        checkpoint = profile.metadata.get("evaluation_checkpoint")
        adapter_name = "offline_" + hashlib.sha256(model_variant_id(profile).encode()).hexdigest()[:16]
        if not base or not checkpoint:
            raise ValueError("teacher-forcing 需要实际 base model 与 adapter checkpoint")
        try:
            import transformers
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("teacher-forcing 需要安装 torch、transformers、peft 及模型对应依赖") from exc
        if self.base_path != base:
            self.close()
            loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if loader is None:
                loader = getattr(transformers, "AutoModelForVision2Seq", None)
            if loader is None:
                raise RuntimeError("Transformers 版本不支持视觉语言模型的 AutoModel，请更新服务器环境")
            self.processor = transformers.AutoProcessor.from_pretrained(
                base, trust_remote_code=self.trust_remote_code, max_pixels=self.max_pixels,
            )
            model = loader.from_pretrained(base, device_map="auto", torch_dtype="auto",
                                           trust_remote_code=self.trust_remote_code)
            self.model = PeftModel.from_pretrained(model, checkpoint, adapter_name=adapter_name, is_trainable=False)
            self.model.eval()
            self.base_path = base
            self.adapters = {adapter_name}
        elif adapter_name not in self.adapters:
            self.model.load_adapter(checkpoint, adapter_name=adapter_name, is_trainable=False)
            self.adapters.add(adapter_name)
        self.model.set_adapter(adapter_name)
        self.model.eval()

    def score(self, profile: ModelProfile, sample: Mapping[str, Any], skill: SkillRecord | None) -> float:
        self._activate(profile)
        import torch

        messages = visual_messages(sample, skill)
        prefix = self.processor.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt")
        full = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                   return_dict=True, return_tensors="pt")
        prefix_length = prefix["input_ids"].shape[-1]
        full_length = full["input_ids"].shape[-1]
        if full_length > self.max_length or prefix_length >= full_length:
            raise ValueError(f"参考样本长度无效/超过上限: prefix={prefix_length}, full={full_length}, limit={self.max_length}")
        if not torch.equal(prefix["input_ids"], full["input_ids"][:, :prefix_length]):
            raise ValueError("chat template 的 assistant 前缀不一致，不能安全计算仅参考响应的 loss")
        full.pop("token_type_ids", None)
        labels = full["input_ids"].clone()
        labels[:, :prefix_length] = -100
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in full.items()}
        with torch.inference_mode():
            result = self.model(**inputs, labels=labels.to(device), use_cache=False, return_dict=True)
        return -float(result.loss.detach().float().cpu())

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.base_path = None
        self.adapters.clear()
        gc.collect()
        # 不为 prior 模式导入/初始化 CUDA。
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
