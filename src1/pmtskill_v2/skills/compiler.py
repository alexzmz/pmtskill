"""用云侧模型把通用 SKVM skill 编译成 AndroidWorld raw skill topology。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..core.models import PrimitiveSpec, SkillRecord, SkillTopology, utc_now
from ..inference.vlm import VLModelClient
from .store import SkillStore


class RawSkillCompiler(Protocol):
    compiler_id: str

    def compile(self, skill: SkillRecord) -> SkillRecord: ...


@dataclass(slots=True)
class RawSkillCompileSummary:
    examined: int = 0
    approved: int = 0
    rejected: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "examined": self.examined,
            "approved": self.approved,
            "rejected": self.rejected,
            "failed": self.failed,
        }


def _json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    value = (fenced.group(1) if fenced else text).strip()
    if not value.startswith("{"):
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("raw skill compiler 必须返回 JSON object")
    return parsed


class LLMRawSkillCompiler:
    """严格受 26 原语约束的 raw skill 编译器。

    编译只是准入和结构化，不会直接把技能晋升成 polished/active；在线结果仍受
    后续 trial、晋升和回滚机制约束。
    """

    compiler_id = "llm-raw-skill-compiler-v1"

    def __init__(self, client: VLModelClient, primitives: Sequence[PrimitiveSpec]):
        self.client = client
        self.primitives = tuple(primitives)

    def compile(self, skill: SkillRecord) -> SkillRecord:
        print(skill.name)
        catalog = [
            {
                "primitive_id": item.primitive_id,
                "description": item.description,
            }
            for item in self.primitives
        ]
        prompt = (
            "你正在将通用 Agent skill 编译为 AndroidWorld GUI skill。"
            "请判断该 skill 所表达的任务目标，是否可以通过 Android 手机上的应用、"
            "系统界面或常见 GUI 交互合理实现。"
            "不要要求原始 skill 本身包含 Android、点击、滑动等字样。"
            "只要其任务语义可以合理落地到 Android GUI，就设 android_relevant=true。"
            "只有明显属于纯后端、纯编程、纯服务器、纯专业知识推理、"
            "且无法通过 Android GUI 实现的任务，才设为 false。"
            "\n"
            "若可适配，请给出完成该任务语义所需的最小 primitive 执行序列。"
            "primitives 必须按执行顺序，且只能使用给定 primitive ID。"
            "不要输出分析过程，不要解释，不要使用 markdown。"
            "reason 不超过 40 个中文字符。"
            "只输出严格 JSON："
            '{"android_relevant":true|false,"primitives":[],'
            '"adapted_instruction":"","reason":""}。\n'
            f"技能名：{skill.name}\n"
            f"描述：{skill.description}\n"
            f"技能正文：{skill.body[:6000]}\n"
            f"允许原语：{json.dumps(catalog, ensure_ascii=False)}"
        )
        response = self.client.generate(prompt, max_tokens=1200)
        value = _json_object(response.text)
        known = {item.primitive_id for item in self.primitives}
        sequence = tuple(item for item in value.get("primitives", []) if item in known)
        relevant = bool(value.get("android_relevant", False)) and bool(sequence)
        skill.metadata["raw_skill_compiler"] = self.compiler_id
        skill.metadata["raw_skill_compiler_model"] = self.client.model_id
        skill.metadata["raw_skill_compile_reason"] = str(value.get("reason", ""))
        skill.metadata["raw_skill_compiled"] = True
        skill.metadata["approved_for_planning"] = relevant
        print(
            "RELEVANT",
            relevant,
            "PRIMITIVES",
            sequence,
        )
        if relevant:
            skill.topology = SkillTopology.from_sequence(sequence)
            instruction = str(value.get("adapted_instruction", "")).strip()
            if instruction:
                skill.metadata["original_body"] = skill.body
                skill.body = instruction
        skill.updated_at = utc_now()
        return skill


def compile_imported_raw_skills(
    store: SkillStore,
    compiler: RawSkillCompiler,
    *,
    limit: int = 8,
) -> RawSkillCompileSummary:
    """逐批编译尚未处理的 raw skills，避免一次维护周期产生过多云调用。"""

    pending = [
        skill
        for skill in store.list_skills(kind="raw")
        if not skill.metadata.get("raw_skill_compiled")
    ][: max(0, limit)]
    summary = RawSkillCompileSummary()
    for skill in pending:
        summary.examined += 1
        try:
            compiled = compiler.compile(skill)
            store.upsert_skill(compiled)
            if compiled.metadata.get("approved_for_planning"):
                summary.approved += 1
            else:
                summary.rejected += 1
        except Exception as exc:
            # 不把失败标为 compiled，下一维护周期仍可重试。
            store.log_maintenance_event(
                "raw_skill_compile_failed", skill.skill_id, {"error": str(exc)}
            )
            summary.failed += 1
    store.log_maintenance_event("raw_skill_compile_batch", None, summary.to_dict())
    return summary
