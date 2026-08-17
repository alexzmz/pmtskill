"""任务到 raw skill、再到原语拓扑的两阶段规划。

PPT 中在线预算要求拓扑转换尽量小于 1 秒，因此设备端默认使用确定性
``KeywordSkillPlanner``；有充足算力时可启用 ``LLMSkillPlanner`` 获取更准确分解。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..core.models import PrimitiveSpec, SkillRecord, SkillTopology
from ..inference.vlm import VLModelClient


@dataclass(slots=True)
class TaskDecomposition:
    """第一阶段的任务→技能结果。"""

    goal: str
    raw_skill_ids: tuple[str, ...]
    extra_primitives: tuple[str, ...] = ()
    reasoning: str = ""
    planner_id: str = "unknown"


class SkillTopologyPlanner(Protocol):
    planner_id: str

    def decompose(self, goal: str, raw_skills: Sequence[SkillRecord]) -> TaskDecomposition:
        ...


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]{1,4}", text.lower()))
    stopwords = {
        "a", "an", "and", "the", "to", "of", "for", "on", "in", "with",
        "from", "user", "using", "use", "task", "skill", "app", "please",
    }
    return terms - stopwords


class KeywordSkillPlanner:
    """低延迟、完全离线的 raw skill 检索与任务分解。"""

    planner_id = "keyword-planner-v1"

    _GOAL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
        (("open", "打开", "launch"), "action.open_app"),
        (("add", "create", "new", "添加", "新建"), "action.click"),
        (("add", "create", "new", "添加", "新建"), "action.type"),
        (("type", "input", "填写", "输入"), "action.type"),
        (("scroll", "向下", "向上", "滚动"), "action.scroll"),
        (("back", "返回"), "action.back"),
        (("click", "tap", "选择", "点击"), "action.click"),
        (("find", "read", "查看", "查找"), "perceive.ocr"),
    )

    def __init__(self, maximum_skills: int = 4):
        self.maximum_skills = maximum_skills

    def decompose(self, goal: str, raw_skills: Sequence[SkillRecord]) -> TaskDecomposition:
        goal_terms = _terms(goal)
        ranked: list[tuple[float, SkillRecord]] = []
        for skill in raw_skills:
            text = " ".join((skill.name, skill.description, skill.body[:1500]))
            overlap = goal_terms.intersection(_terms(text))
            if overlap:
                relevance = len(overlap) / max(1, len(goal_terms))
                if skill.metadata.get("android_relevant"):
                    relevance += 0.10
                ranked.append((relevance, skill))
        ranked.sort(key=lambda item: (-item[0], item[1].skill_id))
        selected = tuple(skill.skill_id for _, skill in ranked[: self.maximum_skills])
        extras = ["reason.intent", "reason.decompose"]
        lowered = goal.lower()
        for keywords, primitive in self._GOAL_RULES:
            if any(keyword in lowered for keyword in keywords):
                extras.append(primitive)
        extras.extend(("reason.verify", "control.finish"))
        return TaskDecomposition(
            goal=goal,
            raw_skill_ids=selected,
            extra_primitives=tuple(dict.fromkeys(extras)),
            reasoning="设备端关键词检索；云侧不可用时的低延迟回退。",
            planner_id=self.planner_id,
        )


def _extract_json(text: str) -> dict[str, Any]:
    """接受纯 JSON 或 markdown fenced JSON，但拒绝无法解释的自由文本。"""

    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 < end else candidate
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("规划器必须返回 JSON object")
    return value


class LLMSkillPlanner:
    """让在线/云侧模型输出受约束的 skill topology。"""

    planner_id = "llm-skill-planner-v1"

    def __init__(
        self,
        client: VLModelClient,
        primitives: Sequence[PrimitiveSpec],
        fallback: SkillTopologyPlanner | None = None,
        maximum_skills_in_prompt: int = 24,
    ):
        self.client = client
        self.primitives = tuple(primitives)
        self.fallback = fallback or KeywordSkillPlanner()
        self.maximum_skills_in_prompt = maximum_skills_in_prompt

    def decompose(self, goal: str, raw_skills: Sequence[SkillRecord]) -> TaskDecomposition:
        skill_catalog = [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description[:300],
                "primitives": list(skill.topology.primitive_sequence()),
            }
            for skill in raw_skills[: self.maximum_skills_in_prompt]
        ]
        primitive_ids = [primitive.primitive_id for primitive in self.primitives]
        prompt = (
            "把 Android 操作目标分解为可复用 raw skills 和少量补充原语。"
            "只能从给定 ID 中选择，按执行顺序输出严格 JSON："
            '{"raw_skill_ids":[],"extra_primitives":[],"reasoning":""}。\n'
            f"目标：{goal}\n"
            f"raw skills：{json.dumps(skill_catalog, ensure_ascii=False)}\n"
            f"原语：{json.dumps(primitive_ids, ensure_ascii=False)}"
        )
        try:
            result = self.client.generate(prompt, max_tokens=800)
            value = _extract_json(result.text)
            known_skills = {skill.skill_id for skill in raw_skills}
            known_primitives = set(primitive_ids)
            selected = tuple(
                item for item in value.get("raw_skill_ids", []) if item in known_skills
            )
            extras = tuple(
                item for item in value.get("extra_primitives", []) if item in known_primitives
            )
            if not selected and not extras:
                raise ValueError("模型返回空规划")
            return TaskDecomposition(
                goal=goal,
                raw_skill_ids=selected,
                extra_primitives=extras,
                reasoning=str(value.get("reasoning", "")),
                planner_id=self.planner_id,
            )
        except Exception:
            return self.fallback.decompose(goal, raw_skills)


class PrimitiveTopologyGenerator:
    """第二阶段：把 raw skill topology 展开成统一原语 DAG。"""

    generator_id = "primitive-expander-v1"

    def expand(
        self, decomposition: TaskDecomposition, raw_skills: Sequence[SkillRecord]
    ) -> SkillTopology:
        by_id = {skill.skill_id: skill for skill in raw_skills}
        sequence: list[str] = []
        for skill_id in decomposition.raw_skill_ids:
            skill = by_id.get(skill_id)
            if skill:
                sequence.extend(skill.topology.primitive_sequence())
        sequence.extend(decomposition.extra_primitives)
        # 连续重复原语通常是 skill 拼接边界造成的，设备执行无需重复。
        compacted: list[str] = []
        for primitive in sequence:
            if not compacted or compacted[-1] != primitive:
                compacted.append(primitive)
        if not compacted:
            compacted = ["reason.intent", "reason.decompose", "reason.verify", "control.finish"]
        return SkillTopology.from_sequence(compacted)


class PlannerPipeline:
    """对外提供一次调用的任务→原语拓扑接口。"""

    def __init__(self, planner: SkillTopologyPlanner, generator: PrimitiveTopologyGenerator):
        self.planner = planner
        self.generator = generator

    def plan(
        self, goal: str, raw_skills: Sequence[SkillRecord]
    ) -> tuple[TaskDecomposition, SkillTopology]:
        decomposition = self.planner.decompose(goal, raw_skills)
        return decomposition, self.generator.expand(decomposition, raw_skills)
