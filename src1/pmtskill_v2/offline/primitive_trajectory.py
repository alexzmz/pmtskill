"""Teacher 受约束原语规划与 AndroidWorld VL 包装器。

本模块只负责“如何规划、如何在每一步约束 Teacher、如何记录计划修订”，不负责
启动 emulator。这样后续替换规划算法时，只需实现 :class:`PrimitivePlanGenerator`，
AndroidWorld 采集、SQLite 和技能维护代码都无需改动。

三种数据来源使用同一协议：

* ``task``：输入 AndroidWorld 任务，Teacher 从任务目标生成初始计划；
* ``database_skill``/兼容名 ``raw_skill``：读取 SQLite 中已有技能；
* ``skill_cluster``：先导入外部技能包，再按同样方式理解技能并规划。

计划中的 polished skill 始终展开为标准原语；``skill_id`` 只作为来源与执行方式的
审计字段。因此两种入口最终都能得到同构的原语序列。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..core.models import PrimitiveSpec, SkillRecord
from ..inference.vlm import VLModelClient
from ..online.planner import KeywordSkillPlanner
from ..skills.identity import skill_version_id


PRIMITIVE_TRACE_SCHEMA = "pmtskill.teacher-primitive-trace/v1"


def _json_object(text: str) -> dict[str, Any]:
    """从纯 JSON 或 markdown fenced JSON 中读取一个 object。"""

    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Teacher primitive planner 必须返回 JSON object")
    return value


@dataclass(frozen=True, slots=True)
class PrimitivePlanUnit:
    """一个环境 step 要完成的原语单元。

    一个单元可以同时包含感知、定位、推理和一个动作原语。例如“读取 UI → 定位
    Allow → 点击”属于一个环境动作单元。若引用 polished skill，则 ``skill_id`` 保留
    技能身份，而 ``primitive_ids`` 保存其完全展开后的原语，便于统一挖掘。
    """

    unit_id: str
    primitive_ids: tuple[str, ...]
    instruction: str
    skill_id: str | None = None

    @property
    def kind(self) -> str:
        return "polished_skill" if self.skill_id else "primitive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "skill_id": self.skill_id,
            "primitive_ids": list(self.primitive_ids),
            "instruction": self.instruction,
        }


@dataclass(frozen=True, slots=True)
class PrimitivePlanDecision:
    """一次初始规划或剩余路径修订的确定性返回值。"""

    units: tuple[PrimitivePlanUnit, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SkillTaskSelection:
    """Teacher 将通用 raw skill 绑定到可验证 AndroidWorld 任务的结果。"""

    tasks: tuple[str, ...]
    reason: str


class PrimitivePlanGenerator(Protocol):
    """可替换的 Teacher 原语规划接口。"""

    planner_id: str

    def initial_plan(
        self,
        goal: str,
        *,
        source_skill: SkillRecord | None,
        images: Sequence[Any] = (),
    ) -> PrimitivePlanDecision: ...

    def revise_plan(
        self,
        goal: str,
        *,
        source_skill: SkillRecord | None,
        completed_units: Sequence[PrimitivePlanUnit],
        remaining_units: Sequence[PrimitivePlanUnit],
        latest_step: Mapping[str, Any],
        android_context: str,
        images: Sequence[Any] = (),
    ) -> PrimitivePlanDecision: ...

    def bind_skill_to_tasks(
        self,
        skill: SkillRecord,
        available_tasks: Sequence[str],
        *,
        maximum_tasks: int,
    ) -> SkillTaskSelection: ...


class LLMPrimitivePlanGenerator:
    """让 Teacher 在固定原语/技能目录中生成并动态修订执行计划。"""

    planner_id = "teacher-primitive-planner-v1"

    def __init__(
        self,
        client: VLModelClient,
        primitives: Sequence[PrimitiveSpec],
        optimized_skills: Sequence[SkillRecord] = (),
        *,
        maximum_plan_units: int = 50,
        maximum_skills_in_prompt: int = 32,
    ):
        self.client = client
        self.primitives = tuple(primitives)
        self.primitive_by_id = {item.primitive_id: item for item in self.primitives}
        self.optimized_skills = {
            skill.skill_id: skill for skill in optimized_skills
        }
        self.maximum_plan_units = max(1, maximum_plan_units)
        self.maximum_skills_in_prompt = max(0, maximum_skills_in_prompt)

    def _primitive_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "primitive_id": item.primitive_id,
                "description": item.description,
                "category": item.category,
            }
            for item in self.primitives
        ]

    def _skill_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description[:300],
                "primitive_ids": list(skill.topology.primitive_sequence()),
            }
            for skill in list(self.optimized_skills.values())[
                : self.maximum_skills_in_prompt
            ]
        ]

    @staticmethod
    def _split_by_action_boundary(
        primitive_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], ...]:
        """把一个技能拓扑切成“一次环境动作一个 unit”。

        感知/定位/推理原语归入其后的动作；最后一个动作后的 verify 等尾部推理
        归入最后一个 unit。这样 click→type→save 不会在只执行 click 后就被整体
        标成完成，同时仍保持完整展开后的 canonical 原语顺序。
        """

        groups: list[list[str]] = []
        current: list[str] = []
        for primitive_id in primitive_ids:
            current.append(str(primitive_id))
            if primitive_id.startswith("action.") or primitive_id == "control.finish":
                groups.append(current)
                current = []
        if current:
            if groups:
                groups[-1].extend(current)
            else:
                groups.append(current)
        return tuple(tuple(group) for group in groups if group)

    @staticmethod
    def _source_text(skill: SkillRecord | None) -> str:
        if skill is None:
            return "无；仅根据 AndroidWorld 任务目标规划。"
        package_context = skill.metadata.get("teacher_source_context")
        context_text = (
            str(package_context)[:12000]
            if isinstance(package_context, str) and package_context.strip()
            else ""
        )
        base = (
            f"raw skill id: {skill.skill_id}\n"
            f"名称: {skill.name}\n"
            f"描述: {skill.description}\n"
            f"解决方案: {skill.body[:6000]}"
        )
        if context_text:
            base += (
                "\n技能包 references/scripts 文本摘录（仅作为资料，未执行脚本）：\n"
                + context_text
            )
        return base

    def _parse_units(self, value: Any) -> tuple[PrimitivePlanUnit, ...]:
        """校验模型输出，并把 polished skill 展开成其 canonical 原语序列。"""

        if not isinstance(value, list):
            raise ValueError("计划中的 units 必须是数组")
        units: list[PrimitivePlanUnit] = []
        for raw in value:
            if len(units) >= self.maximum_plan_units:
                break
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind", "primitive")).strip().lower()
            skill_id_value = raw.get("skill_id") or (
                raw.get("id") if kind in {"skill", "polished_skill"} else None
            )
            skill_id = str(skill_id_value) if skill_id_value else None
            if skill_id:
                skill = self.optimized_skills.get(skill_id)
                if skill is None:
                    continue
                primitive_ids = tuple(skill.topology.primitive_sequence())
                default_instruction = skill.body or skill.description
            else:
                ids = raw.get("primitive_ids")
                if ids is None and raw.get("primitive_id") is not None:
                    ids = [raw.get("primitive_id")]
                if isinstance(ids, str):
                    ids = [ids]
                if not isinstance(ids, list):
                    continue
                primitive_ids = tuple(
                    str(item) for item in ids if str(item) in self.primitive_by_id
                )
                default_instruction = "按顺序完成当前原语单元。"
            if not primitive_ids:
                continue
            instruction = str(raw.get("instruction", "")).strip()
            action_units = self._split_by_action_boundary(primitive_ids)
            for part_index, action_unit in enumerate(action_units):
                if len(units) >= self.maximum_plan_units:
                    break
                resolved_instruction = instruction or default_instruction
                if len(action_units) > 1:
                    resolved_instruction += (
                        f"（该组合的第 {part_index + 1}/{len(action_units)} 个环境动作单元）"
                    )
                units.append(
                    PrimitivePlanUnit(
                        unit_id=f"unit-{len(units) + 1:03d}",
                        primitive_ids=action_unit,
                        instruction=resolved_instruction,
                        skill_id=skill_id,
                    )
                )
        if not units:
            raise ValueError("Teacher 返回了空计划或全部使用未知原语/技能")
        return tuple(units)

    def _fallback_units(
        self, goal: str, source_skill: SkillRecord | None
    ) -> tuple[PrimitivePlanUnit, ...]:
        """模型 JSON 失败时仍生成可审计、受约束的确定性计划。"""

        if source_skill is not None and source_skill.topology.primitive_sequence():
            sequence = source_skill.topology.primitive_sequence()
        else:
            sequence = KeywordSkillPlanner(maximum_skills=0).decompose(
                goal, ()
            ).extra_primitives
        known_sequence = tuple(
            primitive_id
            for primitive_id in sequence
            if primitive_id in self.primitive_by_id
        )
        groups = self._split_by_action_boundary(known_sequence)
        return tuple(
            PrimitivePlanUnit(
                unit_id=f"unit-{index + 1:03d}",
                primitive_ids=group,
                instruction="；".join(
                    self.primitive_by_id[primitive_id].description
                    for primitive_id in group
                ),
            )
            for index, group in enumerate(groups[: self.maximum_plan_units])
        )

    def initial_plan(
        self,
        goal: str,
        *,
        source_skill: SkillRecord | None,
        images: Sequence[Any] = (),
    ) -> PrimitivePlanDecision:
        """在 episode 第一动作前生成完整计划和明确的动作单元数量。"""

        prompt = (
            "你是 AndroidWorld Teacher 轨迹规划器。先根据目标制定完整计划，再由"
            "执行器逐步完成；后续界面变化时允许修改尚未执行的部分。\n"
            "每个 unit 对应一次 AndroidWorld M3A 环境动作，可以包含若干感知、"
            "定位、推理原语，但通常应包含一个 action.* 或最终 control.finish。\n"
            "只能使用目录中的 primitive_id；也可引用给出的 polished skill。引用"
            "技能时 kind=polished_skill 且填写 skill_id，不要自行复制其内部原语。\n"
            "输出严格 JSON，不要 markdown："
            '{"planned_step_count":2,"units":['
            '{"kind":"primitive","primitive_ids":[],"instruction":""},'
            '{"kind":"polished_skill","skill_id":"","instruction":""}],'
            '"reason":""}。planned_step_count 必须等于 units 数量。\n'
            f"任务目标：{goal}\n"
            f"输入技能：{self._source_text(source_skill)}\n"
            f"标准原语：{json.dumps(self._primitive_catalog(), ensure_ascii=False)}\n"
            f"可选 polished skills：{json.dumps(self._skill_catalog(), ensure_ascii=False)}"
        )
        try:
            result = self.client.generate(
                prompt, images, temperature=0.0, max_tokens=1800
            )
            value = _json_object(result.text)
            units = self._parse_units(value.get("units", value.get("steps", [])))
            return PrimitivePlanDecision(units, str(value.get("reason", "")))
        except Exception as exc:
            fallback = self._fallback_units(goal, source_skill)
            if not fallback:
                raise RuntimeError("Teacher 初始规划失败且无确定性回退") from exc
            return PrimitivePlanDecision(
                fallback, f"模型规划失败，使用确定性回退：{exc}"
            )

    def revise_plan(
        self,
        goal: str,
        *,
        source_skill: SkillRecord | None,
        completed_units: Sequence[PrimitivePlanUnit],
        remaining_units: Sequence[PrimitivePlanUnit],
        latest_step: Mapping[str, Any],
        android_context: str,
        images: Sequence[Any] = (),
    ) -> PrimitivePlanDecision:
        """结合上一步执行结果和当前截图，只修订尚未执行的路径。"""

        prompt = (
            "你是 AndroidWorld Teacher 动态重规划器。检查上一步结果和当前界面。"
            "已经完成的单元不可修改；只返回从当前时刻开始的 remaining_units。"
            "如果原路径仍正确，change=false；若动作失败、界面不符、出现弹窗或"
            "计划耗尽但任务未结束，change=true 并给出新路径。\n"
            "每个 unit 的格式与初始规划一致，只能使用给定原语/技能。输出严格 JSON："
            '{"change":true|false,"remaining_units":[],"reason":""}。\n'
            f"任务目标：{goal}\n"
            f"输入技能：{self._source_text(source_skill)}\n"
            f"已完成：{json.dumps([item.to_dict() for item in completed_units], ensure_ascii=False)}\n"
            f"原剩余路径：{json.dumps([item.to_dict() for item in remaining_units], ensure_ascii=False)}\n"
            f"上一步：{json.dumps(dict(latest_step), ensure_ascii=False, default=str)}\n"
            f"AndroidWorld 当前上下文：{android_context[-6000:]}\n"
            f"标准原语：{json.dumps(self._primitive_catalog(), ensure_ascii=False)}\n"
            f"可选 polished skills：{json.dumps(self._skill_catalog(), ensure_ascii=False)}"
        )
        try:
            result = self.client.generate(
                prompt, images, temperature=0.0, max_tokens=1600
            )
            value = _json_object(result.text)
            if not bool(value.get("change", False)):
                return PrimitivePlanDecision(
                    tuple(remaining_units), str(value.get("reason", "保留原路径"))
                )
            units = self._parse_units(value.get("remaining_units", []))
            return PrimitivePlanDecision(units, str(value.get("reason", "动态修订")))
        except Exception as exc:
            # 动态修订失败不应破坏正在运行的 episode；保留旧路径并记录原因。
            return PrimitivePlanDecision(
                tuple(remaining_units), f"重规划失败，保留原路径：{exc}"
            )

    def bind_skill_to_tasks(
        self,
        skill: SkillRecord,
        available_tasks: Sequence[str],
        *,
        maximum_tasks: int,
    ) -> SkillTaskSelection:
        """选择具有 AndroidWorld ground-truth evaluator 的任务来承载 raw skill。"""

        limit = max(1, maximum_tasks)
        prompt = (
            "把一个通用技能绑定到语义最接近、确实能验证该技能的 AndroidWorld"
            "任务。只能从候选 task name 中选择；没有合适任务时返回空数组。"
            f"最多选择 {limit} 个。只输出严格 JSON："
            '{"tasks":[],"reason":""}。\n'
            f"技能资料：\n{self._source_text(skill)}\n"
            f"候选任务：{json.dumps(list(available_tasks), ensure_ascii=False)}"
        )
        try:
            result = self.client.generate(prompt, temperature=0.0, max_tokens=600)
            value = _json_object(result.text)
            available = set(available_tasks)
            raw_tasks = value.get("tasks", [])
            if not isinstance(raw_tasks, list):
                raw_tasks = []
            tasks = tuple(
                dict.fromkeys(
                    str(item) for item in raw_tasks if str(item) in available
                )
            )[:limit]
            return SkillTaskSelection(tasks, str(value.get("reason", "")))
        except Exception as exc:
            return SkillTaskSelection((), f"自动任务绑定失败：{exc}")


@dataclass(slots=True)
class TeacherPrimitiveVLWrapper:
    """在原生 M3A 外包裹 Teacher，使每个动作服从可修订的原语计划。"""

    client: Any
    planner: PrimitivePlanGenerator
    replan_every_steps: int = 1
    source_kind: str = "task"
    source_skill: SkillRecord | None = None
    collection_context: dict[str, Any] = field(default_factory=dict)
    goal: str | None = None
    initial_units: tuple[PrimitivePlanUnit, ...] = ()
    remaining_units: list[PrimitivePlanUnit] = field(default_factory=list)
    completed_units: list[PrimitivePlanUnit] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    latest_step: dict[str, Any] = field(default_factory=dict)
    _pending_unit: PrimitivePlanUnit | None = None
    _action_calls: int = 0

    def reset(self) -> None:
        self.goal = None
        self.initial_units = ()
        self.remaining_units = []
        self.completed_units = []
        self.revisions = []
        self.latest_step = {}
        self._pending_unit = None
        self._action_calls = 0

    def set_episode(
        self,
        goal: str,
        *,
        source_kind: str,
        source_skill: SkillRecord | None = None,
    ) -> None:
        """绑定当前 episode；相同 goal 的连续 M3A step 不重复初始化。"""

        if self.goal == goal:
            return
        self.reset()
        self.goal = goal
        self.source_kind = source_kind
        self.source_skill = source_skill

    @staticmethod
    def _is_summary(prompt: str) -> bool:
        lowered = prompt.lower()
        return (
            "summerize the latest step" in lowered
            or "summarize the latest step" in lowered
        )

    def _ensure_initial_plan(self, images: Sequence[Any]) -> None:
        if self.initial_units:
            return
        if not self.goal:
            raise RuntimeError("TeacherPrimitiveVLWrapper 尚未设置 episode goal")
        decision = self.planner.initial_plan(
            self.goal, source_skill=self.source_skill, images=images
        )
        self.initial_units = decision.units
        self.remaining_units = list(decision.units)
        self.revisions.append(
            {
                "revision": 0,
                "trigger": "initial_plan",
                "reason": decision.reason,
                "remaining_units": [item.to_dict() for item in decision.units],
            }
        )

    def _maybe_replan(self, prompt: str, images: Sequence[Any]) -> None:
        if not self.goal or self._action_calls <= 0:
            return
        interval_due = (
            self.replan_every_steps > 0
            and self._action_calls % self.replan_every_steps == 0
        )
        if not interval_due and self.remaining_units:
            return
        previous = tuple(self.remaining_units)
        decision = self.planner.revise_plan(
            self.goal,
            source_skill=self.source_skill,
            completed_units=tuple(self.completed_units),
            remaining_units=previous,
            latest_step=self.latest_step,
            android_context=prompt,
            images=images,
        )
        self.remaining_units = list(decision.units)
        changed = tuple(decision.units) != previous
        self.revisions.append(
            {
                "revision": len(self.revisions),
                "trigger": "path_changed" if changed else "path_confirmed",
                "reason": decision.reason,
                "remaining_units": [item.to_dict() for item in decision.units],
            }
        )

    def _trace_snapshot(self, unit: PrimitivePlanUnit) -> dict[str, Any]:
        optimized = (
            getattr(self.planner, "optimized_skills", {}).get(unit.skill_id)
            if unit.skill_id
            else None
        )
        source = {
            "kind": self.source_kind,
            "skill_id": self.source_skill.skill_id if self.source_skill else None,
            "skill_name": self.source_skill.name if self.source_skill else None,
            "skill_description": (
                self.source_skill.description[:1000] if self.source_skill else None
            ),
            "skill_version": self.source_skill.version if self.source_skill else None,
            "skill_source_hash": (
                self.source_skill.source_hash if self.source_skill else None
            ),
            "skill_import_namespace": (
                self.source_skill.metadata.get("import_namespace")
                if self.source_skill
                else None
            ),
            "skill_package_hash": (
                self.source_skill.metadata.get("package_hash")
                if self.source_skill
                else None
            ),
        }
        return {
            "trajectory_schema": PRIMITIVE_TRACE_SCHEMA,
            "metric_scope": "skill_discovery",
            "routing_mode": "teacher_primitive_guided",
            "model_id": str(getattr(self.client, "model_id", "unknown")),
            # raw skill 是规划输入，不是一次被调用的优化技能；只有计划单元真正引用
            # polished skill 时才写入 skill_id，避免污染逐模型技能成功率。
            "skill_id": unit.skill_id,
            "skill_version_id": (
                skill_version_id(optimized) if optimized is not None else None
            ),
            "primitive_ids": list(unit.primitive_ids),
            "source": source,
            "collection_context": dict(self.collection_context),
            "planner_id": self.planner.planner_id,
            "planned_step_count": len(self.initial_units),
            "current_revision": max(0, len(self.revisions) - 1),
            "planned_unit": unit.to_dict(),
            "initial_plan": [item.to_dict() for item in self.initial_units],
            "plan_revisions": list(self.revisions),
            "completed_units": [item.to_dict() for item in self.completed_units],
        }

    def predict_mm(
        self, text_prompt: str, images: list[Any]
    ) -> tuple[str, bool | None, dict[str, Any] | None]:
        if self._is_summary(text_prompt):
            return self.client.predict_mm(text_prompt, images)

        self._ensure_initial_plan(images)
        self._maybe_replan(text_prompt, images)
        if not self.remaining_units:
            # 极端情况下重规划器仍返回空路径，使用可验证的安全结束单元，避免重复
            # 上一个 action 并制造虚假轨迹。
            self.remaining_units = [
                PrimitivePlanUnit(
                    "unit-finish",
                    ("reason.verify", "control.finish"),
                    "核验当前界面；已完成则报告成功，否则报告失败。",
                )
            ]
        unit = self.remaining_units[0]
        optimized = (
            getattr(self.planner, "optimized_skills", {}).get(unit.skill_id)
            if unit.skill_id
            else None
        )
        source_hint = ""
        if self.source_skill is not None:
            package_context = self.source_skill.metadata.get("teacher_source_context")
            context_hint = (
                f"\n技能包资料摘录：{str(package_context)[:5000]}"
                if isinstance(package_context, str) and package_context.strip()
                else ""
            )
            source_hint = (
                "\n- 输入 raw skill 仅作为任务与解法参考："
                f"{(self.source_skill.body or self.source_skill.description)[:2500]}"
                + context_hint
            )
        skill_hint = ""
        if optimized is not None:
            skill_hint = (
                "\n- 当前采用的 polished skill："
                f"{(optimized.body or optimized.description)[:2500]}"
            )
        augmented = (
            text_prompt
            + "\n\nTeacher 原语轨迹约束：\n"
            + f"- 初始计划共 {len(self.initial_units)} 个环境动作单元。\n"
            + f"- 当前单元：{unit.unit_id}。\n"
            + f"- 当前原语：{', '.join(unit.primitive_ids)}。\n"
            + f"- 当前具体要求：{unit.instruction}。\n"
            + source_hint
            + skill_hint
            + "\n请结合当前真实截图完成当前单元。若界面与预期不同，先安全恢复；"
            + "下一步规划器可以修改剩余路径。必须保持 AndroidWorld 要求的 "
            + "Reason/Action 输出格式。"
        )
        output, safe, raw = self.client.predict_mm(augmented, images)
        self._pending_unit = unit
        self._action_calls += 1
        if raw is not None:
            raw = dict(raw)
            raw.setdefault("_pmtskill", {}).update(self._trace_snapshot(unit))
        return output, safe, raw

    def observe_step(self, step_data: Any) -> None:
        """接收 M3A 的真实动作解析结果，并完成当前计划单元的提交/重试。"""

        data = dict(step_data) if isinstance(step_data, Mapping) else {}
        parsed_action = data.get("action_output_json")
        parsed = isinstance(parsed_action, Mapping) or (
            parsed_action is not None
            and isinstance(getattr(parsed_action, "action_type", None), str)
        )
        summary_text = str(data.get("summary", ""))
        failed_markers = (
            "no action is performed",
            "can not parse",
            "cannot parse",
            "out of range",
            "can not execute",
            "cannot execute",
        )
        action_committed = parsed and not any(
            marker in summary_text.lower() for marker in failed_markers
        )
        unit = self._pending_unit
        if unit is not None and action_committed:
            if self.remaining_units and self.remaining_units[0] == unit:
                self.remaining_units.pop(0)
            self.completed_units.append(unit)
        self.latest_step = {
            "action_parsed": parsed,
            "action_committed": action_committed,
            "action_output": str(data.get("action_output", ""))[:2000],
            "summary": summary_text[:2000],
            "error": str(data.get("error", data.get("error_message", "")))[:2000],
            "planned_unit": unit.to_dict() if unit else None,
        }
        raw = data.get("action_raw_response")
        if isinstance(raw, dict):
            route = raw.get("_pmtskill")
            if isinstance(route, dict):
                route["action_parsed"] = parsed
                route["action_committed"] = action_committed
                route["completed_units"] = [
                    item.to_dict() for item in self.completed_units
                ]
                route["remaining_units_after_step"] = [
                    item.to_dict() for item in self.remaining_units
                ]
        self._pending_unit = None

    def predict(self, text_prompt: str):
        return self.predict_mm(text_prompt, [])
