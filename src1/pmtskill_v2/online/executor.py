"""在线计划执行、polished skill 失败降级与 VL 路由 wrapper。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..core.models import (
    ExecutionPlan,
    ExecutionTrace,
    ModelProfile,
    RouteStep,
    SkillRecord,
    SkillTopology,
    TraceEvent,
)
from ..inference.model_pool import ModelPool
from ..skills.store import SkillStore
from .router import DynamicProgrammingRouter


@dataclass(slots=True)
class StepExecutionResult:
    success: bool
    action: str | None
    observation: str | None
    latency_ms: float
    context_updates: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ExecutionBackend(Protocol):
    """设备控制层替换点，可实现 Android、浏览器或离线模拟器。"""

    def execute(
        self, step: RouteStep, context: dict[str, Any], skill: SkillRecord | None
    ) -> StepExecutionResult:
        ...


class VLInferenceBackend:
    """执行一次模型/技能推理；设备动作可由上层 AndroidWorld M3A 解析。"""

    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool

    def execute(
        self, step: RouteStep, context: dict[str, Any], skill: SkillRecord | None
    ) -> StepExecutionResult:
        skill_instruction = skill.body if skill else "逐个执行所给原语，并严格校验结果。"
        prompt = (
            f"任务目标：{context.get('goal', '')}\n"
            f"当前原语：{', '.join(step.primitive_ids)}\n"
            f"技能指令：{skill_instruction}\n"
            f"历史：{context.get('history', '')}\n"
            "根据当前截图输出下一步 AndroidWorld M3A 格式的 Reason 和 Action。"
        )
        try:
            with self.model_pool.use(step.model_id) as client:
                result = client.generate(prompt, context.get("images", ()), max_tokens=1024)
            return StepExecutionResult(
                success=bool(result.text),
                action=result.text,
                observation=None,
                latency_ms=result.latency_ms,
            )
        except Exception as exc:
            return StepExecutionResult(False, None, None, 0.0, error=str(exc))


class OnlineExecutor:
    """通用计划执行器，负责记录轨迹与 polished skill 安全降级。"""

    def __init__(
        self,
        router: DynamicProgrammingRouter,
        backend: ExecutionBackend,
        store: SkillStore,
        models: Sequence[ModelProfile],
        skills: Sequence[SkillRecord],
    ):
        self.router = router
        self.backend = backend
        self.store = store
        self.models = tuple(models)
        self.skills = tuple(skills)
        self.skills_by_id = {skill.skill_id: skill for skill in skills}

    def execute(
        self,
        goal: str,
        task_name: str,
        topology: SkillTopology,
        *,
        context: dict[str, Any] | None = None,
        planner_id: str = "unknown",
    ) -> ExecutionTrace:
        """执行路由；组合技能失败时禁用该技能并重路由其 fallback。"""

        runtime_context = dict(context or {})
        runtime_context["goal"] = goal
        plan = self.router.route(
            goal, topology, self.models, self.skills, planner_id=planner_id
        )
        events: list[TraceEvent] = []
        started = time.perf_counter()
        all_successful = True
        event_index = 0
        for route_step in plan.steps:
            skill = self.skills_by_id.get(route_step.skill_id or "")
            result = self.backend.execute(route_step, runtime_context, skill)
            events.append(
                TraceEvent(
                    index=event_index,
                    model_id=route_step.model_id,
                    skill_id=route_step.skill_id,
                    primitive_ids=route_step.primitive_ids,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    action=result.action,
                    observation=result.observation,
                    error=result.error,
                )
            )
            event_index += 1
            runtime_context.update(result.context_updates)
            if result.success:
                continue
            if skill and skill.fallback_topology:
                fallback = self.router.route(
                    goal,
                    skill.fallback_topology,
                    self.models,
                    self.skills,
                    planner_id=planner_id,
                    banned_skill_ids={skill.skill_id},
                )
                for fallback_step in fallback.steps:
                    fallback_skill = self.skills_by_id.get(fallback_step.skill_id or "")
                    fallback_result = self.backend.execute(
                        fallback_step, runtime_context, fallback_skill
                    )
                    events.append(
                        TraceEvent(
                            index=event_index,
                            model_id=fallback_step.model_id,
                            skill_id=fallback_step.skill_id,
                            primitive_ids=fallback_step.primitive_ids,
                            success=fallback_result.success,
                            latency_ms=fallback_result.latency_ms,
                            action=fallback_result.action,
                            observation=fallback_result.observation,
                            error=fallback_result.error,
                            metadata={"fallback_from": skill.skill_id},
                        )
                    )
                    event_index += 1
                    runtime_context.update(fallback_result.context_updates)
                    if not fallback_result.success:
                        all_successful = False
                        break
            else:
                all_successful = False
            if not all_successful:
                break
        trace = ExecutionTrace.new(
            goal,
            task_name,
            all_successful,
            events,
            plan=plan,
            reward=float(all_successful),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        self.store.append_trace(trace)
        return trace


class RoutedVLWrapper:
    """让 AndroidWorld M3A 每一步按计划动态选择模型与技能。

    M3A 每个环境 step 会调用两次 VL：动作选择和动作总结。两次保持同一个模型，
    只有下一次动作选择才前进到下一 route step，从而减少 adapter 切换。
    """

    def __init__(self, model_pool: ModelPool, skills: Sequence[SkillRecord]):
        self.model_pool = model_pool
        self.skills = {skill.skill_id: skill for skill in skills}
        self.plan: ExecutionPlan | None = None
        self.action_call_index = 0
        self._last_action_step: RouteStep | None = None
        self.call_log: list[dict[str, Any]] = []

    def set_plan(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.action_call_index = 0
        self._last_action_step = None
        self.call_log = []

    def _route_step(self) -> RouteStep:
        if self.plan is None or not self.plan.steps:
            raise RuntimeError("RoutedVLWrapper 尚未设置执行计划")
        index = min(self.action_call_index, len(self.plan.steps) - 1)
        return self.plan.steps[index]

    def predict_mm(
        self, text_prompt: str, images: list[Any]
    ) -> tuple[str, bool | None, dict[str, Any] | None]:
        is_summary = "summerize the latest step" in text_prompt.lower()
        # 动作总结必须沿用刚才动作选择的模型，不能提前切到下一 route step。
        route_step = (
            self._last_action_step
            if is_summary and self._last_action_step is not None
            else self._route_step()
        )
        skill = self.skills.get(route_step.skill_id or "")
        augmented = text_prompt
        if not is_summary:
            augmented += (
                "\n\nPMT-Skill 路由约束：\n"
                f"- 当前原语：{', '.join(route_step.primitive_ids)}\n"
                f"- 当前技能：{skill.name if skill else 'primitive fallback'}\n"
                f"- 技能说明：{skill.body[:3000] if skill else '按原语完成一个可靠动作。'}\n"
                "以上约束不能改变 AndroidWorld 要求的 Reason/Action 输出格式。"
            )
        client = self.model_pool.client(route_step.model_id)
        output, safe, raw = client.predict_mm(augmented, images)  # type: ignore[attr-defined]
        route_metadata = {
            "model_id": route_step.model_id,
            "skill_id": route_step.skill_id,
            "primitive_ids": list(route_step.primitive_ids),
            "is_summary": is_summary,
        }
        if raw is not None:
            raw = dict(raw)
            raw.setdefault("_pmtskill", {}).update(route_metadata)
        self.call_log.append(route_metadata)
        if not is_summary:
            self._last_action_step = route_step
            self.action_call_index += 1
        return output, safe, raw

    def predict(self, text_prompt: str):
        return self.predict_mm(text_prompt, [])
