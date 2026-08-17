"""在 ``模型池 × polished-skill 图`` 上进行动态规划路由。"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..core.config import RoutingConfig
from ..core.models import (
    ExecutionPlan,
    ModelProfile,
    RouteStep,
    SkillRecord,
    SkillStatus,
    SkillTopology,
    geometric_success,
)
from ..skills.store import SkillStore


class RoutingAlgorithm(Protocol):
    """路由算法替换接口。"""

    algorithm_id: str

    def route(
        self,
        goal: str,
        topology: SkillTopology,
        models: Sequence[ModelProfile],
        skills: Sequence[SkillRecord],
        **kwargs,
    ) -> ExecutionPlan:
        ...


@dataclass(slots=True)
class _Candidate:
    primitive_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    skill: SkillRecord | None


@dataclass(slots=True)
class _State:
    score: float
    steps: tuple[RouteStep, ...]


class DynamicProgrammingRouter:
    """最小化延迟与切换、最大化预计成功率的可解释 DP。

    当前版本按 DAG 的稳定拓扑序路由。并行节点仍保留依赖信息，但实际并行调度
    留给执行器；未来要换成图搜索/强化学习，只需实现 :class:`RoutingAlgorithm`。
    """

    algorithm_id = "dp-model-skill-router-v1"

    def __init__(self, config: RoutingConfig, store: SkillStore | None = None):
        self.config = config
        self.store = store

    def _candidates(
        self,
        position: int,
        primitives: tuple[str, ...],
        node_ids: tuple[str, ...],
        skills: Sequence[SkillRecord],
        banned: set[str],
        include_candidates: bool,
    ) -> list[_Candidate]:
        # 单原语路径永远存在，保证 polished skill 失效时可降级。
        candidates = [_Candidate((primitives[position],), (node_ids[position],), None)]
        allowed_status = {SkillStatus.ACTIVE}
        if include_candidates:
            allowed_status.add(SkillStatus.CANDIDATE)
        for skill in skills:
            if skill.skill_id in banned or skill.kind != "polished" or skill.status not in allowed_status:
                continue
            covered = skill.topology.primitive_sequence()
            if not covered:
                continue
            end = position + len(covered)
            if primitives[position:end] == covered:
                candidates.append(
                    _Candidate(covered, node_ids[position:end], skill)
                )
        candidates.sort(
            key=lambda candidate: (
                -len(candidate.primitive_ids),
                candidate.skill.skill_id if candidate.skill else "",
            )
        )
        return candidates[: self.config.maximum_candidates_per_position]

    def _probability(
        self, model: ModelProfile, candidate: _Candidate
    ) -> tuple[float, float]:
        probability = geometric_success(
            model.capability(primitive) for primitive in candidate.primitive_ids
        )
        latency = model.average_latency_ms
        if candidate.skill and self.store:
            metrics = self.store.skill_metrics(candidate.skill.skill_id, model.model_id)
            # 真实样本达到 3 次后逐渐覆盖能力画像先验。
            trials = int(metrics["trials"])
            if trials:
                weight = min(0.90, trials / (trials + 5))
                probability = (
                    weight * float(metrics["smoothed_success_rate"])
                    + (1 - weight) * probability
                )
                if metrics["average_latency_ms"]:
                    latency = float(metrics["average_latency_ms"])
        return probability, latency

    def route(
        self,
        goal: str,
        topology: SkillTopology,
        models: Sequence[ModelProfile],
        skills: Sequence[SkillRecord],
        *,
        planner_id: str = "unknown",
        banned_skill_ids: set[str] | None = None,
        include_candidates: bool = False,
    ) -> ExecutionPlan:
        topology.validate()
        ordered = topology.topological_nodes()
        primitives = tuple(node.primitive_id for node in ordered)
        node_ids = tuple(node.node_id for node in ordered)
        enabled_models = [model for model in models if model.enabled]
        if not enabled_models:
            raise ValueError("路由失败：没有启用模型")
        if not primitives:
            return ExecutionPlan(goal, topology, (), 0.0, planner_id)

        states: dict[tuple[int, str | None], _State] = {(0, None): _State(0.0, ())}
        banned = banned_skill_ids or set()
        for position in range(len(primitives)):
            position_states = [
                (key, state) for key, state in states.items() if key[0] == position
            ]
            if not position_states:
                continue
            candidates = self._candidates(
                position,
                primitives,
                node_ids,
                skills,
                banned,
                include_candidates,
            )
            for (_, previous_model_id), state in position_states:
                for candidate in candidates:
                    for model in enabled_models:
                        capabilities = [model.capability(p) for p in candidate.primitive_ids]
                        if min(capabilities) < self.config.minimum_capability:
                            continue
                        probability, latency = self._probability(model, candidate)
                        switch_cost = (
                            model.switch_cost_ms
                            if previous_model_id is not None and previous_model_id != model.model_id
                            else 0.0
                        )
                        polished_gain = (
                            self.config.polished_bonus * (len(candidate.primitive_ids) - 1)
                            if candidate.skill
                            else 0.0
                        )
                        degradation = 0.0 if candidate.skill else self.config.degradation_weight
                        success_term = self.config.success_weight * math.log(
                            max(0.000001, probability)
                        )
                        latency_term = self.config.latency_weight * latency
                        switch_term = self.config.switch_weight * switch_cost
                        delta = success_term - latency_term - switch_term + polished_gain - degradation
                        step = RouteStep(
                            step_id=uuid.uuid4().hex,
                            model_id=model.model_id,
                            skill_id=candidate.skill.skill_id if candidate.skill else None,
                            primitive_ids=candidate.primitive_ids,
                            node_ids=candidate.node_ids,
                            expected_success=probability,
                            expected_latency_ms=latency,
                            score=delta,
                            is_polished=candidate.skill is not None,
                            score_detail={
                                "log_success": success_term,
                                "latency_penalty": latency_term,
                                "switch_penalty": switch_term,
                                "polished_bonus": polished_gain,
                                "degradation_penalty": degradation,
                            },
                        )
                        next_position = position + len(candidate.primitive_ids)
                        key = (next_position, model.model_id)
                        new_state = _State(state.score + delta, state.steps + (step,))
                        old_state = states.get(key)
                        if old_state is None or new_state.score > old_state.score:
                            states[key] = new_state

        finals = [state for (position, _), state in states.items() if position == len(primitives)]
        if not finals:
            raise RuntimeError("路由失败：能力阈值过滤了所有模型候选")
        best = max(finals, key=lambda state: state.score)
        return ExecutionPlan(
            goal=goal,
            topology=topology,
            steps=best.steps,
            total_score=best.score,
            planner_id=planner_id,
            metadata={"algorithm": self.algorithm_id},
        )

