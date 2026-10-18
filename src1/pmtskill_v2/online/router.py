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
from ..skills.identity import model_variant_id, skill_version_id
from ..evaluation.resume import fingerprint, model_resume_identity


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
    success_score: float = 0.0


class DynamicProgrammingRouter:
    """最小化延迟与切换、最大化预计成功率的可解释 DP。

    当前版本按 DAG 的稳定拓扑序路由。并行节点仍保留依赖信息，但实际并行调度
    留给执行器；未来要换成图搜索/强化学习，只需实现 :class:`RoutingAlgorithm`。
    """

    algorithm_id = "dp-model-skill-router-v2"

    def __init__(self, config: RoutingConfig, store: SkillStore | None = None):
        self.config = config
        self.store = store
        # 一轮在线评测中标定矩阵不变；缓存避免 DP 内层为同一 pair 反复扫描 trial。
        self._calibration_cache: dict[tuple[str, str], dict[str, object]] = {}
        self._pair_status_cache: dict[
            tuple[str, str], dict[str, object] | None
        ] = {}
        self._skill_has_matrix_cache: dict[str, bool] = {}
        self._estimate_cache: dict[tuple[str, str], dict | None] = {}
        self._model_weight_cache: dict[str, str] = {}
        self._score_sources: dict[tuple[str, str], str] = {}

    def _estimate(self, model: ModelProfile, skill: SkillRecord) -> dict | None:
        if not self.store or not self.config.use_skill_estimates:
            return None
        key = (skill_version_id(skill), model_variant_id(model))
        if key not in self._estimate_cache:
            estimate = self.store.skill_model_estimate(skill, model)
            if estimate is not None:
                if key[1] not in self._model_weight_cache:
                    self._model_weight_cache[key[1]] = fingerprint(model_resume_identity(model))
                if estimate.get("model_weight_fingerprint") != self._model_weight_cache[key[1]]:
                    estimate = None
            self._estimate_cache[key] = estimate
        return self._estimate_cache[key]

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
            positions = skill.metadata.get("routing_positions")
            if positions is not None and position not in positions:
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
        selected = candidates[:max(1, self.config.maximum_candidates_per_position)]
        # Truncating the catalogue must not accidentally remove the only fallback.
        if not any(candidate.skill is None for candidate in selected):
            fallback = next(candidate for candidate in candidates if candidate.skill is None)
            if len(selected) == 1:
                selected.append(fallback)
            else:
                selected[-1] = fallback
        return selected

    def model_unit_support(self, model: ModelProfile, step: RouteStep, skill: SkillRecord | None,
                           *, include_candidates: bool = False) -> tuple[float, float, str] | None:
        """Use the same lifecycle/matrix evidence in online model re-selection."""
        candidate = _Candidate(step.primitive_ids, step.node_ids, skill)
        if not self._model_skill_allowed(model, candidate, include_candidates=include_candidates):
            return None
        probability, latency = self._probability(model, candidate)
        if skill is None and step.primitive_ids:
            # These are simultaneous FEATURES of one GUI action, not a path of
            # independent executed actions. Normalize rather than exponentiating
            # uncertainty merely because more context features were available.
            probability = math.exp(sum(math.log(model.capability(p)) for p in step.primitive_ids)
                                   / len(step.primitive_ids))
        if probability < self.config.minimum_capability and min(
                (model.capability(p) for p in step.primitive_ids), default=probability) < self.config.minimum_capability:
            return None
        source = self._score_sources.get((skill_version_id(skill), model_variant_id(model)), "primitive_prior") if skill else "primitive_prior"
        return probability, latency, source

    def _rank(self, state: _State) -> tuple:
        # Equivalent products computed through different segmentations differ
        # at machine precision; do not let that defeat the intended tie-break.
        return (round(state.success_score, 12), state.score) if self.config.objective == "success_first" else (state.score,)

    def _probability(
        self, model: ModelProfile, candidate: _Candidate
    ) -> tuple[float, float]:
        probability = geometric_success(
            model.capability(primitive) for primitive in candidate.primitive_ids
        )
        latency = model.average_latency_ms
        if candidate.skill and self.store:
            pair_key = (
                skill_version_id(candidate.skill),
                model_variant_id(model),
            )
            calibration = self._calibration_cache.get(pair_key)
            if calibration is None:
                calibration = self.store.calibration_metrics(
                    candidate.skill.skill_id,
                    model_id=model.model_id,
                    model_variant_id_value=pair_key[1],
                )
                self._calibration_cache[pair_key] = calibration
            calibration_trials = int(calibration["valid_trials"])
            if calibration_trials:
                self._score_sources[pair_key] = "observed_calibration"
                # SKVM-style 配对标定有明确 no-skill 对照，且绑定真实 adapter 版本，
                # 优先级高于在线自然流量；小样本仍与原语先验平滑混合。
                weight = min(0.95, calibration_trials / (calibration_trials + 5))
                probability = (
                    weight * float(calibration["smoothed_skill_success_rate"])
                    + (1 - weight) * probability
                )
            else:
                metrics = self.store.skill_metrics(
                    candidate.skill.skill_id, model.model_id
                )
                trials = int(metrics["trials"])
                if trials:
                    self._score_sources[pair_key] = "online_observations"
                    weight = min(0.90, trials / (trials + 5))
                    probability = (
                        weight * float(metrics["smoothed_success_rate"])
                        + (1 - weight) * probability
                    )
                    if metrics["average_latency_ms"]:
                        latency = float(metrics["average_latency_ms"])
                else:
                    estimate = self._estimate(model, candidate.skill)
                    if estimate is not None:
                        probability = min(0.999999, max(0.000001, float(estimate["estimated_support_score"])))
                        self._score_sources[pair_key] = "estimated_not_sr"
        return probability, latency

    def _model_skill_allowed(
        self,
        model: ModelProfile,
        candidate: _Candidate,
        *,
        include_candidates: bool,
    ) -> bool:
        """应用模型专属生命周期；未迁移的旧技能库保持原有行为。"""

        if candidate.skill is None or self.store is None:
            return True
        pair_key = (skill_version_id(candidate.skill), model_variant_id(model))
        if pair_key not in self._pair_status_cache:
            self._pair_status_cache[pair_key] = self.store.skill_model_status(
                candidate.skill, model
            )
        pair = self._pair_status_cache[pair_key]
        if pair is not None:
            return pair["status"] == SkillStatus.ACTIVE.value or (
                include_candidates
                and pair["status"] == SkillStatus.CANDIDATE.value
            )
        if self._estimate(model, candidate.skill) is not None:
            # 估计不是已验证 active；仅允许已被上游全局生命周期筛选的技能候选。
            # 已有当前模型实测状态时，上面的分支始终优先，不能用估计绕过回滚。
            return True
        # 一旦当前技能版本已有兼容矩阵，未标定的新模型视为 unknown，而不是
        # 沿用其他 adapter 的成功率。灰度模式仍允许探索它。
        if candidate.skill.skill_id not in self._skill_has_matrix_cache:
            self._skill_has_matrix_cache[candidate.skill.skill_id] = (
                self.store.has_current_skill_model_status(candidate.skill)
                or self.store.has_any_skill_model_status(candidate.skill.skill_id)
            )
        has_matrix = self._skill_has_matrix_cache[candidate.skill.skill_id]
        return include_candidates if has_matrix else True

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
                        if not self._model_skill_allowed(
                            model,
                            candidate,
                            include_candidates=include_candidates,
                        ):
                            continue
                        probability, latency = self._probability(model, candidate)
                        if probability < self.config.minimum_capability and min(
                                model.capability(p) for p in candidate.primitive_ids) < self.config.minimum_capability:
                            continue
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
                        # Existing TOMLs may retain old, large weights. In this
                        # mode they cannot trade away even a small success gain.
                        latency_term = self.config.latency_weight * latency if self.config.objective == "weighted" else 0.0
                        switch_term = self.config.switch_weight * switch_cost if self.config.objective == "weighted" else 0.0
                        delta = success_term - latency_term - switch_term + polished_gain - degradation
                        step = RouteStep(
                            step_id=uuid.uuid4().hex,
                            model_id=model.model_id,
                            skill_id=candidate.skill.skill_id if candidate.skill else None,
                            primitive_ids=candidate.primitive_ids,
                            node_ids=candidate.node_ids,
                            expected_success=probability,
                            expected_latency_ms=latency,
                            score=success_term if self.config.objective == "success_first" else delta,
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
                        new_state = _State(state.score + delta, state.steps + (step,), state.success_score + success_term)
                        old_state = states.get(key)
                        if old_state is None or self._rank(new_state) > self._rank(old_state):
                            states[key] = new_state

        finals = [state for (position, _), state in states.items() if position == len(primitives)]
        if not finals:
            raise RuntimeError("路由失败：能力阈值过滤了所有模型候选")
        best = max(finals, key=self._rank)
        return ExecutionPlan(
            goal=goal,
            topology=topology,
            steps=best.steps,
            total_score=best.success_score if self.config.objective == "success_first" else best.score,
            planner_id=planner_id,
            metadata={"algorithm": self.algorithm_id,
                      "objective": self.config.objective, "log_success_objective": best.success_score,
                      "skill_score_sources": {step.step_id: self._score_sources.get(
                          (skill_version_id(next(skill for skill in skills if skill.skill_id == step.skill_id)),
                           model_variant_id(next(model for model in enabled_models if model.model_id == step.model_id))),
                          "primitive_prior") for step in best.steps if step.skill_id}},
        )
