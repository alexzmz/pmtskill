"""动态模型/技能路由版 AndroidWorld 在线评测。"""

from __future__ import annotations

import datetime as dt
import random
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from ..core.config import ProjectConfig
from ..core.io import load_primitives
from ..core.models import (
    ExecutionPlan,
    ExecutionTrace,
    ModelProfile,
    RouteStep,
    SkillRecord,
    SkillStatus,
    SkillTopology,
    TraceEvent,
)
from ..inference.model_pool import ModelPool
from ..inference.vlm import OpenAICompatibleVLClient
from ..offline.collector import bootstrap_android_world
from ..offline.dataset import infer_action_primitives
from ..online.executor import (
    ForcedSkillVLWrapper,
    RoutedVLWrapper,
    SimpleSkillVLWrapper,
)
from ..online.planner import (
    KeywordSkillPlanner,
    LLMSkillPlanner,
    PlannerPipeline,
    PrimitiveTopologyGenerator,
)
from ..online.router import DynamicProgrammingRouter
from ..skills.importer import relevant_raw_skills
from ..skills.store import SkillStore
from .compaction import compact_m3a_step_result
from .reporter import (
    EvaluationArtifacts,
    episode_data_is_usable,
    episode_step_values,
    finite_float_value,
    normalize_episode_data,
    successful_episode_value,
    write_evaluation_report,
)
from .recovery import (
    ensure_valid_evaluation_episodes,
    recover_infrastructure_failures,
)


DEFAULT_EVALUATION_MAX_STEPS = 30


def _evaluation_max_steps(value: int) -> int:
    if value <= 0:
        raise ValueError("评测 max_steps 必须是正整数")
    return value


def _extract_route_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        value = raw.get("_pmtskill", {})
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _is_parsed_action(value: Any) -> bool:
    """兼容 checkpoint 中的 JSON dict 与 AndroidWorld ``JSONAction`` dataclass。"""

    return isinstance(value, Mapping) or (
        value is not None and isinstance(getattr(value, "action_type", None), str)
    )


def _execution_plan_from_route(goal: str, route: Mapping[str, Any]) -> ExecutionPlan | None:
    """把 Teacher wrapper 的初始 action-unit 计划恢复成强类型审计对象。"""

    raw_units = route.get("initial_plan")
    if not isinstance(raw_units, list):
        return None
    primitive_sequence: list[str] = []
    normalized: list[tuple[Mapping[str, Any], tuple[str, ...], tuple[str, ...]]] = []
    for raw_unit in raw_units:
        if not isinstance(raw_unit, Mapping):
            continue
        raw_primitives = raw_unit.get("primitive_ids", ())
        if isinstance(raw_primitives, str):
            raw_primitives = (raw_primitives,)
        if not isinstance(raw_primitives, (list, tuple)):
            continue
        primitives = tuple(str(item) for item in raw_primitives if str(item))
        if not primitives:
            continue
        start = len(primitive_sequence)
        primitive_sequence.extend(primitives)
        node_ids = tuple(f"n{index:04d}" for index in range(start, len(primitive_sequence)))
        normalized.append((raw_unit, primitives, node_ids))
    if not primitive_sequence:
        return None
    topology = SkillTopology.from_sequence(
        primitive_sequence, topology_id=f"teacher-plan:{uuid.uuid4().hex[:16]}"
    )
    model_id = str(route.get("model_id", "unknown"))
    steps = tuple(
        RouteStep(
            step_id=str(raw.get("unit_id", f"unit-{index + 1:03d}")),
            model_id=model_id,
            skill_id=(str(raw["skill_id"]) if raw.get("skill_id") else None),
            primitive_ids=primitives,
            node_ids=node_ids,
            expected_success=0.0,
            expected_latency_ms=0.0,
            score=0.0,
            is_polished=bool(raw.get("skill_id")),
        )
        for index, (raw, primitives, node_ids) in enumerate(normalized)
    )
    return ExecutionPlan(
        goal=goal,
        topology=topology,
        steps=steps,
        total_score=0.0,
        planner_id=str(route.get("planner_id", "unknown")),
        metadata={
            "trajectory_schema": route.get("trajectory_schema"),
            "source": route.get("source"),
            "planned_step_count": route.get("planned_step_count", len(steps)),
        },
    )


def episodes_to_traces(episodes: Sequence[dict[str, Any]]) -> list[ExecutionTrace]:
    """把 M3A episode 转成 backend 接收的轻量轨迹并进行保守信用分配。"""

    traces: list[ExecutionTrace] = []
    for episode in episodes:
        successful = successful_episode_value(episode.get("is_successful", False))
        raw_episode_data = episode.get("episode_data", {})
        episode_data = normalize_episode_data(raw_episode_data)
        raw_responses = episode_step_values(
            episode_data.get("action_raw_response")
        )
        actions = episode_step_values(episode_data.get("action_output"))
        parsed_actions = episode_step_values(
            episode_data.get("action_output_json")
        )
        summaries = episode_step_values(episode_data.get("summary"))
        events: list[TraceEvent] = []
        route_snapshots: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_responses):
            route = _extract_route_metadata(raw)
            if not route:
                continue
            route_snapshots.append(route)
            action_parsed = index < len(parsed_actions) and _is_parsed_action(
                parsed_actions[index]
            )
            action_committed = bool(route.get("action_committed", action_parsed))
            # 组合技能只有在动作可解析且 episode 最终成功时计为成功，避免把局部
            # 看似正确、实际使任务失败的动作错误地用于技能晋升。
            event_success = bool(action_committed and successful)
            action_text = str(actions[index]) if index < len(actions) else None
            actual_primitives = (
                infer_action_primitives(action_text) if action_text else ()
            )
            planned_actions = {
                str(item)
                for item in episode_step_values(route.get("primitive_ids"))
                if str(item).startswith("action.") or str(item) == "control.finish"
            }
            actual_actions = {
                item
                for item in actual_primitives
                if item.startswith("action.") or item == "control.finish"
            }
            aligned = (
                bool(planned_actions.intersection(actual_actions))
                if planned_actions
                else None
            )
            planned_primitives = tuple(
                str(item)
                for item in episode_step_values(route.get("primitive_ids"))
                if str(item)
            )
            executed_primitives = planned_primitives
            if route.get("trajectory_schema") and action_committed and actual_actions:
                # 标准 Teacher trace 同时保存 planned_unit 和真实执行序列。模型若为
                # 恢复弹窗临时选择了不同动作，技能挖掘必须使用真实 action，而不能
                # 把原计划误写成已执行事实。感知/定位/推理原语仍予以保留。
                non_actions = tuple(
                    item
                    for item in planned_primitives
                    if not item.startswith("action.") and item != "control.finish"
                )
                executed_primitives = non_actions + tuple(sorted(actual_actions))
            events.append(
                TraceEvent(
                    index=len(events),
                    model_id=str(route.get("model_id", "unknown")),
                    skill_id=(
                        str(route["skill_id"])
                        if route.get("skill_id") is not None
                        else None
                    ),
                    primitive_ids=executed_primitives,
                    success=event_success,
                    latency_ms=finite_float_value(route.get("latency_ms")),
                    action=action_text,
                    observation=(
                        str(summaries[index]) if index < len(summaries) else None
                    ),
                    metadata={"credit_assignment": "parsed_action_and_episode_success"},
                )
            )
            events[-1].metadata.update(
                {
                    "trajectory_schema": route.get("trajectory_schema"),
                    "routing_mode": route.get("routing_mode"),
                    "source": route.get("source"),
                    "planner_id": route.get("planner_id"),
                    "plan_revision": route.get("current_revision"),
                    "planned_unit": route.get("planned_unit"),
                    "planned_primitive_ids": list(planned_primitives),
                    "action_parsed": action_parsed,
                    "action_committed": action_committed,
                    "actual_action_primitives": list(actual_primitives),
                    "action_primitive_aligned": aligned,
                }
            )
        first_route = route_snapshots[0] if route_snapshots else {}
        last_route = route_snapshots[-1] if route_snapshots else {}
        aligned_values = [
            event.metadata.get("action_primitive_aligned")
            for event in events
            if event.metadata.get("action_primitive_aligned") is not None
        ]
        plan = _execution_plan_from_route(str(episode.get("goal", "")), first_route)
        traces.append(
            ExecutionTrace.new(
                goal=str(episode.get("goal", "")),
                task_name=str(episode.get("task_template", "unknown")),
                successful=successful,
                events=events,
                plan=plan,
                reward=float(successful),
                duration_ms=finite_float_value(episode.get("run_time")) * 1000,
                metadata={
                    "source": "android_world_m3a",
                    "episode_data_valid": episode_data_is_usable(raw_episode_data),
                    "had_exception": bool(episode.get("exception_info")),
                    "instance_id": episode.get("instance_id"),
                    "agent_name": episode.get("agent_name"),
                    "trajectory_schema": first_route.get("trajectory_schema"),
                    "metric_scope": first_route.get("metric_scope", "online"),
                    "collection_source": first_route.get("source"),
                    "collection_context": first_route.get("collection_context"),
                    "teacher_model_id": first_route.get("model_id"),
                    "planner_id": first_route.get("planner_id"),
                    "planned_step_count": first_route.get("planned_step_count"),
                    "initial_plan": first_route.get("initial_plan"),
                    "plan_revisions": last_route.get("plan_revisions", ()),
                    "completed_units": last_route.get("completed_units", ()),
                    "executed_primitive_sequence": [
                        primitive for event in events for primitive in event.primitive_ids
                    ],
                    "action_primitive_alignment_rate": (
                        sum(bool(value) for value in aligned_values) / len(aligned_values)
                        if aligned_values
                        else None
                    ),
                },
            )
        )
    return traces


def sample_android_world_tasks(
    config: ProjectConfig,
    *,
    tasks: Sequence[str] | None,
    task_count: int | None,
    seed: int,
    family: str = "android_world",
) -> list[str]:
    """解析一组固定评测任务，供所有训练阶段公平复用。

    显式传入任务时保留用户顺序且仅去重；未传入且 task_count 为 ``None`` 时
    返回 family 全部任务，否则按 seed 抽样。这里只读取任务注册表，不连接 emulator。
    """

    if tasks:
        return list(dict.fromkeys(str(item) for item in tasks))
    if task_count is not None and task_count <= 0:
        raise ValueError("评测 task_count 必须是正整数")
    bootstrap_android_world(config.paths.android_world_root)
    from android_world import registry

    available = sorted(registry.TaskRegistry().get_registry(family=family))
    if not available:
        raise ValueError(f"AndroidWorld family 没有可评测任务: {family}")
    if task_count is None:
        return available
    count = min(task_count, len(available))
    return sorted(random.Random(seed).sample(available, count))


class AndroidWorldStandaloneEvaluator:
    """用单个 VL 模型和原生 M3A 评测，不注入任何技能或动态路由。"""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def run(
        self,
        *,
        profile: ModelProfile,
        tasks: Sequence[str] | None,
        n_task_combinations: int = 1,
        seed: int = 42,
        family: str = "android_world",
        max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
        output_dir: str | Path | None = None,
    ) -> EvaluationArtifacts:
        max_steps = _evaluation_max_steps(max_steps)
        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = Path(output_dir).resolve() if output_dir else (
            self.config.paths.state_dir
            / "evaluations"
            / f"standalone_{run_stamp}_{uuid.uuid4().hex[:8]}"
        )
        checkpoint_dir = target / "checkpoints"
        model = OpenAICompatibleVLClient(profile)

        class MemoryBoundedM3A(m3a.M3A):
            """评测不跨 step 保留截图，避免长 suite 被宿主 OOM killer 杀死。"""

            def step(self, goal: str):
                return compact_m3a_step_result(super().step(goal))

        environment = env_launcher.load_and_setup_env(
            console_port=self.config.android_world.console_port,
            emulator_setup=self.config.android_world.emulator_setup,
            adb_path=self.config.android_world.adb_path,
        )
        try:
            task_registry = registry.TaskRegistry()
            suite = suite_utils.create_suite(
                task_registry.get_registry(family=family),
                n_task_combinations=n_task_combinations,
                seed=seed,
                tasks=list(tasks) if tasks else None,
                env=environment,
            )
            suite.suite_family = family
            agent = MemoryBoundedM3A(
                environment,
                model,
                name=f"standalone:{profile.model_id}",
                wait_after_action_seconds=(
                    self.config.android_world.wait_after_action_seconds
                ),
            )
            with recover_infrastructure_failures(
                suite_utils, environment, self.config.android_world
            ):
                episodes = suite_utils.run(
                    suite,
                    agent,
                    checkpointer=checkpointer_lib.IncrementalCheckpointer(
                        str(checkpoint_dir)
                    ),
                    demo_mode=False,
                    return_full_episode_data=True,
                    max_n_steps_override=max_steps,
                    stop_on_task_success=self.config.android_world.stop_on_task_success,
                )
            ensure_valid_evaluation_episodes(
                episodes,
                expected_episodes=sum(len(instances) for instances in suite.values()),
            )
        finally:
            environment.close()

        traces = episodes_to_traces(episodes)
        return write_evaluation_report(
            target,
            episodes,
            traces,
            metadata={
                "evaluation_mode": "standalone",
                "model_id": profile.model_id,
                "served_model": profile.served_model,
                "adapter": profile.adapter,
                "evaluation_checkpoint": profile.metadata.get(
                    "evaluation_checkpoint"
                ),
                "adapter_resolution": dict(profile.metadata),
                "family": family,
                "tasks": list(tasks) if tasks else "all",
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "max_steps": max_steps,
                "stop_on_task_success": self.config.android_world.stop_on_task_success,
                "episode_storage": "compact_without_screenshots",
            },
        )


class AndroidWorldSimpleSkillEvaluator:
    """单个 VL adapter + 单次关键词技能检索的消融评测。

    与 :class:`AndroidWorldOnlineEvaluator` 的关键区别是：这里没有原语拓扑展开、
    动态规划路由或模型切换。每个 episode 仅检索一个技能并注入同一个模型的 prompt，
    因而可以单独衡量“简单使用技能库”相对裸模型带来的变化。
    """

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def _skills(self, include_candidates: bool) -> list[SkillRecord]:
        raw = relevant_raw_skills(self.store.list_skills(kind="raw"))
        allowed = {SkillStatus.ACTIVE}
        if include_candidates:
            allowed.add(SkillStatus.CANDIDATE)
        polished = [
            skill
            for skill in self.store.list_skills(kind="polished")
            if skill.status in allowed
        ]
        # 同一 skill_id 只保留一次，顺序固定以保证三次复现实验选择一致。
        return list({skill.skill_id: skill for skill in (*polished, *raw)}.values())

    def run(
        self,
        *,
        profile: ModelProfile,
        tasks: Sequence[str] | None,
        n_task_combinations: int = 1,
        seed: int = 42,
        family: str = "android_world",
        max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
        include_candidate_skills: bool = False,
        output_dir: str | Path | None = None,
        record_traces: bool = False,
    ) -> EvaluationArtifacts:
        max_steps = _evaluation_max_steps(max_steps)
        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        skills = self._skills(include_candidate_skills)
        wrapper = SimpleSkillVLWrapper(
            OpenAICompatibleVLClient(profile), skills, model_id=profile.model_id
        )
        run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = Path(output_dir).resolve() if output_dir else (
            self.config.paths.state_dir
            / "evaluations"
            / f"simple_skill_{run_stamp}_{uuid.uuid4().hex[:8]}"
        )
        checkpoint_dir = target / "checkpoints"
        self_outer = self

        class SimpleSkillM3A(m3a.M3A):
            def __init__(self, environment):
                super().__init__(
                    environment,
                    wrapper,
                    name=f"simple-skill:{profile.model_id}",
                    wait_after_action_seconds=(
                        self_outer.config.android_world.wait_after_action_seconds
                    ),
                )
                self.current_goal: str | None = None

            def reset(self, go_home_on_reset: bool = False):
                self.current_goal = None
                wrapper.reset()
                return super().reset(go_home_on_reset)

            def step(self, goal: str):
                if goal != self.current_goal:
                    wrapper.set_goal(goal)
                    self.current_goal = goal
                return compact_m3a_step_result(super().step(goal))

        environment = env_launcher.load_and_setup_env(
            console_port=self.config.android_world.console_port,
            emulator_setup=self.config.android_world.emulator_setup,
            adb_path=self.config.android_world.adb_path,
        )
        try:
            task_registry = registry.TaskRegistry()
            suite = suite_utils.create_suite(
                task_registry.get_registry(family=family),
                n_task_combinations=n_task_combinations,
                seed=seed,
                tasks=list(tasks) if tasks else None,
                env=environment,
            )
            suite.suite_family = family
            with recover_infrastructure_failures(
                suite_utils, environment, self.config.android_world
            ):
                episodes = suite_utils.run(
                    suite,
                    SimpleSkillM3A(environment),
                    checkpointer=checkpointer_lib.IncrementalCheckpointer(
                        str(checkpoint_dir)
                    ),
                    demo_mode=False,
                    return_full_episode_data=True,
                    max_n_steps_override=max_steps,
                    stop_on_task_success=self.config.android_world.stop_on_task_success,
                )
            ensure_valid_evaluation_episodes(
                episodes,
                expected_episodes=sum(len(instances) for instances in suite.values()),
            )
        finally:
            environment.close()

        traces = episodes_to_traces(episodes)
        if record_traces:
            for trace in traces:
                self.store.append_trace(trace)
        return write_evaluation_report(
            target,
            episodes,
            traces,
            metadata={
                "evaluation_mode": "single_model_with_simple_skill_retrieval",
                "model_id": profile.model_id,
                "served_model": profile.served_model,
                "evaluation_checkpoint": profile.metadata.get(
                    "evaluation_checkpoint"
                ),
                "adapter_resolution": dict(profile.metadata),
                "skill_database": str(self.store.database.resolve()),
                "available_skills": len(skills),
                "include_candidate_skills": include_candidate_skills,
                "record_traces": record_traces,
                "family": family,
                "tasks": list(tasks) if tasks else "all",
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "max_steps": max_steps,
                "stop_on_task_success": self.config.android_world.stop_on_task_success,
                "episode_storage": "compact_without_screenshots",
            },
        )


class AndroidWorldForcedSkillEvaluator:
    """运行单模型、单技能的 SKVM-style forced-skill condition。

    与关键词技能基线不同，本评测不做检索，也不做 PMT-Skill 路由；调用方指定的
    skill 会在每个动作决策中被强制注入。它专用于与同任务、同 seed 的裸模型结果
    做配对标定，不会直接写入在线 ``skill_metrics``。
    """

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def run(
        self,
        *,
        profile: ModelProfile,
        skill: SkillRecord,
        tasks: Sequence[str] | None,
        n_task_combinations: int = 1,
        seed: int = 42,
        family: str = "android_world",
        max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
        output_dir: str | Path | None = None,
    ) -> EvaluationArtifacts:
        max_steps = _evaluation_max_steps(max_steps)
        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        wrapper = ForcedSkillVLWrapper(
            OpenAICompatibleVLClient(profile), skill, model_id=profile.model_id
        )
        run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = Path(output_dir).resolve() if output_dir else (
            self.config.paths.state_dir
            / "evaluations"
            / f"forced_skill_{run_stamp}_{uuid.uuid4().hex[:8]}"
        )
        checkpoint_dir = target / "checkpoints"
        self_outer = self

        class ForcedSkillM3A(m3a.M3A):
            def __init__(self, environment):
                super().__init__(
                    environment,
                    wrapper,
                    name=f"forced-skill:{profile.model_id}:{skill.skill_id}",
                    wait_after_action_seconds=(
                        self_outer.config.android_world.wait_after_action_seconds
                    ),
                )

            def reset(self, go_home_on_reset: bool = False):
                wrapper.reset()
                return super().reset(go_home_on_reset)

            def step(self, goal: str):
                return compact_m3a_step_result(super().step(goal))

        environment = env_launcher.load_and_setup_env(
            console_port=self.config.android_world.console_port,
            emulator_setup=self.config.android_world.emulator_setup,
            adb_path=self.config.android_world.adb_path,
        )
        try:
            task_registry = registry.TaskRegistry()
            suite = suite_utils.create_suite(
                task_registry.get_registry(family=family),
                n_task_combinations=n_task_combinations,
                seed=seed,
                tasks=list(tasks) if tasks else None,
                env=environment,
            )
            suite.suite_family = family
            with recover_infrastructure_failures(
                suite_utils, environment, self.config.android_world
            ):
                episodes = suite_utils.run(
                    suite,
                    ForcedSkillM3A(environment),
                    checkpointer=checkpointer_lib.IncrementalCheckpointer(
                        str(checkpoint_dir)
                    ),
                    demo_mode=False,
                    return_full_episode_data=True,
                    max_n_steps_override=max_steps,
                    stop_on_task_success=(
                        self.config.android_world.stop_on_task_success
                    ),
                )
            ensure_valid_evaluation_episodes(
                episodes,
                expected_episodes=sum(len(instances) for instances in suite.values()),
            )
        finally:
            environment.close()

        traces = episodes_to_traces(episodes)
        return write_evaluation_report(
            target,
            episodes,
            traces,
            metadata={
                "evaluation_mode": "skvm_forced_skill_calibration",
                "calibration_protocol": "skvm-paired-android-world-v1",
                "model_id": profile.model_id,
                "served_model": profile.served_model,
                "evaluation_checkpoint": profile.metadata.get(
                    "evaluation_checkpoint"
                ),
                "adapter_resolution": dict(profile.metadata),
                "skill_database": str(self.store.database.resolve()),
                "forced_skill_id": skill.skill_id,
                "forced_skill_name": skill.name,
                "family": family,
                "tasks": list(tasks) if tasks else "all",
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "max_steps": max_steps,
                "stop_on_task_success": (
                    self.config.android_world.stop_on_task_success
                ),
                "episode_storage": "compact_without_screenshots",
            },
        )


class AndroidWorldOnlineEvaluator:
    """建立动态 M3A agent、运行 suite 并生成报告。"""

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def _planner(
        self,
        planner_model_id: str | None,
        profiles: Sequence[ModelProfile],
    ):
        primitives = load_primitives()
        if not planner_model_id:
            return KeywordSkillPlanner()
        profile = next(
            (item for item in profiles if item.model_id == planner_model_id),
            None,
        )
        if profile is None:
            profile = self.config.model(planner_model_id)
        return LLMSkillPlanner(OpenAICompatibleVLClient(profile), primitives)

    def run(
        self,
        *,
        tasks: Sequence[str] | None,
        n_task_combinations: int = 1,
        seed: int = 42,
        family: str = "android_world",
        max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
        planner_model_id: str | None = None,
        include_candidate_skills: bool = False,
        output_dir: str | Path | None = None,
        model_profiles: Sequence[ModelProfile] | None = None,
        record_traces: bool = True,
    ) -> EvaluationArtifacts:
        max_steps = _evaluation_max_steps(max_steps)
        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        raw_skills = relevant_raw_skills(self.store.list_skills(kind="raw"))
        polished = self.store.list_skills(kind="polished")
        profiles = list(model_profiles or ())
        if not profiles:
            profiles = self.store.list_model_profiles()
        if not profiles:
            profiles = [profile for profile in self.config.models if profile.enabled]
        model_pool = ModelPool(profiles)
        routed_wrapper = RoutedVLWrapper(model_pool, polished)
        planner_pipeline = PlannerPipeline(
            self._planner(planner_model_id, profiles), PrimitiveTopologyGenerator()
        )
        router = DynamicProgrammingRouter(self.config.routing, self.store)

        class DynamicRoutingM3A(m3a.M3A):
            """在每个新 episode 首步生成计划，随后复用到 M3A 各动作。"""

            def __init__(self, environment):
                super().__init__(
                    environment,
                    routed_wrapper,
                    name="PMT-Skill-v2",
                    wait_after_action_seconds=self_outer.config.android_world.wait_after_action_seconds,
                )
                self.current_goal: str | None = None
                self.latest_plan = None

            def reset(self, go_home_on_reset: bool = False):
                self.current_goal = None
                self.latest_plan = None
                model_pool.reset_counters()
                return super().reset(go_home_on_reset)

            def step(self, goal: str):
                if self.current_goal != goal:
                    decomposition, topology = planner_pipeline.plan(goal, raw_skills)
                    self.latest_plan = router.route(
                        goal,
                        topology,
                        profiles,
                        polished,
                        planner_id=decomposition.planner_id,
                        include_candidates=include_candidate_skills,
                    )
                    routed_wrapper.set_plan(self.latest_plan)
                    self.current_goal = goal
                return compact_m3a_step_result(super().step(goal))

        self_outer = self
        run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = Path(output_dir).resolve() if output_dir else (
            self.config.paths.state_dir
            / "evaluations"
            / f"pmtskill_online_{run_stamp}_{uuid.uuid4().hex[:8]}"
        )
        checkpoint_dir = target / "checkpoints"
        environment = env_launcher.load_and_setup_env(
            console_port=self.config.android_world.console_port,
            emulator_setup=self.config.android_world.emulator_setup,
            adb_path=self.config.android_world.adb_path,
        )
        try:
            task_registry = registry.TaskRegistry()
            suite = suite_utils.create_suite(
                task_registry.get_registry(family=family),
                n_task_combinations=n_task_combinations,
                seed=seed,
                tasks=list(tasks) if tasks else None,
                env=environment,
            )
            suite.suite_family = family
            agent = DynamicRoutingM3A(environment)
            with recover_infrastructure_failures(
                suite_utils, environment, self.config.android_world
            ):
                episodes = suite_utils.run(
                    suite,
                    agent,
                    checkpointer=checkpointer_lib.IncrementalCheckpointer(str(checkpoint_dir)),
                    demo_mode=False,
                    return_full_episode_data=True,
                    max_n_steps_override=max_steps,
                    stop_on_task_success=self.config.android_world.stop_on_task_success,
                )
            ensure_valid_evaluation_episodes(
                episodes,
                expected_episodes=sum(len(instances) for instances in suite.values()),
            )
        finally:
            environment.close()

        traces = episodes_to_traces(episodes)
        if record_traces:
            for trace in traces:
                self.store.append_trace(trace)
        return write_evaluation_report(
            target,
            episodes,
            traces,
            metadata={
                "family": family,
                "tasks": list(tasks) if tasks else "all",
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "planner_model_id": planner_model_id,
                "include_candidate_skills": include_candidate_skills,
                "evaluation_mode": "model_with_skill_library",
                "model_ids": [profile.model_id for profile in profiles],
                "evaluation_checkpoints": {
                    profile.model_id: profile.metadata.get("evaluation_checkpoint")
                    for profile in profiles
                },
                "adapter_resolutions": {
                    profile.model_id: dict(profile.metadata) for profile in profiles
                },
                "skill_database": str(self.store.database.resolve()),
                "record_traces": record_traces,
                "max_steps": max_steps,
                "stop_on_task_success": self.config.android_world.stop_on_task_success,
                "episode_storage": "compact_without_screenshots",
            },
        )
