"""动态模型/技能路由版 AndroidWorld 在线评测。"""

from __future__ import annotations

import datetime as dt
import random
import uuid
from pathlib import Path
from typing import Any, Sequence

from ..core.config import ProjectConfig
from ..core.io import load_primitives
from ..core.models import ExecutionTrace, ModelProfile, TraceEvent
from ..inference.model_pool import ModelPool
from ..inference.vlm import OpenAICompatibleVLClient
from ..offline.collector import bootstrap_android_world
from ..online.executor import RoutedVLWrapper
from ..online.planner import (
    KeywordSkillPlanner,
    LLMSkillPlanner,
    PlannerPipeline,
    PrimitiveTopologyGenerator,
)
from ..online.router import DynamicProgrammingRouter
from ..skills.importer import relevant_raw_skills
from ..skills.store import SkillStore
from .reporter import (
    EvaluationArtifacts,
    successful_episode_value,
    write_evaluation_report,
)


def _extract_route_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        value = raw.get("_pmtskill", {})
        return value if isinstance(value, dict) else {}
    return {}


def episodes_to_traces(episodes: Sequence[dict[str, Any]]) -> list[ExecutionTrace]:
    """把 M3A episode 转成 backend 接收的轻量轨迹并进行保守信用分配。"""

    traces: list[ExecutionTrace] = []
    for episode in episodes:
        successful = successful_episode_value(episode.get("is_successful", False))
        episode_data = episode.get("episode_data", {}) or {}
        raw_responses = episode_data.get("action_raw_response", [])
        actions = episode_data.get("action_output", [])
        parsed_actions = episode_data.get("action_output_json", [])
        events: list[TraceEvent] = []
        for index, raw in enumerate(raw_responses):
            route = _extract_route_metadata(raw)
            if not route:
                continue
            action_parsed = index < len(parsed_actions) and parsed_actions[index] is not None
            # 组合技能只有在动作可解析且 episode 最终成功时计为成功，避免把局部
            # 看似正确、实际使任务失败的动作错误地用于技能晋升。
            event_success = bool(action_parsed and successful)
            events.append(
                TraceEvent(
                    index=len(events),
                    model_id=str(route.get("model_id", "unknown")),
                    skill_id=route.get("skill_id"),
                    primitive_ids=tuple(route.get("primitive_ids", ())),
                    success=event_success,
                    latency_ms=float(route.get("latency_ms", 0.0) or 0.0),
                    action=str(actions[index]) if index < len(actions) else None,
                    metadata={"credit_assignment": "parsed_action_and_episode_success"},
                )
            )
        traces.append(
            ExecutionTrace.new(
                goal=str(episode.get("goal", "")),
                task_name=str(episode.get("task_template", "unknown")),
                successful=successful,
                events=events,
                reward=float(successful),
                duration_ms=float(episode.get("run_time", 0.0) or 0.0) * 1000,
                metadata={"source": "android_world_m3a"},
            )
        )
    return traces


def sample_android_world_tasks(
    config: ProjectConfig,
    *,
    tasks: Sequence[str] | None,
    task_count: int,
    seed: int,
    family: str = "android_world",
) -> list[str]:
    """解析一组固定评测任务，供所有训练阶段公平复用。

    显式传入任务时保留用户顺序且仅去重；未传入时从 family 的完整注册表中
    按 seed 抽样。这里只读取任务注册表，不会连接 emulator。
    """

    if tasks:
        return list(dict.fromkeys(str(item) for item in tasks))
    if task_count <= 0:
        raise ValueError("评测 task_count 必须是正整数")
    bootstrap_android_world(config.paths.android_world_root)
    from android_world import registry

    available = sorted(registry.TaskRegistry().get_registry(family=family))
    if not available:
        raise ValueError(f"AndroidWorld family 没有可评测任务: {family}")
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
        output_dir: str | Path | None = None,
    ) -> EvaluationArtifacts:
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
            agent = m3a.M3A(
                environment,
                model,
                name=f"standalone:{profile.model_id}",
                wait_after_action_seconds=(
                    self.config.android_world.wait_after_action_seconds
                ),
            )
            episodes = suite_utils.run(
                suite,
                agent,
                checkpointer=checkpointer_lib.IncrementalCheckpointer(
                    str(checkpoint_dir)
                ),
                demo_mode=False,
                return_full_episode_data=True,
                max_n_steps_override=self.config.android_world.max_steps,
                stop_on_task_success=self.config.android_world.stop_on_task_success,
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
                "family": family,
                "tasks": list(tasks) if tasks else "all",
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "max_steps": self.config.android_world.max_steps,
                "stop_on_task_success": self.config.android_world.stop_on_task_success,
            },
        )


class AndroidWorldOnlineEvaluator:
    """建立动态 M3A agent、运行 suite 并生成报告。"""

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def _planner(self, planner_model_id: str | None):
        primitives = load_primitives()
        if not planner_model_id:
            return KeywordSkillPlanner()
        profile = self.config.model(planner_model_id)
        return LLMSkillPlanner(OpenAICompatibleVLClient(profile), primitives)

    def run(
        self,
        *,
        tasks: Sequence[str] | None,
        n_task_combinations: int = 1,
        seed: int = 42,
        family: str = "android_world",
        planner_model_id: str | None = None,
        include_candidate_skills: bool = False,
        output_dir: str | Path | None = None,
        model_profiles: Sequence[ModelProfile] | None = None,
        record_traces: bool = True,
    ) -> EvaluationArtifacts:
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
            self._planner(planner_model_id), PrimitiveTopologyGenerator()
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
                return super().step(goal)

        self_outer = self
        run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        target = Path(output_dir).resolve() if output_dir else (
            self.config.paths.state_dir / "evaluations" / f"run_{run_stamp}_{uuid.uuid4().hex[:8]}"
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
            episodes = suite_utils.run(
                suite,
                agent,
                checkpointer=checkpointer_lib.IncrementalCheckpointer(str(checkpoint_dir)),
                demo_mode=False,
                return_full_episode_data=True,
                max_n_steps_override=self.config.android_world.max_steps,
                stop_on_task_success=self.config.android_world.stop_on_task_success,
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
                "record_traces": record_traces,
                "max_steps": self.config.android_world.max_steps,
                "stop_on_task_success": self.config.android_world.stop_on_task_success,
            },
        )
