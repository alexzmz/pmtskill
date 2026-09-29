"""面向技能库优化的 AndroidWorld Teacher 标准轨迹采集器。

旧 ``collect`` 服务于监督蒸馏数据生产；本模块服务于技能发现，二者刻意分开，避免
改变既有训练实验。这里的 task/database-skill/skill-cluster 三种入口在创建 source binding 后，共用
完全相同的 planner、M3A wrapper、episode converter、报告和 SQLite 写入路径。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.config import ProjectConfig
from ..core.io import load_primitives, write_json_atomic
from ..core.models import ModelProfile, SkillRecord
from ..evaluation.android_world import episodes_to_traces
from ..evaluation.recovery import (
    ensure_valid_evaluation_episodes,
    recover_infrastructure_failures,
)
from ..evaluation.reporter import EvaluationArtifacts, write_evaluation_report
from ..inference.vlm import OpenAICompatibleVLClient
from ..skills.store import SkillStore, classify_trajectory_quality
from .collector import (
    bootstrap_android_world,
    enforce_episode_step_limit,
    resolve_episode_step_limit,
)
from .primitive_trajectory import (
    LLMPrimitivePlanGenerator,
    PRIMITIVE_TRACE_SCHEMA,
    PrimitivePlanGenerator,
    SkillTaskSelection,
    TeacherPrimitiveVLWrapper,
)


@dataclass(frozen=True, slots=True)
class SkillTaskBinding:
    """一个 raw skill 与可执行 AndroidWorld 任务之间的可审计绑定。"""

    skill_id: str
    tasks: tuple[str, ...]
    source: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "tasks": list(self.tasks),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(slots=True)
class OptimizationTraceCollectionOptions:
    """一次标准轨迹采集的全部选项。"""

    source_kind: str
    output_dir: Path
    tasks: tuple[str, ...] | None = None
    source_skills: tuple[SkillRecord, ...] = ()
    skill_task_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    family: str = "android_world"
    combinations: int = 1
    seed: int = 42
    max_steps: int | None = None
    replan_every_steps: int = 1
    auto_bind_skills: bool = True
    maximum_tasks_per_skill: int = 3
    record_traces: bool = True
    # 外部技能簇的 root/namespace/import manifest 等审计信息；task/DB 模式可为空。
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationTraceCollectionResult:
    """CLI 和日志系统可直接序列化的采集结果。"""

    artifacts: EvaluationArtifacts
    source_kind: str
    bindings: tuple[SkillTaskBinding, ...]
    skipped_skills: dict[str, str]
    traces_inserted: int
    checkpoint_dir: Path
    bindings_json: Path
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema": PRIMITIVE_TRACE_SCHEMA,
            "source_kind": self.source_kind,
            "episodes": self.artifacts.summary.get("episodes_evaluated", 0),
            "successes": self.artifacts.summary.get("successes", 0),
            "success_rate": self.artifacts.summary.get("success_rate_micro", 0.0),
            "traces_inserted": self.traces_inserted,
            "bindings": [item.to_dict() for item in self.bindings],
            "skipped_skills": self.skipped_skills,
            "output_dir": str(self.artifacts.output_dir),
            "summary_json": str(self.artifacts.summary_json),
            "report_markdown": str(self.artifacts.report_markdown),
            "traces_jsonl": str(self.artifacts.traces_jsonl),
            "checkpoint_dir": str(self.checkpoint_dir),
            "bindings_json": str(self.bindings_json),
            "source_metadata": dict(self.source_metadata),
        }


def _metadata_tasks(skill: SkillRecord) -> tuple[str, ...]:
    """读取显式维护的 task 绑定；trajectory_tasks 优先于 calibration_tasks。"""

    for key in ("trajectory_tasks", "calibration_tasks"):
        value = skill.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return (value.strip(),)
        if isinstance(value, (list, tuple, set)):
            tasks = tuple(
                dict.fromkeys(str(item).strip() for item in value if str(item).strip())
            )
            if tasks:
                return tasks
    return ()


def resolve_skill_task_bindings(
    skills: Sequence[SkillRecord],
    available_tasks: Sequence[str],
    planner: PrimitivePlanGenerator,
    *,
    explicit_map: Mapping[str, tuple[str, ...]] | None = None,
    shared_tasks: Sequence[str] | None = None,
    auto_bind: bool = True,
    maximum_tasks_per_skill: int = 3,
) -> tuple[tuple[SkillTaskBinding, ...], dict[str, str]]:
    """在启动 emulator 前解析 raw skill→AndroidWorld task 绑定。

    优先级为：``--skill-task-map`` → ``--tasks`` 共享任务 → skill metadata →
    Teacher 自动语义绑定。所有结果都必须属于当前 AndroidWorld registry；不存在
    evaluator 的自由文本任务不会被伪装成有 ground truth 的有效轨迹。
    """

    known = set(available_tasks)
    explicit = explicit_map or {}
    shared = tuple(shared_tasks or ())
    bindings: list[SkillTaskBinding] = []
    skipped: dict[str, str] = {}
    for skill in skills:
        candidates: tuple[str, ...]
        source: str
        reason = ""
        if skill.skill_id in explicit:
            candidates = tuple(explicit[skill.skill_id])
            source = "skill_task_map"
        elif shared:
            candidates = shared
            source = "shared_tasks"
        else:
            candidates = _metadata_tasks(skill)
            source = "skill_metadata"
            if not candidates and auto_bind:
                selection: SkillTaskSelection = planner.bind_skill_to_tasks(
                    skill,
                    available_tasks,
                    maximum_tasks=maximum_tasks_per_skill,
                )
                candidates = selection.tasks
                source = "teacher_auto_binding"
                reason = selection.reason
        unknown = sorted({item for item in candidates if item not in known})
        if unknown:
            raise ValueError(
                f"技能 {skill.skill_id} 绑定了未知 AndroidWorld tasks: {unknown}"
            )
        tasks = tuple(dict.fromkeys(candidates))[: max(1, maximum_tasks_per_skill)]
        if not tasks:
            skipped[skill.skill_id] = (
                reason
                or "没有可验证的 AndroidWorld task；请提供 --skill-task-map 或 --tasks"
            )
            continue
        bindings.append(SkillTaskBinding(skill.skill_id, tasks, source, reason))
    return tuple(bindings), skipped


def _safe_component(value: str) -> str:
    """把任意 skill ID 转成不会逃逸输出目录的短文件夹名。"""

    compact = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{(compact or 'skill')[:80]}_{digest}"


def _compact_step_source_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """去掉批量 ID 清单，避免同一大对象被复制到每个 episode 的每一步。"""

    compact: dict[str, Any] = {}
    if value.get("cli_source") is not None:
        compact["cli_source"] = value.get("cli_source")
    cluster = value.get("skill_cluster")
    if isinstance(cluster, Mapping):
        compact["skill_cluster"] = {
            key: cluster.get(key)
            for key in (
                "root",
                "namespace",
                "scanned",
                "resources",
                "skipped_resources",
                "write_database",
            )
            if cluster.get(key) is not None
        }
    return compact


class AndroidWorldOptimizationTraceCollector:
    """运行受约束 Teacher，并把三类来源写成统一标准轨迹。"""

    def __init__(
        self,
        config: ProjectConfig,
        store: SkillStore,
        *,
        planner: PrimitivePlanGenerator | None = None,
    ):
        self.config = config
        self.store = store
        self._planner = planner

    def _run_suite(
        self,
        *,
        suite_utils: Any,
        checkpointer_lib: Any,
        registry_module: Any,
        m3a_module: Any,
        environment: Any,
        profile: ModelProfile,
        client: Any,
        planner: PrimitivePlanGenerator,
        tasks: Sequence[str] | None,
        source_kind: str,
        source_skill: SkillRecord | None,
        binding_source: str,
        options: OptimizationTraceCollectionOptions,
        checkpoint_dir: Path,
        step_limit: int,
    ) -> list[dict[str, Any]]:
        """运行一个同源 suite；skill 模式每个 skill 独占 checkpoint 子目录。"""

        wrapper = TeacherPrimitiveVLWrapper(
            client,
            planner,
            replan_every_steps=max(0, options.replan_every_steps),
            source_kind=source_kind,
            source_skill=source_skill,
            collection_context={
                "family": options.family,
                "suite_seed": options.seed,
                "task_combinations": options.combinations,
                "bound_tasks": list(tasks) if tasks else "all",
                "binding_source": binding_source,
                "input_mode": options.source_kind,
                "source_metadata": _compact_step_source_metadata(
                    options.source_metadata
                ),
            },
        )

        class GuidedCollectionM3A(m3a_module.M3A):
            """把 M3A 每个真实 step 的解析结果反馈给动态规划器。"""

            def __init__(self):
                super().__init__(
                    environment,
                    wrapper,
                    name=f"primitive-teacher:{profile.model_id}",
                    wait_after_action_seconds=(
                        self_outer.config.android_world.wait_after_action_seconds
                    ),
                )

            def reset(self, go_home_on_reset: bool = False):
                wrapper.reset()
                return super().reset(go_home_on_reset)

            def step(self, goal: str):
                wrapper.set_episode(
                    goal,
                    source_kind=source_kind,
                    source_skill=source_skill,
                )
                result = super().step(goal)
                wrapper.observe_step(getattr(result, "data", None))
                return result

        self_outer = self
        task_registry = registry_module.TaskRegistry()
        suite = suite_utils.create_suite(
            task_registry.get_registry(family=options.family),
            n_task_combinations=options.combinations,
            seed=options.seed,
            tasks=list(tasks) if tasks else None,
            env=environment,
        )
        suite.suite_family = options.family
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with recover_infrastructure_failures(
            suite_utils, environment, self.config.android_world
        ):
            with enforce_episode_step_limit(suite_utils, step_limit):
                episodes = suite_utils.run(
                    suite,
                    GuidedCollectionM3A(),
                    checkpointer=checkpointer_lib.IncrementalCheckpointer(
                        str(checkpoint_dir)
                    ),
                    demo_mode=False,
                    return_full_episode_data=True,
                    max_n_steps_override=step_limit,
                    stop_on_task_success=self.config.android_world.stop_on_task_success,
                )
        ensure_valid_evaluation_episodes(
            episodes,
            expected_episodes=sum(len(instances) for instances in suite.values()),
        )
        return list(episodes)

    def run(
        self,
        profile: ModelProfile,
        options: OptimizationTraceCollectionOptions,
        *,
        optimized_skills: Sequence[SkillRecord] = (),
    ) -> OptimizationTraceCollectionResult:
        """完成采集、标准化、SQLite 写入和人机可读报告生成。"""

        allowed_source_kinds = {"task", "raw_skill", "database_skill", "skill_cluster"}
        if options.source_kind not in allowed_source_kinds:
            raise ValueError(
                "source_kind 必须是 task、database_skill、skill_cluster 或兼容 raw_skill"
            )
        if options.combinations <= 0:
            raise ValueError("combinations 必须是正整数")
        if options.maximum_tasks_per_skill <= 0:
            raise ValueError("maximum_tasks_per_skill 必须是正整数")
        is_skill_source = options.source_kind != "task"
        if is_skill_source and not options.source_skills:
            raise ValueError("技能来源模式至少需要一个 source skill")

        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        client = OpenAICompatibleVLClient(profile)
        planner = self._planner or LLMPrimitivePlanGenerator(
            client,
            load_primitives(),
            optimized_skills,
            maximum_plan_units=resolve_episode_step_limit(
                self.config.android_world.max_steps, options.max_steps
            ),
        )
        available = sorted(registry.TaskRegistry().get_registry(family=options.family))
        if not available:
            raise ValueError(f"AndroidWorld family 没有任务: {options.family}")
        explicit_tasks = tuple(options.tasks or ())
        unknown_tasks = sorted({item for item in explicit_tasks if item not in available})
        if unknown_tasks:
            raise ValueError(f"未知 AndroidWorld tasks: {unknown_tasks}")

        bindings: tuple[SkillTaskBinding, ...] = ()
        skipped: dict[str, str] = {}
        if is_skill_source:
            bindings, skipped = resolve_skill_task_bindings(
                options.source_skills,
                available,
                planner,
                explicit_map=options.skill_task_map,
                shared_tasks=explicit_tasks,
                auto_bind=options.auto_bind_skills,
                maximum_tasks_per_skill=options.maximum_tasks_per_skill,
            )
            if not bindings:
                details = "; ".join(f"{key}: {value}" for key, value in skipped.items())
                raise ValueError(f"没有 raw skill 能绑定到可验证任务。{details}")

        target = options.output_dir.expanduser().resolve()
        checkpoint_root = target / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        bindings_path = target / "source_bindings.json"
        write_json_atomic(
            bindings_path,
            {
                "trace_schema": PRIMITIVE_TRACE_SCHEMA,
                "source_kind": options.source_kind,
                "bindings": [item.to_dict() for item in bindings],
                "skipped_skills": skipped,
                "source_metadata": dict(options.source_metadata),
            },
        )

        by_id = {skill.skill_id: skill for skill in options.source_skills}
        step_limit = resolve_episode_step_limit(
            self.config.android_world.max_steps, options.max_steps
        )
        all_episodes: list[dict[str, Any]] = []
        environment = env_launcher.load_and_setup_env(
            console_port=self.config.android_world.console_port,
            emulator_setup=self.config.android_world.emulator_setup,
            adb_path=self.config.android_world.adb_path,
        )
        try:
            if options.source_kind == "task":
                all_episodes.extend(
                    self._run_suite(
                        suite_utils=suite_utils,
                        checkpointer_lib=checkpointer_lib,
                        registry_module=registry,
                        m3a_module=m3a,
                        environment=environment,
                        profile=profile,
                        client=client,
                        planner=planner,
                        tasks=explicit_tasks or None,
                        source_kind="task",
                        source_skill=None,
                        binding_source="direct_task_input",
                        options=options,
                        checkpoint_dir=checkpoint_root / "tasks",
                        step_limit=step_limit,
                    )
                )
            else:
                for binding in bindings:
                    skill = by_id[binding.skill_id]
                    all_episodes.extend(
                        self._run_suite(
                            suite_utils=suite_utils,
                            checkpointer_lib=checkpointer_lib,
                            registry_module=registry,
                            m3a_module=m3a,
                            environment=environment,
                            profile=profile,
                            client=client,
                            planner=planner,
                            tasks=binding.tasks,
                            source_kind=options.source_kind,
                            source_skill=skill,
                            binding_source=binding.source,
                            options=options,
                            checkpoint_dir=(
                                checkpoint_root / "skills" / _safe_component(skill.skill_id)
                            ),
                            step_limit=step_limit,
                        )
                    )
        finally:
            environment.close()

        traces = episodes_to_traces(all_episodes)
        for trace in traces:
            # 即使本轮 --no-record-traces，文件报告也带相同的 deterministic 分级。
            trace.metadata["trajectory_quality"] = classify_trajectory_quality(trace)
        traces_inserted = 0
        if options.record_traces:
            for trace in traces:
                # 标准 Teacher 数据用于技能发现，不是在线自然流量，不能直接累计
                # skill_metrics；候选仍需 calibrate-skills 的配对实测才能晋升。
                traces_inserted += int(
                    self.store.append_trace(trace, update_skill_metrics=False)
                )
        alignment_values = [
            float(trace.metadata["action_primitive_alignment_rate"])
            for trace in traces
            if trace.metadata.get("action_primitive_alignment_rate") is not None
        ]
        initial_lengths = [
            len(trace.metadata.get("initial_plan", ()))
            for trace in traces
            if isinstance(trace.metadata.get("initial_plan"), (list, tuple))
        ]
        trajectory_quality = {
            "traces": len(traces),
            "successful": sum(trace.successful for trace in traces),
            "failed": sum(not trace.successful for trace in traces),
            "library_statuses": {
                status: sum(
                    trace.metadata.get("trajectory_quality", {}).get("status") == status
                    for trace in traces
                )
                for status in ("excellent", "candidate", "failed", "rejected")
            },
            "with_events": sum(bool(trace.events) for trace in traces),
            "with_initial_plan": sum(trace.plan is not None for trace in traces),
            "average_initial_action_units": (
                sum(initial_lengths) / len(initial_lengths) if initial_lengths else 0.0
            ),
            "total_plan_revisions": sum(
                max(0, len(trace.metadata.get("plan_revisions", ())) - 1)
                for trace in traces
                if isinstance(trace.metadata.get("plan_revisions"), (list, tuple))
            ),
            "average_action_primitive_alignment": (
                sum(alignment_values) / len(alignment_values)
                if alignment_values
                else None
            ),
        }
        artifacts = write_evaluation_report(
            target,
            all_episodes,
            traces,
            metadata={
                "evaluation_mode": "teacher_primitive_trace_collection",
                "trace_schema": PRIMITIVE_TRACE_SCHEMA,
                "metric_scope": "skill_discovery",
                "source_kind": options.source_kind,
                "teacher_model_id": profile.model_id,
                "served_model": profile.served_model,
                "tasks": list(explicit_tasks) if explicit_tasks else "bound_or_all",
                "bindings": [item.to_dict() for item in bindings],
                "skipped_skills": skipped,
                "combinations": options.combinations,
                "seed": options.seed,
                "max_steps": step_limit,
                "replan_every_steps": options.replan_every_steps,
                "record_traces": options.record_traces,
                "trajectory_quality": trajectory_quality,
                "skill_database": str(self.store.database.resolve()),
                "source_metadata": dict(options.source_metadata),
                "created_at": dt.datetime.now().astimezone().isoformat(),
                "run_id": uuid.uuid4().hex,
            },
        )
        with artifacts.report_markdown.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## 标准原语轨迹\n\n"
                f"- Schema：`{PRIMITIVE_TRACE_SCHEMA}`\n"
                f"- 来源模式：`{options.source_kind}`\n"
                f"- 初始规划后每 {options.replan_every_steps} 个环境 step 检查剩余路径；"
                "0 表示只在路径耗尽时重规划。\n"
                f"- 写入技能库：{'是' if options.record_traces else '否'}\n"
                f"- 含有效原语事件：{trajectory_quality['with_events']}/"
                f"{trajectory_quality['traces']}\n"
                f"- 平均初始 action units："
                f"{trajectory_quality['average_initial_action_units']:.2f}\n"
                f"- 动态计划修订总数：{trajectory_quality['total_plan_revisions']}\n"
                "- AndroidWorld evaluator 是最终成功真值；失败轨迹会保留，但不会直接"
                "生成 polished candidate。\n"
            )
        return OptimizationTraceCollectionResult(
            artifacts=artifacts,
            source_kind=options.source_kind,
            bindings=bindings,
            skipped_skills=skipped,
            traces_inserted=traces_inserted,
            checkpoint_dir=checkpoint_root,
            bindings_json=bindings_path,
            source_metadata=dict(options.source_metadata),
        )


def default_optimization_trace_output(config: ProjectConfig) -> Path:
    """返回带时间戳和随机后缀的默认输出目录，避免不同采集互相覆盖。"""

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return (
        config.paths.state_dir
        / "optimization_trajectories"
        / f"{stamp}_{uuid.uuid4().hex[:8]}"
    )
