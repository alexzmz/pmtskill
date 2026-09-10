"""云侧 polished skill 发现、验证、晋升和回滚。

当前实现使用透明的频繁子序列挖掘；如果未来换成 LLM/序列模型，只需替换
``SkillCompiler``，数据库和在线路由无需变化。
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from ..core.config import MaintenanceConfig
from ..core.models import ExecutionTrace, SkillRecord, SkillStatus, SkillTopology
from .store import SkillStore, wilson_lower_bound
from .identity import skill_version_id


class SkillCompiler(Protocol):
    """把高频原语序列编译成可执行技能的可替换接口。"""

    compiler_id: str

    def compile(self, primitives: tuple[str, ...], support: int) -> SkillRecord:
        ...


class TemplateSkillCompiler:
    """离线可用的确定性编译器。

    它生成清晰的执行约束和 fallback。生产环境可以换成云端 VL/LLM 编译器，
    但候选技能仍必须经过 AndroidWorld 真实 trial 才能晋升。
    """

    compiler_id = "template-compiler-v1"

    def compile(self, primitives: tuple[str, ...], support: int) -> SkillRecord:
        digest = hashlib.sha256("|".join(primitives).encode("utf-8")).hexdigest()[:16]
        topology = SkillTopology.from_sequence(primitives, topology_id=f"polished:{digest}")
        body = (
            "这是一个由高频成功轨迹固化出的组合技能。\n"
            f"覆盖原语：{', '.join(primitives)}。\n"
            "执行时应在一次模型推理中完成尽可能多的内部判断；若当前界面与"
            "预期不一致，立即停止该组合技能并按 fallback 原语拓扑逐步执行。"
        )
        return SkillRecord(
            skill_id=f"polished:{digest}:v1",
            name=f"polished_{digest}",
            description=f"由 {support} 条成功轨迹发现的高频组合。",
            kind="polished",
            status=SkillStatus.CANDIDATE,
            level=max(2, len(primitives)),
            topology=topology,
            fallback_topology=topology,
            body=body,
            metadata={"support": support, "compiler": self.compiler_id},
        )


@dataclass(slots=True)
class MaintenanceReport:
    traces_consumed: int
    subsequences_found: int
    candidates_created: list[str]
    promoted: list[str]
    rolled_back: list[str]
    model_pair_updates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "traces_consumed": self.traces_consumed,
            "subsequences_found": self.subsequences_found,
            "candidates_created": self.candidates_created,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "model_pair_updates": self.model_pair_updates,
        }


def mine_successful_subsequences(
    traces: Iterable[ExecutionTrace], minimum_length: int, maximum_length: int
) -> Counter[tuple[str, ...]]:
    """按 episode 去重统计高频连续子序列，避免长轨迹单次刷高 support。"""

    counts: Counter[tuple[str, ...]] = Counter()
    for trace in traces:
        if not trace.successful:
            continue
        sequence = trace.primitive_sequence()
        seen_in_episode: set[tuple[str, ...]] = set()
        for length in range(minimum_length, maximum_length + 1):
            for start in range(0, len(sequence) - length + 1):
                seen_in_episode.add(sequence[start : start + length])
        counts.update(seen_in_episode)
    return counts


def subsequence_supporting_tasks(
    traces: Iterable[ExecutionTrace], minimum_length: int, maximum_length: int
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """记录每个候选序列曾在哪些成功任务出现，供无 LLM 标定选择任务。"""

    tasks: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for trace in traces:
        if not trace.successful:
            continue
        sequence = trace.primitive_sequence()
        for length in range(minimum_length, maximum_length + 1):
            for start in range(0, len(sequence) - length + 1):
                tasks[sequence[start : start + length]].add(trace.task_name)
    return {
        primitives: tuple(sorted(task_names))
        for primitives, task_names in tasks.items()
    }


class SkillMaintainer:
    """backend 定时运行的技能库维护服务。"""

    def __init__(
        self,
        store: SkillStore,
        config: MaintenanceConfig,
        compiler: SkillCompiler | None = None,
    ):
        self.store = store
        self.config = config
        self.compiler = compiler or TemplateSkillCompiler()

    def discover_candidates(self, traces: list[ExecutionTrace]) -> tuple[list[str], int]:
        counts = mine_successful_subsequences(
            traces,
            self.config.minimum_subsequence_length,
            self.config.maximum_subsequence_length,
        )
        supporting_tasks = subsequence_supporting_tasks(
            traces,
            self.config.minimum_subsequence_length,
            self.config.maximum_subsequence_length,
        )
        created: list[str] = []
        frequent = [item for item in counts.items() if item[1] >= self.config.minimum_support]
        # 优先固化更长、支持度更高的路径。
        frequent.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
        for primitives, support in frequent:
            skill = self.compiler.compile(primitives, support)
            skill.metadata["calibration_tasks"] = list(
                supporting_tasks.get(primitives, ())
            )
            existing = self.store.get_skill(skill.skill_id)
            if existing:
                previous = existing.metadata.get("calibration_tasks", ())
                if isinstance(previous, str):
                    previous = (previous,)
                elif not isinstance(previous, (list, tuple, set)):
                    previous = ()
                existing.metadata["calibration_tasks"] = sorted(
                    {
                        *(str(item) for item in previous),
                        *skill.metadata["calibration_tasks"],
                    }
                )
                existing.metadata["support"] = max(
                    int(existing.metadata.get("support", 0)), support
                )
                self.store.upsert_skill(existing)
                continue
            self.store.upsert_skill(skill)
            self.store.log_maintenance_event(
                "candidate_created",
                skill.skill_id,
                {"primitives": list(primitives), "support": support},
            )
            created.append(skill.skill_id)
        return created, len(frequent)

    def promote_and_rollback(self) -> tuple[list[str], list[str]]:
        """先维护模型专属状态，再更新技能的全局候选生命周期。

        有配对标定时绝不再跨模型汇总：任一模型版本通过即可让 candidate 技能进入
        全局 active，在线路由仍只允许通过标定的模型–技能组合。没有标定数据的旧库
        继续使用在线 ``skill_metrics``，保证平滑迁移。
        """

        self.reconcile_calibrations()

        promoted: list[str] = []
        rolled_back: list[str] = []
        for skill in self.store.list_skills(kind="polished"):
            current_version = skill_version_id(skill)
            pair_statuses = [
                row
                for row in self.store.list_skill_model_statuses(skill.skill_id)
                if row["skill_version_id"] == current_version
            ]
            if pair_statuses:
                statuses = {str(row["status"]) for row in pair_statuses}
                if (
                    skill.status == SkillStatus.CANDIDATE
                    and SkillStatus.ACTIVE.value in statuses
                ):
                    self.store.set_skill_status(skill.skill_id, SkillStatus.ACTIVE)
                    self.store.log_maintenance_event(
                        "candidate_promoted_from_model_calibration",
                        skill.skill_id,
                        {"model_pair_statuses": pair_statuses},
                    )
                    promoted.append(skill.skill_id)
                elif (
                    skill.status == SkillStatus.ACTIVE
                    and statuses == {SkillStatus.DEPRECATED.value}
                ):
                    self.store.set_skill_status(
                        skill.skill_id, SkillStatus.DEPRECATED
                    )
                    self.store.log_maintenance_event(
                        "skill_rolled_back_from_model_calibration",
                        skill.skill_id,
                        {"model_pair_statuses": pair_statuses},
                    )
                    rolled_back.append(skill.skill_id)
                continue

            metrics = self.store.skill_metrics(skill.skill_id)
            trials = int(metrics["trials"])
            successes = int(metrics["successes"])
            rate = float(metrics["success_rate"])
            lower = wilson_lower_bound(successes, trials)
            if (
                skill.status == SkillStatus.CANDIDATE
                and trials >= self.config.minimum_candidate_trials
                and rate >= self.config.promotion_success_rate
                and lower >= max(0.0, self.config.promotion_success_rate - 0.20)
            ):
                self.store.set_skill_status(skill.skill_id, SkillStatus.ACTIVE)
                self.store.log_maintenance_event(
                    "candidate_promoted", skill.skill_id, {**metrics, "wilson_lower": lower}
                )
                promoted.append(skill.skill_id)
            elif (
                skill.status == SkillStatus.ACTIVE
                and trials >= self.config.minimum_candidate_trials
                and rate < self.config.rollback_success_rate
            ):
                self.store.set_skill_status(skill.skill_id, SkillStatus.DEPRECATED)
                self.store.log_maintenance_event(
                    "skill_rolled_back", skill.skill_id, metrics
                )
                rolled_back.append(skill.skill_id)
        return promoted, rolled_back

    def reconcile_calibrations(self) -> list[dict[str, Any]]:
        """按模型版本将配对标定指标转换为独立技能状态。

        晋升同时要求绝对 SR、Wilson 下界和相对裸模型 uplift。候选组合出现明显负
        增益或低于回滚 SR 时标记 deprecated；样本不足保持 candidate/unknown。
        """

        existing = {
            (str(row["skill_version_id"]), str(row["model_variant_id"])): str(
                row["status"]
            )
            for row in self.store.list_skill_model_statuses()
        }
        updates: list[dict[str, Any]] = []
        for skill in self.store.list_skills(kind="polished"):
            for metrics in self.store.calibration_metrics_by_model(skill.skill_id):
                trials = int(metrics["valid_trials"])
                if trials <= 0:
                    continue
                current = existing.get(
                    (
                        str(metrics["skill_version_id"]),
                        str(metrics["model_variant_id"]),
                    )
                )
                rate = float(metrics["skill_success_rate"])
                uplift = float(metrics["success_rate_uplift"])
                lower = float(metrics["skill_success_wilson_lower"])
                enough = trials >= self.config.minimum_candidate_trials
                promotes = (
                    enough
                    and rate >= self.config.promotion_success_rate
                    and uplift >= self.config.baseline_margin
                    and lower
                    >= max(0.0, self.config.promotion_success_rate - 0.20)
                )
                harmful = enough and (
                    rate < self.config.rollback_success_rate
                    or uplift < -self.config.baseline_margin
                )
                if promotes:
                    target = SkillStatus.ACTIVE
                elif harmful:
                    target = SkillStatus.DEPRECATED
                else:
                    target = SkillStatus.CANDIDATE
                if current == target.value:
                    continue
                self.store.set_skill_model_status(
                    skill,
                    model_id=str(metrics["model_id"]),
                    model_variant_id_value=str(metrics["model_variant_id"]),
                    status=target,
                    reason=metrics,
                )
                update = {
                    "skill_id": skill.skill_id,
                    "skill_version_id": metrics["skill_version_id"],
                    "model_id": metrics["model_id"],
                    "model_variant_id": metrics["model_variant_id"],
                    "previous_status": current or "unknown",
                    "status": target.value,
                    "valid_trials": trials,
                    "skill_success_rate": rate,
                    "baseline_success_rate": metrics["baseline_success_rate"],
                    "success_rate_uplift": uplift,
                    "skill_success_wilson_lower": lower,
                }
                self.store.log_maintenance_event(
                    "skill_model_calibration_status", skill.skill_id, update
                )
                updates.append(update)
        return updates

    def run_cycle(self) -> MaintenanceReport:
        """消费尚未处理的设备轨迹并完成一次维护周期。"""

        traces = self.store.list_traces(processed=False)
        created, subsequences_found = self.discover_candidates(traces)
        model_pair_updates = self.reconcile_calibrations()
        promoted, rolled_back = self.promote_and_rollback()
        self.store.mark_traces_processed([trace.trace_id for trace in traces])
        report = MaintenanceReport(
            traces_consumed=len(traces),
            subsequences_found=subsequences_found,
            candidates_created=created,
            promoted=promoted,
            rolled_back=rolled_back,
            model_pair_updates=model_pair_updates,
        )
        self.store.log_maintenance_event("maintenance_cycle", None, report.to_dict())
        return report
