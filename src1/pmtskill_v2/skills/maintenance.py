"""云侧 polished skill 发现、验证、晋升和回滚。

当前实现使用透明的频繁子序列挖掘；如果未来换成 LLM/序列模型，只需替换
``SkillCompiler``，数据库和在线路由无需变化。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

from ..core.config import MaintenanceConfig
from ..core.models import ExecutionTrace, SkillRecord, SkillStatus, SkillTopology
from .store import SkillStore, wilson_lower_bound


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

    def to_dict(self) -> dict[str, object]:
        return {
            "traces_consumed": self.traces_consumed,
            "subsequences_found": self.subsequences_found,
            "candidates_created": self.candidates_created,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
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
        created: list[str] = []
        frequent = [item for item in counts.items() if item[1] >= self.config.minimum_support]
        # 优先固化更长、支持度更高的路径。
        frequent.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
        for primitives, support in frequent:
            skill = self.compiler.compile(primitives, support)
            if self.store.get_skill(skill.skill_id):
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
        """依据真实 trial 的置信下界晋升，依据近期总成功率回滚。"""

        promoted: list[str] = []
        rolled_back: list[str] = []
        for skill in self.store.list_skills(kind="polished"):
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

    def run_cycle(self) -> MaintenanceReport:
        """消费尚未处理的设备轨迹并完成一次维护周期。"""

        traces = self.store.list_traces(processed=False)
        created, subsequences_found = self.discover_candidates(traces)
        promoted, rolled_back = self.promote_and_rollback()
        self.store.mark_traces_processed([trace.trace_id for trace in traces])
        report = MaintenanceReport(
            traces_consumed=len(traces),
            subsequences_found=subsequences_found,
            candidates_created=created,
            promoted=promoted,
            rolled_back=rolled_back,
        )
        self.store.log_maintenance_event("maintenance_cycle", None, report.to_dict())
        return report
