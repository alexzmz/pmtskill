"""SKVM bench condition 在 AndroidWorld 上的适配接口。

SKVM 原生 ``no-skill`` 与 ``original/custom-skill`` condition 面向文件型 agent
任务；AndroidWorld 使用 Python M3A 和 emulator，不能直接把 TypeScript runner 嵌入。
本模块保留相同的 condition 边界，让标定编排、假 runner 单测和未来其他 GUI harness
共享接口，真实判分仍完全交给 AndroidWorld task evaluator。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..core.config import ProjectConfig
from ..core.models import ModelProfile, SkillRecord
from ..evaluation.android_world import (
    AndroidWorldForcedSkillEvaluator,
    AndroidWorldStandaloneEvaluator,
)
from ..skills.store import SkillStore


@dataclass(slots=True)
class SkillConditionResult:
    """一次 condition 的逐 episode 结果与标准 AndroidWorld 汇总。"""

    condition: str
    output_dir: Path
    summary: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]
    model_id: str
    skill_id: str | None = None


class SkillCalibrationConditionRunner(Protocol):
    """可替换的条件执行器，接口语义对应 SKVM bench conditions。"""

    protocol_id: str

    def run_no_skill(
        self,
        *,
        profile: ModelProfile,
        tasks: Sequence[str] | None,
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        output_dir: Path,
    ) -> SkillConditionResult:
        ...

    def run_forced_skill(
        self,
        *,
        profile: ModelProfile,
        skill: SkillRecord,
        tasks: Sequence[str] | None,
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        output_dir: Path,
    ) -> SkillConditionResult:
        ...


class AndroidWorldSkillConditionRunner:
    """使用 AndroidWorld 原生 evaluator 实现 no-skill/forced-skill 配对。"""

    protocol_id = "skvm-paired-android-world-v1"

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store

    def run_no_skill(
        self,
        *,
        profile: ModelProfile,
        tasks: Sequence[str] | None,
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        output_dir: Path,
    ) -> SkillConditionResult:
        artifacts = AndroidWorldStandaloneEvaluator(self.config).run(
            profile=profile,
            tasks=tasks,
            n_task_combinations=combinations,
            seed=seed,
            family=family,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        return SkillConditionResult(
            condition="no-skill",
            output_dir=artifacts.output_dir,
            summary=artifacts.summary,
            episodes=artifacts.episodes,
            model_id=profile.model_id,
        )

    def run_forced_skill(
        self,
        *,
        profile: ModelProfile,
        skill: SkillRecord,
        tasks: Sequence[str] | None,
        combinations: int,
        seed: int,
        family: str,
        max_steps: int,
        output_dir: Path,
    ) -> SkillConditionResult:
        artifacts = AndroidWorldForcedSkillEvaluator(self.config, self.store).run(
            profile=profile,
            skill=skill,
            tasks=tasks,
            n_task_combinations=combinations,
            seed=seed,
            family=family,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        return SkillConditionResult(
            condition="forced-skill",
            output_dir=artifacts.output_dir,
            summary=artifacts.summary,
            episodes=artifacts.episodes,
            model_id=profile.model_id,
            skill_id=skill.skill_id,
        )
