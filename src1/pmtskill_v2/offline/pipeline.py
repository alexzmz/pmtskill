"""离线蒸馏阶段的编排门面。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.config import ProjectConfig
from .collector import CollectionResult, collect_teacher_trajectories
from .dataset import AndroidWorldDistillationDatasetBuilder, DatasetBuildResult
from .trainer import AdapterJob, MSSwiftLoraTrainer, default_training_job


class OfflineDistillationPipeline:
    """把采集、转换和训练组织成可单独运行的三个阶段。"""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def collect(
        self,
        tasks: Sequence[str] | None,
        *,
        combinations: int = 1,
        seed: int = 42,
    ) -> CollectionResult:
        return collect_teacher_trajectories(
            self.config,
            tasks=tasks,
            n_task_combinations=combinations,
            seed=seed,
        )

    def build_dataset(
        self,
        trajectory_root: Path | None = None,
        *,
        successful_only: bool | None = None,
    ) -> DatasetBuildResult:
        builder = AndroidWorldDistillationDatasetBuilder(
            self.config.offline.dataset_dir,
            successful_only=(
                self.config.offline.successful_only
                if successful_only is None
                else successful_only
            ),
            validation_ratio=self.config.offline.validation_ratio,
        )
        return builder.build(trajectory_root or self.config.offline.trajectory_dir)

    def train(self, job: AdapterJob | None = None, *, dry_run: bool = False) -> int:
        trainer = MSSwiftLoraTrainer(self.config)
        return trainer.run(job or default_training_job(self.config), dry_run=dry_run)
