"""可替换训练算法与 ms-swift LoRA 实现。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from ..core.config import ProjectConfig


@dataclass(slots=True)
class AdapterJob:
    """一个定向能力 adapter 的训练任务。

    可先用 ``filter_dataset_by_primitives`` 从主数据集筛出 grounding、planning、
    action 等分支，再并行训练多个 LoRA。默认 job 则训练一个全能力 LoRA。
    """

    name: str
    train_dataset: Path
    validation_dataset: Path | None
    output_dir: Path
    primitive_filter: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()


class TrainingAlgorithm(Protocol):
    """训练算法替换点；实现者只需构建和运行 job。"""

    algorithm_id: str

    def build_command(self, job: AdapterJob) -> list[str]:
        ...

    def run(self, job: AdapterJob, *, dry_run: bool = False) -> int:
        ...


class MSSwiftLoraTrainer:
    """调用仓库内 ``libs/ms-swift`` 完成多模态 LoRA SFT。"""

    algorithm_id = "ms-swift-lora-sft-v1"

    def __init__(self, config: ProjectConfig):
        self.config = config

    @property
    def swift_cli(self) -> Path:
        # 当前 vendored ms-swift 的统一子命令入口是 main.py（可执行 `sft`）。
        return self.config.paths.ms_swift_root / "swift" / "cli" / "main.py"

    def build_command(self, job: AdapterJob) -> list[str]:
        if not self.swift_cli.is_file():
            raise FileNotFoundError(f"ms-swift CLI 不存在: {self.swift_cli}")
        offline = self.config.offline
        command = [
            sys.executable,
            str(self.swift_cli),
            "sft",
            "--model",
            offline.student_model_path,
            "--tuner_type",
            "lora",
            "--dataset",
            str(job.train_dataset.resolve()),
            "--split_dataset_ratio",
            "0",
            "--output_dir",
            str(job.output_dir.resolve()),
            "--num_train_epochs",
            str(offline.epochs),
            "--learning_rate",
            str(offline.learning_rate),
            "--lora_rank",
            str(offline.lora_rank),
            "--lora_alpha",
            str(offline.lora_alpha),
            "--max_pixels",
            str(offline.max_pixels),
            "--freeze_vit",
            str(offline.freeze_vit).lower(),
        ]
        if job.validation_dataset and job.validation_dataset.is_file():
            command.extend(("--val_dataset", str(job.validation_dataset.resolve())))
        command.extend(job.extra_args)
        return command

    def run(self, job: AdapterJob, *, dry_run: bool = False) -> int:
        command = self.build_command(job)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "training_command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if dry_run:
            return 0
        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.config.paths.ms_swift_root) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        completed = subprocess.run(
            command,
            cwd=self.config.paths.ms_swift_root,
            env=environment,
            check=False,
        )
        return completed.returncode


def filter_dataset_by_primitives(
    source: Path, destination: Path, primitives: Sequence[str]
) -> int:
    """按样本 metadata 原语标签生成定向 LoRA 子数据集。"""

    selected = set(primitives)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            item = json.loads(line)
            labels = set(item.get("metadata", {}).get("primitives", []))
            if selected and not labels.intersection(selected):
                continue
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def default_training_job(config: ProjectConfig) -> AdapterJob:
    """返回使用完整蒸馏数据训练单个通用 LoRA 的默认 job。"""

    validation = config.offline.dataset_dir / "validation.jsonl"
    return AdapterJob(
        name="android_world_all",
        train_dataset=config.offline.dataset_dir / "train.jsonl",
        validation_dataset=validation if validation.exists() else None,
        output_dir=config.offline.output_dir / "android_world_all",
    )
