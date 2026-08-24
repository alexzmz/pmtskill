"""可替换训练算法与 ms-swift LoRA 实现。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import re
import threading
from dataclasses import dataclass, replace
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
    # 分段训练时使用累计 epoch 目标，例如先训练到 1，再从 checkpoint 恢复到 2。
    num_train_epochs: float | None = None
    resume_from_checkpoint: Path | None = None


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
            str(
                job.num_train_epochs
                if job.num_train_epochs is not None
                else offline.epochs
            ),
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
        if job.resume_from_checkpoint is not None:
            command.extend(
                (
                    "--resume_from_checkpoint",
                    str(job.resume_from_checkpoint.resolve()),
                )
            )
        command.extend(job.extra_args)
        return command

    def run(self, job: AdapterJob, *, dry_run: bool = False) -> int:
        command = self.build_command(job)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        command_name = "training_command.json"
        if job.num_train_epochs is not None:
            epoch_text = str(job.num_train_epochs).replace(".", "_")
            command_name = f"training_command_to_epoch_{epoch_text}.json"
        (job.output_dir / command_name).write_text(
            json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if dry_run:
            return 0
        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.config.paths.ms_swift_root) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        # 使用两个 PIPE 并发转发，保证 ms-swift 的 stdout 进入 runtime.log，
        # stderr 同时进入 runtime.log 与 errors.log；并发读取可避免管道写满死锁。
        process = subprocess.Popen(
            command,
            cwd=self.config.paths.ms_swift_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        def forward(stream, target) -> None:
            for line in stream:
                print(line, end="", file=target, flush=True)

        stdout_thread = threading.Thread(
            target=forward, args=(process.stdout, sys.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=forward, args=(process.stderr, sys.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        return return_code


_CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)$")


def find_latest_adapter_checkpoint(output_dir: Path) -> Path:
    """寻找 ms-swift 生成的最新、可加载 LoRA checkpoint。

    ms-swift 默认可能在 output_dir 下增加版本子目录，因此这里递归查找，并且
    只有包含 ``adapter_config.json`` 的目录才视为完整 checkpoint。
    """

    candidates: list[tuple[int, float, Path]] = []
    if output_dir.is_dir():
        for child in output_dir.rglob("checkpoint-*"):
            match = _CHECKPOINT_NAME.fullmatch(child.name)
            if not match or not child.is_dir():
                continue
            if not (child / "adapter_config.json").is_file():
                continue
            candidates.append((int(match.group(1)), child.stat().st_mtime, child))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2].resolve()
    if (output_dir / "adapter_config.json").is_file():
        return output_dir.resolve()
    raise FileNotFoundError(f"训练输出中没有完整 LoRA checkpoint: {output_dir}")


def validate_staged_training_args(job: AdapterJob) -> None:
    """在启动昂贵的基线评测前，拒绝会破坏分段恢复的额外参数。"""

    reserved = {
        "--num_train_epochs",
        "--output_dir",
        "--resume_from_checkpoint",
        "--add_version",
        "--save_strategy",
        "--create_checkpoint_symlink",
    }
    conflicts = sorted(reserved.intersection(job.extra_args))
    if conflicts:
        raise ValueError(
            "带 AndroidWorld 评测的分段训练不能通过 --extra-arg 覆盖编排参数: "
            + ", ".join(conflicts)
        )


def staged_training_job(
    job: AdapterJob,
    *,
    output_dir: Path,
    target_epoch: float,
    resume_from_checkpoint: Path | None,
) -> AdapterJob:
    """构造一次可恢复的累计 epoch 训练 job。

    ``add_version=false`` 让各段共享同一 checkpoint 目录；``save_strategy=epoch``
    保证每个评测点都有完整 adapter 和 optimizer 状态可用于下一段恢复。
    """

    validate_staged_training_args(job)
    orchestration_args = (
        "--add_version",
        "false",
        "--save_strategy",
        "epoch",
        "--create_checkpoint_symlink",
        "true",
    )
    return replace(
        job,
        output_dir=output_dir,
        num_train_epochs=target_epoch,
        resume_from_checkpoint=resume_from_checkpoint,
        extra_args=tuple(job.extra_args) + orchestration_args,
    )


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
