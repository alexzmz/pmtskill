"""可替换训练算法与 ms-swift LoRA 实现。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import re
import threading
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..core.config import ProjectConfig
from ..core.io import write_json_atomic, write_jsonl


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


@dataclass(slots=True)
class PreparedTrainingJob:
    """本次训练使用的固定 JSONL 索引及其兼容性检查报告。"""

    job: AdapterJob
    manifest_path: Path
    manifest: dict[str, Any]


def _image_candidates(
    value: str, *, source_dataset: Path, configured_dataset_dir: Path
) -> list[Path]:
    """给旧绝对路径和相对路径生成确定性的本地候选位置。"""

    original = Path(value).expanduser()
    candidates: list[Path] = []
    if original.is_absolute():
        candidates.append(original)
    else:
        candidates.extend(
            (source_dataset.parent / original, configured_dataset_dir / original)
        )

    # 合并/迁移数据集后，JSONL 常仍指向旧的 .../dataset/images/...。
    # 只从稳定的 images 锚点以后重定位，不根据文件名做模糊搜索。
    normalized_parts = Path(value.replace("\\", "/")).parts
    image_indexes = [
        index
        for index, part in enumerate(normalized_parts)
        if part.lower() == "images"
    ]
    if image_indexes:
        relative = Path(*normalized_parts[image_indexes[-1] :])
        candidates.append(configured_dataset_dir / relative)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _readable_image(path: Path, cache: dict[Path, bool]) -> bool:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        from PIL import Image
    except ModuleNotFoundError:
        # 轻量单测环境可能没有 Pillow；至少用常见格式魔数拒绝空文件/文本文件。
        try:
            header = path.read_bytes()[:16]
            readable = bool(
                header.startswith(b"\x89PNG\r\n\x1a\n")
                or header.startswith(b"\xff\xd8\xff")
                or header.startswith((b"GIF87a", b"GIF89a", b"BM"))
                or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            )
        except OSError:
            readable = False
    else:
        try:
            with Image.open(path) as image:
                image.verify()
            readable = True
        except (OSError, SyntaxError, ValueError, TypeError):
            readable = False
    cache[path] = readable
    return readable


def _align_leading_image_placeholders(row: dict[str, Any], image_count: int) -> None:
    """在部分图片损坏但仍有可用图片时，同步构建器生成的前置占位符。"""

    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("<image>"):
            return
        body = content
        while body.startswith("<image>"):
            body = body[len("<image>") :]
        message["content"] = "<image>" * image_count + body
        return


def _prepare_dataset_split(
    source: Path,
    destination: Path,
    *,
    configured_dataset_dir: Path,
) -> dict[str, Any]:
    """冻结一个 JSONL split，并修复路径、过滤真正不可读取的图片样本。"""

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []
    image_cache: dict[Path, bool] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counters["source_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counters["rejected_invalid_json"] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append(
                        {"line": line_number, "reason": "invalid_json"}
                    )
                continue
            if not isinstance(row, dict):
                counters["rejected_non_object"] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append(
                        {"line": line_number, "reason": "non_object"}
                    )
                continue

            raw_images = row.get("images")
            if raw_images is None:
                rows.append(row)
                counters["accepted_rows"] += 1
                counters["text_only_rows"] += 1
                continue
            references = raw_images if isinstance(raw_images, list) else [raw_images]
            valid_images: list[str] = []
            invalid_reasons: list[str] = []
            for reference in references:
                if not isinstance(reference, str) or not reference.strip():
                    invalid_reasons.append("invalid_image_reference")
                    counters["invalid_image_references"] += 1
                    continue
                if reference.startswith(("http://", "https://", "data:")):
                    valid_images.append(reference)
                    counters["remote_images_unchecked"] += 1
                    continue
                candidates = _image_candidates(
                    reference,
                    source_dataset=source,
                    configured_dataset_dir=configured_dataset_dir,
                )
                existing = [candidate for candidate in candidates if candidate.is_file()]
                selected = next(
                    (
                        candidate
                        for candidate in existing
                        if _readable_image(candidate, image_cache)
                    ),
                    None,
                )
                if selected is None:
                    reason = "corrupt_image" if existing else "missing_image"
                    invalid_reasons.append(reason)
                    counters[f"{reason}_references"] += 1
                    continue
                selected_text = str(selected)
                valid_images.append(selected_text)
                original = Path(reference).expanduser().resolve(strict=False)
                if selected != original:
                    counters["rebased_image_references"] += 1

            if references and not valid_images:
                reason = (
                    "corrupt_or_invalid_images"
                    if any(item != "missing_image" for item in invalid_reasons)
                    else "missing_images"
                )
                counters[f"rejected_{reason}"] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append(
                        {
                            "line": line_number,
                            "reason": reason,
                            "declared_images": len(references),
                        }
                    )
                continue
            if invalid_reasons:
                counters["repaired_partial_image_rows"] += 1
                counters["dropped_image_references"] += len(invalid_reasons)
                _align_leading_image_placeholders(row, len(valid_images))
            row["images"] = valid_images
            rows.append(row)
            counters["accepted_rows"] += 1

    write_jsonl(destination, rows)
    return {
        "source_path": str(source.resolve()),
        "snapshot_path": str(destination.resolve()),
        **dict(sorted(counters.items())),
        "rejected_rows": counters["source_rows"] - counters["accepted_rows"],
        "rejected_examples": rejected_examples,
    }


def prepare_training_job(
    job: AdapterJob,
    *,
    configured_dataset_dir: Path,
    snapshot_dir: Path,
) -> PreparedTrainingJob:
    """为一次训练生成不可漂移、可审计且已验证图片的 JSONL 快照。"""

    if not job.train_dataset.is_file():
        raise FileNotFoundError(f"训练数据集不存在: {job.train_dataset}")
    snapshot_dir = snapshot_dir.resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    train_path = snapshot_dir / "train.jsonl"
    train_report = _prepare_dataset_split(
        job.train_dataset.resolve(),
        train_path,
        configured_dataset_dir=configured_dataset_dir.resolve(),
    )
    validation_path: Path | None = None
    validation_report: dict[str, Any] | None = None
    if job.validation_dataset is not None and job.validation_dataset.is_file():
        candidate = snapshot_dir / "validation.jsonl"
        validation_report = _prepare_dataset_split(
            job.validation_dataset.resolve(),
            candidate,
            configured_dataset_dir=configured_dataset_dir.resolve(),
        )
        if int(validation_report.get("accepted_rows", 0)) > 0:
            validation_path = candidate

    manifest = {
        "schema_version": 1,
        "configured_dataset_dir": str(configured_dataset_dir.resolve()),
        "policy": (
            "rebase old paths at the images/ anchor; keep readable rows; "
            "drop only invalid JSON or rows without any readable declared image"
        ),
        "train": train_report,
        "validation": validation_report,
    }
    manifest_path = snapshot_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    if int(train_report.get("accepted_rows", 0)) <= 0:
        raise ValueError(
            "训练集预检后没有可用样本；请查看 " f"{manifest_path}"
        )
    return PreparedTrainingJob(
        job=replace(
            job,
            train_dataset=train_path,
            validation_dataset=validation_path,
        ),
        manifest_path=manifest_path,
        manifest=manifest,
    )


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

    def build_environment(self) -> dict[str, str]:
        """构造训练子进程环境，并只在已配置时限制可见物理 GPU。"""

        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.config.paths.ms_swift_root) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        if self.config.offline.cuda_visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = (
                self.config.offline.cuda_visible_devices
            )
        return environment

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
        environment = self.build_environment()
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
        "--save_total_limit",
        "--create_checkpoint_symlink",
        "--load_args",
        "--load_data_args",
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

    每段使用独立的 epoch 目录；``save_strategy=epoch`` 保证段末有完整 adapter 和
    optimizer 状态可恢复，``save_total_limit=1`` 则只保留该段最新 checkpoint。
    """

    validate_staged_training_args(job)
    orchestration_args = (
        "--add_version",
        "false",
        "--save_strategy",
        "epoch",
        "--save_total_limit",
        "1",
        "--create_checkpoint_symlink",
        "true",
        # 防止未来 ms-swift 默认值变化时，从 checkpoint/args.json 回载旧数据路径。
        "--load_args",
        "false",
        "--load_data_args",
        "false",
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
