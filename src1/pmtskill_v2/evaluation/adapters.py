"""评测入口的训练运行解析与单服务多 LoRA 部署。

用户传入的是 adapter 的顶层训练目录，而 ms-swift 推理需要实际 checkpoint。
本模块把目录选择规则集中到一处，三种 AndroidWorld 评测因而一定加载同一份权重。
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..core.config import ProjectConfig, TrainingEvaluationConfig
from ..core.models import ModelProfile
from ..offline.trainer import find_latest_adapter_checkpoint
from .deployment import MSSwiftEvaluationDeployment


_EPOCH_DIRECTORY = re.compile(r"epoch_(\d+)(?:_(\d+))?$")
_MODEL_ID_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]+")
_MODEL_ID = re.compile(r"[a-zA-Z0-9_.-]+$")


def _epoch_value(path: Path) -> float | None:
    """把 ``epoch_003``/``epoch_002_500`` 转为可比较的累计 epoch。"""

    match = _EPOCH_DIRECTORY.fullmatch(path.name)
    if not match:
        return None
    whole = int(match.group(1))
    fraction = match.group(2)
    return float(whole) if fraction is None else whole + int(fraction) / 10 ** len(fraction)


def adapter_model_id(path: str | Path) -> str:
    """从 adapter 顶层目录生成可作为 OpenAI ``model`` 的稳定 ID。"""

    name = Path(path).expanduser().name or "adapter"
    value = _MODEL_ID_UNSAFE.sub("-", name).strip("-._")
    return value or "adapter"


@dataclass(frozen=True, slots=True)
class ResolvedAdapterCheckpoint:
    """一次 adapter 目录解析的完整、可审计结果。"""

    input_path: Path
    adapter_root: Path
    run_dir: Path | None
    epoch_dir: Path | None
    checkpoint_dir: Path
    adapter_config: Path
    optimizer_state: Path
    scheduler_state: Path
    base_model_path: str | None
    lora_rank: int | None
    selection: str

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataclasses.asdict(self).items()
        }


def _latest_run(training_runs: Path) -> Path:
    candidates = [
        child
        for child in training_runs.iterdir()
        if child.is_dir() and (child / "training").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"training_runs 中没有含 training/ 的运行: {training_runs}")
    # 训练运行名以可排序时间戳开头；mtime 只作为同名/非标准名称的补充依据。
    return max(candidates, key=lambda item: (item.name, item.stat().st_mtime_ns))


def _latest_epoch(training_dir: Path) -> Path:
    candidates = [
        (value, child)
        for child in training_dir.iterdir()
        if child.is_dir() and (value := _epoch_value(child)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"training/ 中没有 epoch_XXX 目录: {training_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _root_from_nested_path(path: Path) -> Path:
    """输入本身已经位于 training_runs 内时，仍返回 adapter 顶层目录。"""

    for parent in (path, *path.parents):
        if parent.name == "training_runs":
            return parent.parent.resolve()
    return path.resolve()


def resolve_adapter_checkpoint(
    adapter_path: str | Path,
    *,
    require_training_state: bool = True,
) -> ResolvedAdapterCheckpoint:
    """解析 adapter 顶层路径到最新运行、最后 epoch 的最佳 checkpoint。

    首选布局为 ``adapter/training_runs/<time>/training/<last_epoch>/best``。
    为兼容旧运行，缺少 ``best`` 时回退到该 epoch 最新的 ``checkpoint-*``；也允许
    直接传入完整 checkpoint。``optimizer.pt`` 和 ``scheduler.pt`` 是训练恢复状态，
    推理不会反序列化它们，但默认要求两者存在，以防误选到未完整落盘的 checkpoint。
    """

    source = Path(adapter_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"adapter 路径不存在或不是目录: {source}")

    run_dir: Path | None = None
    epoch_dir: Path | None = None
    selection = "direct_checkpoint"
    if (source / "adapter_config.json").is_file():
        checkpoint = source
        adapter_root = _root_from_nested_path(source)
    else:
        adapter_root = _root_from_nested_path(source)
        if source.name == "training_runs":
            training_runs = source
            adapter_root = source.parent.resolve()
        elif (source / "training_runs").is_dir():
            training_runs = source / "training_runs"
            adapter_root = source.resolve()
        elif (source / "training").is_dir():
            run_dir = source
            training_runs = None
        elif source.name == "training" and source.is_dir():
            run_dir = source.parent
            training_runs = None
        elif _epoch_value(source) is not None:
            epoch_dir = source
            run_dir = source.parent.parent if source.parent.name == "training" else None
            training_runs = None
        else:
            raise FileNotFoundError(
                "无法识别 adapter 目录；需要 adapter 顶层、training_runs、运行目录、"
                f"epoch 目录或完整 checkpoint: {source}"
            )
        if run_dir is None and epoch_dir is None:
            assert training_runs is not None
            run_dir = _latest_run(training_runs)
        if epoch_dir is None:
            assert run_dir is not None
            epoch_dir = _latest_epoch(run_dir / "training")
        best = epoch_dir / "best"
        if best.exists() and best.is_dir():
            checkpoint = best.resolve(strict=True)
            selection = "latest_run_last_epoch_best"
        else:
            checkpoint = find_latest_adapter_checkpoint(epoch_dir)
            selection = "latest_run_last_epoch_checkpoint_fallback"

    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"checkpoint 缺少 adapter_config.json: {checkpoint}")
    optimizer = checkpoint / "optimizer.pt"
    scheduler = checkpoint / "scheduler.pt"
    missing_state = [path.name for path in (optimizer, scheduler) if not path.is_file()]
    if require_training_state and missing_state:
        raise FileNotFoundError(
            f"checkpoint 训练状态不完整，缺少 {', '.join(missing_state)}: {checkpoint}"
        )
    try:
        config_value = json.loads(adapter_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"adapter_config.json 无法读取: {adapter_config}") from exc
    base_model = config_value.get("base_model_name_or_path")
    raw_rank = config_value.get("r")
    lora_rank = int(raw_rank) if raw_rank is not None else None
    if lora_rank is not None and lora_rank <= 0:
        raise ValueError(f"adapter_config.json 的 LoRA rank 无效: {raw_rank!r}")
    return ResolvedAdapterCheckpoint(
        input_path=source,
        adapter_root=adapter_root,
        run_dir=run_dir.resolve() if run_dir is not None else None,
        epoch_dir=epoch_dir.resolve() if epoch_dir is not None else None,
        checkpoint_dir=checkpoint.resolve(),
        adapter_config=adapter_config.resolve(),
        optimizer_state=optimizer.resolve(),
        scheduler_state=scheduler.resolve(),
        base_model_path=str(base_model) if base_model else None,
        lora_rank=lora_rank,
        selection=selection,
    )


@dataclass(frozen=True, slots=True)
class AdapterDeploymentBinding:
    """一个路由 model_id、能力画像和实际 LoRA checkpoint 的绑定。"""

    model_id: str
    checkpoint: ResolvedAdapterCheckpoint
    template_profile: ModelProfile


class MSSwiftAdapterDeployment(MSSwiftEvaluationDeployment):
    """在一个 ms-swift OpenAI 服务中加载一个或多个命名 LoRA adapter。"""

    def __init__(
        self,
        config: ProjectConfig,
        settings: TrainingEvaluationConfig,
        bindings: Sequence[AdapterDeploymentBinding],
        *,
        base_model_path: str | None = None,
    ):
        if not bindings:
            raise ValueError("评测至少需要一个 adapter")
        model_ids = [binding.model_id for binding in bindings]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"adapter model_id 不能重复: {model_ids}")
        invalid_ids = [item for item in model_ids if not _MODEL_ID.fullmatch(item)]
        if invalid_ids:
            raise ValueError(
                "adapter model_id 只能包含字母、数字、点、下划线和连字符: "
                + ", ".join(invalid_ids)
            )
        if any("=" in str(binding.checkpoint.checkpoint_dir) for binding in bindings):
            raise ValueError("adapter checkpoint 路径不能包含 '='，否则 ms-swift 无法解析命名映射")
        reserved = {
            "--model",
            "--adapters",
            "--infer_backend",
            "--host",
            "--port",
            "--served_model_name",
            "--max_new_tokens",
            "--verbose",
            "--vllm_max_model_len",
            "--vllm_gpu_memory_utilization",
            "--vllm_max_lora_rank",
        }
        conflicts = sorted(
            {
                token.split("=", 1)[0]
                for token in settings.deploy_extra_args
                if token.split("=", 1)[0] in reserved
            }
        )
        if conflicts:
            raise ValueError(
                "--deploy-extra-arg 不能覆盖评测编排参数: " + ", ".join(conflicts)
            )
        if len(bindings) > 1 and settings.infer_backend != "vllm":
            raise ValueError("多 adapter 在线路由目前要求 infer_backend=vllm")
        super().__init__(config, settings, bindings[0].template_profile)
        self.bindings = tuple(bindings)
        self.base_model_path = base_model_path or self._infer_base_model()

    def _infer_base_model(self) -> str:
        candidates = {
            binding.checkpoint.base_model_path
            for binding in self.bindings
            if binding.checkpoint.base_model_path
        }
        if len(candidates) > 1:
            raise ValueError(
                "多个 adapter 的 base_model_name_or_path 不一致，请显式传 --base-model-path: "
                + ", ".join(sorted(candidates))
            )
        if candidates:
            return next(iter(candidates))  # type: ignore[return-value]
        if self.config.offline.student_model_path:
            return self.config.offline.student_model_path
        raise ValueError("无法推断 base model；请传 --base-model-path")

    def build_adapter_command(self) -> list[str]:
        """生成命名 LoRA 映射命令；请求的 model 字段就是路由 model_id。"""

        if not self.swift_cli.is_file():
            raise FileNotFoundError(f"ms-swift CLI 不存在: {self.swift_cli}")
        command = [
            sys.executable,
            str(self.swift_cli),
            "deploy",
            "--model",
            self.base_model_path,
            "--adapters",
        ]
        command.extend(
            f"{binding.model_id}={binding.checkpoint.checkpoint_dir}"
            for binding in self.bindings
        )
        command.extend(
            (
                "--infer_backend",
                self.settings.infer_backend,
                "--host",
                self.settings.deploy_host,
                "--port",
                str(self.settings.deploy_port),
                "--served_model_name",
                "pmtskill-base",
                "--max_new_tokens",
                str(self.settings.max_new_tokens),
            )
        )
        command.extend(self.settings.deploy_extra_args)
        command.extend(
            (
                "--verbose",
                "false",
                "--vllm_max_model_len",
                str(self.settings.max_model_len),
                "--vllm_gpu_memory_utilization",
                str(self.settings.gpu_memory_utilization),
                "--vllm_max_lora_rank",
                str(
                    max(
                        self.config.offline.lora_rank,
                        *(binding.checkpoint.lora_rank or 1 for binding in self.bindings),
                    )
                ),
            )
        )
        return command

    def profiles(self) -> tuple[ModelProfile, ...]:
        endpoint = (
            f"http://{self.settings.deploy_host}:{self.settings.deploy_port}/v1"
        )
        return tuple(
            dataclasses.replace(
                binding.template_profile,
                model_id=binding.model_id,
                served_model=binding.model_id,
                base_url=endpoint,
                adapter=None,
                api_key_env=None,
                enabled=True,
                metadata={
                    **binding.template_profile.metadata,
                    "evaluation_checkpoint": str(
                        binding.checkpoint.checkpoint_dir
                    ),
                    "adapter_input_path": str(binding.checkpoint.input_path),
                    "adapter_run_dir": (
                        str(binding.checkpoint.run_dir)
                        if binding.checkpoint.run_dir is not None
                        else None
                    ),
                    "adapter_epoch_dir": (
                        str(binding.checkpoint.epoch_dir)
                        if binding.checkpoint.epoch_dir is not None
                        else None
                    ),
                    "adapter_selection": binding.checkpoint.selection,
                    "lora_rank": binding.checkpoint.lora_rank,
                    "optimizer_state": str(binding.checkpoint.optimizer_state),
                    "scheduler_state": str(binding.checkpoint.scheduler_state),
                },
            )
            for binding in self.bindings
        )

    @contextlib.contextmanager
    def activate_adapters(self) -> Iterator[tuple[ModelProfile, ...]]:
        """启动命名 adapter 服务，并产出可直接交给 evaluator 的画像列表。"""

        with self._activate_command(self.build_adapter_command()):
            yield self.profiles()
