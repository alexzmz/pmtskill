"""AndroidWorld 教师 episode 到 ms-swift 多模态 JSONL 的转换。"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import write_json_atomic, write_jsonl


def infer_action_primitives(action_output: str) -> tuple[str, ...]:
    """从 M3A Action JSON 中提取原语标签，供后续分 adapter 训练。"""

    lowered = action_output.lower()
    mapping = (
        ("action.double_tap", ("double_tap", "double tap")),
        ("action.long_press", ("long_press", "long press")),
        ("action.type", ("input_text", "type")),
        ("action.scroll", ("scroll",)),
        ("action.swipe", ("swipe",)),
        ("action.back", ('"back"', "go_back")),
        ("action.home", ('"home"',)),
        ("action.open_app", ("open_app",)),
        ("action.wait", ('"wait"',)),
        ("control.finish", ('"status"', "goal_status")),
        ("action.click", ('"click"', '"tap"')),
    )
    matches = [
        primitive for primitive, tokens in mapping if any(t in lowered for t in tokens)
    ]
    return tuple(matches or ("reason.decompose",))


def _episode_value(
    episode: Mapping[str, Any], *keys: str, default: Any = None
) -> Any:
    for key in keys:
        if key in episode:
            return episode[key]
    return default


_EPISODE_HINT_KEYS = frozenset(
    {"episode_data", "goal", "task_template", "task_name", "is_successful"}
)


def _iter_episode_records(value: Any):
    """兼容新旧 checkpointer 的 list、单 dict 和嵌套 tuple 包装。"""

    if isinstance(value, Mapping):
        if _EPISODE_HINT_KEYS.intersection(value):
            yield dict(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_episode_records(item)


def load_episode_files(root: str | Path) -> list[dict[str, Any]]:
    """递归读取 IncrementalCheckpointer 产生的所有 gzip pickle。"""

    episodes: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - 文件来自本地 AndroidWorld。
        episodes.extend(_iter_episode_records(value))
    return episodes


def _is_successful(value: Any) -> bool:
    """解析 AndroidWorld 的 0/1 成功分数，并把失败占位 NaN 判为 False。"""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "success", "successful"}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.5


def _normalize_episode_data(value: Any) -> Mapping[str, Any] | None:
    """把字段列表或按 step 保存的字典列表归一化为 dict-of-lists。"""

    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return {}
    if not all(isinstance(step, Mapping) for step in value):
        return None
    keys = {
        key
        for step in value
        for key in step
        if isinstance(key, str)
    }
    return {key: [step.get(key) for step in value] for key in keys}


def _step_values(value: Any) -> list[Any]:
    """允许旧轨迹把单步字段直接保存成标量，而不是长度为 1 的列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _safe_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return result[:80] or "episode"


def _save_image(image: Any, path: Path) -> None:
    from PIL import Image

    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, format="PNG", optimize=True)


@dataclass(slots=True)
class DatasetBuildResult:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_samples: int
    validation_samples: int
    accepted_episodes: int
    rejected_episodes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "manifest_path": str(self.manifest_path),
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "accepted_episodes": self.accepted_episodes,
            "rejected_episodes": self.rejected_episodes,
        }


class AndroidWorldDistillationDatasetBuilder:
    """构建 ``[截图, SoM 截图, prompt] -> [Reason, Action]`` 监督样本。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        successful_only: bool = True,
        validation_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.successful_only = successful_only
        if not 0 <= validation_ratio < 1:
            raise ValueError("validation_ratio 必须在 [0, 1) 内")
        self.validation_ratio = validation_ratio
        self.seed = seed

    def _convert_episode(
        self, episode: dict[str, Any], episode_index: int
    ) -> list[dict[str, Any]]:
        episode_data = _normalize_episode_data(
            _episode_value(episode, "episode_data", default={})
        )
        if episode_data is None:
            return []
        actions = _step_values(episode_data.get("action_output"))
        prompts = _step_values(episode_data.get("action_prompt"))
        raw_images = _step_values(episode_data.get("raw_screenshot"))
        som_images = _step_values(episode_data.get("before_screenshot_with_som"))
        task_name = str(
            _episode_value(episode, "task_template", "task_name", default="unknown")
        )
        goal = str(_episode_value(episode, "goal", default=""))
        episode_key = hashlib.sha256(
            f"{task_name}|{episode_index}|{goal}".encode("utf-8")
        ).hexdigest()[:16]
        samples: list[dict[str, Any]] = []
        step_count = min(len(actions), len(prompts), len(raw_images), len(som_images))
        for step in range(step_count):
            action = actions[step]
            prompt = prompts[step]
            if (
                not action
                or not prompt
                or raw_images[step] is None
                or som_images[step] is None
            ):
                continue
            image_dir = self.output_dir / "images" / _safe_name(task_name) / episode_key
            raw_path = (image_dir / f"step_{step:04d}_raw.png").resolve()
            som_path = (image_dir / f"step_{step:04d}_som.png").resolve()
            _save_image(raw_images[step], raw_path)
            _save_image(som_images[step], som_path)
            # <image> 占位符数必须与 images 数量严格一致，这是 ms-swift 的格式要求。
            samples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image><image>\n" + str(prompt),
                        },
                        {"role": "assistant", "content": str(action)},
                    ],
                    "images": [str(raw_path), str(som_path)],
                    "metadata": {
                        "task_name": task_name,
                        "goal": goal,
                        "episode_id": episode_key,
                        "step": step,
                        "primitives": list(infer_action_primitives(str(action))),
                    },
                }
            )
        return samples

    def build(self, trajectory_root: str | Path) -> DatasetBuildResult:
        """转换并按 episode 切分 train/validation，返回完整 manifest。"""

        episodes = load_episode_files(trajectory_root)
        accepted: list[list[dict[str, Any]]] = []
        rejected = 0
        rejection_reasons = {"unsuccessful": 0, "invalid_or_empty": 0}
        for index, episode in enumerate(episodes):
            successful = _is_successful(
                _episode_value(episode, "is_successful", "successful", default=False)
            )
            if self.successful_only and not successful:
                rejected += 1
                rejection_reasons["unsuccessful"] += 1
                continue
            samples = self._convert_episode(episode, index)
            if samples:
                accepted.append(samples)
            else:
                rejected += 1
                rejection_reasons["invalid_or_empty"] += 1

        rng = random.Random(self.seed)
        rng.shuffle(accepted)
        validation_episodes = int(round(len(accepted) * self.validation_ratio))
        if self.validation_ratio > 0 and len(accepted) > 1:
            validation_episodes = max(1, min(len(accepted) - 1, validation_episodes))
        validation_groups = accepted[:validation_episodes]
        train_groups = accepted[validation_episodes:]
        train_rows = [sample for group in train_groups for sample in group]
        validation_rows = [sample for group in validation_groups for sample in group]
        if not train_rows:
            raise ValueError(
                "没有可用训练样本；"
                f"读取 episode={len(episodes)}，"
                f"unsuccessful={rejection_reasons['unsuccessful']}，"
                f"invalid_or_empty={rejection_reasons['invalid_or_empty']}；"
                "请检查成功标记和截图字段"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / "train.jsonl"
        validation_path = self.output_dir / "validation.jsonl"
        manifest_path = self.output_dir / "manifest.json"
        train_count = write_jsonl(train_path, train_rows)
        validation_count = write_jsonl(validation_path, validation_rows)
        manifest = {
            "format": "ms-swift-multimodal-jsonl",
            "teacher_target": "M3A Reason + Action",
            "trajectory_root": str(Path(trajectory_root).resolve()),
            "successful_only": self.successful_only,
            "split_unit": "episode",
            "seed": self.seed,
            "train_samples": train_count,
            "validation_samples": validation_count,
            "accepted_episodes": len(accepted),
            "rejected_episodes": rejected,
            "rejection_reasons": rejection_reasons,
        }
        write_json_atomic(manifest_path, manifest)
        return DatasetBuildResult(
            train_path,
            validation_path,
            manifest_path,
            train_count,
            validation_count,
            len(accepted),
            rejected,
        )
