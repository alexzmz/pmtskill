"""AndroidWorld 教师 episode 到 ms-swift 多模态 JSONL 的转换。"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import random
import re
from collections import Counter
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


def _episode_outcome(value: Any) -> str:
    """返回 successful、failed 或 unknown，避免把 NaN 当成成功。"""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "success", "successful"}:
            return "successful"
        if normalized in {"0", "false", "fail", "failed", "failure"}:
            return "failed"
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(numeric):
        return "unknown"
    return "successful" if numeric > 0.5 else "failed"


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


def _value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _usable_text(value: Any) -> str | None:
    """只接受有实际语义的文本，过滤 None/NaN 等占位值。"""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _action_json_text(value: Any) -> str | None:
    """兼容 dict 和 AndroidWorld JSONAction，作为 action_output 的回退。"""

    direct = _usable_text(value)
    if direct:
        return direct
    payload: Any
    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "as_dict", None)):
        payload = value.as_dict()
    else:
        return None
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _training_target(
    action_output: Any,
    raw_response: Any,
    action_reason: Any,
    parsed_action: Any,
) -> tuple[str | None, str | None]:
    """尽量保留教师思考；仅在所有可辨认输出都缺失时拒绝该 step。"""

    direct = _usable_text(action_output)
    if direct:
        return direct, "action_output"
    raw = _usable_text(raw_response)
    if raw:
        return raw, "action_raw_response"
    reason = _usable_text(action_reason)
    action_json = _action_json_text(parsed_action)
    if reason and action_json:
        return f"Reason: {reason}\nAction: {action_json}", "reason_and_parsed_action"
    if action_json:
        return f"Action: {action_json}", "parsed_action"
    if reason:
        return f"Reason: {reason}", "action_reason"
    return None, None


def _safe_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return result[:80] or "episode"


def _save_image(image: Any, path: Path) -> bool:
    from PIL import Image

    try:
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        if pil_image.width <= 0 or pil_image.height <= 0:
            return False
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
    except Exception:  # 输入轨迹中的截图对象可能是 NaN、标量或损坏数组。
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, format="PNG", optimize=True)
    return True


@dataclass(slots=True)
class _EpisodeConversion:
    samples: list[dict[str, Any]]
    candidate_steps: int
    rejected_steps: int
    rejection_reasons: dict[str, int]


@dataclass(slots=True)
class DatasetBuildResult:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_samples: int
    validation_samples: int
    accepted_episodes: int
    rejected_episodes: int
    candidate_steps: int = 0
    rejected_steps: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "manifest_path": str(self.manifest_path),
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "accepted_episodes": self.accepted_episodes,
            "rejected_episodes": self.rejected_episodes,
            "candidate_steps": self.candidate_steps,
            "accepted_steps": self.train_samples + self.validation_samples,
            "rejected_steps": self.rejected_steps,
        }


class AndroidWorldDistillationDatasetBuilder:
    """构建 ``[截图, SoM 截图, prompt] -> [Reason, Action]`` 监督样本。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        successful_only: bool = False,
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
        self,
        episode: dict[str, Any],
        episode_index: int,
        episode_outcome: str,
    ) -> _EpisodeConversion:
        episode_data = _normalize_episode_data(
            _episode_value(episode, "episode_data", default={})
        )
        if episode_data is None:
            return _EpisodeConversion([], 0, 0, {})
        actions = _step_values(episode_data.get("action_output"))
        raw_responses = _step_values(episode_data.get("action_raw_response"))
        action_reasons = _step_values(episode_data.get("action_reason"))
        parsed_actions = _step_values(episode_data.get("action_output_json"))
        prompts = _step_values(episode_data.get("action_prompt"))
        raw_images = _step_values(
            _episode_value(
                episode_data,
                "raw_screenshot",
                "screenshot",
                "before_screenshot",
            )
        )
        som_images = _step_values(
            _episode_value(
                episode_data,
                "before_screenshot_with_som",
                "after_screenshot_with_som",
            )
        )
        task_name = str(
            _episode_value(episode, "task_template", "task_name", default="unknown")
        )
        goal = str(_episode_value(episode, "goal", default=""))
        episode_key = hashlib.sha256(
            f"{task_name}|{episode_index}|{goal}".encode("utf-8")
        ).hexdigest()[:16]
        samples: list[dict[str, Any]] = []
        rejection_reasons: Counter[str] = Counter()
        step_count = max(
            len(actions),
            len(raw_responses),
            len(action_reasons),
            len(parsed_actions),
            len(prompts),
            len(raw_images),
            len(som_images),
            0,
        )
        for step in range(step_count):
            target, target_source = _training_target(
                _value_at(actions, step),
                _value_at(raw_responses, step),
                _value_at(action_reasons, step),
                _value_at(parsed_actions, step),
            )
            if target is None:
                rejection_reasons["missing_target"] += 1
                continue
            prompt = _usable_text(_value_at(prompts, step))
            prompt_source = "action_prompt"
            if prompt is None:
                prompt = _usable_text(goal)
                prompt_source = "goal"
            if prompt is None:
                rejection_reasons["missing_prompt_and_goal"] += 1
                continue

            image_dir = self.output_dir / "images" / _safe_name(task_name) / episode_key
            raw_path = (image_dir / f"step_{step:04d}_raw.png").resolve()
            som_path = (image_dir / f"step_{step:04d}_som.png").resolve()
            image_paths: list[str] = []
            raw_image = _value_at(raw_images, step)
            som_image = _value_at(som_images, step)
            if raw_image is not None and _save_image(raw_image, raw_path):
                image_paths.append(str(raw_path))
            if som_image is not None and _save_image(som_image, som_path):
                image_paths.append(str(som_path))
            if not image_paths:
                rejection_reasons["missing_or_invalid_image"] += 1
                continue

            # <image> 占位符数必须与 images 数量严格一致，这是 ms-swift 的格式要求。
            samples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>" * len(image_paths) + "\n" + prompt,
                        },
                        {"role": "assistant", "content": target},
                    ],
                    "images": image_paths,
                    "metadata": {
                        "task_name": task_name,
                        "goal": goal,
                        "episode_id": episode_key,
                        "episode_outcome": episode_outcome,
                        "step": step,
                        "prompt_source": prompt_source,
                        "target_source": target_source,
                        "image_count": len(image_paths),
                        "primitives": list(infer_action_primitives(target)),
                    },
                }
            )
        rejected_steps = sum(rejection_reasons.values())
        return _EpisodeConversion(
            samples,
            step_count,
            rejected_steps,
            dict(sorted(rejection_reasons.items())),
        )

    def build(self, trajectory_root: str | Path) -> DatasetBuildResult:
        """转换并按 episode 切分 train/validation，返回完整 manifest。"""

        episodes = load_episode_files(trajectory_root)
        accepted: list[list[dict[str, Any]]] = []
        rejected = 0
        candidate_steps = 0
        rejected_steps = 0
        outcome_counts: Counter[str] = Counter()
        accepted_outcome_counts: Counter[str] = Counter()
        step_rejection_reasons: Counter[str] = Counter()
        rejection_reasons = {"filtered_by_success": 0, "invalid_or_empty": 0}
        for index, episode in enumerate(episodes):
            outcome = _episode_outcome(
                _episode_value(episode, "is_successful", "successful", default=None)
            )
            outcome_counts[outcome] += 1
            if self.successful_only and outcome != "successful":
                rejected += 1
                rejection_reasons["filtered_by_success"] += 1
                continue
            conversion = self._convert_episode(episode, index, outcome)
            candidate_steps += conversion.candidate_steps
            rejected_steps += conversion.rejected_steps
            step_rejection_reasons.update(conversion.rejection_reasons)
            if conversion.samples:
                accepted.append(conversion.samples)
                accepted_outcome_counts[outcome] += 1
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
                f"filtered_by_success={rejection_reasons['filtered_by_success']}，"
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
            "episode_outcomes": dict(sorted(outcome_counts.items())),
            "accepted_episode_outcomes": dict(
                sorted(accepted_outcome_counts.items())
            ),
            "candidate_steps": candidate_steps,
            "accepted_steps": train_count + validation_count,
            "rejected_steps": rejected_steps,
            "step_rejection_reasons": dict(sorted(step_rejection_reasons.items())),
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
            candidate_steps,
            rejected_steps,
        )
