"""按任务实例恢复完整评测轨迹，不依赖 AndroidWorld 的 metadata-only 恢复。

只复用同一实验中正常完成的 episode（包括真正的任务失败）。异常、损坏或未写完
的 checkpoint 会重跑；不会仅凭任务名或上一次成功就跳过其他参数组合。
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import sqlite3
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Iterator

from ..core.io import write_json_atomic
from ..core.models import ModelProfile, SkillRecord
from ..skills.identity import model_variant_id, skill_definition_hash
from .reporter import (
    STORED_SUMMARY_KEY, compact_result_for_storage, episode_data_is_usable, successful_episode_value,
)


PROTOCOL = "androidworld-evaluation-resume-v1"
STAMP_KEY = "_evaluation_resume"
ATTEMPT_KEY = "_evaluation_attempt_id"


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if callable(getattr(value, "tolist", None)):
        return value.tolist()
    return str(value)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=_json_default,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def model_resume_identity(profile: ModelProfile) -> dict[str, Any]:
    """地址/端口不构成模型版本；本地 LoRA 权重另做内容校验。"""

    checkpoint = profile.metadata.get("evaluation_checkpoint")
    weights: dict[str, str] = {}
    if checkpoint:
        root = Path(checkpoint)
        for path in sorted(root.glob("adapter*")):
            if path.is_file() and path.suffix in {".json", ".bin", ".safetensors"}:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                weights[path.name] = digest.hexdigest()
    return {
        "variant": model_variant_id(profile),
        "weights": weights,
        "capabilities": profile.capabilities,
        "average_latency_ms": profile.average_latency_ms,
        "switch_cost_ms": profile.switch_cost_ms,
        "enabled": profile.enabled,
    }


def skills_resume_identity(skills: Sequence[SkillRecord]) -> str:
    return fingerprint(sorted((
        {
            "id": skill.skill_id,
            "definition": skill_definition_hash(skill),
            "status": skill.status.value,
            "metadata": skill.metadata,
        }
        for skill in skills
    ), key=lambda item: item["id"]))


def routing_statistics_identity(database: Path, *, include_online_metrics: bool) -> str:
    """标定 SR 变化即使没有触发生命周期变化，也不能悄悄混入旧基准。"""

    with contextlib.closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        calibration = connection.execute(
            "SELECT skill_version_id, model_variant_id, COUNT(*), SUM(skill_success) "
            "FROM skill_calibration_trials WHERE valid = 1 "
            "GROUP BY skill_version_id, model_variant_id ORDER BY skill_version_id, model_variant_id"
        ).fetchall()
        online = connection.execute(
            "SELECT skill_id, model_id, successes, trials, latency_sum_ms FROM skill_metrics "
            "ORDER BY skill_id, model_id"
        ).fetchall() if include_online_metrics else "mutable-trace-recording"
        estimates = connection.execute(
            "SELECT skill_version_id, model_variant_id, estimate_json FROM skill_model_estimates "
            "ORDER BY skill_version_id, model_variant_id"
        ).fetchall()
    return fingerprint({"calibration": calibration, "online": online, "estimates": estimates})


def validate_resume_output(resume: bool, output_dir: Any) -> None:
    if resume and not output_dir:
        raise ValueError("--resume / evaluation.resume=true 必须指定同一个 --output-dir")


def _completed(episode: Mapping[str, Any]) -> bool:
    aux = episode.get("aux_data")
    if episode.get("_evaluation_episode_data_valid") is False or episode.get("exception_info") or (
        isinstance(aux, Mapping) and aux.get("infrastructure_failure_skipped")
    ):
        return False
    try:
        reward = float(episode["is_successful"])
    except (KeyError, ValueError, TypeError):
        return False
    return math.isfinite(reward) and episode_data_is_usable(episode.get("episode_data"))


def validate_evaluation_directory(
    output_dir: Path, specification: Mapping[str, Any], *, resume: bool,
) -> dict[str, Any] | None:
    """非修改性预检；V-Droid 父进程在写 resolved_request 前也使用它。"""

    target = output_dir.resolve()
    manifest_path = target / "evaluation_manifest.json"
    spec = json.loads(json.dumps(specification, default=_json_default))
    if manifest_path.is_file():
        if not resume:
            raise ValueError("评测目录已有记录；请用 --resume 续跑或指定新的 --output-dir")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL or manifest.get("specification") != spec:
            old = manifest.get("specification", {})
            changed = sorted(key for key in set(old) | set(spec) if old.get(key) != spec.get(key))
            raise ValueError("评测续跑配置不一致: " + ", ".join(changed)
                             + "；不能混合结果，请使用新的 --output-dir")
        return manifest
    checkpoint_dir = target / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError("已有 checkpoint 缺少 evaluation_manifest.json，不能验证实验身份；"
                         "请保留旧结果并使用新的 --output-dir")
    if (target / "summary.json").is_file():
        raise ValueError("已有旧版报告但缺少 evaluation_manifest.json；请使用新的 --output-dir")
    return None


def vdroid_resume_specification(request: Mapping[str, Any]) -> dict[str, Any]:
    spec = {key: value for key, value in request.items()
            if key not in {"resume", "output_dir", "config_path", "repo_root", "store_trajectories"}}
    # GPU 编号只影响部署位置，不改变固定 verifier。
    spec["model"] = {key: value for key, value in request["model"].items()
                     if key != "cuda_visible_devices"}
    return spec


class EvaluationProgress:
    """保存实验身份、原子 episode checkpoint 和便于查看的 progress.json。"""

    def __init__(
        self, output_dir: Path, specification: Mapping[str, Any], *, resume: bool,
        store_trajectories: bool = True,
        result_compactor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.target = output_dir.resolve()
        self.checkpoint_dir = self.target / "checkpoints"
        self.manifest_path = self.target / "evaluation_manifest.json"
        self.progress_path = self.target / "progress.json"
        self.resume = resume
        self.store_trajectories = store_trajectories
        self.result_compactor = result_compactor or (lambda episode: compact_result_for_storage(episode, []))
        # 转成与落盘后完全一致的 JSON 类型（tuple 与 list 不应造成假冲突）。
        self.spec = json.loads(json.dumps(specification, default=_json_default))
        self.cached: dict[str, dict[str, Any]] = {}
        self.current: dict[str, dict[str, Any]] = {}
        self.skipped = 0
        self.executed = 0
        self.corrupt = 0
        manifest = validate_evaluation_directory(self.target, self.spec, resume=resume)
        if manifest is not None:
            self.run_id = str(manifest["run_id"])
            self._load_checkpoints()
            if store_trajectories and any(STORED_SUMMARY_KEY in item for item in self.cached.values()):
                raise ValueError("已有实例仅保存了轻量结果，无法恢复其完整轨迹；请继续 --no-store-trajectories 或使用新目录")
        else:
            self.run_id = uuid.uuid4().hex
            write_json_atomic(self.manifest_path, {
                "protocol": PROTOCOL, "run_id": self.run_id, "specification": self.spec,
            })
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_checkpoints(self) -> None:
        # 仅加载此输出目录自己的 pickle；不要恢复来源不可信的 checkpoint。
        paths = [*self.checkpoint_dir.glob("*.pkl.gz"), *self.checkpoint_dir.glob("*.json")]
        latest: dict[str, Path] = {}
        for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns):
            latest[path.name[:-5] if path.suffix == ".json" else path.name[:-7]] = path
        for group, path in sorted(latest.items()):
            try:
                if path.suffix == ".json":
                    episodes = [json.loads(path.read_text(encoding="utf-8"))]
                else:
                    with gzip.open(path, "rb") as handle:
                        episodes = pickle.load(handle)
                if not isinstance(episodes, list) or len(episodes) != 1:
                    raise ValueError("checkpoint 必须只有一个任务实例")
                episode = episodes[0]
                stamp = episode.get(STAMP_KEY, {})
                if stamp.get("run_id") != self.run_id or stamp.get("group") != group:
                    raise ValueError("checkpoint 身份与 manifest 不一致")
                self.cached[stamp["group"]] = episode
            except Exception:
                self.corrupt += 1
                logging.warning("EVALUATION_RESUME_CORRUPT: %s 将重新执行", path, exc_info=True)

    @contextlib.contextmanager
    def watch(self, suite_utils: Any, suite: Any) -> Iterator[None]:
        """在设备初始化/模型调用之前返回缓存；保留原 instance_id 和完整数据。"""

        original = suite_utils._run_task
        stamps = {}
        for instances in suite.values():
            for index, task in enumerate(instances):
                group = f"{task.name}_{index}"
                if Path(group).name != group or "/" in group or "\\" in group:
                    raise ValueError(f"不安全的任务实例名: {group}")
                identity = {
                    "task": str(task.name), "instance_id": index,
                    "goal": str(task.goal), "params": getattr(task, "params", {}),
                }
                stamps[id(task)] = {
                    "run_id": self.run_id, "group": group,
                    "identity": fingerprint(identity),
                }
        groups = {stamp["group"] for stamp in stamps.values()}
        if set(self.cached) - groups:
            raise ValueError("任务实例范围与原 checkpoint 不一致；请使用新的输出目录")
        for stamp in stamps.values():
            previous = self.cached.get(stamp["group"])
            if previous is not None and previous[STAMP_KEY] != stamp:
                raise ValueError(f"任务实例参数发生变化: {stamp['group']}；请使用新的输出目录")
        self.current = {}
        self.expected = len(stamps)

        def run_task(*args: Any, **kwargs: Any) -> Any:
            task = args[0] if args else kwargs["task"]
            stamp = stamps[id(task)]
            previous = self.cached.get(stamp["group"])
            if previous is not None:
                if _completed(previous):
                    self.skipped += 1
                    logging.info("EVALUATION_RESUME_SKIP: %s", stamp["group"])
                    return copy.deepcopy(previous)
            self.executed += 1
            episode = original(*args, **kwargs)
            if isinstance(episode, dict):
                episode[STAMP_KEY] = stamp
                episode[ATTEMPT_KEY] = uuid.uuid4().hex
            return episode

        self._write_progress()
        suite_utils._run_task = run_task
        try:
            yield
        finally:
            suite_utils._run_task = original
            self._write_progress()

    def load(self, fields=None) -> list[dict[str, Any]]:
        # 禁用原生 metadata-only skip。watch 会把旧完整 episode 返回到本轮结果中，
        # 因而既不触发 return_full_episode_data 的恢复异常，也不丢掉旧轨迹。
        return []

    def save_episodes(self, task_episodes: list[dict[str, Any]], task_name: str) -> None:
        if len(task_episodes) != 1 or task_episodes[0].get(STAMP_KEY, {}).get("group") != task_name:
            raise ValueError("评测 checkpoint 缺少匹配的任务实例身份")
        episode = task_episodes[0]
        if not self.store_trajectories:
            result = self.result_compactor(episode)
            write_json_atomic(self.checkpoint_dir / f"{task_name}.json", json.loads(
                json.dumps(result, default=_json_default),
            ))
            self.current[task_name] = episode
            self.cached[task_name] = episode
            self._write_progress()
            return
        path = self.checkpoint_dir / f"{task_name}.pkl.gz"
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=5) as compressed:
                    pickle.dump(task_episodes, compressed, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self.current[task_name] = episode
        self.cached[task_name] = episode
        self._write_progress()

    def episodes(self) -> list[dict[str, Any]]:
        return list(self.current.values())

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.resume, "run_id": self.run_id,
            "episodes_skipped": self.skipped, "episodes_executed": self.executed,
            "corrupt_checkpoints": self.corrupt, "progress_json": str(self.progress_path),
            "checkpoint_storage": "full_episode_pickle" if self.store_trajectories else "result_only_json",
        }

    def _write_progress(self) -> None:
        # 缓存尚未被本轮遍历时也显示已完成记录，便于中断后直接查看。
        records = {**self.cached, **self.current}
        completed = [
            {"instance": group, "task": item.get("task_template"),
             "instance_id": item.get("instance_id"), "task_seed": item.get("task_seed"),
             "successful": successful_episode_value(item.get("is_successful"))}
            for group, item in sorted(records.items()) if _completed(item)
        ]
        write_json_atomic(self.progress_path, {
            **self.metadata(), "expected_episodes": getattr(self, "expected", None),
            "completed_count": len(completed), "completed_instances": completed,
            "retry_instances": [group for group, item in sorted(records.items()) if not _completed(item)],
        })

    def trace_id(self, episode: Mapping[str, Any]) -> str:
        # 重启后生成相同 trace_id，使 --record-traces 不重复累计 SQLite 统计。
        # 真正重试异常实例是新 attempt，不能被旧异常 trace 的 INSERT OR IGNORE 吞掉。
        return "evaluation:" + fingerprint({
            "instance": episode.get(STAMP_KEY), "attempt": episode.get(ATTEMPT_KEY),
        })
