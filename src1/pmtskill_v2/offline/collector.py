"""用教师 VL 模型在 AndroidWorld 上生成监督轨迹。"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..core.config import ProjectConfig
from ..inference.vlm import OpenAICompatibleVLClient
from .reporting import summarize_collection


HARD_EPISODE_STEP_LIMIT = 50


def bootstrap_android_world(android_world_root: str | Path) -> None:
    """把本仓库的 AndroidWorld 加入导入路径，不修改 site-packages。"""

    root = str(Path(android_world_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_episode_step_limit(configured: int, override: int | None = None) -> int:
    """解析采集步数，并保证任何 episode 都不会超过硬上限 50。"""

    if override is not None:
        if override <= 0:
            raise ValueError("--max-steps 必须是正整数")
        return min(override, HARD_EPISODE_STEP_LIMIT)
    if configured <= 0:
        return HARD_EPISODE_STEP_LIMIT
    return min(configured, HARD_EPISODE_STEP_LIMIT)


def _supports_keyword(function: Any, name: str) -> bool:
    """兼容是否原生支持 max_n_steps_override 的不同 AndroidWorld 版本。"""

    parameters = inspect.signature(function).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


@contextmanager
def enforce_episode_step_limit(suite_utils: Any, step_limit: int) -> Iterator[None]:
    """在 AndroidWorld episode_runner 外层施加不可绕过的步数上限。"""

    runner_module = suite_utils.episode_runner
    original = runner_module.run_episode

    @functools.wraps(original)
    def bounded_run_episode(*args: Any, **kwargs: Any):
        positional = list(args)
        if "max_n_steps" in kwargs:
            requested = kwargs["max_n_steps"]
            kwargs["max_n_steps"] = min(max(1, int(requested)), step_limit)
        elif len(positional) >= 3:
            positional[2] = min(max(1, int(positional[2])), step_limit)
        else:
            kwargs["max_n_steps"] = step_limit
        result = original(*positional, **kwargs)
        step_data = getattr(result, "step_data", None)
        step_numbers = (
            step_data.get("step_number", []) if isinstance(step_data, dict) else []
        )
        if len(step_numbers) >= step_limit and not bool(getattr(result, "done", False)):
            aux_data = getattr(result, "aux_data", None)
            aux_data = dict(aux_data) if isinstance(aux_data, dict) else {}
            aux_data.update(
                {
                    "collector_termination_reason": "max_steps",
                    "collector_step_limit": step_limit,
                }
            )
            result.aux_data = aux_data
        return result

    runner_module.run_episode = bounded_run_episode
    try:
        yield
    finally:
        runner_module.run_episode = original


def _successful_episode(value: Any) -> bool:
    """只把有限且大于 0.5 的 AndroidWorld reward 计为成功。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.5


def _episode_length(value: Any) -> int:
    """把异常 episode 中的 NaN 长度安全地归零。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric) or numeric <= 0:
        return 0
    return int(numeric)


@dataclass(slots=True)
class CollectionResult:
    output_dir: Path
    episodes: int
    successes: int
    tasks: list[str]
    episode_step_limit: int = HARD_EPISODE_STEP_LIMIT
    episodes_at_step_limit: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "episodes": self.episodes,
            "successes": self.successes,
            "tasks": self.tasks,
            "success_rate": self.successes / self.episodes if self.episodes else 0.0,
            "episode_step_limit": self.episode_step_limit,
            "episodes_at_step_limit": self.episodes_at_step_limit,
            "summary": self.summary,
        }


def collect_teacher_trajectories(
    config: ProjectConfig,
    *,
    tasks: Sequence[str] | None = None,
    n_task_combinations: int = 1,
    seed: int = 42,
    family: str = "android_world",
    output_dir: Path | None = None,
    max_steps: int | None = None,
) -> CollectionResult:
    """启动 AndroidWorld suite，用教师 M3A 逐 episode 写 ``.pkl.gz``。

    该函数要求用户已启动 emulator 和教师模型服务。每个任务完成后立即 checkpoint，
    中断后再次使用同一目录会自动续跑。
    """

    bootstrap_android_world(config.paths.android_world_root)
    from android_world import checkpointer as checkpointer_lib
    from android_world import registry, suite_utils
    from android_world.agents import m3a
    from android_world.env import env_launcher

    episode_step_limit = resolve_episode_step_limit(
        config.android_world.max_steps, max_steps
    )
    teacher = OpenAICompatibleVLClient(config.model(config.offline.teacher_model_id))
    trajectory_dir = (output_dir or config.offline.trajectory_dir).resolve()
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    environment = env_launcher.load_and_setup_env(
        console_port=config.android_world.console_port,
        emulator_setup=config.android_world.emulator_setup,
        adb_path=config.android_world.adb_path,
    )
    try:
        task_registry = registry.TaskRegistry()
        suite = suite_utils.create_suite(
            task_registry.get_registry(family=family),
            n_task_combinations=n_task_combinations,
            seed=seed,
            tasks=list(tasks) if tasks else None,
            env=environment,
        )
        suite.suite_family = family
        agent = m3a.M3A(
            environment,
            teacher,
            name=f"teacher:{teacher.model_id}",
            wait_after_action_seconds=config.android_world.wait_after_action_seconds,
        )
        run_options: dict[str, Any] = {
            "checkpointer": checkpointer_lib.IncrementalCheckpointer(
                str(trajectory_dir)
            ),
            "demo_mode": False,
        }
        if _supports_keyword(suite_utils.run, "max_n_steps_override"):
            run_options["max_n_steps_override"] = episode_step_limit
        if _supports_keyword(suite_utils.run, "stop_on_task_success"):
            run_options["stop_on_task_success"] = (
                config.android_world.stop_on_task_success
            )
        with enforce_episode_step_limit(suite_utils, episode_step_limit):
            results = suite_utils.run(suite, agent, **run_options)
    finally:
        environment.close()

    successes = sum(
        _successful_episode(item.get("is_successful", False)) for item in results
    )
    episodes_at_step_limit = sum(
        _episode_length(item.get("episode_length", 0)) >= episode_step_limit
        for item in results
    )
    # Checkpointer 中保存了完整 step_data，可用于最终的 primitive 调用统计。
    # 读取失败不应让已经完成的 AndroidWorld 采集整体失败，因此退化为空明细。
    try:
        full_episodes = checkpointer_lib.IncrementalCheckpointer(
            str(trajectory_dir)
        ).load()
    except Exception:
        logging.warning(
            "无法读取完整 checkpoint，collect 的 per_primitive 汇总将为空。",
            exc_info=True,
        )
        full_episodes = []
    summary = summarize_collection(
        results,
        full_episodes=full_episodes,
        episode_step_limit=episode_step_limit,
    )
    return CollectionResult(
        output_dir=trajectory_dir,
        episodes=len(results),
        successes=successes,
        tasks=list(tasks or suite.keys()),
        episode_step_limit=episode_step_limit,
        episodes_at_step_limit=episodes_at_step_limit,
        summary=summary,
    )

