"""用教师 VL 模型在 AndroidWorld 上生成监督轨迹。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..core.config import ProjectConfig
from ..inference.vlm import OpenAICompatibleVLClient


def bootstrap_android_world(android_world_root: str | Path) -> None:
    """把本仓库的 AndroidWorld 加入导入路径，不修改 site-packages。"""

    root = str(Path(android_world_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(slots=True)
class CollectionResult:
    output_dir: Path
    episodes: int
    successes: int
    tasks: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "episodes": self.episodes,
            "successes": self.successes,
            "tasks": self.tasks,
            "success_rate": self.successes / self.episodes if self.episodes else 0.0,
        }


def collect_teacher_trajectories(
    config: ProjectConfig,
    *,
    tasks: Sequence[str] | None = None,
    n_task_combinations: int = 1,
    seed: int = 42,
    family: str = "android_world",
    output_dir: Path | None = None,
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
        results = suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer_lib.IncrementalCheckpointer(str(trajectory_dir)),
            demo_mode=False,
            max_n_steps_override=config.android_world.max_steps,
            stop_on_task_success=config.android_world.stop_on_task_success,
        )
    finally:
        environment.close()

    successes = sum(bool(item.get("is_successful", False)) for item in results)
    return CollectionResult(
        output_dir=trajectory_dir,
        episodes=len(results),
        successes=successes,
        tasks=list(tasks or suite.keys()),
    )

