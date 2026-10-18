"""A shared boundary for additional agents, not a replacement task evaluator."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from ..pmtskill_v2.core.environment import load_android_world_environment
from ..pmtskill_v2.evaluation.android_world import (
    _close_environment, _evaluation_progress, _run_evaluation_suite, episodes_to_traces,
)
from ..pmtskill_v2.evaluation.compaction import compact_m3a_step_result
from ..pmtskill_v2.evaluation.recovery import ensure_valid_evaluation_episodes
from ..pmtskill_v2.evaluation.reporter import write_evaluation_report
from ..pmtskill_v2.evaluation.resume import fingerprint
from ..pmtskill_v2.offline.collector import bootstrap_android_world


@contextlib.contextmanager
def episode_context(suite_utils, suite, client):
    """Bind stochastic agents to an instance, independent of skipped episodes.

    This hook is outside the common resume/recovery guards. Only executed task
    attempts reach it; a retried instance gets the same initial random stream.
    It never changes the benchmark's global random generator or task params.
    """
    original = suite_utils._run_task
    identities = {
        id(task): fingerprint({"task": task.name, "instance_id": index,
                               "goal": task.goal, "params": getattr(task, "params", {})})
        for instances in suite.values() for index, task in enumerate(instances)
    }

    def run_task(*args, **kwargs):
        task = args[0] if args else kwargs["task"]
        if hasattr(client, "start_episode"):
            client.start_episode(task.goal, identities[id(task)])
        return original(*args, **kwargs)

    suite_utils._run_task = run_task
    try:
        yield
    finally:
        suite_utils._run_task = original


def run_gui_baseline(config, profiles, args, tasks, *, protocol, prepare_client,
                     agent_kind="m3a", skills=(), store=None, extra=None, metadata=None):
    """Keep original agents/actions, task truth, bounds, recovery and reports.

    prepare_client runs AFTER immutable experiment identity validation and before
    emulator task initialization. It returns a client and preparation metadata.
    """
    if agent_kind not in {"m3a", "t3a"}:
        raise ValueError("agent_kind must be m3a or t3a")
    target = Path(args.output_dir).expanduser().resolve()
    progress = _evaluation_progress(
        target, config, mode=protocol, profiles=profiles, tasks=tasks,
        combinations=args.combinations, seed=args.seed, family=args.family,
        max_steps=args.max_steps, resume=bool(args.resume), skills=skills, store=store,
        store_trajectories=args.store_trajectories, extra=dict(extra or {}),
    )
    client, prepared = prepare_client(target)
    bootstrap_android_world(config.paths.android_world_root)
    from android_world import registry, suite_utils
    from android_world.env import env_launcher
    if agent_kind == "m3a":
        from android_world.agents import m3a
        OfficialAgent = m3a.M3A
    else:
        from android_world.agents import t3a
        OfficialAgent = t3a.T3A

    class Agent(OfficialAgent):
        def reset(self, *positional, **kwargs):
            if hasattr(client, "reset"):
                client.reset()
            return super().reset(*positional, **kwargs)

        def step(self, goal):
            if hasattr(client, "begin_step"):
                client.begin_step(goal)
            result = super().step(goal)
            if hasattr(client, "observe_step"):
                client.observe_step(getattr(result, "data", {}))
            return compact_m3a_step_result(result)

    environment = load_android_world_environment(env_launcher, config.android_world)
    try:
        suite = suite_utils.create_suite(
            registry.TaskRegistry().get_registry(family=args.family),
            n_task_combinations=args.combinations, seed=args.seed,
            tasks=list(tasks) if tasks else None, env=environment,
        )
        suite.suite_family = args.family
        options = {"name": protocol}
        if agent_kind == "m3a":
            options["wait_after_action_seconds"] = config.android_world.wait_after_action_seconds
        agent = Agent(environment, client, **options)
        with episode_context(suite_utils, suite, client):
            episodes = _run_evaluation_suite(
                suite_utils, suite, agent, environment, config, progress, max_steps=args.max_steps,
            )
        ensure_valid_evaluation_episodes(episodes, expected_episodes=sum(map(len, suite.values())))
    finally:
        _close_environment(environment, tolerate_failure=sys.exc_info()[0] is not None)
    return write_evaluation_report(
        target, episodes, episodes_to_traces(episodes), store_trajectories=args.store_trajectories,
        metadata={
            "evaluation_mode": protocol, "official_agent": agent_kind,
            "model_ids": [p.model_id for p in profiles],
            "evaluation_checkpoints": {p.model_id: p.metadata.get("evaluation_checkpoint") for p in profiles},
            "android_world_connection": {"console_port": config.android_world.console_port,
                                         "grpc_port": config.android_world.grpc_port},
            "tasks": tasks or "all", "family": args.family, "n_task_combinations": args.combinations,
            "seed": args.seed, "max_steps": args.max_steps, "resume": progress.metadata(),
            **dict(metadata or {}), **prepared,
            "runtime": client.runtime_stats() if hasattr(client, "runtime_stats") else {},
        },
    )
