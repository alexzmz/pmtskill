"""Isolated native V-Droid runner.

Run only through :mod:`src1.baseline_vdroid.evaluator`.  Isolation is required
because V-Droid vendors a different ``android_world`` package than PORTS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _configure_assistant_environment(assistant: Mapping[str, Any]) -> None:
    service = str(assistant["service"])
    key_name = str(assistant.get("api_key_env") or "")
    key_value = os.environ.get(key_name, "") if key_name else ""
    if service in {"openai", "azure_openai"} and not key_value:
        raise RuntimeError(
            f"assistant API key environment variable is not set: {key_name}"
        )
    prefix = {
        "openai": "OPENAI",
        "trapi": "TRAPI",
        "azure_openai": "AZURE_OPENAI",
    }[service]
    os.environ[f"{prefix}_MODEL_NAME"] = str(assistant["model"])
    if assistant.get("base_url"):
        os.environ[f"{prefix}_ENDPOINT"] = str(assistant["base_url"])
    if key_value:
        os.environ[f"{prefix}_API_KEY"] = key_value
    if assistant.get("api_version"):
        os.environ[f"{prefix}_API_VERSION"] = str(assistant["api_version"])


def _instrument_vdroid_agent(agent_class: type, *, max_steps: int) -> None:
    """Record verifier statistics and enforce the shared episode horizon."""

    original_score = agent_class.score_by_batch
    original_complete = agent_class._action_completion
    original_search = agent_class.search

    def score_by_batch(self, node, available_actions, memory, child_node_info, *args, **kwargs):
        children = original_score(
            self,
            node,
            available_actions,
            memory,
            child_node_info,
            *args,
            **kwargs,
        )
        for child in children:
            if child is None:
                continue
            child.node_info = dict(child.node_info or {})
            child.node_info["action_output"] = child.action
            child.node_info["_vdroid"] = {
                "candidate_count": len(available_actions),
                "selected_score": float(child.score),
                "selected_action_before_completion": child.action,
                "action_completed": False,
            }
        return children

    def action_completion(self, node):
        before = node.action
        completed = original_complete(self, node)
        completed.node_info = dict(completed.node_info or {})
        metadata = dict(completed.node_info.get("_vdroid", {}))
        metadata["action_completed"] = completed.action != before
        metadata["selected_action_after_completion"] = completed.action
        completed.node_info["_vdroid"] = metadata
        completed.node_info["action_output"] = completed.action
        return completed

    def search(self):
        # V-Droid's vendored episode_runner expands the requested budget by 10
        # before calling search(). Clamp both internal limits back to the
        # experiment's declared horizon so PORTS and V-Droid are comparable.
        self.depth_limit = min(int(self.depth_limit), max_steps)
        self.explore_step_count_limit = min(
            int(self.explore_step_count_limit), max_steps
        )
        return original_search(self)

    agent_class.score_by_batch = score_by_batch
    agent_class._action_completion = action_completion
    agent_class.search = search


def _compact_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in episode.items()
        if key not in {"tree", "screen_config"}
    }
    raw_data = episode.get("episode_data", {})
    if isinstance(raw_data, Mapping):
        keep = {
            "step_number",
            "action_output",
            "summary",
            "summary_prompt",
            "_vdroid",
        }
        result["episode_data"] = {
            key: value for key, value in raw_data.items() if key in keep
        }
    else:
        result["episode_data"] = {}
    return result


class _CompactCheckpointer:
    def __init__(self, delegate: Any):
        self.delegate = delegate

    def save_episodes(self, task_episodes, task_name):
        self.delegate.save_episodes(
            [_compact_episode(item) for item in task_episodes], task_name
        )

    def load(self, fields=None):
        return self.delegate.load(fields=fields)


def _select_tasks(registry_items: Mapping[str, Any], request: Mapping[str, Any]):
    explicit = [str(item) for item in request.get("tasks", ())]
    available = sorted(registry_items)
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"V-Droid registry does not contain tasks: {missing}")
        return list(dict.fromkeys(explicit))
    count = request.get("task_count")
    if count is None:
        return None
    count = min(int(count), len(available))
    return sorted(random.Random(int(request["seed"])).sample(available, count))


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    v_droid_root = Path(str(request["v_droid_root"])).resolve()
    output_dir = Path(str(request["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = dict(request["model"])
    assistant = dict(request["assistant"])
    device = dict(request["device"])

    if model.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])
    _configure_assistant_environment(assistant)

    os.chdir(v_droid_root)
    sys.path.insert(0, str(v_droid_root))

    # These imports must happen only after the V-Droid root is first on sys.path.
    import ray
    from android_world import registry, suite_utils
    from android_world.agents import vdroid
    from android_world.env import env_launcher

    from src1.baseline_vdroid.reporting import compact_vdroid_result_for_storage, write_vdroid_report
    from src1.pmtskill_v2.core.io import write_json_atomic
    from src1.pmtskill_v2.evaluation.resume import EvaluationProgress, vdroid_resume_specification

    progress = EvaluationProgress(
        output_dir, vdroid_resume_specification(request), resume=bool(request.get("resume", False)),
        store_trajectories=bool(request.get("store_trajectories", True)),
        result_compactor=lambda episode: compact_vdroid_result_for_storage(episode, model_id=str(model["model_id"])),
    )

    _instrument_vdroid_agent(
        vdroid.VDroidAgent, max_steps=int(request["max_steps"])
    )
    # The upstream runner derives the horizon from task complexity.  Override
    # only this budget so V-Droid and PORTS use the exact same experimental cap.
    suite_utils._allocate_step_budget = lambda _complexity: int(request["max_steps"])

    environment = env_launcher.load_and_setup_env(
        console_port=int(device["console_port"]),
        emulator_setup=bool(device["emulator_setup"]),
        freeze_datetime=bool(device["freeze_datetime"]),
        adb_path=str(device["adb_path"]),
        grpc_port=int(device["grpc_port"]),
        device_name=str(device["device_name"]),
        family=str(request["family"]),
    )
    try:
        if bool(device.get("verify_api_level", True)):
            env_launcher.verify_api_level(environment)
        task_registry = registry.TaskRegistry()
        registry_items = task_registry.get_registry(family=str(request["family"]))
        selected_tasks = _select_tasks(registry_items, request)
        suite = suite_utils.create_suite(
            registry_items,
            n_task_combinations=int(request["combinations"]),
            seed=int(request["seed"]),
            tasks=selected_tasks,
            use_identical_params=bool(request["fixed_task_seed"]),
        )
        suite.suite_family = str(request["family"])
        agent = vdroid.VDroidAgent(
            environment,
            str(model["base_model"]),
            adapter_dir=str(model["verifier_adapter"]),
            llm_name=str(assistant["model"]),
            service_name=str(assistant["service"]),
            n_iters=int(model["search_iterations"]),
            family=str(request["family"]),
            summary_mode=str(model["summary_mode"]),
            num_actors=int(model["num_actors"]),
            if_store_screen=bool(request.get("store_screenshots", False)),
            wait_after_action_seconds=float(device["wait_after_action_seconds"]),
        )
        # Upstream VDroidAgent does not forward wait_after_action_seconds to its
        # base transition_pause property, so set it explicitly for a fair cap.
        agent.transition_pause = float(device["wait_after_action_seconds"])
        agent.name = f"vdroid:{model['model_id']}"

        checkpoint_dir = output_dir / "checkpoints"
        checkpointer = _CompactCheckpointer(
            progress
        )
        with progress.watch(suite_utils, suite):
            suite_utils.run(
                suite,
                agent,
                checkpointer=checkpointer,
                demo_mode=False,
                save_name=output_dir.name,
            )
        episodes = progress.episodes()
        metadata = {
            "evaluation_mode": "vdroid",
            "report_title": "V-Droid AndroidWorld Baseline",
            "protocol": request["protocol"],
            "comparison_protocol": "same-tasks-seed-combinations-and-step-cap",
            "family": request["family"],
            "tasks": selected_tasks or "all",
            "n_task_combinations": request["combinations"],
            "seed": request["seed"],
            "max_steps": request["max_steps"],
            "model_id": model["model_id"],
            "served_model": model["base_model"],
            "verifier_adapter": model["verifier_adapter"],
            "assistant_model": assistant["model"],
            "assistant_service": assistant["service"],
            "summary_mode": model["summary_mode"],
            "search_iterations": model["search_iterations"],
            "num_actors": model["num_actors"],
            "skill_database": None,
            "skill_adaptation": False,
            "dynamic_model_routing": False,
            "adapter_switching": False,
            "episode_storage": "compact_vdroid_without_screenshots_or_tree",
            "resume": progress.metadata(),
        }
        artifacts = write_vdroid_report(
            output_dir,
            episodes,
            metadata=metadata,
            store_trajectories=bool(request.get("store_trajectories", True)),
        )
        result = {
            "evaluation_mode": "vdroid",
            "summary": artifacts.summary,
            "summary_json": str(artifacts.summary_json),
            "report_markdown": str(artifacts.report_markdown),
            "traces_jsonl": str(artifacts.traces_jsonl) if artifacts.traces_jsonl else None,
            "checkpoint_dir": str(checkpoint_dir),
            "skill_database": None,
            "adapters": [],
            "fixed_models": {
                "verifier": model["model_id"],
                "assistant": assistant["model"],
            },
        }
        write_json_atomic(output_dir / "worker_result.json", result)
        return result
    finally:
        try:
            environment.close()
        finally:
            if ray.is_initialized():
                ray.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="isolated native V-Droid worker")
    parser.add_argument("--request", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    run_request(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

