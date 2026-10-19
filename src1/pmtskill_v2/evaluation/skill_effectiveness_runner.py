"""Real PORTS wrapper + AndroidWorld task truth for representation ablations."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import re
import time
from pathlib import Path

from ..core.environment import load_android_world_environment
from ..core.io import load_primitives, write_json_atomic
from ..core.models import ExecutionPlan, RouteStep
from ..inference.model_pool import ModelPool
from ..inference.vlm import OpenAICompatibleVLClient
from ..offline.collector import bootstrap_android_world
from ..online.executor import RoutedVLWrapper
from ..skills.evidence_miner import GUARD
from .android_world import _run_evaluation_suite, episodes_to_traces
from .compaction import compact_m3a_step_result
from .reporter import compact_result_for_storage, successful_episode_value, write_evaluation_report
from .resume import EvaluationProgress, fingerprint, model_resume_identity
from .skill_effectiveness import PROTOCOL, slug
from . import task_parameters

METRICS = "_skill_effectiveness"


def compact_result(episode):
    result = compact_result_for_storage(episode, episodes_to_traces([episode]))
    if METRICS in episode:
        result[METRICS] = dict(episode[METRICS])
    return result


def frozen_suite(suite_utils, registry, tasks, *, combinations, seed, family, environment, path):
    """Persist exact parameters before any condition, including wall-clock names."""
    spec = {"tasks": list(tasks), "combinations": combinations, "seed": seed, "family": family}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["specification"] != spec:
            raise ValueError("冻结任务参数的实验范围发生变化，请使用新目录")
    else:
        suite = suite_utils.create_suite(registry, n_task_combinations=combinations,
                                         seed=seed, tasks=list(tasks), env=environment)
        saved = {"specification": spec, "params_encoding": task_parameters.PROTOCOL, "instances": [
            {"task": name, "instance_id": i, "params": task_parameters.encode(task.params), "goal": task.goal}
            for name, instances in suite.items() for i, task in enumerate(instances)]}
    encoding = saved.get("params_encoding")
    if encoding not in {None, task_parameters.PROTOCOL}:
        raise ValueError(f"Unsupported frozen task parameter encoding: {encoding}")
    result = {}
    for row in saved["instances"]:
        params = task_parameters.decode(row["params"]) if encoding else copy.deepcopy(row["params"])
        task = registry[row["task"]](params)
        if task.goal != row["goal"]:
            raise ValueError(f"冻结任务的 goal 无法一致恢复: {row['task']}；请检查 harness 版本")
        instances = result.setdefault(row["task"], [])
        if row["instance_id"] != len(instances):
            raise ValueError("冻结任务 instance_id 顺序损坏")
        instances.append(task)
    if set(result) != set(tasks) or any(len(v) != combinations for v in result.values()):
        raise ValueError("冻结任务参数缺失或任务集合不匹配")
    if not path.exists():
        # Round-trip validation must pass BEFORE saving the shared parameters.
        write_json_atomic(path, saved)
    suite = suite_utils.Suite(result)
    suite.suite_family = family
    return suite, fingerprint(saved)


class EffectivenessWrapper(RoutedVLWrapper):
    """Keep the real PORTS prompt path; pin one model and one representation.

    Primitive condition receives the entire flat sequence, not one artificial
    primitive per model call. Macro calls are NOT counted as a single GUI step.
    """
    def __init__(self, profile, pair, condition, *, guidance_max_chars=12000, client=None, supported_apps=()):
        self.profile, self.pair, self.condition = profile, pair, condition
        self.limit = guidance_max_chars
        self.app_names = tuple(sorted(set(supported_apps)))
        self.client = client or OpenAICompatibleVLClient(profile)
        skill = pair.raw if condition == "raw" else pair.polished if condition == "polished" else None
        self.selected = copy.deepcopy(skill)
        if self.selected:
            self.selected.body = self.selected.body or self.selected.description
            if condition == "raw":
                self.selected.metadata.pop("prompt_mode", None)
            if self.selected.metadata.get("prompt_mode") != "stepwise" and len(self.selected.body) > guidance_max_chars:
                raise ValueError("技能原文超过 guidance-max-chars；不截断原文，请增大预算")
        pool = ModelPool([profile], client_factory=lambda _: self.client)
        super().__init__(pool, [self.selected] if self.selected else [], skill_body_max_chars=guidance_max_chars)
        definitions = {p.primitive_id: p for p in load_primitives()}
        flat = []
        for i, node in enumerate(pair.polished.topology.topological_nodes()):
            if node.primitive_id not in definitions:
                raise ValueError(f"未知原语: {node.primitive_id}")
            primitive = definitions[node.primitive_id]
            flat.append(f"{i + 1}. {node.primitive_id}: {primitive.description}; params="
                        + json.dumps(node.params, ensure_ascii=False, sort_keys=True))
        self.flat_guidance = "\n".join(flat)
        if len(self.flat_guidance) > guidance_max_chars:
            raise ValueError("原语序列超过 guidance-max-chars；不能截掉动作造成不等价，请增大预算")
        self.reset()

    def reset(self):
        super().reset()
        self.metrics = {"model_invocations": 0, "action_calls": 0, "summary_calls": 0,
                        "failed_calls": 0, "model_latency_seconds": 0.0,
                        "latency_seconds": 0.0, "interaction_length": 0}

    def set_goal(self, goal):
        topology = self.pair.polished.topology
        chosen_topology = self.selected.topology if self.selected else topology
        step = RouteStep("representation", self.profile.model_id,
                         self.selected.skill_id if self.selected else None,
                         chosen_topology.primitive_sequence(),
                         tuple(n.node_id for n in chosen_topology.topological_nodes()),
                         0.5, 0.0, 0.0, is_polished=self.condition == "polished")
        self.set_plan(ExecutionPlan(goal, chosen_topology, (step,), 0.0, planner_id=PROTOCOL,
                                   metadata={"task_skill_id": self.selected.skill_id if self.selected else None}))

    def predict_mm(self, text_prompt, images):
        is_action = bool(self._pending_action) if self._pending_action is not None else not any(
            marker in text_prompt.lower() for marker in ("summerize the latest step", "summarize the latest step"))
        if is_action:
            if self.condition == "primitives":
                text_prompt += "\n\nFlat primitive-level implementation (no skill text):\n" + self.flat_guidance + "\n" + GUARD
            else:
                text_prompt += "\n" + GUARD
            text_prompt += self.app_name_guidance()
        self.metrics["action_calls" if is_action else "summary_calls"] += 1
        started = time.perf_counter()
        before = self.client.http_request_count
        try:
            result = super().predict_mm(text_prompt, images)
            if result[2] is None:
                self.metrics["failed_calls"] += 1
            return result
        except Exception:
            self.metrics["failed_calls"] += 1
            raise
        finally:
            self.metrics["model_invocations"] += self.client.http_request_count - before
            self.metrics["model_latency_seconds"] += time.perf_counter() - started

    def app_name_guidance(self):
        """Identical public harness contract in every arm; no goal/answer hints."""
        text = ("\n\nAndroidWorld application-name contract (shared by all representations):\n"
                "Skill names/IDs are metadata, NOT Android app names or package IDs. "
                "For open_app, use an exact supported app name, not a skill name.\n")
        if self.app_names:
            text += "Supported names (installation not guaranteed): " + ", ".join(self.app_names) + ".\n"
            if "files" in self.app_names:
                text += "The Android file manager is named 'files'; do NOT use 'file-manager' or 'Files by Google'.\n"
            if "simple calendar pro" in self.app_names:
                text += "For Simple Calendar Pro use 'simple calendar pro', not the skill name 'calendar'.\n"
        return text


@contextlib.contextmanager
def record_task_metrics(suite_utils, wrapper):
    original = suite_utils._run_task

    def run_task(*args, **kwargs):
        wrapper.reset()
        result = original(*args, **kwargs)
        if isinstance(result, dict):
            result[METRICS] = dict(wrapper.metrics)
            if wrapper.metrics["failed_calls"]:
                result["_evaluation_episode_data_valid"] = False
                result["exception_info"] = result.get("exception_info") or "Model API failure during skill effectiveness evaluation"
        return result

    suite_utils._run_task = run_task
    try:
        yield
    finally:
        suite_utils._run_task = original


class AndroidWorldEffectivenessRunner:
    def __init__(self, config, profile, store, target, *, guidance_max_chars):
        self.config, self.profile, self.store, self.target = config, profile, store, target
        self.limit = guidance_max_chars

    def run(self, pair, condition, *, combinations, seed, max_steps, family, resume, store_trajectories):
        bootstrap_android_world(self.config.paths.android_world_root)
        from android_world import registry, suite_utils
        from android_world.agents import m3a
        from android_world.env import env_launcher

        from android_world.env import adb_utils
        supported_apps = sorted({name for pattern in getattr(adb_utils, "_PATTERN_TO_ACTIVITY", {})
                                 for name in pattern.split("|") if re.fullmatch(r"[a-zA-Z0-9 _-]+", name)})

        target = self.target / "conditions" / slug(pair.pair_id) / condition
        wrapper = EffectivenessWrapper(self.profile, pair, condition, guidance_max_chars=self.limit,
                                        supported_apps=supported_apps)
        # Force the same declared horizon in all three arms, independently of
        # the evaluator's legacy task-complexity budget.
        config = dataclasses.replace(self.config, evaluation=dataclasses.replace(self.config.evaluation, fixed_step_budget=True))
        environment = load_android_world_environment(env_launcher, config.android_world)
        try:
            items = registry.TaskRegistry().get_registry(family=family)
            missing = set(pair.tasks) - set(items)
            if missing:
                raise ValueError(f"AndroidWorld 不存在任务: {sorted(missing)}")
            suite, suite_hash = frozen_suite(suite_utils, items, pair.tasks,
                                            combinations=combinations, seed=seed, family=family,
                                            environment=environment,
                                            path=self.target / "suites" / (slug(pair.pair_id) + ".json"))
            progress = EvaluationProgress(target, {
                "protocol": PROTOCOL, "pair": pair.identity(), "condition": condition,
                "model": model_resume_identity(self.profile), "suite_hash": suite_hash,
                "max_steps": max_steps, "guidance_max_chars": self.limit,
                "supported_apps": supported_apps, "params_encoding": task_parameters.PROTOCOL,
                "android_world": dataclasses.asdict(config.android_world),
            }, resume=resume, store_trajectories=store_trajectories, result_compactor=compact_result)
            outer = self

            class Agent(m3a.M3A):
                def __init__(self):
                    super().__init__(environment, wrapper, name=f"ports-effectiveness:{condition}",
                                     wait_after_action_seconds=outer.config.android_world.wait_after_action_seconds)
                    self.current_goal = None

                def reset(self, go_home_on_reset=False):
                    wrapper.reset()
                    self.current_goal = None
                    return super().reset(go_home_on_reset)

                def step(self, goal):
                    if self.current_goal is None:
                        wrapper.set_goal(goal)
                        self.current_goal = goal
                    wrapper.begin_step()
                    started = time.perf_counter()
                    try:
                        result = super().step(goal)
                        wrapper.observe_step(result.data)
                        return compact_m3a_step_result(result)
                    finally:
                        wrapper.metrics["latency_seconds"] += time.perf_counter() - started
                        wrapper.metrics["interaction_length"] += 1

            with record_task_metrics(suite_utils, wrapper):
                episodes = _run_evaluation_suite(suite_utils, suite, Agent(), environment,
                                                 config, progress, max_steps=max_steps)
            write_evaluation_report(target, episodes, episodes_to_traces(episodes),
                                    store_trajectories=store_trajectories,
                                    metadata={"evaluation_mode": PROTOCOL, "condition": condition,
                                              "model_id": self.profile.model_id, "pair_id": pair.pair_id,
                                              "tasks": list(pair.tasks), "seed": seed, "max_steps": max_steps,
                                              "supported_apps": supported_apps, "params_encoding": task_parameters.PROTOCOL,
                                              "n_task_combinations": combinations,
                                              "suite_hash": suite_hash, "resume": progress.metadata()})
            rows = []
            for episode in episodes:
                metrics = episode.get(METRICS, {})
                valid = not episode.get("exception_info") and bool(metrics) and not metrics.get("failed_calls")
                rows.append({"pair_id": pair.pair_id, "condition": condition,
                             "model_id": self.profile.model_id,
                             "raw_skill_id": pair.raw.skill_id, "polished_skill_id": pair.polished.skill_id,
                             "task": episode["task_template"], "instance_id": episode["instance_id"],
                             "goal": episode.get("goal"), "suite_hash": suite_hash,
                             "valid": valid, "success": successful_episode_value(episode.get("is_successful")) if valid else None,
                             "error": episode.get("exception_info") or ("model API failure" if metrics.get("failed_calls") else ""),
                             "episode_wall_time_seconds": episode.get("run_time") if valid else None,
                             **metrics})
            return rows
        finally:
            environment.close()
