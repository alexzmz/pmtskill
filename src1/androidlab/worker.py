"""Isolated benchmark process (V-Droid vendors its own android_world package)."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import PROTOCOL
from .avd import AvdDevice, verify_identity
from .environment import (AndroidLabEnvironment, DockerDevice, InfrastructureError,
                          JudgeError, StepBudgetReached, bridge_adb_helpers)
from .reporting import write_report
from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.core.models import ModelProfile
from ..pmtskill_v2.evaluation.resume import fingerprint


class EpisodeTimeout(BaseException):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("AndroidLab evaluation requires POSIX SIGALRM (Linux/macOS); Windows supports prepare/inspect/dry-run")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise EpisodeTimeout(f"Episode exceeded {seconds}s")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def install_answer_judge(profile):
    """Official answer-comparison prompt, configured transport, explicit errors.

    Upstream catches all exceptions in check_answer and turns API outages into
    False. Only that error boundary is changed; task ground truths remain native.
    """
    from evaluation import task as native
    from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
    client = OpenAICompatibleVLClient(profile) if profile else None
    def check_answer(self, line):
        def fail(message, cause=None):
            error = JudgeError(message)
            # Some native task subclasses have their own bare except around
            # check_answer. The environment inspects this latch after judge().
            self._ports_answer_judge_error = error
            raise error from cause
        parsed = line["parsed_action"]
        if parsed.get("action") != "finish" and parsed.get("type") != "finish":
            return False
        if self.final_ground_truth is None:
            return False
        if client is None:
            fail("An AndroidLab answer task requires [androidlab].judge_model_id")
        answer = parsed.get("kwargs", {}).get("message", parsed.get("input", ""))
        prompt = ("You need to judge the model answer is True or False based on Standard Answer we provided. "
                  "You should whether answer [True] or [False]. \n\nQuestion: " + line["target"]
                  + "\n\nModel Answer: " + str(answer) + "\n\nStandard Answer: " + str(self.final_ground_truth))
        try:
            text = client.generate(prompt, temperature=0.001, max_tokens=512).text.strip()
        except Exception as exc:
            fail(f"AndroidLab answer judge API failed: {exc}", exc)
        if "True" in text and "False" not in text:
            return True
        if "False" in text and "True" not in text:
            return False
        fail("AndroidLab answer judge did not produce an unambiguous True/False")
    native.SingleTask.check_answer = check_answer


def episode_key(task, repeat):
    return f"{task['task_id']}__{repeat:03d}"


def empty_episode(task, repeat):
    return {"task_id": task["task_id"], "repeat": repeat, "app": task["app"],
            "capability": task["capability"], "status": "pending", "successful": None,
            "steps": 0, "duration_ms": 0, "switch_count": 0, "model_calls": None,
            "error": None}


def run_episodes(tasks, repeats, target, identity, run_one, *, resume, continue_on_error, metadata):
    """Shared, testable checkpoint loop; successful AND unsuccessful tasks resume."""
    target = Path(target)
    manifest = target / "evaluation_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("identity") != identity or not resume:
            raise ValueError("AndroidLab evaluation inputs/protocol changed, or resume disabled; use a new output-dir")
    elif any((target / "episodes").glob("*.json")):
        raise ValueError("Episode checkpoints exist without a manifest; refusing unsafe resume")
    else:
        write_json_atomic(manifest, {"protocol": PROTOCOL, "identity": identity, "metadata": metadata})
    rows = [empty_episode(task, repeat) for task in tasks for repeat in range(repeats)]
    for row in rows:
        key = episode_key(row, row["repeat"])
        path = target / "episodes" / f"{key}.json"
        if resume and path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("identity") != identity or old.get("task_id") != row["task_id"] or old.get("repeat") != row["repeat"]:
                    raise ValueError(f"Checkpoint identity mismatch: {path}")
                if old.get("status") == "completed" and isinstance(old.get("successful"), bool):
                    row.update(old)
            except json.JSONDecodeError:
                pass  # Atomic-write interruption/corrupt file is retried, not trusted.
    write_report(target, rows, len(rows), metadata)
    by_id = {task["task_id"]: task for task in tasks}
    for index, row in enumerate(rows):
        if row["status"] == "completed":
            print(f"[AndroidLab] resume {index + 1}/{len(rows)} {row['task_id']} repeat={row['repeat']}", flush=True)
            continue
        error = None
        started = time.perf_counter()
        try:
            result = run_one(by_id[row["task_id"]], row["repeat"])
            if not isinstance(result.get("successful"), bool):
                raise JudgeError("Episode returned no Boolean official result")
            row.update(result, status="completed", error=None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (Exception, EpisodeTimeout) as exc:
            error = exc
            row.update(status="unavailable", successful=None, error=f"{type(exc).__name__}: {exc}")
        row.update(identity=identity, total_episode_ms=(time.perf_counter() - started) * 1000)
        write_json_atomic(target / "episodes" / (episode_key(row, row["repeat"]) + ".json"), row)
        summary = write_report(target, rows, len(rows), metadata)
        print(f"[AndroidLab] {index + 1}/{len(rows)} {row['task_id']} status={row['status']} "
              f"successful={row['successful']} valid={summary['valid_episodes']}", flush=True)
        if error is not None and not continue_on_error:
            raise error
    return write_report(target, rows, len(rows), metadata)


def bootstrap(request, options):
    # Native task modules use top-level evaluation/utils_mobile, AW agents use
    # android_world. V-Droid must be selected BEFORE either AW agent is imported.
    lab = str(options.root)
    sys.path.insert(0, lab)
    source = Path(request["agent_source"])
    sys.path.insert(0, str(source))
    os.chdir(options.root)
    from android_world.env import interface  # noqa: F401 -- verify selected agent dependencies
    from utils_mobile.xml_tool import UIXMLTree  # noqa: F401
    import numpy as np
    np.random.seed(request["args"]["seed"])
    random.seed(request["args"]["seed"])


def preflight(request):
    from .config import AndroidLabConfig
    if os.name != "posix" or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Run AndroidLab evaluation on Linux/macOS; Windows supports prepare/inspect/dry-run")
    options = AndroidLabConfig.load(request["config"])
    if options.backend == "avd":
        verify_identity(request["device_identity"])
    bootstrap(request, options)
    for task in request["tasks"]:
        module = importlib.import_module(task["metric_module"])
        if task["task_id"] not in module.function_map:
            raise ValueError(f"Official AndroidLab predicate is missing: {task['task_id']}")
    if request["args"]["method"] == "vdroid":
        from android_world.agents import vdroid  # noqa: F401
        import ray  # noqa: F401
    else:
        from android_world.agents import m3a, t3a  # noqa: F401
    from ..pmtskill_v2.core.config import load_config
    config = load_config(request["config"])
    if options.judge_model_id and any(task["metric_type"] == "query_detect" for task in request["tasks"]):
        profile = config.model(options.judge_model_id)
        if profile.api_key_env and not os.environ.get(profile.api_key_env):
            raise RuntimeError(f"AndroidLab judge API key environment variable is missing: {profile.api_key_env}")
    print("[AndroidLab] native task/agent dependencies checked; no model or device started", flush=True)


def make_device(options, request, log_dir):
    identity = request["device_identity"]
    if options.backend != identity["backend"]:
        raise ValueError("AndroidLab backend changed after preflight")
    if options.backend == "avd":
        return AvdDevice(options, identity, log_dir)
    return DockerDevice(options, identity["image_id"])


def run_request(request):
    from .agents import AgentPolicy
    from .config import AndroidLabConfig
    from ..pmtskill_v2.core.config import load_config
    from ..pmtskill_v2.skills.store import SkillStore
    config = load_config(request["config"])
    options = AndroidLabConfig.load(request["config"])
    args = argparse.Namespace(**request["args"])
    target = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(target)
    bootstrap(request, options)
    profiles = [ModelProfile.from_dict(value) for value in request["profiles"]]
    judge = config.model(options.judge_model_id) if options.judge_model_id else None
    install_answer_judge(judge)
    store = None
    if request.get("skill_database"):
        store = SkillStore(Path(request["skill_database"]))
        store.initialize()
    apps = {task["app"].replace(" ", "").casefold(): task["package"] for task in request["all_tasks"]}
    # Exact package names are accepted too; no fuzzy match to unrelated AW apps.
    apps.update({package.casefold(): package for package in list(apps.values())})
    app_names = sorted({task["app"] for task in request["all_tasks"]})
    args.lab_wait_after_action_seconds = options.wait_after_action_seconds
    completed = any(p.is_file() for p in (target / "episodes").glob("*.json"))
    preparation = target / "preparation_identity.json"
    if completed and args.method in {"skvm", "random-skvm"}:
        if args.skvm_prepare == "force" or not preparation.is_file():
            raise ValueError("Existing AndroidLab SKVM episodes require frozen variants; use a new output-dir")
    policy = AgentPolicy(config, store, profiles, args, target, app_names)
    artifacts = [(p.relative_to(target).as_posix(), p.read_text(encoding="utf-8"))
                 for p in sorted((target / "models").rglob("variant-identity.json"))]
    prepared = fingerprint(artifacts)
    if completed and artifacts and json.loads(preparation.read_text(encoding="utf-8"))["identity"] != prepared:
        raise ValueError("SKVM variants changed; refusing mixed AndroidLab resume")
    write_json_atomic(preparation, {"identity": prepared})
    metadata = {**request["metadata"], **policy.metadata, "answer_judge_model_id": options.judge_model_id,
                "evaluation_truth": "official-AndroidLab-predicates-last-eligible-page",
                "task_ground_truth_available_to_agent": False,
                "model_calls_scope": "agent generation calls; V-Droid remote verifier calls not included"}

    def run_one(task, repeat):
        # A resumed/retried episode gets its own stable random stream.
        seed = int(fingerprint([args.seed, task["task_id"], repeat])[:8], 16)
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        key = episode_key(task, repeat)
        with time_limit(options.episode_timeout_seconds):
            with make_device(options, request, target / "avd_logs" / key) as device:
                device.initialize_task(task)
                metric_type = importlib.import_module(task["metric_module"]).function_map[task["task_id"]]
                metric = metric_type(argparse.Namespace(judge_model="configured", api_key=None))
                with tempfile.TemporaryDirectory(prefix="pmtskill-androidlab-") as scratch:
                    work = target / "trajectories" / key / ("attempt-" + uuid.uuid4().hex[:12]) if args.store_trajectories else Path(scratch)
                    environment = AndroidLabEnvironment(device, task, options, work, metric, args.max_steps,
                                                         retain=args.store_trajectories)
                    environment.app_packages = apps
                    before = policy.stats()
                    records_before = len(policy.contract.records) if policy.contract else 0
                    started = time.perf_counter()
                    agent = None
                    steps = 0
                    try:
                        with bridge_adb_helpers():
                            agent = policy.make_agent(environment, key)
                            agent.reset(False)
                            agent.set_max_steps(args.max_steps)
                            if args.method == "vdroid":
                                try:
                                    _, output = agent.search()
                                    for entry in output:
                                        if entry.get("action_output_json") is not None:
                                            environment.finalize_step(entry)
                                except StepBudgetReached:
                                    pass
                                steps = environment.actions
                            else:
                                for steps in range(1, args.max_steps + 1):
                                    environment.begin_step()
                                    policy.begin_step(environment.goal)
                                    result = agent.step(environment.goal)
                                    data = getattr(result, "data", {})
                                    policy.observe_step(data)
                                    environment.finalize_step(data)
                                    if result.done:
                                        break
                            successful = environment.finish()
                            duration_ms = max(0.0, (time.perf_counter() - started) * 1000 - environment.judge_time_ms)
                    finally:
                        if agent:
                            policy.close_agent(agent)
                    after = policy.stats()
                    records = policy.contract.records[records_before:] if policy.contract else []
                    actions = [record for record in records if not record["is_summary"] and record["model_id"]]
                    model_ids = [record["model_id"] for record in actions]
                    from collections import Counter
                    result = {"successful": successful, "steps": environment.actions, "gui_actions": environment.actions,
                              "decision_steps": steps if args.method != "vdroid" else None,
                              "duration_ms": duration_ms, "judge_time_ms": environment.judge_time_ms,
                              "switch_count": sum(a != b for a, b in zip(model_ids, model_ids[1:])),
                              "model_usage": dict(Counter(model_ids)),
                              "skill_usage": dict(Counter(r["skill_id"] for r in actions if r["skill_id"])),
                              "model_calls": after["calls"] - before["calls"] if after["calls"] is not None else None,
                              "official_result": environment.final_result, "judge_calls": environment.judge_calls}
                    if args.store_trajectories:
                        result["trajectory_dir"] = str(work)
                    if policy.plan:
                        result["skill_score_sources"] = dict(Counter(policy.plan.metadata.get("skill_score_sources", {}).values()))
                    # Logs needed by the next step live in the original agent,
                    # but no images/history escape into compact checkpoints.
                    return result
    try:
        return run_episodes(request["tasks"], args.combinations, target, request["identity"], run_one,
                            resume=args.resume, continue_on_error=args.continue_on_error, metadata=metadata)
    finally:
        if args.method == "vdroid":
            import ray
            if ray.is_initialized():
                ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if args.check:
        preflight(request)
        return 0
    def terminate(*unused):
        raise KeyboardInterrupt("AndroidLab worker terminated; cleaning up its owned device resources")
    signal.signal(signal.SIGTERM, terminate)
    result = run_request(request)
    return 0 if result["evaluation_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
