"""CLI orchestration, immutable inputs and isolated model deployment."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

from . import METHODS, MULTI_MODEL_METHODS, PROTOCOL, SKILL_METHODS
from .catalog import file_digest, load_catalog, source_identity
from .config import AndroidLabConfig
from .environment import docker_image_identity
from ..pmtskill_v2.core.config import load_config
from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.evaluation.resume import fingerprint, model_resume_identity


def snapshot_database(source, target):
    """SQLite backup includes WAL; never initialize or mutate the source DB."""
    source, target = Path(source).expanduser().resolve(), Path(target).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Skill database does not exist: {source}")
    if source == target:
        raise ValueError("Evaluation database snapshot must not be the source database")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".backup-tmp")
    if temporary.exists():
        raise ValueError(f"Incomplete database backup exists: {temporary}; inspect it before retrying")
    with contextlib.closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with contextlib.closing(sqlite3.connect(temporary)) as destination:
            original.backup(destination)
    os.replace(temporary, target)
    return target


def prepare(config, options, args):
    """Freeze shared tasks/library without starting Docker, GPU or paid APIs."""
    tasks = load_catalog(options, tasks=args.tasks, task_count=args.task_count, seed=args.seed)
    target = Path(args.output_dir).expanduser().resolve()
    document = {"protocol": PROTOCOL, "source_identity": source_identity(options.root),
                "tasks": tasks, "seed": args.seed, "combinations": args.combinations,
                "max_steps": args.max_steps}
    if args.combinations <= 0 or args.max_steps <= 0:
        raise ValueError("combinations/max-steps must be positive")
    catalog = target / "inputs" / "tasks.json"
    if catalog.exists() and json.loads(catalog.read_text(encoding="utf-8")) != document:
        raise ValueError("Prepared AndroidLab suite differs; use a new comparison output-dir")
    source = Path(args.skill_database).expanduser().resolve() if args.skill_database else config.paths.database
    database = target / "inputs" / "skill_library.sqlite3"
    if not args.dry_run:
        write_json_atomic(catalog, document)
        if source.is_file() and not database.exists():
            snapshot_database(source, database)
        for method in METHODS:
            (target / method / "cli_logs").mkdir(parents=True, exist_ok=True)
    return {"suite_file": str(catalog), "skill_database": str(database) if source.is_file() or database.exists() else None,
            "task_count": len(tasks), "episodes_per_method": len(tasks) * args.combinations,
            "benchmark": "AndroidLab", "docker_started": False, "model_started": False,
            "note": "Existing input snapshot is retained, never refreshed from a changing source library"}


def command_prepare(args):
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    config = load_config(args.config)
    print(json.dumps(prepare(config, AndroidLabConfig.load(args.config), args), ensure_ascii=False, indent=2))
    return 0


def _selected_suite(options, args):
    all_tasks = load_catalog(options)
    tasks = load_catalog(options, tasks=args.tasks, task_count=args.task_count, seed=args.seed)
    if args.suite_file:
        document = json.loads(Path(args.suite_file).expanduser().read_text(encoding="utf-8"))
        if document.get("protocol") != PROTOCOL or document.get("source_identity") != source_identity(options.root):
            raise ValueError("AndroidLab prepared suite/source version mismatch")
        if args.tasks or args.task_count is not None:
            if document["tasks"] != tasks:
                raise ValueError("--tasks/--task-count differ from frozen AndroidLab suite")
        by_id = {task["task_id"]: task for task in all_tasks}
        if any(by_id.get(task["task_id"]) != task for task in document["tasks"]):
            raise ValueError("Frozen AndroidLab task instructions/predicates changed")
        for name in ("seed", "combinations", "max_steps"):
            if document.get(name) != getattr(args, name):
                raise ValueError(f"--{name.replace('_', '-')} differs from prepared suite")
        tasks = document["tasks"]
    return tasks, all_tasks


def _agent_source(config, args):
    if args.method == "vdroid":
        candidates = [Path(args.vdroid_root).expanduser().resolve()] if args.vdroid_root else [
            config.paths.repo_root / "libs" / name for name in ("V-Droid", "v-droid")]
        source = next((p for p in candidates if (p / "android_world/agents/vdroid.py").is_file()), candidates[0])
        if not (source / "android_world/agents/vdroid.py").is_file():
            raise FileNotFoundError(f"V-Droid source missing: {source}")
    else:
        source = config.paths.android_world_root
        if not (source / "android_world/agents/m3a.py").is_file():
            raise FileNotFoundError(f"AndroidWorld agent source missing: {source}")
    return source


@contextlib.contextmanager
def output_lock(target):
    target.mkdir(parents=True, exist_ok=True)
    path = target / "evaluation.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Evaluation directory is locked: {path}. Check the recorded PID before removing a stale lock") from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def command_evaluate(args):
    from ..pmtskill_v2.cli import (_adapter_deployment, _evaluation_resume, _evaluation_store_trajectories,
                                  _evaluation_settings)
    from ..pmtskill_v2.skills.store import SkillStore
    config = load_config(args.config)
    options = AndroidLabConfig.load(args.config)
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    if args.combinations <= 0 or args.max_steps <= 0:
        raise ValueError("combinations/max-steps must be positive")
    if not math.isfinite(args.preparation_timeout_seconds) or args.preparation_timeout_seconds <= 0:
        raise ValueError("--preparation-timeout-seconds must be finite and positive")
    if args.record_traces:
        raise ValueError("AndroidLab evaluation never writes test trajectories into the skill library")
    args.resume = _evaluation_resume(config, args)
    args.store_trajectories = _evaluation_store_trajectories(config, args)
    if args.family != "android_lab":
        raise ValueError("evaluate-androidlab only supports --family android_lab")
    tasks, all_tasks = _selected_suite(options, args)
    if any(t["metric_type"] == "query_detect" for t in tasks):
        if not options.judge_model_id:
            raise ValueError("Answer/query tasks need [androidlab].judge_model_id; use a separate configured judge model")
        config.model(options.judge_model_id)  # Reject unknown models before GPU/Docker.
    source = _agent_source(config, args)
    target = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(target)
    # The worker changes cwd to the native benchmark for its relative imports.
    # Resolve user-owned resources before crossing that process boundary.
    for name in ("memory_file", "embedding_model_path", "skvm_cache_dir"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if args.method == "echotrail-gui":
        if not args.memory_file or not args.embedding_model_path:
            raise ValueError("EchoTrail requires --memory-file and --embedding-model-path")
        from ..baselines.echotrail import validate_memory
        validate_memory(args, check_encoder=not args.dry_run)
    if args.method == "vdroid" and min(args.vdroid_score_workers, args.vdroid_search_iterations) <= 0:
        raise ValueError("V-Droid worker/iteration counts must be positive")
    with output_lock(target) if not args.dry_run else contextlib.nullcontext():
        store = None
        database = None
        if args.method in SKILL_METHODS:
            original = Path(args.skill_database).expanduser().resolve() if args.skill_database else config.paths.database
            if not original.is_file():
                raise FileNotFoundError(f"{args.method} requires an existing --skill-database: {original}")
            if not args.dry_run:
                database = snapshot_database(original, target / "inputs" / "skill_library.sqlite3")
                store = SkillStore(database)
                store.initialize()
        paths = list(args.adapter_paths or ([args.adapter_path] if args.adapter_path else []))
        multi = args.method in MULTI_MODEL_METHODS
        if args.method == "random-skvm" and len(paths or args.model_ids or []) < 2:
            raise ValueError("random-skvm requires at least two adapter/model profiles")
        deployment = None
        if paths:
            selected_paths = paths if multi else paths[:1]
            ids = list(args.adapter_model_ids or [])
            if ids and len(ids) != len(paths):
                raise ValueError("adapter-model-ids must match adapter-paths")
            deployment, _ = _adapter_deployment(config, store, args, selected_paths, ids if multi else ids[:1])
            profiles = deployment.profiles()
            context = deployment.activate_adapters()
        else:
            ids = list(args.model_ids or ([args.model_id] if args.model_id else []))
            if not ids:
                raise ValueError("Provide --adapter-paths, --adapter-path, --model-ids, or --model-id")
            profiles = [config.model(key) for key in (ids if multi else ids[:1])]
            if len({profile.model_id for profile in profiles}) != len(profiles):
                raise ValueError("Model IDs must be distinct")
            context = contextlib.nullcontext(profiles)
        if not profiles or any(not profile.enabled for profile in profiles):
            raise ValueError("Every evaluation model must be enabled")
        source_files = [p for directory in ("android_world", "html_representation", "MCTS")
                        for p in (source / directory).rglob("*.py")]
        source_files += list(source.glob("*.py"))
        policy_root = Path(__file__).resolve().parents[1]
        policy_files = [p for directory in ("pmtskill_v2/online", "pmtskill_v2/inference", "pmtskill_v2/skills",
                                            "baselines", "baseline_skvm", "baseline_vdroid")
                        for p in (policy_root / directory).rglob("*.py")]
        identity = {"protocol": PROTOCOL, "method": args.method, "tasks": tasks,
                    "official_source": source_identity(options.root),
                    "agent_code": fingerprint([(str(p.relative_to(source)), file_digest(p))
                                               for p in sorted(source_files)]),
                    "policy_code": fingerprint([(str(p.relative_to(policy_root)), file_digest(p)) for p in sorted(policy_files)]),
                    "bridge_code": fingerprint([(p.name, file_digest(p)) for p in sorted(Path(__file__).parent.glob("*.py"))]),
                    "options": options.to_dict(), "profiles": [model_resume_identity(p) for p in profiles],
                    "routing": dataclasses.asdict(config.routing),
                    "generation": dataclasses.asdict(_evaluation_settings(config, args)),
                    "planner_model_id": args.planner_model,
                    "seed": args.seed, "combinations": args.combinations, "max_steps": args.max_steps,
                    "store_trajectories": args.store_trajectories, "include_candidates": args.include_candidates,
                    "skill_database": file_digest(database) if database else None,
                    "method_options": {key: getattr(args, key) for key in (
                        "androidworld_agent", "routing_seed", "memory_alpha", "memory_top_k", "memory_max_chars",
                        "skvm_profile_instances", "skvm_aot_pass_sets", "skvm_failure_policy", "skvm_compile_retries",
                        "vdroid_score_workers", "vdroid_search_iterations")},
                    "memory": file_digest(args.memory_file) if args.method == "echotrail-gui" else None,
                    "judge": model_resume_identity(config.model(options.judge_model_id)) if options.judge_model_id else None}
        if args.dry_run:
            print(json.dumps({"method": args.method, "tasks": [t["task_id"] for t in tasks],
                              "profiles": [p.model_id for p in profiles], "androidlab": options.to_dict(),
                              "deployment_command": deployment.build_adapter_command() if deployment else None,
                              "note": "No Docker/GPU/API work performed; image/dependency checks occur at run time"},
                             ensure_ascii=False, indent=2))
            return 0
        image_id = docker_image_identity(options.docker_image)
        identity["image_id"] = image_id
        identity_hash = fingerprint(identity)
        manifest = target / "evaluation_manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if not args.resume or old.get("identity") != identity_hash:
                raise ValueError("AndroidLab experiment identity changed; use a new output-dir (not --resume)")
        request = {"config": str(config.config_path), "profiles": [p.to_dict() for p in profiles],
                       "tasks": tasks, "all_tasks": all_tasks, "skill_database": str(database) if database else None,
                       "agent_source": str(source), "image_id": image_id, "identity": identity_hash,
                       "metadata": {"method": args.method, "benchmark": "AndroidLab", "protocol": PROTOCOL,
                                    "image_id": image_id, "source_identity": identity["official_source"],
                                    "model_ids": [p.model_id for p in profiles], "seed": args.seed,
                                    "combinations": args.combinations, "max_steps": args.max_steps}}
        request["args"] = {key: value for key, value in vars(args).items() if key != "handler"}
        write_json_atomic(target / "worker_request.json", request)
        preflight = run_worker(target / "worker_request.json", timeout=180, check=True)
        if preflight:
            raise RuntimeError("AndroidLab dependency/predicate preflight failed before model deployment; see CLI runtime.log")
        with context as running_profiles:
            request["profiles"] = [p.to_dict() for p in running_profiles]
            write_json_atomic(target / "worker_request.json", request)
            timeout = options.episode_timeout_seconds * len(tasks) * args.combinations + args.preparation_timeout_seconds
            code = run_worker(target / "worker_request.json", timeout=timeout)
        if code:
            raise RuntimeError(f"AndroidLab {args.method} worker failed (exit={code}); "
                               f"see {target / 'report.md'} and the current CLI runtime.log")
        print(json.dumps({"summary": str(target / "summary.json"), "report": str(target / "report.md")}, indent=2))
        return 0


def run_worker(request_path, *, timeout, check=False):
    """Forward worker logs and allow its finally blocks to clean owned Docker resources."""
    root = str(Path(__file__).resolve().parents[2])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "src1.androidlab.worker", str(request_path)]
    if check:
        command.append("--check")
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               start_new_session=os.name != "nt")
    def forward():
        for line in process.stdout:
            print(line, end="", flush=True)
    thread = threading.Thread(target=forward, daemon=True)
    thread.start()
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
        thread.join(timeout=5)
        process.stdout.close()


def add_arguments(parser, *, evaluation=True):
    from ..pmtskill_v2.cli import _add_common_evaluation_arguments, _add_skill_evaluation_arguments, _add_skvm_arguments
    _add_common_evaluation_arguments(parser)
    _add_skill_evaluation_arguments(parser)
    parser.set_defaults(family="android_lab")
    if evaluation:
        parser.add_argument("--method", choices=METHODS, required=True)
        sources = parser.add_mutually_exclusive_group()
        sources.add_argument("--adapter-paths", nargs="+")
        sources.add_argument("--adapter-path")
        sources.add_argument("--model-ids", nargs="+")
        sources.add_argument("--model-id")
        parser.add_argument("--adapter-model-ids", nargs="+")
        parser.add_argument("--suite-file")
        parser.add_argument("--planner-model")
        parser.add_argument("--continue-on-error", action="store_true",
                            help="保留 unavailable 并继续后续任务，最终仍以非零退出")
        parser.add_argument("--preparation-timeout-seconds", type=float, default=10800)
        _add_skvm_arguments(parser)
        parser.add_argument("--routing-seed", type=int, default=42)
        parser.add_argument("--androidworld-agent", choices=["m3a", "t3a"], default="m3a")
        parser.add_argument("--vdroid-root")
        parser.add_argument("--vdroid-score-workers", type=int, default=4)
        parser.add_argument("--vdroid-search-iterations", type=int, default=1)
        parser.add_argument("--memory-file")
        parser.add_argument("--embedding-model-path")
        parser.add_argument("--memory-alpha", type=float, default=0.5)
        parser.add_argument("--memory-top-k", type=int, default=2)
        parser.add_argument("--memory-max-chars", type=int, default=12000)
