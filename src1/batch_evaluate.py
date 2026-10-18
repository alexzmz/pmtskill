"""Serial five-method AndroidWorld experiment. Run from the repository root.

python -m src1.batch_evaluate --config src1/config.local_control.toml \
  --cuda-visible-devices 3 --deploy-port 8005 --output-dir src1/runtime/evaluations/compare_five

Baselines use the FIRST adapter only; PORTS uses the entire three-adapter pool.
No GPU, API or emulator work is performed by --dry-run.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from .pmtskill_v2.core.config import load_config
from .pmtskill_v2.core.io import write_json_atomic
from .pmtskill_v2.evaluation.capabilities import CAPABILITY_ORDER
from .baseline_skvm.protocol import SKVM_TOOL_PROTOCOL, output_for_protocol


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("standalone", "simple-skill", "skvm", "vdroid", "ports")
COMMANDS = dict(zip(METHODS, ("evaluate-standalone", "evaluate-simple-skills",
                            "evaluate-skvm", "evaluate-vdroid-qwen", "evaluate-pmtskill")))
DEFAULT_ADAPTERS = (
    "src1/runtime/checkpoints/adapter_control01",
    "src1/runtime/checkpoints/adapter_crossapp01",
    "src1/runtime/checkpoints/deprecated/adapter_all02",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tasks(values):
    return list(dict.fromkeys(part.strip() for value in values or ()
                              for part in value.split(",") if part.strip()))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="五种方法共用同一个 PORTS TOML")
    parser.add_argument("--output-dir", required=True, help="固定批次目录；再次运行即续跑")
    parser.add_argument("--skill-database", help="默认读取 TOML；首次运行创建一致性快照")
    parser.add_argument("--adapter-paths", nargs=3, default=DEFAULT_ADAPTERS)
    parser.add_argument("--adapter-model-ids", nargs=3,
                        default=("adapter_control01", "adapter_crossapp01", "adapter_all02"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--combinations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--family", default="android_world")
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--deploy-port", type=int, default=8005)
    parser.add_argument("--base-model-path")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--include-candidates", action="store_true", help="默认仅 active polished（不自动晋升技能）")
    parser.add_argument("--store-trajectories", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true", help="单个方法失败后继续；错误结果仍显示 N/A")
    parser.add_argument("--dry-run", action="store_true", help="只打印五个命令，不写文件、不启动服务")
    parser.add_argument("--skvm-command")
    parser.add_argument("--skvm-prepare", choices=("auto", "reuse", "force"), default="auto")
    parser.add_argument("--skvm-profile-instances", type=int, default=3)
    parser.add_argument("--vdroid-root", help="默认查找 libs/V-Droid 或 libs/v-droid")
    parser.add_argument("--vdroid-score-workers", type=int, default=4)
    parser.add_argument("--vdroid-search-iterations", type=int, default=1)
    return parser


def build_commands(args, adapters=None, tasks=None):
    """Pure command construction, also usable on a laptop without checkpoints."""
    output = Path(args.output_dir).expanduser().resolve()
    adapters = list(adapters or args.adapter_paths)
    common = ["--combinations", str(args.combinations), "--seed", str(args.seed),
              "--max-steps", str(args.max_steps), "--family", args.family,
              "--cuda-visible-devices", args.cuda_visible_devices,
              "--deploy-port", str(args.deploy_port), "--infer-backend", "vllm",
              "--no-require-training-state", "--resume" if args.resume else "--no-resume",
              "--store-trajectories" if args.store_trajectories else "--no-store-trajectories"]
    for name in ("base_model_path", "max_model_len", "gpu_memory_utilization"):
        value = getattr(args, name)
        if value is not None:
            common += ["--" + name.replace("_", "-"), str(value)]
    selected = tasks if tasks is not None else parse_tasks(args.tasks)
    if selected:
        common += ["--tasks", *selected]
    elif args.task_count is not None:
        common += ["--task-count", str(args.task_count)]
    jobs = []
    for method in dict.fromkeys(args.methods):
        folder = output / method.replace("-", "_")
        if method == "skvm":
            folder = output_for_protocol(folder)
        command = [sys.executable, "-m", "src1", "--config", str(Path(args.config).resolve()),
                   "--log-dir", str(folder / "cli_logs"), COMMANDS[method], *common,
                   "--output-dir", str(folder)]
        if method == "ports":
            command += ["--adapter-paths", *map(str, adapters),
                        "--adapter-model-ids", *args.adapter_model_ids]
        else:
            command += ["--adapter-path", str(adapters[0]), "--model-id", args.adapter_model_ids[0]]
        if method in {"simple-skill", "skvm", "ports"}:
            command += ["--skill-database", str(folder / "skill_library.sqlite3"), "--no-record-traces"]
            if args.include_candidates:
                command += ["--include-candidates"]
        if method == "skvm":
            command += ["--skvm-prepare", args.skvm_prepare,
                        "--skvm-profile-instances", str(args.skvm_profile_instances)]
            if args.skvm_command:
                command += ["--skvm-command", args.skvm_command]
        if method == "vdroid":
            command += ["--vdroid-score-workers", str(args.vdroid_score_workers),
                        "--vdroid-search-iterations", str(args.vdroid_search_iterations)]
            if args.vdroid_root:
                command += ["--vdroid-root", str(Path(args.vdroid_root).expanduser().resolve())]
        jobs.append({"method": method, "output_dir": str(folder), "command": command})
    return jobs


def snapshot_database(source, target):
    """SQLite backup includes a live WAL, unlike copying only the main file."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"技能库不存在: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        with contextlib.closing(sqlite3.connect(target)) as dest:
            src.backup(dest)


def validate_dependencies(args, config):
    if "skvm" in args.methods and args.skvm_prepare != "reuse":
        if args.skvm_command:
            available = Path(args.skvm_command).is_file() or shutil.which(args.skvm_command)
        else:
            available = shutil.which("skvm") or (
                shutil.which("bun") and (config.paths.repo_root / "libs/skvm/src/index.ts").is_file())
        if not available:
            raise FileNotFoundError("SKVM 需要 skvm 或 Bun，请安装或指定 --skvm-command")
    if "vdroid" in args.methods:
        if importlib.util.find_spec("ray") is None:
            raise RuntimeError("V-Droid-Qwen 需要 Ray；请先在评测 Python 环境安装 V-Droid 依赖")


def read_summary(job, expected):
    path = Path(job["output_dir"]) / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        complete = (summary.get("evaluation_available") is True
                    and summary.get("episodes_total") == expected
                    and summary.get("episodes_evaluated") == expected)
        if job["method"] == "skvm":
            metadata = summary.get("metadata")
            complete = complete and isinstance(metadata, dict) and metadata.get("skvm_tool_protocol") == SKVM_TOOL_PROTOCOL
        return summary, complete
    except (OSError, ValueError, TypeError, AttributeError):
        return {}, False


def comparison_rows(jobs, state, expected):
    rows = []
    for job in jobs:
        result = state.get(job["method"], {})
        summary, complete = read_summary(job, expected)
        valid = result.get("status") == "completed" and complete
        status = result.get("status", "pending")
        if status == "completed" and not complete:
            status = "pending"
        row = {"method": "vdroid-qwen-adapted" if job["method"] == "vdroid" else job["method"],
               "status": status, "expected_episodes": expected,
               "episodes_evaluated": summary.get("episodes_evaluated"),
               "successes": summary.get("successes") if valid else None,
               "micro_sr": summary.get("success_rate_micro") if valid else None,
               "macro_sr": summary.get("success_rate_macro") if valid else None,
               "average_steps": summary.get("average_steps") if valid else None,
               "average_run_time_seconds": summary.get("average_run_time_seconds") if valid else None,
               "average_model_switches": summary.get("average_model_switches") if valid else None,
               "output_dir": job["output_dir"], "error": result.get("error", "")}
        for category in CAPABILITY_ORDER:
            row[category] = (summary.get("capability_distribution", {}).get(category, {}).get("score")
                             if valid else None)
        rows.append(row)
    baseline = next((row["micro_sr"] for row in rows if row["method"] == "standalone"), None)
    for row in rows:
        row["delta_micro_sr_pp"] = (100 * (row["micro_sr"] - baseline)
                                    if row["micro_sr"] is not None and baseline is not None else None)
    return rows


def write_comparison(output, jobs, state, expected):
    rows = comparison_rows(jobs, state, expected)
    write_json_atomic(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    def number(value):
        return "N/A" if value is None else f"{value:.2f}"
    lines = ["# AndroidWorld 五方法对比", "",
             "四个 baseline 固定第一个 adapter；PORTS 使用三个 adapter 池。",
             "V-Droid-Qwen 是改造版：保留原生候选枚举/搜索，改用 Qwen Yes/No 概率评分，"
             "不是原论文的训练 verifier。各方法独立启动服务，顺序使用同一 GPU 和模拟器。",
             "失败、未完成或基础设施不可用的实验标为 N/A，不当作 SR=0。", "",
             "| 方法 | 状态 | 有效/预期 | Micro SR | Macro SR | Δ Micro (pp) | Reasoning | Coordination | Manipulation |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['status']} | {row['episodes_evaluated'] or 0}/{expected} | "
                     f"{percent(row['micro_sr'])} | {percent(row['macro_sr'])} | "
                     f"{number(row['delta_micro_sr_pp'])} | {number(row['reasoning'])} | "
                     f"{number(row['coordination'])} | {number(row['manipulation'])} |")
    lines += ["", "技能库使用批次开始时的冻结快照，三个技能方法各用独立副本。"
              "默认不保存完整轨迹，保留报告、轻量 checkpoint、日志和 SKVM 编译/能力评估产物。",
              "PORTS 的增益包含多 adapter 池的影响，不是等模型资源的纯算法消融。",
              "V-Droid 使用上游独立 AndroidWorld fork；同名任务/seed 不保证两版本的参数实现完全相同，"
              "应同时检查各自 checkpoint 中的 goal/task_seed。", "", "## 详细报告", ""]
    for job in jobs:
        lines.append(f"- [{job['method']}]({Path(job['output_dir']).name}/report.md)")
        if state.get(job["method"], {}).get("error"):
            lines.append(f"  - {state[job['method']]['error']}")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n$ " + shlex.join(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1, start_new_session=os.name != "nt")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()


def run(args):
    if min(args.combinations, args.max_steps, args.skvm_profile_instances,
           args.vdroid_score_workers, args.vdroid_search_iterations) <= 0:
        raise ValueError("组合数、步数和 profiling/worker 参数必须为正数")
    if len(set(args.adapter_model_ids)) != 3:
        raise ValueError("三个 adapter model_id 必须不同")
    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("--task-count 必须为正数")
    # Preview intentionally does not require any Linux adapter paths to exist.
    if args.dry_run:
        jobs = build_commands(args)
        print("DRY RUN: baselines=adapter_control01; PORTS=three-adapter pool")
        for job in jobs:
            print(shlex.join(job["command"]))
        return 0
    from .pmtskill_v2.cli import build_parser as cli_parser, _adapter_deployment
    from .pmtskill_v2.evaluation.android_world import sample_android_world_tasks
    from .pmtskill_v2.evaluation.resume import model_resume_identity

    config = load_config(args.config)
    validate_dependencies(args, config)
    preview = build_commands(args)
    ports = next((item for item in preview if item["method"] == "ports"), None)
    # Record real checkpoint identities and check them before every method.
    # Keep original adapter paths: these also form part of the existing offline
    # model_variant_id, so replacing them with checkpoint paths loses estimates.
    preflight = ports or preview[0]
    resolved_args = cli_parser().parse_args(preflight["command"][3:])
    deployment, _ = _adapter_deployment(config, None, resolved_args,
                                        args.adapter_paths, args.adapter_model_ids)
    profiles = deployment.profiles()
    tasks = sample_android_world_tasks(config, tasks=parse_tasks(args.tasks),
                                      task_count=args.task_count, seed=args.seed, family=args.family)
    output = Path(args.output_dir).expanduser().resolve()
    jobs = build_commands(args, tasks=tasks)
    expected = len(tasks) * args.combinations
    spec = {"protocol": "five-method-fixed-first-v1", "config": str(config.config_path),
            "config_sha256": file_sha256(config.config_path), "tasks": tasks,
            "models": [model_resume_identity(profile) for profile in profiles],
            "methods": [job["method"] for job in jobs], "combinations": args.combinations,
            "seed": args.seed, "max_steps": args.max_steps, "family": args.family,
            "include_candidates": args.include_candidates, "store_trajectories": args.store_trajectories,
            "deployment": {name: getattr(args, name) for name in
                           ("deploy_port", "base_model_path", "max_model_len", "gpu_memory_utilization")},
            "skvm": {"prepare": args.skvm_prepare, "instances": args.skvm_profile_instances,
                     "command": args.skvm_command},
            "vdroid": {"source": args.vdroid_root, "workers": args.vdroid_score_workers,
                       "iterations": args.vdroid_search_iterations}}
    manifest_path = output / "batch_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or manifest["specification"] != spec:
            raise ValueError("批次配置/权重已变化或关闭 resume；请使用新 output-dir，不能混合结果")
        database = output / "inputs" / "skill_library.sqlite3"
        if file_sha256(database) != manifest["skill_database_sha256"]:
            raise ValueError("批次冻结的技能库被修改；请恢复快照或使用新 output-dir")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("批次目录非空且没有 batch_manifest.json；请使用新的目录")
        output.mkdir(parents=True, exist_ok=True)
        database = output / "inputs" / "skill_library.sqlite3"
        source = Path(args.skill_database).expanduser().resolve() if args.skill_database else config.paths.database
        snapshot_database(source, database)
        manifest = {"specification": spec, "skill_database_source": str(source),
                    "skill_database_sha256": file_sha256(database),
                    "created_at": dt.datetime.now().isoformat(), "expected_episodes_per_method": expected}
        write_json_atomic(manifest_path, manifest)
    for job in jobs:
        if job["method"] in {"simple-skill", "skvm", "ports"}:
            copy = Path(job["output_dir"]) / "skill_library.sqlite3"
            if not copy.exists():
                snapshot_database(database, copy)
    write_json_atomic(output / "commands.json", jobs)
    (output / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n"
        + "\n".join(shlex.join(job["command"]) for job in jobs) + "\n", encoding="utf-8")
    state_path = output / "batch_progress.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    write_comparison(output, jobs, state, expected)
    # Read-only/model-free CLI checks for ALL methods before starting the first
    # expensive episode. Failure remains visible and no method is mislabelled.
    for job in jobs:
        _, complete = read_summary(job, expected)
        if state.get(job["method"], {}).get("status") == "completed" and complete:
            continue
        if run_command([*job["command"], "--dry-run"], output / "preflight.log") != 0:
            state[job["method"]] = {"status": "preflight_failed", "error": "入口预检失败，见 preflight.log"}
            write_json_atomic(state_path, state)
            write_comparison(output, jobs, state, expected)
            return 1
    failed = False
    for job in jobs:
        name = job["method"]
        _, complete = read_summary(job, expected)
        if state.get(name, {}).get("status") == "completed" and complete:
            print(f"[SKIP] {name}: 已完成")
            continue
        state[name] = {"status": "running", "started_at": dt.datetime.now().isoformat()}
        write_json_atomic(state_path, state)
        write_comparison(output, jobs, state, expected)
        try:
            current_deployment, _ = _adapter_deployment(
                config, None, resolved_args, args.adapter_paths, args.adapter_model_ids)
            current_models = [model_resume_identity(profile) for profile in current_deployment.profiles()]
            if current_models != spec["models"]:
                raise ValueError("批次期间 adapter checkpoint/权重发生变化；停止以避免混合结果")
            code = run_command(job["command"], Path(job["output_dir"]) / "console.log")
            _, complete = read_summary(job, expected)
            if code == 0 and complete:
                state[name].update(status="completed", return_code=code)
            else:
                failed = True
                state[name].update(status="failed" if code else "incomplete", return_code=code,
                                   error=f"退出码 {code}；有效实例不足或子进程失败，见 {name}/console.log")
        except KeyboardInterrupt:
            state[name].update(status="interrupted", error="用户中断；再次运行同一命令可续跑")
            raise
        except Exception as exc:
            failed = True
            state[name].update(status="failed", error=str(exc))
        finally:
            state[name]["updated_at"] = dt.datetime.now().isoformat()
            write_json_atomic(state_path, state)
            write_comparison(output, jobs, state, expected)
        if failed and not args.continue_on_error:
            break
    print(f"汇总报告: {output / 'comparison.md'}", flush=True)
    return int(failed)


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Batch evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
