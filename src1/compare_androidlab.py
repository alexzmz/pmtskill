"""Prepare and run the existing eight methods against one frozen AndroidLab suite."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .androidlab import METHODS, MULTI_MODEL_METHODS, SKILL_METHODS
from .androidlab.config import AndroidLabConfig
from .androidlab.driver import prepare
from .pmtskill_v2.core.config import load_config
from .pmtskill_v2.core.io import write_json_atomic


def build_parser():
    from .batch_evaluate import DEFAULT_ADAPTERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-paths", nargs="+", default=DEFAULT_ADAPTERS)
    parser.add_argument("--adapter-model-ids", nargs="+", default=["adapter_control01", "adapter_crossapp01", "adapter_all02"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--skill-database")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--combinations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--deploy-port", type=int, default=8002)
    parser.add_argument("--base-model-path")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--planner-model")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--store-trajectories", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--androidworld-agent", choices=["m3a", "t3a"], default="m3a")
    parser.add_argument("--memory-file")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--memory-alpha", type=float, default=.5)
    parser.add_argument("--memory-top-k", type=int, default=2)
    parser.add_argument("--memory-max-chars", type=int, default=12000)
    parser.add_argument("--routing-seed", type=int, default=42)
    parser.add_argument("--vdroid-root")
    parser.add_argument("--skvm-command")
    return parser


def build_commands(args, inputs):
    if len(args.adapter_paths) != len(args.adapter_model_ids):
        raise ValueError("adapter-paths/model-ids length mismatch")
    commands = {}
    target = Path(args.output_dir).expanduser().resolve()
    for method in dict.fromkeys(args.methods):
        common = [sys.executable, "-m", "src1", "--config", str(Path(args.config).expanduser().resolve()),
                  "--log-dir", str(target / method / "cli_logs"), "evaluate-androidlab", "--method", method,
                  "--suite-file", inputs["suite_file"], "--output-dir", str(target / method),
                  "--adapter-paths", *(args.adapter_paths if method in MULTI_MODEL_METHODS else args.adapter_paths[:1]),
                  "--adapter-model-ids", *(args.adapter_model_ids if method in MULTI_MODEL_METHODS else args.adapter_model_ids[:1]),
                  "--combinations", str(args.combinations), "--seed", str(args.seed), "--max-steps", str(args.max_steps),
                  "--cuda-visible-devices", args.cuda_visible_devices, "--deploy-port", str(args.deploy_port),
                  "--infer-backend", "vllm", "--no-require-training-state", "--no-record-traces",
                  "--resume" if args.resume else "--no-resume",
                  "--store-trajectories" if args.store_trajectories else "--no-store-trajectories"]
        if method in SKILL_METHODS:
            if not inputs["skill_database"]:
                raise ValueError(f"{method} requires an existing --skill-database")
            common += ["--skill-database", inputs["skill_database"]]
        for name in ("base_model_path", "max_model_len", "gpu_memory_utilization"):
            if getattr(args, name) is not None:
                common += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.continue_on_error:
            common += ["--continue-on-error"]
        if args.include_candidates:
            common += ["--include-candidates"]
        if method == "ports" and args.planner_model:
            common += ["--planner-model", args.planner_model]
        if method in {"skvm", "random-skvm"} and args.skvm_command:
            common += ["--skvm-command", args.skvm_command]
        if method == "random-skvm":
            common += ["--routing-seed", str(args.routing_seed)]
        if method == "androidworld":
            common += ["--androidworld-agent", args.androidworld_agent]
        if method == "vdroid" and args.vdroid_root:
            common += ["--vdroid-root", args.vdroid_root]
        if method == "echotrail-gui":
            if not args.memory_file or not args.embedding_model_path:
                if not args.prepare_only and not args.dry_run:
                    raise ValueError("EchoTrail needs --memory-file/--embedding-model-path; prepare inputs first if not ready")
            common += ["--memory-file", args.memory_file or "REPLACE_WITH_MEMORY_JSON",
                       "--embedding-model-path", args.embedding_model_path or "REPLACE_WITH_EMBEDDING_DIRECTORY",
                       "--memory-alpha", str(args.memory_alpha), "--memory-top-k", str(args.memory_top_k),
                       "--memory-max-chars", str(args.memory_max_chars)]
        commands[method] = common
    return commands


def comparison(target, methods, codes):
    rows, sets = [], {}
    for method in methods:
        summary_path = target / method / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        available = codes.get(method) == 0 and summary.get("evaluation_available") is True
        rows.append({"method": method, "exit_code": codes.get(method),
                     "success_rate": summary.get("success_rate") if available else None,
                     "valid_episodes": summary.get("valid_episodes", 0) if codes.get(method) is not None else 0,
                     "expected_episodes": summary.get("expected_episodes"),
                     "mean_steps": summary.get("mean_steps") if available else None,
                     "mean_latency_ms": summary.get("mean_latency_ms") if available else None})
        episodes = []
        for path in (target / method / "episodes").glob("*.json") if available else ():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "completed":
                episodes.append(row)
        sets[method] = {(r["task_id"], r["repeat"]): r for r in episodes}
    common = set.intersection(*(set(values) for values in sets.values())) if sets else set()
    for row in rows:
        row["matched_valid_episodes"] = len(common)
        row["matched_subset_sr"] = (sum(sets[row["method"]][key]["successful"] for key in common) / len(common)
                                    if common else None)
    write_json_atomic(target / "comparison.json", {"methods": rows, "matched_subset_is_not_full_benchmark": True})
    with (target / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    percent = lambda value: "N/A" if value is None else f"{value:.2%}"
    lines = ["# AndroidLab comparison", "", "Missing/failed methods remain N/A; no fabricated zero scores.", "",
             "| Method | Exit | Valid | Full-suite SR | Matched-subset SR |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['exit_code']} | {row['valid_episodes']} | "
                     f"{percent(row['success_rate'])} | {percent(row['matched_subset_sr'])} |")
    lines += ["", f"Matched subset size: {len(common)}. This is NOT full-benchmark performance.",
              "Standalone and androidworld=m3a intentionally use the same official agent."]
    (target / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        inputs = prepare(config, AndroidLabConfig.load(args.config), args)
        commands = build_commands(args, inputs)
        if args.dry_run:
            print(json.dumps(commands, ensure_ascii=False, indent=2))
            return 0
        target = Path(args.output_dir).expanduser().resolve()
        write_json_atomic(target / "commands.json", commands)
        (target / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" +
                                            "\n\n".join(shlex.join(c) for c in commands.values()) + "\n", encoding="utf-8")
        if args.prepare_only:
            print(json.dumps(inputs, ensure_ascii=False, indent=2))
            return 0
        codes = {}
        for method, command in commands.items():
            result = subprocess.run(command, cwd=config.paths.repo_root)
            codes[method] = result.returncode
            comparison(target, commands, codes)
            if result.returncode and not args.continue_on_error:
                break
        return 0 if len(codes) == len(commands) and all(code == 0 for code in codes.values()) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"AndroidLab comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
