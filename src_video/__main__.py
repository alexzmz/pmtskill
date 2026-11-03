"""Run one AndroidWorld episode with passive recording, or rerender a capture."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src1.pmtskill_v2.core.config import load_config, tomllib

from .recorder import EpisodeRecorder, write_json


def _positive(value, key):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return value


def _integer(value, key):
    _positive(value, key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def load_settings(path: Path, method_override=None):
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    def local(value):
        return (path.parent / Path(value).expanduser()).resolve()

    evaluation = raw["evaluation"]
    video = raw.get("recording", {})
    config_path = local(evaluation["config"])
    config = load_config(config_path)
    serial = f"emulator-{config.android_world.console_port}"
    if video.get("emulator_serial", serial) != serial:
        raise ValueError(f"recording.emulator_serial must match the evaluator ({serial})")
    method = method_override or evaluation.get("method", "standalone")
    if method not in {"standalone", "ports"}:
        raise ValueError("evaluation.method must be standalone or ports")
    task = evaluation["task"]
    if not isinstance(task, str) or not task.isidentifier():
        raise ValueError("evaluation.task must be a single AndroidWorld task ID")
    adapters = evaluation["adapter_paths"]
    ids = evaluation.get("adapter_model_ids", [])
    if not isinstance(adapters, list) or not adapters or not all(isinstance(x, str) and x for x in adapters):
        raise ValueError("evaluation.adapter_paths must be a nonempty array of paths")
    if not isinstance(ids, list) or (ids and len(ids) != len(adapters)) or not all(
            isinstance(x, str) and x for x in ids):
        raise ValueError("evaluation.adapter_model_ids must be empty or match adapter_paths")
    if method == "standalone" and len(adapters) != 1:
        raise ValueError("standalone requires exactly one adapter path")
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation.adapter_model_ids must be distinct")

    output = local(video.get("output_dir", "../src1/runtime/videos"))
    run = output / f"{task}_{method}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    argv = ["--config", str(config_path), "--log-dir", str(run / "cli_logs")]
    adapter_args = [str(local(x)) for x in adapters]
    if method == "standalone":
        argv += ["evaluate-standalone", "--adapter-path", adapter_args[0]]
        if ids:
            argv += ["--model-id", ids[0]]
    else:
        argv += ["evaluate-pmtskill", "--adapter-paths", *adapter_args, "--no-record-traces"]
        if ids:
            argv += ["--adapter-model-ids", *ids]
        if evaluation.get("skill_database"):
            argv += ["--skill-database", str(local(evaluation["skill_database"]))]
        if evaluation.get("planner_model"):
            argv += ["--planner-model", evaluation["planner_model"]]
    port = _integer(evaluation.get("deploy_port", 8002), "deploy_port")
    if port > 65535:
        raise ValueError("deploy_port must be at most 65535")
    argv += [
        "--tasks", task, "--family", "android_world", "--combinations", "1",
        "--seed", str(evaluation.get("seed", 42)),
        "--max-steps", str(_integer(evaluation.get("max_steps", 30), "max_steps")),
        "--cuda-visible-devices", str(evaluation.get("cuda_visible_devices", "0")),
        "--deploy-port", str(port), "--infer-backend", "vllm",
        "--no-require-training-state", "--no-resume", "--store-trajectories",
        "--output-dir", str(run / "evaluation"),
    ]
    return {
        "run": run, "argv": argv, "serial": serial, "adb": config.android_world.adb_path,
        "capture_fps": _positive(video.get("capture_fps", 4), "capture_fps"),
        "video_fps": _positive(video.get("video_fps", 12), "video_fps"),
        "command_timeout": _positive(video.get("adb_timeout_seconds", 5), "adb_timeout_seconds"),
        "ffmpeg": video.get("ffmpeg", "ffmpeg"),
    }


def render_capture(directory: Path, *, fps, ffmpeg):
    from .render import render_episode

    data = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    result = render_episode(directory, frames=data["frames"], steps=data["steps"],
                            controls=data["controls"], duration=data["duration"], fps=fps, ffmpeg=ffmpeg)
    write_json(directory / "videos.json", result)
    return result


def check_encoder(executable):
    ffmpeg = shutil.which(executable)
    if not ffmpeg:
        raise RuntimeError(f"FFmpeg executable not found: {executable}")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                              check=True, timeout=10).stdout
    if not any(line.split()[1:2] == [b"libx264"] for line in encoders.splitlines()):
        raise RuntimeError("This FFmpeg build does not provide the libx264 encoder")
    return ffmpeg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="recording TOML")
    mode.add_argument("--render-existing", type=Path, metavar="EPISODE_DIR")
    parser.add_argument("--method", choices=("standalone", "ports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.render_existing:
            if args.dry_run or args.method:
                parser.error("--method and --dry-run require --config")
            render_capture(args.render_existing.resolve(), fps=12, ffmpeg="ffmpeg")
            return 0
        settings = load_settings(args.config, args.method)
        if args.dry_run:
            from src1.pmtskill_v2.cli import build_parser

            build_parser().parse_args(settings["argv"])
            print(json.dumps(settings, default=str, ensure_ascii=False, indent=2))
            return 0

        from src1.pmtskill_v2.cli import main as evaluate
        from src1.pmtskill_v2.evaluation.observation import observe_android_world

        ffmpeg = check_encoder(settings["ffmpeg"])
        run = settings["run"]
        recorder = EpisodeRecorder(run, adb=settings["adb"], serial=settings["serial"],
                                   capture_fps=settings["capture_fps"],
                                   command_timeout=settings["command_timeout"])
        recorder.preflight()
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "request.json", {**settings, "run": str(run)})
        print(f"[src_video] recording {settings['serial']} to {run}", flush=True)
        evaluation_code = 1
        errors = []
        try:
            with observe_android_world(recorder):
                evaluation_code = evaluate(settings["argv"])
        except KeyboardInterrupt:
            evaluation_code = 130
        except Exception as exc:
            errors.append(f"evaluation: {exc}")
        finally:
            if recorder.active is not None:
                recorder.end(None, error="Evaluation ended without a benchmark outcome")

        rendered = []
        for directory in recorder.episodes:
            try:
                render_capture(directory, fps=settings["video_fps"], ffmpeg=ffmpeg)
                rendered.append(str(directory))
            except Exception as exc:
                errors.append(f"{directory}: {exc}")
        errors.extend(recorder.errors)
        if not recorder.episodes:
            errors.append("No AndroidWorld episode ran; inspect cli_logs")
        summary = {"evaluation_exit_code": evaluation_code, "episodes_rendered": rendered, "errors": errors}
        write_json(run / "result.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return evaluation_code or int(bool(errors))
    except Exception as exc:
        print(f"[src_video] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
