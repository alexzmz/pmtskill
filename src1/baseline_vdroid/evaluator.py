"""Parent-process orchestration for the isolated V-Droid worker."""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.evaluation.resume import (
    validate_evaluation_directory,
    validate_resume_output,
    vdroid_resume_specification,
)
from .config import VDroidBaselineConfig
from .validation import require_available_evaluation


@dataclasses.dataclass(slots=True)
class VDroidRunOptions:
    tasks: tuple[str, ...] = ()
    task_count: int | None = None
    combinations: int = 1
    seed: int = 42
    family: str = "android_world"
    max_steps: int = 30
    fixed_task_seed: bool = False
    output_dir: Path | None = None
    model_id: str = "vdroid-8b-0323"
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    verifier_adapter: str = "V-Droid-8B-0323"
    assistant_service: str = "openai"
    assistant_model: str = "gpt-4o"
    assistant_base_url: str = "https://api.openai.com/v1"
    assistant_api_key_env: str = "OPENAI_API_KEY"
    assistant_api_version: str | None = None
    summary_mode: str = "llm"
    num_actors: int = 1
    search_iterations: int = 1
    cuda_visible_devices: str | None = None
    store_screenshots: bool = False
    emulator_setup: bool = False
    freeze_datetime: bool = True
    resume: bool = False
    store_trajectories: bool = True

    @classmethod
    def from_config(cls, config: VDroidBaselineConfig) -> "VDroidRunOptions":
        return cls(
            tasks=config.evaluation.tasks,
            task_count=config.evaluation.task_count,
            combinations=config.evaluation.combinations,
            seed=config.evaluation.seed,
            family=config.evaluation.family,
            max_steps=config.evaluation.max_steps,
            fixed_task_seed=config.evaluation.fixed_task_seed,
            model_id=config.model.model_id,
            base_model=config.model.base_model,
            verifier_adapter=config.model.verifier_adapter,
            assistant_service=config.assistant.service,
            assistant_model=config.assistant.model,
            assistant_base_url=config.assistant.base_url,
            assistant_api_key_env=config.assistant.api_key_env,
            assistant_api_version=config.assistant.api_version,
            summary_mode=config.model.summary_mode,
            num_actors=config.model.num_actors,
            search_iterations=config.model.search_iterations,
            cuda_visible_devices=config.model.cuda_visible_devices,
            store_screenshots=config.evaluation.store_screenshots,
            emulator_setup=config.device.emulator_setup,
            freeze_datetime=config.device.freeze_datetime,
            resume=config.evaluation.resume,
            store_trajectories=config.evaluation.store_trajectories,
        )

    def validate(self) -> None:
        validate_resume_output(self.resume, self.output_dir)
        if self.task_count is not None and self.task_count <= 0:
            raise ValueError("--task-count must be positive")
        if self.combinations <= 0:
            raise ValueError("--combinations must be positive")
        if self.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        if self.num_actors <= 0:
            raise ValueError("--num-actors must be positive")
        if self.search_iterations <= 0:
            raise ValueError("--search-iterations must be positive")
        if self.summary_mode not in {"llm", "rule", "skip"}:
            raise ValueError("--summary-mode must be llm, rule, or skip")


class VDroidBaselineEvaluator:
    """Runs the upstream V-Droid agent without importing it into the CLI process."""

    def __init__(self, config: VDroidBaselineConfig, options: VDroidRunOptions):
        self.config = config
        self.options = options
        self.options.validate()

    def target_dir(self) -> Path:
        if self.options.output_dir is not None:
            return self.options.output_dir.expanduser().resolve()
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        return (
            self.config.paths.output_dir
            / f"vdroid_{stamp}_{uuid.uuid4().hex[:8]}"
        ).resolve()

    def dependency_status(self) -> dict[str, bool]:
        return {
            "v_droid_source": (
                self.config.paths.v_droid_root
                / "android_world"
                / "agents"
                / "vdroid.py"
            ).is_file(),
            "ray": importlib.util.find_spec("ray") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "vllm": importlib.util.find_spec("vllm") is not None,
            "peft": importlib.util.find_spec("peft") is not None,
        }

    def request(self, target: Path) -> dict[str, Any]:
        o = self.options
        d = self.config.device
        return {
            "protocol": "vdroid-androidworld-baseline-v1",
            "config_path": str(self.config.config_path),
            "repo_root": str(self.config.paths.repo_root),
            "v_droid_root": str(self.config.paths.v_droid_root),
            "output_dir": str(target),
            "resume": o.resume,
            "store_trajectories": o.store_trajectories,
            "tasks": list(o.tasks),
            "task_count": o.task_count,
            "combinations": o.combinations,
            "seed": o.seed,
            "family": o.family,
            "max_steps": o.max_steps,
            "fixed_task_seed": o.fixed_task_seed,
            "model": {
                "model_id": o.model_id,
                "base_model": o.base_model,
                "verifier_adapter": o.verifier_adapter,
                "num_actors": o.num_actors,
                "search_iterations": o.search_iterations,
                "summary_mode": o.summary_mode,
                "cuda_visible_devices": o.cuda_visible_devices,
            },
            "assistant": {
                "service": o.assistant_service,
                "model": o.assistant_model,
                "base_url": o.assistant_base_url,
                "api_key_env": o.assistant_api_key_env,
                "api_version": o.assistant_api_version,
            },
            "device": {
                "adb_path": d.adb_path,
                "console_port": d.console_port,
                "grpc_port": d.grpc_port,
                "device_name": d.device_name,
                "emulator_setup": o.emulator_setup,
                "freeze_datetime": o.freeze_datetime,
                "verify_api_level": d.verify_api_level,
                "wait_after_action_seconds": d.wait_after_action_seconds,
            },
            "store_screenshots": o.store_screenshots and o.store_trajectories,
            "isolation": {
                "skill_store": False,
                "skill_adaptation": False,
                "dynamic_model_routing": False,
                "adapter_switching": False,
                "fixed_verifier_model": o.model_id,
            },
        }

    @staticmethod
    def worker_command(request_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src1.baseline_vdroid.worker",
            "--request",
            str(request_path),
        ]

    def dry_run(self) -> dict[str, Any]:
        target = self.target_dir()
        request_path = target / "resolved_request.json"
        return {
            "dry_run": True,
            "evaluation_mode": "vdroid",
            "worker_command": self.worker_command(request_path),
            "resolved_request": self.request(target),
            "dependency_status": self.dependency_status(),
        }

    def run(self) -> dict[str, Any]:
        target = self.target_dir()
        request = self.request(target)
        validate_evaluation_directory(
            target, vdroid_resume_specification(request), resume=self.options.resume,
        )
        target.mkdir(parents=True, exist_ok=True)
        request_path = target / "resolved_request.json"
        write_json_atomic(request_path, request)
        env = os.environ.copy()
        if self.options.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.options.cuda_visible_devices
        process = subprocess.Popen(
            self.worker_command(request_path),
            cwd=self.config.paths.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def pump(source, target_stream) -> None:
            if source is None:
                return
            try:
                for line in iter(source.readline, ""):
                    target_stream.write(line)
                    target_stream.flush()
            finally:
                source.close()

        stdout_thread = threading.Thread(
            target=pump, args=(process.stdout, sys.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=pump, args=(process.stderr, sys.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        if return_code != 0:
            raise RuntimeError(
                "V-Droid worker failed with return code "
                f"{return_code}; see the current CLI runtime.log/errors.log"
            )
        result_path = target / "worker_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"V-Droid worker completed without {result_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        # Exit code zero alone does not guarantee any task was evaluated.
        require_available_evaluation(result.get("summary"))
        result["worker_return_code"] = return_code
        result["resolved_request"] = str(request_path)
        return result


def apply_cli_overrides(
    options: VDroidRunOptions,
    *,
    tasks: Sequence[str] | None = None,
    **overrides: Any,
) -> VDroidRunOptions:
    values = {
        key: value
        for key, value in overrides.items()
        if value is not None and hasattr(options, key)
    }
    if tasks is not None:
        values["tasks"] = tuple(dict.fromkeys(str(item) for item in tasks))
    return dataclasses.replace(options, **values)

