"""为训练期 AndroidWorld 评测临时部署学生基座或 LoRA checkpoint。"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from ..core.config import ProjectConfig, TrainingEvaluationConfig
from ..core.models import ModelProfile


class MSSwiftEvaluationDeployment:
    """使用 vendored ms-swift 启停一个 OpenAI-compatible 临时服务。

    每次 activation 只服务一个权重版本。训练进程退出后才启动部署进程，因此
    不会让训练和 AndroidWorld 评测同时争用同一块 GPU。
    """

    def __init__(
        self,
        config: ProjectConfig,
        settings: TrainingEvaluationConfig,
        template_profile: ModelProfile,
    ):
        self.config = config
        self.settings = settings
        self.template_profile = template_profile

    @property
    def swift_cli(self) -> Path:
        return self.config.paths.ms_swift_root / "swift" / "cli" / "main.py"

    def served_model_name(self, checkpoint: Path | None) -> str:
        suffix = "base" if checkpoint is None else checkpoint.name
        return f"pmtskill-{self.template_profile.model_id}-{suffix}"

    def build_command(self, checkpoint: Path | None) -> list[str]:
        """生成可直接审计/复现的 ``swift deploy`` 命令。"""

        if not self.swift_cli.is_file():
            raise FileNotFoundError(f"ms-swift CLI 不存在: {self.swift_cli}")
        command = [sys.executable, str(self.swift_cli), "deploy"]
        if checkpoint is None:
            command.extend(("--model", self.config.offline.student_model_path))
        else:
            command.extend(("--adapters", str(checkpoint.resolve())))
        command.extend(
            (
                "--infer_backend",
                self.settings.infer_backend,
                "--host",
                self.settings.deploy_host,
                "--port",
                str(self.settings.deploy_port),
                "--served_model_name",
                self.served_model_name(checkpoint),
                "--max_new_tokens",
                str(self.settings.max_new_tokens),
            )
        )
        command.extend(self.settings.deploy_extra_args)
        # 放在 extra args 之后，确保正式 TOML 字段是最终生效值。
        command.extend(
            (
                "--vllm_max_model_len",
                str(self.settings.max_model_len),
                "--vllm_gpu_memory_utilization",
                str(self.settings.gpu_memory_utilization),
            )
        )
        return command

    def profile(self, checkpoint: Path | None) -> ModelProfile:
        """返回指向当前临时服务、可被 M3A 使用的模型画像。"""

        return dataclasses.replace(
            self.template_profile,
            served_model=self.served_model_name(checkpoint),
            base_url=(
                f"http://{self.settings.deploy_host}:"
                f"{self.settings.deploy_port}/v1"
            ),
            # LoRA 已由 swift deploy 加载并成为 served_model；这里不能再给请求附加
            # adapter metadata，否则部分 vLLM OpenAI 端点会把未知字段判为 422。
            adapter=None,
            api_key_env=None,
            enabled=True,
            metadata={
                **self.template_profile.metadata,
                "evaluation_checkpoint": (
                    str(checkpoint.resolve()) if checkpoint is not None else None
                ),
            },
        )

    def build_environment(self) -> dict[str, str]:
        """构造评测服务环境；GPU 可见性与训练进程完全独立。"""

        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.config.paths.ms_swift_root) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        environment.setdefault("MAX_PIXELS", str(self.config.offline.max_pixels))
        if self.settings.cuda_visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = (
                self.settings.cuda_visible_devices
            )
        return environment

    def _assert_port_available(self) -> None:
        """提前拒绝端口冲突，避免 ms-swift 自动换端口后客户端一直空等。"""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(
                (self.settings.deploy_host, self.settings.deploy_port)
            ) == 0:
                raise RuntimeError(
                    "训练评测部署端口已被占用: "
                    f"{self.settings.deploy_host}:{self.settings.deploy_port}"
                )

    def _wait_until_ready(self, process: subprocess.Popen[str]) -> None:
        endpoint = (
            f"http://{self.settings.deploy_host}:"
            f"{self.settings.deploy_port}/v1/models"
        )
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"ms-swift deploy 提前退出，return_code={code}")
            try:
                with urllib.request.urlopen(
                    endpoint,
                    timeout=max(1.0, self.settings.startup_poll_seconds),
                ) as response:
                    if 200 <= response.status < 300:
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            time.sleep(max(0.2, self.settings.startup_poll_seconds))
        raise TimeoutError(
            f"等待评测模型服务超时: {endpoint}; last_error={last_error}"
        )

    @staticmethod
    def _forward(stream, target) -> None:
        for line in stream:
            print(line, end="", file=target, flush=True)

    @staticmethod
    def _stop(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)

    @contextlib.contextmanager
    def activate(self, checkpoint: Path | None) -> Iterator[ModelProfile]:
        """启动指定权重，服务就绪后产出 profile，离开上下文时可靠回收。"""

        self._assert_port_available()
        command = self.build_command(checkpoint)
        environment = self.build_environment()
        process = subprocess.Popen(
            command,
            cwd=self.config.paths.ms_swift_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name != "nt",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=self._forward, args=(process.stdout, sys.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._forward, args=(process.stderr, sys.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            self._wait_until_ready(process)
            yield self.profile(checkpoint)
        finally:
            self._stop(process)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
