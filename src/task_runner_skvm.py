"""Run Android World with a real SkVM profile/AOT/runtime adaptation path.

This runner keeps Android World's task generation, T3A agent, checkpoints,
official success signals, and reports identical to ``task_runner_detail.py``.
Its model path is deliberately different:

1. start or reuse one OpenAI-compatible vLLM server;
2. inventory every SKILL.md under the configured skill roots;
3. use SkVM's ``bare-agent`` profiler to build a TCP for that same server;
4. use SkVM's compiler to emit AOT p1 and p1+p3 variants (including SCR and
   capability-gap annotations);
5. before *every* T3A model request, select a skill, variant, SCR purposes, and
   relevant sections for the current goal/phase;
6. prefer an existing SkVM JIT-optimize best round when one is available.

The online decisions are written to ``<run_dir>/skvm/adaptations.jsonl``.
The runner also emits ``skvm_report.json``, ``skvm_report.md``, numeric TCP
capability tables, compilation-gap tables, and immutable skill-variant
snapshots.  The result therefore measures ``model + model-adapted skill``
rather than merely prepending an unchanged SKILL.md.

Example:

    python src/task_runner_skvm.py \
      --model_path /models/Qwen \
      --tasks ContactsAddContact ClockCreateTimer

The first run profiles/compiles and can be slow. Later runs reuse the SkVM
cache. Use ``--skvm_prepare=reuse`` to require cached artifacts, or
``--skvm_prepare=force`` to regenerate them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import task_runner_detail
import skvm_reporting


REPO_ROOT = task_runner_detail.REPO_ROOT
SKVM_ROOT = REPO_ROOT / "libs" / "skvm"
DEFAULT_SKILLS_ROOT = REPO_ROOT / "src" / "skills"
DEFAULT_SKVM_CACHE = REPO_ROOT / "results" / "skvm-cache"
DEFAULT_COMPILER_MODEL = "target"
DEFAULT_AOT_PASS_SETS = ("1", "1,3")
SKVM_ADAPTER = "bare-agent"
SKVM_VLLM_KEY_ENV = "SKVM_ANDROID_VLLM_API_KEY"
LOAD_SKILL_RE = re.compile(
    r"<?load-skill>\s*(.*?)\s*</load-skill>", re.IGNORECASE
)
FRONTMATTER_RE = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL
)
HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.IGNORECASE)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "user",
    "with",
}

APP_LABELS: dict[str, tuple[str, ...]] = {
    "alarm-clock": ("alarm", "clock", "stopwatch", "timer"),
    "browser": ("browser", "website", "web page", "url"),
    "calendar": ("calendar", "event", "schedule", "recurrence"),
    "camera-gallery": ("camera", "photo", "picture", "gallery", "image"),
    "contacts": ("contact", "phone number", "address book"),
    "files": ("file", "folder", "directory", "filename"),
    "maps": ("map", "route", "location", "navigate"),
    "messages": ("message", "sms", "text", "recipient"),
    "music": ("music", "song", "playlist", "audio"),
    "notes": ("note", "memo", "markor", "todo", "task"),
    "phone": ("call", "dial", "voicemail"),
    "settings": (
        "setting",
        "wifi",
        "bluetooth",
        "brightness",
        "airplane mode",
        "dark mode",
    ),
}

STATE_LABELS: dict[str, tuple[str, ...]] = {
    "confirmation": ("confirm", "are you sure", "allow", "permission"),
    "date-time-picker": ("date picker", "time picker", "am", "pm", "hour"),
    "editable-form": ("editable", "text field", "input", "checkbox"),
    "keyboard-visible": ("keyboard", "ime", "enter key"),
    "loading-or-blocked": ("loading", "not responding", "wait", "dialog"),
    "scrollable-list": ("scrollable", "list item", "recycler"),
}


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _slug(text: str, limit: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return (value or "model")[:limit]


def _safe_model_name(model: str) -> str:
    return model.replace("/", "--").replace(":", "_")


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def _parse_frontmatter(content: str, skill_dir: Path) -> dict[str, str]:
    metadata = {
        "name": skill_dir.name,
        "description": f"Skill loaded from {skill_dir}",
    }
    match = FRONTMATTER_RE.match(content)
    if not match:
        return metadata
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        normalized_key = key.strip()
        if separator and normalized_key in metadata:
            cleaned = value.strip().strip("'\"")
            if cleaned:
                metadata[normalized_key] = cleaned
    return metadata


def _normalize_pass_set(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid AOT pass set {raw!r}; expected e.g. 1 or 1,3."
            ) from exc
        if value not in {1, 2, 3}:
            raise ValueError(
                f"Invalid AOT pass {value}; SkVM currently exposes 1, 2, 3."
            )
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("An AOT pass set cannot be empty.")
    return tuple(sorted(values))


def _pass_tag(passes: Iterable[int]) -> str:
    return "".join(f"p{value}" for value in sorted(set(passes)))


def _split_csv_values(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(result)) or None


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    task_runner_detail._write_json(path, payload)  # pylint: disable=protected-access


@dataclass(frozen=True)
class SkillSource:
    skill_id: str
    description: str
    path: Path
    content: str
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": len(self.content.encode("utf-8")),
        }


@dataclass(frozen=True)
class SkillVariant:
    skill: SkillSource
    source: str
    tag: str
    path: Path
    content: str
    sha256: str
    plan: Mapping[str, Any] = field(default_factory=dict)
    proposal_id: str | None = None

    def manifest(self) -> dict[str, Any]:
        artifacts = self.plan.get("artifacts") or {}
        scr = artifacts.get("scr") or {}
        return {
            "skill_id": self.skill.skill_id,
            "source": self.source,
            "tag": self.tag,
            "path": str(self.path),
            "sha256": self.sha256,
            "changed_from_original": self.sha256 != self.skill.sha256,
            "purpose_count": len(scr.get("purposes") or []),
            "gap_count": len(artifacts.get("gaps") or []),
            "guard_passed": self.plan.get("guardPassed"),
            "proposal_id": self.proposal_id,
        }


def _load_skill(path: Path) -> SkillSource:
    resolved = path.expanduser().resolve()
    skill_path = (
        resolved
        if resolved.name.lower() == "skill.md"
        else resolved / "SKILL.md"
    )
    if not skill_path.is_file():
        raise FileNotFoundError(f"SKILL.md not found: {skill_path}")
    content = skill_path.read_text(encoding="utf-8")
    metadata = _parse_frontmatter(content, skill_path.parent)
    return SkillSource(
        skill_id=metadata["name"],
        description=metadata["description"],
        path=skill_path,
        content=content,
        sha256=_sha256(content),
    )


def discover_skills(
    roots: Sequence[Path], explicit_paths: Sequence[Path]
) -> list[SkillSource]:
    """Build a deterministic, duplicate-checked catalog of all SKILL.md files."""
    paths: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Skill root does not exist: {resolved}")
        if resolved.is_file():
            paths.append(resolved)
        else:
            paths.extend(sorted(resolved.rglob("SKILL.md")))
    paths.extend(explicit_paths)

    by_path: dict[Path, SkillSource] = {}
    by_id: dict[str, SkillSource] = {}
    for path in paths:
        skill = _load_skill(path)
        if skill.path in by_path:
            continue
        previous = by_id.get(skill.skill_id)
        if previous is not None and previous.path != skill.path:
            raise ValueError(
                f"Duplicate skill name {skill.skill_id!r}: "
                f"{previous.path} and {skill.path}"
            )
        by_path[skill.path] = skill
        by_id[skill.skill_id] = skill

    if not by_path:
        raise ValueError("No SKILL.md files were found in the selected roots.")
    return sorted(by_path.values(), key=lambda item: item.skill_id)


def _get_json(url: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib_request.Request(url, headers=headers, method="GET")
    with urllib_request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _infer_tool_parser(model_path: str) -> str | None:
    lowered = model_path.lower()
    if "qwen3" in lowered:
        return "qwen3_xml"
    if "qwen" in lowered:
        return "hermes"
    if "mistral" in lowered or "mixtral" in lowered:
        return "mistral"
    if "llama" in lowered:
        return "llama3_json"
    return None


class ManagedVLLMServer:
    """Own a vLLM API subprocess, or validate a user-managed endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        process: subprocess.Popen[str] | None = None,
        log_handle: Any = None,
        log_path: Path | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.process = process
        self._log_handle = log_handle
        self.log_path = log_path

    @classmethod
    def connect(
        cls, base_url: str, api_key: str, timeout_s: float
    ) -> "ManagedVLLMServer":
        server = cls(base_url=base_url, api_key=api_key)
        server.wait_ready(timeout_s)
        return server

    @classmethod
    def start(
        cls, args: argparse.Namespace, run_dir: Path
    ) -> "ManagedVLLMServer":
        host = str(args.vllm_host)
        port = int(args.vllm_port) or _free_tcp_port(host)
        served_name = (
            args.vllm_served_model_name
            or _slug(Path(str(args.model_path).rstrip("/\\")).name, 120)
        )
        log_path = run_dir / "skvm" / "vllm-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open(
            "a", encoding="utf-8", errors="replace", buffering=1
        )

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(args.model_path),
            "--served-model-name",
            served_name,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--api-key",
            str(args.vllm_api_key),
        ]
        if args.max_model_len is not None:
            command.extend(["--max-model-len", str(args.max_model_len)])

        parser_name = args.vllm_tool_call_parser
        if parser_name == "auto":
            parser_name = _infer_tool_parser(str(args.model_path))
        if parser_name and parser_name.lower() != "none":
            command.extend(
                [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    parser_name,
                ]
            )
        if args.vllm_reasoning_parser:
            command.extend(
                ["--reasoning-parser", str(args.vllm_reasoning_parser)]
            )
        command.extend(args.vllm_server_arg or [])

        print(
            f"Starting shared vLLM server on http://{host}:{port} "
            f"(log: {log_path})",
            flush=True,
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except BaseException:
            log_handle.close()
            raise

        server = cls(
            base_url=f"http://{host}:{port}/v1",
            api_key=str(args.vllm_api_key),
            process=process,
            log_handle=log_handle,
            log_path=log_path,
        )
        try:
            server.wait_ready(float(args.vllm_startup_timeout_s))
        except BaseException:
            server.close()
            tail = ""
            try:
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            except OSError:
                pass
            if tail:
                print(f"Last vLLM server log lines:\n{tail}", file=sys.stderr)
            raise
        return server

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {self.process.returncode}. "
                    f"See {self.log_path}."
                )
            try:
                payload = _get_json(
                    f"{self.base_url}/models", self.api_key, timeout_s=5.0
                )
                if payload.get("data"):
                    return
            except (
                OSError,
                TimeoutError,
                ValueError,
                urllib_error.URLError,
            ) as exc:
                last_error = exc
            time.sleep(1.0)
        raise TimeoutError(
            f"vLLM server was not ready after {timeout_s:.0f}s: {last_error}"
        )

    def discover_model(self, requested: str | None) -> str:
        payload = _get_json(
            f"{self.base_url}/models", self.api_key, timeout_s=30.0
        )
        model_ids = [
            str(item.get("id"))
            for item in payload.get("data") or []
            if isinstance(item, Mapping) and item.get("id")
        ]
        if requested:
            if model_ids and requested not in model_ids:
                raise ValueError(
                    f"--vllm_served_model_name={requested!r} is not exposed "
                    f"by {self.base_url}; available: {model_ids}"
                )
            return requested
        if not model_ids:
            raise RuntimeError(f"{self.base_url}/models returned no model IDs.")
        return model_ids[0]

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log_handle is not None and not self._log_handle.closed:
            self._log_handle.close()


class SkVMKernel:
    """Drive the vendored SkVM CLI and load its typed on-disk artifacts."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        skills: Sequence[SkillSource],
        run_dir: Path,
        vllm_base_url: str,
        served_model: str,
    ) -> None:
        self.args = args
        self.skills = list(skills)
        self.run_dir = run_dir
        self.cache_dir = args.skvm_cache_dir.expanduser().resolve()
        self.vllm_base_url = _normalize_base_url(vllm_base_url)
        self.served_model = served_model
        self.target_model = (
            str(args.skvm_target_model)
            if args.skvm_target_model
            else f"vllm/{served_model}"
        )
        if "/" not in self.target_model:
            raise ValueError(
                "--skvm_target_model must use SkVM's <provider>/<model> form."
            )
        self.compiler_model = (
            self.target_model
            if args.skvm_compiler_model == "target"
            else str(args.skvm_compiler_model)
        )
        self.pass_sets = [
            _normalize_pass_set(value) for value in args.skvm_aot_pass_sets
        ]
        self.log_path = run_dir / "skvm" / "kernel.log"
        self._command: list[str] | None = None

    def _variant_dir(self, skill_id: str, passes: Iterable[int]) -> Path:
        return (
            self.cache_dir
            / "proposals"
            / "aot-compile"
            / SKVM_ADAPTER
            / _safe_model_name(self.target_model)
            / skill_id
            / _pass_tag(passes)
        )

    def _resolve_command(self) -> list[str]:
        if self._command is not None:
            return list(self._command)
        if self.args.skvm_command:
            configured = Path(self.args.skvm_command).expanduser()
            if not configured.exists() and shutil.which(str(configured)) is None:
                raise FileNotFoundError(
                    f"SkVM executable not found: {configured}"
                )
            self._command = [str(configured)]
            return list(self._command)

        installed = shutil.which("skvm")
        if installed:
            self._command = [installed]
            return list(self._command)

        bun = shutil.which("bun")
        entry = SKVM_ROOT / "src" / "index.ts"
        if bun and entry.is_file():
            self._command = [bun, "run", str(entry)]
            return list(self._command)

        raise FileNotFoundError(
            "SkVM is required for --skvm_prepare=auto/force, but neither an "
            "`skvm` executable nor `bun` was found. Install SkVM/Bun or use "
            "--skvm_prepare=reuse with an existing cache."
        )

    def _configure_local_route(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.cache_dir / "skvm.config.json"
        config: dict[str, Any] = {}
        if config_path.is_file():
            loaded = _read_json(config_path)
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid SkVM config object: {config_path}")
            config = loaded

        providers = config.setdefault("providers", {})
        if not isinstance(providers, dict):
            raise ValueError(f"Invalid providers block in {config_path}")
        routes = providers.setdefault("routes", [])
        if not isinstance(routes, list):
            raise ValueError(f"Invalid providers.routes in {config_path}")

        local_route = {
            "match": self.target_model,
            "kind": "openai-compatible",
            "apiKeyEnv": SKVM_VLLM_KEY_ENV,
            "baseUrl": self.vllm_base_url,
        }
        routes[:] = [
            route
            for route in routes
            if not (
                isinstance(route, Mapping)
                and route.get("match") == self.target_model
            )
        ]
        routes.insert(0, local_route)
        _write_json(config_path, config)

    def _run(self, command_args: Sequence[str], label: str) -> None:
        command = [*self._resolve_command(), *command_args]
        env = os.environ.copy()
        env["SKVM_CACHE"] = str(self.cache_dir)
        env[SKVM_VLLM_KEY_ENV] = str(self.args.vllm_api_key)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open(
            "a", encoding="utf-8", errors="replace", buffering=1
        ) as log:
            rendered = subprocess.list2cmdline(command)
            header = f"\n[{dt.datetime.now().isoformat()}] {rendered}\n"
            log.write(header)
            print(f"\nSkVM {label}: {rendered}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=str(SKVM_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"SkVM {label} failed with exit code {return_code}. "
                f"See {self.log_path}."
            )

    def _missing_variants(
        self, passes: tuple[int, ...]
    ) -> list[SkillSource]:
        return [
            skill
            for skill in self.skills
            if not (self._variant_dir(skill.skill_id, passes) / "SKILL.md").is_file()
        ]

    def prepare(self) -> list[SkillVariant]:
        mode = self.args.skvm_prepare
        if mode == "off":
            return self.load_variants(include_aot=False)

        missing_by_pass = {
            passes: self._missing_variants(passes) for passes in self.pass_sets
        }
        missing_count = sum(len(items) for items in missing_by_pass.values())
        if mode == "reuse" and missing_count:
            details = [
                f"{skill.skill_id}/{_pass_tag(passes)}"
                for passes, skills in missing_by_pass.items()
                for skill in skills
            ]
            raise FileNotFoundError(
                "Missing cached SkVM AOT variants: " + ", ".join(details)
            )

        should_compile = mode == "force" or missing_count > 0
        if should_compile:
            self._configure_local_route()
            needs_profile = any(1 in passes for passes in self.pass_sets)
            if needs_profile:
                profile_command = [
                    "profile",
                    f"--model={self.target_model}",
                    f"--adapter={SKVM_ADAPTER}",
                    f"--instances={self.args.skvm_profile_instances}",
                    f"--concurrency={self.args.skvm_profile_concurrency}",
                    f"--timeout-ms={self.args.skvm_profile_timeout_ms}",
                ]
                primitives = _split_csv_values(
                    self.args.skvm_profile_primitives
                )
                if primitives:
                    profile_command.append(
                        f"--primitives={','.join(primitives)}"
                    )
                if mode == "force":
                    profile_command.append("--force")
                self._run(profile_command, "model profiling")

            for passes in self.pass_sets:
                compile_skills = (
                    self.skills
                    if mode == "force"
                    else missing_by_pass[passes]
                )
                if not compile_skills:
                    continue
                paths = [str(skill.path) for skill in compile_skills]
                if any("," in path for path in paths):
                    raise ValueError(
                        "SkVM's multi-skill CLI cannot accept a path "
                        "containing a comma."
                    )
                compile_command = [
                    "aot-compile",
                    f"--skill={','.join(paths)}",
                    f"--model={self.target_model}",
                    f"--adapter={SKVM_ADAPTER}",
                    f"--pass={','.join(str(value) for value in passes)}",
                    f"--compiler-model={self.compiler_model}",
                    f"--concurrency={self.args.skvm_compile_concurrency}",
                    f"--timeout-ms={self.args.skvm_compile_timeout_ms}",
                ]
                self._run(
                    compile_command,
                    f"AOT compile ({_pass_tag(passes)})",
                )

        variants = self.load_variants(include_aot=True)
        self._validate_aot_annotations(variants)
        return variants

    def _validate_aot_annotations(
        self, variants: Sequence[SkillVariant]
    ) -> None:
        errors: list[str] = []
        for variant in variants:
            if variant.source != "aot" or "p1" not in variant.tag:
                continue
            artifacts = variant.plan.get("artifacts") or {}
            pass_runs = variant.plan.get("passRuns") or {}
            if not artifacts.get("scr"):
                errors.append(f"{variant.skill.skill_id}/{variant.tag}: no SCR")
            rewrite = pass_runs.get("rewrite-skill") or {}
            if rewrite.get("status") != "ok":
                errors.append(
                    f"{variant.skill.skill_id}/{variant.tag}: "
                    "rewrite-skill pass not successful"
                )
        if errors:
            raise RuntimeError(
                "SkVM produced incomplete AOT artifacts:\n- "
                + "\n- ".join(errors)
            )

    def _latest_jit_variant(
        self, skill: SkillSource, annotation_plan: Mapping[str, Any]
    ) -> SkillVariant | None:
        root = (
            self.cache_dir
            / "proposals"
            / "jit-optimize"
            / SKVM_ADAPTER
            / _safe_model_name(self.target_model)
            / skill.skill_id
        )
        if not root.is_dir():
            return None
        for proposal_dir in sorted(
            (path for path in root.iterdir() if path.is_dir()), reverse=True
        ):
            meta_path = proposal_dir / "meta.json"
            try:
                meta = _read_json(meta_path)
                if meta.get("status") == "infra-blocked":
                    continue
                best_round = int(meta.get("bestRound", 0))
                skill_path = proposal_dir / f"round-{best_round}" / "SKILL.md"
                if not skill_path.is_file():
                    continue
                content = skill_path.read_text(encoding="utf-8")
                proposal_id = str(
                    proposal_dir.relative_to(
                        self.cache_dir / "proposals" / "jit-optimize"
                    )
                ).replace("\\", "/")
                return SkillVariant(
                    skill=skill,
                    source="jit",
                    tag=f"jit-r{best_round}",
                    path=skill_path,
                    content=content,
                    sha256=_sha256(content),
                    plan=annotation_plan,
                    proposal_id=proposal_id,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return None

    def load_variants(self, *, include_aot: bool) -> list[SkillVariant]:
        variants: list[SkillVariant] = []
        aot_by_skill: dict[str, list[SkillVariant]] = {}
        for skill in self.skills:
            original = SkillVariant(
                skill=skill,
                source="original",
                tag="original",
                path=skill.path,
                content=skill.content,
                sha256=skill.sha256,
            )
            variants.append(original)
            if not include_aot:
                continue
            for passes in self.pass_sets:
                variant_dir = self._variant_dir(skill.skill_id, passes)
                skill_path = variant_dir / "SKILL.md"
                plan_path = variant_dir / "compilation-plan.json"
                if not skill_path.is_file():
                    continue
                content = skill_path.read_text(encoding="utf-8")
                plan: Mapping[str, Any] = {}
                if plan_path.is_file():
                    loaded = _read_json(plan_path)
                    if isinstance(loaded, Mapping):
                        plan = loaded
                variant = SkillVariant(
                    skill=skill,
                    source="aot",
                    tag=_pass_tag(passes),
                    path=skill_path,
                    content=content,
                    sha256=_sha256(content),
                    plan=plan,
                )
                variants.append(variant)
                aot_by_skill.setdefault(skill.skill_id, []).append(variant)

            plans = aot_by_skill.get(skill.skill_id) or []
            annotation_plan = (
                next(
                    (
                        item.plan
                        for item in plans
                        if item.tag == "p1p3"
                    ),
                    None,
                )
                or next((item.plan for item in plans if "p1" in item.tag), {})
            )
            jit_variant = self._latest_jit_variant(skill, annotation_plan)
            if jit_variant is not None:
                variants.append(jit_variant)
        return variants

    def manifest(self, variants: Sequence[SkillVariant]) -> dict[str, Any]:
        package_version = None
        try:
            package_version = _read_json(SKVM_ROOT / "package.json").get(
                "version"
            )
        except (OSError, AttributeError, json.JSONDecodeError):
            pass
        return {
            "delivery": "skvm-kernel-adaptive",
            "mode": self.args.skill_mode,
            "skvm_version": package_version,
            "skvm_root": str(SKVM_ROOT),
            "skvm_cache": str(self.cache_dir),
            "target_model": self.target_model,
            "compiler_model": self.compiler_model,
            "adapter": SKVM_ADAPTER,
            "prepare_mode": self.args.skvm_prepare,
            "aot_pass_sets": [list(item) for item in self.pass_sets],
            "variant_policy": self.args.skvm_variant_policy,
            "skills": [skill.manifest() for skill in self.skills],
            "variants": [variant.manifest() for variant in variants],
            "kernel_references": [
                str((SKVM_ROOT / "src" / "profiler" / "index.ts").resolve()),
                str((SKVM_ROOT / "src" / "compiler" / "index.ts").resolve()),
                str(
                    (
                        SKVM_ROOT
                        / "src"
                        / "compiler"
                        / "passes"
                        / "rewrite-skill"
                        / "extractor.ts"
                    ).resolve()
                ),
                str(
                    (
                        SKVM_ROOT / "src" / "proposals" / "storage.ts"
                    ).resolve()
                ),
            ],
        }


def _words(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOP_WORDS
    }


def _extract_goal(prompt: str) -> str:
    patterns = (
        r"current user goal/request is:\s*(.*?)(?:\n\n|\Z)",
        r"\(overall\) user goal/request is:\s*(.*?)(?:\n|\Z)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).strip().split())
    return ""


def _prompt_phase(prompt: str) -> str:
    lowered = prompt.lower()
    if (
        "summary of this step" in lowered
        or "summerize the latest step" in lowered
    ):
        return "summarization"
    return "action-selection"


def _intent_labels(goal: str) -> list[str]:
    lowered = goal.lower()
    labels = [
        label
        for label, keywords in APP_LABELS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return labels or ["general-android"]


def _runtime_context(prompt: str, goal: str, limit: int = 10_000) -> str:
    """Extract step-specific UI/history text without the static T3A preamble."""
    sections: list[str] = [goal]
    patterns = (
        (
            r"history of what you have done so far:\s*(.*?)"
            r"(?:\n\nHere is a list|\Z)"
        ),
        (
            r"list of descriptions for some UI elements on the current "
            r"screen:\s*(.*?)(?:\nHere are some useful guidelines|\Z)"
        ),
        (
            r"description for the before screenshot:\s*(.*?)"
            r"(?:\nHere is the description for the after screenshot|\Z)"
        ),
        (
            r"description for the after screenshot:\s*(.*?)"
            r"(?:\nThis is the action you picked|\Z)"
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            sections.append(match.group(1).strip())
    return "\n".join(sections)[-limit:]


def _state_labels(context: str) -> list[str]:
    lowered = context.lower()
    return [
        label
        for label, keywords in STATE_LABELS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def _is_multi_intent(goal: str, labels: Sequence[str]) -> bool:
    lowered = f" {goal.lower()} "
    return len(labels) > 1 or any(
        marker in lowered
        for marker in (" and ", " then ", " after ", " before ", ";")
    )


def _variant_purposes(variant: SkillVariant) -> list[Mapping[str, Any]]:
    artifacts = variant.plan.get("artifacts") or {}
    scr = artifacts.get("scr") or {}
    purposes = scr.get("purposes") or []
    return [item for item in purposes if isinstance(item, Mapping)]


def _variant_gaps(variant: SkillVariant) -> list[Mapping[str, Any]]:
    artifacts = variant.plan.get("artifacts") or {}
    gaps = artifacts.get("gaps") or []
    return [item for item in gaps if isinstance(item, Mapping)]


def _has_parallelism(variant: SkillVariant) -> bool:
    artifacts = variant.plan.get("artifacts") or {}
    dag = artifacts.get("dag") or {}
    return bool(dag.get("parallelism"))


def _section_chunks(content: str) -> list[tuple[str, str]]:
    body = FRONTMATTER_RE.sub("", content, count=1).strip()
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("skill", body)]
    chunks: list[tuple[str, str]] = []
    prefix = body[: matches[0].start()].strip()
    if prefix:
        chunks.append(("overview", prefix))
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(body)
        )
        chunks.append((match.group(1).strip(), body[match.start() : end].strip()))
    return chunks


class SkVMAdaptiveWrapper:
    """JIT-compose SkVM AOT/JIT variants for every Android World request."""

    def __init__(
        self,
        base_llm: Any,
        *,
        variants: Sequence[SkillVariant],
        target_model: str,
        mode: str,
        variant_policy: str,
        max_skills: int,
        max_skill_chars: int,
        trace_path: Path,
        trace_include_goal: bool,
    ) -> None:
        if mode not in {"inject", "discover"}:
            raise ValueError(f"Unsupported skill mode: {mode}")
        self.base_llm = base_llm
        self.variants = list(variants)
        self.target_model = target_model
        self.mode = mode
        self.variant_policy = variant_policy
        self.max_skills = max_skills
        self.max_skill_chars = max_skill_chars
        self.trace_path = trace_path
        self.trace_include_goal = trace_include_goal
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._session_skill_loaded = mode == "inject"
        self._request_count = 0
        self._skill_load_requests = 0
        self._adapted_requests = 0
        self._discovery_requests = 0
        self._injected_chars = 0
        self._source_counts: Counter[str] = Counter()
        self._tag_counts: Counter[str] = Counter()
        self._intent_counts: Counter[str] = Counter()
        self._state_counts: Counter[str] = Counter()
        self._phase_counts: Counter[str] = Counter()
        self._environment_event_counts: Counter[str] = Counter()
        self._environment_last_errors: list[str] = []
        self._restore_trace_stats()
        if self.variant_policy == "jit-only":
            # Fail before Android World starts rather than halfway through the
            # first episode when no proposal can satisfy the policy.
            self._candidate_variants()

    def _restore_trace_stats(self) -> None:
        """Continue request IDs and adaptation counters on checkpoint resume."""
        if not self.trace_path.is_file():
            return
        try:
            lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                event = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(event, Mapping):
                continue
            try:
                self._request_count = max(
                    self._request_count, int(event.get("request_id") or 0)
                )
                self._injected_chars += int(event.get("injected_chars") or 0)
            except (TypeError, ValueError):
                pass
            self._adapted_requests += 1
            self._phase_counts[str(event.get("phase") or "unknown")] += 1
            for label in event.get("intent_labels") or []:
                self._intent_counts[str(label)] += 1
            for label in event.get("ui_state_labels") or []:
                self._state_counts[str(label)] += 1
            for selection in event.get("selected") or []:
                if not isinstance(selection, Mapping):
                    continue
                self._source_counts[str(selection.get("source") or "unknown")] += 1
                self._tag_counts[str(selection.get("variant") or "unknown")] += 1

    def reset_skill_session(self) -> None:
        self._session_skill_loaded = self.mode == "inject"

    def record_environment_event(self, kind: str, error: BaseException) -> None:
        self._environment_event_counts[kind] += 1
        self._environment_last_errors.append(
            f"{type(error).__name__}: {error}"
        )
        self._environment_last_errors = self._environment_last_errors[-10:]

    def _candidate_variants(self) -> list[SkillVariant]:
        by_skill: dict[str, list[SkillVariant]] = {}
        for variant in self.variants:
            by_skill.setdefault(variant.skill.skill_id, []).append(variant)

        candidates: list[SkillVariant] = []
        for skill_variants in by_skill.values():
            jit = [item for item in skill_variants if item.source == "jit"]
            aot = [item for item in skill_variants if item.source == "aot"]
            original = [
                item for item in skill_variants if item.source == "original"
            ]
            if self.variant_policy == "jit-only":
                if not jit:
                    skill_id = skill_variants[0].skill.skill_id
                    raise FileNotFoundError(
                        f"No usable SkVM JIT proposal for {skill_id!r}."
                    )
                candidates.extend(jit)
            elif self.variant_policy == "prefer-jit" and jit:
                candidates.extend(jit)
            elif aot:
                candidates.extend(aot)
            else:
                candidates.extend(original)
        return candidates

    def _select_variants(
        self, goal: str, phase: str, labels: Sequence[str]
    ) -> list[SkillVariant]:
        goal_words = _words(goal)
        multi_intent = _is_multi_intent(goal, labels)
        scored: list[tuple[float, SkillVariant]] = []
        for variant in self._candidate_variants():
            metadata_text = (
                f"{variant.skill.skill_id} {variant.skill.description} "
                + " ".join(
                    f"{purpose.get('id', '')} "
                    f"{purpose.get('description', '')}"
                    for purpose in _variant_purposes(variant)
                )
            )
            overlap = len(goal_words & _words(metadata_text))
            score = float(overlap * 5)
            if variant.skill.skill_id == "android-world-t3a":
                score += 3
            if variant.source == "jit":
                score += 2
            if variant.source == "aot":
                score += 1
            if (
                multi_intent
                and variant.tag == "p1p3"
                and _has_parallelism(variant)
            ):
                score += 3
            elif variant.tag == "p1":
                score += 1
            if phase == "summarization" and "summary" in metadata_text.lower():
                score += 1
            scored.append((score, variant))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].source == "jit",
                item[1].tag == "p1p3",
                item[1].skill.skill_id,
            ),
            reverse=True,
        )
        selected: list[SkillVariant] = []
        seen_skills: set[str] = set()
        for score, variant in scored:
            if variant.skill.skill_id in seen_skills:
                continue
            if selected and score < 3:
                # Always keep one fallback, but do not inject unrelated skills
                # merely because the catalog contains them.
                continue
            selected.append(variant)
            seen_skills.add(variant.skill.skill_id)
            if len(selected) >= self.max_skills:
                break
        return selected

    def _select_purposes(
        self, variant: SkillVariant, context: str
    ) -> list[Mapping[str, Any]]:
        context_words = _words(context)
        purposes = _variant_purposes(variant)
        scored = sorted(
            purposes,
            key=lambda purpose: len(
                context_words
                & _words(
                    f"{purpose.get('id', '')} "
                    f"{purpose.get('description', '')}"
                )
            ),
            reverse=True,
        )
        matched = [
            item
            for item in scored
            if context_words
            & _words(
                f"{item.get('id', '')} {item.get('description', '')}"
            )
        ]
        return (matched or scored)[:3]

    def _select_sections(
        self, variant: SkillVariant, context: str, phase: str, budget: int
    ) -> str:
        chunks = _section_chunks(variant.content)
        if sum(len(text) for _, text in chunks) <= budget:
            return "\n\n".join(text for _, text in chunks)

        context_words = _words(context)
        ranked: list[tuple[float, int, str]] = []
        for index, (heading, text) in enumerate(chunks):
            heading_lower = heading.lower()
            score = float(
                len(context_words & _words(f"{heading} {text}"))
            )
            if index == 0 or heading_lower in {"core policy", "output discipline"}:
                score += 100
            if phase == "summarization" and (
                "output" in heading_lower or "summary" in text.lower()
            ):
                score += 50
            ranked.append((score, index, text))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        selected: list[tuple[int, str]] = []
        used = 0
        for _, index, text in ranked:
            if selected and used + len(text) > budget:
                continue
            selected.append((index, text))
            used += len(text)
            if used >= budget:
                break
        selected.sort()
        return "\n\n".join(text for _, text in selected)

    def _adapt(self, prompt: str) -> tuple[str, dict[str, Any]]:
        phase = _prompt_phase(prompt)
        goal = _extract_goal(prompt)
        labels = _intent_labels(goal)
        context = _runtime_context(prompt, goal)
        state_labels = _state_labels(context)
        selected = self._select_variants(goal, phase, labels)

        remaining = self.max_skill_chars
        skill_blocks: list[str] = []
        selections: list[dict[str, Any]] = []
        for variant in selected:
            if remaining <= 0:
                break
            purposes = self._select_purposes(variant, context)
            purpose_ids = [
                str(item.get("id")) for item in purposes if item.get("id")
            ]
            relevant_purpose_ids = set(purpose_ids)
            gaps = [
                gap
                for gap in _variant_gaps(variant)
                if not relevant_purpose_ids
                or gap.get("purposeId") in relevant_purpose_ids
            ]
            gap_labels = [
                (
                    f"{gap.get('primitiveId')} "
                    f"{gap.get('requiredLevel')}->{gap.get('modelLevel')}"
                )
                for gap in gaps[:8]
            ]
            section_text = self._select_sections(
                variant, context, phase, remaining
            )
            remaining -= len(section_text)
            skill_blocks.append(
                "\n".join(
                    [
                        (
                            f'<skill id="{variant.skill.skill_id}" '
                            f'source="{variant.source}" '
                            f'variant="{variant.tag}">'
                        ),
                        section_text,
                        "</skill>",
                    ]
                )
            )
            selections.append(
                {
                    "skill_id": variant.skill.skill_id,
                    "source": variant.source,
                    "variant": variant.tag,
                    "sha256": variant.sha256,
                    "purposes": purpose_ids,
                    "gaps": gap_labels,
                    "proposal_id": variant.proposal_id,
                }
            )

        annotation_lines = [
            "<skvm-adaptation>",
            f"target_model: {self.target_model}",
            f"phase: {phase}",
            f"intent_labels: {', '.join(labels)}",
            f"ui_state_labels: {', '.join(state_labels) or 'none'}",
            (
                "selected_variants: "
                + ", ".join(
                    f"{item['skill_id']}:{item['source']}:{item['variant']}"
                    for item in selections
                )
            ),
            (
                "selected_scr_purposes: "
                + ", ".join(
                    purpose
                    for item in selections
                    for purpose in item["purposes"]
                )
            ),
            (
                "compensated_capability_gaps: "
                + ", ".join(
                    gap for item in selections for gap in item["gaps"]
                )
            ),
            "</skvm-adaptation>",
        ]
        adaptation = (
            "SkVM has selected a target-model-specific skill variant for this "
            "request. Follow it when relevant and preserve the surrounding "
            "prompt's exact response format.\n"
            + "\n".join(annotation_lines)
            + "\n\n"
            + "\n\n".join(skill_blocks)
            + "\n\n"
        )
        event = {
            "request_id": self._request_count,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "phase": phase,
            "intent_labels": labels,
            "ui_state_labels": state_labels,
            "goal_sha256": _sha256(goal),
            "context_sha256": _sha256(context),
            "selected": selections,
            "injected_chars": len(adaptation),
        }
        if self.trace_include_goal:
            event["goal"] = goal
        return adaptation + prompt, event

    def _record_event(self, event: Mapping[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _discovery_prompt(
        self, prompt: str, variants: Sequence[SkillVariant]
    ) -> str:
        lines = [
            "You have access to target-model-adapted skills.",
            "To load one, respond with EXACTLY:",
            "<load-skill>SKILL_NAME</load-skill>",
            "",
            "Available skills:",
        ]
        lines.extend(
            f"- **{item.skill.skill_id}**: {item.skill.description}"
            for item in variants
        )
        return "\n".join(lines) + "\n\n" + prompt

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        self._request_count += 1
        if self.mode == "discover" and not self._session_skill_loaded:
            goal = _extract_goal(text_prompt)
            phase = _prompt_phase(text_prompt)
            labels = _intent_labels(goal)
            candidates = self._select_variants(goal, phase, labels)
            self._discovery_requests += 1
            response = self.base_llm.predict(
                self._discovery_prompt(text_prompt, candidates)
            )
            output, _, _ = response
            match = LOAD_SKILL_RE.search(output or "")
            if not match:
                return response
            requested = match.group(1).strip()
            if requested not in {
                item.skill.skill_id for item in candidates
            }:
                return response
            self._skill_load_requests += 1
            self._session_skill_loaded = True

        adapted_prompt, event = self._adapt(text_prompt)
        self._adapted_requests += 1
        self._injected_chars += int(event["injected_chars"])
        self._phase_counts[str(event["phase"])] += 1
        for label in event["intent_labels"]:
            self._intent_counts[str(label)] += 1
        for label in event["ui_state_labels"]:
            self._state_counts[str(label)] += 1
        for selection in event["selected"]:
            self._source_counts[str(selection["source"])] += 1
            self._tag_counts[str(selection["variant"])] += 1
        self._record_event(event)
        return self.base_llm.predict(adapted_prompt)

    def predict_mm(
        self, text_prompt: str, images: Optional[list[Any]] = None
    ) -> tuple[str, Optional[bool], Any]:
        del images
        return self.predict(text_prompt)

    def get_stats(self) -> dict[str, Any]:
        base_stats = (
            self.base_llm.get_stats()
            if hasattr(self.base_llm, "get_stats")
            else {}
        )
        return dict(base_stats) | {
            "skvm_target_model": self.target_model,
            "skvm_skill_mode": self.mode,
            "skvm_variant_policy": self.variant_policy,
            "skvm_online_requests": self._request_count,
            "skvm_adapted_requests": self._adapted_requests,
            "skvm_discovery_requests": self._discovery_requests,
            "skill_load_requests": self._skill_load_requests,
            "skvm_injected_chars": self._injected_chars,
            "skvm_variant_source_counts": dict(self._source_counts),
            "skvm_variant_tag_counts": dict(self._tag_counts),
            "skvm_intent_counts": dict(self._intent_counts),
            "skvm_ui_state_counts": dict(self._state_counts),
            "skvm_phase_counts": dict(self._phase_counts),
            "skvm_adaptation_trace": str(self.trace_path),
            "android_environment_events": dict(
                self._environment_event_counts
            ),
            "android_environment_last_errors": list(
                self._environment_last_errors
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = task_runner_detail.build_parser(description=__doc__)

    server = parser.add_argument_group("Shared vLLM OpenAI server")
    server.add_argument(
        "--vllm_base_url",
        default=None,
        help=(
            "Reuse an existing vLLM OpenAI-compatible endpoint. When omitted, "
            "this runner starts one and shuts it down after evaluation."
        ),
    )
    server.add_argument("--vllm_host", default="127.0.0.1")
    server.add_argument(
        "--vllm_port",
        type=int,
        default=0,
        help="Managed server port; zero selects a free local port.",
    )
    server.add_argument(
        "--vllm_api_key",
        default=os.environ.get("VLLM_API_KEY", "skvm-local"),
    )
    server.add_argument(
        "--vllm_served_model_name",
        default=None,
        help=(
            "Model ID exposed by /v1/models. The managed server defaults to "
            "the model directory name; an external server is queried."
        ),
    )
    server.add_argument("--vllm_startup_timeout_s", type=float, default=900.0)
    server.add_argument("--vllm_request_timeout_s", type=float, default=300.0)
    server.add_argument(
        "--vllm_eval_api",
        choices=("completions", "chat"),
        default="completions",
        help=(
            "Use raw completions by default to match task_runner_detail.py's "
            "in-process LLM.generate prompt semantics."
        ),
    )
    server.add_argument(
        "--vllm_tool_call_parser",
        default="auto",
        help=(
            "vLLM auto-tool parser for SkVM profiling/compilation. 'auto' "
            "infers common Qwen/Mistral/Llama parsers; 'none' disables it."
        ),
    )
    server.add_argument("--vllm_reasoning_parser", default=None)
    server.add_argument(
        "--vllm_server_arg",
        action="append",
        default=[],
        help="Extra argument passed verbatim to the managed vLLM server.",
    )

    catalog = parser.add_argument_group("Skill catalog")
    catalog.add_argument(
        "--skills_root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Recursively inventory SKILL.md files under this root. Repeat for "
            "multiple roots. Defaults to src/skills."
        ),
    )
    catalog.add_argument(
        "--skill_path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional skill directory or SKILL.md. Repeatable; retained for "
            "compatibility with the old single-skill runner."
        ),
    )

    prepare = parser.add_argument_group("SkVM kernel preparation")
    prepare.add_argument(
        "--skvm_prepare",
        choices=("auto", "force", "reuse", "off"),
        default="auto",
        help=(
            "auto builds missing TCP/AOT artifacts; force regenerates them; "
            "reuse requires cached artifacts; off uses original skills."
        ),
    )
    prepare.add_argument(
        "--skvm_command",
        default=None,
        help="Explicit SkVM executable. Defaults to PATH, then vendored Bun.",
    )
    prepare.add_argument(
        "--skvm_cache_dir",
        type=Path,
        default=DEFAULT_SKVM_CACHE,
        help="Persistent SkVM profiles/proposals/config cache.",
    )
    prepare.add_argument(
        "--skvm_target_model",
        default=None,
        help=(
            "SkVM provider-prefixed target ID. Defaults to vllm/<served-model>."
        ),
    )
    prepare.add_argument(
        "--skvm_compiler_model",
        default=DEFAULT_COMPILER_MODEL,
        help=(
            "'target' uses the same vLLM model for compiler calls. A separate "
            "SkVM provider-prefixed model can produce stronger AOT rewrites."
        ),
    )
    prepare.add_argument(
        "--skvm_aot_pass_sets",
        nargs="+",
        default=list(DEFAULT_AOT_PASS_SETS),
        metavar="PASSES",
        help=(
            "AOT variants to generate, e.g. --skvm_aot_pass_sets 1 1,3. "
            "Pass 2 is intentionally omitted by default for Android UI tasks."
        ),
    )
    prepare.add_argument(
        "--skvm_profile_primitives",
        nargs="+",
        default=None,
        help=(
            "Optional primitive subset. Omit to profile SkVM's complete "
            "primitive catalog."
        ),
    )
    prepare.add_argument("--skvm_profile_instances", type=int, default=3)
    prepare.add_argument("--skvm_profile_concurrency", type=int, default=1)
    prepare.add_argument("--skvm_compile_concurrency", type=int, default=1)
    prepare.add_argument(
        "--skvm_profile_timeout_ms", type=int, default=120_000
    )
    prepare.add_argument(
        "--skvm_compile_timeout_ms", type=int, default=300_000
    )

    online = parser.add_argument_group("Online skill adaptation")
    online.add_argument(
        "--skill_mode",
        choices=("inject", "discover"),
        default="inject",
    )
    online.add_argument(
        "--skvm_variant_policy",
        choices=("aot", "prefer-jit", "jit-only"),
        default="prefer-jit",
        help=(
            "prefer-jit consumes SkVM's latest usable JIT-optimize best round "
            "when present and otherwise uses AOT."
        ),
    )
    online.add_argument("--skvm_max_skills_per_request", type=int, default=2)
    online.add_argument("--skvm_max_skill_chars", type=int, default=12_000)
    online.add_argument(
        "--skvm_trace_include_goal",
        action="store_true",
        help="Include plaintext task goals in adaptations.jsonl.",
    )
    recovery = parser.add_argument_group("Android environment recovery")
    recovery.add_argument(
        "--android_reset_retries",
        type=int,
        default=2,
        help="Retries after an accessibility-tree failure during episode reset.",
    )
    recovery.add_argument(
        "--android_reset_retry_wait_s",
        type=float,
        default=2.0,
        help="Wait between accessibility reset retries.",
    )
    recovery.add_argument(
        "--a11y_fallback",
        choices=("none", "uiautomator"),
        default="uiautomator",
        help=(
            "Fall back to adb uiautomator when Android World's a11y gRPC "
            "forwarder remains unavailable after retries."
        ),
    )
    return parser


def validate_skvm_args(args: argparse.Namespace) -> None:
    task_runner_detail.validate_args(args)
    if args.vllm_port < 0 or args.vllm_port > 65535:
        raise ValueError("--vllm_port must be between 0 and 65535.")
    if args.vllm_startup_timeout_s <= 0:
        raise ValueError("--vllm_startup_timeout_s must be positive.")
    if args.vllm_request_timeout_s <= 0:
        raise ValueError("--vllm_request_timeout_s must be positive.")
    if args.skvm_profile_instances < 1:
        raise ValueError("--skvm_profile_instances must be at least 1.")
    if args.skvm_profile_concurrency < 1:
        raise ValueError("--skvm_profile_concurrency must be at least 1.")
    if args.skvm_compile_concurrency < 1:
        raise ValueError("--skvm_compile_concurrency must be at least 1.")
    if args.skvm_profile_timeout_ms < 1:
        raise ValueError("--skvm_profile_timeout_ms must be positive.")
    if args.skvm_compile_timeout_ms < 1:
        raise ValueError("--skvm_compile_timeout_ms must be positive.")
    if args.skvm_max_skills_per_request < 1:
        raise ValueError("--skvm_max_skills_per_request must be at least 1.")
    if args.skvm_max_skill_chars < 500:
        raise ValueError("--skvm_max_skill_chars must be at least 500.")
    if args.android_reset_retries < 0:
        raise ValueError("--android_reset_retries cannot be negative.")
    if args.android_reset_retry_wait_s < 0:
        raise ValueError("--android_reset_retry_wait_s cannot be negative.")
    for value in args.skvm_aot_pass_sets:
        _normalize_pass_set(value)


def _agent_factory(
    env: Any,
    llm: SkVMAdaptiveWrapper,
    t3a_module: Any,
    *,
    reset_retries: int,
    retry_wait_s: float,
    a11y_fallback: str,
) -> Any:
    class SkVMAdaptiveT3A(t3a_module.T3A):
        def reset(self, go_home_on_reset: bool = False) -> None:
            llm.reset_skill_session()
            last_error: RuntimeError | None = None
            for attempt in range(reset_retries + 1):
                try:
                    super().reset(go_home_on_reset)
                    return
                except RuntimeError as exc:
                    if "Could not get a11y tree" not in str(exc):
                        raise
                    last_error = exc
                    llm.record_environment_event("a11y_reset_failure", exc)
                    if attempt >= reset_retries:
                        break
                    print(
                        "Android a11y reset failed; refreshing the controller "
                        f"and retrying ({attempt + 1}/{reset_retries})...",
                        flush=True,
                    )
                    try:
                        self.env.controller.refresh_env()
                    except Exception as refresh_error:  # pylint: disable=broad-exception-caught
                        llm.record_environment_event(
                            "a11y_refresh_failure", refresh_error
                        )
                    if retry_wait_s:
                        time.sleep(retry_wait_s)

            if a11y_fallback == "uiautomator":
                from android_world.env import android_world_controller

                print(
                    "Android a11y gRPC forwarder is unavailable; falling "
                    "back to uiautomator for UI extraction.",
                    flush=True,
                )
                self.env.controller._a11y_method = (  # pylint: disable=protected-access
                    android_world_controller.A11yMethod.UIAUTOMATOR
                )
                assert last_error is not None
                llm.record_environment_event(
                    "a11y_uiautomator_fallback", last_error
                )
                super().reset(go_home_on_reset)
                return
            assert last_error is not None
            raise last_error

    return SkVMAdaptiveT3A(env, llm, name="t3a_vllm_skvm_adaptive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        return task_runner_detail.list_available_tasks()
    validate_skvm_args(args)

    roots = args.skills_root or [DEFAULT_SKILLS_ROOT]
    skills = discover_skills(roots, args.skill_path)
    print(
        f"SkVM skill catalog: {len(skills)} skill(s): "
        + ", ".join(skill.skill_id for skill in skills),
        flush=True,
    )

    condition = "skvm_adaptive"
    run_dir = task_runner_detail._resolve_run_dir(  # pylint: disable=protected-access
        args, condition
    )
    args.run_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    server: ManagedVLLMServer | None = None
    try:
        if args.vllm_base_url:
            print(
                f"Connecting to shared vLLM server: {args.vllm_base_url}",
                flush=True,
            )
            server = ManagedVLLMServer.connect(
                str(args.vllm_base_url),
                str(args.vllm_api_key),
                float(args.vllm_startup_timeout_s),
            )
        else:
            server = ManagedVLLMServer.start(args, run_dir)
        served_model = server.discover_model(args.vllm_served_model_name)
        print(f"vLLM served model: {served_model}", flush=True)

        kernel = SkVMKernel(
            args,
            skills=skills,
            run_dir=run_dir,
            vllm_base_url=server.base_url,
            served_model=served_model,
        )
        variants = kernel.prepare()
        manifest = kernel.manifest(variants)
        catalog_path = run_dir / "skvm" / "skill-catalog.json"
        _write_json(catalog_path, manifest)
        print(
            f"SkVM variants ready: {len(variants)} "
            f"(catalog: {catalog_path})",
            flush=True,
        )

        from vllm_wrapper import VLLMOpenAIWrapper

        base_llm = VLLMOpenAIWrapper(
            base_url=server.base_url,
            model=served_model,
            api_key=str(args.vllm_api_key),
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout_s=args.vllm_request_timeout_s,
            raise_on_error=True,
            use_chat_completions=args.vllm_eval_api == "chat",
        )
        adaptive_llm = SkVMAdaptiveWrapper(
            base_llm,
            variants=variants,
            target_model=kernel.target_model,
            mode=args.skill_mode,
            variant_policy=args.skvm_variant_policy,
            max_skills=args.skvm_max_skills_per_request,
            max_skill_chars=args.skvm_max_skill_chars,
            trace_path=run_dir / "skvm" / "adaptations.jsonl",
            trace_include_goal=args.skvm_trace_include_goal,
        )

        skill_info = dict(manifest)
        skill_info["catalog_path"] = str(catalog_path)
        skill_info["adaptation_trace"] = str(
            run_dir / "skvm" / "adaptations.jsonl"
        )
        skill_info["name"] = (
            skills[0].skill_id
            if len(skills) == 1
            else f"catalog-{len(skills)}-skills"
        )
        skill_info["path"] = str(catalog_path)
        skill_info["sha256"] = _sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        )
        skill_info["vllm"] = {
            "served_model": served_model,
            "managed": server.process is not None,
            "eval_api": args.vllm_eval_api,
            # A managed server uses a fresh free port on each process start.
            # Do not put that ephemeral port into the resume signature.
            "base_url": (
                None if server.process is not None else server.base_url
            ),
            "server_log": (
                str(server.log_path) if server.log_path is not None else None
            ),
        }
        skill_info["android_recovery"] = {
            "reset_retries": args.android_reset_retries,
            "retry_wait_s": args.android_reset_retry_wait_s,
            "a11y_fallback": args.a11y_fallback,
            "max_consecutive_infrastructure_errors": (
                args.max_consecutive_infrastructure_errors
            ),
            "grpc_port": args.grpc_port,
        }
        skill_info["skvm_evaluation"] = {
            "json": str(run_dir / "skvm_report.json"),
            "markdown": str(run_dir / "skvm_report.md"),
            "capabilities_csv": str(
                run_dir / "skvm" / "capabilities.csv"
            ),
        }

        def model_factory(_: argparse.Namespace) -> SkVMAdaptiveWrapper:
            return adaptive_llm

        def agent_factory(env: Any, llm: Any, t3a_module: Any) -> Any:
            return _agent_factory(
                env,
                llm,
                t3a_module,
                reset_retries=args.android_reset_retries,
                retry_wait_s=args.android_reset_retry_wait_s,
                a11y_fallback=args.a11y_fallback,
            )

        evaluation_error: BaseException | None = None
        reporting_error: Exception | None = None
        exit_code = 1
        try:
            exit_code = task_runner_detail.run_evaluation(
                args,
                condition=condition,
                model_factory=model_factory,
                skill_info=skill_info,
                agent_factory=agent_factory,
            )
        except BaseException as exc:
            # task_runner_detail has already materialized its checkpoint-based
            # report. Preserve the original failure until the SkVM-specific
            # report has captured the same partial run.
            evaluation_error = exc

        try:
            skvm_reporting.write_skvm_evaluation(
                run_dir=run_dir,
                cache_dir=kernel.cache_dir,
                target_model=kernel.target_model,
                adapter=SKVM_ADAPTER,
                variants=variants,
                manifest=manifest,
                runtime_stats=adaptive_llm.get_stats(),
            )
            print(
                f"SkVM evaluation: {run_dir / 'skvm_report.md'}",
                flush=True,
            )
            print(
                f"SkVM structured evaluation: "
                f"{run_dir / 'skvm_report.json'}",
                flush=True,
            )
        except Exception as exc:  # Do not hide the Android failure.
            reporting_error = exc
            print(
                f"Failed to write SkVM evaluation artifacts: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if evaluation_error is not None:
            raise evaluation_error
        if reporting_error is not None:
            raise reporting_error
        return exit_code
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
