"""Evaluate Android World with a SkVM-compatible skill delivery layer.

The runner keeps Android World's environment, tasks, agent and scoring identical
to ``task_runner_detail.py``.  The only experimental variable is delivery of a
SKILL.md using SkVM's ``inject`` or ``discover`` semantics.

The default skill is a general Android World T3A playbook.  ``--skill_path`` can
also point at an original skill or at a SKILL.md produced by SkVM AOT/JIT
optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import task_runner_detail


DEFAULT_SKILL_PATH = (
    task_runner_detail.REPO_ROOT
    / "src"
    / "skills"
    / "android_world"
    / "SKILL.md"
)
SKVM_ROOT = task_runner_detail.REPO_ROOT / "libs" / "skvm"
LOAD_SKILL_RE = re.compile(
    r"<?load-skill>\s*(.*?)\s*</load-skill>", re.IGNORECASE
)


def _parse_frontmatter(content: str, skill_dir: Path) -> dict[str, str]:
    """Parse the simple name/description contract used by SkVM skills."""
    metadata = {
        "name": skill_dir.name,
        "description": "User-specified skill injected by SkVM",
    }
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return metadata
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in metadata:
            cleaned = value.strip().strip("'\"")
            if cleaned:
                metadata[key.strip()] = cleaned
    return metadata


def load_skill(path: Path) -> tuple[str, dict[str, Any]]:
    """Load a directory or SKILL.md exactly as SkVM's loader accepts it."""
    resolved = path.expanduser().resolve()
    skill_path = resolved if resolved.suffix.lower() == ".md" else resolved / "SKILL.md"
    if not skill_path.is_file():
        raise FileNotFoundError(f"SKILL.md not found: {skill_path}")
    content = skill_path.read_text(encoding="utf-8")
    metadata = _parse_frontmatter(content, skill_path.parent)
    info: dict[str, Any] = {
        "delivery": "skvm-compatible",
        "mode": None,
        "name": metadata["name"],
        "description": metadata["description"],
        "path": str(skill_path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "bytes": len(content.encode("utf-8")),
    }
    package_path = SKVM_ROOT / "package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
        info["skvm_version"] = package.get("version")
    except (OSError, json.JSONDecodeError):
        info["skvm_version"] = None
    info["skvm_root"] = str(SKVM_ROOT.resolve())
    info["semantics_reference"] = [
        str((SKVM_ROOT / "src" / "core" / "skill-loader.ts").resolve()),
        str((SKVM_ROOT / "src" / "adapters" / "bare-agent.ts").resolve()),
    ]
    return content, info


class SkVMSkillWrapper:
    """Apply SkVM skill delivery semantics to an Android World LLM wrapper."""

    def __init__(
        self,
        base_llm: Any,
        *,
        skill_content: str,
        skill_name: str,
        skill_description: str,
        mode: str,
    ) -> None:
        if mode not in {"inject", "discover"}:
            raise ValueError(f"Unsupported SkVM skill mode: {mode}")
        self.base_llm = base_llm
        self.skill_content = skill_content
        self.skill_name = skill_name
        self.skill_description = skill_description
        self.mode = mode
        self._session_skill_loaded = mode == "inject"
        self._skill_load_requests = 0
        self._skill_augmented_requests = 0
        self._discovery_requests = 0

    def reset_skill_session(self) -> None:
        """Reset discover-mode state at the Android World episode boundary."""
        self._session_skill_loaded = self.mode == "inject"

    def _injected_prompt(self, prompt: str) -> str:
        self._skill_augmented_requests += 1
        return (
            "The following domain skill is available for this request. Follow "
            "it when relevant while preserving the required response format.\n"
            f"<skill>\n{self.skill_content}\n</skill>\n\n"
            f"{prompt}"
        )

    def _discovery_prompt(self, prompt: str) -> str:
        self._discovery_requests += 1
        return f"""You have access to domain-specific skills.
To load the relevant skill, respond with EXACTLY:
<load-skill>{self.skill_name}</load-skill>

The opening and closing tags are both required. Once loaded, answer the
original request using its required output format.

Available skills:
- **{self.skill_name}**: {self.skill_description}

{prompt}"""

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        if self.mode == "inject" or self._session_skill_loaded:
            return self.base_llm.predict(self._injected_prompt(text_prompt))

        discovery_response = self.base_llm.predict(
            self._discovery_prompt(text_prompt)
        )
        output, _, _ = discovery_response
        match = LOAD_SKILL_RE.search(output or "")
        if not match or match.group(1).strip() != self.skill_name:
            return discovery_response

        # This mirrors SkVM bare-agent discover mode: intercept the load marker,
        # make the skill available, and retry the original request.
        self._skill_load_requests += 1
        self._session_skill_loaded = True
        return self.base_llm.predict(self._injected_prompt(text_prompt))

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
            "skvm_skill_mode": self.mode,
            "skill_load_requests": self._skill_load_requests,
            "skill_augmented_requests": self._skill_augmented_requests,
            "skill_discovery_requests": self._discovery_requests,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = task_runner_detail.build_parser(description=__doc__)
    skill = parser.add_argument_group("SkVM skill delivery")
    skill.add_argument(
        "--skill_path",
        type=Path,
        default=DEFAULT_SKILL_PATH,
        help=(
            "Skill directory or SKILL.md. This may be a SkVM AOT/JIT optimized "
            "skill artifact."
        ),
    )
    skill.add_argument(
        "--skill_mode",
        choices=("inject", "discover"),
        default="inject",
        help=(
            "SkVM delivery mode. inject always supplies the skill; discover "
            "requires the model to request it by name."
        ),
    )
    return parser


def _agent_factory(env: Any, llm: SkVMSkillWrapper, t3a_module: Any) -> Any:
    class SkVMT3A(t3a_module.T3A):
        def reset(self, go_home_on_reset: bool = False) -> None:
            llm.reset_skill_session()
            super().reset(go_home_on_reset)

    return SkVMT3A(env, llm, name="t3a_vllm_skvm")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        return task_runner_detail.list_available_tasks()

    skill_content, skill_info = load_skill(args.skill_path)
    skill_info["mode"] = args.skill_mode

    def model_factory(model_args: argparse.Namespace) -> SkVMSkillWrapper:
        base_llm = task_runner_detail.create_vllm(model_args)
        return SkVMSkillWrapper(
            base_llm,
            skill_content=skill_content,
            skill_name=str(skill_info["name"]),
            skill_description=str(skill_info["description"]),
            mode=args.skill_mode,
        )

    return task_runner_detail.run_evaluation(
        args,
        condition=f"skvm_{args.skill_mode}",
        model_factory=model_factory,
        skill_info=skill_info,
        agent_factory=_agent_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
