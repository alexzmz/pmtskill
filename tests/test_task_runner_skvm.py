from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import task_runner_skvm as runner  # noqa: E402
from vllm_wrapper import VLLMOpenAIWrapper  # noqa: E402


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def predict(self, prompt: str):
        self.prompts.append(prompt)
        return (
            'Reason: done\nAction: {"action_type":"status",'
            ' "goal_status":"complete"}',
            None,
            {"ok": True},
        )

    def get_stats(self):
        return {"request_count": len(self.prompts)}


def make_skill(path: Path, name: str = "android-world-t3a"):
    content = f"""---
name: {name}
description: Android contacts and calendar operation.
---

# Android

## Core policy

Inspect the UI before acting.

## Contacts

Add a contact and verify it.

## Calendar

Create an event and verify it.

## Output discipline

Return the exact Android World action schema.
"""
    path.mkdir(parents=True, exist_ok=True)
    skill_path = path / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    return runner._load_skill(skill_path)


def make_plan(*, parallel: bool):
    return {
        "artifacts": {
            "scr": {
                "skillName": "android-world-t3a",
                "purposes": [
                    {
                        "id": "operate-contacts",
                        "description": "Add and edit Android contacts",
                    },
                    {
                        "id": "operate-calendar",
                        "description": "Create Android calendar events",
                    },
                ],
            },
            "gaps": [
                {
                    "purposeId": "operate-contacts",
                    "primitiveId": "reason.planning",
                    "requiredLevel": "L3",
                    "modelLevel": "L1",
                }
            ],
            "dag": {
                "steps": [{"id": "contacts"}, {"id": "calendar"}],
                "parallelism": (
                    [{"type": "tlp", "steps": ["contacts", "calendar"]}]
                    if parallel
                    else []
                ),
            },
        },
        "passRuns": {"rewrite-skill": {"status": "ok"}},
        "guardPassed": True,
    }


class SkillCatalogTests(unittest.TestCase):
    def test_discovers_all_skills_in_deterministic_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_skill(root / "zeta", "zeta")
            make_skill(root / "alpha", "alpha")
            skills = runner.discover_skills([root], [])
            self.assertEqual([item.skill_id for item in skills], ["alpha", "zeta"])

    def test_rejects_duplicate_frontmatter_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_skill(root / "one", "same")
            make_skill(root / "two", "same")
            with self.assertRaisesRegex(ValueError, "Duplicate skill name"):
                runner.discover_skills([root], [])


class AdaptiveWrapperTests(unittest.TestCase):
    def _variants(self, root: Path):
        skill = make_skill(root / "skill")
        p1_content = skill.content + "\nP1 marker.\n"
        p13_content = skill.content + "\nP13 marker.\n"
        return skill, [
            runner.SkillVariant(
                skill=skill,
                source="original",
                tag="original",
                path=skill.path,
                content=skill.content,
                sha256=runner._sha256(skill.content),
            ),
            runner.SkillVariant(
                skill=skill,
                source="aot",
                tag="p1",
                path=root / "p1" / "SKILL.md",
                content=p1_content,
                sha256=runner._sha256(p1_content),
                plan=make_plan(parallel=False),
            ),
            runner.SkillVariant(
                skill=skill,
                source="aot",
                tag="p1p3",
                path=root / "p1p3" / "SKILL.md",
                content=p13_content,
                sha256=runner._sha256(p13_content),
                plan=make_plan(parallel=True),
            ),
        ]

    def test_selects_parallel_aot_variant_and_records_scr_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, variants = self._variants(root)
            fake = FakeLLM()
            trace = root / "adaptations.jsonl"
            wrapper = runner.SkVMAdaptiveWrapper(
                fake,
                variants=variants,
                target_model="vllm/test",
                mode="inject",
                variant_policy="aot",
                max_skills=1,
                max_skill_chars=12_000,
                trace_path=trace,
                trace_include_goal=False,
            )

            prompt = (
                "The current user goal/request is: Add a contact and then "
                "create a calendar event\n\n"
                "Here is a list of descriptions for some UI elements on the "
                "current screen:\nUI element 1: editable text field\n"
                "Here are some useful guidelines\nNow output an action."
            )
            wrapper.predict(prompt)

            self.assertIn('source="aot" variant="p1p3"', fake.prompts[0])
            self.assertIn("operate-contacts", fake.prompts[0])
            event = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(event["selected"][0]["variant"], "p1p3")
            self.assertIn("contacts", event["intent_labels"])
            self.assertIn("calendar", event["intent_labels"])
            self.assertIn("editable-form", event["ui_state_labels"])
            self.assertNotIn("goal", event)

    def test_prefers_real_jit_best_round_when_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill, variants = self._variants(root)
            jit_content = skill.content + "\nJIT marker.\n"
            variants.append(
                runner.SkillVariant(
                    skill=skill,
                    source="jit",
                    tag="jit-r2",
                    path=root / "round-2" / "SKILL.md",
                    content=jit_content,
                    sha256=runner._sha256(jit_content),
                    plan=make_plan(parallel=False),
                    proposal_id="bare-agent/vllm--test/skill/time",
                )
            )
            fake = FakeLLM()
            wrapper = runner.SkVMAdaptiveWrapper(
                fake,
                variants=variants,
                target_model="vllm/test",
                mode="inject",
                variant_policy="prefer-jit",
                max_skills=1,
                max_skill_chars=12_000,
                trace_path=root / "adaptations.jsonl",
                trace_include_goal=True,
            )

            wrapper.predict(
                "The current user goal/request is: Add a contact\n\nAct now."
            )
            self.assertIn('source="jit" variant="jit-r2"', fake.prompts[0])
            self.assertIn("JIT marker.", fake.prompts[0])
            stats = wrapper.get_stats()
            self.assertEqual(stats["skvm_variant_source_counts"], {"jit": 1})

    def test_jit_only_fails_before_first_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, variants = self._variants(root)
            with self.assertRaisesRegex(FileNotFoundError, "No usable SkVM JIT"):
                runner.SkVMAdaptiveWrapper(
                    FakeLLM(),
                    variants=variants,
                    target_model="vllm/test",
                    mode="inject",
                    variant_policy="jit-only",
                    max_skills=1,
                    max_skill_chars=12_000,
                    trace_path=root / "adaptations.jsonl",
                    trace_include_goal=False,
                )


class KernelArtifactTests(unittest.TestCase):
    def test_reuse_loads_aot_annotations_and_jit_best_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            cache = root / "cache"
            safe_model = "vllm--test"
            aot_root = (
                cache
                / "proposals"
                / "aot-compile"
                / "bare-agent"
                / safe_model
                / skill.skill_id
            )
            for tag, parallel in (("p1", False), ("p1p3", True)):
                variant_dir = aot_root / tag
                variant_dir.mkdir(parents=True)
                (variant_dir / "SKILL.md").write_text(
                    skill.content + f"\n{tag}\n", encoding="utf-8"
                )
                (variant_dir / "compilation-plan.json").write_text(
                    json.dumps(make_plan(parallel=parallel)),
                    encoding="utf-8",
                )

            proposal = (
                cache
                / "proposals"
                / "jit-optimize"
                / "bare-agent"
                / safe_model
                / skill.skill_id
                / "20260101T000000000Z"
            )
            (proposal / "round-2").mkdir(parents=True)
            (proposal / "meta.json").write_text(
                json.dumps({"status": "pending", "bestRound": 2}),
                encoding="utf-8",
            )
            (proposal / "round-2" / "SKILL.md").write_text(
                skill.content + "\nJIT best\n", encoding="utf-8"
            )

            args = SimpleNamespace(
                skvm_cache_dir=cache,
                skvm_target_model="vllm/test",
                skvm_compiler_model="target",
                skvm_aot_pass_sets=["1", "1,3"],
                skvm_prepare="reuse",
                skvm_variant_policy="prefer-jit",
                skill_mode="inject",
            )
            kernel = runner.SkVMKernel(
                args,
                skills=[skill],
                run_dir=root / "run",
                vllm_base_url="http://127.0.0.1:8000/v1",
                served_model="test",
            )
            variants = kernel.prepare()

            self.assertEqual(
                {(item.source, item.tag) for item in variants},
                {
                    ("original", "original"),
                    ("aot", "p1"),
                    ("aot", "p1p3"),
                    ("jit", "jit-r2"),
                },
            )
            jit = next(item for item in variants if item.source == "jit")
            self.assertEqual(
                jit.proposal_id,
                (
                    "bare-agent/vllm--test/android-world-t3a/"
                    "20260101T000000000Z"
                ),
            )
            self.assertEqual(len(runner._variant_purposes(jit)), 2)


class VLLMServerWrapperTests(unittest.TestCase):
    def test_raw_completion_mode_matches_baseline_prompt_shape(self):
        response_payload = {
            "choices": [{"text": "Reason: ok"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        wrapper = VLLMOpenAIWrapper(
            base_url="http://127.0.0.1:8000/v1",
            model="test",
            use_chat_completions=False,
        )
        with mock.patch(
            "vllm_wrapper.urllib_request.urlopen", return_value=Response()
        ) as urlopen:
            text, safe, raw = wrapper.predict("plain prompt")

        self.assertEqual(text, "Reason: ok")
        self.assertIsNone(safe)
        self.assertEqual(raw, response_payload)
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/v1/completions"))
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["prompt"], "plain prompt")
        self.assertNotIn("messages", sent)


if __name__ == "__main__":
    unittest.main()
