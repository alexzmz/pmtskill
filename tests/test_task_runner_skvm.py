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
import skvm_reporting  # noqa: E402
import task_runner_detail  # noqa: E402
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


class AndroidRecoveryTests(unittest.TestCase):
    def test_a11y_reset_refreshes_environment_and_retries(self):
        class Controller:
            def __init__(self):
                self.ready = False
                self.refresh_count = 0

            def refresh_env(self):
                self.refresh_count += 1
                self.ready = True

        class T3A:
            def __init__(self, env, llm, name):
                self.env = env
                self.llm = llm
                self.name = name
                self.reset_count = 0

            def reset(self, go_home_on_reset=False):
                del go_home_on_reset
                self.reset_count += 1
                if not self.env.controller.ready:
                    raise RuntimeError("Could not get a11y tree.")

        class AdaptiveLLM:
            def __init__(self):
                self.session_resets = 0
                self.events = []

            def reset_skill_session(self):
                self.session_resets += 1

            def record_environment_event(self, kind, error):
                self.events.append((kind, str(error)))

        controller = Controller()
        adaptive = AdaptiveLLM()
        agent = runner._agent_factory(
            SimpleNamespace(controller=controller),
            adaptive,
            SimpleNamespace(T3A=T3A),
            reset_retries=1,
            retry_wait_s=0,
            a11y_fallback="none",
        )

        agent.reset()

        self.assertEqual(agent.reset_count, 2)
        self.assertEqual(controller.refresh_count, 1)
        self.assertEqual(adaptive.session_resets, 1)
        self.assertEqual(adaptive.events[0][0], "a11y_reset_failure")

    def test_fail_fast_saves_diagnostic_episode_before_aborting(self):
        class Delegate:
            def __init__(self):
                self.saved = []

            def save_episodes(self, episodes, task_name):
                self.saved.append((task_name, episodes))

            def load(self, fields=None):
                del fields
                return [
                    episode
                    for _, episodes in self.saved
                    for episode in episodes
                ]

        delegate = Delegate()
        checkpointer = task_runner_detail._FailFastCheckpointer(
            delegate, limit=2
        )
        error_episode = {
            "exception_info": "RuntimeError: Could not get a11y tree."
        }

        checkpointer.save_episodes([error_episode], "task-1")
        with self.assertRaisesRegex(
            RuntimeError, "2 consecutive infrastructure errors"
        ):
            checkpointer.save_episodes([error_episode], "task-2")

        self.assertEqual(len(delegate.saved), 2)
        self.assertEqual(len(checkpointer.load()), 2)


class SkVMReportingTests(unittest.TestCase):
    def test_marks_an_all_error_run_invalid_for_effect_comparison(self):
        validity = skvm_reporting._performance_validity(
            {
                "summary": {
                    "attempted_episodes": 116,
                    "scored_episodes": 0,
                    "error_episodes": 116,
                }
            },
            {"skvm_adapted_requests": 0},
            {"status": "complete"},
        )

        self.assertFalse(validity["valid_for_skvm_effect_comparison"])
        self.assertIn(
            "No episode received an Android World score.",
            validity["invalid_reasons"],
        )
        self.assertIn(
            "No model request received online SkVM skill adaptation.",
            validity["invalid_reasons"],
        )

    def test_writes_capability_values_and_valid_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            cache = root / "cache"
            skill = make_skill(root / "skill")
            p1_path = root / "p1" / "SKILL.md"
            p1_path.parent.mkdir(parents=True)
            p1_content = skill.content + "\nP1 adapted.\n"
            p1_path.write_text(p1_content, encoding="utf-8")
            variants = [
                runner.SkillVariant(
                    skill=skill,
                    source="original",
                    tag="original",
                    path=skill.path,
                    content=skill.content,
                    sha256=skill.sha256,
                ),
                runner.SkillVariant(
                    skill=skill,
                    source="aot",
                    tag="p1",
                    path=p1_path,
                    content=p1_content,
                    sha256=runner._sha256(p1_content),
                    plan=make_plan(parallel=False),
                ),
            ]

            profile_path = (
                cache
                / "profiles"
                / "bare-agent"
                / "vllm--test"
                / "latest.json"
            )
            profile_path.parent.mkdir(parents=True)
            profile_path.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "model": "vllm/test",
                        "harness": "bare-agent",
                        "profiledAt": "2026-01-01T00:00:00Z",
                        "capabilities": {
                            "reason.planning": "L1",
                            "follow.format": "L3",
                        },
                        "details": [
                            {
                                "primitiveId": "reason.planning",
                                "levelResults": [
                                    {"passCount": 1, "totalCount": 3}
                                ],
                            },
                            {
                                "primitiveId": "follow.format",
                                "levelResults": [
                                    {"passCount": 3, "totalCount": 3}
                                ],
                            },
                        ],
                        "isPartial": False,
                    }
                ),
                encoding="utf-8",
            )
            run_dir.mkdir(parents=True)
            (run_dir / "report.json").write_text(
                json.dumps(
                    {
                        "run": {"status": "completed"},
                        "summary": {
                            "attempted_episodes": 2,
                            "scored_episodes": 2,
                            "error_episodes": 0,
                            "task_success_rate": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = skvm_reporting.write_skvm_evaluation(
                run_dir=run_dir,
                cache_dir=cache,
                target_model="vllm/test",
                adapter="bare-agent",
                variants=variants,
                manifest={"skills": [{"skill_id": skill.skill_id}]},
                runtime_stats={
                    "skvm_adapted_requests": 4,
                    "skvm_variant_source_counts": {"aot": 4},
                },
            )

            capability = report["model_capability_evaluation"]
            self.assertEqual(capability["mean_level_value"], 2.0)
            self.assertEqual(
                capability["normalized_capability_score"], 0.6667
            )
            self.assertEqual(capability["microbenchmark_pass_rate"], 0.6667)
            self.assertTrue(
                report["performance_validity"][
                    "valid_for_skvm_effect_comparison"
                ]
            )
            self.assertTrue((run_dir / "skvm_report.json").is_file())
            self.assertTrue((run_dir / "skvm_report.md").is_file())
            self.assertTrue(
                (run_dir / "skvm" / "capabilities.csv").is_file()
            )
            self.assertTrue(
                (
                    run_dir
                    / "skvm"
                    / "artifacts"
                    / skill.skill_id
                    / "aot-p1"
                    / "SKILL.md"
                ).is_file()
            )


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
