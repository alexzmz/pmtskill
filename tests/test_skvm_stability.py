from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_skvm_preparation import legacy_plan, make_kernel, save_variant
from tests.test_task_runner_skvm import FakeLLM, make_skill, runner, skvm_reporting


def stable_kernel(root, skills, mode="auto", retries=1):
    kernel = make_kernel(root, skills, mode)
    kernel.args.skvm_failure_policy = "fallback"
    kernel.args.skvm_compile_retries = retries
    kernel.args.skill_mode = "inject"
    kernel.args.skvm_variant_policy = "aot"
    kernel.failure_policy = "fallback"
    return kernel


def save_profile(kernel, *, partial=False):
    path = kernel._profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": kernel.target_model, "harness": "bare-agent", "isPartial": partial,
        "capabilities": {"tool.browser": "L2"}, "details": [],
    }), encoding="utf-8")


def bad_plan(skill):
    plan = legacy_plan(skill)
    plan.update(guardPassed=False, guardViolations=["Frontmatter modified"])
    return plan


def wrapper(root, variants):
    return runner.SkVMAdaptiveWrapper(
        FakeLLM(), variants=variants, target_model="vllm/test", mode="inject", variant_policy="aot",
        max_skills=2, max_skill_chars=12000, trace_path=root / "adaptations.jsonl", trace_include_goal=False)


class StablePreparationTest(unittest.TestCase):
    def test_bad_p1p3_uses_cached_p1_without_recompiling_or_faking_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill])
            save_profile(kernel)
            bad = save_variant(kernel, skill, (1, 3), bad_plan(skill))
            save_variant(kernel, skill, (1,), legacy_plan(skill))
            before = (bad / "compilation-plan.json").read_bytes()
            with mock.patch.object(kernel, "_run") as cli:
                variants = kernel.prepare()
            cli.assert_not_called()
            aot = next(v for v in variants if v.source == "aot")
            self.assertEqual(aot.tag, "p1")
            self.assertIn("guard not passed", aot.fallback_reason)
            self.assertEqual((bad / "compilation-plan.json").read_bytes(), before)
            self.assertEqual(kernel.preparation_summary["alternate_aot_skills"], 1)
            self.assertEqual(kernel.preparation_summary["guard_failed_variants"], 1)
            self.assertEqual(kernel.preparation_summary["aot_coverage"], 1)

    def test_valid_p1p3_is_preferred_and_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            for passes in kernel.pass_sets:
                save_variant(kernel, skill, passes, legacy_plan(skill))
            variants = kernel.prepare()
            aot = next(v for v in variants if v.source == "aot")
            self.assertEqual(aot.tag, "p1p3")
            self.assertIsNone(aot.fallback_reason)
            self.assertTrue(aot.path.is_relative_to(kernel.run_dir))
            self.assertTrue((kernel.run_dir / "skvm/preparation.json").is_file())

    def test_two_invalid_variants_fall_back_to_original_without_dropping_skill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            for passes in kernel.pass_sets:
                save_variant(kernel, skill, passes, bad_plan(skill))
            with mock.patch.object(kernel, "_run") as cli:
                variants = kernel.prepare()
            cli.assert_not_called()
            self.assertEqual(len(variants), 1)
            self.assertEqual(variants[0].source, "original")
            self.assertEqual(variants[0].content, skill.content)
            self.assertEqual(variants[0].plan, {})
            self.assertEqual(kernel.preparation_summary["original_fallback_skills"], 1)
            self.assertEqual(kernel.preparation_summary["aot_coverage"], 0)

    def test_compile_retries_are_bounded_then_original_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill])
            save_profile(kernel)
            def compile_bad(command, label):
                passes = (1, 3) if "--pass=1,3" in command else (1,)
                save_variant(kernel, skill, passes, bad_plan(skill))
            with mock.patch.object(kernel, "_run", side_effect=compile_bad) as cli:
                variants = kernel.prepare()
            self.assertEqual(cli.call_count, 4)  # two variants × (one attempt + one retry)
            self.assertEqual(kernel.preparation_summary["compile_attempts"], 4)
            self.assertEqual(variants[0].source, "original")

    def test_partial_nonzero_compile_keeps_successes_and_isolates_bad_skill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = [make_skill(root / name, name) for name in ("good", "bad")]
            kernel = stable_kernel(root, skills, retries=0)
            save_profile(kernel)
            def partially_bad(command, label):
                passes = (1, 3) if "--pass=1,3" in command else (1,)
                if "--pass=1,3" in command:
                    save_variant(kernel, skills[0], passes, legacy_plan(skills[0]))
                else:
                    self.assertIn(f"--skill={skills[1].path}", command)
                raise runner.SkVMCommandError("one skill failed", recoverable=True)
            with mock.patch.object(kernel, "_run", side_effect=partially_bad):
                variants = kernel.prepare()
            self.assertEqual({(v.skill.skill_id, v.source) for v in variants},
                             {("good", "aot"), ("good", "original"), ("bad", "original")})
            self.assertEqual(kernel.preparation_summary["skills_with_aot"], 1)
            self.assertEqual(kernel.preparation_summary["aot_coverage"], .5)

    def test_provider_or_unexpected_error_does_not_become_original_fallback(self):
        for failure in (runner.SkVMCommandError("API error 401", recoverable=False),
                        RuntimeError("unexpected implementation failure")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                kernel = stable_kernel(root, [make_skill(root / "source")])
                save_profile(kernel)
                with mock.patch.object(kernel, "_run", side_effect=failure), self.assertRaises(type(failure)):
                    kernel.prepare()
                self.assertFalse((kernel.run_dir / "skvm/preparation.json").exists())

    def test_zero_exit_auth_failure_is_classified_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = stable_kernel(root, [])
            process = SimpleNamespace(stdout=io.StringIO("[1/1] skill: FAILED: API error 401: invalid key\n"),
                                      wait=lambda: 0)
            with mock.patch.object(kernel, "_resolve_command", return_value=["skvm"]), mock.patch.object(
                    runner.subprocess, "Popen", return_value=process), self.assertRaises(runner.SkVMCommandError) as caught:
                kernel._run(["aot-compile"], "test")
            self.assertFalse(caught.exception.recoverable)

    def test_frozen_original_choice_does_not_silently_upgrade_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            original = kernel.prepare()[0]
            save_variant(kernel, skill, (1, 3), legacy_plan(skill))
            (kernel.run_dir / "checkpoints").mkdir()
            (kernel.run_dir / "checkpoints/task.json").write_text("{}", encoding="utf-8")
            resumed = stable_kernel(root, [skill])
            with mock.patch.object(resumed, "_run") as cli:
                variants = resumed.prepare()
            cli.assert_not_called()
            self.assertEqual(variants[0].source, "original")
            self.assertEqual(variants[0].sha256, original.sha256)
            self.assertEqual(variants[0].path, original.path)

    def test_snapshot_corruption_and_force_after_episodes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            variant = kernel.prepare()[0]
            (kernel.run_dir / "checkpoints").mkdir()
            (kernel.run_dir / "checkpoints/task.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot rebuild"):
                stable_kernel(root, [skill], "force").prepare()
            variant.path.write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot was modified"):
                stable_kernel(root, [skill]).prepare()

    def test_partial_tcp_is_not_treated_as_low_capability_or_complete_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = stable_kernel(root, [make_skill(root / "source")], "reuse")
            save_profile(kernel, partial=True)
            with self.assertRaisesRegex(RuntimeError, "complete TCP"):
                kernel.prepare()

    def test_changed_experiment_cannot_reuse_frozen_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            kernel.prepare()
            resumed = stable_kernel(root, [skill])
            resumed.pass_sets = [(1,)]
            with self.assertRaisesRegex(ValueError, "does not match"):
                resumed.prepare()

    def test_strict_mode_still_requires_all_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = make_kernel(root, [skill], "reuse")
            save_variant(kernel, skill, (1,), legacy_plan(skill))
            save_variant(kernel, skill, (1, 3), bad_plan(skill))
            with self.assertRaises(FileNotFoundError):
                kernel.prepare()


class FallbackReportingTest(unittest.TestCase):
    def test_aggregate_restore_replaces_trace_counters_without_double_counting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            variant = runner.SkillVariant(skill, "original", "original", skill.path,
                                           skill.content, skill.sha256, fallback_reason="no valid AOT")
            first = wrapper(root, [variant])
            first.predict("The current user goal/request is: Add a contact\n\nAct.")
            stats = first.get_stats()
            resumed = wrapper(root, [variant])  # Trace already restored once.
            resumed.restore_runtime_stats(stats)
            self.assertEqual(resumed.get_stats()["skvm_injected_requests"], 1)
            resumed.predict("The current user goal/request is: Add a contact\n\nAct.")
            self.assertEqual(resumed.get_stats()["skvm_injected_requests"], 2)
            self.assertEqual(resumed.get_stats()["skvm_adapted_requests"], 0)
            bad = dict(stats, skvm_adapted_requests=-1)
            with self.assertRaises(ValueError):
                resumed.restore_runtime_stats(bad)
            self.assertEqual(resumed.get_stats()["skvm_injected_requests"], 2)

    def test_original_fallback_is_not_counted_as_adaptation_even_after_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill = make_skill(root / "source")
            kernel = stable_kernel(root, [skill], "reuse")
            save_profile(kernel)
            variants = kernel.prepare()
            adaptive = wrapper(root, variants)
            adaptive.predict("The current user goal/request is: Add a contact\n\nAct.")
            stats = adaptive.get_stats()
            self.assertEqual(stats["skvm_adapted_requests"], 0)
            self.assertEqual(stats["skvm_original_only_requests"], 1)
            self.assertEqual(stats["skvm_fallback_selection_counts"], {"original": 1})
            self.assertIn("ORIGINAL skills are unadapted", adaptive.base_llm.prompts[0])
            resumed = wrapper(root, variants)
            self.assertEqual(resumed.get_stats()["skvm_original_only_requests"], 1)
            self.assertEqual(resumed.get_stats()["skvm_adapted_requests"], 0)

    def test_real_preparation_runtime_and_report_keep_official_scores_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = [make_skill(root / name, name) for name in ("contacts", "calendar")]
            kernel = stable_kernel(root, skills, "reuse")
            save_profile(kernel)
            save_variant(kernel, skills[0], (1,), legacy_plan(skills[0]))
            save_variant(kernel, skills[0], (1, 3), bad_plan(skills[0]))
            variants = kernel.prepare()
            adaptive = wrapper(root, variants)
            adaptive.predict("The current user goal/request is: Add a contact and calendar event\n\nAct.")
            # Simulate the normal reporter's official task results: no filtering
            # or success edits when preparation degraded for one of the skills.
            (kernel.run_dir / "summary.json").write_text(json.dumps({
                "evaluation_available": True, "episodes_total": 2, "episodes_evaluated": 2,
                "success_rate_micro": .5, "metadata": {},
            }), encoding="utf-8")
            report = skvm_reporting.write_skvm_evaluation(
                run_dir=kernel.run_dir, cache_dir=kernel.cache_dir, target_model=kernel.target_model,
                adapter="bare-agent", variants=variants, manifest=kernel.manifest(variants),
                runtime_stats=adaptive.get_stats())
            self.assertEqual(report["android_world"]["summary"]["scored_episodes"], 2)
            self.assertEqual(report["android_world"]["summary"]["task_success_rate"], .5)
            self.assertEqual(report["preparation"]["original_fallback_skills"], 1)
            self.assertEqual(report["preparation"]["alternate_aot_skills"], 1)
            self.assertEqual(report["preparation"]["aot_coverage"], .5)
            self.assertIn("Original (unadapted) fallbacks", (kernel.run_dir / "skvm_report.md").read_text(encoding="utf-8"))
            self.assertTrue(report["model_capability_evaluation"]["profile_source_path"].endswith("tcp.json"))
            self.assertEqual(report["runtime_adaptation"]["skvm_adapted_selection_coverage"], .5)

    def test_all_original_report_does_not_claim_skvm_adaptation_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = stable_kernel(root, [make_skill(root / "source")], "reuse")
            save_profile(kernel)
            variants = kernel.prepare()
            adaptive = wrapper(root, variants)
            adaptive.predict("The current user goal/request is: Add a contact\n\nAct.")
            (kernel.run_dir / "summary.json").write_text(json.dumps({
                "evaluation_available": True, "episodes_total": 1, "episodes_evaluated": 1,
                "success_rate_micro": 0, "metadata": {},
            }), encoding="utf-8")
            report = skvm_reporting.write_skvm_evaluation(
                run_dir=kernel.run_dir, cache_dir=kernel.cache_dir, target_model=kernel.target_model,
                adapter="bare-agent", variants=variants, manifest=kernel.manifest(variants),
                runtime_stats=adaptive.get_stats())
            self.assertEqual(report["android_world"]["summary"]["task_success_rate"], 0)
            self.assertFalse(report["performance_validity"]["valid_for_skvm_effect_comparison"])

    def test_batch_comparison_declares_original_fallback_instead_of_hiding_it(self):
        from src1 import batch_evaluate as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps({
                "evaluation_available": True, "episodes_total": 1, "episodes_evaluated": 1,
                "success_rate_micro": 0, "metadata": {
                    "skvm_tool_protocol": batch.SKVM_TOOL_PROTOCOL,
                    "skvm_context_policy": batch.SKVM_CONTEXT_POLICY,
                    "skvm_stability_policy": batch.SKVM_STABILITY_POLICY,
                    "skvm_compilation": {"aot_coverage": 0, "original_fallback_skills": 1},
                }}), encoding="utf-8")
            jobs = [{"method": "skvm", "output_dir": str(root)}]
            batch.write_comparison(root, jobs, {"skvm": {"status": "completed"}}, 1)
            row = json.loads((root / "comparison.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(row["skvm_original_fallback_skills"], 1)
            self.assertEqual(row["micro_sr"], 0)
            self.assertIn("不代表 SKVM 适配效果", (root / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
