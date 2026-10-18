from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_task_runner_skvm import make_plan, make_skill, runner
from skvm_artifacts import annotation_errors, normalize_compilation_plan


def legacy_plan(skill, *, primitive="tool.browser", parallel=False):
    # Same flat schema as the two user-supplied compilation-plan.json files.
    return {
        "skillName": skill.skill_id, "model": "vllm/test", "harness": "bare-agent",
        "compiledAt": "2026-10-18T05:25:21.288Z",
        "scr": {"skillName": skill.skill_id, "purposes": [{
            "id": "set-date", "description": "Set date on visible picker",
            "currentPath": {"primitives": [{"id": primitive, "minLevel": "L2"}]},
            "alternativePaths": [],
        }]},
        "gaps": [], "dependencies": [],
        "pass3": {"dag": {"steps": [{"id": "a"}] if parallel else [], "parallelism": []}},
        "guardPassed": True, "guardViolations": [],
    }


def make_kernel(root, skills, mode="auto"):
    args = SimpleNamespace(
        skvm_cache_dir=root / "cache", skvm_target_model="vllm/test",
        skvm_compiler_model="target", skvm_aot_pass_sets=["1", "1,3"], skvm_prepare=mode,
        skvm_profile_instances=3, skvm_profile_concurrency=1, skvm_profile_timeout_ms=120000,
        skvm_profile_primitives=None, skvm_compile_concurrency=1, skvm_compile_timeout_ms=300000,
        vllm_api_key="local", skvm_command=None,
    )
    return runner.SkVMKernel(args, skills=skills, run_dir=root / "run",
                            vllm_base_url="http://127.0.0.1:8002/v1", served_model="test")


def save_variant(kernel, skill, passes, plan=None):
    folder = kernel._variant_dir(skill.skill_id, passes)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(skill.content + "\nAdapted.\n", encoding="utf-8")
    if plan is not None:
        (folder / "compilation-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return folder


class CompilationPlanTest(unittest.TestCase):
    def test_user_flat_schema_is_normalized_without_fabricating_pass_runs(self):
        skill = SimpleNamespace(skill_id="skill-c8344afe6ffc2226f751")
        for passes in ((1,), (1, 3)):
            raw = legacy_plan(skill)
            before = copy.deepcopy(raw)
            plan = normalize_compilation_plan(raw)
            self.assertEqual(plan["artifacts"]["scr"], raw["scr"])
            self.assertEqual(plan["artifacts"]["dag"], raw["pass3"]["dag"])
            self.assertNotIn("passRuns", plan)
            self.assertEqual(raw, before)
            self.assertEqual(annotation_errors(plan, skill_id=skill.skill_id, model="vllm/test",
                                               harness="bare-agent", passes=passes), [])

    def test_incomplete_modern_plan_does_not_fall_back_to_flat_fields(self):
        skill = SimpleNamespace(skill_id="test")
        plan = legacy_plan(skill)
        plan["passRuns"] = {"rewrite-skill": {"status": "failed"}}
        normalized = normalize_compilation_plan(plan)
        errors = annotation_errors(normalized, skill_id="test", model="vllm/test", harness="bare-agent")
        self.assertIn("no valid SCR purposes", errors)
        self.assertIn("rewrite-skill pass not successful", errors)

    def test_guard_mismatch_missing_completion_and_failed_pass3_are_rejected(self):
        skill = SimpleNamespace(skill_id="test")
        for mutate in (lambda p: p.update(guardPassed=False),
                       lambda p: p.update(model="vllm/wrong"),
                       lambda p: p.pop("compiledAt"),
                       lambda p: p["scr"].update(purposes=[])):
            plan = legacy_plan(skill)
            mutate(plan)
            self.assertTrue(annotation_errors(normalize_compilation_plan(plan), skill_id="test",
                                              model="vllm/test", harness="bare-agent"))
        modern = make_plan(parallel=True)
        modern["passRuns"]["extract-parallelism"]["status"] = "failed"
        self.assertIn("extract-parallelism pass not successful", annotation_errors(
            modern, skill_id="test", model="vllm/test", harness="bare-agent", passes=(1, 3)))


class PreparationCacheTest(unittest.TestCase):
    def test_legacy_reuse_keeps_json_unchanged_and_exposes_real_scr_to_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = make_skill(root / "source")
            kernel = make_kernel(root, [skill], "reuse")
            files = []
            for passes in kernel.pass_sets:
                folder = save_variant(kernel, skill, passes, legacy_plan(skill))
                files.append(folder / "compilation-plan.json")
            before = [p.read_bytes() for p in files]
            with mock.patch.object(kernel, "_run") as cli:
                variants = kernel.prepare()
            cli.assert_not_called()
            self.assertEqual([p.read_bytes() for p in files], before)
            aot = [v for v in variants if v.source == "aot"]
            self.assertEqual(len(aot), 2)
            self.assertTrue(all(v.manifest()["purpose_count"] == 1 for v in aot))
            self.assertTrue(all("passRuns" not in v.plan for v in aot))
            self.assertEqual(runner._variant_purposes(aot[0])[0]["id"], "set-date")

    def test_auto_retries_only_missing_plan_not_successful_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad = make_skill(root / "good", "good"), make_skill(root / "bad", "bad")
            kernel = make_kernel(root, [good, bad])
            for passes in kernel.pass_sets:
                save_variant(kernel, good, passes, legacy_plan(good))
            save_variant(kernel, bad, (1,))  # Native pre-copy, no completed plan.
            save_variant(kernel, bad, (1, 3), legacy_plan(bad))
            calls = []
            def run(command, label):
                calls.append((command, label))
                if command[0] == "aot-compile":
                    self.assertIn(f"--skill={bad.path}", command)
                    self.assertIn("--pass=1", command)
                    save_variant(kernel, bad, (1,), legacy_plan(bad))
            with mock.patch.object(kernel, "_run", side_effect=run):
                variants = kernel.prepare()
            self.assertEqual(sum(command[0] == "aot-compile" for command, _ in calls), 1)
            self.assertEqual(len([v for v in variants if v.source == "aot"]), 4)

    def test_corrupt_json_and_guard_failures_are_missing_but_originals_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = make_skill(root / "source")
            kernel = make_kernel(root, [skill], "reuse")
            first = save_variant(kernel, skill, (1,))
            (first / "compilation-plan.json").write_text("{", encoding="utf-8")
            failed = legacy_plan(skill)
            failed["guardPassed"] = False
            save_variant(kernel, skill, (1, 3), failed)
            self.assertEqual(kernel._missing_variants((1,)), [skill])
            self.assertEqual(kernel._missing_variants((1, 3)), [skill])
            self.assertEqual([v.source for v in kernel.load_variants(include_aot=True)], ["original"])
            with self.assertRaisesRegex(FileNotFoundError, "Missing cached"):
                kernel.prepare()

    def test_recompile_failure_cannot_pair_old_plan_with_precopied_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = make_skill(root / "source")
            kernel = make_kernel(root, [skill])
            folder = save_variant(kernel, skill, (1,), legacy_plan(skill))
            old_text = (folder / "SKILL.md").read_text(encoding="utf-8")
            def fail(*args):
                (folder / "SKILL.md").write_text(skill.content, encoding="utf-8")
                raise RuntimeError("partial compile failure")
            with mock.patch.object(kernel, "_run", side_effect=fail), self.assertRaises(RuntimeError):
                kernel._compile_variants([skill], (1,), ["aot-compile"])
            self.assertTrue(kernel._cached_plan(skill, (1,))[1])
            backups = list((kernel.cache_dir / "kernel-backups").rglob("SKILL.md"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), old_text)

    def test_context_tainted_profile_is_repaired_selectively_and_dependent_aot_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skills = [make_skill(root / "affected", "affected"), make_skill(root / "healthy", "healthy")]
            kernel = make_kernel(root, skills)
            for skill in skills:
                for passes in kernel.pass_sets:
                    save_variant(kernel, skill, passes, legacy_plan(
                        skill, primitive="tool.browser" if skill.skill_id == "affected" else "reason.planning"))
            profile = {"details": [{"primitiveId": "tool.browser", "levelResults": [{
                "failureDetails": ["Error: max_tokens must be at least 1, got -169"]}]}]}
            kernel._profile_path().parent.mkdir(parents=True)
            kernel._profile_path().write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not real capability"):
                make_kernel(root, skills, "reuse").prepare()
            calls = []
            def run(command, label):
                calls.append(command)
                if command[0] == "profile" and "--force" in command:
                    self.assertIn("--primitives=tool.browser", command)
                    profile["details"][0]["levelResults"][0]["failureDetails"] = []
                    kernel._profile_path().write_text(json.dumps(profile), encoding="utf-8")
                if command[0] == "aot-compile":
                    self.assertIn(f"--skill={skills[0].path}", command)
                    passes = (1, 3) if "--pass=1,3" in command else (1,)
                    save_variant(kernel, skills[0], passes, legacy_plan(skills[0]))
            with mock.patch.object(kernel, "_run", side_effect=run):
                kernel.prepare()
            self.assertEqual(sum(c[0] == "aot-compile" for c in calls), 2)
            self.assertEqual(sum("--force" in c for c in calls), 1)
            self.assertEqual(json.loads((kernel.cache_dir / "aot-profile-repair.json").read_text())["pending_primitives"], [])
            self.assertTrue(kernel._variant_stamp_path(skills[0], (1,)).is_file())

    def test_partial_repaired_batch_stamps_successes_for_next_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skills = [make_skill(root / name, name) for name in ("one", "two")]
            kernel = make_kernel(root, skills)
            kernel._repair_primitives = {"tool.browser"}
            for skill in skills:
                save_variant(kernel, skill, (1,), legacy_plan(skill))
            def partial(*args):
                save_variant(kernel, skills[0], (1,), legacy_plan(skills[0]))
                raise RuntimeError("one failure")
            with mock.patch.object(kernel, "_run", side_effect=partial), self.assertRaises(RuntimeError):
                kernel._compile_variants(skills, (1,), ["aot-compile"])
            self.assertEqual(kernel._missing_variants((1,)), [skills[1]])

    def test_unresolved_profile_context_error_stops_before_aot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = make_skill(root / "source")
            kernel = make_kernel(root, [skill])
            kernel._profile_path().parent.mkdir(parents=True)
            kernel._profile_path().write_text(json.dumps({"details": [{
                "primitiveId": "tool.browser", "levelResults": [{"failureDetails": [
                    "Error: SKVM_CONTEXT_WINDOW_EXCEEDED"]}],
            }]}), encoding="utf-8")
            with mock.patch.object(kernel, "_run") as cli, self.assertRaisesRegex(
                    RuntimeError, "AndroidWorld evaluation not started"):
                kernel.prepare()
            self.assertEqual(cli.call_count, 1)
            self.assertEqual(cli.call_args.args[0][0], "profile")
            self.assertTrue((kernel.run_dir / "skvm/protocol-compat.json").is_file())
            self.assertTrue((kernel.cache_dir / "aot-profile-repair.json").is_file())


if __name__ == "__main__":
    unittest.main()
