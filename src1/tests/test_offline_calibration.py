"""离线估计不伪装成实测、可恢复、可作为路由回退；不需要 GPU/模拟器。"""
from __future__ import annotations

import dataclasses
import contextlib
import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.calibration.likelihood import TransformersReferenceScorer, visual_messages
from src1.pmtskill_v2.calibration.offline import OfflineOptions, run_offline_calibration
from src1.pmtskill_v2.calibration.workflow import CalibrationOptions
from src1.pmtskill_v2.cli import build_parser, command_calibrate_skills
from src1.pmtskill_v2.core.config import RoutingConfig, load_config
from src1.pmtskill_v2.core.io import write_jsonl
from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology
from src1.pmtskill_v2.evaluation.adapters import AdapterDeploymentBinding, resolve_adapter_checkpoint
from src1.pmtskill_v2.evaluation.resume import fingerprint, model_resume_identity
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.skills.identity import model_variant_id
from src1.pmtskill_v2.skills.store import SkillStore


def _binding(root, name, *, trained=True):
    run = root / name / "training_runs" / "run-1"
    checkpoint = run / "training" / "epoch_001" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "local-base", "r": 8}))
    (checkpoint / "adapter_model.safetensors").write_bytes(name.encode())
    resolved = dataclasses.replace(resolve_adapter_checkpoint(checkpoint, require_training_state=False), run_dir=run)
    profile = ModelProfile(name, name, "http://local/v1", {"action.click": 0.8},
                           metadata={"capability_prior_source": "model-specific"})
    binding = AdapterDeploymentBinding(name, resolved, profile)
    if trained:
        write_jsonl(run / "dataset_snapshot" / "train.jsonl", [_sample(root, episode="training", task="ContactsAddContact")])
    return binding


def _sample(root, *, episode="heldout", task="ContactsAddContact"):
    image = root / "input.png"
    image.write_bytes(b"image-fixture")
    return {"messages": [{"role": "user", "content": "<image>Choose an action."},
                         {"role": "assistant", "content": 'Reason: Open it. Action: {"action_type":"click","index":1}'}],
            "images": [str(image)],
            "metadata": {"task_name": task, "episode_id": episode, "episode_outcome": "successful", "primitives": ["action.click"]}}


def _skill(identifier="skill:contact"):
    return SkillRecord(identifier, "Add contact", "", "polished", SkillStatus.ACTIVE, 3,
                       SkillTopology.from_sequence(["action.click"]), body="Open Contacts and add a contact.",
                       metadata={"calibration_tasks": ["ContactsAddContact"]})


class OfflineCalibrationTest(unittest.TestCase):
    def test_prior_is_versioned_nonobservational_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            skill = _skill()
            skill.status = SkillStatus.CANDIDATE
            store.upsert_skill(skill)
            binding = _binding(root, "student")
            target = root / "estimate"
            selection = CalibrationOptions(target)
            options = OfflineOptions(target)
            with mock.patch("src1.pmtskill_v2.calibration.runner.AndroidWorldStandaloneEvaluator.run", side_effect=AssertionError("GUI")):
                result = run_offline_calibration(store, [binding], selection, options)
                resumed = run_offline_calibration(store, [binding], selection, options)
            self.assertEqual(result["summary"]["run_id"], resumed["summary"]["run_id"])
            self.assertEqual(result["summary"]["estimated_pairs"], 1)
            self.assertFalse(result["summary"]["score_is_success_rate"])
            profile = binding.profile(base_model_path="local-base")
            estimate = store.skill_model_estimate(skill, profile)
            self.assertGreater(estimate["estimated_support_score"], 0)
            self.assertEqual(store.list_calibration_trials(), [])
            self.assertEqual(store.list_skill_model_statuses(), [])
            self.assertEqual(store.get_skill(skill.skill_id).status, SkillStatus.CANDIDATE)
            self.assertEqual(tuple(result["summary"]["capability_estimates"]), ("reasoning", "coordination", "manipulation"))
            self.assertIsNone(result["summary"]["capability_estimates"]["reasoning"]["model_scores"]["student"])
            self.assertIn("不是 SR", Path(result["report_markdown"]).read_text(encoding="utf-8"))
            (binding.checkpoint.checkpoint_dir / "adapter_model.safetensors").write_bytes(b"new weights")
            with self.assertRaisesRegex(ValueError, "续跑配置"):
                run_offline_calibration(store, [binding], selection, options)

    def test_prior_does_not_infer_ability_from_adapter_name_or_teacher_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            store.upsert_skill(_skill())
            bindings = [_binding(root, name, trained=False) for name in ("adapter_query001", "adapter_create02")]
            bindings = [dataclasses.replace(binding, template_profile=dataclasses.replace(
                binding.template_profile, metadata={"capability_prior_source": "inherited-template:teacher"})) for binding in bindings]
            result = run_offline_calibration(store, bindings, CalibrationOptions(root / "out"), OfflineOptions(root / "out"))
            rows = result["summary"]["model_skill_estimates"]
            self.assertEqual(rows[0]["estimated_support_score"], rows[1]["estimated_support_score"])
            self.assertEqual(rows[0]["known_primitive_count"], 0)
            self.assertIn("no_training_snapshot", rows[0]["warnings"])

    def test_teacher_forcing_pairs_reference_scores_and_excludes_training_overlap(self):
        class Scorer:
            def __init__(self):
                self.calls = []
            def score(self, profile, sample, skill):
                self.calls.append((sample["metadata"]["episode_id"], skill is not None))
                return -0.6 if skill else -1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            store.upsert_skill(_skill("s1"))
            store.upsert_skill(_skill("s2"))
            binding = _binding(root, "student")
            dataset = root / "reference.jsonl"
            write_jsonl(dataset, [_sample(root, episode="training"), _sample(root)])
            scorer = Scorer()
            options = OfflineOptions(root / "out", scorer="teacher-forcing", dataset=dataset)
            result = run_offline_calibration(store, [binding], CalibrationOptions(root / "out"), options, scorer=scorer)
            self.assertEqual(result["summary"]["reference_rejections"]["training_episode_overlap"], 1)
            self.assertEqual(len(scorer.calls), 3)  # 一个裸模型参考结果被两个 skill 复用。
            self.assertTrue(all(episode == "heldout" for episode, _ in scorer.calls))
            for row in result["summary"]["model_skill_estimates"]:
                self.assertAlmostEqual(row["likelihood_gain_nats_per_token"], 0.4)
                self.assertEqual(row["reference_samples"], 1)
            self.assertEqual(store.list_calibration_trials(), [])

    def test_teacher_forcing_rejects_missing_visual_data_or_task_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            store.upsert_skill(_skill())
            binding = _binding(root, "student", trained=False)
            dataset = root / "reference.jsonl"
            sample = _sample(root)
            sample["images"] = []
            write_jsonl(dataset, [sample])
            options = OfflineOptions(root / "out", scorer="teacher-forcing", dataset=dataset)
            with self.assertRaisesRegex(ValueError, "没有可用"):
                run_offline_calibration(store, [binding], CalibrationOptions(root / "out"), options, scorer=mock.Mock())
            write_jsonl(dataset, [_sample(root, task="BrowserMultiply")])
            with self.assertRaisesRegex(ValueError, "没有交集"):
                run_offline_calibration(store, [binding], CalibrationOptions(root / "out"), options, scorer=mock.Mock())

    def test_router_uses_estimate_without_fabricating_active_or_overriding_deprecation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            skill = _skill()
            store.upsert_skill(skill)
            binding = _binding(root, "student")
            profile = binding.profile(base_model_path="local-base")
            store.set_skill_model_estimate(skill, profile, {
                "estimated_support_score": 0.97, "source": "estimated",
                "model_weight_fingerprint": fingerprint(model_resume_identity(profile)),
            })
            config = RoutingConfig(latency_weight=0, switch_weight=0)
            plan = DynamicProgrammingRouter(config, store).route("Add contact", skill.topology, [profile], [skill])
            self.assertEqual(plan.steps[0].skill_id, skill.skill_id)
            self.assertEqual(plan.steps[0].expected_success, 0.97)
            self.assertEqual(plan.metadata["skill_score_sources"][plan.steps[0].step_id], "estimated_not_sr")
            store.set_skill_model_status(skill, model_id=profile.model_id, model_variant_id_value=model_variant_id(profile),
                                         status=SkillStatus.DEPRECATED, reason={})
            plan = DynamicProgrammingRouter(config, store).route("Add contact", skill.topology, [profile], [skill])
            self.assertIsNone(plan.steps[0].skill_id)

    def test_cli_offline_path_never_deploys_or_reads_android_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            store.upsert_skill(_skill())
            binding = _binding(root, "student")
            config = load_config(Path(__file__).parents[1] / "config.example.toml")
            config.paths.state_dir = root / "runtime"
            config.paths.log_dir = root / "logs"
            config.offline.trajectory_dir = root / "trajectories"
            config.offline.dataset_dir = root / "dataset"
            config.offline.output_dir = root / "checkpoints"
            args = build_parser().parse_args(["calibrate-skills", "--mode", "offline", "--adapter-paths", "unused",
                                               "--skill-database", str(store.database), "--output-dir", str(root / "out")])
            with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config), mock.patch(
                "src1.pmtskill_v2.cli._adapter_bindings", return_value=(binding,)
            ), mock.patch("src1.pmtskill_v2.cli._adapter_deployment", side_effect=AssertionError("server")), mock.patch(
                "src1.pmtskill_v2.cli._evaluation_tasks", side_effect=AssertionError("registry")
            ), mock.patch("src1.pmtskill_v2.cli._print") as printer:
                self.assertEqual(command_calibrate_skills(args), 0)
            self.assertEqual(printer.call_args.args[0]["calibration_mode"], "offline")

    def test_visual_messages_use_real_images_and_dont_score_skill_text_as_response(self):
        sample = {"messages": [{"role": "user", "content": "<image>Do it"},
                               {"role": "assistant", "content": "Action: click"}], "images": ["/input.png"]}
        messages = visual_messages(sample, _skill())
        self.assertEqual(messages[0]["content"][0], {"type": "image", "url": "/input.png"})
        self.assertIn("必须使用技能", messages[0]["content"][-1]["text"])
        self.assertEqual(messages[-1]["content"], "Action: click")
        self.assertNotIn("<image>", messages[0]["content"][-1]["text"])

    def test_teacher_forcing_masks_prompt_tokens_and_never_generates(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))
            def clone(self):
                return Tensor(list(self.values))
            def __getitem__(self, key):
                return Tensor(self.values[key[1]])
            def __setitem__(self, key, value):
                indices = range(*key[1].indices(len(self.values)))
                for index in indices:
                    self.values[index] = value
            def to(self, device):
                return self

        class Loss:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return 0.7

        calls = []
        def forward(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(loss=Loss())
        class Model:
            def get_input_embeddings(self):
                return types.SimpleNamespace(weight=types.SimpleNamespace(device="cpu"))
            def __call__(self, **kwargs):
                return forward(**kwargs)
        processor = mock.Mock()
        processor.apply_chat_template.side_effect = [
            {"input_ids": Tensor([1, 2, 3])},
            {"input_ids": Tensor([1, 2, 3, 4, 5]), "pixel_values": Tensor([9])},
        ]
        scorer = TransformersReferenceScorer()
        scorer.model = Model()
        scorer.processor = processor
        torch = types.SimpleNamespace(equal=lambda a, b: a.values == b.values, inference_mode=contextlib.nullcontext)
        sample = {"messages": [{"role": "user", "content": "<image>Do it"},
                               {"role": "assistant", "content": "Action: click"}], "images": ["/input.png"]}
        with mock.patch.object(scorer, "_activate"), mock.patch.dict(sys.modules, {"torch": torch}):
            score = scorer.score(mock.Mock(), sample, None)
        self.assertEqual(score, -0.7)  # 返回负的参考响应 token 平均 loss。
        self.assertEqual(calls[0]["labels"].values, [-100, -100, -100, 4, 5])
        self.assertFalse(calls[0]["use_cache"])
        self.assertEqual(len(calls), 1)

    def test_reference_backend_reuses_base_model_and_safe_adapter_names(self):
        base_loader = mock.Mock(return_value=object())
        processor_loader = mock.Mock(return_value=object())
        model = mock.Mock()
        peft_loader = mock.Mock(return_value=model)
        transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=types.SimpleNamespace(from_pretrained=base_loader),
            AutoProcessor=types.SimpleNamespace(from_pretrained=processor_loader),
        )
        peft = types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=peft_loader))
        first = ModelProfile("student.with.dot", "student.with.dot", "http://local/v1", {},
                             metadata={"base_model_path": "base", "evaluation_checkpoint": "a"})
        second = dataclasses.replace(first, model_id="student-2", metadata={**first.metadata, "evaluation_checkpoint": "b"})
        scorer = TransformersReferenceScorer()
        with mock.patch.dict(sys.modules, {"transformers": transformers, "peft": peft}):
            scorer._activate(first)
            scorer._activate(second)
        self.assertEqual(base_loader.call_count, 1)
        self.assertEqual(peft_loader.call_count, 1)
        self.assertEqual(model.load_adapter.call_count, 1)
        self.assertNotIn(".", peft_loader.call_args.kwargs["adapter_name"])
        self.assertEqual(model.set_adapter.call_count, 2)

    def test_estimate_weight_fingerprint_mismatch_is_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SkillStore(root / "library.sqlite3")
            store.initialize()
            skill = _skill()
            store.upsert_skill(skill)
            binding = _binding(root, "student")
            profile = binding.profile(base_model_path="local-base")
            store.set_skill_model_estimate(skill, profile, {
                "estimated_support_score": 0.97, "source": "estimated", "model_weight_fingerprint": "old",
            })
            router = DynamicProgrammingRouter(RoutingConfig(latency_weight=0), store)
            plan = router.route("Add contact", skill.topology, [profile], [skill])
            self.assertNotEqual(plan.steps[0].expected_success, 0.97)

    def test_cli_defaults_keep_original_mode_and_config_selects_offline(self):
        args = build_parser().parse_args(["calibrate-skills", "--adapter-paths", "a"])
        config = load_config(Path(__file__).parents[1] / "config.example.toml")
        self.assertIsNone(args.mode)
        self.assertEqual(config.calibration.mode, "androidworld")
        config.calibration.mode = "offline"
        with mock.patch("src1.pmtskill_v2.cli.load_config", return_value=config), mock.patch.object(
            type(config), "ensure_runtime_dirs"
        ), mock.patch("src1.pmtskill_v2.cli._open_evaluation_skill_store", return_value=mock.Mock()), mock.patch(
            "src1.pmtskill_v2.cli._command_calibrate_offline", return_value=0
        ) as offline, mock.patch("src1.pmtskill_v2.cli._adapter_deployment", side_effect=AssertionError("deployed")):
            self.assertEqual(command_calibrate_skills(args), 0)
        offline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
