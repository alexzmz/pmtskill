"""关闭完整轨迹后的报告、轻量 checkpoint 和 CLI 配置回归。"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.baseline_vdroid.config import load_vdroid_config
from src1.baseline_vdroid.evaluator import VDroidBaselineEvaluator, VDroidRunOptions, apply_cli_overrides
from src1.baseline_vdroid.reporting import (
    compact_vdroid_result_for_storage, episodes_to_vdroid_traces, summarize_vdroid_runtime,
)
from src1.pmtskill_v2.cli import _evaluation_store_trajectories, build_parser
from src1.pmtskill_v2.core.config import load_config
from src1.pmtskill_v2.evaluation.android_world import episodes_to_traces
from src1.pmtskill_v2.evaluation.reporter import compact_result_for_storage, summarize_episodes, write_evaluation_report
from src1.pmtskill_v2.evaluation.resume import EvaluationProgress


def _episode(*, success=0.0):
    return {
        "goal": "Add a contact", "task_template": "ContactsAddContact", "instance_id": 0,
        "task_seed": 42, "is_successful": success, "episode_length": 2, "run_time": 3.5,
        "exception_info": None, "aux_data": None,
        "episode_data": {
            "action_output": ['{"action_type":"click","index":1}', '{"action_type":"click","index":2}'],
            "action_output_json": [{"action_type": "click"}, {"action_type": "click"}],
            "action_raw_response": [{"_pmtskill": {"model_id": model, "skill_id": "contact",
                                                      "primitive_ids": ["action.click"], "action_committed": True}}
                                    for model in ("model-a", "model-b")],
            "action_prompt": ["PRIVATE_TRAJECTORY_PROMPT"], "summary": ["PRIVATE_TRAJECTORY_RESPONSE"],
            "screenshot": b"PRIVATE_TRAJECTORY_SCREENSHOT",
            "_vdroid": [{"candidate_count": 5, "selected_score": 0.2, "action_completed": True},
                        {"candidate_count": 7, "selected_score": 0.9, "action_completed": False}],
        },
    }


class EvaluationStorageTest(unittest.TestCase):
    def test_result_only_snapshot_preserves_metrics_without_step_data(self):
        episode = _episode()
        expected = summarize_episodes([episode], episodes_to_traces([episode]))
        compact = compact_result_for_storage(episode, episodes_to_traces([episode]))
        self.assertEqual(compact["episode_data"], {})
        encoded = json.dumps(compact)
        self.assertNotIn("PRIVATE_TRAJECTORY", encoded)
        self.assertNotIn("action_output", encoded)
        actual = summarize_episodes([compact], episodes_to_traces([compact]))
        self.assertEqual(actual, expected)
        self.assertEqual(actual["average_model_switches"], 1)
        self.assertEqual(actual["failure_reasons"], {"environment_not_satisfied": 1})

    def test_report_storage_disabled_returns_no_trace_path_or_file(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "src1.pmtskill_v2.evaluation.reporter.render_capability_plot", return_value={},
        ):
            episode = _episode(success=1.0)
            artifacts = write_evaluation_report(directory, [episode], episodes_to_traces([episode]), store_trajectories=False)
            self.assertIsNone(artifacts.traces_jsonl)
            self.assertFalse((Path(directory) / "traces.jsonl").exists())
            self.assertTrue(artifacts.summary_json.is_file())
            self.assertTrue(artifacts.report_markdown.is_file())
            self.assertEqual(artifacts.summary["success_rate_micro"], 1.0)
            self.assertFalse(artifacts.summary["metadata"]["store_trajectories"])

    def test_summary_checkpoints_resume_and_keep_failed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            task = types.SimpleNamespace(name="ContactsAddContact", goal="Add a contact", params={"task_seed": 42})
            suite = {task.name: [task]}
            harness = types.SimpleNamespace(_run_task=lambda task: _episode())
            compactor = lambda episode: compact_result_for_storage(episode, episodes_to_traces([episode]))
            initial = EvaluationProgress(target, {}, resume=False, store_trajectories=False, result_compactor=compactor)
            with initial.watch(harness, suite):
                episode = harness._run_task(task)
                initial.save_episodes([episode], f"{task.name}_0")
            self.assertEqual(list(initial.checkpoint_dir.glob("*.pkl.gz")), [])
            file = initial.checkpoint_dir / f"{task.name}_0.json"
            self.assertNotIn("PRIVATE_TRAJECTORY", file.read_text(encoding="utf-8"))
            runner = mock.Mock(side_effect=AssertionError("already completed"))
            harness._run_task = runner
            resumed = EvaluationProgress(target, {}, resume=True, store_trajectories=False, result_compactor=compactor)
            with resumed.watch(harness, suite):
                cached = harness._run_task(task)
                resumed.save_episodes([cached], f"{task.name}_0")
            runner.assert_not_called()
            self.assertEqual(cached["is_successful"], 0.0)
            self.assertEqual(initial.trace_id(episode), resumed.trace_id(cached))
            self.assertEqual(summarize_episodes([episode], episodes_to_traces([episode])),
                             summarize_episodes([cached], episodes_to_traces([cached])))
            with self.assertRaisesRegex(ValueError, "无法恢复其完整轨迹"):
                EvaluationProgress(target, {}, resume=True, store_trajectories=True)

    def test_vdroid_result_only_snapshot_keeps_verifier_statistics(self):
        episode = _episode()
        compact = compact_vdroid_result_for_storage(episode, model_id="fixed")
        self.assertNotIn("PRIVATE_TRAJECTORY", json.dumps(compact))
        self.assertEqual(summarize_vdroid_runtime([episode]), summarize_vdroid_runtime([compact]))
        self.assertEqual(summarize_episodes([episode], episodes_to_vdroid_traces([episode], model_id="fixed")),
                         summarize_episodes([compact], episodes_to_vdroid_traces([compact], model_id="fixed")))

    def test_cli_overrides_toml_and_vdroid_disables_screenshots(self):
        parser = build_parser()
        config = load_config(Path(__file__).parents[1] / "config.example.toml")
        for mode, adapters in (("evaluate-standalone", ["--adapter-path", "a"]),
                               ("evaluate-simple-skills", ["--adapter-path", "a"]),
                               ("evaluate-pmtskill", ["--adapter-paths", "a"]),
                               ("evaluate", ["--adapter-paths", "a"])):
            args = parser.parse_args([mode, *adapters])
            self.assertTrue(_evaluation_store_trajectories(config, args))
            config.evaluation.store_trajectories = False
            self.assertFalse(_evaluation_store_trajectories(config, args))
            enabled = parser.parse_args([mode, *adapters, "--store-trajectories"])
            self.assertTrue(_evaluation_store_trajectories(config, enabled))
            config.evaluation.store_trajectories = True
            disabled = parser.parse_args([mode, *adapters, "--no-store-trajectories"])
            self.assertFalse(_evaluation_store_trajectories(config, disabled))
        vdroid = parser.parse_args(["evaluate-vdroid", "--no-store-trajectories"])
        self.assertFalse(vdroid.store_trajectories)
        baseline = load_vdroid_config(Path(__file__).parents[1] / "baseline_vdroid" / "config.example.toml")
        options = apply_cli_overrides(VDroidRunOptions.from_config(baseline), store_trajectories=False,
                                      store_screenshots=True)
        request = VDroidBaselineEvaluator(baseline, options).request(Path("unused"))
        self.assertFalse(request["store_trajectories"])
        self.assertFalse(request["store_screenshots"])


if __name__ == "__main__":
    unittest.main()
