from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src1.baseline_vdroid.config import load_vdroid_config
from src1.baseline_vdroid.evaluator import (
    VDroidBaselineEvaluator,
    VDroidRunOptions,
    apply_cli_overrides,
)
from src1.baseline_vdroid.reporting import (
    episodes_to_vdroid_traces,
    summarize_vdroid_runtime,
    write_vdroid_report,
)
from src1.baseline_vdroid.worker import _instrument_vdroid_agent
from src1.pmtskill_v2.cli import build_parser


class VDroidBaselineTest(unittest.TestCase):
    @property
    def example_config(self) -> Path:
        return Path(__file__).parents[1] / "baseline_vdroid" / "config.example.toml"

    def test_example_config_is_fixed_model_no_skill_baseline(self):
        config = load_vdroid_config(self.example_config)

        self.assertEqual(config.model.model_id, "vdroid-8b-0323")
        self.assertEqual(config.model.search_iterations, 1)
        self.assertEqual(config.evaluation.max_steps, 30)
        self.assertEqual(config.evaluation.tasks, ())
        self.assertTrue(
            (config.paths.v_droid_root / "android_world" / "agents" / "vdroid.py")
            .is_file()
        )

    def test_cli_exposes_evaluate_vdroid_without_adapter_or_skill_arguments(self):
        args = build_parser().parse_args(
            [
                "evaluate-vdroid",
                "--tasks",
                "ContactsAddContact",
                "--combinations",
                "2",
                "--max-steps",
                "25",
                "--dry-run",
            ]
        )

        self.assertEqual(args.handler.__name__, "command_evaluate_vdroid")
        self.assertEqual(args.tasks, ["ContactsAddContact"])
        self.assertEqual(args.combinations, 2)
        self.assertEqual(args.max_steps, 25)
        self.assertFalse(hasattr(args, "skill_database"))
        self.assertFalse(hasattr(args, "adapter_paths"))

    def test_dry_run_declares_isolation_and_does_not_create_output(self):
        config = load_vdroid_config(self.example_config)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            options = apply_cli_overrides(
                VDroidRunOptions.from_config(config),
                tasks=["ContactsAddContact"],
                output_dir=target,
                max_steps=19,
            )
            result = VDroidBaselineEvaluator(config, options).dry_run()

            self.assertFalse(target.exists())
            self.assertEqual(result["evaluation_mode"], "vdroid")
            request = result["resolved_request"]
            self.assertFalse(request["isolation"]["skill_adaptation"])
            self.assertFalse(request["isolation"]["dynamic_model_routing"])
            self.assertEqual(request["max_steps"], 19)
            self.assertEqual(request["tasks"], ["ContactsAddContact"])

    def test_trace_conversion_uses_one_model_and_no_skills(self):
        episodes = [
            {
                "goal": "Add a contact",
                "task_template": "ContactsAddContact",
                "instance_id": 0,
                "is_successful": 1.0,
                "run_time": 4.0,
                "episode_data": {
                    "action_output": [
                        None,
                        '{"action_type":"click","index":3}',
                        '{"action_type":"status","goal_status":"complete"}',
                    ],
                    "summary": [None, "clicked", "done"],
                    "_vdroid": [
                        None,
                        {
                            "candidate_count": 11,
                            "selected_score": 0.8,
                            "action_completed": False,
                        },
                        {
                            "candidate_count": 7,
                            "selected_score": 0.9,
                            "action_completed": False,
                        },
                    ],
                },
            }
        ]

        traces = episodes_to_vdroid_traces(episodes, model_id="vdroid-fixed")
        metrics = summarize_vdroid_runtime(episodes)

        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0].events), 2)
        self.assertEqual({event.model_id for event in traces[0].events}, {"vdroid-fixed"})
        self.assertTrue(all(event.skill_id is None for event in traces[0].events))
        self.assertEqual(metrics["candidate_actions_verified"], 18)
        self.assertEqual(metrics["decision_count"], 2)
        self.assertEqual(metrics["model_switches"], 0)
        self.assertEqual(metrics["skill_calls"], 0)

    def test_worker_is_not_imported_by_cli_parser(self):
        # Parsing and --dry-run must remain usable without CUDA/ray/vLLM imports.
        import src1.baseline_vdroid.evaluator as evaluator

        source = inspect.getsource(evaluator)
        self.assertNotIn("from android_world", source)
        self.assertNotIn("import ray", source)

    def test_worker_clamps_vdroid_internal_plus_ten_budget(self):
        class FakeAgent:
            def score_by_batch(self, *args, **kwargs):
                return []

            def _action_completion(self, node):
                return node

            def search(self):
                return self.depth_limit, self.explore_step_count_limit

        _instrument_vdroid_agent(FakeAgent, max_steps=25)
        agent = FakeAgent()
        agent.depth_limit = 35
        agent.explore_step_count_limit = 35

        self.assertEqual(agent.search(), (25, 25))

    def test_report_uses_common_schema_and_adds_vdroid_metrics(self):
        episodes = [
            {
                "goal": "Add a contact",
                "task_template": "ContactsAddContact",
                "instance_id": 0,
                "agent_name": "vdroid:fixed",
                "is_successful": 0.0,
                "episode_length": 1,
                "run_time": 2.0,
                "exception_info": None,
                "episode_data": {
                    "action_output": ['{"action_type":"click","index":3}'],
                    "summary": ["clicked"],
                    "_vdroid": [
                        {"candidate_count": 8, "selected_score": 0.4}
                    ],
                },
            }
        ]
        metadata = {
            "evaluation_mode": "vdroid",
            "model_id": "vdroid-fixed",
            "assistant_model": "gpt-4o",
            "skill_database": None,
            "tasks": ["ContactsAddContact"],
            "n_task_combinations": 1,
            "seed": 42,
            "max_steps": 30,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "src1.pmtskill_v2.evaluation.reporter.render_capability_plot",
            return_value={"png": "capability_distribution.png"},
        ):
            artifacts = write_vdroid_report(directory, episodes, metadata=metadata)

            self.assertEqual(artifacts.summary["success_rate_micro"], 0.0)
            self.assertEqual(
                artifacts.summary["vdroid"]["candidate_actions_verified"], 8
            )
            markdown = artifacts.report_markdown.read_text(encoding="utf-8")
            self.assertIn("# V-Droid AndroidWorld Baseline 评测报告", markdown)
            self.assertIn("动态模型/adapter 切换：关闭", markdown)
            self.assertIn("PORTS/SKVM 技能检索与技能适配：关闭", markdown)


if __name__ == "__main__":
    unittest.main()

