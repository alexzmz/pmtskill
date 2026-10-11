"""分层经验挖掘的负例、去重、参数变化及在线短提示回归。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import EvidenceMiningConfig, MaintenanceConfig, RoutingConfig, load_config
from src1.pmtskill_v2.core.io import load_primitives
from src1.pmtskill_v2.core.models import ExecutionPlan, ExecutionTrace, ModelProfile, RouteStep, SkillRecord, SkillStatus, SkillTopology, TraceEvent
from src1.pmtskill_v2.inference.model_pool import ModelPool
from src1.pmtskill_v2.online.executor import RoutedVLWrapper
from src1.pmtskill_v2.online.planner import KeywordSkillPlanner, PlannerPipeline, PrimitiveTopologyGenerator, scope_hierarchical_skills
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.skills.evidence_miner import COMPILER_ID, build_evidence_library, canonical_operations, export_evidence_library
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore


class EvidenceMinerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SkillStore(self.root / "library.sqlite3")
        self.store.initialize()
        self.store.sync_primitive_catalog(load_primitives())
        self.rules = EvidenceMiningConfig()

    def tearDown(self):
        self.temp.cleanup()

    def trace(self, task="NotesCreate", value="Alice", instance=0, success=True, invalid=False):
        units = [
            ("action.open_app", "Open Notes.", {"action_type": "open_app", "app_name": "Notes"}),
            ("action.click", "Click 'New' to create a note.", {"action_type": "click", "index": 2}),
            ("action.type", f"Type '{value}' into the title field.", {"action_type": "input_text", "text": value}),
            ("action.click", "Click 'Save'.", {"action_type": "click", "index": 8}),
            ("control.finish", "Verify completion.", {"action_type": "status", "goal_status": "complete"}),
        ]
        return ExecutionTrace.new(
            f"Create a note named '{value}' in Notes.", task, success,
            [TraceEvent(i, "teacher", None, (p,), success, 1, action="Reason: " + instruction + "\nAction: " + json.dumps(action),
                        metadata={"action_committed": True, "planned_unit": {"instruction": instruction, "primitive_ids": [p]}})
             for i, (p, instruction, action) in enumerate(units)],
            metadata={"episode_data_valid": not invalid, "instance_id": instance,
                      "collection_context": {"suite_seed": 42}, "teacher_model_id": "teacher"},
        )

    def append(self, *traces):
        for trace in traces:
            self.store.append_trace(trace, update_skill_metrics=False)

    def build(self, *traces):
        self.append(*traces)
        return build_evidence_library(self.store, self.rules)[1]

    def four_successes(self):
        return [self.trace(task, value, instance) for task in ("NotesCreate", "NotesCreateOther")
                for instance, value in enumerate(("Alice", "Bob"))]

    def skills(self, level):
        return [s for s in self.store.list_skills(kind="polished") if s.level == level]

    def test_semantic_dedup_uses_actual_action_not_multi_action_plan(self):
        traces = self.four_successes()
        traces[0].events[3].metadata["planned_unit"] = {"primitive_ids": ["action.back", "action.click", "action.type"],
                                                        "instruction": "Go back then Delete the note then type its name."}
        report = self.build(*traces)
        self.assertEqual(report["levels"]["L1"], 5)
        self.assertEqual(report["active_levels"]["L1"], 5)
        self.assertTrue(all(s.topology.primitive_sequence() == ("action.click",) for s in self.skills(1) if "save" in s.name))
        self.assertFalse(any("delete" in s.name for s in self.skills(1)))

    def test_failed_episode_is_task_denominator_even_if_path_differs(self):
        failure = self.trace(value="Carol", instance=2, success=False)
        failure.events = failure.events[:1]
        self.build(*self.four_successes(), failure)
        skill = next(s for s in self.skills(3) if s.metadata["task_names"] == ["NotesCreate"])
        self.assertEqual(skill.status, SkillStatus.CANDIDATE)
        self.assertEqual(skill.metadata["evidence"]["trials"], 3)
        self.assertAlmostEqual(skill.metadata["evidence"]["contextual_episode_success_rate"], 2 / 3)

    def test_visited_fragment_failures_are_counted_but_not_local_truth(self):
        self.build(*self.four_successes(), self.trace(value="Carol", instance=2, success=False))
        operation = next(s for s in self.skills(1) if s.metadata["semantic_roles"] == ["save"])
        self.assertEqual(operation.metadata["evidence"]["failures"], 1)
        self.assertEqual(operation.metadata["evidence_scope"], "episodes_containing_semantic_fragment")
        self.assertEqual(self.store.skill_metrics(operation.skill_id)["trials"], 0)
        self.assertEqual(self.store.list_skill_model_statuses(), [])

    def test_duplicate_conflicting_collection_counts_once_as_failure(self):
        report = self.build(self.trace(), self.trace(), self.trace(success=False), self.trace(value="Bob", instance=1))
        self.assertEqual(report["episode_cohorts"], 2)
        self.assertEqual(report["duplicate_trace_count"], 2)
        self.assertEqual(len(report["conflicting_episode_ids"]), 1)
        task = self.skills(3)[0]
        self.assertEqual(task.metadata["evidence"]["successes"], 1)
        self.assertEqual(task.metadata["evidence"]["failures"], 1)
        self.assertEqual(task.status, SkillStatus.CANDIDATE)

    def test_rejected_infrastructure_does_not_poison_denominator(self):
        report = self.build(*self.four_successes(), self.trace(instance=99, success=False, invalid=True))
        self.assertEqual(report["rejected_traces"], 1)
        self.assertEqual(report["episode_cohorts"], 4)
        self.assertTrue(all(s.status == SkillStatus.ACTIVE for s in self.skills(3)))

    def test_parameterized_task_needs_goal_variation(self):
        self.build(self.trace(instance=0), self.trace(instance=1))
        skill = self.skills(3)[0]
        self.assertEqual(skill.status, SkillStatus.CANDIDATE)
        self.assertIn("successful_goal_variants<2", skill.metadata["admission_reasons"])

    def test_fixed_goal_does_not_require_artificial_parameters(self):
        traces = [self.trace(instance=i) for i in (0, 1)]
        for trace in traces:
            trace.goal = "Enable wifi in Android Settings."
            trace.events = [trace.events[0], trace.events[-1]]
        self.build(*traces)
        self.assertEqual(self.skills(3)[0].status, SkillStatus.ACTIVE)

    def test_subflows_are_repeated_closed_patterns_not_arbitrary_chunks(self):
        report = self.build(*self.four_successes())
        self.assertGreater(report["redundant_closed_subflows_discarded"], 0)
        self.assertEqual(report["active_levels"]["L2"], 2)
        self.assertTrue(all(s.metadata["operation_count"] == 4 for s in self.skills(2)))
        self.assertTrue(all(s.metadata["evidence"]["successful_task_count"] == 2 for s in self.skills(2)))

    def test_unknown_action_cannot_bridge_a_subflow_or_complete_task(self):
        traces = self.four_successes()
        for trace in traces:
            trace.events[2].action = 'Action: {"action_type":"unknown"}'
        report = self.build(*traces)
        self.assertEqual(report["levels"]["L3"], 0)
        self.assertTrue(all(s.metadata["operation_count"] == 2 for s in self.skills(2)))
        self.assertFalse(any(s.metadata["semantic_roles"] == ["create", "save"] for s in self.skills(2)))

    def test_save_and_delete_remain_distinct(self):
        trace = self.trace()
        trace.events[3].action = 'Reason: I will click Delete.\nAction: {"action_type":"click","index":8}'
        trace.events[3].metadata["planned_unit"]["instruction"] = "Click Delete."
        self.build(self.trace(), trace)
        self.assertEqual({s.metadata["semantic_roles"][0] for s in self.skills(1)} & {"save", "delete"}, {"save", "delete"})

    def test_raw_taxonomy_is_not_derivation_or_evidence(self):
        raw = SkillRecord("raw:note", "note", "Android notes", "raw", SkillStatus.IMPORTED, 1,
                          SkillTopology.from_sequence(["action.type"]), metadata={"approved_for_planning": True})
        self.store.upsert_skill(raw)
        self.build(*self.four_successes())
        self.assertEqual(self.store.get_skill(raw.skill_id).to_dict(), raw.to_dict())
        relations = self.store.list_skill_relations()
        related = [r for r in relations if r["relation_type"] == "related_raw"]
        self.assertTrue(related)
        self.assertTrue(all(r["evidence_count"] == 0 for r in related))
        self.assertFalse(any(r["relation_type"] == "derived_from_raw" for r in relations))

    def test_repeat_build_is_idempotent_and_config_tightening_demotes(self):
        self.build(*self.four_successes())
        before = {s.skill_id: (s.version, s.metadata) for s in self.store.list_skills()}
        changed, _ = build_evidence_library(self.store, self.rules)
        self.assertEqual(changed, [])
        self.assertEqual(before, {s.skill_id: (s.version, s.metadata) for s in self.store.list_skills()})
        self.rules.l3_min_successes = 3
        build_evidence_library(self.store, self.rules)
        self.assertTrue(all(s.status == SkillStatus.CANDIDATE for s in self.skills(3)))

    def test_planner_router_and_exports_are_compatible(self):
        self.build(*self.four_successes())
        skills = self.store.list_skills(kind="polished")
        goal = "Create a note named 'Carol' in Notes."
        decomposition, topology = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(goal, [], skills)
        self.assertIsNotNone(decomposition.task_skill_id)
        scoped = scope_hierarchical_skills(skills, decomposition.task_skill_id)
        profile = ModelProfile("student", "student", "http://test/v1", {})
        plan = DynamicProgrammingRouter(RoutingConfig(), self.store).route(goal, topology, [profile], scoped)
        self.assertTrue(any(s.is_polished for s in plan.steps))
        _, report = build_evidence_library(self.store, self.rules)
        export_evidence_library(self.store, self.root / "export", report)
        self.assertTrue((self.root / "export" / "skill_evidence.csv").is_file())
        self.assertTrue((self.root / "export" / "raw_catalog.json").is_file())

    def test_long_skill_injects_tail_stage_under_budget_without_summary_advance(self):
        self.build(*self.four_successes())
        root = self.skills(3)[0]
        root.metadata["instructions"] = [f"Stage {i}: inspect current UI and use current goal." for i in range(80)]
        root.metadata["leaf_primitive_ids"] = ["action.click"] * 80
        root.metadata["operation_count"] = 80
        profile = ModelProfile("student", "student", "http://test/v1", {})
        class Client:
            def predict_mm(self, prompt, images):
                self.prompt = prompt
                return "Action: {}", True, {}
        client = Client()
        pool = ModelPool([profile], client_factory=lambda _: client)
        wrapper = RoutedVLWrapper(pool, [root])
        step = RouteStep("a", "student", root.skill_id, (), (), .5, 1, 0, is_polished=True)
        wrapper.set_plan(ExecutionPlan("goal", root.topology, (step,), 0, metadata={"task_skill_id": root.skill_id}))
        wrapper.action_call_index = 75
        wrapper.predict_mm("base prompt", [])
        self.assertIn("Stage 75", client.prompt)
        self.assertNotIn("Stage 0:", client.prompt)
        self.assertLessEqual(wrapper.call_log[-1]["guidance_chars"], 1200)
        wrapper.predict_mm("summerize the latest step", [])
        self.assertEqual(wrapper.action_call_index, 76)
        self.assertEqual(client.prompt, "summerize the latest step")

    def test_cli_opt_in_and_default_rule_config(self):
        args = build_parser().parse_args(["maintain", "--evidence-based", "--skip-import"])
        self.assertTrue(args.evidence_based)
        config = load_config(Path(__file__).parents[1] / "config.example.toml")
        self.assertEqual(config.maintenance.admission_policy, "calibrated")
        self.assertEqual(config.maintenance.mining.l2_min_success_rate, .85)
        self.append(*self.four_successes())
        report = SkillMaintainer(self.store, MaintenanceConfig(admission_policy="evidence_based", build_hierarchy=True)).run_cycle()
        self.assertEqual(report.hierarchy["compiler"], COMPILER_ID)

    def test_goal_structure_disambiguates_batch_and_single_object(self):
        traces = []
        for task, count in (("ExpenseSingle", 1), ("ExpenseMultiple", 3)):
            for index in (0, 1):
                trace = self.trace(task, value=str(index), instance=index)
                trace.goal = "Add the following expenses into the pro expense:\n" + "\n".join(
                    f"Expense: item{j}_{index}\n amount_dollars: ${10 + j}" for j in range(count))
                traces.append(trace)
        self.build(*traces)
        skills = self.store.list_skills(kind="polished")
        goal = "Add the following expenses into the pro expense:\nExpense: A\nExpense: B\nExpense: C"
        decomposition, _ = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(goal, [], skills)
        self.assertEqual(self.store.get_skill(decomposition.task_skill_id).metadata["task_names"], ["ExpenseMultiple"])

    def test_following_payload_is_parameterized_even_without_input_text(self):
        traces = [self.trace(value=value, instance=i) for i, value in enumerate(("Alice", "Bob"))]
        for trace in traces:
            trace.goal = "Delete the following recipes from Broccoli app: " + ("Pizza" if trace.metadata["instance_id"] == 0 else "Noodles")
            trace.events = [trace.events[0], trace.events[3]]
        self.build(*traces)
        skill = self.skills(3)[0]
        self.assertEqual(skill.metadata["goal_templates"], ["Delete the following recipes from Broccoli app: <goal_value>"])
        decomposition, _ = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(
            "Delete the following recipes from Broccoli app: New Recipe", [], self.store.list_skills(kind="polished"))
        self.assertIsNotNone(decomposition.task_skill_id)

    def test_goal_intent_does_not_retrieve_opposite_operation_in_same_app(self):
        self.build(*self.four_successes())
        decomposition, _ = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(
            "Delete a note named 'Carol' in Notes.", [], self.store.list_skills(kind="polished"))
        self.assertIsNone(decomposition.task_skill_id)

    def test_path_consensus_excludes_self_vote(self):
        first = self.trace()
        second = self.trace(value="Bob", instance=1)
        second.events = [second.events[0], second.events[-1]]
        self.build(first, second)
        skill = self.skills(3)[0]
        self.assertAlmostEqual(skill.metadata["evidence"]["shared_path_fraction"], .4)
        self.assertEqual(skill.status, SkillStatus.CANDIDATE)
        self.assertIn("shared_path_fraction<0.6", skill.metadata["admission_reasons"])

    def test_payload_abstraction_keeps_draft_constraint_and_inline_cardinality(self):
        from src1.pmtskill_v2.skills.goal_schema import abstract_payload, goal_shape
        goal = "Go to the new contact screen and enter the following details: First Name: Eva. Do NOT hit save."
        self.assertIn("Do NOT hit save", abstract_payload(goal))
        self.assertEqual(goal_shape("Delete the following expenses from pro expense: Rent, Taxi, Meals."), {"payload_item_count": 3})
        traces = [self.trace(value=value, instance=i) for i, value in enumerate(("Alice", "Bob"))]
        for trace in traces:
            trace.goal = goal.replace("Eva", "Alice" if trace.metadata["instance_id"] == 0 else "Bob")
        self.build(*traces)
        decomposition, _ = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(goal, [], self.store.list_skills())
        self.assertIsNotNone(decomposition.task_skill_id)


if __name__ == "__main__":
    unittest.main()
