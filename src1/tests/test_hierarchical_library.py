"""成功一次准入、参数化层级库与在线规划的端到端离线回归。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import MaintenanceConfig, RoutingConfig, load_config
from src1.pmtskill_v2.core.io import load_primitives
from src1.pmtskill_v2.core.models import ExecutionTrace, ModelProfile, SkillRecord, SkillStatus, SkillTopology, TraceEvent
from src1.pmtskill_v2.inference.model_pool import ModelPool
from src1.pmtskill_v2.online.executor import RoutedVLWrapper
from src1.pmtskill_v2.online.planner import KeywordSkillPlanner, PlannerPipeline, PrimitiveTopologyGenerator, scope_hierarchical_skills
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.skills.hierarchy import copy_skill_database, export_hierarchical_library
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore, classify_trajectory_quality


class HierarchicalLibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SkillStore(self.root / "source.sqlite3")
        self.store.initialize()
        self.store.sync_primitive_catalog(load_primitives())
        self.config = MaintenanceConfig(admission_policy="success_once", build_hierarchy=True)

    def tearDown(self):
        self.temp.cleanup()

    def trace(self, *, successful=True, invalid=False, name="ContactsAddContact", last_instruction="Click Save."):
        units = [
            (("action.open_app",), "Open Contacts.", {"action_type": "open_app", "app_name": "Contacts"}),
            (("action.click",), "Click Create new contact.", {"action_type": "click", "index": 99}),
            (("action.type",), "Type 'Alice' in the First name field.", {"action_type": "input_text", "text": "Alice"}),
            (("action.type",), "Type '+123456789' in the Phone field.", {"action_type": "input_text", "text": "+123456789"}),
            (("action.click",), last_instruction, {"action_type": "click", "index": 7}),
        ]
        return ExecutionTrace.new(
            "Create a new contact for Alice. Their number is +123456789.", name, successful,
            [TraceEvent(i, "teacher", None, p, successful, 1,
                        action="Reason: demo\nAction: " + json.dumps(action),
                        metadata={"action_committed": True, "planned_unit": {"instruction": instruction}})
             for i, (p, instruction, action) in enumerate(units)],
            metadata={"trajectory_schema": "pmtskill.teacher-primitive-trace/v1",
                      "routing_mode": "teacher_primitive_guided", "episode_data_valid": not invalid,
                      "initial_plan": [{"primitive_ids": ["action.open_app"]}],
                      "planned_step_count": 1, "action_primitive_alignment_rate": 0.5},
        )

    def build(self, *traces):
        for trace in traces or (self.trace(),):
            self.store.append_trace(trace, update_skill_metrics=False)
        return SkillMaintainer(self.store, self.config).run_cycle()

    def test_success_once_accepts_candidate_quality_without_fabricating_model_metrics(self):
        success = self.trace()
        self.assertEqual(classify_trajectory_quality(success)["status"], "candidate")
        result = self.build(success, self.trace(successful=False, name="FailedTask"), self.trace(invalid=True, name="BrokenTask"))
        self.assertEqual(result.hierarchy["covered_tasks"], ["ContactsAddContact"])
        self.assertEqual(result.hierarchy["uncovered_tasks"], ["BrokenTask", "FailedTask"])
        skills = self.store.list_skills(kind="polished")
        self.assertEqual({skill.level for skill in skills}, {1, 2, 3})
        self.assertTrue(all(skill.status == SkillStatus.ACTIVE for skill in skills))
        self.assertTrue(all(skill.metadata["support"] == 1 for skill in skills))
        self.assertTrue(all(self.store.skill_metrics(skill.skill_id)["trials"] == 0 for skill in skills))
        self.assertEqual(self.store.list_skill_model_statuses(), [])

    def test_strict_default_stays_strict(self):
        self.store.append_trace(self.trace(), update_skill_metrics=False)
        result = SkillMaintainer(self.store, MaintenanceConfig()).run_cycle()
        self.assertEqual(result.candidates_created, [])
        self.assertEqual(self.store.list_skills(kind="polished"), [])

    def test_hierarchy_calibrated_mode_does_not_auto_activate(self):
        trace = self.trace()
        trace.metadata = {"episode_data_valid": True}
        self.store.append_trace(trace, update_skill_metrics=False)
        result = SkillMaintainer(self.store, MaintenanceConfig(build_hierarchy=True, minimum_support=1)).run_cycle()
        self.assertTrue(result.candidates_created)
        self.assertTrue(all(skill.status == SkillStatus.CANDIDATE for skill in self.store.list_skills(kind="polished")))

    def test_parameterization_removes_demo_values_and_action_indices(self):
        self.build()
        task = next(skill for skill in self.store.list_skills() if skill.level == 3)
        self.assertIn("<goal_value>", task.body)
        self.assertNotIn("Alice", task.body)
        self.assertNotIn("123456789", task.body)
        self.assertNotIn('"index": 99', task.body)
        self.assertLess(len(task.body), 3000)
        self.assertIn("Click Save", task.body)

    def test_semantic_operations_are_not_merged_by_topology(self):
        self.build(self.trace(), self.trace(name="DeleteContact", last_instruction="Click Delete."))
        operations = [skill for skill in self.store.list_skills() if skill.level == 1]
        names = [skill.body for skill in operations if skill.topology.primitive_sequence() == ("action.click",)]
        self.assertTrue(any("Click Save" in body for body in names))
        self.assertTrue(any("Click Delete" in body for body in names))

    def test_repeat_maintenance_is_idempotent_and_preserves_raw(self):
        raw = SkillRecord("raw:test", "Original", "Keep me", "raw", SkillStatus.IMPORTED, 1, SkillTopology.from_sequence(["action.click"]))
        self.store.upsert_skill(raw)
        self.build()
        before = [skill.to_dict() for skill in self.store.list_skills()]
        relations = self.store.list_skill_relations()
        second = SkillMaintainer(self.store, self.config).run_cycle()
        after = [skill.to_dict() for skill in self.store.list_skills()]
        self.assertEqual(second.candidates_created, [])
        self.assertEqual([(s["skill_id"], s["version"], s["metadata"]) for s in before],
                         [(s["skill_id"], s["version"], s["metadata"]) for s in after])
        self.assertEqual(len(self.store.list_skill_relations()), len(relations))
        self.assertEqual(self.store.get_skill(raw.skill_id).to_dict(), raw.to_dict())

    def test_task_planning_preserves_repeated_actions_and_routes_without_calibration(self):
        self.build()
        skills = self.store.list_skills(kind="polished")
        goal = "Create a new contact for Bob. Their number is +987654321."
        decomposition, topology = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(goal, [], skills)
        self.assertIsNotNone(decomposition.task_skill_id)
        self.assertEqual(topology.primitive_sequence().count("action.type"), 2)
        scoped = scope_hierarchical_skills(skills, decomposition.task_skill_id)
        profile = ModelProfile("student", "student", "http://test/v1", {})
        plan = DynamicProgrammingRouter(RoutingConfig(), self.store).route(goal, topology, [profile], scoped)
        self.assertEqual(plan.steps[0].skill_id, decomposition.task_skill_id)
        self.assertTrue(all(step.is_polished for step in plan.steps))
        self.assertEqual(scope_hierarchical_skills(skills, None), [])

    def test_unrelated_goal_does_not_use_task_demo(self):
        self.build()
        decomposition, _ = PlannerPipeline(KeywordSkillPlanner(), PrimitiveTopologyGenerator()).plan(
            "Open the camera and take a photo.", [], self.store.list_skills())
        self.assertIsNone(decomposition.task_skill_id)

    def test_goal_payload_does_not_drown_out_app_and_operation_retrieval(self):
        from src1.pmtskill_v2.online.planner import retrieve_task_skill

        task = SkillRecord("polished:message", "reply", "", "polished", SkillStatus.ACTIVE, 3,
                           SkillTopology.from_sequence(["action.type"]),
                           metadata={"hierarchy_role": "task", "goal_templates": ["Reply to <goal_value> with message: <goal_value> in Simple SMS Messenger"]})
        goal = "Reply to +123456789 with message: You can't make an omelette without breaking a few eggs. in Simple SMS Messenger"
        self.assertEqual(retrieve_task_skill(goal, [task]), task)
        task.metadata["goal_templates"] = ["Add the following recipes into the Broccoli app:\ntitle|description\n<goal_value>"]
        self.assertEqual(retrieve_task_skill("Add the following recipes into the Broccoli app:\nA very long recipe payload" * 2, [task]), task)
        task.metadata["goal_templates"] = ["Add the following recipes into the Broccoli app: title|description|servings Raspberry Almond Smoothie|A tasty recipe|2 servings"]
        self.assertEqual(retrieve_task_skill("Add the following recipes into the Broccoli app:\nOther|Contents|4", [task]), task)

    def test_fallback_components_are_bound_to_their_original_path_positions(self):
        self.build()
        skills = self.store.list_skills()
        task = next(skill for skill in skills if skill.level == 3)
        scoped = scope_hierarchical_skills(skills, task.skill_id)
        operations = [skill for skill in scoped if skill.level == 1]
        profile = ModelProfile("student", "student", "http://test/v1", {})
        plan = DynamicProgrammingRouter(RoutingConfig(), self.store).route("goal", task.topology, [profile], operations)
        selected = {skill.skill_id: skill for skill in skills}
        self.assertIn("First name", selected[plan.steps[2].skill_id].body)
        self.assertIn("Phone field", selected[plan.steps[3].skill_id].body)
        self.assertIn("Click Save", selected[plan.steps[4].skill_id].body)

    def test_rejected_and_unknown_primitive_cannot_create_full_task(self):
        trace = self.trace()
        trace.events[0].primitive_ids = ("action.unknown",)
        result = self.build(trace, self.trace(invalid=True))
        self.assertEqual(result.hierarchy["covered_task_count"], 0)
        self.assertIn(trace.trace_id, result.hierarchy["skipped_successful_trace_ids"])

    def test_hierarchical_wrapper_keeps_subflow_for_its_real_action_count(self):
        self.build()
        skills = self.store.list_skills()
        subflow = next(skill for skill in skills if skill.level == 2 and skill.metadata["operation_count"] == 3)
        from src1.pmtskill_v2.core.models import ExecutionPlan, RouteStep

        profile = ModelProfile("student", "student", "http://test/v1", {})
        wrapper = RoutedVLWrapper(ModelPool([profile]), skills)
        steps = (RouteStep("a", "student", subflow.skill_id, (), (), .5, 1, 0), RouteStep("b", "student", None, (), (), .5, 1, 0))
        wrapper.set_plan(ExecutionPlan("goal", SkillTopology.from_sequence(["action.click"]), steps, 0))
        for index in range(3):
            wrapper.action_call_index = index
            self.assertEqual(wrapper._route_step().step_id, "a")
        wrapper.action_call_index = 3
        self.assertEqual(wrapper._route_step().step_id, "b")

    def test_existing_calibration_deprecated_skill_is_not_reactivated(self):
        self.build()
        task = next(skill for skill in self.store.list_skills() if skill.level == 3)
        profile = ModelProfile("student", "student", "http://test/v1", {})
        from src1.pmtskill_v2.skills.identity import model_variant_id

        self.store.upsert_model_profile(profile)
        self.store.register_model_variant(profile)
        self.store.set_skill_model_status(task, model_id="student", model_variant_id_value=model_variant_id(profile), status=SkillStatus.DEPRECATED, reason={"test": True})
        self.store.set_skill_status(task.skill_id, SkillStatus.DEPRECATED)
        SkillMaintainer(self.store, self.config).run_cycle()
        self.assertEqual(self.store.get_skill(task.skill_id).status, SkillStatus.DEPRECATED)

    def test_database_backup_and_export_preserve_source(self):
        self.build()
        before = hashlib.sha256(self.store.database.read_bytes()).hexdigest()
        target = self.root / "output.sqlite3"
        copy_skill_database(self.store.database, target)
        copy = SkillStore(target)
        copy.initialize()
        self.assertEqual(len(copy.list_traces()), 1)
        self.assertEqual(hashlib.sha256(self.store.database.read_bytes()).hexdigest(), before)
        with self.assertRaises(ValueError):
            copy_skill_database(self.store.database, target)
        report = SkillMaintainer(copy, self.config).run_cycle().hierarchy
        export_hierarchical_library(copy, self.root / "report", report)
        self.assertTrue((self.root / "report" / "hierarchy.json").is_file())
        self.assertTrue((self.root / "report" / "coverage.md").is_file())

    def test_cli_and_config_expose_opt_in_mode(self):
        args = build_parser().parse_args(["maintain", "--success-once", "--hierarchical", "--skip-import",
                                         "--source-database", "source.sqlite3", "--output-database", "output.sqlite3"])
        self.assertTrue(args.success_once and args.hierarchical and args.skip_import)
        default = load_config(Path(__file__).parents[1] / "config.example.toml")
        self.assertEqual(default.maintenance.admission_policy, "calibrated")
        self.assertFalse(default.maintenance.build_hierarchy)


if __name__ == "__main__":
    unittest.main()
