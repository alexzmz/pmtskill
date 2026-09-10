"""Teacher 标准原语轨迹、累计证据和双来源绑定测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src1.pmtskill_v2.cli import build_parser
from src1.pmtskill_v2.core.config import MaintenanceConfig
from src1.pmtskill_v2.core.models import (
    ExecutionTrace,
    SkillRecord,
    SkillStatus,
    SkillTopology,
    TraceEvent,
)
from src1.pmtskill_v2.evaluation.android_world import episodes_to_traces
from src1.pmtskill_v2.offline.dataset import infer_action_primitives
from src1.pmtskill_v2.offline.optimization_collector import (
    resolve_skill_task_bindings,
)
from src1.pmtskill_v2.offline.primitive_trajectory import (
    PRIMITIVE_TRACE_SCHEMA,
    PrimitivePlanDecision,
    PrimitivePlanUnit,
    SkillTaskSelection,
    TeacherPrimitiveVLWrapper,
)
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore


def _raw_skill(skill_id: str = "raw:test") -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name="Add a contact",
        description="Create a contact and save it",
        kind="raw",
        status=SkillStatus.IMPORTED,
        level=1,
        topology=SkillTopology.from_sequence(("action.click", "action.type")),
        body="Open Contacts, add the name and phone, then save.",
    )


class FakeClient:
    model_id = "teacher"

    def predict_mm(self, prompt, images):
        del prompt, images
        return '{"action_type":"click"}', True, {"_pmtskill": {"latency_ms": 3.0}}


class FakePlanner:
    planner_id = "fake-planner"
    optimized_skills = {}

    def __init__(self):
        self.revisions = 0
        self.bind_calls = 0

    def initial_plan(self, goal, *, source_skill, images=()):
        del goal, source_skill, images
        return PrimitivePlanDecision(
            (
                PrimitivePlanUnit("unit-001", ("action.click",), "click"),
                PrimitivePlanUnit("unit-002", ("action.type",), "type"),
            ),
            "initial",
        )

    def revise_plan(
        self,
        goal,
        *,
        source_skill,
        completed_units,
        remaining_units,
        latest_step,
        android_context,
        images=(),
    ):
        del (
            goal,
            source_skill,
            completed_units,
            remaining_units,
            latest_step,
            android_context,
            images,
        )
        self.revisions += 1
        return PrimitivePlanDecision(
            (PrimitivePlanUnit("unit-001", ("action.scroll",), "recover"),),
            "screen changed",
        )

    def bind_skill_to_tasks(self, skill, available_tasks, *, maximum_tasks):
        del skill, maximum_tasks
        self.bind_calls += 1
        return SkillTaskSelection((available_tasks[0],), "semantic match")


class PrimitiveWrapperTest(unittest.TestCase):
    def test_initial_plan_summary_and_dynamic_revision_share_schema(self):
        planner = FakePlanner()
        wrapper = TeacherPrimitiveVLWrapper(
            FakeClient(), planner, replan_every_steps=1
        )
        wrapper.set_episode("Add Alice", source_kind="task")

        _, _, first_raw = wrapper.predict_mm("choose next action", [])
        first_route = first_raw["_pmtskill"]
        self.assertEqual(first_route["trajectory_schema"], PRIMITIVE_TRACE_SCHEMA)
        self.assertEqual(first_route["planned_step_count"], 2)
        self.assertEqual(first_route["primitive_ids"], ["action.click"])

        # summary 调用不能推进计划或触发重规划。
        wrapper.predict_mm("Summerize the latest step", [])
        self.assertEqual(planner.revisions, 0)
        wrapper.observe_step(
            {
                "action_output": '{"action_type":"click"}',
                "action_output_json": {"action_type": "click"},
                "action_raw_response": first_raw,
            }
        )

        _, _, second_raw = wrapper.predict_mm("choose next action", [])
        second_route = second_raw["_pmtskill"]
        self.assertEqual(planner.revisions, 1)
        self.assertEqual(second_route["primitive_ids"], ["action.scroll"])
        self.assertEqual(second_route["current_revision"], 1)
        self.assertEqual(len(second_route["completed_units"]), 1)

    def test_raw_skill_source_only_changes_provenance_not_schema(self):
        wrapper = TeacherPrimitiveVLWrapper(FakeClient(), FakePlanner())
        skill = _raw_skill()
        wrapper.set_episode(
            "Add Alice", source_kind="raw_skill", source_skill=skill
        )
        _, _, raw = wrapper.predict_mm("choose next action", [])
        route = raw["_pmtskill"]
        self.assertEqual(route["trajectory_schema"], PRIMITIVE_TRACE_SCHEMA)
        self.assertEqual(route["source"]["kind"], "raw_skill")
        self.assertEqual(route["source"]["skill_id"], skill.skill_id)
        # raw skill 是输入提示，不应伪装成被调用的 polished skill。
        self.assertIsNone(route["skill_id"])


class TraceConversionTest(unittest.TestCase):
    def test_action_type_field_does_not_false_positive_as_text_input(self):
        self.assertEqual(
            infer_action_primitives('{"action_type":"click","index":3}'),
            ("action.click",),
        )
        self.assertEqual(
            infer_action_primitives('{"action_type":"input_text","text":"Alice"}'),
            ("action.type",),
        )

    def test_standard_metadata_and_typed_plan_round_trip(self):
        route = {
            "trajectory_schema": PRIMITIVE_TRACE_SCHEMA,
            "metric_scope": "skill_discovery",
            "routing_mode": "teacher_primitive_guided",
            "model_id": "teacher",
            "skill_id": None,
            "primitive_ids": ["ground.text", "action.click"],
            "source": {"kind": "task", "skill_id": None},
            "planner_id": "fake-planner",
            "planned_step_count": 1,
            "current_revision": 0,
            "planned_unit": {
                "unit_id": "unit-001",
                "primitive_ids": ["ground.text", "action.click"],
                "instruction": "click",
            },
            "initial_plan": [
                {
                    "unit_id": "unit-001",
                    "primitive_ids": ["ground.text", "action.click"],
                    "instruction": "click",
                }
            ],
            "plan_revisions": [{"revision": 0}],
            "completed_units": [],
        }
        traces = episodes_to_traces(
            [
                {
                    "goal": "Tap Allow",
                    "task_template": "PermissionTask",
                    "is_successful": True,
                    "run_time": 1.0,
                    "episode_data": [
                        {
                            "action_output": '{"action_type":"click"}',
                            "action_output_json": {"action_type": "click"},
                            "action_raw_response": {"_pmtskill": route},
                        }
                    ],
                }
            ]
        )
        trace = traces[0]
        self.assertEqual(trace.metadata["trajectory_schema"], PRIMITIVE_TRACE_SCHEMA)
        self.assertEqual(trace.metadata["metric_scope"], "skill_discovery")
        self.assertEqual(
            trace.metadata["executed_primitive_sequence"],
            ["ground.text", "action.click"],
        )
        self.assertEqual(trace.metadata["action_primitive_alignment_rate"], 1.0)
        self.assertIsNotNone(trace.plan)
        restored = ExecutionTrace.from_dict(trace.to_dict())
        self.assertIsNotNone(restored.plan)
        self.assertEqual(
            restored.plan.topology.primitive_sequence(),
            ("ground.text", "action.click"),
        )


class SkillBindingTest(unittest.TestCase):
    def test_explicit_metadata_and_teacher_binding_precedence(self):
        planner = FakePlanner()
        explicit = _raw_skill("raw:explicit")
        metadata = _raw_skill("raw:metadata")
        metadata.metadata["trajectory_tasks"] = ["TaskB"]
        automatic = _raw_skill("raw:auto")
        bindings, skipped = resolve_skill_task_bindings(
            (explicit, metadata, automatic),
            ("TaskA", "TaskB"),
            planner,
            explicit_map={explicit.skill_id: ("TaskA",)},
        )
        by_id = {item.skill_id: item for item in bindings}
        self.assertEqual(by_id[explicit.skill_id].source, "skill_task_map")
        self.assertEqual(by_id[metadata.skill_id].source, "skill_metadata")
        self.assertEqual(by_id[automatic.skill_id].source, "teacher_auto_binding")
        self.assertEqual(skipped, {})
        self.assertEqual(planner.bind_calls, 1)

    def test_unknown_android_task_is_rejected_before_emulator(self):
        with self.assertRaisesRegex(ValueError, "未知 AndroidWorld"):
            resolve_skill_task_bindings(
                (_raw_skill(),),
                ("TaskA",),
                FakePlanner(),
                explicit_map={"raw:test": ("NotRegistered",)},
            )


class CumulativeMaintenanceEvidenceTest(unittest.TestCase):
    @staticmethod
    def _trace(successful: bool, source_kind: str) -> ExecutionTrace:
        return ExecutionTrace.new(
            "Add contact",
            "ContactsAddContact",
            successful,
            (
                TraceEvent(0, "teacher", None, ("action.click",), successful, 1.0),
                TraceEvent(1, "teacher", None, ("action.type",), successful, 1.0),
            ),
            metadata={
                "trajectory_schema": PRIMITIVE_TRACE_SCHEMA,
                "metric_scope": "skill_discovery",
                "collection_source": {
                    "kind": source_kind,
                    "skill_id": "raw:contact" if source_kind == "raw_skill" else None,
                },
                "executed_primitive_sequence": ["action.click", "action.type"],
            },
        )

    def test_support_accumulates_across_maintenance_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SkillStore(Path(directory) / "skills.sqlite3")
            store.initialize()
            config = MaintenanceConfig(
                minimum_support=3,
                minimum_subsequence_length=2,
                maximum_subsequence_length=2,
            )
            maintainer = SkillMaintainer(store, config)

            for source in ("task", "raw_skill"):
                store.append_trace(self._trace(True, source))
            first = maintainer.run_cycle()
            self.assertEqual(first.candidates_created, [])

            # 失败证据保留风险统计，但不计入 candidate 的成功 support。
            store.append_trace(self._trace(False, "raw_skill"))
            store.append_trace(self._trace(True, "task"))
            second = maintainer.run_cycle()
            self.assertEqual(len(second.candidates_created), 1)
            skill = store.get_skill(second.candidates_created[0])
            self.assertEqual(skill.metadata["support"], 3)
            self.assertEqual(skill.metadata["failure_support"], 1)
            self.assertEqual(
                skill.metadata["trajectory_source_kinds"], ["raw_skill", "task"]
            )
            self.assertEqual(skill.metadata["source_skill_ids"], ["raw:contact"])

    def test_skill_discovery_trace_does_not_pollute_online_skill_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SkillStore(Path(directory) / "skills.sqlite3")
            store.initialize()
            skill = SkillRecord(
                skill_id="polished:test",
                name="test",
                description="test",
                kind="polished",
                status=SkillStatus.CANDIDATE,
                level=2,
                topology=SkillTopology.from_sequence(("action.click",)),
                body="click",
            )
            store.upsert_skill(skill)
            trace = ExecutionTrace.new(
                "goal",
                "Task",
                True,
                (TraceEvent(0, "teacher", skill.skill_id, ("action.click",), True, 1.0),),
                metadata={"metric_scope": "skill_discovery"},
            )
            store.append_trace(trace)
            self.assertEqual(store.skill_metrics(skill.skill_id)["trials"], 0)


class OptimizationTraceCliTest(unittest.TestCase):
    def test_parser_exposes_both_sources_and_safe_defaults(self):
        parser = build_parser()
        task = parser.parse_args(["collect-optimization-traces", "--tasks", "TaskA"])
        self.assertEqual(task.source, "task")
        self.assertTrue(task.record_traces)
        self.assertEqual(task.replan_every_steps, 1)
        skill = parser.parse_args(
            [
                "collect-guided",
                "--source",
                "raw-skill",
                "--skills",
                "raw:test",
            ]
        )
        self.assertEqual(skill.source, "raw-skill")


if __name__ == "__main__":
    unittest.main()
