"""No GPU/emulator: success-first objective, live routing and compact resume."""
from __future__ import annotations

import unittest
from unittest import mock

from src1.pmtskill_v2.core.config import RoutingConfig
from src1.pmtskill_v2.core.models import ModelProfile, SkillRecord, SkillStatus, SkillTopology, ExecutionPlan, RouteStep
from src1.pmtskill_v2.inference.model_pool import ModelPool
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.online.executor import RoutedVLWrapper
from src1.pmtskill_v2.online.runtime import current_ui_text, grounded_summary, screen_key
from src1.pmtskill_v2.evaluation.android_world import episodes_to_traces
from src1.pmtskill_v2.evaluation.reporter import summarize_episodes, compact_result_for_storage


def model(name, capabilities=None, **kwargs):
    return ModelProfile(name, name, "http://unused/v1", capabilities or {}, **kwargs)


def skill():
    return SkillRecord(skill_id="task", name="Task guide", description="", body="Use the current goal",
                       kind="polished", status=SkillStatus.ACTIVE, level=3,
                       topology=SkillTopology.from_sequence(["action.open_app", "action.type"]),
                       metadata={"hierarchy_role": "task", "prompt_mode": "stepwise", "compiler": "evidence-hierarchy-v2",
                                 "operation_count": 100, "leaf_primitive_ids": ["action.open_app", "action.type"],
                                 "instructions": ["Open target app", "Type current goal values"]})


class Client:
    def __init__(self, profile):
        self.profile = profile
        self.prompts = []
        self.summary_output = "summarized"

    def predict_mm(self, prompt, images):
        self.prompts.append((prompt, images))
        text = self.summary_output if "summerize" in prompt else 'Reason: inspect\nAction: {"action_type":"click","index":1}'
        return text, True, {"_pmtskill": {"model_id": self.profile.model_id}}


def wrapper(profiles=None, root=None):
    profiles = profiles or [model("a"), model("b")]
    root = root or skill()
    clients = {p.model_id: Client(p) for p in profiles}
    pool = ModelPool(profiles, client_factory=lambda p: clients[p.model_id])
    router = DynamicProgrammingRouter(RoutingConfig())
    wrapped = RoutedVLWrapper(pool, [root], router=router, supported_apps=["files", "chrome"])
    step = RouteStep("whole-task", profiles[0].model_id, root.skill_id, root.topology.primitive_sequence(), (), .5, 1, 0, is_polished=True)
    wrapped.set_plan(ExecutionPlan("Enter current goal values", root.topology, (step,), 0,
                                   metadata={"task_skill_id": root.skill_id}))
    return wrapped, clients, root


class PortsRuntimeTest(unittest.TestCase):
    def test_success_first_ignores_old_large_latency_and_switch_penalties(self):
        topology = SkillTopology.from_sequence(["ground.text", "action.click"])
        profiles = [model("reader", {"ground.text": .99, "action.click": .6}, average_latency_ms=100000, switch_cost_ms=100000),
                    model("actor", {"ground.text": .6, "action.click": .99}, average_latency_ms=1, switch_cost_ms=100000)]
        plan = DynamicProgrammingRouter(RoutingConfig(latency_weight=10, switch_weight=10)).route("goal", topology, profiles, [])
        self.assertEqual([s.model_id for s in plan.steps], ["reader", "actor"])
        self.assertEqual(plan.switch_count, 1)
        self.assertTrue(all(s.score_detail["switch_penalty"] == s.score_detail["latency_penalty"] == 0 for s in plan.steps))

    def test_polished_bonus_cannot_beat_a_more_successful_route(self):
        root = skill()
        router = DynamicProgrammingRouter(RoutingConfig(polished_bonus=10000, degradation_weight=10000))
        with mock.patch.object(router, "_probability", side_effect=lambda m, c: (.1 if c.skill else .9, 1)):
            plan = router.route("goal", root.topology, [model("a")], [root])
        self.assertTrue(all(step.skill_id is None for step in plan.steps))

    def test_l3_task_does_not_lock_the_adapter_and_summary_is_pinned(self):
        wrapped, clients, root = wrapper([model("navigator", {"action.open_app": .95, "action.type": .4}),
                                           model("typist", {"action.open_app": .4, "action.type": .95})])
        images = [object()]
        wrapped.begin_step()
        _, _, raw = wrapped.predict_mm("first", images)
        self.assertEqual(raw["_pmtskill"]["model_id"], "navigator")
        wrapped.observe_step({"action_output_json": {"action_type": "open_app", "app_name": "contacts"},
                              "before_ui_elements": [{"text": "Home"}], "after_ui_elements": [{"text": "Editor"}]})
        wrapped.begin_step()
        _, _, raw = wrapped.predict_mm("consistent with the ones in the labeled screenshot:\nUI element: is_editable: True\nHere are some useful guidelines", images)
        self.assertEqual(raw["_pmtskill"]["model_id"], "typist")
        wrapped.predict_mm("Now I want you to summerize the latest step", [object(), object()])
        self.assertEqual(wrapped.model_pool.current_model_id, "typist")
        self.assertEqual(wrapped.model_pool.switch_count, 1)
        self.assertIs(clients["navigator"].prompts[0][1], images)

    def test_two_screen_toggle_triggers_model_switch_and_bare_skill_recovery(self):
        wrapped, clients, _ = wrapper()
        for index in range(5):
            wrapped.begin_step()
            wrapped.predict_mm("select action", [])
            wrapped.observe_step({"action_output_json": {"action_type": "click", "index": 32 if index % 2 == 0 else 0},
                                  "before_ui_elements": [{"text": "Editor" if index % 2 == 0 else "Dropdown"}],
                                  "after_ui_elements": [{"text": "Dropdown" if index % 2 == 0 else "Editor"}]})
        self.assertTrue(wrapped.runtime.recovery_pending)
        wrapped.begin_step()
        _, _, raw = wrapped.predict_mm("select action", [])
        route = raw["_pmtskill"]
        self.assertEqual(route["model_id"], "b")
        self.assertIsNone(route["skill_id"])
        self.assertTrue(route["ports_runtime"]["stall_recovery"])
        self.assertTrue(route["ports_runtime"]["skill_fallback"])
        self.assertIn("RECOVERY", clients["b"].prompts[-1][0])
        self.assertIn("files, chrome", clients["b"].prompts[-1][0])
        self.assertEqual(wrapped.runtime.stage, 0)  # clicks did not complete open_app

    def test_repeated_actions_with_new_ui_or_no_observation_are_not_fake_failures(self):
        for observed in (True, False):
            wrapped, _, _ = wrapper()
            for i in range(12):
                wrapped.begin_step()
                wrapped.predict_mm("select action", [])
                data = {"action_output_json": {"action_type": "click", "index": 1}}
                if observed:
                    data.update(before_ui_elements=[{"text": str(i)}], after_ui_elements=[{"text": str(i + 1)}])
                wrapped.observe_step(data)
            self.assertFalse(wrapped.runtime.failures)
            self.assertEqual(wrapped.model_pool.switch_count, 0)

    def test_invalid_actions_trigger_recovery_and_episode_reset_clears_it(self):
        wrapped, _, _ = wrapper()
        plan = wrapped.plan
        for _ in range(3):
            wrapped.begin_step()
            wrapped.predict_mm("select", [])
            wrapped.observe_step({"summary": "Output for action selection is not in the correct format"})
        self.assertTrue(wrapped.runtime.recovery_pending)
        wrapped.set_plan(plan)
        self.assertFalse(wrapped.runtime.failures)
        self.assertFalse(wrapped.runtime.recovery_pending)

    def test_disabled_skill_pair_is_not_injected_on_new_adapter(self):
        wrapped, _, _ = wrapper()
        original = wrapped.runtime.router.model_unit_support
        with mock.patch.object(wrapped.runtime.router, "model_unit_support",
                               side_effect=lambda m, s, sk, **kw: None if m.model_id == "b" and sk else original(m, s, sk, **kw)):
            wrapped.runtime.failures["a"] = 1
            wrapped.begin_step()
            _, _, raw = wrapped.predict_mm("select", [])
            self.assertEqual(raw["_pmtskill"]["model_id"], "a")

    def test_ui_extraction_does_not_read_the_historical_preamble(self):
        text = "Examples EditText input_text\nconsistent with the ones in the labeled screenshot:\nCURRENT UI\nHere are some useful guidelines: more text"
        self.assertEqual(current_ui_text(text).strip(), "CURRENT UI")
        clock = lambda text, package: {"before_ui_elements": [{"text": text, "package_name": package}]}
        self.assertEqual(screen_key(clock("12:01", "com.android.systemui")), screen_key(clock("12:02", "com.android.systemui")))
        self.assertNotEqual(screen_key(clock("12:01", "calendar")), screen_key(clock("12:02", "calendar")))

    def test_action_in_summary_is_replaced_by_grounded_observation_not_another_action(self):
        wrapped, clients, _ = wrapper()
        clients["a"].summary_output = '{"action_type":"click","index":6}'
        wrapped.begin_step()
        _, _, action_raw = wrapped.predict_mm("select", [])
        prompt = ("Now I want you to summerize the latest step.\nin the before screenshot:\nUI0\n"
                  "Here is the list for the after screenshot:\nUI0\nThis is the action you picked: click\nBased on the reason: inspect")
        summary, _, summary_raw = wrapped.predict_mm(prompt, [])
        self.assertIn("not proof of success", summary)
        self.assertNotIn('"index":6', summary)
        self.assertTrue(summary_raw["_pmtskill"]["summary_fallback"])
        data = {"action_raw_response": action_raw, "summary_raw_response": summary_raw}
        wrapped.observe_step(data)
        self.assertTrue(action_raw["_pmtskill"]["ports_runtime"]["summary_fallback"])

    def test_runtime_statistics_survive_result_only_resume_without_reclassifying_sr(self):
        episode = {"goal": "goal", "task_template": "ContactsAddContact", "is_successful": 0.,
                   "episode_length": 2, "exception_info": None,
                   "aux_data": {"collector_termination_reason": "max_steps", "collector_step_limit": 2},
                   "episode_data": {"step_number": [0, 1], "action_output": ['Action: {"action_type":"click"}'] * 2,
                                    "action_output_json": [{"action_type": "click"}] * 2,
                                    "action_raw_response": [{"_pmtskill": {"model_id": name, "ports_runtime": {
                                        "runtime_decision": True, "adapter_switched": i > 0, "stall_recovery": i > 0}}}
                                        for i, name in enumerate(("a", "b"))]}}
        traces = episodes_to_traces([episode])
        first = summarize_episodes([episode], traces)
        compact = compact_result_for_storage(episode, traces)
        resumed = summarize_episodes([compact], episodes_to_traces([compact]))
        self.assertEqual(first["routing_runtime"], resumed["routing_runtime"])
        self.assertEqual(resumed["routing_runtime"]["stall_recovery"], 1)
        self.assertEqual(resumed["average_model_switches"], 1.)
        self.assertEqual(resumed["success_rate_micro"], 0.)
        self.assertEqual(resumed["failure_reasons"], {"step_budget_exhausted": 1})


if __name__ == "__main__":
    unittest.main()
