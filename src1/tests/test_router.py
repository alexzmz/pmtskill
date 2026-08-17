"""动态规划模型/技能路由测试。"""

from __future__ import annotations

import unittest

from src1.pmtskill_v2.core.config import RoutingConfig
from src1.pmtskill_v2.core.models import (
    ModelProfile,
    SkillRecord,
    SkillStatus,
    SkillTopology,
)
from src1.pmtskill_v2.online.router import DynamicProgrammingRouter
from src1.pmtskill_v2.core.models import ExecutionPlan, RouteStep
from src1.pmtskill_v2.inference.model_pool import ModelPool
from src1.pmtskill_v2.online.executor import RoutedVLWrapper


def model(model_id: str, capabilities: dict[str, float]) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        served_model=model_id,
        base_url="http://test/v1",
        capabilities=capabilities,
        average_latency_ms=1000,
        switch_cost_ms=50,
    )


class RouterTest(unittest.TestCase):
    def test_prefers_verified_polished_skill_for_matching_sequence(self):
        topology = SkillTopology.from_sequence(("action.click", "reason.verify"))
        polished = SkillRecord(
            skill_id="polished:test:v1",
            name="click_and_verify",
            description="",
            kind="polished",
            status=SkillStatus.ACTIVE,
            level=2,
            topology=SkillTopology.from_sequence(("action.click", "reason.verify")),
        )
        profile = model("m", {"action.click": 0.9, "reason.verify": 0.9})
        plan = DynamicProgrammingRouter(RoutingConfig()).route(
            "test", topology, [profile], [polished]
        )
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].skill_id, polished.skill_id)
        self.assertTrue(plan.steps[0].is_polished)

    def test_high_switch_penalty_keeps_one_model(self):
        topology = SkillTopology.from_sequence(("ground.text", "action.click"))
        first = model("grounder", {"ground.text": 0.99, "action.click": 0.60})
        second = model("actor", {"ground.text": 0.60, "action.click": 0.99})
        low_penalty = RoutingConfig(
            latency_weight=0,
            switch_weight=0,
            polished_bonus=0,
            degradation_weight=0,
        )
        split = DynamicProgrammingRouter(low_penalty).route(
            "test", topology, [first, second], []
        )
        self.assertEqual(split.switch_count, 1)

        high_penalty = RoutingConfig(
            latency_weight=0,
            switch_weight=0.02,
            polished_bonus=0,
            degradation_weight=0,
        )
        stable = DynamicProgrammingRouter(high_penalty).route(
            "test", topology, [first, second], []
        )
        self.assertEqual(stable.switch_count, 0)

    def test_banned_polished_skill_expands_to_primitives(self):
        topology = SkillTopology.from_sequence(("action.click", "reason.verify"))
        polished = SkillRecord(
            skill_id="polished:test:v1",
            name="click_and_verify",
            description="",
            kind="polished",
            status=SkillStatus.ACTIVE,
            level=2,
            topology=topology,
        )
        profile = model("m", {"action.click": 0.9, "reason.verify": 0.9})
        plan = DynamicProgrammingRouter(RoutingConfig()).route(
            "test",
            topology,
            [profile],
            [polished],
            banned_skill_ids={polished.skill_id},
        )
        self.assertEqual(len(plan.steps), 2)
        self.assertTrue(all(step.skill_id is None for step in plan.steps))

    def test_action_summary_reuses_model_before_next_route_step(self):
        class FakeClient:
            def __init__(self, profile):
                self.model_id = profile.model_id

            def predict_mm(self, prompt, images):
                return "ok", True, {}

        profiles = [model("first", {}), model("second", {})]
        pool = ModelPool(profiles, client_factory=FakeClient)
        topology = SkillTopology.from_sequence(("reason.intent", "action.click"))
        steps = (
            RouteStep("1", "first", None, ("reason.intent",), ("n0000",), 0.5, 1, 0),
            RouteStep("2", "second", None, ("action.click",), ("n0001",), 0.5, 1, 0),
        )
        wrapper = RoutedVLWrapper(pool, [])
        wrapper.set_plan(ExecutionPlan("goal", topology, steps, 0))
        wrapper.predict_mm("action selection", [])
        wrapper.predict_mm("Now I want you to summerize the latest step", [])
        self.assertEqual(pool.current_model_id, "first")
        self.assertEqual(pool.switch_count, 0)
        wrapper.predict_mm("next action selection", [])
        self.assertEqual(pool.current_model_id, "second")
        self.assertEqual(pool.switch_count, 1)


if __name__ == "__main__":
    unittest.main()
