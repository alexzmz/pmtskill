from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from src1.pmtskill_v2.evaluation.observation import (
    ObservedM3A,
    _observation_boxes,
    observe_android_world,
    watch_observed_episodes,
)


class ObservationTest(unittest.TestCase):
    def test_optional_clock_timestamp_precedes_ui_geometry_and_prediction(self):
        events = []
        pixels = object()

        class FakeM3A:
            def step(self, goal):
                self.env.get_state()
                events.append("predict")

        class Agent(ObservedM3A, FakeM3A):
            pass

        agent = Agent()
        agent.env = SimpleNamespace(get_state=lambda: SimpleNamespace(pixels=pixels), execute_action=lambda _: None)
        task = SimpleNamespace(name="OsmAndTrack", tear_down=lambda env: None)

        def run_task(task, run_episode, env, demo_mode):
            run_episode(task)
            task.tear_down(env)
            return {"is_successful": 1.0}

        suite = SimpleNamespace(_run_task=run_task)
        observer = SimpleNamespace(
            begin=lambda _: None, end=lambda *args: None,
            observe_at=lambda *args: events.append(args),
        )

        def monotonic():
            events.append("clock")
            return 123.0

        def geometry(*args):
            events.append("geometry")
            return []

        with mock.patch("src1.pmtskill_v2.evaluation.observation.time.monotonic", side_effect=monotonic), \
             mock.patch("src1.pmtskill_v2.evaluation.observation._observation_boxes", side_effect=geometry):
            with observe_android_world(observer), watch_observed_episodes(suite):
                suite._run_task(task, lambda _: agent.step("goal"), agent.env, False)
        self.assertEqual(events, ["clock", "geometry", (1, pixels, [], 123.0), "predict"])

    def test_exact_step_images_and_post_action_boxes_without_extra_states(self):
        events = []
        before, internal, after = object(), object(), object()

        class Env:
            def get_state(self):
                image = next(images)
                events.append(("state", image))
                return SimpleNamespace(pixels=image)

            def execute_action(self, action):
                self.get_state()  # AndroidWorld actuation's internal index resolution.
                events.append(("action", action))

        class Task:
            name = "OsmAndTrack"

            def tear_down(self, env):
                events.append(("teardown",))

        class FakeM3A:
            def step(self, goal):
                self.env.get_state()
                events.append(("predict", goal))
                self.env.execute_action("tap")
                self.env.get_state()

        class Agent(ObservedM3A, FakeM3A):
            pass

        images = iter((before, internal, after))
        task, env, agent = Task(), Env(), Agent()
        agent.env = env

        def run_task(task, run_episode, env, demo_mode):
            events.append(("setup",))
            run_episode(task)
            events.append(("verify",))
            task.tear_down(env)
            return {"is_successful": 1.0, "exception_info": None}

        def run_episode(task):
            agent.step(task.name)

        observer = SimpleNamespace(
            begin=lambda task_id: events.append(("start", task_id)),
            observe=lambda step, pixels, boxes: events.append(("observation", step, pixels, boxes)),
            stop_capture=lambda: events.append(("stop",)),
            end=lambda success, error: events.append(("end", success, error)),
        )
        suite = SimpleNamespace(_run_task=run_task)
        with mock.patch("src1.pmtskill_v2.evaluation.observation._observation_boxes", return_value=[]):
            with observe_android_world(observer), watch_observed_episodes(suite):
                result = suite._run_task(task, run_episode, env, False)
        self.assertEqual(result["is_successful"], 1.0)
        self.assertEqual(events, [
            ("setup",), ("start", "OsmAndTrack"), ("state", before),
            ("observation", 1, before, []), ("predict", "OsmAndTrack"),
            ("state", internal), ("action", "tap"), ("state", after),
            ("observation", None, after, []), ("verify",), ("stop",),
            ("teardown",), ("end", True, None),
        ])
        self.assertIs(suite._run_task, run_task)
        self.assertNotIn("tear_down", vars(task))
        self.assertNotIn("get_state", vars(env))
        self.assertNotIn("execute_action", vars(env))

    def test_native_failure_and_exception_are_not_conflated(self):
        observer = mock.Mock()
        task = SimpleNamespace(name="OsmAndTrack", tear_down=lambda env: None)

        def run_task(task, run_episode, env, demo_mode):
            return run_episode(task)

        suite = SimpleNamespace(_run_task=run_task)
        with observe_android_world(observer), watch_observed_episodes(suite):
            suite._run_task(task, lambda _: {"is_successful": 0.0}, None, False)
            suite._run_task(task, lambda _: {"exception_info": "adb offline"}, None, False)
        self.assertEqual(observer.begin.call_count, 2)
        self.assertEqual(observer.stop_capture.call_count, 2)
        self.assertEqual(observer.end.call_args_list, [mock.call(False, None), mock.call(None, "adb offline")])

    def test_observer_failure_is_reported_after_native_result_and_teardown(self):
        observer = mock.Mock()
        observer.begin.side_effect = RuntimeError("disk is full")
        task = SimpleNamespace(name="OsmAndTrack", tear_down=mock.Mock())

        def run_task(task, run_episode, env, demo_mode):
            run_episode(task)
            task.tear_down(env)
            return {"is_successful": 1.0}

        suite = SimpleNamespace(_run_task=run_task)
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "video observation failed"):
            with observe_android_world(observer), watch_observed_episodes(suite):
                self.assertEqual(suite._run_task(task, lambda _: None, None, False)["is_successful"], 1.0)
        observer.stop_capture.assert_called_once()
        observer.end.assert_called_once_with(True, None)
        task.tear_down.assert_called_once()
        self.assertIs(suite._run_task, run_task)

    def test_boxes_filter_inactive_controls_and_keep_model_indices(self):
        utils = SimpleNamespace(
            validate_ui_element=lambda item, size: item.is_visible,
            get_ui_element_bbox_pixels=mock.Mock(return_value=SimpleNamespace(
                x_min=-10, y_min=40, x_max=50, y_max=180)),
        )
        aw, agents = ModuleType("android_world"), ModuleType("android_world.agents")
        aw.agents, agents.m3a_utils = agents, utils
        env = SimpleNamespace(logical_screen_size=(100, 200), physical_frame_boundary=(0, 0, 100, 200), orientation=0)
        state = SimpleNamespace(ui_elements=[
            SimpleNamespace(is_visible=True),
            SimpleNamespace(is_visible=True, is_enabled=False, is_clickable=True),
            SimpleNamespace(is_visible=True, is_enabled=True, is_editable=True),
            SimpleNamespace(is_visible=False, is_clickable=True),
        ])
        with mock.patch.dict(sys.modules, {"android_world": aw, "android_world.agents": agents}):
            self.assertEqual(_observation_boxes(state, env), [{"bounds": [0.0, 0.2, 0.5, 0.9], "label": "2"}])

    def test_older_m3a_without_bbox_helper_keeps_letterbox_origin_and_rotation(self):
        aw, agents = ModuleType("android_world"), ModuleType("android_world.agents")
        aw.agents, agents.m3a_utils = agents, SimpleNamespace(validate_ui_element=lambda item, size: True)
        env = SimpleNamespace(logical_screen_size=(200, 100), physical_frame_boundary=(10, 20, 110, 220), orientation=1)
        element = SimpleNamespace(is_enabled=True, is_clickable=True, bbox_pixels=SimpleNamespace(
            x_min=30, y_min=10, x_max=100, y_max=40))
        state = SimpleNamespace(ui_elements=[element])
        with mock.patch.dict(sys.modules, {"android_world": aw, "android_world.agents": agents}):
            self.assertEqual(_observation_boxes(state, env), [{
                "bounds": [70 / 110, 50 / 220, 100 / 110, 120 / 220], "label": "0",
            }])


if __name__ == "__main__":
    unittest.main()
