"""Optional, in-process observers for real AndroidWorld M3A episodes."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from .reporter import successful_episode_value


class AndroidWorldObserver(Protocol):
    def begin(self, task_id: str) -> None: ...
    def observe(self, step: int | None, pixels: Any, boxes: list[dict[str, Any]]) -> None: ...
    def end(self, success: bool | None, error: str | None = None) -> None: ...


@dataclass
class _Session:
    observer: AndroidWorldObserver
    failure: Exception | None = None

    def call(self, callback, *args):
        try:
            callback(*args)
        except Exception as error:
            # A recorder error must never change the benchmark reward or trigger
            # AndroidWorld infrastructure recovery. Report it after evaluation.
            if self.failure is None:
                self.failure = error
            logging.exception("ANDROIDWORLD_OBSERVER_FAILED: %s", callback)


@dataclass
class _Attempt:
    session: _Session
    step: int = 0
    started: bool = False
    stopped: bool = False

    def stop(self):
        if self.started and not self.stopped:
            self.stopped = True
            callback = getattr(self.session.observer, "stop_capture", None)
            if callback is not None:
                self.session.call(callback)


_session: contextvars.ContextVar[_Session | None] = contextvars.ContextVar("aw_observer", default=None)
_attempt: contextvars.ContextVar[_Attempt | None] = contextvars.ContextVar("aw_observer_attempt", default=None)


@contextlib.contextmanager
def observe_android_world(observer: AndroidWorldObserver) -> Iterator[None]:
    """Register an observer for evaluations invoked in this synchronous context."""
    if _session.get() is not None:
        raise RuntimeError("AndroidWorld observation contexts cannot be nested")
    session = _Session(observer)
    token = _session.set(session)
    try:
        yield
        if session.failure is not None:
            raise RuntimeError("AndroidWorld evaluation finished, but video observation failed") from session.failure
    finally:
        _session.reset(token)


@contextlib.contextmanager
def _override(instance, name, replacement):
    """Restore descriptors as descriptors, not as newly bound instance methods."""
    values = vars(instance)
    existed, old = name in values, values.get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(instance, name, old)
        else:
            delattr(instance, name)


@contextlib.contextmanager
def watch_observed_episodes(suite_utils: Any) -> Iterator[None]:
    """Start after task setup, stop before cleanup, report native task outcome.

    Installed inside recovery/resume wrappers: each real retry is an attempt,
    and a resumed cached task never triggers recording callbacks.
    """
    session = _session.get()
    if session is None:
        yield
        return

    original = suite_utils._run_task
    signature = inspect.signature(original)

    @functools.wraps(original)
    def run_task(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        task = bound.arguments["task"]
        run_episode = bound.arguments["run_episode"]
        tear_down = task.tear_down
        attempt = _Attempt(session)
        token = _attempt.set(attempt)
        success, error = None, None

        def observed_episode(*episode_args, **episode_kwargs):
            if session.failure is None:
                attempt.started = True
                session.call(session.observer.begin, str(task.name))
            return run_episode(*episode_args, **episode_kwargs)

        def observed_teardown(*teardown_args, **teardown_kwargs):
            attempt.stop()
            return tear_down(*teardown_args, **teardown_kwargs)

        bound.arguments["run_episode"] = observed_episode
        try:
            with _override(task, "tear_down", observed_teardown):
                result = original(*bound.args, **bound.kwargs)
            if isinstance(result, Mapping) and "is_successful" in result and not result.get("exception_info"):
                success = successful_episode_value(result.get("is_successful"))
            else:
                reason = result.get("exception_info") if isinstance(result, Mapping) else None
                error = str(reason or "No AndroidWorld verdict")
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if attempt.started:
                    attempt.stop()
                    session.call(session.observer.end, success, error)
            finally:
                _attempt.reset(token)

    suite_utils._run_task = run_task
    try:
        yield
    finally:
        suite_utils._run_task = original


def _legacy_physical_bounds(element, logical_size, physical_frame, orientation):
    """Equivalent SoM transform for older M3A without the public bbox helper."""
    bbox = element.bbox_pixels
    if bbox is None:
        return None
    origin_x, origin_y, right, bottom = physical_frame
    logical_width, logical_height = logical_size
    width, height = right - origin_x, bottom - origin_y

    def transform(x, y):
        if orientation == 0:
            return origin_x + int(x * width / logical_width), origin_y + int(y * height / logical_height)
        if orientation == 1:
            return right - int(y * width / logical_height), origin_y + int(x * height / logical_width)
        if orientation == 2:
            return right - int(x * width / logical_width), bottom - int(y * height / logical_height)
        if orientation == 3:
            return origin_x + int(y * width / logical_height), bottom - int(x * height / logical_width)
        raise ValueError(f"Unsupported AndroidWorld orientation: {orientation}")

    x1, y1 = transform(int(bbox.x_min), int(bbox.y_min))
    x2, y2 = transform(int(bbox.x_max), int(bbox.y_max))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _observation_boxes(state: Any, environment: Any) -> list[dict[str, Any]]:
    # Use the same validation and orientation transform as the model's SoM.
    from android_world.agents import m3a_utils

    logical_size = environment.logical_screen_size
    physical_frame = environment.physical_frame_boundary
    orientation = environment.orientation
    # M3A rescales physical x/y by screenshot width/right and height/bottom.
    # Those screenshot dimensions cancel when expressing bounds in [0, 1].
    width, height = physical_frame[2:]
    if width <= 0 or height <= 0:
        return []

    def clip(value):
        return min(1.0, max(0.0, value))

    boxes = []
    physical_bbox = getattr(m3a_utils, "get_ui_element_bbox_pixels", None)
    for index, element in enumerate(state.ui_elements):
        if (getattr(element, "is_enabled", None) is False
                or not any(getattr(element, f"is_{kind}", False) for kind in
                           ("clickable", "long_clickable", "scrollable", "editable", "checkable"))
                or not m3a_utils.validate_ui_element(element, logical_size)):
            continue
        if physical_bbox is None:
            bounds_px = _legacy_physical_bounds(element, logical_size, physical_frame, orientation)
        else:
            bbox = physical_bbox(element, logical_size, physical_frame, orientation)
            bounds_px = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) if bbox is not None else None
        if bounds_px is None:
            continue
        left, top, right, bottom = bounds_px
        bounds = [clip(left / width), clip(top / height), clip(right / width), clip(bottom / height)]
        if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
            boxes.append({"bounds": bounds, "label": str(index)})
    return boxes


class ObservedM3A:
    """Read M3A's own observations before compaction, never poll the a11y tree.

    Video Step 1 is native episode ``step_number=0``. The post-action
    observation updates controls but does not add a new step marker.
    """

    def step(self, goal: str):
        attempt = _attempt.get()
        if attempt is None or not attempt.started or attempt.session.failure is not None:
            return super().step(goal)

        attempt.step += 1
        environment = self.env
        get_state, execute_action = environment.get_state, environment.execute_action
        initial = True
        acting = False

        def emit(step, state, captured_at):
            observer = attempt.session.observer
            boxes = _observation_boxes(state, environment)
            observe_at = getattr(observer, "observe_at", None)
            if callable(observe_at):
                observe_at(step, state.pixels, boxes, captured_at)
            else:
                observer.observe(step, state.pixels, boxes)

        def observed_state(*args, **kwargs):
            nonlocal initial
            state = get_state(*args, **kwargs)
            if not acting and attempt.session.failure is None:
                captured_at = time.monotonic()
                attempt.session.call(emit, attempt.step if initial else None, state, captured_at)
                initial = False
            return state

        def observed_action(*args, **kwargs):
            nonlocal acting
            acting = True
            try:
                return execute_action(*args, **kwargs)
            finally:
                acting = False

        with _override(environment, "get_state", observed_state), _override(environment, "execute_action", observed_action):
            return super().step(goal)
