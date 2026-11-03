"""Timestamp, exact-match and control-overlay tests; no emulator required."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src_video.render import _choose_controls, _timeline, render_episode


class TimelineTests(unittest.TestCase):
    def test_reserves_tick_for_each_step_between_regular_ticks(self):
        frames = [{"t": 0, "path": "initial"},
                  {"t": .05, "path": "one", "source": "trajectory", "step": 1},
                  {"t": .06, "path": "two", "source": "trajectory", "step": 2},
                  {"t": 1, "path": "final"}]
        timeline, marks, matches = _timeline(frames, [{"t": .05, "path": "one", "step": 1},
                                                     {"t": .06, "path": "two", "step": 2}], 1, 2, str)
        self.assertEqual(timeline, [0, 1, 2, 3])
        self.assertEqual(marks, {1: 1, 2: 2})
        self.assertEqual([match["status"] for match in matches], ["exact", "exact"])

    def test_match_prefers_tagged_trajectory_frame_on_static_screen(self):
        frames = [{"t": 0, "path": "static"}, {"t": .49, "path": "static"},
                  {"t": .51, "path": "static", "source": "trajectory", "step": 4},
                  {"t": 1, "path": "end"}]
        _, marks, matches = _timeline(frames, [{"t": .50, "path": "static", "step": 4}], 1, 8, str)
        self.assertEqual(matches[0]["source_t"], .51)
        self.assertEqual(matches[0]["source"], "trajectory")
        self.assertEqual(marks[4], 4)

    def test_different_nearby_screen_is_not_marked(self):
        frames = [{"t": 0, "path": "same"}, {"t": 3, "path": "different"}]
        _, marks, matches = _timeline(frames, [{"t": 3, "path": "same", "step": 1}], 3, 8, str)
        self.assertEqual(marks, {})
        self.assertEqual(matches[0]["status"], "unmatched")

    def test_stale_sample_not_displayed_on_held_screenshot(self):
        sample = {"t": 0, "path": "a"}
        frame = {"t": .5, "path": "a"}
        self.assertIs(_choose_controls([sample], [0], frame, None, 1, display_t=.5), sample)
        self.assertIsNone(_choose_controls([sample], [0], frame, None, 1, display_t=2))

    def test_unlimited_mode_requires_screen_match(self):
        sample = {"t": 0, "path": "a"}
        frame = {"t": 20, "path": "b"}
        self.assertIs(_choose_controls([sample], [0], frame, lambda *_: True, None), sample)
        self.assertIsNone(_choose_controls([sample], [0], frame, lambda *_: False, None))
        self.assertIsNone(_choose_controls([{"t": 0}], [0], frame, None, None))


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for rendering")
class RenderTests(unittest.TestCase):
    def test_two_outputs_share_timeline_but_use_distinct_annotations(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (80, 121), "black").save(root / "a.png")
            Image.new("RGB", (80, 121), "white").save(root / "b.png")
            Image.new("RGB", (80, 121), "black").save(root / "step.png")
            captured = {}

            def encode(_executable, target, size, _fps, images):
                self.assertEqual(size, (80, 122))
                captured[target.stem] = list(images)
                target.write_bytes(b"mock-encoded")

            with patch("src_video.render.shutil.which", return_value="mock-ffmpeg"), patch("src_video.render._encode", encode):
                result = render_episode(root, frames=[{"t": 0, "path": "a.png"}, {"t": .6, "path": "b.png"}],
                                        steps=[{"t": .1, "path": "step.png", "step": 1}],
                                        controls=[{"t": 0, "path": "a.png", "boxes": [{"bounds": [10, 70, 40, 100],
                                                                                         "label": "0"}]}],
                                        duration=1, fps=4)
            self.assertEqual(result["matched_steps"], 1)
            self.assertEqual(result["control_overlay_frames"], 3)
            self.assertEqual(len(captured["steps"]), len(captured["controls"]))
            self.assertEqual(captured["steps"][1].getpixel((3, 70)), (255, 209, 43))
            self.assertEqual(captured["controls"][1].getpixel((10, 99)), (30, 240, 214))
            self.assertEqual(captured["controls"][-1].getpixel((10, 99)), (255, 255, 255))
            self.assertTrue(Path(result["step_video"]).is_file())
            self.assertTrue(Path(result["controls_video"]).is_file())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_ffmpeg_mp4_smoke(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (81, 121), "black").save(root / "a.png")
            result = render_episode(root, frames=[{"t": 0, "path": "a.png"}],
                                    steps=[{"t": 0, "path": "a.png", "step": 1}],
                                    controls=[], duration=.5, fps=4)
            for key in ("step_video", "controls_video"):
                self.assertIn(b"ftyp", Path(result[key]).read_bytes()[:32])


if __name__ == "__main__":
    unittest.main()
