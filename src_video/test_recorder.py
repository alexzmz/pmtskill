from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .__main__ import load_settings
from .recorder import EpisodeRecorder


class RecorderTest(unittest.TestCase):
    def test_example_routes_one_real_episode(self):
        from src1.pmtskill_v2.cli import build_parser

        example = Path(__file__).with_name("config.example.toml")
        for mode in ("standalone", "ports"):
            settings = load_settings(example, mode)
            with patch.dict("sys.modules", {"src1.androidlab.catalog": None,
                                            "src1.androidlab.driver": None}):
                args = build_parser().parse_args(settings["argv"])
            self.assertEqual(settings["serial"], "emulator-5554")
            self.assertEqual(args.tasks, ["OsmAndTrack"])
            self.assertEqual(args.combinations, 1)
            self.assertEqual(args.cuda_visible_devices, "0")
            self.assertTrue(args.store_trajectories)
            self.assertFalse(args.resume)

    def test_sample_and_exact_step_share_timeline(self):
        import numpy as np
        from PIL import Image

        pixels = np.zeros((20, 10, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EpisodeRecorder(Path(temporary), adb="adb", serial="emulator-5554",
                                       capture_fps=10, command_timeout=1)
            with patch.object(recorder, "_adb", return_value=buffer.getvalue()) as adb:
                recorder.begin("Example")
                recorder.observe(1, pixels, [{"bounds": [.2, .3, .9, .8], "label": "0"}])
                recorder.stop_capture()
                recorder.end(False)
            directory = recorder.episodes[0]
            data = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
            self.assertFalse(data["success"])
            self.assertGreater(data["adb_frame_count"], 0)
            self.assertTrue(any(frame.get("final_capture") for frame in data["frames"]))
            self.assertTrue(any(frame["source"] == "trajectory" for frame in data["frames"]))
            self.assertEqual(data["controls"][0]["boxes"][0]["bounds"], [2, 6, 9, 16])
            self.assertEqual(len(list((directory / "frames").glob("*.png"))), 1)
            self.assertTrue(all(call.args == ("exec-out", "screencap", "-p") for call in adb.call_args_list))


if __name__ == "__main__":
    unittest.main()
