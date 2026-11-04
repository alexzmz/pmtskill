import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .ports_workflow import COST, COUNT, EDGES, MODELS, PATH, PRIMITIVES, SWITCH, frame, write_avi


class WorkflowTest(unittest.TestCase):
    def test_chinese_labels_have_no_english_text(self):
        with patch("src_video.ports_workflow.font"), patch("PIL.ImageDraw.ImageDraw.text") as draw:
            for elapsed in (3, 9, 15, 23, 36, 43):
                frame(elapsed, "zh")
        labels = [call.args[1] for call in draw.call_args_list]
        self.assertIn("有向无环图与代价", " ".join(labels))
        for label in labels:
            self.assertNotRegex(label, "[a-zA-Z]")

    def test_dynamic_programming_matches_exhaustive_search(self):
        def paths(position):
            if position == len(PRIMITIVES):
                yield ()
            for edge in EDGES:
                if edge.start == position:
                    for rest in paths(edge.end):
                        yield (edge, *rest)

        def costs(path, previous=None):
            if not path:
                yield 0
            else:
                for model, cost in zip(MODELS, path[0].costs):
                    for remainder in costs(path[1:], model):
                        yield cost + remainder + (SWITCH if previous not in (None, model) else 0)

        routes = list(paths(0))
        self.assertEqual(COUNT, len(routes))
        self.assertEqual(COST, min(value for route in routes for value in costs(route)))
        self.assertEqual([model for _, model in PATH], ["A", "B", "A"])

    def test_avi_header_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sample.avi"
            write_avi(target, 24, iter((frame(3), frame(36))), 2)
            data = target.read_bytes()
            self.assertEqual(data[:4], b"RIFF")
            self.assertEqual(struct.unpack_from("<I", data, 4)[0], len(data) - 8)
            movi = data.index(b"movi")
            index = data.rindex(b"idx1")
            for i in range(2):
                tag, flags, offset, length = struct.unpack_from("<4sIII", data, index + 8 + 16 * i)
                self.assertEqual((tag, flags), (b"00dc", 16))
                self.assertEqual(data[movi + offset:movi + offset + 10],
                                 tag + struct.pack("<I", length) + b"\xff\xd8")
