"""Typed parameter round trips, including the real CalendarEvent class body."""
from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import enum
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src1.pmtskill_v2.evaluation import task_parameters as codec
from src1.pmtskill_v2.evaluation.skill_effectiveness_runner import frozen_suite
from src1.pmtskill_v2.evaluation.resume import fingerprint


def calendar_schema():
    """Load actual upstream field defaults/methods without Android/CUDA imports."""
    path = Path(__file__).parents[2] / "libs/android_world/android_world/task_evals/utils/sqlite_schema_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CalendarEvent")
    module = types.ModuleType(codec.SCHEMA_MODULE)
    module.__dict__.update(dataclasses=dataclasses, datetime=dt,
                           SQLiteRow=dataclasses.make_dataclass("SQLiteRow", [], frozen=True),
                           device_constants=types.SimpleNamespace(TIMEZONE="UTC"),
                           datetime_utils=types.SimpleNamespace(timestamp_to_localized_datetime=
                               lambda ts, **kwargs: dt.datetime.fromtimestamp(ts, dt.timezone.utc)))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, body], type_ignores=[]))
    with mock.patch.dict("sys.modules", {codec.SCHEMA_MODULE: module}):
        exec(compile(code, str(path), "exec"), module.__dict__)
    return module


class Suite(dict):
    pass


class CalendarTask:
    name = "SimpleCalendarAddOneEventRelativeDay"

    def __init__(self, params):
        self.params = params

    @property
    def goal(self):
        event = self.params["rows"][0]
        return f"Create {event.title} at {event.start_ts}, duration {event.duration_mins} minutes"


class TaskParametersTest(unittest.TestCase):
    def round_trip(self, value):
        return codec.decode(json.loads(json.dumps(codec.encode(value), allow_nan=False)))

    def test_actual_calendar_event_and_noise_rows_keep_type_and_properties(self):
        schema = calendar_schema()
        with mock.patch.dict("sys.modules", {codec.SCHEMA_MODULE: schema}):
            target = schema.CalendarEvent(start_ts=1800000000, end_ts=1800001800, title="Target", repeat_interval=86400)
            noise = schema.CalendarEvent(start_ts=1700000000, end_ts=1700003600, title="Noise")
            result = self.round_trip({"rows": [target], "noise_rows": [noise, noise], "seed": 42})
            self.assertIsInstance(result["rows"][0], schema.CalendarEvent)
            self.assertIsInstance(result["noise_rows"][1], schema.CalendarEvent)
            self.assertEqual(dataclasses.asdict(result["rows"][0]), dataclasses.asdict(target))
            self.assertEqual(result["rows"][0].duration_mins, 30)
            self.assertEqual(result["rows"][0].start_datetime, target.start_datetime)
            self.assertEqual(len(result["noise_rows"]), 2)

    def test_containers_dates_and_reserved_dictionary_key_are_lossless(self):
        value = {"tuple": (1, 2, 2), "set": {3, 1}, "frozen": frozenset([4]),
                 "datetime": dt.datetime(2026, 10, 19, 5, 6, 7, 1234, dt.timezone(dt.timedelta(hours=8)), fold=1),
                 "date": dt.date(2026, 10, 19), "time": dt.time(2, 3, fold=1),
                 "delta": dt.timedelta(days=-2, seconds=31, microseconds=18),
                 "bytes": b"\x00\xff", "keys": {(1, 2): "pair", 3: "int"},
                 "literal": {codec.TAG: "not really a tag", "value": {"nested": True}}}
        result = self.round_trip(value)
        self.assertEqual(result, value)
        self.assertIsInstance(result["tuple"], tuple)
        self.assertIsInstance(result["frozen"], frozenset)
        self.assertEqual(result["datetime"].fold, 1)

    def test_schema_enum_is_not_silently_converted_to_int(self):
        schema = calendar_schema()
        cls = enum.IntEnum("AllowedState", {"ACTIVE": 1}, module=codec.SCHEMA_MODULE)
        schema.AllowedState = cls
        with mock.patch.dict("sys.modules", {codec.SCHEMA_MODULE: schema}):
            self.assertIs(self.round_trip(cls.ACTIVE), cls.ACTIVE)

    def test_decoder_rejects_foreign_type_without_importing_it(self):
        with mock.patch.object(codec.importlib, "import_module") as imports:
            with self.assertRaisesRegex(ValueError, "trusted"):
                codec.decode({codec.TAG: "dataclass", "module": "os", "name": "system", "fields": {}})
            imports.assert_not_called()

    def test_changed_schema_fields_and_unknown_tags_are_rejected(self):
        schema = calendar_schema()
        with mock.patch.dict("sys.modules", {codec.SCHEMA_MODULE: schema}):
            encoded = codec.encode(schema.CalendarEvent(1800000000, 1800001800, "target"))
            del encoded["fields"]["repeat_rule"]
            with self.assertRaisesRegex(ValueError, "schema fields changed"):
                codec.decode(encoded)
        with self.assertRaisesRegex(ValueError, "Unknown task parameter"):
            codec.decode({codec.TAG: "pickle", "value": "untrusted"})

    def test_unsupported_objects_and_nonfinite_values_are_not_stringified(self):
        for value in (object(), float("nan"), float("inf")):
            with self.subTest(value=type(value)), self.assertRaises(TypeError):
                codec.encode(value)

    def test_calendar_suite_save_reload_and_resume_do_not_regenerate_parameters(self):
        schema = calendar_schema()
        with mock.patch.dict("sys.modules", {codec.SCHEMA_MODULE: schema}):
            params = {"seed": 42, "rows": [schema.CalendarEvent(1800000000, 1800001800, "saved-timestamp")],
                      "noise_rows": [schema.CalendarEvent(1700000000, 1700003600, "background")]}
            original = CalendarTask(params)
            create = mock.Mock(return_value=Suite({CalendarTask.name: [original]}))
            harness = types.SimpleNamespace(create_suite=create, Suite=Suite)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "suite.json"
                kwargs = dict(combinations=1, seed=42, family="android_world", environment=None, path=path)
                first, first_hash = frozen_suite(harness, {CalendarTask.name: CalendarTask}, [CalendarTask.name], **kwargs)
                stored = path.read_bytes()
                create.side_effect = AssertionError("must not regenerate")
                second, second_hash = frozen_suite(harness, {CalendarTask.name: CalendarTask}, [CalendarTask.name], **kwargs)
                self.assertEqual(first_hash, second_hash)
                self.assertEqual(path.read_bytes(), stored)
                self.assertEqual(second[CalendarTask.name][0].goal, original.goal)
                self.assertEqual(second[CalendarTask.name][0].params, first[CalendarTask.name][0].params)
                self.assertEqual(create.call_count, 1)

    def test_legacy_plain_json_suite_remains_readable_without_rewriting_its_hash(self):
        class PlainTask:
            name = "Plain"
            def __init__(self, params):
                self.params, self.goal = params, params["name"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            saved = {"specification": {"tasks": ["Plain"], "combinations": 1, "seed": 42, "family": "android_world"},
                     "instances": [{"task": "Plain", "instance_id": 0, "params": {"name": "original"}, "goal": "original"}]}
            path.write_text(json.dumps(saved), encoding="utf-8")
            before = path.read_bytes()
            harness = types.SimpleNamespace(Suite=Suite, create_suite=mock.Mock(side_effect=AssertionError("cached")))
            restored, digest = frozen_suite(harness, {"Plain": PlainTask}, ["Plain"], combinations=1, seed=42,
                                             family="android_world", environment=None, path=path)
            self.assertEqual(restored["Plain"][0].goal, "original")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(digest, fingerprint(saved))


if __name__ == "__main__":
    unittest.main()
