"""Lossless JSON task parameters; never unpickle or import user-chosen types.

Calendar task params contain CalendarEvent objects, including background rows.
Converting them to strings/plain dicts breaks the validator on reconstruction.
Only the harness's fixed SQLite-schema module may supply dataclass/enum types.
"""
from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import importlib
import math

PROTOCOL = "androidworld-task-parameters-v1"
TAG = "__pmtskill_parameter_type__"
SCHEMA_MODULE = "android_world.task_evals.utils.sqlite_schema_utils"


def _schema_type(module, name, base_check):
    if module != SCHEMA_MODULE or not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Task parameter type is not in the trusted AndroidWorld schema module")
    cls = getattr(importlib.import_module(SCHEMA_MODULE), name, None)
    if not isinstance(cls, type) or cls.__module__ != module or not base_check(cls):
        raise ValueError(f"Unsupported AndroidWorld parameter type: {module}.{name}")
    return cls


def encode(value):
    """Preserve types and ordered/repeated values, without lossy default=str."""
    if isinstance(value, enum.Enum):
        cls = type(value)
        _schema_type(cls.__module__, cls.__qualname__, lambda c: issubclass(c, enum.Enum))
        return {TAG: "enum", "module": cls.__module__, "name": cls.__qualname__, "value": encode(value.value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Task parameter contains a non-finite float")
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        _schema_type(cls.__module__, cls.__qualname__, dataclasses.is_dataclass)
        fields = dataclasses.fields(value)
        if any(not f.init for f in fields):
            raise TypeError(f"Cannot safely restore non-init fields in {cls.__name__}")
        return {TAG: "dataclass", "module": cls.__module__, "name": cls.__qualname__,
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields}}
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {TAG: type(value).__name__, "value": value.isoformat(),
                **({"fold": value.fold} if isinstance(value, (dt.datetime, dt.time)) else {})}
    if isinstance(value, dt.timedelta):
        return {TAG: "timedelta", "days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}
    if isinstance(value, bytes):
        return {TAG: "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if TAG in value or any(not isinstance(k, str) for k in value):
            return {TAG: "mapping", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, (tuple, set, frozenset)):
        items = [encode(v) for v in value]
        if isinstance(value, (set, frozenset)):
            import json
            items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return {TAG: type(value).__name__, "items": items}
    raise TypeError(f"Unsupported task parameter {type(value).__module__}.{type(value).__qualname__}; "
                    "add a lossless codec rather than converting it to str")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if TAG not in value:
        return {k: decode(v) for k, v in value.items()}
    kind = value[TAG]
    if kind == "dataclass":
        cls = _schema_type(value["module"], value["name"], dataclasses.is_dataclass)
        fields = {f.name for f in dataclasses.fields(cls) if f.init}
        if fields != set(value["fields"]):
            raise ValueError(f"AndroidWorld schema fields changed for {cls.__name__}; use a new experiment directory")
        return cls(**{k: decode(v) for k, v in value["fields"].items()})
    if kind == "enum":
        cls = _schema_type(value["module"], value["name"], lambda c: issubclass(c, enum.Enum))
        return cls(decode(value["value"]))
    if kind in {"datetime", "date", "time"}:
        result = getattr(dt, kind).fromisoformat(value["value"])
        return result.replace(fold=value.get("fold", 0)) if kind != "date" else result
    if kind == "timedelta":
        return dt.timedelta(days=value["days"], seconds=value["seconds"], microseconds=value["microseconds"])
    if kind == "bytes":
        return base64.b64decode(value["value"], validate=True)
    if kind == "mapping":
        return {decode(k): decode(v) for k, v in value["items"]}
    if kind in {"tuple", "set", "frozenset"}:
        return {"tuple": tuple, "set": set, "frozenset": frozenset}[kind](decode(v) for v in value["items"])
    raise ValueError(f"Unknown task parameter encoding: {kind!r}")
