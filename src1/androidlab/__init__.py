"""AndroidLab benchmark adapter; no device or optional imports at import time."""

PROTOCOL = "ports-androidlab-bridge-v1"
METHODS = ("standalone", "simple-skill", "skvm", "vdroid", "androidworld",
           "echotrail-gui", "random-skvm", "ports")
MULTI_MODEL_METHODS = {"ports", "random-skvm"}
SKILL_METHODS = {"simple-skill", "skvm", "random-skvm", "ports"}
