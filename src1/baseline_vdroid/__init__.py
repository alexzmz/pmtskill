"""V-Droid AndroidWorld baseline integration.

This package deliberately keeps V-Droid isolated from PMT-Skill's online
planner, skill store, adapter pool, and dynamic router.  The only shared
surface is the benchmark protocol and the report schema.
"""

from .config import VDroidBaselineConfig, load_vdroid_config

__all__ = ["VDroidBaselineConfig", "load_vdroid_config"]

