"""评测 episode 的内存压缩；collector 轨迹不使用这里的裁剪。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# 训练评测只需要这些字段生成失败分类和 PMT-Skill 轻量 trace。M3A 每一步还会
# 产生三份截图、两份 UI tree 和完整 prompt；把它们跨 task 累积会占用数十 GB。
_EVALUATION_STEP_FIELDS = (
    "action_output",
    "action_output_json",
    "action_reason",
    "action_raw_response",
    "summary",
    "summary_raw_response",
)


def compact_evaluation_step_data(value: Any) -> dict[str, Any]:
    """保留评测/路由所需字段，剥离截图、UI tree 和重复 prompt。"""

    if not isinstance(value, Mapping):
        return {}
    return {
        field: value[field]
        for field in _EVALUATION_STEP_FIELDS
        if field in value
    }


def compact_m3a_step_result(result: Any) -> Any:
    """原地压缩 M3A result.data，也同步释放 agent.history 中的同一字典。"""

    data = getattr(result, "data", None)
    if not isinstance(data, dict):
        return result
    compact = compact_evaluation_step_data(data)
    data.clear()
    data.update(compact)
    return result
