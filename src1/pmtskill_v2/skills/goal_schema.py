"""只从用户 goal 提取输入形状；不能读取 evaluator 真值或 task ID。"""
from __future__ import annotations

import csv
import re


def goal_intent(goal: str) -> str | None:
    header = goal.splitlines()[0].strip().lower() if goal.splitlines() else ""
    for name, pattern in (
        ("delete", r"^(?:please\s+)?(?:delete|remove|clear)\b"),
        ("create", r"^(?:please\s+)?(?:add|create|insert)\b"),
        ("edit", r"^(?:please\s+)?(?:change|edit|update|rename)\b"),
        ("reply", r"^(?:please\s+)?(?:reply|respond)\b"),
        ("send", r"^(?:please\s+)?(?:send|resend)\b"),
        ("record", r"^(?:please\s+)?record\b"),
    ):
        if re.search(pattern, header):
            return name
    return None


def goal_shape(goal: str) -> dict[str, int]:
    listed = re.search(r"^delete the following expenses from[^:]+:\s*(.+)$", goal, re.I)
    if listed:
        items = next(csv.reader([listed[1].rstrip(". ")], skipinitialspace=True))
        return {"payload_item_count": len(items)}
    blocks = re.findall(r"(?:^|\n)\s*Expense\s*:", goal, re.I)
    if blocks:
        return {"payload_item_count": len(blocks)}
    lines = [line.strip() for line in goal.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "|" in line and re.search(r"title|name|amount|description", line, re.I):
            width = len(line.split("|"))
            rows = [row for row in lines[index + 1:] if len(row.split("|")) == width
                    and not re.fullmatch(r"[| :\-]+", row)]
            if rows:
                return {"payload_item_count": len(rows)}
    return {}


def abstract_payload(template: str) -> str:
    # 多行或已压成单行的 following 数据表不是技能定义本身。
    match = re.search(r"\bfollowing\b[^:]{0,140}:", template, re.I)
    if match:
        # 草稿任务的 Do NOT hit save 是约束，不是可丢弃的实例数据。
        constraint = re.search(r"\b(?:do\s+not|don't|without\s+saving)\b.*$", template[match.end():], re.I)
        return template[:match.end()] + " <goal_value>" + (" " + constraint[0] if constraint else "")
    return template
