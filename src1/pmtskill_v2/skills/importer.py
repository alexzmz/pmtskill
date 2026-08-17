"""把 ``libs/skvm/skvm-data/skills`` 转成可审计的 raw skill 记录。

SKVM 目录里的技能并不都针对 Android，因此导入后状态固定为 ``imported``，
不会直接进入在线执行。后续由编译器标注原语拓扑并经真实轨迹验证后才能晋升。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.models import SkillRecord, SkillStatus, SkillTopology, utc_now
from .store import SkillStore


@dataclass(slots=True)
class ImportSummary:
    scanned: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    relevant: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "android_relevant": self.relevant,
        }


def _parse_scalar(raw: str):
    value = raw.strip().strip("\"'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    return value


def parse_skill_markdown(text: str) -> tuple[dict[str, object], str]:
    """解析常见 YAML frontmatter；不要求额外安装 PyYAML。"""

    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return {}, normalized
    metadata: dict[str, object] = {}
    for line in normalized[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)
    return metadata, normalized[end + 5 :].strip()


_PRIMITIVE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("action.open_app", ("open app", "launch app", "android", "mobile app")),
    ("action.click", ("click", "tap", "button", "点击")),
    ("action.type", ("input", "type text", "form", "输入")),
    ("action.scroll", ("scroll", "滚动")),
    ("action.swipe", ("swipe", "滑动")),
    ("action.back", ("go back", "back button", "返回")),
    ("perceive.screenshot", ("screenshot", "screen", "视觉", "屏幕")),
    ("perceive.ocr", ("ocr", "read text", "文字识别")),
    ("ground.text", ("ui element", "selector", "text grounding", "定位")),
    ("reason.verify", ("verify", "validate", "check result", "校验")),
    ("reason.recover", ("retry", "recover", "failure", "重试")),
    ("reason.decompose", ("workflow", "steps", "plan", "任务分解")),
)


def infer_primitives(text: str) -> tuple[tuple[str, ...], bool]:
    """用确定性规则产生初始标签；云侧模型可在之后覆盖它。"""

    lowered = text.lower()
    matched = [
        primitive
        for primitive, keywords in _PRIMITIVE_KEYWORDS
        if any(keyword in lowered for keyword in keywords)
    ]
    android_relevant = any(
        token in lowered
        # 不能用 screen/click 这类过宽词，否则文档、PPT、前端技能都会被误判。
        for token in (
            "android",
            "mobile ui",
            "mobile app",
            "touchscreen",
            "appium",
            "uiautomator",
            "adb command",
            "tap gesture",
        )
    )
    if not matched:
        matched = ["reason.intent", "reason.decompose", "reason.verify"]
    elif matched[0] != "reason.intent":
        matched.insert(0, "reason.intent")
    if matched[-1] != "reason.verify":
        matched.append("reason.verify")
    return tuple(dict.fromkeys(matched)), android_relevant


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return normalized or "unnamed"


def load_skill_file(path: Path, root: Path) -> SkillRecord:
    """把单个 SKILL.md 转换为 raw skill，保留原始正文和哈希。"""

    raw_bytes = path.read_bytes()
    # 某些第三方技能不是严格 UTF-8；替换坏字符优于丢失整个技能。
    text = raw_bytes.decode("utf-8", errors="replace")
    metadata, body = parse_skill_markdown(text)
    relative = path.relative_to(root).as_posix()
    name = str(metadata.get("name") or path.parent.name)
    description = str(metadata.get("description") or "").strip()
    primitives, relevant = infer_primitives("\n".join((name, description, body)))
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return SkillRecord(
        skill_id=f"skvm:{_slug(relative.removesuffix('/SKILL.md'))}",
        name=name,
        description=description,
        kind="raw",
        status=SkillStatus.IMPORTED,
        level=1,
        topology=SkillTopology.from_sequence(primitives, topology_id=f"skvm:{digest[:16]}"),
        body=body,
        source_path=str(path.resolve()),
        source_hash=digest,
        metadata={
            "frontmatter": metadata,
            "android_relevant": relevant,
            "importer": "deterministic-v1",
        },
    )


def import_skvm_skills(root: str | Path, store: SkillStore) -> ImportSummary:
    """递归、幂等导入 SKVM 技能目录。"""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"SKVM 技能目录不存在: {root_path}")
    summary = ImportSummary()
    for path in sorted(root_path.rglob("SKILL.md")):
        summary.scanned += 1
        skill = load_skill_file(path, root_path)
        if skill.metadata["android_relevant"]:
            summary.relevant += 1
        same_hash = store.find_skill_by_source_hash(skill.source_hash or "")
        if same_hash:
            summary.skipped += 1
            continue
        existing = store.get_skill(skill.skill_id)
        if existing:
            # 保留已经验证出的状态与版本；只更新上游正文和来源哈希。
            skill.status = existing.status
            skill.version = existing.version + 1
            skill.created_at = existing.created_at
            skill.updated_at = utc_now()
            store.upsert_skill(skill)
            summary.updated += 1
        else:
            store.upsert_skill(skill)
            summary.inserted += 1
    return summary


def relevant_raw_skills(skills: Iterable[SkillRecord]) -> list[SkillRecord]:
    """只把初筛相关的 raw skill 提供给在线规划器，控制上下文长度。"""

    return [
        skill
        for skill in skills
        if skill.metadata.get("android_relevant")
        or skill.metadata.get("approved_for_planning")
    ]
