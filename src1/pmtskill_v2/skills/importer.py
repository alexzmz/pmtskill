"""把 SKVM 或任意 ``SKILL.md`` 技能簇转成可审计的 raw skill 记录。

导入器只读取技能包，绝不会执行 ``scripts/`` 中的代码。除入口文档外，还会为
``references/``、``scripts/`` 等资源生成有界 manifest 和文本摘要，使 Teacher 在
标准轨迹采集时能理解完整技能，同时避免把一个大目录无界塞进模型上下文。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
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
    resources: int = 0
    skipped_resources: int = 0
    root: str = ""
    namespace: str = ""
    # inserted/updated/skipped 都必须返回 ID。第三种采集入口据此只处理本次目录，
    # 不会误把 SQLite 中其他 imported skill 一起送给 Teacher。
    skill_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "android_relevant": self.relevant,
            "resources": self.resources,
            "skipped_resources": self.skipped_resources,
            "root": self.root,
            "namespace": self.namespace,
            "skill_ids": list(self.skill_ids),
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
    """解析常见 YAML frontmatter；不要求额外安装 PyYAML。

    这里只消费顶层标量，并支持真实技能簇常见的 ``description: >``/``|`` 多行
    写法。嵌套 ``metadata`` 不会被错误拍平成顶层键；完整 frontmatter 仍保留在
    原始 ``SKILL.md`` 的包哈希中。
    """

    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return {}, normalized
    metadata: dict[str, object] = {}
    lines = normalized[4:end].splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            not line.strip()
            or line.lstrip().startswith("#")
            or line[:1].isspace()
            or ":" not in line
        ):
            index += 1
            continue
        key, raw_value = line.split(":", 1)
        marker = raw_value.strip()
        if marker in {">", ">-", ">+", "|", "|-", "|+"}:
            block: list[str] = []
            index += 1
            while index < len(lines):
                nested = lines[index]
                if nested and not nested[:1].isspace():
                    break
                block.append(nested.strip())
                index += 1
            value = (
                "\n".join(block).strip()
                if marker.startswith("|")
                else " ".join(item for item in block if item).strip()
            )
            metadata[key.strip()] = value
            continue
        # 空值通常表示后面是嵌套 mapping/list；本轻量解析器不展开它。
        if marker:
            metadata[key.strip()] = _parse_scalar(marker)
        index += 1
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


_IGNORED_PACKAGE_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
}
_TEXT_RESOURCE_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".sh",
    ".ps1",
    ".js",
    ".ts",
}
_MAX_PACKAGE_FILES = 512
_MAX_CONTEXT_FILE_BYTES = 256 * 1024
_MAX_CONTEXT_CHARS_PER_FILE = 2500
_MAX_CONTEXT_CHARS_TOTAL = 12000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_role(relative: Path) -> str:
    if relative.as_posix() == "SKILL.md":
        return "entrypoint"
    first = relative.parts[0].lower() if relative.parts else ""
    if first == "references":
        return "reference"
    if first == "scripts":
        return "script"
    if first == "assets":
        return "asset"
    return "resource"


def _package_inventory(
    entrypoint: Path,
    package_entrypoints: Iterable[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str, bool]:
    """扫描单个技能包；嵌套技能包有自己的边界，不会被父包重复吸收。"""

    package_root = entrypoint.parent.resolve()
    nested_roots = {
        item.parent.resolve()
        for item in package_entrypoints
        if item.resolve() != entrypoint.resolve()
        and package_root in item.parent.resolve().parents
    }
    candidates: list[Path] = []
    skipped: list[dict[str, object]] = []
    for item in sorted(package_root.rglob("*")):
        if item.is_symlink():
            skipped.append(
                {
                    "path": item.relative_to(package_root).as_posix(),
                    "role": "resource",
                    "bytes": 0,
                    "reason": "symlink_not_followed",
                }
            )
            continue
        if not item.is_file():
            continue
        relative = item.relative_to(package_root)
        if any(part in _IGNORED_PACKAGE_PARTS for part in relative.parts):
            continue
        if any(root == item or root in item.parents for root in nested_roots):
            continue
        if len(candidates) >= _MAX_PACKAGE_FILES:
            raise ValueError(
                f"技能包文件数超过安全上限 {_MAX_PACKAGE_FILES}: {package_root}"
            )
        candidates.append(item)

    manifest: list[dict[str, object]] = []
    context_parts: list[str] = []
    context_chars = 0
    context_truncated = False
    for item in candidates:
        relative = item.relative_to(package_root)
        size = item.stat().st_size
        role = _resource_role(relative)
        manifest.append(
            {
                "path": relative.as_posix(),
                "role": role,
                "bytes": size,
                "sha256": _sha256_file(item),
            }
        )
        if (
            role == "entrypoint"
            or item.suffix.lower() not in _TEXT_RESOURCE_SUFFIXES
            or size > _MAX_CONTEXT_FILE_BYTES
        ):
            continue
        if context_chars >= _MAX_CONTEXT_CHARS_TOTAL:
            context_truncated = True
            continue
        text = item.read_text(encoding="utf-8", errors="replace")
        excerpt = text[:_MAX_CONTEXT_CHARS_PER_FILE]
        remaining = _MAX_CONTEXT_CHARS_TOTAL - context_chars
        excerpt = excerpt[:remaining]
        context_parts.append(f"### {relative.as_posix()}\n{excerpt}")
        context_chars += len(excerpt)
        if len(text) > len(excerpt):
            context_truncated = True

    return manifest, skipped, "\n\n".join(context_parts), context_truncated


def _package_key(path: Path, root: Path) -> str:
    relative_parent = path.parent.resolve().relative_to(root.resolve())
    if not relative_parent.parts:
        return _slug(root.name)
    return _slug(relative_parent.as_posix())


def _package_relative_dir(path: Path, root: Path) -> str:
    relative_parent = path.parent.resolve().relative_to(root.resolve())
    return relative_parent.as_posix() if relative_parent.parts else "."


def _skill_id(path: Path, root: Path, namespace: str) -> str:
    """组合 namespace 与包路径；默认 namespace 已含单包名时避免重复一遍。"""

    key = _package_key(path, root)
    if path.parent.resolve() == root.resolve() and namespace.endswith(f":{key}"):
        return namespace
    return f"{namespace}:{key}"


def load_skill_file(
    path: Path,
    root: Path,
    *,
    namespace: str = "skvm",
    package_entrypoints: Iterable[Path] = (),
) -> SkillRecord:
    """把单个技能包转换为 raw skill，保留正文、资源 manifest 和包级哈希。"""

    raw_bytes = path.read_bytes()
    # 某些第三方技能不是严格 UTF-8；替换坏字符优于丢失整个技能。
    text = raw_bytes.decode("utf-8", errors="replace")
    metadata, body = parse_skill_markdown(text)
    name = str(metadata.get("name") or path.parent.name)
    description = str(metadata.get("description") or "").strip()
    primitives, relevant = infer_primitives("\n".join((name, description, body)))
    entries = tuple(package_entrypoints) or (path,)
    manifest, skipped, source_context, context_truncated = _package_inventory(
        path, entries
    )
    package_hash = hashlib.sha256(raw_bytes).hexdigest()
    if manifest:
        canonical = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        package_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package_key = _package_key(path, root)
    # source_hash 同时包含稳定来源身份和内容。相同内容位于不同 namespace/path 时仍
    # 是两个可独立维护的 raw skill；package_hash 则保留纯内容指纹用于跨库比较。
    source_hash = hashlib.sha256(
        f"{namespace}\0{_package_relative_dir(path, root)}\0{package_hash}".encode(
            "utf-8"
        )
    ).hexdigest()
    package_root = path.parent.resolve()
    return SkillRecord(
        skill_id=_skill_id(path, root, namespace),
        name=name,
        description=description,
        kind="raw",
        status=SkillStatus.IMPORTED,
        level=1,
        topology=SkillTopology.from_sequence(
            primitives, topology_id=f"imported:{package_hash[:16]}"
        ),
        body=body,
        source_path=str(path.resolve()),
        source_hash=source_hash,
        metadata={
            "frontmatter": metadata,
            "android_relevant": relevant,
            "importer": "skill-package-v2",
            "import_namespace": namespace,
            "package_root": str(package_root),
            "package_entrypoint": str(path.resolve()),
            "package_relative_path": package_key,
            "package_relative_dir": _package_relative_dir(path, root),
            "package_hash": package_hash,
            "package_manifest": manifest,
            "package_skipped_resources": skipped,
            "teacher_source_context": source_context,
            "teacher_source_context_truncated": context_truncated,
        },
    )


def scan_skill_cluster(
    root: str | Path,
    *,
    namespace: str,
) -> tuple[SkillRecord, ...]:
    """只读扫描一个技能包或技能簇；既支持 ``root/SKILL.md`` 也支持递归目录。"""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"技能簇目录不存在: {root_path}")
    entrypoints = tuple(
        path
        for path in sorted(root_path.rglob("SKILL.md"))
        if not any(part in _IGNORED_PACKAGE_PARTS for part in path.relative_to(root_path).parts)
    )
    skills = [
        load_skill_file(
            path,
            root_path,
            namespace=namespace,
            package_entrypoints=entrypoints,
        )
        for path in entrypoints
    ]
    # 两条不同路径 slug 后可能相同。只给碰撞项附加“路径哈希”，内容更新不会改变 ID。
    by_id: dict[str, list[SkillRecord]] = {}
    for skill in skills:
        by_id.setdefault(skill.skill_id, []).append(skill)
    for duplicate_id, duplicates in by_id.items():
        if len(duplicates) <= 1:
            continue
        for skill in duplicates:
            relative = str(skill.metadata.get("package_relative_dir", skill.skill_id))
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
            skill.skill_id = f"{duplicate_id}-{suffix}"
    return tuple(skills)


_COMPILE_METADATA_KEYS = {
    "raw_skill_compiler",
    "raw_skill_compiler_model",
    "raw_skill_compile_reason",
    "raw_skill_compiled",
    "approved_for_planning",
    "original_body",
    "original_topology",
}


def import_skill_cluster(
    root: str | Path,
    store: SkillStore,
    *,
    namespace: str,
) -> ImportSummary:
    """递归、幂等导入任意技能簇，并返回本次簇精确对应的 skill IDs。"""

    root_path = Path(root).expanduser().resolve()
    skills = scan_skill_cluster(root_path, namespace=namespace)
    summary = ImportSummary(root=str(root_path), namespace=namespace)
    for skill in skills:
        summary.scanned += 1
        if skill.metadata["android_relevant"]:
            summary.relevant += 1
        summary.resources += len(skill.metadata.get("package_manifest", ()))
        summary.skipped_resources += len(
            skill.metadata.get("package_skipped_resources", ())
        )
        same_hash = store.find_skill_by_source_hash(skill.source_hash or "")
        if same_hash:
            summary.skipped += 1
            summary.skill_ids.append(same_hash.skill_id)
            continue
        existing = store.get_skill(skill.skill_id)
        if existing:
            # 包内容变更会生成新版本。保留人工 task 绑定等扩展 metadata，但清除
            # 旧 compiler 结论，避免“新正文沿用旧标注/标定”的隐性污染。
            preserved = dict(existing.metadata)
            for key in _COMPILE_METADATA_KEYS:
                preserved.pop(key, None)
            preserved.update(skill.metadata)
            skill.metadata = preserved
            skill.status = SkillStatus.IMPORTED
            skill.version = existing.version + 1
            skill.created_at = existing.created_at
            skill.updated_at = utc_now()
            store.upsert_skill(skill)
            summary.updated += 1
        else:
            store.upsert_skill(skill)
            summary.inserted += 1
        summary.skill_ids.append(skill.skill_id)
    summary.skill_ids = list(dict.fromkeys(summary.skill_ids))
    return summary


def import_skvm_skills(root: str | Path, store: SkillStore) -> ImportSummary:
    """兼容旧调用：以稳定 ``skvm:`` namespace 导入默认技能簇。"""

    return import_skill_cluster(root, store, namespace="skvm")


def relevant_raw_skills(skills: Iterable[SkillRecord]) -> list[SkillRecord]:
    """只把初筛相关的 raw skill 提供给在线规划器，控制上下文长度。"""

    return [
        skill
        for skill in skills
        if skill.metadata.get("android_relevant")
        or skill.metadata.get("approved_for_planning")
    ]
