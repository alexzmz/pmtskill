"""可复核的语义技能挖掘：真实动作、去重 episode、分层准入、失败分母。

这里的比例是包含操作/子流程的 episode 的最终成功比例，不是该操作的独立
成功率。数据没有独立片段 evaluator，因而绝不写入学生能力或 model-skill 标定。
raw skill 仅提供分类先验，不能替代执行证据。
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import EvidenceMiningConfig
from ..core.io import load_primitives, write_json_atomic, write_jsonl
from ..core.models import ExecutionTrace, SkillRecord, SkillStatus, SkillTopology, TraceEvent
from .hierarchy import COMPILER_ID as LEGACY_COMPILER, _action, _values, parameterize
from .identity import skill_definition_hash
from .goal_schema import abstract_payload, goal_intent, goal_shape
from .store import SkillStore, classify_trajectory_quality, wilson_lower_bound


COMPILER_ID = "evidence-hierarchy-v2"
ACTION_PRIMITIVES = {
    "click": "action.click", "tap": "action.click", "long_press": "action.long_press",
    "input_text": "action.type", "type": "action.type", "type_text": "action.type",
    "open_app": "action.open_app", "scroll": "action.scroll", "swipe": "action.swipe",
    "navigate_back": "action.back", "navigate_home": "action.home",
    "keyboard_enter": "action.enter", "wait": "action.wait",
    "answer": "control.finish", "status": "control.finish",
}
GUARD = (
    "Use CURRENT goal values and the live screen, never demonstration values/coordinates/indices. "
    "Steps are guidance: skip completed steps and recover from dialogs. "
    "Perform ONE action per call; keep AndroidWorld Reason/Action format. "
    "Only finish after verifying the current goal; recompute answers from current UI."
)
FIELD_RULES = (
    ("first_name", r"first\s*name|given\s*name"), ("last_name", r"last\s*name|surname|family\s*name"),
    ("phone", r"phone|telephone|mobile number"), ("email", r"e-?mail"),
    ("search_query", r"search|query"), ("file_name", r"file\s*name|filename|rename"),
    ("latitude", r"latitude"), ("longitude", r"longitude"),
    ("duration", r"duration|minutes? long"), ("date", r"\bdate\b|birthday"),
    ("hour", r"\bhour\b"), ("minute", r"\bminute\b"), ("time", r"\btime\b"),
    ("amount", r"amount|price|cost|expense"), ("category", r"category"),
    ("body", r"description|content|body|message|note text"),
    ("title", r"title|name|heading"),
)
DOMAIN_RULES = (
    ("calendar", r"calendar|日历|日程"), ("contacts", r"contact|联系人"),
    ("messaging", r"sms|messenger|message|短信"), ("notes", r"markor|note|joplin|记事|笔记"),
    ("tasks", r"todo|to-do|tasks|待办"), ("files", r"file|文件"),
    ("clock", r"clock|alarm|remind|闹钟|提醒"), ("audio", r"audio|recorder|音频|录音"),
    ("maps", r"osmand|map|navigation|地图"), ("recipes", r"broccoli|recipe|食谱"),
    ("finance", r"expense|money|budget|账|财务"), ("system", r"settings|wifi|bluetooth|brightness|系统"),
)


@dataclass(frozen=True)
class Operation:
    role: str
    primitive: str
    instruction: str
    app: str
    segment: int = 0


@dataclass
class Episode:
    episode_id: str
    traces: list[ExecutionTrace]
    successful: bool
    task: str
    goal: str
    paths: list[list[Operation]]
    complete_paths: list[list[Operation]]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def _domain(text: str) -> str:
    for category, pattern in DOMAIN_RULES:
        if re.search(pattern, text, re.I):
            return category
    return "general"


def _field(text: str) -> str:
    for role, pattern in FIELD_RULES:
        if re.search(pattern, text, re.I):
            return role
    return "current_field"


def _current_intent(event: TraceEvent, primitive: str) -> str:
    unit = event.metadata.get("planned_unit", {})
    planned = unit.get("primitive_ids", event.primitive_ids) if isinstance(unit, dict) else ()
    executable = [p for p in planned if p.startswith("action.") or p == "control.finish"]
    if len(executable) == 1 and executable[0] == primitive:
        instruction = str(unit.get("instruction", "")) if isinstance(unit, dict) else ""
        if instruction:
            return instruction
    # 多动作计划不能当作实际一步。使用本次动作 Reason 的末尾当前意图，而不是总结。
    reason = re.split(r"\bAction:", event.action or "", maxsplit=1)[0]
    intent = re.split(r"\b(?:I will|I'll|I should|I need to|I can now)\b", reason, flags=re.I)[-1]
    sentences = re.split(r"(?<=[.!?])\s+", intent)
    verb = {
        "action.click": r"click|tap|select|press", "action.type": r"type|enter|input|fill",
        "action.long_press": r"long.press|hold", "action.scroll": r"scroll",
    }.get(primitive)
    if verb:
        matching = [s for s in sentences if re.search(verb, s, re.I)]
        if matching:
            return matching[-1]
    return intent[-350:]


def _click_role(intent: str) -> str:
    # 仅将已提交动作的意图分成有限操作族，不按整句生成数千个 L1。
    rules = (
        ("delete_all", r"delete all|remove all|clear all"),
        ("confirm_delete", r"confirm.{0,20}(?:delet|remov)|(?:delet|remov).{0,20}confirm"),
        ("delete", r"\bdelete\b|\bremove\b|trash"),
        ("permission", r"allow|grant|permission"),
        ("stop_recording", r"stop.{0,20}record|stop button|stop icon"),
        ("start_recording", r"start.{0,20}record|record button|recording button"),
        ("save", r"\bsave\b|\bdone\b|\bapply\b|\bsubmit\b|\bcommit\b"),
        ("cancel", r"\bcancel\b|\bdismiss\b|\bclose\b"),
        ("select_all", r"select all"), ("confirm", r"\bok\b|\bconfirm\b|\byes\b"),
        ("create", r"\bnew\b|\bcreate\b|\badd\b|plus|\+ button"),
        ("search", r"search|magnif"), ("menu", r"menu|overflow|three.dot"),
        ("settings", r"settings|preferences"), ("share", r"share|send"),
        ("toggle", r"toggle|switch|enable|disable|turn on|turn off|checkbox"),
        ("date_picker", r"date|calendar|month|year"), ("time_picker", r"time|hour|minute"),
        ("edit", r"edit|pencil|rename"),
        ("select_goal_item", r"target|goal|specified|matching|requested|named|titled|list item"),
    )
    if re.search(r"field|text box|textbox|input box", intent, re.I):
        return "focus." + _field(intent)
    for role, pattern in rules:
        if re.search(pattern, intent, re.I):
            return role
    return "activate_control"


def _instruction(role: str) -> str:
    if role.startswith("fill."):
        return f"Enter the CURRENT goal's {role[5:].replace('_', ' ')} into the matching visible field."
    if role.startswith("focus."):
        return f"Focus the visible {role[6:].replace('_', ' ')} field."
    instructions = {
        "launch_app": "Open the app required by the CURRENT goal.",
        "activate_control": "Locate and activate the control needed for the CURRENT goal on the live screen.",
        "select_goal_item": "Select the item matching the CURRENT goal, checking its visible identity.",
        "save": "Save or commit the current edit; confirm it persisted.",
        "create": "Open the creation form for the object requested by the CURRENT goal.",
        "delete": "Delete only the item requested by the CURRENT goal; check its identity first.",
        "delete_all": "Delete all ONLY when explicitly requested by the CURRENT goal.",
        "confirm_delete": "Confirm deletion only for the requested item/scope.",
        "permission": "Handle the permission dialog only as needed for the CURRENT goal.",
        "cancel": "Dismiss the blocking dialog or cancel the unintended edit.",
        "confirm": "Confirm the visible dialog after checking its meaning and the CURRENT goal.",
        "select_all": "Select all only within the list/scope requested by the CURRENT goal.",
        "search": "Open or submit search for the CURRENT goal's item.",
        "menu": "Open the relevant contextual menu on the current screen.",
        "settings": "Open the settings required by the CURRENT goal.",
        "toggle": "Set the requested option to the goal state, not blindly toggle it.",
        "date_picker": "Set/select the CURRENT goal's date on the visible picker.",
        "time_picker": "Set/select the CURRENT goal's time on the visible picker.",
        "edit": "Open editing for the item matching the CURRENT goal.",
        "share": "Send/share only to the destination requested by the CURRENT goal.",
        "start_recording": "Start recording as requested by the CURRENT goal.",
        "stop_recording": "Stop recording after the requested duration and preserve the result.",
        "long_press": "Long-press the item matching the CURRENT goal to open its context actions.",
        "back": "Go back one screen if needed; do not discard an unsaved goal edit.",
        "home": "Go to the Android home screen if required by the CURRENT goal.",
        "enter": "Submit the current input using the keyboard Enter action.",
        "wait": "Wait briefly for the requested operation/UI state, then recheck.",
        "verify_finish": "Verify the CURRENT goal against the live UI; recompute any requested answer before finishing.",
    }
    if role.startswith(("scroll.", "swipe.")):
        kind, direction = role.split(".", 1)
        return f"{kind.title()} {direction} to locate the CURRENT goal's item; inspect the new screen."
    return instructions[role]


def canonical_operations(trace: ExecutionTrace) -> tuple[list[Operation], bool]:
    """只按实际 Action 分类；保留重复动作，完整路径中不能悄悄删未知动作。"""
    result: list[Operation] = []
    complete = True
    # 未出现真实 open_app 时不将同一领域误当作同一 app（例如 Markor/Joplin）。
    # unknown-app 只能在该 task 内计重复，不能伪造跨 app 的 L2 复用证据。
    app = "unknown-app:" + trace.task_name.lower()
    values = _values(trace)
    segment = 0
    for event in trace.events:
        if event.metadata.get("action_committed", True) is False:
            complete = False
            segment += 1
            continue
        action = _action(event)
        action_type = action.get("action_type")
        primitive = ACTION_PRIMITIVES.get(action_type)
        if not primitive:
            complete = False
            segment += 1
            continue
        intent = parameterize(_current_intent(event, primitive), values)
        if primitive == "action.open_app":
            app = re.sub(r"\s+", " ", str(action.get("app_name") or app)).strip().lower()
            role = "launch_app"
        elif primitive == "action.type":
            role = "fill." + _field(intent)
        elif primitive == "action.click":
            role = _click_role(intent)
        elif primitive in {"action.scroll", "action.swipe"}:
            direction = str(action.get("direction", "toward_target")).lower()
            direction = direction if direction in {"up", "down", "left", "right"} else "toward_target"
            role = primitive[7:] + "." + direction
        else:
            role = {"action.long_press": "long_press", "action.back": "back", "action.home": "home",
                    "action.enter": "enter", "action.wait": "wait", "control.finish": "verify_finish"}[primitive]
        instruction = _instruction(role)
        # 短的单动作绑定保留 UI 标签；不使用多动作计划、不回放 input_text/answer。
        if primitive == "action.open_app":
            instruction = f"Open {app}, if it is the app required by the CURRENT goal."
        elif primitive in {"action.click", "action.long_press"}:
            label = re.search(r"(?:click|tap|press|select|hold)\s+(?:on\s+)?(?:the\s+)?['\"]([^'\"]{1,50})['\"]", intent, re.I)
            if label and "<goal_value>" not in label[1]:
                instruction += f" Expected control: {label[1]}."
        result.append(Operation(role, primitive, instruction, app, segment))
    return result, complete and bool(result)


def _episode_key(trace: ExecutionTrace) -> str:
    context = trace.metadata.get("collection_context", {})
    seed = context.get("suite_seed") if isinstance(context, dict) else None
    instance = trace.metadata.get("instance_id")
    if seed is None or instance is None:
        # 没有足够身份字段，不能把不同 trial 误合并。
        return "trace:" + trace.trace_id
    teacher = trace.metadata.get("teacher_model_id") or sorted({e.model_id for e in trace.events})
    return "episode:" + _digest([trace.task_name, trace.goal, seed, instance, teacher])


def _episodes(traces: list[ExecutionTrace]) -> tuple[dict[str, Episode], list[str]]:
    groups: defaultdict[str, list[ExecutionTrace]] = defaultdict(list)
    rejected = []
    for trace in traces:
        if classify_trajectory_quality(trace)["status"] == "rejected":
            rejected.append(trace.trace_id)
        else:
            groups[_episode_key(trace)].append(trace)
    episodes = {}
    for key, group in sorted(groups.items()):
        group.sort(key=lambda t: t.trace_id)
        extracted = [canonical_operations(trace) for trace in group]
        episodes[key] = Episode(key, group, all(t.successful for t in group), group[0].task_name,
                                group[0].goal, [p for p, _ in extracted], [p for p, ok in extracted if ok])
    return episodes, sorted(rejected)


def _lcs(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    row = [0] * (len(b) + 1)
    for x in a:
        next_row = [0]
        for j, y in enumerate(b):
            next_row.append(row[j] + 1 if x == y else max(row[j + 1], next_row[-1]))
        row = next_row
    return row[-1]


def _meaningful(roles: tuple[str, ...]) -> bool:
    boilerplate = {"launch_app", "activate_control", "back", "home", "wait", "verify_finish", "confirm", "cancel"}
    return any(role not in boilerplate and not role.startswith("scroll.") for role in roles)


def _evidence(ids: set[str], episodes: dict[str, Episode]) -> dict[str, Any]:
    success_ids = sorted(i for i in ids if episodes[i].successful)
    failure_ids = sorted(ids - set(success_ids))
    successes, trials = len(success_ids), len(ids)
    return {
        "trials": trials, "successes": successes, "failures": trials - successes,
        "contextual_episode_success_rate": successes / trials if trials else 0.0,
        "wilson_lower_95": wilson_lower_bound(successes, trials),
        "successful_task_count": len({episodes[i].task for i in success_ids}),
        "successful_goal_variants": len({(episodes[i].task, episodes[i].goal) for i in success_ids}),
        "successful_episode_ids": success_ids, "failed_episode_ids": failure_ids,
        "interpretation": "observational episode outcome, NOT independent fragment success or student SR",
    }


def _gate(level: int, evidence: dict[str, Any], config: EvidenceMiningConfig) -> list[str]:
    reasons = []
    for key, threshold in (
        ("successes", getattr(config, f"l{level}_min_successes")),
        ("contextual_episode_success_rate", getattr(config, f"l{level}_min_success_rate")),
        ("wilson_lower_95", getattr(config, f"l{level}_min_wilson_lower")),
    ):
        if evidence[key] < threshold:
            reasons.append(f"{key}<{threshold}")
    if level < 3 and evidence["successful_task_count"] < getattr(config, f"l{level}_min_tasks"):
        reasons.append(f"successful_task_count<{getattr(config, f'l{level}_min_tasks')}")
    return reasons


def raw_catalog(store: SkillStore) -> list[dict[str, Any]]:
    return [{"skill_id": s.skill_id, "name": s.name, "category": _domain(s.name + " " + s.description),
             "approved_for_planning": bool(s.metadata.get("approved_for_planning", False)),
             "role": "semantic_prior_only_not_execution_evidence"}
            for s in store.list_skills(kind="raw")]


def build_evidence_library(store: SkillStore, config: EvidenceMiningConfig) -> tuple[list[str], dict[str, Any]]:
    traces = store.list_traces()
    episodes, rejected = _episodes(traces)
    known = {p.primitive_id for p in load_primitives()}
    operation_support: defaultdict[str, set[str]] = defaultdict(set)
    operation_examples: dict[str, Operation] = {}
    subflow_support: defaultdict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    subflow_examples: dict[tuple[str, tuple[str, ...]], list[Operation]] = {}
    task_support: defaultdict[str, set[str]] = defaultdict(set)
    for episode_id, episode in episodes.items():
        task_support[episode.task].add(episode_id)  # 完整任务分母包括所有有效失败，非只含成功路径。
        for path in episode.paths:
            for operation in path:
                if operation.primitive not in known:
                    raise ValueError(f"原语目录缺少实际动作: {operation.primitive}")
                operation_support[operation.role].add(episode_id)
                operation_examples.setdefault(operation.role, operation)
            for length in range(config.subflow_min_actions, config.subflow_max_actions + 1):
                for start in range(len(path) - length + 1):
                    window = path[start:start + length]
                    roles = tuple(op.role for op in window)
                    if len({(op.app, op.segment) for op in window}) != 1 or not _meaningful(roles):
                        continue
                    key = (window[0].app, roles)
                    subflow_support[key].add(episode_id)
                    subflow_examples.setdefault(key, window)

    # Closed contiguous patterns：短窗口与更长窗口证据完全相同时，只保留信息更多的长者。
    repeated = {key for key, ids in subflow_support.items() if sum(episodes[i].successful for i in ids) >= 2}
    redundant = set()
    for longer in repeated:
        app, roles = longer
        for length in range(config.subflow_min_actions, len(roles)):
            for start in range(len(roles) - length + 1):
                shorter = (app, roles[start:start + length])
                if shorter in repeated and subflow_support[shorter] == subflow_support[longer]:
                    redundant.add(shorter)
    subflow_keys = repeated - redundant
    records: dict[str, SkillRecord] = {}
    operation_ids: dict[str, str] = {}
    subflow_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    catalog = raw_catalog(store)

    def add(level: int, identity: Any, name: str, units: list[Operation], ids: set[str],
            components: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> SkillRecord:
        skill_id = f"polished:evidence:l{level}:{_digest(identity)}:v2"
        evidence = _evidence(ids, episodes)
        reasons = _gate(level, evidence, config)
        evidence.update(extra or {})
        sequence = [op.primitive for op in units]
        roles = [op.role for op in units]
        scope = "whole_task_episode" if level == 3 else "episodes_containing_semantic_fragment"
        task_names = sorted({episodes[i].task for i in ids})
        positive_traces = [t for i in sorted(ids) if episodes[i].successful for t in episodes[i].traces]
        source_traces = [t for i in sorted(ids) for t in episodes[i].traces]
        source_skill_ids = set()
        for trace in positive_traces:
            source = trace.metadata.get("collection_source", {})
            source_id = source.get("skill_id") if isinstance(source, dict) else None
            if source_id and store.get_skill(str(source_id)):
                source_skill_ids.add(str(source_id))
        category = _domain(" ".join(task_names))
        related = [entry["skill_id"] for entry in catalog if entry["category"] == category
                   and category != "general" and entry["approved_for_planning"]]
        instructions = [_instruction(roles[0])] if level == 1 else [op.instruction for op in units]
        goal_templates = sorted({abstract_payload(parameterize(t.goal, _values(t)))[:650] for t in positive_traces}) if level == 3 else []
        skill = SkillRecord(
            skill_id, name, f"L{level} {scope}: " + (goal_templates[0][:350] if goal_templates else " / ".join(roles)),
            "polished", SkillStatus.ACTIVE if not reasons else SkillStatus.CANDIDATE, level,
            SkillTopology.from_sequence(sequence, topology_id=skill_id),
            body=GUARD + "\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(instructions, 1)),
            fallback_topology=SkillTopology.from_sequence(sequence),
            metadata={
                "compiler": COMPILER_ID, "admission_policy": "evidence_based", "student_calibrated": False,
                "hierarchy_role": {1: "operation", 2: "subflow", 3: "task"}[level],
                "operation_count": len(units), "operation_primitive_lengths": [1] * len(units),
                "instructions": instructions, "semantic_roles": roles, "components": components,
                "leaf_primitive_ids": sequence, "task_names": task_names, "calibration_tasks": task_names,
                "goal_templates": goal_templates, "source_trace_ids": sorted(t.trace_id for t in source_traces),
                "goal_shapes": [json.loads(v) for v in sorted({json.dumps(goal_shape(t.goal), sort_keys=True) for t in positive_traces})] if level == 3 else [],
                "goal_intents": sorted({goal_intent(t.goal) for t in positive_traces} - {None}) if level == 3 else [],
                "source_skill_ids": sorted(source_skill_ids),
                "teacher_model_ids": sorted({e.model_id for t in source_traces for e in t.events}),
                "evidence_scope": scope, "evidence": evidence, "support": evidence["successes"],
                "admission_reasons": reasons, "category": category,
                "related_raw_skill_ids": related, "raw_relation_method": "domain_keyword_prior_not_proven_derivation",
                "prompt_mode": "stepwise", "prompt_max_chars": config.prompt_max_chars,
            },
        )
        records[skill_id] = skill
        return skill

    for role in sorted(operation_support):
        example = operation_examples[role]
        skill = add(1, role, "L1 " + role, [example], operation_support[role], [])
        operation_ids[role] = skill.skill_id
    for key in sorted(subflow_keys):
        app, roles = key
        units = subflow_examples[key]
        components = [{"skill_id": operation_ids[role], "start": i, "end": i + 1} for i, role in enumerate(roles)]
        # L2 共享定义不带某条示范的 UI label；L3 保留逐位置绑定。
        generic = [dataclasses.replace(op, instruction=_instruction(op.role)) for op in units]
        skill = add(2, key, f"L2 {app}: " + " → ".join(roles), generic, subflow_support[key], components,
                    {"application": app, "contiguous_pattern": True, "closed_pattern": True})
        subflow_ids[key] = skill.skill_id
    ranked = sorted((s for s in records.values() if s.level == 2 and s.status == SkillStatus.ACTIVE),
                    key=lambda s: (-s.metadata["evidence"]["wilson_lower_95"], -s.metadata["support"],
                                   -s.metadata["operation_count"], s.skill_id))
    for skill in ranked[config.max_active_l2:]:
        skill.status = SkillStatus.CANDIDATE
        skill.metadata["admission_reasons"].append("active_subflow_budget")

    incomplete_success = []
    for task, ids in sorted(task_support.items()):
        successful = [episodes[i] for i in sorted(ids) if episodes[i].successful]
        choices = [(episode.episode_id, path) for episode in successful for path in episode.complete_paths]
        if not choices:
            incomplete_success.extend(t.trace_id for e in successful for t in e.traces)
            continue
        # 每个 episode 对 medoid 投一票，重复采集不能增加某条路线的权重。
        episode_paths = {e.episode_id: [tuple(op.role for op in p) for p in e.complete_paths] for e in successful}
        def similarity(path: list[Operation], origin: str) -> float:
            roles = tuple(op.role for op in path)
            other_episodes = [paths for episode_id, paths in episode_paths.items() if episode_id != origin]
            return sum(max((_lcs(roles, other) / max(len(roles), len(other)) for other in paths), default=0.0)
                       for paths in other_episodes) / max(1, len(other_episodes))
        # 共识只来自其它 episode；自己的 100% 匹配不能替自己投票。
        chosen_id, units = min(choices, key=lambda item: (-similarity(item[1], item[0]), len(item[1]), item[0],
                                                         tuple(op.instruction for op in item[1])))
        roles = tuple(op.role for op in units)
        components = []
        start = 0
        while start < len(units):
            matches = []
            for end in range(start + config.subflow_min_actions, min(len(units), start + config.subflow_max_actions) + 1):
                key = (units[start].app, roles[start:end])
                child_id = subflow_ids.get(key)
                if child_id and len({op.app for op in units[start:end]}) == 1:
                    child = records[child_id]
                    matches.append((child.status == SkillStatus.ACTIVE, end - start, child_id, end))
            if matches:
                _, _, child_id, end = max(matches)
            else:
                child_id, end = operation_ids[units[start].role], start + 1
            components.append({"skill_id": child_id, "start": start, "end": end})
            start = end
        representative = episodes[chosen_id].traces[0]
        template = abstract_payload(parameterize(representative.goal, _values(representative)))
        requires_variation = "<goal_value>" in template
        skill = add(3, task, task + " workflow", units, ids, components,
                    {"representative_episode_id": chosen_id, "shared_path_fraction": similarity(units, chosen_id),
                     "shared_fraction_scope": "mean normalized LCS against OTHER successful episode cohorts; no self vote",
                     "requires_parameter_variation": requires_variation,
                     "representative_path_support": sum(roles in paths for paths in episode_paths.values())})
        reasons = skill.metadata["admission_reasons"]
        if requires_variation and skill.metadata["evidence"]["successful_goal_variants"] < config.l3_min_goal_variants:
            reasons.append(f"successful_goal_variants<{config.l3_min_goal_variants}")
        if similarity(units, chosen_id) < config.l3_min_shared_fraction:
            reasons.append(f"shared_path_fraction<{config.l3_min_shared_fraction}")
        if reasons:
            skill.status = SkillStatus.CANDIDATE

    changed = []
    retired = []
    for previous in store.list_skills(kind="polished"):
        if previous.metadata.get("compiler") in {COMPILER_ID, LEGACY_COMPILER} and previous.skill_id not in records:
            if previous.status != SkillStatus.DEPRECATED and not store.has_any_skill_model_status(previous.skill_id):
                store.set_skill_status(previous.skill_id, SkillStatus.DEPRECATED)
                retired.append(previous.skill_id)
    for skill in sorted(records.values(), key=lambda s: s.skill_id):
        previous = store.get_skill(skill.skill_id)
        if previous:
            skill.created_at = previous.created_at
            skill.version = previous.version + int(skill_definition_hash(previous) != skill_definition_hash(skill))
            if store.has_any_skill_model_status(skill.skill_id):
                skill.status = previous.status  # 不强制覆盖已做的模型配对标定/回滚。
        if (not previous or skill_definition_hash(previous) != skill_definition_hash(skill)
                or skill.metadata != previous.metadata or skill.status != previous.status):
            store.upsert_skill(skill)
            changed.append(skill.skill_id)
    previous_edges = {(r["parent_skill_id"], r["child_skill_id"], r["relation_type"]): r for r in store.list_skill_relations()}
    for skill in records.values():
        relation_children = [(c["skill_id"], "composes_polished", [c["start"], c["end"]]) for c in skill.metadata["components"]]
        relation_children += [(s, "related_raw", None) for s in skill.metadata["related_raw_skill_ids"]]
        relation_children += [(s, "derived_from_raw" if store.get_skill(s).kind == "raw" else "derived_from_input_skill", None)
                              for s in skill.metadata["source_skill_ids"]]
        merged: defaultdict[tuple[str, str], list[list[int]]] = defaultdict(list)
        for child, relation, span in relation_children:
            if span is not None:
                merged[(child, relation)].append(span)
            else:
                merged[(child, relation)]
        for (child, relation), spans in sorted(merged.items()):
            count = 0 if relation == "related_raw" else skill.metadata["support"]
            detail = {"compiler": COMPILER_ID, "operation_spans": sorted(spans),
                      "evidence_scope": "taxonomy_prior" if relation == "related_raw" else "episode_level_composition"}
            prior = previous_edges.get((skill.skill_id, child, relation))
            if not prior or prior["evidence_count"] != count or prior["detail"] != detail:
                store.upsert_skill_relation(skill.skill_id, child, relation, evidence_count=count, detail=detail)

    active = [s for s in records.values() if s.status == SkillStatus.ACTIVE]
    covered = sorted({t for s in active if s.level == 3 for t in s.metadata["task_names"]})
    observed = sorted({t.task_name for t in traces})
    valid_observed = sorted(task_support)
    report = {
        "compiler": COMPILER_ID, "admission_policy": "evidence_based", "student_calibrated": False,
        "rules": dataclasses.asdict(config), "traces_total": len(traces),
        "valid_traces": sum(len(e.traces) for e in episodes.values()), "rejected_traces": len(rejected),
        "rejected_trace_ids": rejected, "quality_counts": dict(Counter(classify_trajectory_quality(t)["status"] for t in traces)),
        "episode_cohorts": len(episodes), "successful_episode_cohorts": sum(e.successful for e in episodes.values()),
        "failed_episode_cohorts": sum(not e.successful for e in episodes.values()),
        "duplicate_trace_count": sum(len(e.traces) - 1 for e in episodes.values()),
        "conflicting_episode_ids": sorted(i for i, e in episodes.items() if len({t.successful for t in e.traces}) > 1),
        "episode_grouping": "task+exact_goal+suite_seed+instance+teacher; missing identity uses trace_id; mixed outcomes count as failure",
        "raw_skills": len(catalog), "raw_planning_approved": sum(r["approved_for_planning"] for r in catalog),
        "primitive_count": len(known), "polished_skills": len(records),
        "levels": {f"L{n}": sum(s.level == n for s in records.values()) for n in (1, 2, 3)},
        "active_levels": {f"L{n}": sum(s.level == n for s in active) for n in (1, 2, 3)},
        "candidate_levels": {f"L{n}": sum(s.level == n and s.status == SkillStatus.CANDIDATE for s in records.values()) for n in (1, 2, 3)},
        "subflow_patterns_seen": len(subflow_support), "subflow_single_success_or_zero_discarded": len(subflow_support) - len(repeated),
        "redundant_closed_subflows_discarded": len(redundant),
        "observed_task_count": len(observed), "valid_observed_task_count": len(valid_observed),
        "covered_task_count": len(covered), "covered_tasks": covered,
        "uncovered_tasks": sorted(set(observed) - set(covered)),
        "coverage_scope": "active L3 among tasks present in input, not entire AndroidWorld or student performance",
        "incomplete_successful_trace_ids": sorted(incomplete_success), "retired_legacy_or_missing_skills": retired,
        "warning": "Teacher observational evidence only. 2/2 is not proof of high SR. Wilson bounds assume independent trials, which this small fixed-seed dataset does not establish. Use held-out student evaluation.",
    }
    store.log_maintenance_event("evidence_library_built", None, report)
    return changed, report


def export_evidence_library(store: SkillStore, directory: Path, report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    skills = [s for s in store.list_skills(kind="polished") if s.metadata.get("compiler") == COMPILER_ID]
    write_json_atomic(directory / "coverage.json", report)
    write_json_atomic(directory / "mining_rules.json", {"compiler": COMPILER_ID, **report["rules"]})
    write_json_atomic(directory / "raw_catalog.json", raw_catalog(store))
    write_jsonl(directory / "skills.jsonl", (s.to_dict() for s in skills))
    write_json_atomic(directory / "hierarchy.json", {
        "primitives": [p.to_dict() for p in store.list_primitive_catalog()],
        "skills": [{"skill_id": s.skill_id, "name": s.name, "level": s.level, "status": s.status.value,
                    "components": s.metadata["components"], "primitive_ids": s.metadata["leaf_primitive_ids"]} for s in skills],
        "relations": store.list_skill_relations(),
    })
    fields = ["skill_id", "name", "level", "status", "category", "successes", "failures", "trials",
              "contextual_episode_success_rate", "wilson_lower_95", "successful_task_count", "successful_goal_variants", "admission_reasons"]
    with (directory / "skill_evidence.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for skill in sorted(skills, key=lambda s: (s.level, s.name)):
            row = {k: skill.metadata["evidence"].get(k, "") for k in fields}
            row.update(skill_id=skill.skill_id, name=skill.name, level=skill.level, status=skill.status.value,
                       category=skill.metadata["category"], admission_reasons="; ".join(skill.metadata["admission_reasons"]))
            writer.writerow(row)
    rows = ["# AndroidWorld 分层证据技能库", "",
            f"输入：{report['raw_skills']} raw；{report['valid_traces']} 有效轨迹；{report['rejected_traces']} 无效轨迹不计分母。",
            f"去重：{report['episode_cohorts']} episode 身份组，{report['successful_episode_cohorts']} 成功、{report['failed_episode_cohorts']} 失败。冲突组保守计失败。", "",
            f"生成：{report['levels']}；active：{report['active_levels']}；candidate：{report['candidate_levels']}。",
            f"active L3 覆盖 {report['covered_task_count']}/{report['observed_task_count']} 个输入任务类。", "",
            "L1：有限语义操作族，以真实 Action 去重，要求跨任务证据。",
            "L2：同 app 的重复连续子流程，closed-pattern 去冗余；比 L1 要求更高观察成功比例。",
            "L3：每任务类一条成功 medoid 路线，所有有效失败均入分母；参数化任务要求不同 goal 成功及路线共识。", "",
            "raw 仅作分类/关联先验；related_raw 的 evidence_count=0，不伪造来源或独立执行成功。",
            "L1/L2 比例是包含片段的整集成功率；L3 是任务观察成功率。均非学生 SR。",
            "当前数据最多常见 2/2，Wilson 下界约 0.34；不能据此宣称真实高成功率或泛化提升。", "",
            "## 每个任务的准入结果", "", "| Task | Status | 成功/有效 | 不准入原因 |", "|---|---|---:|---|"]
    for skill in sorted((s for s in skills if s.level == 3), key=lambda s: s.name):
        evidence = skill.metadata["evidence"]
        rows.append(f"| {skill.metadata['task_names'][0]} | {skill.status.value} | {evidence['successes']}/{evidence['trials']} | {'; '.join(skill.metadata['admission_reasons']) or '通过经验准入'} |")
    # 这是生成程序的输出，不是对已有用户文件的编辑。
    (directory / "coverage.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
