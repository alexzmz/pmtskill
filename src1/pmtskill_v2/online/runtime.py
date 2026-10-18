"""Observation-driven adapter re-selection; never consult task truth/answers.

Runtime discounts are episode-local recovery heuristics, NOT new calibration
success rates. A GUI step is still one action, with its summary pinned to it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
from collections import Counter, deque

PORTS_RUNTIME_PROTOCOL = "ports-success-observation-routing-v2"
ACTION_PRIMITIVES = {"click": "action.click", "long_press": "action.long_press", "input_text": "action.type",
                     "open_app": "action.open_app", "scroll": "action.scroll", "swipe": "action.swipe",
                     "navigate_back": "action.back", "navigate_home": "action.home", "keyboard_enter": "action.enter"}


def action_dict(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if item is not None}


def screen_key(data, before=True):
    """Coarse visual+UI state, not the model's possibly optimistic summary."""
    elements = data.get("before_ui_elements" if before else "after_ui_elements")
    ui = [dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(e) if isinstance(e, dict) else e
          for e in elements] if elements else []
    # Ignore status-bar clock drift, not arbitrary counters/task values.
    for element in ui:
        if isinstance(element, dict) and "com.android.systemui" in str(element.get("package_name", "")):
            for field in ("text", "content_description"):
                if isinstance(element.get(field), str):
                    element[field] = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "<clock>", element[field])
    text = json.dumps(ui, sort_keys=True, default=str)
    visual = data.get("raw_screenshot" if before else "after_screenshot_with_som")
    visual_digest = ""
    if visual is not None:
        try:
            height = visual.shape[0]
            sample = visual[height // 8:7 * height // 8:24, ::24] // 16
            visual_digest = hashlib.sha256(sample.tobytes()).hexdigest()
        except (AttributeError, TypeError, IndexError):
            if isinstance(visual, bytes):
                visual_digest = hashlib.sha256(visual).hexdigest()
    if not ui and not visual_digest:
        return None
    return hashlib.sha256((text + visual_digest).encode()).hexdigest()


def current_ui_text(prompt):
    match = re.search(r"(?:list of descriptions for some UI elements on the current screen:|"
                      r"consistent with the ones in the labeled screenshot:)\s*(.*?)"
                      r"(?:\nHere are some useful guidelines|\Z)", prompt, re.I | re.S)
    return match.group(1) if match else ""


def grounded_summary(prompt):
    """Fallback when an action-trained adapter answers a summary with an action."""
    before = re.search(r"in the before screenshot:\s*(.*?)\nHere is the list for the after screenshot:", prompt, re.S)
    after = re.search(r"Here is the list for the after screenshot:\s*(.*?)\nThis is the action you picked:", prompt, re.S)
    action = re.search(r"This is the action you picked:\s*(.*?)\nBased on the reason:", prompt, re.S)
    if not before or not after:
        return "The previous action was attempted. No reliable summary was produced; inspect the current screenshot rather than assuming success."
    previous = set(before.group(1).splitlines())
    changed = [line.strip()[:220] for line in after.group(1).splitlines() if line.strip() and line not in previous]
    observation = ("New/current UI descriptions: " + " | ".join(changed[:5]) if changed else
                   "UI element descriptions did not change; that is not proof of success. Check the screenshot before repeating.")
    return "Attempted " + ((action.group(1)[:300] + ". ") if action else "the previous action. ") + observation


class AdaptiveRoutingSession:
    def __init__(self, router, pool, skills, *, include_candidates=False):
        self.router, self.pool, self.skills = router, pool, skills
        self.config = router.config
        self.include_candidates = include_candidates
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=8)
        self.failures = Counter()
        self.skill_failures = Counter()
        self.cooldowns = {}
        self.loop_hits = 0
        self.stage = 0
        self.actions = 0
        self.recovery_pending = False
        self.last_step = None
        self.last_decision = {}
        self.last_action = None

    def context_primitives(self, prompt, goal, binding):
        ui = current_ui_text(prompt)
        primitives = ["perceive.ui_tree", "ground.text"]
        leaves = binding.metadata.get("leaf_primitive_ids", []) if binding else []
        if leaves:
            primitives.append(leaves[min(self.stage, len(leaves) - 1)])
        if re.search(r"EditText|is_editable.?[:=].?true|editable.?[:=].?true", ui, re.I):
            primitives.append("action.type")
        if re.search(r"\b(count|total|sum|multiply|product|longest|how many|duration)\b", goal, re.I):
            primitives.extend(("reason.compare", "memory.read"))
        if re.search(r"\b(canvas|maze|drawing)\b", goal + " " + ui, re.I):
            primitives.extend(("ground.bbox", "reason.compare"))
        if self.recovery_pending:
            primitives.extend(("reason.recover", "reason.verify"))
        # Never infer that the goal is complete from a plan/call-count alone.
        return tuple(dict.fromkeys(p for p in primitives if p != "control.finish"))

    def select(self, planned, prompt, goal, binding=None):
        skill = self.skills.get(planned.skill_id or "")
        fallback = bool(skill and self.cooldowns.get(skill.skill_id, -1) > self.actions)
        if fallback:
            skill = None
        context = self.context_primitives(prompt, goal, binding)
        unit = dataclasses.replace(planned, skill_id=skill.skill_id if skill else None,
                                   primitive_ids=planned.primitive_ids if skill else context,
                                   is_polished=skill is not None)
        choices, scores = [], {}
        for model in self.pool.profiles.values():
            evidence = self.router.model_unit_support(model, unit, skill, include_candidates=self.include_candidates)
            if evidence is None:
                continue
            probability, latency, source = evidence
            # Only include available primitive-profile entries; unknowns should
            # not fabricate model specialization from adapter names.
            known = [model.capability(p) for p in context if p in model.capabilities]
            contextual = math.exp(sum(math.log(p) for p in known) / len(known)) if known else probability
            weight = self.config.runtime_context_weight if known else 0.0
            compatibility = math.exp((1 - weight) * math.log(max(probability, 1e-6))
                                     + weight * math.log(max(contextual, 1e-6)))
            compatibility *= self.config.stall_model_discount ** min(4, self.failures[model.model_id])
            switch = model.switch_cost_ms if self.pool.current_model_id not in {None, model.model_id} else 0
            penalty = (self.config.latency_weight * latency + self.config.switch_weight * switch
                       if self.config.objective == "weighted" else 0.0)
            score = self.config.success_weight * math.log(max(compatibility, 1e-6)) - penalty
            scores[model.model_id] = {"unit_support": probability, "context_prior": contextual,
                                       "runtime_compatibility_not_sr": compatibility, "score_source": source,
                                       "stall_discount_count": self.failures[model.model_id]}
            # Staying is only an exact-score tie-break, never a success penalty.
            choices.append((score, model.model_id == self.pool.current_model_id, model, probability, latency))
        if not choices and skill:
            # A disabled/unknown skill pair may not authorize skill injection on
            # another adapter. Explicit bare-model recovery is a distinct route.
            self.cooldowns[skill.skill_id] = self.actions + 1
            return self.select(planned, prompt, goal, binding)
        if not choices:
            raise RuntimeError("PORTS runtime: no eligible model; check capability thresholds/profiles")
        _, _, model, probability, latency = max(choices, key=lambda c: c[:2])
        previous = self.pool.current_model_id
        chosen = dataclasses.replace(unit, model_id=model.model_id, expected_success=probability,
                                     expected_latency_ms=latency)
        decision = {"protocol": PORTS_RUNTIME_PROTOCOL, "runtime_decision": True,
                    "adapter_switched": previous is not None and previous != model.model_id,
                    "stall_recovery": self.recovery_pending, "skill_fallback": fallback,
                    "context_primitives": list(context), "model_scores": scores,
                    "score_source": scores[model.model_id]["score_source"], "stage": self.stage}
        logging.info("PORTS_ROUTE action=%d model=%s previous=%s skill=%s recovery=%s fallback=%s scores=%s",
                     self.actions + 1, model.model_id, previous, chosen.skill_id,
                     self.recovery_pending, fallback, json.dumps(scores, ensure_ascii=False))
        self.last_step, self.last_decision = chosen, decision
        self.recovery_pending = False
        return chosen, decision

    def observe(self, data, binding=None):
        if self.last_step is None:
            return
        self.actions += 1
        action = action_dict(data.get("action_output_json"))
        self.last_action = action
        summary = str(data.get("summary") or "").lower()
        invalid = any(t in summary for t in ("not in the correct format", "can not parse", "out of range", "can not execute", "error calling"))
        before, after = screen_key(data, True), screen_key(data, False)
        kind = (action or {}).get("action_type")
        signature = (before, json.dumps(action, sort_keys=True)) if before is not None and action else None
        repeating = signature is not None and signature in self.recent
        # Repeated waits/answers and legitimate repeated taps on changing screens
        # are not loop evidence. A two-screen A/B toggle is loop evidence.
        bad = invalid or (repeating and kind not in {"wait", "answer", "status"})
        self.loop_hits = self.loop_hits + 1 if bad else max(0, self.loop_hits - 1)
        if signature:
            self.recent.append(signature)
        leaves = binding.metadata.get("leaf_primitive_ids", []) if binding else []
        if leaves and not bad and before and after and before != after:
            expected = leaves[min(self.stage, len(leaves) - 1)]
            if ACTION_PRIMITIVES.get(kind) == expected:
                self.stage = min(self.stage + 1, len(leaves) - 1)
        if self.loop_hits >= self.config.stall_threshold:
            self.failures[self.last_step.model_id] += 1
            if self.last_step.skill_id:
                self.skill_failures[self.last_step.skill_id] += 1
                self.cooldowns[self.last_step.skill_id] = (self.actions + 2 if self.skill_failures[self.last_step.skill_id] == 1
                                                          else math.inf)
            self.recovery_pending = True
            self.loop_hits = 0
            self.recent.clear()
            logging.warning("PORTS_STALL: repeated UI/action or invalid outputs; reselect adapter/skill next step (not an SR label)")

    def recovery_guidance(self):
        if not self.last_decision.get("stall_recovery"):
            return ""
        return ("\nRECOVERY: recent actions repeated without new UI progress or were invalid. "
                "Do not repeat that toggle/tap blindly. Inspect the actual screenshot and supported app names; "
                "choose another visible target or a different valid approach. Do not invent missing controls. "
                "A predicted success or summary is not proof that the goal was achieved.\n")
