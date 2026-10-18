"""Uniform random adapter routing, followed by the selected model's real SKVM.

The baseline never consults PORTS compatibility estimates, DP routes or labels.
Random draws occur once per GUI step. Action and summary use the same adapter.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..baseline_skvm.evaluator import (
    MultimodalAdaptiveClient, MultimodalSkillProxy, legacy_kernel_module,
)
from ..baseline_skvm.protocol import SKVM_CONTEXT_POLICY, SKVM_STABILITY_POLICY, SKVM_TOOL_PROTOCOL
from ..baseline_skvm.protocol import has_episode_artifacts
from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.evaluation.android_world import AndroidWorldSimpleSkillEvaluator
from ..pmtskill_v2.evaluation.resume import fingerprint, model_resume_identity
from ..pmtskill_v2.inference.model_pool import ModelPool
from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
from .runner import run_gui_baseline


PROTOCOL = "uniform-random-routing-skvm-v1"


def prepare_model(config, profile, args, skills, target):
    """One TCP/AOT/variant set per actual adapter; never share another TCP."""
    legacy = legacy_kernel_module()
    sources = []
    for skill in sorted(skills, key=lambda item: item.skill_id):
        name = "skill-" + fingerprint(skill.skill_id)[:20]
        path = target / "skvm" / "sources" / name / "SKILL.md"
        body = ("---\nname: " + name + "\ndescription: "
                + json.dumps(skill.name + ": " + skill.description, ensure_ascii=False)
                + "\n---\n\n" + skill.body + "\n")
        sources.append(legacy.SkillSource(name, skill.name + ": " + skill.description,
                                          path, body, legacy._sha256(body)))
    identity = fingerprint({"model": model_resume_identity(profile),
                            "skills": [(s.skill_id, s.sha256) for s in sources],
                            "passes": args.skvm_aot_pass_sets, "instances": args.skvm_profile_instances})
    cache = (Path(args.skvm_cache_dir).expanduser().resolve() if args.skvm_cache_dir
             else config.paths.state_dir / "random_skvm_cache") / identity
    options = argparse.Namespace(
        skvm_cache_dir=cache, skvm_target_model=None, skvm_compiler_model="target",
        skvm_aot_pass_sets=args.skvm_aot_pass_sets, skvm_command=args.skvm_command,
        skvm_prepare=args.skvm_prepare, skvm_profile_instances=args.skvm_profile_instances,
        skvm_profile_concurrency=1, skvm_compile_concurrency=1, skvm_profile_timeout_ms=120000,
        skvm_compile_timeout_ms=300000, skvm_profile_primitives=None, vllm_api_key="skvm-local",
        skvm_failure_policy=args.skvm_failure_policy, skvm_compile_retries=args.skvm_compile_retries,
        skvm_context_window=args.max_model_len or config.training_evaluation.max_model_len,
        skill_mode="inject", skvm_variant_policy="aot",
    )
    for source in sources:
        source.path.parent.mkdir(parents=True, exist_ok=True)
        source.path.write_text(source.content, encoding="utf-8")
    kernel = legacy.SkVMKernel(options, skills=sources, run_dir=target,
                              vllm_base_url=profile.base_url, served_model=profile.served_model)
    formal = has_episode_artifacts(Path(args.output_dir).expanduser().resolve()) if args.output_dir else False
    frozen_path = target / "skvm" / "variant-identity.json"
    if formal and (args.skvm_prepare == "force" or not frozen_path.is_file()):
        raise ValueError("随机 SKVM 已有正式 episode，不能重建/补建模型变体；请使用新 output-dir")
    if formal and args.skvm_failure_policy == "fallback" and not (target / "skvm" / "preparation.json").is_file():
        raise ValueError("随机 SKVM 的冻结 preparation 丢失；拒绝混合续跑")
    variants = kernel.prepare()
    variant_identity = fingerprint([[v.skill.skill_id, v.source, v.tag, v.sha256, v.plan] for v in variants])
    if formal and json.loads(frozen_path.read_text(encoding="utf-8"))["identity"] != variant_identity:
        raise ValueError("随机 SKVM 变体已改变，请使用新 output-dir")
    write_json_atomic(frozen_path, {"identity": variant_identity})
    write_json_atomic(target / "skvm" / "skill-catalog.json", kernel.manifest(variants))
    write_json_atomic(target / "skvm" / "source-ids.json",
                      {source.skill_id: skill.skill_id for source, skill in
                       zip(sources, sorted(skills, key=lambda item: item.skill_id))})
    proxy = MultimodalSkillProxy(OpenAICompatibleVLClient(profile))
    adaptive = legacy.SkVMAdaptiveWrapper(
        proxy, variants=variants, target_model=kernel.target_model, mode="inject", variant_policy="aot",
        max_skills=2, max_skill_chars=12000, trace_path=target / "skvm" / "adaptations.jsonl", trace_include_goal=False,
    )
    stats_path = target / "skvm" / "runtime-stats.json"
    if stats_path.is_file():
        adaptive.restore_runtime_stats(json.loads(stats_path.read_text(encoding="utf-8")))
    original_record = adaptive._record_event

    def record(event):
        adaptive.last_selected = event.get("selected", [])
        write_json_atomic(stats_path, adaptive.get_stats())
        if args.store_trajectories:
            original_record(event)

    adaptive._record_event = record
    return MultimodalAdaptiveClient(adaptive, proxy), adaptive, kernel.preparation_summary


class RandomSKVMClient:
    def __init__(self, profiles, clients, adaptives, seed):
        self.seed, self.episode_id = seed, "unbound"
        self.pool = ModelPool(list(profiles), client_factory=lambda p: clients[p.model_id])
        self.model_ids = [p.model_id for p in profiles]
        self.adaptives = adaptives
        self.reset()

    def start_episode(self, goal, identity):
        self.episode_id = identity
        self.reset()

    def reset(self):
        self.random = random.Random(int(fingerprint([self.seed, self.episode_id]), 16))
        self.current = None
        self.step_index = -1
        self.pool.reset_counters()
        for adaptive in self.adaptives.values():
            adaptive.reset_skill_session()

    def begin_step(self, goal):
        self.current = self.random.choice(self.model_ids)
        self.step_index += 1

    def predict_mm(self, prompt, images):
        if self.current is None:
            raise RuntimeError("Random SKVM must bind a GUI step before an LLM call")
        output, safe, raw = self.pool.client(self.current).predict_mm(prompt, images)
        if isinstance(raw, dict):
            raw = dict(raw)
            raw["_pmtskill"] = {**raw.get("_pmtskill", {}), "model_id": self.current,
                                "routing_mode": PROTOCOL, "random_step_index": self.step_index,
                                "selection_policy": "uniform-per-gui-step; summary-pinned",
                                "model_specific_skvm": True}
        return output, safe, raw

    def predict(self, prompt):
        return self.predict_mm(prompt, [])

    def runtime_stats(self):
        return {"per_model_skvm": {key: adaptive.get_stats() for key, adaptive in self.adaptives.items()},
                "counter_scope": "all-invocations-including-retries",
                "switch_count_source": "per-episode action metadata in report, not a synthetic target"}


def run_random_skvm(config, store, profiles, args, tasks):
    if len(profiles) < 2:
        raise ValueError("Random routing SKVM 至少需要两个不同 adapter")
    skills = AndroidWorldSimpleSkillEvaluator(config, store)._skills(args.include_candidates)
    if not skills:
        raise ValueError("Random SKVM 没有可用技能；不会退化为裸随机模型")

    def prepare(target):
        clients, adaptives, compilation = {}, {}, {}
        for profile in profiles:
            child = target / "models" / ("model-" + fingerprint(profile.model_id)[:16])
            clients[profile.model_id], adaptives[profile.model_id], compilation[profile.model_id] = prepare_model(
                config, profile, args, skills, child,
            )
        return RandomSKVMClient(profiles, clients, adaptives, args.routing_seed), {"skvm_compilation": compilation}

    return run_gui_baseline(
        config, profiles, args, tasks, protocol=PROTOCOL, prepare_client=prepare, skills=skills, store=store,
        extra={"routing_seed": args.routing_seed, "policy": "uniform-per-gui-step-summary-pinned-v1",
               "skvm_pass_sets": args.skvm_aot_pass_sets, "skvm_instances": args.skvm_profile_instances,
               "skvm_failure_policy": args.skvm_failure_policy, "skvm_compile_retries": args.skvm_compile_retries,
               "skvm_context_window": args.max_model_len or config.training_evaluation.max_model_len,
               "skvm_protocol": [SKVM_TOOL_PROTOCOL, SKVM_CONTEXT_POLICY, SKVM_STABILITY_POLICY]},
        metadata={"implementation": "random-routing-real-skvm", "dynamic_model_routing": True,
                  "selection_policy": "uniform independent draws per GUI step; repeats permitted",
                  "routing_seed": args.routing_seed, "uses_ports_scores": False,
                  "per_model_tcp_and_aot": True, "original_fallback_reported": True},
    )
