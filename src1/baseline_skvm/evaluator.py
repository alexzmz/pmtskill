"""Reuse the existing SKVM kernel, but keep the src1 M3A evaluation protocol.

No PORTS router or offline skill estimates are used here. SKVM profiles the
deployed adapter, compiles AOT variants and adapts every multimodal request.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.evaluation.resume import fingerprint, model_resume_identity


def legacy_kernel_module():
    # The old runner is stdlib-only at import time. Do not bootstrap its T3A
    # harness: only its SKVM compiler/artifact/runtime implementation is reused.
    source = str(Path(__file__).resolve().parents[2] / "src")
    sys.path.insert(0, source)
    try:
        return importlib.import_module("task_runner_skvm")
    finally:
        sys.path.remove(source)


class MultimodalSkillProxy:
    """Preserve images despite the legacy text-only SKVM wrapper's predict()."""

    def __init__(self, client):
        self.client = client
        self.images: list[Any] = []

    def predict(self, prompt):
        return self.client.predict_mm(prompt, self.images)


class MultimodalAdaptiveClient:
    def __init__(self, adaptive, proxy):
        self.adaptive, self.proxy = adaptive, proxy

    def predict_mm(self, prompt, images):
        self.proxy.images = images
        try:
            text, safe, raw = self.adaptive.predict(prompt)
            selected = getattr(self.adaptive, "last_selected", [])
            if isinstance(raw, dict) and selected:
                raw = dict(raw)
                raw["_pmtskill"] = {**raw.get("_pmtskill", {}),
                                    "skill_id": selected[0]["skill_id"],
                                    "skvm_selections": selected,
                                    "routing_mode": "fixed_adapter_skvm_aot"}
            return text, safe, raw
        finally:
            self.proxy.images = []

    def predict(self, prompt):
        return self.predict_mm(prompt, [])


def run_skvm(config, store, profile, args, tasks):
    from ..pmtskill_v2.evaluation.android_world import (
        AndroidWorldSimpleSkillEvaluator, _close_environment, _evaluation_progress,
        _run_evaluation_suite, episodes_to_traces,
    )
    from ..pmtskill_v2.evaluation.compaction import compact_m3a_step_result
    from ..pmtskill_v2.evaluation.recovery import ensure_valid_evaluation_episodes
    from ..pmtskill_v2.evaluation.reporter import write_evaluation_report
    from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
    from ..pmtskill_v2.offline.collector import bootstrap_android_world

    legacy = legacy_kernel_module()
    target = Path(args.output_dir).expanduser().resolve()
    skills = AndroidWorldSimpleSkillEvaluator(config, store)._skills(args.include_candidates)
    if not skills:
        raise ValueError("SKVM 没有可用技能；不会悄悄退化成 standalone")
    # Same corpus as simple-skills, with collision-free IDs and original bodies.
    sources = []
    for skill in sorted(skills, key=lambda item: item.skill_id):
        name = "skill-" + fingerprint(skill.skill_id)[:20]
        path = target / "skvm" / "sources" / name / "SKILL.md"
        body = ("---\nname: " + name + "\ndescription: "
                + json.dumps(skill.name + ": " + skill.description, ensure_ascii=False)
                + "\n---\n\n" + skill.body + "\n")
        sources.append(legacy.SkillSource(name, skill.name + ": " + skill.description,
                                          path, body, legacy._sha256(body)))
    identity = fingerprint({
        "model": model_resume_identity(profile),
        "skills": [(item.skill_id, item.sha256) for item in sources],
        "passes": args.skvm_aot_pass_sets,
        "profile_instances": args.skvm_profile_instances,
    })
    cache = (Path(args.skvm_cache_dir).expanduser().resolve() if args.skvm_cache_dir
             else config.paths.state_dir / "skvm_baseline_cache") / identity
    kernel_args = argparse.Namespace(
        skvm_cache_dir=cache, skvm_target_model=None, skvm_compiler_model="target",
        skvm_aot_pass_sets=args.skvm_aot_pass_sets, skvm_command=args.skvm_command,
        skvm_prepare=args.skvm_prepare, skvm_profile_instances=args.skvm_profile_instances,
        skvm_profile_concurrency=1, skvm_compile_concurrency=1,
        skvm_profile_timeout_ms=120_000, skvm_compile_timeout_ms=300_000,
        skvm_profile_primitives=None, vllm_api_key="skvm-local",
        skill_mode="inject", skvm_variant_policy="aot",
    )
    # Create/validate the resumable experiment before writing compiler artifacts.
    progress = _evaluation_progress(
        target, config, mode="skvm-fixed-adapter", profiles=[profile], tasks=tasks,
        combinations=args.combinations, seed=args.seed, family=args.family,
        max_steps=args.max_steps, resume=bool(args.resume), skills=skills, store=store,
        store_trajectories=args.store_trajectories,
        extra={"skvm_identity": identity, "protocol": "skvm-aot-m3a-v1"},
    )
    for source in sources:
        source.path.parent.mkdir(parents=True, exist_ok=True)
        source.path.write_text(source.content, encoding="utf-8")
    kernel = legacy.SkVMKernel(kernel_args, skills=sources, run_dir=target,
                              vllm_base_url=profile.base_url, served_model=profile.served_model)
    variants = kernel.prepare()
    manifest = kernel.manifest(variants)
    write_json_atomic(target / "skvm" / "skill-catalog.json", manifest)
    write_json_atomic(target / "skvm" / "source-ids.json",
                      {source.skill_id: skill.skill_id for source, skill in
                       zip(sources, sorted(skills, key=lambda item: item.skill_id))})
    proxy = MultimodalSkillProxy(OpenAICompatibleVLClient(profile))
    adaptive = legacy.SkVMAdaptiveWrapper(
        proxy, variants=variants, target_model=kernel.target_model, mode="inject",
        variant_policy="aot", max_skills=2, max_skill_chars=12000,
        trace_path=target / "skvm" / "adaptations.jsonl", trace_include_goal=False,
    )
    original_record = adaptive._record_event

    def record_event(event):
        adaptive.last_selected = event.get("selected", [])
        if args.store_trajectories:
            original_record(event)

    # Keep per-request selection only in memory when trajectory storage is off.
    adaptive._record_event = record_event
    client = MultimodalAdaptiveClient(adaptive, proxy)
    bootstrap_android_world(config.paths.android_world_root)
    from android_world import registry, suite_utils
    from android_world.agents import m3a
    from android_world.env import env_launcher

    class SKVMM3A(m3a.M3A):
        def step(self, goal):
            return compact_m3a_step_result(super().step(goal))

    environment = env_launcher.load_and_setup_env(
        console_port=config.android_world.console_port,
        emulator_setup=config.android_world.emulator_setup,
        adb_path=config.android_world.adb_path,
    )
    try:
        suite = suite_utils.create_suite(
            registry.TaskRegistry().get_registry(family=args.family),
            n_task_combinations=args.combinations, seed=args.seed,
            tasks=list(tasks) if tasks else None, env=environment,
        )
        suite.suite_family = args.family
        agent = SKVMM3A(environment, client, name=f"skvm:{profile.model_id}",
                       wait_after_action_seconds=config.android_world.wait_after_action_seconds)
        episodes = _run_evaluation_suite(suite_utils, suite, agent, environment, config,
                                        progress, max_steps=args.max_steps)
        ensure_valid_evaluation_episodes(
            episodes, expected_episodes=sum(len(items) for items in suite.values()))
        artifacts = write_evaluation_report(
            target, episodes, episodes_to_traces(episodes), store_trajectories=args.store_trajectories,
            metadata={"evaluation_mode": "skvm_fixed_adapter_aot", "model_id": profile.model_id,
                      "evaluation_checkpoint": profile.metadata.get("evaluation_checkpoint"),
                      "skill_database": str(store.database), "available_skills": len(skills),
                      "tasks": tasks or "all", "n_task_combinations": args.combinations,
                      "seed": args.seed, "family": args.family, "max_steps": args.max_steps,
                      "dynamic_model_routing": False, "skill_adaptation": True,
                      "skvm_identity": identity, "skvm_variant_policy": "aot",
                      "skvm_runtime_counter_scope": "all_saved_requests" if args.store_trajectories else "current_invocation",
                      "resume": progress.metadata()},
        )
    finally:
        active_error = sys.exc_info()[0]
        try:
            _close_environment(environment, tolerate_failure=bool(active_error))
        finally:
            try:
                # The kernel report is useful even if task execution was interrupted.
                legacy.skvm_reporting.write_skvm_evaluation(
                    run_dir=target, cache_dir=kernel.cache_dir, target_model=kernel.target_model,
                    adapter=legacy.SKVM_ADAPTER, variants=variants, manifest=manifest,
                    runtime_stats=adaptive.get_stats(),
                )
            except Exception:
                if not active_error:
                    raise
                logging.exception("SKVM report failed while preserving original evaluation error")
    return artifacts
