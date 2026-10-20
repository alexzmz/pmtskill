"""Reuse actual PORTS/baseline policies, changing only the benchmark boundary."""

from __future__ import annotations

from pathlib import Path

from ..pmtskill_v2.evaluation.resume import fingerprint
from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient


class AppContractClient:
    """All arms see the same installed-app contract; no evaluator/answer access."""
    def __init__(self, base, apps, *, fallback_model_id=None):
        self.base, self.apps = base, apps
        self.fallback_model_id = fallback_model_id
        self.calls, self.records = 0, []

    def predict_mm(self, prompt, images, *args, **kwargs):
        prompt += ("\nEnvironment: AndroidLab, not AndroidWorld. The installed benchmark apps are: "
                   + ", ".join(self.apps) + ". Use these exact names for open_app. "
                   "Historical skills are guidance only; re-ground every action on the current screen. "
                   "Follow the output format requested by the CURRENT prompt; "
                   "a summary request requires a summary, not a new action.\n")
        output, safe, raw = self.base.predict_mm(prompt, images, *args, **kwargs)
        if raw is None:
            raise RuntimeError("AndroidLab agent model call failed; this is not a task SR=0")
        self.calls += 1
        route = raw.get("_pmtskill", {}) if isinstance(raw, dict) else {}
        self.records.append({"model_id": route.get("model_id") or self.fallback_model_id or getattr(self.base, "model_id", None),
                             "skill_id": route.get("skill_id"), "is_summary": "summerize the latest step" in prompt.lower()})
        return output, safe, raw

    def predict(self, prompt, *args, **kwargs):
        return self.predict_mm(prompt, [], *args, **kwargs)

    @property
    def model_name(self):
        return getattr(self.base, "model_name", getattr(self.base, "model_id", ""))


class AgentPolicy:
    def __init__(self, config, store, profiles, args, target, apps):
        self.config, self.store, self.profiles, self.args = config, store, profiles, args
        self.target, self.apps = Path(target), apps
        self.method = args.method
        self.polished, self.raw = [], []
        self.metadata = {"implementation": "existing-policy-androidlab-environment-bridge",
                         "skills_cross_benchmark_transfer": False}
        self.clients = []
        self.pool, self.planner, self.router = None, None, None
        self._prepare()

    def _prepare(self):
        from ..pmtskill_v2.online.executor import RoutedVLWrapper, SimpleSkillVLWrapper
        from ..pmtskill_v2.inference.model_pool import ModelPool
        from ..pmtskill_v2.evaluation.android_world import AndroidWorldSimpleSkillEvaluator, AndroidWorldOnlineEvaluator
        from ..pmtskill_v2.online.planner import PlannerPipeline, PrimitiveTopologyGenerator
        from ..pmtskill_v2.online.router import DynamicProgrammingRouter
        from ..pmtskill_v2.skills.importer import relevant_raw_skills
        profile = self.profiles[0]
        method = self.method
        self.client = None
        def factory(profile):
            client = OpenAICompatibleVLClient(profile)
            self.clients.append(client)
            return client
        if method in {"simple-skill", "skvm", "random-skvm", "ports"}:
            if self.store is None:
                raise ValueError(f"{method} requires a frozen skill database")
            self.metadata["skills_cross_benchmark_transfer"] = True
        if method == "ports":
            self.raw = relevant_raw_skills(self.store.list_skills(kind="raw"))
            self.polished = self.store.list_skills(kind="polished")
            self.pool = ModelPool(self.profiles, client_factory=factory)
            self.router = DynamicProgrammingRouter(self.config.routing, self.store)
            planner = AndroidWorldOnlineEvaluator(self.config, self.store)._planner(self.args.planner_model, self.profiles)
            self.planner = PlannerPipeline(planner, PrimitiveTopologyGenerator())
            self.client = RoutedVLWrapper(self.pool, self.polished, router=self.router,
                                           include_candidates=self.args.include_candidates,
                                           supported_apps=self.apps)
            self.metadata.update(dynamic_model_routing=True, skill_adaptation=True,
                                 routing_objective=self.config.routing.objective)
        elif method == "simple-skill":
            self.raw = AndroidWorldSimpleSkillEvaluator(self.config, self.store)._skills(self.args.include_candidates)
            if not self.raw:
                raise ValueError("Simple-skill corpus is empty")
            self.client = SimpleSkillVLWrapper(factory(profile), self.raw, model_id=profile.model_id)
        elif method in {"skvm", "random-skvm"}:
            from ..baselines.random_skvm import prepare_model, RandomSKVMClient
            self.raw = AndroidWorldSimpleSkillEvaluator(self.config, self.store)._skills(self.args.include_candidates)
            if not self.raw:
                raise ValueError("SKVM corpus is empty; not silently running standalone")
            clients, adaptives, preparation = {}, {}, {}
            for item in self.profiles:
                child = self.target / "models" / ("model-" + fingerprint(item.model_id)[:16])
                clients[item.model_id], adaptives[item.model_id], preparation[item.model_id] = prepare_model(
                    self.config, item, self.args, self.raw, child)
            self.client = (RandomSKVMClient(self.profiles, clients, adaptives, self.args.routing_seed)
                           if method == "random-skvm" else clients[profile.model_id])
            self.adaptives = adaptives
            self.metadata.update(skvm_compilation=preparation, uses_real_skvm_kernel=True,
                                 dynamic_model_routing=method == "random-skvm",
                                 original_fallback_is_reported=True)
        elif method == "echotrail-gui":
            from ..baselines.echotrail import validate_memory, HybridMemory, LocalEncoder, EchoTrailClient
            document, embedding = validate_memory(self.args, check_encoder=True)
            memory = HybridMemory(document, LocalEncoder(self.args.embedding_model_path))
            self.client = EchoTrailClient(factory(profile), memory, self.args, self.target / "memory_stats.json")
            self.metadata.update(implementation="echotrail-qwen-paper-adaptation-on-androidlab",
                                 memory_source=document["source_kind"], memory_sha256=fingerprint(document),
                                 embedding=embedding, online_memory_updates=False)
        elif method == "vdroid":
            self.metadata.update(implementation="vdroid-qwen-yes-no-adaptation-on-androidlab",
                                 original_llama_value_head=False, dynamic_model_routing=False)
        else:
            self.client = factory(profile)
            self.metadata.update(official_agent="t3a" if method == "androidworld" and self.args.androidworld_agent == "t3a" else "m3a",
                                 same_agent_as_standalone=method == "androidworld" and self.args.androidworld_agent == "m3a")
        self.contract = AppContractClient(self.client, self.apps,
                                          fallback_model_id=profile.model_id if method not in {"ports", "random-skvm"} else None) if self.client else None

    def make_agent(self, environment, episode_key):
        from android_world.agents import m3a
        self.plan = None
        if self.method == "vdroid":
            return self._vdroid(environment)
        if hasattr(self.client, "start_episode"):
            self.client.start_episode(environment.goal, episode_key)
        elif hasattr(self.client, "reset"):
            self.client.reset()
        if hasattr(self, "adaptives"):
            for adaptive in self.adaptives.values():
                adaptive.reset_skill_session()
        if self.method == "ports":
            from ..pmtskill_v2.online.planner import scope_hierarchical_skills
            self.pool.reset_counters()
            decomposition, topology = self.planner.plan(environment.goal, self.raw, self.polished,
                                                        include_candidates=self.args.include_candidates)
            self.plan = self.router.route(environment.goal, topology, self.profiles,
                                          scope_hierarchical_skills(self.polished, decomposition.task_skill_id),
                                          planner_id=decomposition.planner_id,
                                          include_candidates=self.args.include_candidates)
            self.plan.metadata["task_skill_id"] = decomposition.task_skill_id
            self.client.set_plan(self.plan)
        if self.method == "simple-skill":
            self.client.set_goal(environment.goal)
        kind = self.args.androidworld_agent if self.method == "androidworld" else "m3a"
        if kind == "t3a":
            from android_world.agents.t3a import T3A
            return T3A(environment, self.contract, name="androidlab:t3a")
        return m3a.M3A(environment, self.contract, name="androidlab:" + self.method,
                        wait_after_action_seconds=self.args.lab_wait_after_action_seconds)

    def begin_step(self, goal):
        if self.method == "ports":
            self.client.begin_step()
        elif hasattr(self.client, "begin_step"):
            self.client.begin_step(goal)

    def observe_step(self, data):
        if hasattr(self.client, "observe_step"):
            self.client.observe_step(data)

    def _vdroid(self, environment):
        import ray
        from android_world.agents import vdroid
        from ..baseline_vdroid.qwen import install_qwen_backend, validate_qwen_agent
        profile = self.profiles[0]
        install_qwen_backend(vdroid, ray, profile.to_dict(), self.args.vdroid_score_workers)
        agent = vdroid.VDroidAgent(
            environment, str(profile.metadata.get("base_model_path", "")),
            adapter_dir=str(profile.metadata.get("evaluation_checkpoint", "")),
            llm_name=profile.served_model, service_name="openai", family="android_lab",
            num_actors=1, n_iters=self.args.vdroid_search_iterations, summary_mode="llm",
            depth_limit=self.args.max_steps, explore_step_count_limit=self.args.max_steps,
            if_store_screen=False, wait_after_action_seconds=self.args.lab_wait_after_action_seconds,
            save_dir=str(self.target))
        validate_qwen_agent(agent)
        self.contract = AppContractClient(agent.llm, self.apps, fallback_model_id=profile.model_id)
        agent.llm = self.contract
        agent.transition_pause = self.args.lab_wait_after_action_seconds
        agent.goal, agent.iter_idx = environment.goal, 0
        # Do not use _check_task_finish: official benchmark truth is evaluator-only.
        return agent

    def close_agent(self, agent):
        if self.method == "vdroid":
            import ray
            for actor in getattr(agent, "actors", []):
                ray.kill(actor)

    def stats(self):
        return {"calls": self.contract.calls if self.contract and self.method != "vdroid" else None,
                "http_requests": sum(client.http_request_count for client in self.clients),
                "model_switches": self.pool.switch_count if self.pool else None,
                "metadata": self.metadata}
