"""Explicit Qwen adaptation, NOT the original trained Llama/value-head baseline.

Preserves upstream UI-action enumeration, completion, memory and search. A
fixed Qwen adapter replaces both the learned verifier and auxiliary GPT model.
Verifier scores are normalized Yes/No token probabilities, not task SR.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..pmtskill_v2.core.models import ModelProfile
from ..pmtskill_v2.evaluation.resume import model_resume_identity
from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
from .config import VDroidBaselineConfig, VDroidDeviceConfig, VDroidPathConfig
from .config import VDroidModelConfig, VDroidAssistantConfig, VDroidEvaluationConfig
from .evaluator import VDroidBaselineEvaluator, VDroidRunOptions


def yes_no_score(raw):
    """Require real likelihoods for BOTH labels; never invent missing scores."""
    try:
        first = raw["choices"][0]["logprobs"]["content"][0]
        rows = [*first.get("top_logprobs", []), first]
        labels = {"yes": {}, "no": {}}
        for row in rows:
            token = str(row.get("token", ""))
            label = token.strip().lower()
            if label in labels:
                value = float(row["logprob"])
                if math.isfinite(value):
                    labels[label][token] = value
        if not all(labels.values()):
            raise ValueError("Yes 和 No 未同时出现在 top_logprobs 中")
        offset = max(value for group in labels.values() for value in group.values())
        yes, no = (sum(math.exp(value - offset) for value in labels[label].values())
                   for label in ("yes", "no"))
        return yes / (yes + no)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Qwen V-Droid verifier 需要 vLLM chat logprobs/top_logprobs=20，"
            "且 Yes/No 都有有效概率；无法评分时停止，不伪造零分。"
        ) from exc


class QwenVerifier:
    def __init__(self, profile, workers=4):
        self.client = OpenAICompatibleVLClient(ModelProfile.from_dict(profile))
        self.workers = workers

    def _score(self, prompt):
        payload = {"model": self.client.profile.served_model, "temperature": 0,
                   "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
                   "structured_outputs_regex": "Yes|No",
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [
                       {"role": "system", "content": "Judge whether the proposed Android action helps the user goal. Reply with exactly Yes or No, without reasoning."},
                       {"role": "user", "content": prompt},
                   ]}
        request = urllib.request.Request(
            self.client.endpoint, data=json.dumps(payload).encode("utf-8"),
            headers=self.client._headers(), method="POST")
        with urllib.request.urlopen(request, timeout=self.client.timeout_seconds) as response:
            score = yes_no_score(json.loads(response.read().decode("utf-8")))
        return score, None, [score]

    def predict_scores_batch(self, prompts):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._score, prompts))

    def predict(self, prompts):
        return [self.client.predict(prompt) for prompt in prompts]

    def warm_up(self):
        # Validate the actual deployed adapter and scoring API before any task.
        self._score("Goal: open Settings. Proposed action: open the Settings app. Helpful?")
        return True

    def ping(self):
        return True


class StrictFixedClient(OpenAICompatibleVLClient):
    """V-Droid does not check raw=None like M3A; propagate API failures."""

    def predict_mm(self, prompt, images):
        result = self.generate(prompt, images, max_tokens=2048)
        return result.text, True, result.raw

    def predict(self, prompt):
        return self.predict_mm(prompt, [])


def install_qwen_backend(vdroid, ray, profile, workers=4):
    """Patch only the isolated worker; no edits to upstream V-Droid sources."""
    if not ray.is_initialized():
        root = str(Path(__file__).resolve().parents[2])
        python_path = os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH", ""))))
        ray.init(num_gpus=0, runtime_env={"env_vars": {"PYTHONPATH": python_path}})

    class Actor(QwenVerifier):
        def __init__(self, *args, **kwargs):
            super().__init__(profile, workers)

    remote_actor = ray.remote(num_gpus=0)(Actor)

    class ActorFactory:
        @staticmethod
        def options(**kwargs):
            # GPU is already owned by the parent ms-swift/vLLM service.
            return remote_actor.options(num_gpus=0)

    vdroid.ModelActor = ActorFactory
    vdroid.infer.Gpt4Wrapper = lambda *args, **kwargs: StrictFixedClient(
        ModelProfile.from_dict(profile))


class QwenVDroidEvaluator(VDroidBaselineEvaluator):
    def __init__(self, config, options, profile, workers):
        super().__init__(config, options)
        self.profile, self.workers = profile, workers

    def request(self, target):
        request = super().request(target)
        # Exclude the ephemeral service endpoint from the immutable resume ID.
        request["protocol"] = "vdroid-qwen-adapted-yes-no-v1"
        request["qwen_adapter"] = {
            "profile": self.profile.to_dict(), "identity": model_resume_identity(self.profile),
            "score_workers": self.workers, "scoring": "normalized_yes_no_logprobs",
        }
        return request


def evaluator_for_adapter(config, profile, args, tasks):
    candidates = ([Path(args.vdroid_root).expanduser().resolve()] if args.vdroid_root else
                  [config.paths.repo_root / "libs" / name for name in ("V-Droid", "v-droid")])
    source = next((item for item in candidates
                   if (item / "android_world" / "agents" / "vdroid.py").is_file()), candidates[0])
    target = Path(args.output_dir).expanduser().resolve()
    device = config.android_world
    baseline = VDroidBaselineConfig(
        config.config_path,
        VDroidPathConfig(config.paths.repo_root, source, target, config.paths.log_dir),
        VDroidDeviceConfig(adb_path=device.adb_path, console_port=device.console_port,
                           device_name=f"emulator-{device.console_port}",
                           emulator_setup=device.emulator_setup,
                           wait_after_action_seconds=device.wait_after_action_seconds),
        VDroidModelConfig(), VDroidAssistantConfig(), VDroidEvaluationConfig(),
    )
    baseline.validate()
    options = VDroidRunOptions(
        tasks=tuple(tasks or ()), combinations=args.combinations, seed=args.seed,
        family=args.family, max_steps=args.max_steps, output_dir=target,
        model_id=profile.model_id, base_model=str(profile.metadata["base_model_path"]),
        verifier_adapter=str(profile.metadata["evaluation_checkpoint"]),
        assistant_model=profile.served_model, assistant_base_url=profile.base_url,
        summary_mode="llm", num_actors=1, search_iterations=args.vdroid_search_iterations,
        cuda_visible_devices=args.cuda_visible_devices,
        emulator_setup=device.emulator_setup, resume=bool(args.resume),
        store_trajectories=args.store_trajectories,
    )
    return QwenVDroidEvaluator(baseline, options, profile, args.vdroid_score_workers)
