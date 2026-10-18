"""Official ICLR 2025 AndroidWorld agents with a fixed Qwen adapter backbone."""

from ..pmtskill_v2.inference.vlm import OpenAICompatibleVLClient
from .runner import run_gui_baseline


def run_androidworld(config, profile, args, tasks):
    kind = args.androidworld_agent
    return run_gui_baseline(
        config, [profile], args, tasks, protocol=f"androidworld-official-{kind}-qwen-v1",
        agent_kind=kind, prepare_client=lambda target: (OpenAICompatibleVLClient(profile), {}),
        metadata={"implementation": "official-agent-qwen-backbone",
                  "official_source": "https://github.com/google-research/android_world",
                  "skills_used": False, "dynamic_model_routing": False,
                  "same_agent_as_standalone": kind == "m3a",
                  "backbone_changed_from_paper": True},
    )
