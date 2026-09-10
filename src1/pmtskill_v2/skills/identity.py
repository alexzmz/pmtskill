"""技能定义与实际模型部署的稳定身份。

标定指标不能只绑定人类可读的 ``skill_id`` / ``model_id``：技能正文可能被
backend 优化，LoRA 也可能在同名 adapter 下继续训练。这里生成的短 ID 会随真正影响
行为的内容变化，但不会因为时间戳、服务端口或生命周期状态变化而变化。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..core.models import ModelProfile, SkillRecord, SkillTopology


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _topology_payload(topology: SkillTopology | None) -> list[dict[str, Any]] | None:
    """去掉随机 topology_id，只保留会影响执行语义的 DAG 内容。"""

    if topology is None:
        return None
    return [
        {
            "node_id": node.node_id,
            "primitive_id": node.primitive_id,
            "depends_on": list(node.depends_on),
            "params": node.params,
            "condition": node.condition,
        }
        for node in topology.nodes
    ]


def skill_definition_hash(skill: SkillRecord) -> str:
    """返回技能可执行定义的完整 SHA-256，不受 active/candidate 状态影响。"""

    return _digest(
        {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "kind": skill.kind,
            "level": skill.level,
            "body": skill.body,
            "source_hash": skill.source_hash,
            "topology": _topology_payload(skill.topology),
            "fallback_topology": _topology_payload(skill.fallback_topology),
        }
    )


def skill_version_id(skill: SkillRecord) -> str:
    """返回适合 SQLite 主键和报告展示的技能版本 ID。"""

    return f"{skill.skill_id}@{skill_definition_hash(skill)[:16]}"


def model_variant_hash(profile: ModelProfile) -> str:
    """标识一次实际模型/LoRA 部署，而不是临时 OpenAI 服务地址。

    ``evaluation_checkpoint`` 是当前评测部署解析出的真实 checkpoint。没有 LoRA 时，
    served model、adapter 配置和显式模型元数据共同构成稳定身份。
    """

    metadata_keys = (
        "evaluation_checkpoint",
        "adapter_input_path",
        "adapter_run_dir",
        "adapter_epoch_dir",
        "adapter_selection",
        "lora_rank",
        "base_model_path",
        "model_revision",
    )
    return _digest(
        {
            "model_id": profile.model_id,
            "served_model": profile.served_model,
            "adapter": profile.adapter,
            "deployment": {
                key: profile.metadata.get(key)
                for key in metadata_keys
                if profile.metadata.get(key) is not None
            },
        }
    )


def model_variant_id(profile: ModelProfile) -> str:
    """返回模型逻辑 ID 与部署指纹组合成的版本 ID。"""

    return f"{profile.model_id}@{model_variant_hash(profile)[:16]}"
