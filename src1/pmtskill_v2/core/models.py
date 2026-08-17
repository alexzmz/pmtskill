"""整个框架共享的强类型数据结构。

这些类型只依赖 Python 标准库，刻意不依赖 AndroidWorld、PyTorch 或
ms-swift。路由算法、训练算法和执行器因此可以独立替换和测试。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
import uuid
from collections import deque
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    """返回带时区的 UTC ISO 时间，便于跨机器合并日志。"""

    return dt.datetime.now(dt.timezone.utc).isoformat()


class SkillStatus(str, enum.Enum):
    """技能生命周期。

    imported
        从 SKVM 导入，尚未针对 AndroidWorld 验证。
    candidate
        云侧已生成候选实现，正在灰度验证。
    active
        验证通过，可进入在线路由候选集。
    deprecated
        近期退化或被新版本替代，只保留用于审计和回退。
    """

    IMPORTED = "imported"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclasses.dataclass(slots=True)
class PrimitiveSpec:
    """一个最小可训练/可执行原语的定义。"""

    primitive_id: str
    title: str
    description: str
    category: str
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrimitiveSpec":
        return cls(
            primitive_id=str(value["primitive_id"]),
            title=str(value["title"]),
            description=str(value["description"]),
            category=str(value["category"]),
            aliases=tuple(value.get("aliases", ())),
        )


@dataclasses.dataclass(slots=True)
class TopologyNode:
    """技能/原语拓扑中的一个节点。

    ``depends_on`` 表示当前节点只有在这些前驱完成后才能执行，因此整个
    :class:`SkillTopology` 是 DAG，而不仅限于线性列表。
    """

    node_id: str
    primitive_id: str
    depends_on: tuple[str, ...] = ()
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopologyNode":
        return cls(
            node_id=str(value["node_id"]),
            primitive_id=str(value["primitive_id"]),
            depends_on=tuple(value.get("depends_on", ())),
            params=dict(value.get("params", {})),
            condition=value.get("condition"),
        )


@dataclasses.dataclass(slots=True)
class SkillTopology:
    """任务、raw skill 或 polished skill 的 DAG 表示。"""

    nodes: tuple[TopologyNode, ...]
    topology_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_sequence(
        cls, primitives: Iterable[str], topology_id: str | None = None
    ) -> "SkillTopology":
        """把原语序列转换成线性 DAG，适合多数移动端轨迹。"""

        nodes: list[TopologyNode] = []
        previous: str | None = None
        for index, primitive_id in enumerate(primitives):
            node_id = f"n{index:04d}"
            nodes.append(
                TopologyNode(
                    node_id=node_id,
                    primitive_id=str(primitive_id),
                    depends_on=(previous,) if previous else (),
                )
            )
            previous = node_id
        topology = cls(tuple(nodes), topology_id or uuid.uuid4().hex)
        topology.validate()
        return topology

    def validate(self) -> None:
        """检查节点唯一性、悬空依赖和环；非法拓扑立即报错。"""

        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("拓扑中存在重复 node_id")
        known = set(ids)
        for node in self.nodes:
            missing = set(node.depends_on) - known
            if missing:
                raise ValueError(f"节点 {node.node_id} 存在悬空依赖: {sorted(missing)}")
            if node.node_id in node.depends_on:
                raise ValueError(f"节点 {node.node_id} 不能依赖自身")
        self.topological_nodes()  # Kahn 算法会在有环时抛错。

    def topological_nodes(self) -> tuple[TopologyNode, ...]:
        """以稳定顺序返回拓扑排序结果。"""

        by_id = {node.node_id: node for node in self.nodes}
        indegree = {node.node_id: len(node.depends_on) for node in self.nodes}
        followers: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for parent in node.depends_on:
                followers[parent].append(node.node_id)
        queue = deque(node.node_id for node in self.nodes if indegree[node.node_id] == 0)
        ordered: list[TopologyNode] = []
        while queue:
            node_id = queue.popleft()
            ordered.append(by_id[node_id])
            for child in followers[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("拓扑中存在环，无法执行")
        return tuple(ordered)

    def primitive_sequence(self) -> tuple[str, ...]:
        """按拓扑序给出原语序列，供当前动态规划路由器使用。"""

        return tuple(node.primitive_id for node in self.topological_nodes())

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillTopology":
        topology = cls(
            nodes=tuple(TopologyNode.from_dict(item) for item in value.get("nodes", ())),
            topology_id=str(value.get("topology_id") or uuid.uuid4().hex),
        )
        topology.validate()
        return topology


@dataclasses.dataclass(slots=True)
class SkillRecord:
    """技能库中的一个带版本技能。

    ``topology`` 表示该技能覆盖的原语；``fallback_topology`` 是 polished
    skill 失败后可展开的低层级安全路径。
    """

    skill_id: str
    name: str
    description: str
    kind: str
    status: SkillStatus
    level: int
    topology: SkillTopology
    body: str = ""
    source_path: str | None = None
    source_hash: str | None = None
    version: int = 1
    fallback_topology: SkillTopology | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at: str = dataclasses.field(default_factory=utc_now)
    updated_at: str = dataclasses.field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "status": self.status.value,
            "level": self.level,
            "topology": self.topology.to_dict(),
            "body": self.body,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "version": self.version,
            "fallback_topology": (
                self.fallback_topology.to_dict() if self.fallback_topology else None
            ),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillRecord":
        fallback = value.get("fallback_topology")
        return cls(
            skill_id=str(value["skill_id"]),
            name=str(value["name"]),
            description=str(value.get("description", "")),
            kind=str(value.get("kind", "raw")),
            status=SkillStatus(str(value.get("status", SkillStatus.IMPORTED.value))),
            level=int(value.get("level", 1)),
            topology=SkillTopology.from_dict(value.get("topology", {"nodes": []})),
            body=str(value.get("body", "")),
            source_path=value.get("source_path"),
            source_hash=value.get("source_hash"),
            version=int(value.get("version", 1)),
            fallback_topology=SkillTopology.from_dict(fallback) if fallback else None,
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclasses.dataclass(slots=True)
class ModelProfile:
    """一个 VL 模型或 LoRA adapter 的在线能力画像。"""

    model_id: str
    served_model: str
    base_url: str
    capabilities: dict[str, float]
    adapter: str | None = None
    api_key_env: str | None = None
    average_latency_ms: float = 1000.0
    switch_cost_ms: float = 50.0
    enabled: bool = True
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def capability(self, primitive_id: str, default: float = 0.50) -> float:
        """返回被截断到 (0, 1) 的成功率，避免路由时 log(0)。"""

        value = float(self.capabilities.get(primitive_id, default))
        return min(0.999999, max(0.000001, value))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelProfile":
        return cls(
            model_id=str(value["model_id"]),
            served_model=str(value.get("served_model", value["model_id"])),
            base_url=str(value.get("base_url", "http://127.0.0.1:8000/v1")),
            capabilities={
                str(k): float(v) for k, v in dict(value.get("capabilities", {})).items()
            },
            adapter=value.get("adapter"),
            api_key_env=value.get("api_key_env"),
            average_latency_ms=float(value.get("average_latency_ms", 1000.0)),
            switch_cost_ms=float(value.get("switch_cost_ms", 50.0)),
            enabled=bool(value.get("enabled", True)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclasses.dataclass(slots=True)
class RouteStep:
    """路由器选出的一个执行单元。"""

    step_id: str
    model_id: str
    skill_id: str | None
    primitive_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    expected_success: float
    expected_latency_ms: float
    score: float
    is_polished: bool = False
    fallback_skill_id: str | None = None
    score_detail: dict[str, float] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RouteStep":
        return cls(
            step_id=str(value["step_id"]),
            model_id=str(value["model_id"]),
            skill_id=value.get("skill_id"),
            primitive_ids=tuple(value.get("primitive_ids", ())),
            node_ids=tuple(value.get("node_ids", ())),
            expected_success=float(value.get("expected_success", 0.0)),
            expected_latency_ms=float(value.get("expected_latency_ms", 0.0)),
            score=float(value.get("score", 0.0)),
            is_polished=bool(value.get("is_polished", False)),
            fallback_skill_id=value.get("fallback_skill_id"),
            score_detail={
                str(k): float(v) for k, v in dict(value.get("score_detail", {})).items()
            },
        )


@dataclasses.dataclass(slots=True)
class ExecutionPlan:
    """一次任务的完整动态路由结果。"""

    goal: str
    topology: SkillTopology
    steps: tuple[RouteStep, ...]
    total_score: float
    planner_id: str = "unknown"
    created_at: str = dataclasses.field(default_factory=utc_now)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def switch_count(self) -> int:
        """计算相邻执行单元之间的模型/adapter 切换次数。"""

        return sum(
            previous.model_id != current.model_id
            for previous, current in zip(self.steps, self.steps[1:])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "topology": self.topology.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "total_score": self.total_score,
            "planner_id": self.planner_id,
            "created_at": self.created_at,
            "switch_count": self.switch_count,
            "metadata": self.metadata,
        }


@dataclasses.dataclass(slots=True)
class TraceEvent:
    """设备端上传给 backend 的单个执行事件。"""

    index: int
    model_id: str
    skill_id: str | None
    primitive_ids: tuple[str, ...]
    success: bool
    latency_ms: float
    action: str | None = None
    observation: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        return cls(
            index=int(value["index"]),
            model_id=str(value["model_id"]),
            skill_id=value.get("skill_id"),
            primitive_ids=tuple(value.get("primitive_ids", ())),
            success=bool(value.get("success", False)),
            latency_ms=float(value.get("latency_ms", 0.0)),
            action=value.get("action"),
            observation=value.get("observation"),
            error=value.get("error"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclasses.dataclass(slots=True)
class ExecutionTrace:
    """一个 AndroidWorld episode 的端到端轨迹。"""

    trace_id: str
    goal: str
    task_name: str
    successful: bool
    events: tuple[TraceEvent, ...]
    plan: ExecutionPlan | None = None
    reward: float = 0.0
    duration_ms: float = 0.0
    created_at: str = dataclasses.field(default_factory=utc_now)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def new(
        cls,
        goal: str,
        task_name: str,
        successful: bool,
        events: Iterable[TraceEvent],
        **kwargs: Any,
    ) -> "ExecutionTrace":
        return cls(uuid.uuid4().hex, goal, task_name, successful, tuple(events), **kwargs)

    def primitive_sequence(self) -> tuple[str, ...]:
        """把所有事件覆盖的原语拼成序列，供高频子序列挖掘。"""

        return tuple(p for event in self.events for p in event.primitive_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "task_name": self.task_name,
            "successful": self.successful,
            "events": [event.to_dict() for event in self.events],
            "plan": self.plan.to_dict() if self.plan else None,
            "reward": self.reward,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionTrace":
        # 读取数据库轨迹时 plan 主要用于审计，避免在这里重复实现复杂反序列化。
        return cls(
            trace_id=str(value["trace_id"]),
            goal=str(value.get("goal", "")),
            task_name=str(value.get("task_name", "unknown")),
            successful=bool(value.get("successful", False)),
            events=tuple(TraceEvent.from_dict(item) for item in value.get("events", ())),
            plan=None,
            reward=float(value.get("reward", 0.0)),
            duration_ms=float(value.get("duration_ms", 0.0)),
            created_at=str(value.get("created_at", utc_now())),
            metadata=dict(value.get("metadata", {})),
        )


def geometric_success(probabilities: Iterable[float]) -> float:
    """假设各原语独立时计算整段成功率，用作路由的可解释近似。"""

    log_probability = 0.0
    count = 0
    for probability in probabilities:
        log_probability += math.log(min(0.999999, max(0.000001, probability)))
        count += 1
    return math.exp(log_probability) if count else 1.0

