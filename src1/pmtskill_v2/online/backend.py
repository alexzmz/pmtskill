"""在线轨迹接收与技能库自主维护 backend。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..core.config import ProjectConfig
from ..core.io import write_json_atomic
from ..core.io import load_primitives
from ..core.models import ExecutionTrace
from ..inference.vlm import OpenAICompatibleVLClient
from ..skills.compiler import (
    LLMRawSkillCompiler,
    RawSkillCompileSummary,
    compile_imported_raw_skills,
)
from ..skills.importer import ImportSummary, import_skvm_skills
from ..skills.maintenance import MaintenanceReport, SkillMaintainer
from ..skills.store import SkillStore


@dataclass(slots=True)
class BackendCycleResult:
    import_summary: ImportSummary
    compile_summary: RawSkillCompileSummary | None
    model_profile_updates: dict[str, int]
    maintenance_report: MaintenanceReport
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_import": self.import_summary.to_dict(),
            "raw_skill_compile": (
                self.compile_summary.to_dict() if self.compile_summary else None
            ),
            "model_profile_updates": self.model_profile_updates,
            "maintenance": self.maintenance_report.to_dict(),
            "report_path": str(self.report_path),
        }


class SkillOptimizationBackend:
    """可由 cron/服务循环调用的自主技能维护门面。"""

    def __init__(self, config: ProjectConfig, store: SkillStore):
        self.config = config
        self.store = store
        self.maintainer = SkillMaintainer(store, config.maintenance)

    def ingest_trace(self, trace: ExecutionTrace) -> None:
        """接收设备端 ``problem-topology-result-metrics`` 轨迹。"""

        self.store.append_trace(trace)

    def ingest_trace_json(self, path: str | Path) -> int:
        """导入单条 JSON 或 JSONL 轨迹，方便设备离线上传。"""

        input_path = Path(path)
        count = 0
        with input_path.open("r", encoding="utf-8") as handle:
            text = handle.read().strip()
        values = [json.loads(line) for line in text.splitlines()] if "\n" in text else [json.loads(text)]
        for value in values:
            self.ingest_trace(ExecutionTrace.from_dict(value))
            count += 1
        return count

    def _update_model_profiles(
        self, traces: list[ExecutionTrace]
    ) -> dict[str, int]:
        """用新轨迹自动更新每个模型/LoRA 的原语能力画像。

        采用强度为 5 的配置先验，少量失败不会让画像剧烈跳变；观测累计值存入
        metadata，后续维护周期可以增量更新。
        """

        profiles = {
            profile.model_id: profile
            for profile in self.store.list_model_profiles(enabled_only=False)
        }
        for configured in self.config.models:
            profiles.setdefault(configured.model_id, configured)
        batch: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        for trace in traces:
            for event in trace.events:
                for primitive in event.primitive_ids:
                    batch[event.model_id][primitive][0] += int(event.success)
                    batch[event.model_id][primitive][1] += 1

        updates: dict[str, int] = {}
        for model_id, primitives in batch.items():
            profile = profiles.get(model_id)
            if profile is None:
                continue
            observations = profile.metadata.setdefault("capability_observations", {})
            priors = profile.metadata.setdefault("capability_priors", {})
            for primitive, (successes, trials) in primitives.items():
                prior = float(priors.setdefault(primitive, profile.capability(primitive)))
                old = observations.get(primitive, {"successes": 0, "trials": 0})
                total_successes = int(old.get("successes", 0)) + successes
                total_trials = int(old.get("trials", 0)) + trials
                observations[primitive] = {
                    "successes": total_successes,
                    "trials": total_trials,
                }
                profile.capabilities[primitive] = (
                    prior * 5.0 + total_successes
                ) / (5.0 + total_trials)
            self.store.upsert_model_profile(profile)
            updates[model_id] = len(primitives)
        return updates

    def run_cycle(self) -> BackendCycleResult:
        """同步 SKVM 上游、挖掘候选、晋升/回滚并输出审计报告。"""

        import_summary = import_skvm_skills(
            self.config.paths.skvm_skills_root, self.store
        )
        compile_summary = None
        compiler_model_id = self.config.maintenance.raw_skill_compiler_model_id
        if compiler_model_id:
            client = OpenAICompatibleVLClient(self.config.model(compiler_model_id))
            compiler = LLMRawSkillCompiler(client, load_primitives())
            compile_summary = compile_imported_raw_skills(
                self.store,
                compiler,
                limit=self.config.maintenance.raw_skill_compile_batch_size,
            )
        pending_traces = self.store.list_traces(processed=False)
        profile_updates = self._update_model_profiles(pending_traces)
        maintenance = self.maintainer.run_cycle()
        report_path = self.config.paths.state_dir / "reports" / "maintenance_latest.json"
        result = BackendCycleResult(
            import_summary,
            compile_summary,
            profile_updates,
            maintenance,
            report_path,
        )
        write_json_atomic(report_path, result.to_dict())
        return result
