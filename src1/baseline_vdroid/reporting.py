"""Convert native V-Droid episodes to the PMT-Skill comparison schema."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..pmtskill_v2.core.io import write_json_atomic
from ..pmtskill_v2.core.models import ExecutionTrace, TraceEvent
from ..pmtskill_v2.evaluation.reporter import (
    EvaluationArtifacts,
    episode_step_values,
    finite_float_value,
    normalize_episode_data,
    successful_episode_value,
    write_evaluation_report,
)
from ..pmtskill_v2.offline.dataset import infer_action_primitives


def _step_value(values: Sequence[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def episodes_to_vdroid_traces(
    episodes: Sequence[dict[str, Any]], *, model_id: str
) -> list[ExecutionTrace]:
    traces: list[ExecutionTrace] = []
    for episode in episodes:
        successful = successful_episode_value(episode.get("is_successful", False))
        data = normalize_episode_data(episode.get("episode_data", {}))
        actions = episode_step_values(data.get("action_output"))
        summaries = episode_step_values(data.get("summary"))
        verifier_rows = episode_step_values(data.get("_vdroid"))
        valid_actions = [item for item in actions if item]
        per_step_latency = (
            finite_float_value(episode.get("run_time")) * 1000 / len(valid_actions)
            if valid_actions
            else 0.0
        )
        events: list[TraceEvent] = []
        for index, action in enumerate(actions):
            if not action:
                continue
            raw_metadata = _step_value(verifier_rows, index)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            events.append(
                TraceEvent(
                    index=len(events),
                    model_id=model_id,
                    skill_id=None,
                    primitive_ids=tuple(infer_action_primitives(str(action))),
                    success=successful,
                    latency_ms=per_step_latency,
                    action=str(action),
                    observation=(
                        str(_step_value(summaries, index))
                        if _step_value(summaries, index) is not None
                        else None
                    ),
                    metadata={
                        "baseline": "vdroid",
                        "routing_mode": "fixed_verifier",
                        "skill_adaptation": False,
                        **metadata,
                    },
                )
            )
        traces.append(
            ExecutionTrace.new(
                goal=str(episode.get("goal", "")),
                task_name=str(episode.get("task_template", "unknown")),
                successful=successful,
                events=events,
                reward=float(successful),
                duration_ms=finite_float_value(episode.get("run_time")) * 1000,
                metadata={
                    "source": "vdroid_native_android_world",
                    "baseline": "vdroid",
                    "fixed_model_id": model_id,
                    "model_switching": False,
                    "skill_adaptation": False,
                    "instance_id": episode.get("instance_id"),
                    "agent_name": episode.get("agent_name", "VDroid"),
                },
            )
        )
    return traces


def summarize_vdroid_runtime(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts: list[int] = []
    selected_scores: list[float] = []
    completed_actions = 0
    decisions = 0
    for episode in episodes:
        data = normalize_episode_data(episode.get("episode_data", {}))
        for raw in episode_step_values(data.get("_vdroid")):
            if not isinstance(raw, Mapping):
                continue
            count = int(finite_float_value(raw.get("candidate_count", 0)))
            if count > 0:
                candidate_counts.append(count)
                decisions += 1
            score = raw.get("selected_score")
            if score is not None:
                selected_scores.append(finite_float_value(score))
            completed_actions += int(bool(raw.get("action_completed", False)))
    return {
        "decision_count": decisions,
        "candidate_actions_verified": sum(candidate_counts),
        "average_candidates_per_decision": (
            statistics.fmean(candidate_counts) if candidate_counts else 0.0
        ),
        "average_selected_verifier_score": (
            statistics.fmean(selected_scores) if selected_scores else None
        ),
        "action_completion_calls": completed_actions,
        "model_switches": 0,
        "skill_calls": 0,
    }


def write_vdroid_report(
    output_dir: str | Path,
    episodes: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> EvaluationArtifacts:
    model_id = str(metadata["model_id"])
    traces = episodes_to_vdroid_traces(episodes, model_id=model_id)
    artifacts = write_evaluation_report(
        output_dir,
        episodes,
        traces,
        metadata=metadata,
    )
    vdroid_metrics = summarize_vdroid_runtime(episodes)
    artifacts.summary["vdroid"] = vdroid_metrics
    write_json_atomic(artifacts.summary_json, artifacts.summary)

    markdown = artifacts.report_markdown.read_text(encoding="utf-8")
    markdown = markdown.replace(
        "# PMT-Skill AndroidWorld 评测报告",
        "# V-Droid AndroidWorld Baseline 评测报告",
        1,
    )
    score = vdroid_metrics["average_selected_verifier_score"]
    score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
    markdown += (
        "\n## V-Droid 固定基线协议\n\n"
        f"- 固定 verifier：`{model_id}`\n"
        f"- 固定工作记忆/动作补全模型：`{metadata['assistant_model']}`\n"
        "- 动态模型/adapter 切换：关闭\n"
        "- PORTS/SKVM 技能检索与技能适配：关闭\n"
        "- 动作选择：枚举当前 UI 候选动作，由 verifier 批量评分后选择最高分动作\n\n"
        "| V-Droid 指标 | 数值 |\n"
        "|---|---:|\n"
        f"| verifier 决策次数 | {vdroid_metrics['decision_count']} |\n"
        f"| 已评分候选动作总数 | {vdroid_metrics['candidate_actions_verified']} |\n"
        f"| 每次决策平均候选数 | {vdroid_metrics['average_candidates_per_decision']:.2f} |\n"
        f"| 被选动作平均 verifier 分数 | {score_text} |\n"
        f"| 动作补全次数 | {vdroid_metrics['action_completion_calls']} |\n"
    )
    artifacts.report_markdown.write_text(markdown, encoding="utf-8")
    return artifacts

