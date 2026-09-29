"""AndroidWorld 七维能力分类、统计与可视化。

本模块是评测和训练报告共享的唯一能力分类来源。能力分数使用该类中实际参与
评测的全部有效 episode 的 micro success rate，并转换到 ``[0, 100]``。
未被当前抽样覆盖的能力使用 ``None``，避免把“没有测试”误写成“能力为 0”。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

CAPABILITY_ORDER = (
    "control",
    "create",
    "edit",
    "delete",
    "query",
    "cross_app",
    "interactive",
)

CAPABILITY_LABELS = {
    "control": "Control",
    "create": "Create",
    "edit": "Edit",
    "delete": "Delete",
    "query": "Query",
    "cross_app": "Cross-App",
    "interactive": "Interactive",
}

# 与论文实验口径一致的七类任务。使用 frozenset 防止报告运行时意外修改分类。
TASK_CLASSES: dict[str, frozenset[str]] = {
    "control": frozenset(
        {
            "AudioRecorderRecordAudio",
            "AudioRecorderRecordAudioWithFileName",
            "CameraTakePhoto",
            "CameraTakeVideo",
            "ClockStopWatchPausedVerify",
            "ClockStopWatchRunning",
            "ClockTimerEntry",
            "OpenAppTaskEval",
            "SystemBluetoothTurnOff",
            "SystemBluetoothTurnOffVerify",
            "SystemBluetoothTurnOn",
            "SystemBluetoothTurnOnVerify",
            "SystemBrightnessMax",
            "SystemBrightnessMaxVerify",
            "SystemBrightnessMin",
            "SystemBrightnessMinVerify",
            "SystemWifiTurnOff",
            "SystemWifiTurnOffVerify",
            "SystemWifiTurnOn",
            "SystemWifiTurnOnVerify",
            "TurnOffWifiAndTurnOnBluetooth",
            "TurnOnWifiAndOpenApp",
        }
    ),
    "create": frozenset(
        {
            "ContactsAddContact",
            "ContactsNewContactDraft",
            "ExpenseAddMultiple",
            "ExpenseAddSingle",
            "MarkorCreateFolder",
            "MarkorCreateNote",
            "RecipeAddMultipleRecipes",
            "RecipeAddSingleRecipe",
            "RetroCreatePlaylist",
            "RetroSavePlaylist",
            "SimpleCalendarAddOneEvent",
            "SimpleCalendarAddOneEventInTwoWeeks",
            "SimpleCalendarAddOneEventRelativeDay",
            "SimpleCalendarAddOneEventTomorrow",
            "SimpleCalendarAddRepeatingEvent",
            "SimpleDrawProCreateDrawing",
            "SimpleSmsReply",
            "SimpleSmsReplyMostRecent",
            "SimpleSmsResend",
            "SimpleSmsSend",
            "VlcCreatePlaylist",
            "VlcCreateTwoPlaylists",
        }
    ),
    "edit": frozenset(
        {
            "FilesMoveFile",
            "MarkorAddNoteHeader",
            "MarkorChangeNoteContent",
            "MarkorEditNote",
            "MarkorMergeNotes",
            "MarkorMoveNote",
            "SystemCopyToClipboard",
        }
    ),
    "delete": frozenset(
        {
            "ExpenseDeleteDuplicates",
            "ExpenseDeleteDuplicates2",
            "ExpenseDeleteMultiple",
            "ExpenseDeleteMultiple2",
            "ExpenseDeleteSingle",
            "FilesDeleteFile",
            "MarkorDeleteAllNotes",
            "MarkorDeleteNewestNote",
            "MarkorDeleteNote",
            "RecipeDeleteDuplicateRecipes",
            "RecipeDeleteDuplicateRecipes2",
            "RecipeDeleteDuplicateRecipes3",
            "RecipeDeleteMultipleRecipes",
            "RecipeDeleteMultipleRecipesWithConstraint",
            "RecipeDeleteMultipleRecipesWithNoise",
            "RecipeDeleteSingleRecipe",
            "RecipeDeleteSingleWithRecipeWithNoise",
            "SimpleCalendarDeleteEvents",
            "SimpleCalendarDeleteEventsOnRelativeDay",
            "SimpleCalendarDeleteOneEvent",
        }
    ),
    "query": frozenset(
        {
            "NotesIsTodo",
            "NotesMeetingAttendeeCount",
            "NotesRecipeIngredientCount",
            "NotesTodoItemCount",
            "RetroPlayingQueue",
            "RetroPlaylistDuration",
            "SimpleCalendarAnyEventsOnDate",
            "SimpleCalendarEventOnDateAtTime",
            "SimpleCalendarEventsInNextWeek",
            "SimpleCalendarEventsInTimeRange",
            "SimpleCalendarEventsOnDate",
            "SimpleCalendarFirstEventAfterStartTime",
            "SimpleCalendarLocationOfEvent",
            "SimpleCalendarNextEvent",
            "SimpleCalendarNextMeetingWithPerson",
            "SportsTrackerActivitiesCountForWeek",
            "SportsTrackerActivitiesOnDate",
            "SportsTrackerActivityDuration",
            "SportsTrackerLongestDistanceActivity",
            "SportsTrackerTotalDistanceForCategoryOverInterval",
            "SportsTrackerTotalDurationForCategoryThisWeek",
            "TasksCompletedTasksForDate",
            "TasksDueNextWeek",
            "TasksDueOnDate",
            "TasksHighPriorityTasks",
            "TasksHighPriorityTasksDueOnDate",
            "TasksIncompleteTasksOnDate",
        }
    ),
    "cross_app": frozenset(
        {
            "ExpenseAddMultipleFromGallery",
            "ExpenseAddMultipleFromMarkor",
            "MarkorCreateNoteAndSms",
            "MarkorCreateNoteFromClipboard",
            "MarkorTranscribeReceipt",
            "MarkorTranscribeVideo",
            "RecipeAddMultipleRecipesFromImage",
            "RecipeAddMultipleRecipesFromMarkor",
            "RecipeAddMultipleRecipesFromMarkor2",
            "SaveCopyOfReceiptTaskEval",
            "SimpleSmsSendClipboardContent",
            "SimpleSmsSendReceivedAddress",
        }
    ),
    "interactive": frozenset(
        {
            "BrowserDraw",
            "BrowserMaze",
            "BrowserMultiply",
            "OsmAndFavorite",
            "OsmAndMarker",
            "OsmAndTrack",
        }
    ),
}


def build_task_to_capability() -> dict[str, str]:
    """建立 task -> capability 索引，并在分类重复时立即报错。"""

    result: dict[str, str] = {}
    for capability in CAPABILITY_ORDER:
        for task in TASK_CLASSES[capability]:
            previous = result.get(task)
            if previous is not None:
                raise ValueError(
                    f"AndroidWorld task {task!r} 同时属于 {previous!r} 和 "
                    f"{capability!r}"
                )
            result[task] = capability
    return result


TASK_TO_CAPABILITY = build_task_to_capability()


def summarize_capabilities(
    per_task: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """由每任务统计汇总七维能力分数。

    ``per_task`` 的行需包含 ``successes`` 和 ``episodes``。返回值中的
    ``score``、``task_coverage_percent`` 均采用 ``[0, 100]`` 标度。
    """

    buckets: dict[str, dict[str, Any]] = {
        capability: {
            "label": CAPABILITY_LABELS[capability],
            "successes": 0,
            "episodes": 0,
            "evaluated_tasks": 0,
            "defined_tasks": len(TASK_CLASSES[capability]),
        }
        for capability in CAPABILITY_ORDER
    }
    unclassified: list[str] = []
    for task, task_row in per_task.items():
        capability = TASK_TO_CAPABILITY.get(str(task))
        if capability is None:
            unclassified.append(str(task))
            continue
        episodes = max(0, int(task_row.get("episodes", 0)))
        successes = min(
            episodes,
            max(0, int(task_row.get("successes", 0))),
        )
        bucket = buckets[capability]
        bucket["successes"] += successes
        bucket["episodes"] += episodes
        if episodes:
            bucket["evaluated_tasks"] += 1

    for capability in CAPABILITY_ORDER:
        bucket = buckets[capability]
        episodes = int(bucket["episodes"])
        evaluated_tasks = int(bucket["evaluated_tasks"])
        defined_tasks = int(bucket["defined_tasks"])
        bucket["success_rate"] = (
            float(bucket["successes"]) / episodes if episodes else None
        )
        bucket["score"] = (
            float(bucket["successes"]) / episodes * 100.0 if episodes else None
        )
        bucket["task_coverage_percent"] = (
            evaluated_tasks / defined_tasks * 100.0 if defined_tasks else 0.0
        )
    return buckets, sorted(unclassified)


def render_capability_plot(
    capability_distribution: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    title: str = "Capability Distribution of Model",
    stem: str = "capability_distribution",
) -> dict[str, str]:
    """生成论文风格的单模型极坐标柱状图，返回相对文件名。

    绘图依赖采用延迟导入，因此没有安装 matplotlib/numpy 时不会让评测本身
    失败；调用者会在 JSON/Markdown 中记录绘图错误。
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 仅精简运行环境触发
        raise RuntimeError(
            "生成能力图需要 matplotlib 和 numpy，请安装 AndroidWorld/ms-swift 依赖"
        ) from exc

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    categories = [CAPABILITY_LABELS[key] for key in CAPABILITY_ORDER]
    raw_scores = [
        capability_distribution.get(key, {}).get("score")
        for key in CAPABILITY_ORDER
    ]
    values = np.array(
        [float(score) if score is not None and math.isfinite(float(score)) else 0.0 for score in raw_scores]
    )
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    width = 2 * np.pi / len(categories) * 0.88
    palette = plt.get_cmap("gist_earth")(
        np.linspace(0.08, 0.92, len(categories))
    )

    with plt.rc_context(
        {
            "figure.dpi": 200,
            "font.family": "sans-serif",
            "font.size": 16,
        }
    ):
        fig, ax = plt.subplots(figsize=(10.5, 10.5), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for index in range(len(categories)):
            ax.bar(
                angles[index],
                100,
                width=width,
                bottom=0,
                color=palette[index],
                alpha=0.12,
                edgecolor="white",
                linewidth=1.5,
                align="center",
                zorder=1,
            )
        ax.bar(
            angles,
            values,
            width=width,
            bottom=0,
            color=palette,
            alpha=0.88,
            edgecolor="white",
            linewidth=2,
            align="center",
            zorder=3,
        )
        ax.set_ylim(0, 112)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([])
        ax.yaxis.grid(True, linestyle="--", linewidth=1.0, alpha=0.45, zorder=0)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.20, zorder=0)
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        for angle, label in zip(angles, categories):
            ax.text(angle, 110, label, fontsize=15, ha="center", va="center")
        for angle, score, value in zip(angles, raw_scores, values):
            radius = max(min(float(value) + 12, 85), 20)
            value_text = "N/A" if score is None else f"{float(score):.0f}"
            ax.text(
                angle,
                radius,
                value_text,
                fontsize=17,
                ha="center",
                va="center",
                color="black",
                zorder=5,
            )
        center_circle = plt.Circle(
            (0, 0),
            0.055,
            transform=ax.transAxes,
            color="white",
            ec="0.7",
            linewidth=1.2,
            zorder=10,
        )
        ax.add_artist(center_circle)
        ax.spines["polar"].set_visible(False)
        fig.suptitle(title, fontsize=22, fontweight="normal", y=0.98)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.91, bottom=0.04)
        png_path = target / f"{stem}.png"
        pdf_path = target / f"{stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    LOGGER.info("已生成七维能力图: %s, %s", png_path, pdf_path)
    return {"png": png_path.name, "pdf": pdf_path.name}


def format_capability_score(value: Any) -> str:
    """将可空能力分数格式化为报告中的 ``xx.xx`` 或 ``N/A``。"""

    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.2f}" if math.isfinite(numeric) else "N/A"


def capability_table_lines(
    capability_distribution: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """生成可嵌入任意 Markdown 报告的七维能力表。"""

    lines = [
        "| 能力 | 成功/有效 episode | 已测/定义任务 | 任务覆盖率 | 能力分数 [0,100] |",
        "|---|---:|---:|---:|---:|",
    ]
    for capability in CAPABILITY_ORDER:
        row = capability_distribution.get(capability, {})
        lines.append(
            f"| {CAPABILITY_LABELS[capability]} | "
            f"{int(row.get('successes', 0))}/{int(row.get('episodes', 0))} | "
            f"{int(row.get('evaluated_tasks', 0))}/"
            f"{int(row.get('defined_tasks', len(TASK_CLASSES[capability])))} | "
            f"{float(row.get('task_coverage_percent', 0.0)):.2f}% | "
            f"{format_capability_score(row.get('score'))} |"
        )
    return lines

