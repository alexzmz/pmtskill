from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

SOURCE_DIR = Path("/home/zmz/Workspace/gui/src1/runtime/trajectories")

OUTPUT_ROOT = Path("/home/zmz/Workspace/gui/src1/runtime/trajectories_classes")


TASK_CLASSES: dict[str, set[str]] = {
    # 系统控制、Clock、简单设备操作
    "os": {
        "OpenAppTaskEval",
        "SystemBluetoothTurnOff",
        "SystemBluetoothTurnOffVerify",
        "SystemBluetoothTurnOn",
        "SystemBluetoothTurnOnVerify",
        "SystemBrightnessMax",
        "SystemBrightnessMaxVerify",
        "SystemBrightnessMin",
        "SystemBrightnessMinVerify",
        "SystemCopyToClipboard",
        "SystemWifiTurnOff",
        "SystemWifiTurnOffVerify",
        "SystemWifiTurnOn",
        "SystemWifiTurnOnVerify",
        "TurnOffWifiAndTurnOnBluetooth",
        "TurnOnWifiAndOpenApp",
        "ClockStopWatchPausedVerify",
        "ClockStopWatchRunning",
        "ClockTimerEntry",
    },
    # Calendar 创建/删除
    "calendar_crud": {
        "SimpleCalendarAddOneEvent",
        "SimpleCalendarAddOneEventInTwoWeeks",
        "SimpleCalendarAddOneEventRelativeDay",
        "SimpleCalendarAddOneEventTomorrow",
        "SimpleCalendarAddRepeatingEvent",
        "SimpleCalendarDeleteEvents",
        "SimpleCalendarDeleteEventsOnRelativeDay",
        "SimpleCalendarDeleteOneEvent",
    },
    # Calendar / Tasks / Notes 查询推理
    "query_reasoning": {
        "SimpleCalendarEventsOnDate",
        "SimpleCalendarNextEvent",
        "SimpleCalendarEventOnDateAtTime",
        "SimpleCalendarAnyEventsOnDate",
        "SimpleCalendarNextMeetingWithPerson",
        "SimpleCalendarLocationOfEvent",
        "SimpleCalendarEventsInNextWeek",
        "SimpleCalendarFirstEventAfterStartTime",
        "SimpleCalendarEventsInTimeRange",
        "TasksDueOnDate",
        "TasksHighPriorityTasks",
        "TasksHighPriorityTasksDueOnDate",
        "TasksDueNextWeek",
        "TasksCompletedTasksForDate",
        "TasksIncompleteTasksOnDate",
        "NotesRecipeIngredientCount",
        "NotesMeetingAttendeeCount",
        "NotesIsTodo",
        "NotesTodoItemCount",
    },
    # Contacts / SMS / 通信
    "communication": {
        "ContactsAddContact",
        "ContactsNewContactDraft",
        "SimpleSmsReply",
        "SimpleSmsReplyMostRecent",
        "SimpleSmsResend",
        "SimpleSmsSend",
        "SimpleSmsSendClipboardContent",
        "SimpleSmsSendReceivedAddress",
        "MarkorCreateNoteAndSms",
    },
    # Files / Markor 文本和文件操作
    "file_note": {
        "FilesDeleteFile",
        "FilesMoveFile",
        "MarkorAddNoteHeader",
        "MarkorChangeNoteContent",
        "MarkorCreateFolder",
        "MarkorCreateNote",
        "MarkorCreateNoteFromClipboard",
        "MarkorDeleteAllNotes",
        "MarkorDeleteNewestNote",
        "MarkorDeleteNote",
        "MarkorEditNote",
        "MarkorMergeNotes",
        "MarkorMoveNote",
        "MarkorTranscribeReceipt",
        "MarkorTranscribeVideo",
    },
    # Expense / Recipe 结构化 CRUD
    "structured_crud": {
        "ExpenseAddMultiple",
        "ExpenseAddMultipleFromGallery",
        "ExpenseAddMultipleFromMarkor",
        "ExpenseAddSingle",
        "ExpenseDeleteDuplicates",
        "ExpenseDeleteDuplicates2",
        "ExpenseDeleteMultiple",
        "ExpenseDeleteMultiple2",
        "ExpenseDeleteSingle",
        "RecipeAddMultipleRecipes",
        "RecipeAddMultipleRecipesFromImage",
        "RecipeAddMultipleRecipesFromMarkor",
        "RecipeAddMultipleRecipesFromMarkor2",
        "RecipeAddSingleRecipe",
        "RecipeDeleteDuplicateRecipes",
        "RecipeDeleteDuplicateRecipes2",
        "RecipeDeleteDuplicateRecipes3",
        "RecipeDeleteMultipleRecipes",
        "RecipeDeleteMultipleRecipesWithConstraint",
        "RecipeDeleteMultipleRecipesWithNoise",
        "RecipeDeleteSingleRecipe",
        "RecipeDeleteSingleWithRecipeWithNoise",
    },
    # 媒体 / 地图 / 绘图 / Camera
    "media_geo": {
        "AudioRecorderRecordAudio",
        "AudioRecorderRecordAudioWithFileName",
        "CameraTakePhoto",
        "CameraTakeVideo",
        "RetroCreatePlaylist",
        "RetroPlayingQueue",
        "RetroPlaylistDuration",
        "RetroSavePlaylist",
        "VlcCreatePlaylist",
        "VlcCreateTwoPlaylists",
        "OsmAndFavorite",
        "OsmAndMarker",
        "OsmAndTrack",
        "SimpleDrawProCreateDrawing",
        "BrowserDraw",
        "BrowserMaze",
        "BrowserMultiply",
        "SaveCopyOfReceiptTaskEval",
    },
    # SportsTracker 查询 / 数值统计
    "sports": {
        "SportsTrackerActivitiesOnDate",
        "SportsTrackerActivitiesCountForWeek",
        "SportsTrackerActivityDuration",
        "SportsTrackerLongestDistanceActivity",
        "SportsTrackerTotalDurationForCategoryThisWeek",
        "SportsTrackerTotalDistanceForCategoryOverInterval",
    },
}


def build_task_to_class() -> dict[str, str]:
    """构造 task_name -> class_name，并检查重复分类。"""

    result: dict[str, str] = {}

    for class_name, tasks in TASK_CLASSES.items():
        for task in tasks:
            if task in result:
                raise ValueError(
                    f"Task {task!r} 同时属于 " f"{result[task]!r} 和 {class_name!r}"
                )
            result[task] = class_name

    return result


def extract_task_name(path: Path, known_tasks: set[str]) -> str | None:
    """
    从文件名解析 task 名。

    例如：
        BrowserMaze_7.pkl.gz
        -> BrowserMaze

        TasksIncompleteTasksOnDate_0.pkl.gz
        -> TasksIncompleteTasksOnDate

    用已知 task 集合匹配，避免 task 名自身含下划线时出错。
    """

    filename = path.name

    if not filename.endswith(".pkl.gz"):
        return None

    stem = filename[: -len(".pkl.gz")]

    # 优先使用最长 task 名匹配，避免名称前缀冲突。
    for task in sorted(known_tasks, key=len, reverse=True):
        prefix = task + "_"

        if stem == task or stem.startswith(prefix):
            return task

    return None


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"轨迹源目录不存在: {SOURCE_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    task_to_class = build_task_to_class()
    known_tasks = set(task_to_class)

    # 创建所有分类目录
    for class_name in TASK_CLASSES:
        destination = OUTPUT_ROOT / f"trajectory_{class_name}"
        destination.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    unmatched: list[Path] = []

    files = sorted(SOURCE_DIR.glob("*.pkl.gz"))

    print(f"发现轨迹文件: {len(files)}")
    print()

    for source in files:
        task_name = extract_task_name(source, known_tasks)

        if task_name is None:
            unmatched.append(source)
            continue

        class_name = task_to_class[task_name]

        destination_dir = OUTPUT_ROOT / f"trajectory_{class_name}"

        destination = destination_dir / source.name

        # copy2 保留时间戳等文件元信息
        shutil.copy2(source, destination)

        counts[class_name] += 1
        task_counts[task_name] += 1

    print("=" * 70)
    print("分类完成")
    print("=" * 70)

    total_copied = 0

    for class_name in TASK_CLASSES:
        count = counts[class_name]
        total_copied += count

        print(f"trajectory_{class_name:<20} " f"{count:4d} trajectories")

    print("-" * 70)
    print(f"总文件数:     {len(files)}")
    print(f"成功复制:     {total_copied}")
    print(f"未匹配:       {len(unmatched)}")

    print()
    print("=" * 70)
    print("各 task 数量")
    print("=" * 70)

    for class_name, tasks in TASK_CLASSES.items():
        print(f"\n[trajectory_{class_name}]")

        for task in sorted(tasks):
            print(f"  {task:<55} " f"{task_counts[task]:3d}")

    if unmatched:
        print()
        print("=" * 70)
        print("未匹配文件")
        print("=" * 70)

        for path in unmatched:
            print(path.name)


if __name__ == "__main__":
    main()
