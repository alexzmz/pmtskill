from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

SOURCE_DIR = Path("/home/zmz/Workspace/gui/src1/runtime/trajectories")
OUTPUT_ROOT = Path("/home/zmz/Workspace/gui/src1/runtime/trajectories_capa_classes")

TASK_CLASSES: dict[str, set[str]] = {
    "control": {
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
    },

    "create": {
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
    },

    "edit": {
        "FilesMoveFile",
        "MarkorAddNoteHeader",
        "MarkorChangeNoteContent",
        "MarkorEditNote",
        "MarkorMergeNotes",
        "MarkorMoveNote",
        "SystemCopyToClipboard",
    },

    "delete": {
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
    },

    "query": {
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
    },

    "cross_app": {
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
    },

    "interactive": {
        "BrowserDraw",
        "BrowserMaze",
        "BrowserMultiply",
        "OsmAndFavorite",
        "OsmAndMarker",
        "OsmAndTrack",
    },
}


def build_task_to_class() -> dict[str, str]:
    result = {}
    for class_name, tasks in TASK_CLASSES.items():
        for task in tasks:
            if task in result:
                raise ValueError(
                    f"{task!r} 同时属于 {result[task]!r} 和 {class_name!r}"
                )
            result[task] = class_name

    if len(result) != 116:
        raise ValueError(f"分类任务数应为 116，当前为 {len(result)}")

    return result


def extract_task_name(path: Path, known_tasks: set[str]) -> str | None:
    if not path.name.endswith(".pkl.gz"):
        return None

    stem = path.name[:-len(".pkl.gz")]

    for task in sorted(known_tasks, key=len, reverse=True):
        if stem == task or stem.startswith(task + "_"):
            return task

    return None


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"轨迹源目录不存在: {SOURCE_DIR}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    task_to_class = build_task_to_class()
    known_tasks = set(task_to_class)

    for class_name in TASK_CLASSES:
        (OUTPUT_ROOT / f"trajectory_{class_name}").mkdir(parents=True, exist_ok=True)

    counts = Counter()
    task_counts = Counter()
    unmatched = []

    files = sorted(SOURCE_DIR.glob("*.pkl.gz"))

    print(f"发现轨迹文件: {len(files)}")
    print(f"能力类别数: {len(TASK_CLASSES)}")
    print(f"已定义任务数: {len(task_to_class)}")
    print()

    for source in files:
        task_name = extract_task_name(source, known_tasks)

        if task_name is None:
            unmatched.append(source)
            continue

        class_name = task_to_class[task_name]
        destination = OUTPUT_ROOT / f"trajectory_{class_name}" / source.name

        shutil.copy2(source, destination)

        counts[class_name] += 1
        task_counts[task_name] += 1

    print("=" * 70)
    print("能力分类完成")
    print("=" * 70)

    total_copied = 0

    for class_name in TASK_CLASSES:
        count = counts[class_name]
        total_copied += count
        print(
            f"trajectory_{class_name:<15} "
            f"{len(TASK_CLASSES[class_name]):3d} tasks   "
            f"{count:5d} trajectories"
        )

    print("-" * 70)
    print(f"总轨迹文件:   {len(files)}")
    print(f"成功复制:     {total_copied}")
    print(f"未匹配:       {len(unmatched)}")

    print()
    print("=" * 70)
    print("各能力类别中的 task")
    print("=" * 70)

    for class_name, tasks in TASK_CLASSES.items():
        print(f"\n[trajectory_{class_name}]")

        for task in sorted(tasks):
            print(f"  {task:<55} {task_counts[task]:4d}")

    if unmatched:
        print()
        print("=" * 70)
        print("未匹配文件")
        print("=" * 70)

        for path in unmatched:
            print(path.name)


if __name__ == "__main__":
    main()