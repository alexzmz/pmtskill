from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

OPTIMIZATION_ROOT = Path("/home/zmz/Workspace/gui/src1/runtime/optimization_trajectories")
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


def get_next_index(destination_dir: Path, task_name: str) -> int:
    max_index = -1
    pattern = re.compile(rf"^{re.escape(task_name)}_(\d+)\.pkl\.gz$")

    for path in destination_dir.glob(f"{task_name}_*.pkl.gz"):
        match = pattern.match(path.name)

        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def get_destination(source: Path, destination_dir: Path, task_name: str) -> Path:
    destination = destination_dir / source.name

    if not destination.exists():
        return destination

    next_index = get_next_index(destination_dir, task_name)

    return destination_dir / f"{task_name}_{next_index}.pkl.gz"


def main() -> None:
    if not OPTIMIZATION_ROOT.is_dir():
        raise FileNotFoundError(
            f"optimization trajectories 目录不存在: {OPTIMIZATION_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    task_to_class = build_task_to_class()
    known_tasks = set(task_to_class)

    for class_name in TASK_CLASSES:
        (OUTPUT_ROOT / f"trajectory_{class_name}").mkdir(
            parents=True,
            exist_ok=True
        )

    source_dirs = sorted(
        path
        for path in OPTIMIZATION_ROOT.glob("*/checkpoints/tasks")
        if path.is_dir()
    )

    print(f"发现 optimization trajectory 目录: {len(source_dirs)}")

    for source_dir in source_dirs:
        print(f"  {source_dir}")

    print()

    counts = Counter()
    task_counts = Counter()
    renamed = 0
    unmatched = []
    total_files = 0

    for source_dir in source_dirs:
        files = sorted(source_dir.glob("*.pkl.gz"))
        total_files += len(files)

        print(f"[{source_dir.parent.parent.name}] {len(files)} trajectories")

        for source in files:
            task_name = extract_task_name(source, known_tasks)

            if task_name is None:
                unmatched.append(source)
                continue

            class_name = task_to_class[task_name]
            destination_dir = OUTPUT_ROOT / f"trajectory_{class_name}"
            destination = get_destination(
                source,
                destination_dir,
                task_name
            )

            if destination.name != source.name:
                renamed += 1
                print(
                    f"  rename: {source.name} -> {destination.name}"
                )

            shutil.copy2(source, destination)

            counts[class_name] += 1
            task_counts[task_name] += 1

    print()
    print("=" * 70)
    print("Optimization trajectories 添加完成")
    print("=" * 70)

    total_copied = 0

    for class_name in TASK_CLASSES:
        count = counts[class_name]
        total_copied += count

        current_total = len(
            list(
                (
                    OUTPUT_ROOT / f"trajectory_{class_name}"
                ).glob("*.pkl.gz")
            )
        )

        print(
            f"trajectory_{class_name:<15} "
            f"+{count:4d}    "
            f"total={current_total:5d}"
        )

    print("-" * 70)
    print(f"扫描轨迹:       {total_files}")
    print(f"新增轨迹:       {total_copied}")
    print(f"重命名轨迹:     {renamed}")
    print(f"未匹配轨迹:     {len(unmatched)}")

    if unmatched:
        print()
        print("=" * 70)
        print("未匹配文件")
        print("=" * 70)

        for path in unmatched:
            print(path)


if __name__ == "__main__":
    main()