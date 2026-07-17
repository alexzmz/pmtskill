# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import os
from pathlib import Path
import random
import sys
from typing import Type

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANDROID_WORLD_ROOT_ENV = os.environ.get("ANDROID_WORLD_ROOT")
_ANDROID_WORLD_ROOT = (
    Path(_ANDROID_WORLD_ROOT_ENV).expanduser().resolve()
    if _ANDROID_WORLD_ROOT_ENV
    else _REPO_ROOT / "libs" / "android_world"
)
for _path in (_ANDROID_WORLD_ROOT, _REPO_ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from absl import app
from absl import flags
from absl import logging

from android_world import registry
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.task_evals import task_eval
from vllm_wrapper import VLLMWrapper

logging.set_verbosity(logging.WARNING)

os.environ["GRPC_VERBOSITY"] = "ERROR"  # Only show errors
os.environ["GRPC_TRACE"] = "none"  # Disable tracing


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    env_adb_path = os.environ.get("ADB_PATH")
    if env_adb_path and os.path.isfile(env_adb_path):
        return env_adb_path

    potential_paths = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("/home/zmz/Workspace/gui/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    return "adb"


_ADB_PATH = flags.DEFINE_string(
    "adb_path",
    _find_adb_directory(),
    "Path to adb. Set if not installed through SDK.",
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup",
    False,
    "Whether to perform emulator setup. This must be done once and only once"
    " before running Android World. After an emulator is setup, this flag"
    " should always be False.",
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port",
    5554,
    "The console port of the running Android device. This can usually be"
    " retrieved by looking at the output of `adb devices`. In general, the"
    " first connected device is port 5554, the second is 5556, and"
    " so on.",
)

_TASK = flags.DEFINE_string(
    "task",
    None,
    "A specific task to run.",
)
_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    os.environ.get("LOCAL_MODEL_PATH", "/home/zmz/Workspace/models/qwen3.5-4b"),
    "Path to the local model used by vLLM.",
)


def _main() -> None:
    """Runs a single task."""
    print("Loading Android environment...", flush=True)
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    print("Resetting Android environment...", flush=True)
    env.reset(go_home=True)

    print("Selecting task...", flush=True)
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    if _TASK.value:
        if _TASK.value not in aw_registry:
            raise ValueError("Task {} not found in registry.".format(_TASK.value))
        task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
    else:
        task_type: Type[task_eval.TaskEval] = random.choice(list(aw_registry.values()))
    params = task_type.generate_random_params()
    task = task_type(params)
    print(f"Initializing task: {task_type.__name__}", flush=True)
    task.initialize_task(env)

    print(f"Loading local model with vLLM: {_MODEL_PATH.value}", flush=True)
    agent = t3a.T3A(
        env,
        VLLMWrapper(
            model_path=_MODEL_PATH.value,
            temperature=0.7,
            tensor_parallel_size=1,
        ),
    )
    print("Agent initialized.", flush=True)

    print("Goal: " + str(task.goal))
    is_done = False
    for _ in range(int(task.complexity * 10)):
        response = agent.step(task.goal)
        if response.done:
            is_done = True
            break
    agent_successful = is_done and task.is_successful(env) == 1
    print(
        f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
        f" {task.goal}"
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
