#!/usr/bin/env python3
"""
Quick AndroidWorld emulator setup script.

This script performs the one-time emulator initialization required by
AndroidWorld. It is equivalent to running AndroidWorld with:

    --perform_emulator_setup

Usage:

    python src/setup_androidworld.py --console_port 5556

After setup, do NOT run setup again on the same emulator.
"""

import argparse
import os
import time

from android_world.env import env_launcher


DEFAULT_ADB_PATH = (
    "/home/zmz/Workspace/gui/Android/Sdk/platform-tools/adb"
)


def check_adb(adb_path: str):
    """Check adb exists."""
    if not os.path.exists(adb_path):
        raise FileNotFoundError(
            f"adb not found: {adb_path}"
        )


def wait_for_device(console_port: int, timeout: int = 60):
    """
    Wait until emulator is connected.

    AndroidWorld uses console port:
        5554 -> emulator-5554
        5556 -> emulator-5556
    """

    import subprocess

    serial = f"emulator-{console_port}"

    print(f"Waiting for {serial} ...")

    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run(
            [
                "adb",
                "-s",
                serial,
                "get-state",
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip() == "device":
            print(f"{serial} is ready.")
            return

        time.sleep(2)

    raise RuntimeError(
        f"{serial} is not ready after {timeout}s"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Initialize AndroidWorld emulator"
    )

    parser.add_argument(
        "--console_port",
        type=int,
        default=5556,
        help="Android emulator console port",
    )

    parser.add_argument(
        "--adb_path",
        default=DEFAULT_ADB_PATH,
        help="Path to adb",
    )

    args = parser.parse_args()

    check_adb(args.adb_path)

    # AndroidWorld internally expects adb in PATH
    os.environ["PATH"] = (
        os.path.dirname(args.adb_path)
        + ":"
        + os.environ["PATH"]
    )

    wait_for_device(args.console_port)

    print("=" * 60)
    print("Starting AndroidWorld emulator setup...")
    print("=" * 60)

    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=True,
        adb_path=args.adb_path,
    )

    print("=" * 60)
    print("AndroidWorld setup finished successfully.")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
