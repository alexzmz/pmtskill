"""One explicit ADB/gRPC binding for AndroidWorld evaluators and collectors."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from .config import AndroidWorldConfig, resolve_android_world_grpc_port


def load_android_world_environment(launcher: Any, config: AndroidWorldConfig) -> Any:
    """Pass both ports together; an old launcher must fail instead of misbinding.

    Importing this module needs neither AndroidWorld nor a running emulator.
    ``launcher`` is provided after the caller bootstraps its chosen harness.
    """
    grpc_port = resolve_android_world_grpc_port(config.console_port, config.grpc_port)
    load = launcher.load_and_setup_env
    parameters = inspect.signature(load).parameters
    if "grpc_port" not in parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        raise RuntimeError(
            "当前 AndroidWorld env_launcher.load_and_setup_env 不支持 grpc_port；"
            "请更新 android_world_root 指向的 AndroidWorld 版本。"
            "不能忽略该参数后继续评测，否则可能操作和截图连接不同模拟器。"
        )
    logging.info(
        "ANDROIDWORLD_CONNECTION: ADB=emulator-%s; emulator_gRPC=localhost:%s; "
        "both must belong to the same emulator process (not the model server port)",
        config.console_port, grpc_port,
    )
    return load(
        console_port=config.console_port,
        grpc_port=grpc_port,
        emulator_setup=config.emulator_setup,
        adb_path=config.adb_path,
    )
