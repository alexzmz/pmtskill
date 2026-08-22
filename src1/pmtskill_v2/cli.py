"""PMT-Skill v2 统一命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from .core.config import ProjectConfig, load_config
from .core.io import load_primitives
from .offline.pipeline import OfflineDistillationPipeline
from .offline.trainer import (
    AdapterJob,
    MSSwiftLoraTrainer,
    default_training_job,
    filter_dataset_by_primitives,
)
from .online.backend import SkillOptimizationBackend
from .inference.vlm import OpenAICompatibleVLClient
from .online.planner import (
    KeywordSkillPlanner,
    PlannerPipeline,
    PrimitiveTopologyGenerator,
)
from .online.router import DynamicProgrammingRouter
from .skills.importer import import_skvm_skills, relevant_raw_skills
from .skills.compiler import LLMRawSkillCompiler, compile_imported_raw_skills
from .skills.store import SkillStore

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.example.toml"


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _tasks(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result or None


def _open(config_path: str) -> tuple[ProjectConfig, SkillStore]:
    config = load_config(config_path)
    config.ensure_runtime_dirs()
    store = SkillStore(config.paths.database)
    store.initialize()
    return config, store


def command_init(args: argparse.Namespace) -> int:
    config, store = _open(args.config)
    for profile in config.models:
        store.upsert_model_profile(profile)
    summary = import_skvm_skills(config.paths.skvm_skills_root, store)
    _print(
        {
            "database": str(config.paths.database),
            "models": [profile.model_id for profile in config.models],
            "primitives": len(load_primitives()),
            "skills": summary.to_dict(),
        }
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks = {
        "config": config.config_path.is_file(),
        "android_world": (config.paths.android_world_root / "android_world").is_dir(),
        "skvm_skills": config.paths.skvm_skills_root.is_dir(),
        "ms_swift_cli": (
            config.paths.ms_swift_root / "swift" / "cli" / "main.py"
        ).is_file(),
        "student_model_path": Path(config.offline.student_model_path)
        .expanduser()
        .exists(),
        "teacher_model_configured": any(
            profile.model_id == config.offline.teacher_model_id
            for profile in config.models
        ),
        "primitive_count_is_26": len(load_primitives()) == 26,
    }
    _print({"checks": checks, "healthy": all(checks.values())})
    # doctor 只诊断，不因为跨机器模型路径不存在而阻断查看结果。
    return 0 if all(checks.values()) else 2


def command_collect(args: argparse.Namespace) -> int:
    config, _ = _open(args.config)
    result = OfflineDistillationPipeline(config).collect(
        _tasks(args.tasks), combinations=args.combinations, seed=args.seed
    )
    _print(result.to_dict())
    return 0


def command_build_dataset(args: argparse.Namespace) -> int:
    config, _ = _open(args.config)
    root = Path(args.trajectory_dir).resolve() if args.trajectory_dir else None
    result = OfflineDistillationPipeline(config).build_dataset(
        root, successful_only=args.successful_only
    )
    _print(result.to_dict())
    return 0


def command_train(args: argparse.Namespace) -> int:
    config, _ = _open(args.config)
    trainer = MSSwiftLoraTrainer(config)
    if args.primitives:
        name = args.adapter_name or "directional_adapter"
        filtered_dir = config.offline.dataset_dir / "adapter_datasets" / name
        train_path = filtered_dir / "train.jsonl"
        validation_path = filtered_dir / "validation.jsonl"
        train_count = filter_dataset_by_primitives(
            config.offline.dataset_dir / "train.jsonl", train_path, args.primitives
        )
        source_validation = config.offline.dataset_dir / "validation.jsonl"
        validation_count = (
            filter_dataset_by_primitives(
                source_validation, validation_path, args.primitives
            )
            if source_validation.exists()
            else 0
        )
        if train_count == 0:
            raise ValueError(f"原语筛选后没有训练样本: {args.primitives}")
        job = AdapterJob(
            name=name,
            train_dataset=train_path,
            validation_dataset=validation_path if validation_count else None,
            output_dir=config.offline.output_dir / name,
            primitive_filter=tuple(args.primitives),
            extra_args=tuple(args.extra_arg or ()),
        )
    else:
        job = default_training_job(config)
        job.extra_args = tuple(args.extra_arg or ())
        if args.adapter_name:
            job.name = args.adapter_name
            job.output_dir = config.offline.output_dir / args.adapter_name
    command = trainer.build_command(job)
    if args.dry_run:
        _print({"dry_run": True, "job": job.name, "command": command})
        return 0
    code = trainer.run(job)
    _print({"job": job.name, "return_code": code, "output_dir": job.output_dir})
    return code


def command_maintain(args: argparse.Namespace) -> int:
    config, store = _open(args.config)
    backend = SkillOptimizationBackend(config, store)
    cycles = 0
    while True:
        result = backend.run_cycle()
        _print(result.to_dict())
        cycles += 1
        if not args.watch or (args.max_cycles and cycles >= args.max_cycles):
            break
        time.sleep(max(1.0, args.interval_seconds))
    return 0


def command_compile_skills(args: argparse.Namespace) -> int:
    """显式运行一批 SKVM→Android raw skill 云侧编译。"""

    config, store = _open(args.config)
    print(len(store.list_skills()))
    model_id = args.model_id or config.maintenance.raw_skill_compiler_model_id
    if not model_id:
        raise ValueError(
            "请传 --model-id，或在 [maintenance] 配置 raw_skill_compiler_model_id"
        )
    compiler = LLMRawSkillCompiler(
        OpenAICompatibleVLClient(config.model(model_id)), load_primitives()
    )
    result = compile_imported_raw_skills(store, compiler, limit=args.limit)
    _print({"compiler_model_id": model_id, **result.to_dict()})
    return 0


def command_skills(args: argparse.Namespace) -> int:
    _, store = _open(args.config)
    skills = store.list_skills(status=args.status, kind=args.kind)
    _print(
        [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "kind": skill.kind,
                "status": skill.status.value,
                "level": skill.level,
                "primitives": skill.topology.primitive_sequence(),
                "android_relevant": skill.metadata.get("android_relevant"),
                "metrics": store.skill_metrics(skill.skill_id),
            }
            for skill in skills
        ]
    )
    return 0


def command_profile(args: argparse.Namespace) -> int:
    config, store = _open(args.config)
    profile = store.list_model_profiles(enabled_only=False)
    by_id = {item.model_id: item for item in profile}
    target = by_id.get(args.model_id) or config.model(args.model_id)
    for assignment in args.capability:
        if "=" not in assignment:
            raise ValueError(f"能力必须写成 primitive_id=0.75: {assignment}")
        primitive, raw_score = assignment.split("=", 1)
        score = float(raw_score)
        if not 0 <= score <= 1:
            raise ValueError("能力分数必须在 [0,1]")
        target.capabilities[primitive] = score
    store.upsert_model_profile(target)
    _print(target.to_dict())
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config, store = _open(args.config)
    raw = relevant_raw_skills(store.list_skills(kind="raw"))
    decomposition, topology = PlannerPipeline(
        KeywordSkillPlanner(), PrimitiveTopologyGenerator()
    ).plan(args.goal, raw)
    profiles = store.list_model_profiles() or list(config.models)
    polished = store.list_skills(kind="polished")
    plan = DynamicProgrammingRouter(config.routing, store).route(
        args.goal,
        topology,
        profiles,
        polished,
        planner_id=decomposition.planner_id,
        include_candidates=args.include_candidates,
    )
    _print(
        {
            "decomposition": {
                "raw_skill_ids": decomposition.raw_skill_ids,
                "extra_primitives": decomposition.extra_primitives,
                "reasoning": decomposition.reasoning,
            },
            "plan": plan.to_dict(),
        }
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    # 延迟导入，确保没有 Android emulator 时也能使用其余 CLI。
    from .evaluation.android_world import AndroidWorldOnlineEvaluator

    config, store = _open(args.config)
    artifacts = AndroidWorldOnlineEvaluator(config, store).run(
        tasks=_tasks(args.tasks),
        n_task_combinations=args.combinations,
        seed=args.seed,
        family=args.family,
        planner_model_id=args.planner_model,
        include_candidate_skills=args.include_candidates,
        output_dir=args.output_dir,
    )
    _print(
        {
            "summary": artifacts.summary,
            "summary_json": artifacts.summary_json,
            "report_markdown": artifacts.report_markdown,
            "traces_jsonl": artifacts.traces_jsonl,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src1",
        description="PMT-Skill v2：AndroidWorld VL 蒸馏、动态模型/技能路由和技能维护",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML 配置文件")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init", help="初始化数据库、模型画像并导入 SKVM skills"
    )
    init.set_defaults(handler=command_init)

    doctor = subparsers.add_parser("doctor", help="检查本地路径和必要组件")
    doctor.set_defaults(handler=command_doctor)

    collect = subparsers.add_parser(
        "collect", help="教师 VL 模型采集 AndroidWorld 轨迹"
    )
    collect.add_argument(
        "--tasks", nargs="*", help="任务名，可空格或逗号分隔；默认全部"
    )
    collect.add_argument("--combinations", type=int, default=1, help="每任务参数组合数")
    collect.add_argument("--seed", type=int, default=42)
    collect.set_defaults(handler=command_collect)

    dataset = subparsers.add_parser(
        "build-dataset", help="把轨迹转换成 ms-swift VL JSONL"
    )
    dataset.add_argument("--trajectory-dir", help="覆盖配置中的轨迹目录")
    outcome_filter = dataset.add_mutually_exclusive_group()
    outcome_filter.add_argument(
        "--include-failed",
        dest="successful_only",
        action="store_false",
        help="包含失败/未知结果 episode 中可学习的 step（默认）",
    )
    outcome_filter.add_argument(
        "--successful-only",
        dest="successful_only",
        action="store_true",
        help="仅转换 AndroidWorld 判定成功的 episode",
    )
    dataset.set_defaults(successful_only=None)
    dataset.set_defaults(handler=command_build_dataset)

    train = subparsers.add_parser("train", help="运行 ms-swift 多模态 LoRA SFT")
    train.add_argument("--adapter-name", help="输出 adapter 名称")
    train.add_argument("--primitives", nargs="*", help="仅训练指定原语的定向 LoRA")
    train.add_argument(
        "--extra-arg",
        action="append",
        help="传给 ms-swift 的单个额外参数；可重复，例如 --extra-arg=--bf16",
    )
    train.add_argument("--dry-run", action="store_true", help="只显示命令，不启动训练")
    train.set_defaults(handler=command_train)

    maintain = subparsers.add_parser("maintain", help="运行一次 backend 技能优化周期")
    maintain.add_argument("--watch", action="store_true", help="持续运行自主维护循环")
    maintain.add_argument(
        "--interval-seconds", type=float, default=300.0, help="维护周期间隔"
    )
    maintain.add_argument(
        "--max-cycles", type=int, default=0, help="watch 最多循环次数；0 表示持续运行"
    )
    maintain.set_defaults(handler=command_maintain)

    compile_skills = subparsers.add_parser(
        "compile-skills", help="用云模型把 SKVM raw skills 编译为 Android 原语拓扑"
    )
    compile_skills.add_argument(
        "--model-id", help="云侧编译模型；默认读取 maintenance 配置"
    )
    compile_skills.add_argument("--limit", type=int, default=8, help="本批最大技能数")
    compile_skills.set_defaults(handler=command_compile_skills)

    skills = subparsers.add_parser("skills", help="查看技能及其真实在线统计")
    skills.add_argument(
        "--status", choices=["imported", "candidate", "active", "deprecated"]
    )
    skills.add_argument("--kind", choices=["raw", "polished"])
    skills.set_defaults(handler=command_skills)

    profile = subparsers.add_parser("profile", help="更新模型/LoRA 的原语能力画像")
    profile.add_argument("--model-id", required=True)
    profile.add_argument(
        "--capability",
        action="append",
        required=True,
        help="primitive_id=score，可重复",
    )
    profile.set_defaults(handler=command_profile)

    plan = subparsers.add_parser("plan", help="仅查看任务的动态规划路由，不连接模拟器")
    plan.add_argument("--goal", required=True)
    plan.add_argument("--include-candidates", action="store_true")
    plan.set_defaults(handler=command_plan)

    evaluate = subparsers.add_parser(
        "evaluate", help="运行动态模型/技能 AndroidWorld 评测"
    )
    evaluate.add_argument("--tasks", nargs="*", help="任务名；默认全部")
    evaluate.add_argument("--combinations", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--family", default="android_world")
    evaluate.add_argument("--planner-model", help="可选；不填则使用低延迟关键词规划")
    evaluate.add_argument(
        "--include-candidates", action="store_true", help="灰度试用候选技能"
    )
    evaluate.add_argument("--output-dir")
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
