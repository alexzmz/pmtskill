"""PMT-Skill v2 统一命令行入口。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Sequence

from .core.config import ProjectConfig, load_config
from .core.io import load_primitives
from .core.run_records import CommandRunLogger, active_run_logger
from .offline.pipeline import OfflineDistillationPipeline
from .offline.trainer import (
    AdapterJob,
    MSSwiftLoraTrainer,
    default_training_job,
    filter_dataset_by_primitives,
    find_latest_adapter_checkpoint,
    load_prepared_training_job,
    prepare_training_job,
    staged_training_job,
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
    current_run = active_run_logger()
    if current_run is not None:
        current_run.record_result(value)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _run_label(args: argparse.Namespace) -> str:
    """从命令参数中提取便于人眼识别、不会过长的日志目录标签。"""

    task_values = _tasks(getattr(args, "tasks", None))
    if task_values:
        suffix = f"+{len(task_values) - 3}more" if len(task_values) > 3 else ""
        return "+".join(task_values[:3]) + suffix
    for name in ("adapter_name", "model_id", "family"):
        value = getattr(args, name, None)
        if value:
            return str(value)
    goal = getattr(args, "goal", None)
    if goal:
        return str(goal)[:56]
    if getattr(args, "command", "").startswith("evaluate") or getattr(
        args, "command", None
    ) == "collect":
        return "all-tasks"
    return ""


def _resolve_log_root(args: argparse.Namespace) -> Path:
    """优先使用 CLI 覆盖值；配置损坏时仍把失败日志写到默认 runtime。"""

    if args.log_dir:
        return Path(args.log_dir).expanduser().resolve()
    try:
        return load_config(args.config).paths.log_dir
    except Exception:
        return (DEFAULT_CONFIG.parent / "runtime" / "logs").resolve()


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
        "skill_library": config.paths.database.is_file(),
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
        _tasks(args.tasks),
        combinations=args.combinations,
        seed=args.seed,
        max_steps=args.max_steps,
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
    config, store = _open(args.config)
    if args.train_cuda_visible_devices is not None:
        config.offline.cuda_visible_devices = args.train_cuda_visible_devices
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
    with_evaluation = (
        args.with_evaluation
        if args.with_evaluation is not None
        else config.training_evaluation.enabled
    )
    settings = _training_evaluation_settings(config, args)
    # 完整评测可关闭，但默认 SR 早退仍需要 baseline/逐 epoch standalone probe。
    use_staged_workflow = with_evaluation or settings.early_stopping_enabled
    resume_checkpoint = None
    if args.resume and not use_staged_workflow:
        try:
            resume_checkpoint = find_latest_adapter_checkpoint(job.output_dir)
        except FileNotFoundError:
            pass
        if resume_checkpoint is not None:
            job.resume_from_checkpoint = resume_checkpoint
    command = trainer.build_command(job)
    if args.dry_run:
        if not use_staged_workflow:
            _print(
                {
                    "dry_run": True,
                    "job": job.name,
                    "command": command,
                    "early_stopping": {"enabled": False},
                }
            )
            return 0
        from .evaluation.deployment import MSSwiftEvaluationDeployment
        from .offline.training_workflow import (
            build_epoch_plan,
            epoch_label,
            resolve_student_profile,
        )

        run_dir = _training_evaluation_output_dir(job, args)
        profile = resolve_student_profile(config, settings.model_id)
        deployment = MSSwiftEvaluationDeployment(config, settings, profile)
        plan = build_epoch_plan(
            config.offline.epochs,
            1 if settings.early_stopping_enabled else settings.every_epochs,
            settings.checkpoint_every_epochs,
        )
        stage_commands = []
        for index, stage in enumerate(plan):
            target = stage.target_epoch
            stage_job = staged_training_job(
                job,
                output_dir=run_dir / "training" / epoch_label(target),
                target_epoch=target,
                resume_from_checkpoint=(
                    Path("CHECKPOINT_FROM_PREVIOUS_STAGE") if index > 0 else None
                ),
            )
            stage_commands.append(
                {
                    "stage": epoch_label(target),
                    "target_epoch": target,
                    "evaluate_after_stage": stage.evaluate,
                    "retain_checkpoint": stage.retain_checkpoint,
                    "resume_from_previous_checkpoint": index > 0,
                    "command": trainer.build_command(stage_job),
                }
            )
        resumed_tasks = _resume_evaluation_tasks(run_dir) if args.resume else None
        selected_tasks = (
            _tasks(args.eval_tasks)
            or resumed_tasks
            or list(settings.tasks)
        )
        _print(
            {
                "dry_run": True,
                "job": job.name,
                "with_android_evaluation": with_evaluation,
                "evaluation_mode": (
                    "full" if with_evaluation else "early_stopping_probe_only"
                ),
                "early_stopping": {
                    "enabled": settings.early_stopping_enabled,
                    "metric": "standalone_micro_sr",
                    "patience": settings.early_stopping_patience,
                    "min_delta": settings.early_stopping_min_delta,
                },
                "evaluation_output_dir": run_dir,
                "resume": args.resume,
                "resuming_existing_run": (run_dir / "history.json").is_file(),
                "dataset_source": job.train_dataset,
                "dataset_snapshot_at_runtime": run_dir / "dataset_snapshot",
                "resource_assignment": {
                    "training_cuda_visible_devices": (
                        config.offline.cuda_visible_devices
                    ),
                    "evaluation_cuda_visible_devices": (
                        settings.cuda_visible_devices
                    ),
                    "evaluation_max_steps": settings.max_steps,
                    "evaluation_max_model_len": settings.max_model_len,
                    "evaluation_gpu_memory_utilization": (
                        settings.gpu_memory_utilization
                    ),
                },
                "tasks": selected_tasks or (
                    "all"
                    if settings.task_count is None
                    else {
                        "sample_count": settings.task_count,
                        "seed": settings.seed,
                        "family": settings.family,
                    }
                ),
                "baseline_deploy_command": deployment.build_command(None),
                "checkpoint_deploy_command_template": deployment.build_command(
                    Path("CHECKPOINT_FROM_PREVIOUS_STAGE")
                ),
                "training_stages": stage_commands,
                "evaluation_sequence": [
                    "baseline_standalone",
                    *(["baseline_skills"] if with_evaluation else []),
                    *[
                        f"{epoch_label(stage.target_epoch)}_standalone"
                        for stage in plan
                        if stage.evaluate
                    ],
                    *(["final_skills"] if with_evaluation else []),
                ],
            }
        )
        return 0
    if use_staged_workflow:
        # 完整评测关闭时仍延迟加载早退 probe 所需的 AndroidWorld 模块。
        from .evaluation.android_world import sample_android_world_tasks
        from .evaluation.deployment import MSSwiftEvaluationDeployment
        from .offline.training_workflow import (
            TrainingEvaluationOptions,
            TrainingEvaluationWorkflow,
            resolve_student_profile,
        )

        run_dir = _training_evaluation_output_dir(job, args)
        resumed_tasks = _resume_evaluation_tasks(run_dir) if args.resume else None
        explicit_tasks = _tasks(args.eval_tasks)
        if explicit_tasks is not None:
            selected_tasks = sample_android_world_tasks(
                config,
                tasks=explicit_tasks,
                task_count=settings.task_count,
                seed=settings.seed,
                family=settings.family,
            )
        elif resumed_tasks is not None:
            selected_tasks = resumed_tasks
        else:
            requested_tasks = list(settings.tasks) or None
            selected_tasks = sample_android_world_tasks(
                config,
                tasks=requested_tasks,
                task_count=settings.task_count,
                seed=settings.seed,
                family=settings.family,
            )
        profile = resolve_student_profile(config, settings.model_id)
        deployment = MSSwiftEvaluationDeployment(config, settings, profile)
        result = TrainingEvaluationWorkflow(
            config,
            store,
            trainer,
            deployment=deployment,
        ).run(
            job,
            TrainingEvaluationOptions(
                output_dir=run_dir,
                tasks=tuple(selected_tasks),
                family=settings.family,
                combinations=settings.combinations,
                seed=settings.seed,
                max_steps=settings.max_steps,
                every_epochs=settings.every_epochs,
                checkpoint_every_epochs=settings.checkpoint_every_epochs,
                include_candidate_skills=settings.include_candidate_skills,
                full_evaluation=with_evaluation,
                early_stopping_enabled=settings.early_stopping_enabled,
                early_stopping_patience=settings.early_stopping_patience,
                early_stopping_min_delta=settings.early_stopping_min_delta,
                training_cuda_visible_devices=(
                    config.offline.cuda_visible_devices
                ),
                evaluation_cuda_visible_devices=(
                    settings.cuda_visible_devices
                ),
                evaluation_max_model_len=settings.max_model_len,
                evaluation_gpu_memory_utilization=(
                    settings.gpu_memory_utilization
                ),
                resume=args.resume,
            ),
        )
        _print(
            {
                "job": job.name,
                "with_android_evaluation": with_evaluation,
                "evaluation_mode": (
                    "full" if with_evaluation else "early_stopping_probe_only"
                ),
                "evaluation_max_steps": settings.max_steps,
                **result.to_dict(),
            }
        )
        return result.return_code
    snapshot_dir = job.output_dir / "dataset_snapshot"
    if resume_checkpoint is not None:
        for parent in resume_checkpoint.parents:
            candidate = parent / "dataset_snapshot"
            if (candidate / "manifest.json").is_file():
                snapshot_dir = candidate
                break
    if resume_checkpoint is not None and (snapshot_dir / "manifest.json").is_file():
        prepared = load_prepared_training_job(job, snapshot_dir=snapshot_dir)
    else:
        prepared = prepare_training_job(
            job,
            configured_dataset_dir=config.offline.dataset_dir,
            snapshot_dir=snapshot_dir,
        )
    code = trainer.run(prepared.job)
    _print(
        {
            "job": job.name,
            "return_code": code,
            "output_dir": job.output_dir,
            "dataset_manifest": prepared.manifest_path,
            "resumed_from_checkpoint": resume_checkpoint,
        }
    )
    return code


def _training_evaluation_settings(
    config: ProjectConfig, args: argparse.Namespace
):
    """合并 TOML 与 train CLI 覆盖值，不修改全局配置对象。"""

    current = config.training_evaluation
    resolved = dataclasses.replace(
        current,
        model_id=args.eval_model_id or current.model_id,
        task_count=(
            args.eval_task_count
            if args.eval_task_count is not None
            else current.task_count
        ),
        combinations=(
            args.eval_combinations
            if args.eval_combinations is not None
            else current.combinations
        ),
        seed=args.eval_seed if args.eval_seed is not None else current.seed,
        max_steps=(
            args.eval_max_steps
            if args.eval_max_steps is not None
            else current.max_steps
        ),
        every_epochs=(
            args.eval_every_epochs
            if args.eval_every_epochs is not None
            else current.every_epochs
        ),
        early_stopping_enabled=(
            args.early_stopping
            if args.early_stopping is not None
            else current.early_stopping_enabled
        ),
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else current.early_stopping_patience
        ),
        early_stopping_min_delta=(
            args.early_stopping_min_delta
            if args.early_stopping_min_delta is not None
            else current.early_stopping_min_delta
        ),
        checkpoint_every_epochs=(
            args.checkpoint_every_epochs
            if args.checkpoint_every_epochs is not None
            else current.checkpoint_every_epochs
        ),
        include_candidate_skills=(
            args.eval_include_candidates
            if args.eval_include_candidates is not None
            else current.include_candidate_skills
        ),
        deploy_port=(
            args.eval_deploy_port
            if args.eval_deploy_port is not None
            else current.deploy_port
        ),
        infer_backend=args.eval_infer_backend or current.infer_backend,
        max_model_len=(
            args.eval_max_model_len
            if args.eval_max_model_len is not None
            else current.max_model_len
        ),
        gpu_memory_utilization=(
            args.eval_gpu_memory_utilization
            if args.eval_gpu_memory_utilization is not None
            else current.gpu_memory_utilization
        ),
        cuda_visible_devices=(
            args.eval_cuda_visible_devices
            if args.eval_cuda_visible_devices is not None
            else current.cuda_visible_devices
        ),
    )
    if resolved.task_count is not None and resolved.task_count <= 0:
        raise ValueError("--eval-task-count 必须是正整数")
    if resolved.combinations <= 0:
        raise ValueError("--eval-combinations 必须是正整数")
    if resolved.every_epochs <= 0:
        raise ValueError("--eval-every-epochs 必须是正整数")
    if resolved.max_steps <= 0:
        raise ValueError("--eval-max-steps 必须是正整数")
    if resolved.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience 必须是正整数")
    if not 0 <= resolved.early_stopping_min_delta <= 1:
        raise ValueError("--early-stopping-min-delta 必须在 [0, 1]")
    if resolved.checkpoint_every_epochs < 0:
        raise ValueError("--checkpoint-every-epochs 必须是非负整数")
    if not 1 <= resolved.deploy_port <= 65535:
        raise ValueError("--eval-deploy-port 必须在 [1, 65535]")
    if resolved.max_model_len <= 0:
        raise ValueError("--eval-max-model-len 必须是正整数")
    if not 0 < resolved.gpu_memory_utilization <= 1:
        raise ValueError("--eval-gpu-memory-utilization 必须在 (0, 1]")
    if resolved.startup_timeout_seconds <= 0:
        raise ValueError("training_evaluation.startup_timeout_seconds 必须为正数")
    if resolved.startup_poll_seconds <= 0:
        raise ValueError("training_evaluation.startup_poll_seconds 必须为正数")
    if resolved.max_new_tokens <= 0:
        raise ValueError("training_evaluation.max_new_tokens 必须是正整数")
    return resolved


def _training_evaluation_output_dir(
    job: AdapterJob, args: argparse.Namespace
) -> Path:
    """解析训练运行目录；默认优先续接同 adapter 最近一次运行。"""

    if args.eval_output_dir:
        return Path(args.eval_output_dir).expanduser().resolve()
    if args.resume:
        runs_root = job.output_dir / "training_runs"
        candidates: list[tuple[float, Path]] = []
        if runs_root.is_dir():
            for child in runs_root.iterdir():
                history_path = child / "history.json"
                if not child.is_dir() or not history_path.is_file():
                    continue
                try:
                    state = json.loads(history_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                manifest = state.get("manifest", {})
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("job") != job.name
                ):
                    continue
                candidates.append((history_path.stat().st_mtime, child))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1].resolve()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return (
        job.output_dir / "training_runs" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    ).resolve()


def _resume_evaluation_tasks(run_dir: Path) -> list[str] | None:
    """续训沿用原运行固定的评测任务，避免前后 epoch 样本集合漂移。"""

    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return None
    try:
        state = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest = state.get("manifest", {})
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
        return None
    return tasks


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


def _evaluation_settings(
    config: ProjectConfig, args: argparse.Namespace
):
    """用 CLI 参数覆盖公共部署配置，三种评测保持完全相同的服务设置。"""

    if args.max_steps <= 0:
        raise ValueError("--max-steps 必须是正整数")
    current = config.training_evaluation
    resolved = dataclasses.replace(
        current,
        deploy_host=args.deploy_host or current.deploy_host,
        deploy_port=(
            args.deploy_port if args.deploy_port is not None else current.deploy_port
        ),
        infer_backend=args.infer_backend or current.infer_backend,
        max_model_len=(
            args.max_model_len
            if args.max_model_len is not None
            else current.max_model_len
        ),
        gpu_memory_utilization=(
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else current.gpu_memory_utilization
        ),
        cuda_visible_devices=(
            args.cuda_visible_devices
            if args.cuda_visible_devices is not None
            else current.cuda_visible_devices
        ),
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else current.max_new_tokens
        ),
        deploy_extra_args=(
            tuple(current.deploy_extra_args) + tuple(args.deploy_extra_arg or ())
        ),
    )
    if not 1 <= resolved.deploy_port <= 65535:
        raise ValueError("--deploy-port 必须在 [1, 65535]")
    if resolved.max_model_len <= 0:
        raise ValueError("--max-model-len 必须是正整数")
    if not 0 < resolved.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization 必须在 (0, 1]")
    if resolved.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须是正整数")
    return resolved


def _evaluation_tasks(config: ProjectConfig, args: argparse.Namespace):
    """显式任务原样使用；指定 task-count 时按 seed 固定抽样，否则评测全部。"""

    if args.combinations <= 0:
        raise ValueError("--combinations 必须是正整数")
    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("--task-count 必须是正整数")
    selected = _tasks(args.tasks)
    if selected or args.task_count is None:
        return selected
    from .evaluation.android_world import sample_android_world_tasks

    return sample_android_world_tasks(
        config,
        tasks=None,
        task_count=args.task_count,
        seed=args.seed,
        family=args.family,
    )


def _open_evaluation_skill_store(
    config: ProjectConfig, database: str | None
) -> SkillStore:
    """只打开用户指定的既有技能库，避免路径写错时静默创建空库。"""

    path = (
        Path(database).expanduser().resolve()
        if database
        else config.paths.database.resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"技能库不存在: {path}；请设置 [paths].skill_library_db 或 --skill-database"
        )
    store = SkillStore(path)
    store.initialize()
    return store


def _evaluation_profile_template(
    config: ProjectConfig,
    store: SkillStore | None,
    model_id: str,
):
    """尽量继承同名能力画像；新 adapter 则继承训练评测模板的能力先验。"""

    profiles = store.list_model_profiles(enabled_only=False) if store else []
    profiles.extend(config.models)
    exact = next((item for item in profiles if item.model_id == model_id), None)
    if exact is not None:
        return exact
    fallback_id = config.training_evaluation.model_id
    fallback = next(
        (item for item in profiles if fallback_id and item.model_id == fallback_id),
        next((item for item in profiles if item.enabled), None),
    )
    if fallback is None:
        raise ValueError("没有可用于 adapter 的模型能力画像")
    return dataclasses.replace(fallback, model_id=model_id)


def _adapter_deployment(
    config: ProjectConfig,
    store: SkillStore | None,
    args: argparse.Namespace,
    adapter_paths: Sequence[str],
    model_ids: Sequence[str] | None,
):
    """解析 adapter 路径并构造命名 LoRA 部署器。"""

    from .evaluation.adapters import (
        AdapterDeploymentBinding,
        MSSwiftAdapterDeployment,
        adapter_model_id,
        resolve_adapter_checkpoint,
    )

    if args.combinations <= 0:
        raise ValueError("--combinations 必须是正整数")
    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("--task-count 必须是正整数")
    resolved = [
        resolve_adapter_checkpoint(
            path,
            require_training_state=args.require_training_state,
        )
        for path in adapter_paths
    ]
    ids = list(model_ids or ())
    if ids and len(ids) != len(resolved):
        raise ValueError("--adapter-model-ids 数量必须与 --adapter-paths 相同")
    if not ids:
        ids = [adapter_model_id(item.adapter_root) for item in resolved]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "自动生成的 adapter model_id 重复，请用 --adapter-model-ids 显式指定"
        )
    bindings = tuple(
        AdapterDeploymentBinding(
            model_id=model_id,
            checkpoint=checkpoint,
            template_profile=_evaluation_profile_template(
                config, store, model_id
            ),
        )
        for model_id, checkpoint in zip(ids, resolved)
    )
    deployment = MSSwiftAdapterDeployment(
        config,
        _evaluation_settings(config, args),
        bindings,
        base_model_path=args.base_model_path,
    )
    return deployment, bindings


def _evaluation_result(
    *,
    mode: str,
    artifacts,
    bindings,
    skill_database: Path | None,
) -> dict[str, Any]:
    return {
        "evaluation_mode": mode,
        "summary": artifacts.summary,
        "summary_json": artifacts.summary_json,
        "report_markdown": artifacts.report_markdown,
        "traces_jsonl": artifacts.traces_jsonl,
        "checkpoint_dir": artifacts.output_dir / "checkpoints",
        "adapters": [
            {
                "model_id": binding.model_id,
                **binding.checkpoint.to_dict(),
            }
            for binding in bindings
        ],
        "skill_database": str(skill_database) if skill_database else None,
    }


def _dry_evaluation_result(mode: str, deployment, bindings, args) -> int:
    _print(
        {
            "dry_run": True,
            "evaluation_mode": mode,
            "deployment_command": deployment.build_adapter_command(),
            "profiles": [profile.to_dict() for profile in deployment.profiles()],
            "adapters": [
                {
                    "model_id": binding.model_id,
                    **binding.checkpoint.to_dict(),
                }
                for binding in bindings
            ],
            "tasks": _tasks(args.tasks) or (
                f"sample:{args.task_count}" if args.task_count is not None else "all"
            ),
            "combinations": args.combinations,
            "seed": args.seed,
            "family": args.family,
            "max_steps": args.max_steps,
            "output_dir": args.output_dir,
        }
    )
    return 0


def command_evaluate_standalone(args: argparse.Namespace) -> int:
    """接口一：单 adapter 裸模型 + 原生 M3A，不读取或注入技能。"""

    from .evaluation.android_world import AndroidWorldStandaloneEvaluator

    config = load_config(args.config)
    config.ensure_runtime_dirs()
    deployment, bindings = _adapter_deployment(
        config,
        None,
        args,
        [args.adapter_path],
        [args.model_id] if args.model_id else None,
    )
    if args.dry_run:
        return _dry_evaluation_result("standalone", deployment, bindings, args)
    tasks = _evaluation_tasks(config, args)
    with deployment.activate_adapters() as profiles:
        artifacts = AndroidWorldStandaloneEvaluator(config).run(
            profile=profiles[0],
            tasks=tasks,
            n_task_combinations=args.combinations,
            seed=args.seed,
            family=args.family,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
    _print(
        _evaluation_result(
            mode="standalone",
            artifacts=artifacts,
            bindings=bindings,
            skill_database=None,
        )
    )
    return 0


def command_evaluate_simple_skills(args: argparse.Namespace) -> int:
    """接口二：单 adapter + 确定性关键词检索出的一个技能。"""

    from .evaluation.android_world import AndroidWorldSimpleSkillEvaluator

    config = load_config(args.config)
    config.ensure_runtime_dirs()
    store = _open_evaluation_skill_store(config, args.skill_database)
    deployment, bindings = _adapter_deployment(
        config,
        store,
        args,
        [args.adapter_path],
        [args.model_id] if args.model_id else None,
    )
    if args.dry_run:
        return _dry_evaluation_result("simple_skills", deployment, bindings, args)
    tasks = _evaluation_tasks(config, args)
    with deployment.activate_adapters() as profiles:
        artifacts = AndroidWorldSimpleSkillEvaluator(config, store).run(
            profile=profiles[0],
            tasks=tasks,
            n_task_combinations=args.combinations,
            seed=args.seed,
            family=args.family,
            max_steps=args.max_steps,
            include_candidate_skills=args.include_candidates,
            output_dir=args.output_dir,
            record_traces=args.record_traces,
        )
    _print(
        _evaluation_result(
            mode="simple_skills",
            artifacts=artifacts,
            bindings=bindings,
            skill_database=store.database.resolve(),
        )
    )
    return 0


def command_evaluate_pmtskill(args: argparse.Namespace) -> int:
    """接口三：多 adapter 模型池 + PMT-Skill 规划、技能和动态路由。"""

    from .evaluation.android_world import AndroidWorldOnlineEvaluator

    config = load_config(args.config)
    config.ensure_runtime_dirs()
    store = _open_evaluation_skill_store(config, args.skill_database)
    route_overrides = {
        "success_weight": args.routing_success_weight,
        "latency_weight": args.routing_latency_weight,
        "switch_weight": args.routing_switch_weight,
        "polished_bonus": args.routing_polished_bonus,
        "degradation_weight": args.routing_degradation_weight,
        "minimum_capability": args.routing_minimum_capability,
        "maximum_candidates_per_position": args.routing_maximum_candidates,
    }
    for name in (
        "routing_success_weight",
        "routing_latency_weight",
        "routing_switch_weight",
        "routing_polished_bonus",
        "routing_degradation_weight",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是非负数")
    if (
        args.routing_minimum_capability is not None
        and not 0 <= args.routing_minimum_capability <= 1
    ):
        raise ValueError("--routing-minimum-capability 必须在 [0, 1]")
    if (
        args.routing_maximum_candidates is not None
        and args.routing_maximum_candidates <= 0
    ):
        raise ValueError("--routing-maximum-candidates 必须是正整数")
    config.routing = dataclasses.replace(
        config.routing,
        **{key: value for key, value in route_overrides.items() if value is not None},
    )
    deployment, bindings = _adapter_deployment(
        config,
        store,
        args,
        args.adapter_paths,
        args.adapter_model_ids,
    )
    if args.dry_run:
        return _dry_evaluation_result("pmtskill_online", deployment, bindings, args)
    tasks = _evaluation_tasks(config, args)
    with deployment.activate_adapters() as profiles:
        artifacts = AndroidWorldOnlineEvaluator(config, store).run(
            tasks=tasks,
            n_task_combinations=args.combinations,
            seed=args.seed,
            family=args.family,
            max_steps=args.max_steps,
            planner_model_id=args.planner_model,
            include_candidate_skills=args.include_candidates,
            output_dir=args.output_dir,
            model_profiles=profiles,
            record_traces=args.record_traces,
        )
    _print(
        _evaluation_result(
            mode="pmtskill_online",
            artifacts=artifacts,
            bindings=bindings,
            skill_database=store.database.resolve(),
        )
    )
    return 0


def _add_common_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """三种评测共享的任务选择、部署资源和输出参数。"""

    parser.add_argument("--tasks", nargs="*", help="任务名；默认全部")
    parser.add_argument(
        "--task-count",
        type=int,
        help="不传 --tasks 时按 seed 抽样的任务数；不传表示全部",
    )
    parser.add_argument("--combinations", type=int, default=1, help="每任务参数组合数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", default="android_world")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="每个 episode 的步数上限；默认 30",
    )
    parser.add_argument("--output-dir", help="summary/report/traces/checkpoints 输出目录")
    parser.add_argument("--base-model-path", help="覆盖 adapter_config 中记录的基座模型")
    parser.add_argument(
        "--require-training-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "要求选中的 best checkpoint 同时包含 optimizer.pt 和 scheduler.pt（默认）；"
            "评测导出的纯 adapter 可用 --no-require-training-state"
        ),
    )
    parser.add_argument("--deploy-host", help="临时 ms-swift 服务监听地址")
    parser.add_argument("--deploy-port", type=int, help="临时 ms-swift 服务端口")
    parser.add_argument(
        "--cuda-visible-devices",
        help="评测服务使用的物理 GPU，例如 1 或 1,2",
    )
    parser.add_argument(
        "--infer-backend",
        choices=["vllm", "transformers", "sglang", "lmdeploy"],
        help="临时模型服务推理后端；多 adapter 路由要求 vllm",
    )
    parser.add_argument("--max-model-len", type=int, help="vLLM 最大上下文长度")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, help="vLLM 显存使用比例 (0,1]"
    )
    parser.add_argument("--max-new-tokens", type=int, help="单次模型调用最大输出 token")
    parser.add_argument(
        "--deploy-extra-arg",
        action="append",
        help="传给 swift deploy 的额外单个参数；可重复",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 checkpoint 并输出部署命令，不启动 GPU/emulator",
    )


def _add_skill_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skill-database",
        help="覆盖 [paths].skill_library_db，必须指向已有 skill_library.sqlite3",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="除 active 外也允许 candidate polished skills",
    )
    parser.add_argument(
        "--record-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否把本轮轻量轨迹写回技能库；默认关闭，避免基准测试污染路由统计",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src1",
        description="PMT-Skill v2：AndroidWorld VL 蒸馏、动态模型/技能路由和技能维护",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML 配置文件")
    parser.add_argument(
        "--log-dir",
        help="覆盖 [paths].log_dir；每次调用仍会在其中创建独立运行目录",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="runtime.log 中 Python logging 的最低级别",
    )
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
    collect.add_argument(
        "--max-steps",
        type=int,
        help="每个 episode 的步数上限；可调低，但无论配置为何都不会超过 50",
    )
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
        "--train-cuda-visible-devices",
        help=(
            "ms-swift SFT 使用的物理 GPU，例如 2 或 2,3；"
            "覆盖 [offline].cuda_visible_devices"
        ),
    )
    train.add_argument(
        "--extra-arg",
        action="append",
        help="传给 ms-swift 的单个额外参数；可重复，例如 --extra-arg=--bf16",
    )
    train.add_argument("--dry-run", action="store_true", help="只显示命令，不启动训练")
    train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "自动续接同名 adapter 最近一次训练（默认启用）；"
            "使用 --no-resume 强制创建新运行或拒绝非空的显式输出目录"
        ),
    )
    evaluation_switch = train.add_mutually_exclusive_group()
    evaluation_switch.add_argument(
        "--with-evaluation",
        dest="with_evaluation",
        action="store_true",
        help="启用训练前/逐 epoch/训练后的 AndroidWorld 固定子集评测",
    )
    evaluation_switch.add_argument(
        "--without-evaluation",
        dest="with_evaluation",
        action="store_false",
        help=(
            "关闭完整模型/技能库报告；默认仍运行早退所需的轻量 AndroidWorld "
            "standalone probe，配合 --no-early-stopping 才是纯 SFT"
        ),
    )
    train.set_defaults(with_evaluation=None)
    train.add_argument(
        "--eval-model-id",
        help="学生模型画像 ID；默认读取 [training_evaluation].model_id",
    )
    train.add_argument(
        "--eval-tasks",
        nargs="*",
        help="固定评测任务；不传且未设置数量时默认全部任务",
    )
    train.add_argument(
        "--eval-task-count",
        type=int,
        help="未显式指定任务时的抽样数；不传且 TOML 未设置时评测全部任务",
    )
    train.add_argument("--eval-combinations", type=int, help="每个任务的参数组合数")
    train.add_argument("--eval-seed", type=int, help="任务抽样和实例参数随机种子")
    train.add_argument(
        "--eval-max-steps",
        type=int,
        help="训练期每个评测 episode 的步数上限；默认 30",
    )
    train.add_argument(
        "--eval-every-epochs",
        type=int,
        help="每完成多少个 epoch 做一次裸模型 SR；默认每 1 个",
    )
    train.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "按固定 AndroidWorld 子集的 standalone Micro SR 早退（默认读取 TOML，"
            "默认启用）；--no-early-stopping 可关闭"
        ),
    )
    train.add_argument(
        "--early-stopping-patience",
        type=int,
        help="连续多少个 epoch 未显著提升后停止，默认 3",
    )
    train.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help="显著提升的绝对 SR 阈值；0.01 表示 1 个百分点",
    )
    train.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        help="永久保留 checkpoint 的 epoch 间隔；默认 1，0 表示只保留最终结果",
    )
    train.add_argument(
        "--eval-include-candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="模型+技能库评测时也允许 candidate 技能参与灰度",
    )
    train.add_argument("--eval-output-dir", help="本次训练、checkpoint 和评测总目录")
    train.add_argument("--eval-deploy-port", type=int, help="临时 ms-swift 服务端口")
    train.add_argument(
        "--eval-cuda-visible-devices",
        help=(
            "评测 swift deploy 使用的物理 GPU，例如 1；"
            "覆盖 [training_evaluation].cuda_visible_devices"
        ),
    )
    train.add_argument(
        "--eval-max-model-len",
        type=int,
        help="评测 vLLM 最大上下文；默认 32768，用于限制 KV cache",
    )
    train.add_argument(
        "--eval-gpu-memory-utilization",
        type=float,
        help="评测 vLLM 可使用的显存比例，范围 (0,1]",
    )
    train.add_argument(
        "--eval-infer-backend",
        choices=["vllm", "transformers", "sglang", "lmdeploy"],
        help="临时模型服务推理后端",
    )
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

    standalone = subparsers.add_parser(
        "evaluate-standalone",
        help="单 adapter 裸模型 + 原生 M3A AndroidWorld 评测",
    )
    standalone.add_argument(
        "--adapter-path",
        required=True,
        help="adapter 顶层目录；自动选择最新 training_run/最后 epoch/best",
    )
    standalone.add_argument("--model-id", help="报告中的 adapter ID；默认取目录名")
    _add_common_evaluation_arguments(standalone)
    standalone.set_defaults(handler=command_evaluate_standalone)

    simple = subparsers.add_parser(
        "evaluate-simple-skills",
        help="单 adapter + 简单关键词技能调用 AndroidWorld 评测",
    )
    simple.add_argument(
        "--adapter-path",
        required=True,
        help="adapter 顶层目录；自动选择最新 training_run/最后 epoch/best",
    )
    simple.add_argument("--model-id", help="报告中的 adapter ID；默认取目录名")
    _add_common_evaluation_arguments(simple)
    _add_skill_evaluation_arguments(simple)
    simple.set_defaults(handler=command_evaluate_simple_skills)

    evaluate = subparsers.add_parser(
        "evaluate-pmtskill",
        aliases=["evaluate"],
        help="多 adapter + PMT-Skill 动态模型/技能路由 AndroidWorld 评测",
    )
    evaluate.add_argument(
        "--adapter-paths",
        nargs="+",
        required=True,
        help="一个或多个 adapter 顶层目录",
    )
    evaluate.add_argument(
        "--adapter-model-ids",
        nargs="*",
        help="与 adapter-paths 一一对应的路由 ID；默认使用各目录名",
    )
    _add_common_evaluation_arguments(evaluate)
    _add_skill_evaluation_arguments(evaluate)
    evaluate.add_argument(
        "--planner-model",
        help="第一阶段任务分解模型 ID；可引用本次部署的任一 adapter，默认关键词规划",
    )
    evaluate.add_argument("--routing-success-weight", type=float)
    evaluate.add_argument("--routing-latency-weight", type=float)
    evaluate.add_argument("--routing-switch-weight", type=float)
    evaluate.add_argument("--routing-polished-bonus", type=float)
    evaluate.add_argument("--routing-degradation-weight", type=float)
    evaluate.add_argument("--routing-minimum-capability", type=float)
    evaluate.add_argument("--routing-maximum-candidates", type=int)
    evaluate.set_defaults(handler=command_evaluate_pmtskill)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    argv_list = list(argv) if argv is not None else list(sys.argv[1:])
    run_logger = CommandRunLogger(
        _resolve_log_root(args),
        command=args.command,
        label=_run_label(args),
        argv=argv_list,
        arguments=vars(args),
        log_level=args.log_level,
    )
    exit_code = 1
    error: BaseException | None = None
    traceback_text: str | None = None
    with run_logger.capture():
        try:
            exit_code = int(args.handler(args))
        except KeyboardInterrupt as exc:
            error = exc
            exit_code = 130
            print("用户中断。", file=sys.stderr)
        except Exception as exc:
            error = exc
            exit_code = 1
            print(f"错误：{exc}", file=sys.stderr)
            traceback_text = traceback.format_exc()
            print(traceback_text, file=sys.stderr, end="")
        finally:
            run_logger.finalize(
                exit_code,
                error=error,
                traceback_text=traceback_text,
            )
    return exit_code
