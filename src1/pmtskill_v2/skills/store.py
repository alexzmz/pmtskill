"""SQLite 技能图与运行轨迹持久化。

SQLite 适合当前单机/单设备原型：不需要额外服务，支持事务、WAL 和并发读。
未来接入远端 backend 时只需实现相同方法，不影响路由器。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..core.models import (
    ExecutionTrace,
    ModelProfile,
    SkillRecord,
    SkillStatus,
    utc_now,
)
from .identity import (
    model_variant_hash,
    model_variant_id,
    skill_definition_hash,
    skill_version_id,
)


class SkillStore:
    """技能、模型画像、统计与轨迹的统一仓库。"""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """提供自动提交/回滚的短事务。"""

        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """幂等创建数据库表和查询索引。"""

        schema = """
        CREATE TABLE IF NOT EXISTS skills (
            skill_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            level INTEGER NOT NULL,
            source_hash TEXT,
            record_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_source_hash
            ON skills(source_hash) WHERE source_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_skills_status_kind
            ON skills(status, kind);

        CREATE TABLE IF NOT EXISTS skill_versions (
            skill_version_id TEXT PRIMARY KEY,
            skill_id TEXT NOT NULL,
            definition_hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(skill_id, definition_hash),
            FOREIGN KEY(skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_skill_versions_skill
            ON skill_versions(skill_id, created_at);

        CREATE TABLE IF NOT EXISTS skill_metrics (
            skill_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            successes INTEGER NOT NULL DEFAULT 0,
            trials INTEGER NOT NULL DEFAULT 0,
            latency_sum_ms REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(skill_id, model_id),
            FOREIGN KEY(skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS model_profiles (
            model_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_variants (
            model_variant_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            variant_hash TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(model_id, variant_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_model_variants_model
            ON model_variants(model_id, created_at);

        CREATE TABLE IF NOT EXISTS calibration_runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            protocol TEXT NOT NULL,
            config_json TEXT NOT NULL,
            output_dir TEXT,
            summary_json TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT
        );

        CREATE TABLE IF NOT EXISTS skill_calibration_trials (
            trial_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            skill_version_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_variant_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            task_seed TEXT NOT NULL,
            baseline_success INTEGER NOT NULL,
            skill_success INTEGER NOT NULL,
            baseline_steps INTEGER NOT NULL DEFAULT 0,
            skill_steps INTEGER NOT NULL DEFAULT 0,
            baseline_run_time_ms REAL NOT NULL DEFAULT 0,
            skill_run_time_ms REAL NOT NULL DEFAULT 0,
            baseline_parse_rate REAL NOT NULL DEFAULT 0,
            skill_parse_rate REAL NOT NULL DEFAULT 0,
            valid INTEGER NOT NULL DEFAULT 1,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, model_variant_id, skill_version_id, task_name, task_seed),
            FOREIGN KEY(run_id) REFERENCES calibration_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY(skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE,
            FOREIGN KEY(skill_version_id) REFERENCES skill_versions(skill_version_id),
            FOREIGN KEY(model_variant_id) REFERENCES model_variants(model_variant_id)
        );
        CREATE INDEX IF NOT EXISTS idx_calibration_pair
            ON skill_calibration_trials(skill_id, model_id, model_variant_id, valid);
        CREATE INDEX IF NOT EXISTS idx_calibration_run
            ON skill_calibration_trials(run_id, model_variant_id, skill_version_id);

        CREATE TABLE IF NOT EXISTS skill_model_status (
            skill_id TEXT NOT NULL,
            skill_version_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_variant_id TEXT NOT NULL,
            status TEXT NOT NULL,
            reason_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(skill_version_id, model_variant_id),
            FOREIGN KEY(skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE,
            FOREIGN KEY(skill_version_id) REFERENCES skill_versions(skill_version_id),
            FOREIGN KEY(model_variant_id) REFERENCES model_variants(model_variant_id)
        );
        CREATE INDEX IF NOT EXISTS idx_skill_model_status_lookup
            ON skill_model_status(skill_id, model_id, status);

        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            successful INTEGER NOT NULL,
            trace_json TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_traces_processed
            ON traces(processed, successful, created_at);

        CREATE TABLE IF NOT EXISTS skill_sequence_evidence (
            sequence_hash TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            primitives_json TEXT NOT NULL,
            successful INTEGER NOT NULL,
            is_full_sequence INTEGER NOT NULL DEFAULT 0,
            task_name TEXT NOT NULL,
            source_kind TEXT,
            source_skill_id TEXT,
            trajectory_schema TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(sequence_hash, trace_id),
            FOREIGN KEY(trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_sequence_evidence_lookup
            ON skill_sequence_evidence(sequence_hash, successful, created_at);

        CREATE TABLE IF NOT EXISTS maintenance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            subject_id TEXT,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        with self.transaction() as connection:
            connection.executescript(schema)

    def upsert_skill(self, skill: SkillRecord) -> bool:
        """新增或更新技能；返回是否为首次插入。"""

        payload = json.dumps(skill.to_dict(), ensure_ascii=False)
        with self.transaction() as connection:
            existed = connection.execute(
                "SELECT 1 FROM skills WHERE skill_id = ?", (skill.skill_id,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO skills(skill_id, name, kind, status, level, source_hash,
                                   record_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    name=excluded.name,
                    kind=excluded.kind,
                    status=excluded.status,
                    level=excluded.level,
                    source_hash=excluded.source_hash,
                    record_json=excluded.record_json,
                    updated_at=excluded.updated_at
                """,
                (
                    skill.skill_id,
                    skill.name,
                    skill.kind,
                    skill.status.value,
                    skill.level,
                    skill.source_hash,
                    payload,
                    skill.updated_at,
                ),
            )
            definition_hash = skill_definition_hash(skill)
            connection.execute(
                """
                INSERT OR IGNORE INTO skill_versions(
                    skill_version_id, skill_id, definition_hash, version,
                    record_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_version_id(skill),
                    skill.skill_id,
                    definition_hash,
                    skill.version,
                    payload,
                    utc_now(),
                ),
            )
        return existed is None

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT record_json FROM skills WHERE skill_id = ?", (skill_id,)
            ).fetchone()
        return SkillRecord.from_dict(json.loads(row[0])) if row else None

    def find_skill_by_source_hash(self, source_hash: str) -> SkillRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT record_json FROM skills WHERE source_hash = ?", (source_hash,)
            ).fetchone()
        return SkillRecord.from_dict(json.loads(row[0])) if row else None

    def list_skills(
        self,
        *,
        status: SkillStatus | str | None = None,
        kind: str | None = None,
    ) -> list[SkillRecord]:
        """按生命周期和类型筛选技能。"""

        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value if isinstance(status, SkillStatus) else status)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        sql = "SELECT record_json FROM skills"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY level DESC, name"
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, values).fetchall()
        return [SkillRecord.from_dict(json.loads(row[0])) for row in rows]

    def set_skill_status(self, skill_id: str, status: SkillStatus) -> None:
        """原子更新索引列和 JSON，保证两者永远一致。"""

        skill = self.get_skill(skill_id)
        if skill is None:
            raise KeyError(f"技能不存在: {skill_id}")
        skill.status = status
        skill.updated_at = utc_now()
        self.upsert_skill(skill)

    def upsert_model_profile(self, profile: ModelProfile) -> None:
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO model_profiles(model_id, profile_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at
                """,
                (profile.model_id, payload, utc_now()),
            )

    def register_skill_version(self, skill: SkillRecord) -> str:
        """登记当前技能定义并返回版本 ID；重复调用不会产生重复版本。"""

        version_id = skill_version_id(skill)
        payload = json.dumps(skill.to_dict(), ensure_ascii=False)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO skill_versions(
                    skill_version_id, skill_id, definition_hash, version,
                    record_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    skill.skill_id,
                    skill_definition_hash(skill),
                    skill.version,
                    payload,
                    utc_now(),
                ),
            )
        return version_id

    def register_model_variant(self, profile: ModelProfile) -> str:
        """登记实际 base/adapter 部署版本，避免同名 LoRA 的历史指标互相污染。"""

        variant_id = model_variant_id(profile)
        payload = json.dumps(profile.to_dict(), ensure_ascii=False, default=str)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO model_variants(
                    model_variant_id, model_id, variant_hash, profile_json, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    variant_id,
                    profile.model_id,
                    model_variant_hash(profile),
                    payload,
                    utc_now(),
                ),
            )
        return variant_id

    def start_calibration_run(
        self,
        run_id: str,
        config: Mapping[str, Any],
        output_dir: str | Path,
        *,
        protocol: str = "skvm-paired-android-world-v1",
    ) -> None:
        """创建或恢复一次标定运行；配置原样保存，便于审计和复现实验。"""

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO calibration_runs(
                    run_id, status, protocol, config_json, output_dir,
                    summary_json, started_at, finished_at
                ) VALUES(?, 'running', ?, ?, ?, NULL, ?, NULL)
                ON CONFLICT(run_id) DO UPDATE SET
                    status='running',
                    protocol=excluded.protocol,
                    config_json=excluded.config_json,
                    output_dir=excluded.output_dir,
                    finished_at=NULL
                """,
                (
                    run_id,
                    protocol,
                    json.dumps(dict(config), ensure_ascii=False, default=str),
                    str(Path(output_dir).resolve()),
                    utc_now(),
                ),
            )

    def finish_calibration_run(
        self, run_id: str, *, status: str, summary: Mapping[str, Any]
    ) -> None:
        """结束标定运行；失败运行同样保留已经完成的 trial，允许后续恢复。"""

        with self.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE calibration_runs
                SET status = ?, summary_json = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    json.dumps(dict(summary), ensure_ascii=False, default=str),
                    utc_now(),
                    run_id,
                ),
            ).rowcount
        if not updated:
            raise KeyError(f"标定运行不存在: {run_id}")

    def record_skill_calibration_trial(
        self,
        *,
        run_id: str,
        pair_id: str,
        skill: SkillRecord,
        profile: ModelProfile,
        task_name: str,
        task_seed: str | int | None,
        baseline_success: bool,
        skill_success: bool,
        baseline_steps: int = 0,
        skill_steps: int = 0,
        baseline_run_time_ms: float = 0.0,
        skill_run_time_ms: float = 0.0,
        baseline_parse_rate: float = 0.0,
        skill_parse_rate: float = 0.0,
        valid: bool = True,
        detail: Mapping[str, Any] | None = None,
    ) -> str:
        """保存一条 no-skill/forced-skill 配对观测。

        这里不更新旧 ``skill_metrics``。旧表代表在线自然流量；标定数据有强制选择，
        必须独立保存，防止被当成无偏在线样本。
        """

        skill_version = self.register_skill_version(skill)
        model_variant = self.register_model_variant(profile)
        trial_id = uuid.uuid4().hex
        payload = {
            "pair_id": pair_id,
            "task_name": task_name,
            "task_seed": task_seed,
            "baseline_success": bool(baseline_success),
            "skill_success": bool(skill_success),
            "baseline_steps": max(0, int(baseline_steps)),
            "skill_steps": max(0, int(skill_steps)),
            "baseline_run_time_ms": max(0.0, float(baseline_run_time_ms)),
            "skill_run_time_ms": max(0.0, float(skill_run_time_ms)),
            "baseline_parse_rate": min(1.0, max(0.0, float(baseline_parse_rate))),
            "skill_parse_rate": min(1.0, max(0.0, float(skill_parse_rate))),
            "valid": bool(valid),
            "detail": dict(detail or {}),
        }
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO skill_calibration_trials(
                    trial_id, run_id, pair_id, skill_id, skill_version_id,
                    model_id, model_variant_id, task_name, task_seed,
                    baseline_success, skill_success, baseline_steps, skill_steps,
                    baseline_run_time_ms, skill_run_time_ms,
                    baseline_parse_rate, skill_parse_rate, valid,
                    detail_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    run_id, model_variant_id, skill_version_id, task_name, task_seed
                ) DO UPDATE SET
                    trial_id=excluded.trial_id,
                    pair_id=excluded.pair_id,
                    baseline_success=excluded.baseline_success,
                    skill_success=excluded.skill_success,
                    baseline_steps=excluded.baseline_steps,
                    skill_steps=excluded.skill_steps,
                    baseline_run_time_ms=excluded.baseline_run_time_ms,
                    skill_run_time_ms=excluded.skill_run_time_ms,
                    baseline_parse_rate=excluded.baseline_parse_rate,
                    skill_parse_rate=excluded.skill_parse_rate,
                    valid=excluded.valid,
                    detail_json=excluded.detail_json,
                    created_at=excluded.created_at
                """,
                (
                    trial_id,
                    run_id,
                    pair_id,
                    skill.skill_id,
                    skill_version,
                    profile.model_id,
                    model_variant,
                    task_name,
                    str(task_seed if task_seed is not None else "unknown"),
                    int(payload["baseline_success"]),
                    int(payload["skill_success"]),
                    payload["baseline_steps"],
                    payload["skill_steps"],
                    payload["baseline_run_time_ms"],
                    payload["skill_run_time_ms"],
                    payload["baseline_parse_rate"],
                    payload["skill_parse_rate"],
                    int(payload["valid"]),
                    json.dumps(payload["detail"], ensure_ascii=False, default=str),
                    utc_now(),
                ),
            )
        return trial_id

    def list_calibration_trials(
        self,
        *,
        run_id: str | None = None,
        skill_id: str | None = None,
        model_id: str | None = None,
        model_variant_id_value: str | None = None,
    ) -> list[dict[str, Any]]:
        """读取逐次标定结果；保留无效 trial 供排查基础设施问题。"""

        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("run_id", run_id),
            ("skill_id", skill_id),
            ("model_id", model_id),
            ("model_variant_id", model_variant_id_value),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        sql = "SELECT * FROM skill_calibration_trials"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY model_id, skill_id, task_name, task_seed, created_at"
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, values).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["baseline_success"] = bool(item["baseline_success"])
            item["skill_success"] = bool(item["skill_success"])
            item["valid"] = bool(item["valid"])
            item["detail"] = json.loads(item.pop("detail_json") or "{}")
            result.append(item)
        return result

    def calibration_metrics(
        self,
        skill_id: str,
        *,
        model_id: str | None = None,
        model_variant_id_value: str | None = None,
        run_id: str | None = None,
        current_skill_version_only: bool = True,
    ) -> dict[str, Any]:
        """汇总配对标定指标，并明确区分未标定与真实 0% 成功率。"""

        rows = self.list_calibration_trials(
            run_id=run_id,
            skill_id=skill_id,
            model_id=model_id,
            model_variant_id_value=model_variant_id_value,
        )
        current_version: str | None = None
        if current_skill_version_only:
            skill = self.get_skill(skill_id)
            current_version = skill_version_id(skill) if skill is not None else None
            rows = [row for row in rows if row["skill_version_id"] == current_version]
        valid = [row for row in rows if row["valid"]]
        trials = len(valid)
        skill_successes = sum(bool(row["skill_success"]) for row in valid)
        baseline_successes = sum(bool(row["baseline_success"]) for row in valid)
        wins = sum(
            bool(row["skill_success"]) and not bool(row["baseline_success"])
            for row in valid
        )
        losses = sum(
            bool(row["baseline_success"]) and not bool(row["skill_success"])
            for row in valid
        )
        ties = trials - wins - losses

        def average(key: str) -> float:
            return (
                sum(float(row[key]) for row in valid) / trials if trials else 0.0
            )

        skill_rate = skill_successes / trials if trials else 0.0
        baseline_rate = baseline_successes / trials if trials else 0.0
        return {
            "known": trials > 0,
            "skill_id": skill_id,
            "skill_version_id": current_version,
            "model_id": model_id,
            "model_variant_id": model_variant_id_value,
            "trials_total": len(rows),
            "valid_trials": trials,
            "invalid_trials": len(rows) - trials,
            "skill_successes": skill_successes,
            "baseline_successes": baseline_successes,
            "skill_success_rate": skill_rate,
            "baseline_success_rate": baseline_rate,
            "success_rate_uplift": skill_rate - baseline_rate,
            "smoothed_skill_success_rate": (skill_successes + 1) / (trials + 2),
            "skill_success_wilson_lower": wilson_lower_bound(
                skill_successes, trials
            ),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "average_baseline_steps": average("baseline_steps"),
            "average_skill_steps": average("skill_steps"),
            "average_baseline_run_time_ms": average("baseline_run_time_ms"),
            "average_skill_run_time_ms": average("skill_run_time_ms"),
            "average_baseline_parse_rate": average("baseline_parse_rate"),
            "average_skill_parse_rate": average("skill_parse_rate"),
        }

    def calibration_metrics_by_model(self, skill_id: str) -> list[dict[str, Any]]:
        """返回当前技能版本的每个实际模型版本指标，而不是跨模型汇总。"""

        skill = self.get_skill(skill_id)
        if skill is None:
            return []
        current_version = skill_version_id(skill)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT model_id, model_variant_id
                FROM skill_calibration_trials
                WHERE skill_id = ? AND skill_version_id = ?
                ORDER BY model_id, model_variant_id
                """,
                (skill_id, current_version),
            ).fetchall()
        return [
            self.calibration_metrics(
                skill_id,
                model_id=str(row["model_id"]),
                model_variant_id_value=str(row["model_variant_id"]),
            )
            for row in rows
        ]

    def set_skill_model_status(
        self,
        skill: SkillRecord,
        *,
        model_id: str,
        model_variant_id_value: str,
        status: SkillStatus,
        reason: Mapping[str, Any],
    ) -> None:
        """更新某一技能版本对某一模型版本的独立生命周期状态。"""

        version_id = self.register_skill_version(skill)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO skill_model_status(
                    skill_id, skill_version_id, model_id, model_variant_id,
                    status, reason_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_version_id, model_variant_id) DO UPDATE SET
                    status=excluded.status,
                    reason_json=excluded.reason_json,
                    updated_at=excluded.updated_at
                """,
                (
                    skill.skill_id,
                    version_id,
                    model_id,
                    model_variant_id_value,
                    status.value,
                    json.dumps(dict(reason), ensure_ascii=False, default=str),
                    utc_now(),
                ),
            )

    def skill_model_status(
        self, skill: SkillRecord, profile: ModelProfile
    ) -> dict[str, Any] | None:
        """查询当前 skill/model 版本的标定状态；无记录返回 None，而不是失败。"""

        version_id = skill_version_id(skill)
        variant_id = model_variant_id(profile)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT status, reason_json, updated_at
                FROM skill_model_status
                WHERE skill_version_id = ? AND model_variant_id = ?
                """,
                (version_id, variant_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "status": str(row["status"]),
            "reason": json.loads(row["reason_json"] or "{}"),
            "updated_at": str(row["updated_at"]),
            "skill_version_id": version_id,
            "model_variant_id": variant_id,
        }

    def has_current_skill_model_status(self, skill: SkillRecord) -> bool:
        """当前技能版本是否已经开始构建模型兼容矩阵。"""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM skill_model_status WHERE skill_version_id = ? LIMIT 1",
                (skill_version_id(skill),),
            ).fetchone()
        return row is not None

    def has_any_skill_model_status(self, skill_id: str) -> bool:
        """技能是否曾进入版本化兼容矩阵，用于识别更新后尚未重标定的版本。"""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM skill_model_status WHERE skill_id = ? LIMIT 1",
                (skill_id,),
            ).fetchone()
        return row is not None

    def list_skill_model_statuses(
        self,
        skill_id: str | None = None,
        *,
        current_skill_version_only: bool = False,
    ) -> list[dict[str, Any]]:
        """供 CLI/报告展示所有模型专属技能状态。"""

        clauses: list[str] = []
        values: list[Any] = []
        if skill_id is not None:
            clauses.append("skill_id = ?")
            values.append(skill_id)
        if current_skill_version_only:
            if skill_id is None:
                raise ValueError(
                    "current_skill_version_only=True 时必须同时提供 skill_id"
                )
            skill = self.get_skill(skill_id)
            if skill is None:
                return []
            clauses.append("skill_version_id = ?")
            values.append(skill_version_id(skill))
        sql = "SELECT * FROM skill_model_status"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY skill_id, model_id, model_variant_id"
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["reason"] = json.loads(item.pop("reason_json") or "{}")
            result.append(item)
        return result

    def rollback_raw_skill_compile(self, skill_id: str) -> SkillRecord:
        """
        将一个经过 raw-skill compiler 的 skill
        恢复到编译前的 raw-skill 状态。

        返回回溯后的 SkillRecord。
        """

        skill = self.get_skill(skill_id)
        if skill is None:
            raise KeyError(f"技能不存在: {skill_id}")

        if skill.kind != "raw":
            raise ValueError(f"技能 {skill_id} 不是 raw skill，当前 kind={skill.kind}")

        if not skill.metadata.get("raw_skill_compiled"):
            raise ValueError(f"技能 {skill_id} 尚未经过 raw skill compile")

        # 1. 恢复 compiler 可能修改的正文
        original_body = skill.metadata.get("original_body")
        if original_body is not None:
            skill.body = original_body

        # 2. 恢复 topology
        skill.topology = None

        # 3. 清除 compiler 产生的 metadata
        compile_metadata_keys = (
            "raw_skill_compiler",
            "raw_skill_compiler_model",
            "raw_skill_compile_reason",
            "raw_skill_compiled",
            "approved_for_planning",
            "original_body",
            "original_topology",
        )

        for key in compile_metadata_keys:
            skill.metadata.pop(key, None)

        # 4. 强制保证还是 raw
        skill.kind = "raw"
        skill.updated_at = utc_now()

        # 5. 写回同一条记录
        self.upsert_skill(skill)

        # 6. 留 maintenance log
        self.log_maintenance_event(
            "raw_skill_compile_rollback",
            skill.skill_id,
            {
                "skill_name": skill.name,
            },
        )

        return skill

    def list_model_profiles(self, *, enabled_only: bool = True) -> list[ModelProfile]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT profile_json FROM model_profiles ORDER BY model_id"
            ).fetchall()
        profiles = [ModelProfile.from_dict(json.loads(row[0])) for row in rows]
        return (
            [profile for profile in profiles if profile.enabled]
            if enabled_only
            else profiles
        )

    def record_skill_trial(
        self, skill_id: str, model_id: str, success: bool, latency_ms: float
    ) -> None:
        """记录 polished/raw skill 在某模型上的一次真实执行结果。"""

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO skill_metrics(
                    skill_id, model_id, successes, trials, latency_sum_ms, updated_at
                ) VALUES(?, ?, ?, 1, ?, ?)
                ON CONFLICT(skill_id, model_id) DO UPDATE SET
                    successes=successes + excluded.successes,
                    trials=trials + 1,
                    latency_sum_ms=latency_sum_ms + excluded.latency_sum_ms,
                    updated_at=excluded.updated_at
                """,
                (skill_id, model_id, int(success), max(0.0, latency_ms), utc_now()),
            )

    def skill_metrics(
        self, skill_id: str, model_id: str | None = None
    ) -> dict[str, Any]:
        """汇总技能统计；Beta(1,1) 平滑避免小样本得到 0 或 1。"""

        sql = "SELECT SUM(successes), SUM(trials), SUM(latency_sum_ms) FROM skill_metrics WHERE skill_id = ?"
        values: list[Any] = [skill_id]
        if model_id is not None:
            sql += " AND model_id = ?"
            values.append(model_id)
        with closing(self._connect()) as connection:
            row = connection.execute(sql, values).fetchone()
        successes = int(row[0] or 0)
        trials = int(row[1] or 0)
        latency_sum = float(row[2] or 0.0)
        return {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials if trials else 0.0,
            "smoothed_success_rate": (successes + 1) / (trials + 2),
            "average_latency_ms": latency_sum / trials if trials else 0.0,
        }

    def append_trace(
        self,
        trace: ExecutionTrace,
        *,
        update_skill_metrics: bool | None = None,
    ) -> None:
        """幂等保存一条设备轨迹，并按数据作用域决定是否累计在线统计。

        ``skill_metrics`` 表示模型在在线自然流量中调用技能的观测，不应被 Teacher
        生成的技能发现数据污染。调用方可以显式传入 ``False``；未指定时读取 trace
        metadata 的 ``metric_scope``，``skill_discovery``/``maintenance_only`` 会只
        保存轨迹。旧轨迹没有该字段，继续保持原有的统计行为。
        """

        payload = json.dumps(trace.to_dict(), ensure_ascii=False)
        if update_skill_metrics is None:
            update_skill_metrics = str(
                trace.metadata.get("metric_scope", "online")
            ) not in {"skill_discovery", "maintenance_only"}
        with self.transaction() as connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO traces(
                    trace_id, task_name, successful, trace_json, processed, created_at
                ) VALUES(?, ?, ?, ?, 0, ?)
                """,
                (
                    trace.trace_id,
                    trace.task_name,
                    int(trace.successful),
                    payload,
                    trace.created_at,
                ),
            ).rowcount
            if inserted and update_skill_metrics:
                for event in trace.events:
                    if not event.skill_id:
                        continue
                    known_skill = connection.execute(
                        "SELECT 1 FROM skills WHERE skill_id = ?", (event.skill_id,)
                    ).fetchone()
                    # 允许设备先上传含新技能 ID 的轨迹；待技能元数据同步后再统计。
                    if known_skill is None:
                        continue
                    connection.execute(
                        """
                        INSERT INTO skill_metrics(
                            skill_id, model_id, successes, trials, latency_sum_ms, updated_at
                        ) VALUES(?, ?, ?, 1, ?, ?)
                        ON CONFLICT(skill_id, model_id) DO UPDATE SET
                            successes=successes + excluded.successes,
                            trials=trials + 1,
                            latency_sum_ms=latency_sum_ms + excluded.latency_sum_ms,
                            updated_at=excluded.updated_at
                        """,
                        (
                            event.skill_id,
                            event.model_id,
                            int(event.success),
                            max(0.0, event.latency_ms),
                            utc_now(),
                        ),
                    )

    def list_traces(
        self,
        *,
        successful: bool | None = None,
        processed: bool | None = None,
        limit: int | None = None,
    ) -> list[ExecutionTrace]:
        clauses: list[str] = []
        values: list[Any] = []
        if successful is not None:
            clauses.append("successful = ?")
            values.append(int(successful))
        if processed is not None:
            clauses.append("processed = ?")
            values.append(int(processed))
        sql = "SELECT trace_json FROM traces"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            values.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, values).fetchall()
        return [ExecutionTrace.from_dict(json.loads(row[0])) for row in rows]

    def mark_traces_processed(self, trace_ids: list[str]) -> None:
        if not trace_ids:
            return
        with self.transaction() as connection:
            connection.executemany(
                "UPDATE traces SET processed = 1 WHERE trace_id = ?",
                [(trace_id,) for trace_id in trace_ids],
            )

    def record_sequence_evidence(
        self,
        trace: ExecutionTrace,
        sequences: Mapping[tuple[str, ...], bool],
    ) -> int:
        """幂等保存一条轨迹提供的序列证据。

        ``sequences`` 的 value 表示该序列是否也是该 trace 的完整执行序列。同一
        trace 中重复出现的子序列只计一次，避免长 episode 刷高 support。
        """

        source = trace.metadata.get("collection_source")
        source = dict(source) if isinstance(source, Mapping) else {}
        rows: list[tuple[Any, ...]] = []
        for primitives, is_full in sequences.items():
            canonical = json.dumps(list(primitives), ensure_ascii=False, separators=(",", ":"))
            sequence_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            rows.append(
                (
                    sequence_hash,
                    trace.trace_id,
                    canonical,
                    int(trace.successful),
                    int(is_full),
                    trace.task_name,
                    source.get("kind"),
                    source.get("skill_id"),
                    trace.metadata.get("trajectory_schema"),
                    trace.created_at,
                )
            )
        if not rows:
            return 0
        with self.transaction() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT INTO skill_sequence_evidence(
                    sequence_hash, trace_id, primitives_json, successful,
                    is_full_sequence, task_name, source_kind, source_skill_id,
                    trajectory_schema, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sequence_hash, trace_id) DO UPDATE SET
                    is_full_sequence=MAX(
                        skill_sequence_evidence.is_full_sequence,
                        excluded.is_full_sequence
                    )
                """,
                rows,
            )
            return connection.total_changes - before

    def list_sequence_evidence(self) -> list[dict[str, Any]]:
        """返回技能发现的累计成功/失败证据，供维护算法透明聚合。"""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT sequence_hash, trace_id, primitives_json, successful,
                       is_full_sequence, task_name, source_kind, source_skill_id,
                       trajectory_schema, created_at
                FROM skill_sequence_evidence
                ORDER BY created_at, trace_id, sequence_hash
                """
            ).fetchall()
        return [
            {
                "sequence_hash": str(row["sequence_hash"]),
                "trace_id": str(row["trace_id"]),
                "primitives": tuple(json.loads(row["primitives_json"])),
                "successful": bool(row["successful"]),
                "is_full_sequence": bool(row["is_full_sequence"]),
                "task_name": str(row["task_name"]),
                "source_kind": row["source_kind"],
                "source_skill_id": row["source_skill_id"],
                "trajectory_schema": row["trajectory_schema"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def log_maintenance_event(
        self, event_type: str, subject_id: str | None, detail: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO maintenance_events(event_type, subject_id, detail_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (
                    event_type,
                    subject_id,
                    json.dumps(detail, ensure_ascii=False),
                    utc_now(),
                ),
            )


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """二项分布 Wilson 置信区间下界，比裸成功率更适合晋升判断。"""

    if trials <= 0:
        return 0.0
    probability = successes / trials
    denominator = 1 + z * z / trials
    centre = probability + z * z / (2 * trials)
    margin = z * (
        (probability * (1 - probability) / trials + z * z / (4 * trials**2)) ** 0.5
    )
    return max(0.0, (centre - margin) / denominator)
