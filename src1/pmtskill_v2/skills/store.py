"""SQLite 技能图与运行轨迹持久化。

SQLite 适合当前单机/单设备原型：不需要额外服务，支持事务、WAL 和并发读。
未来接入远端 backend 时只需实现相同方法，不影响路由器。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..core.models import ExecutionTrace, ModelProfile, SkillRecord, SkillStatus, utc_now


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

    def list_model_profiles(self, *, enabled_only: bool = True) -> list[ModelProfile]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT profile_json FROM model_profiles ORDER BY model_id"
            ).fetchall()
        profiles = [ModelProfile.from_dict(json.loads(row[0])) for row in rows]
        return [profile for profile in profiles if profile.enabled] if enabled_only else profiles

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

    def skill_metrics(self, skill_id: str, model_id: str | None = None) -> dict[str, Any]:
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

    def append_trace(self, trace: ExecutionTrace) -> None:
        """幂等保存一条设备轨迹，并同步累计技能统计。"""

        payload = json.dumps(trace.to_dict(), ensure_ascii=False)
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
            if inserted:
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

    def log_maintenance_event(
        self, event_type: str, subject_id: str | None, detail: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO maintenance_events(event_type, subject_id, detail_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (event_type, subject_id, json.dumps(detail, ensure_ascii=False), utc_now()),
            )


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """二项分布 Wilson 置信区间下界，比裸成功率更适合晋升判断。"""

    if trials <= 0:
        return 0.0
    probability = successes / trials
    denominator = 1 + z * z / trials
    centre = probability + z * z / (2 * trials)
    margin = z * ((probability * (1 - probability) / trials + z * z / (4 * trials**2)) ** 0.5)
    return max(0.0, (centre - margin) / denominator)
