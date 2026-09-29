"""三入口技能包、轨迹目录与技能层级关系的回归测试。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src1.pmtskill_v2.cli import build_parser, command_collect_optimization_traces
from src1.pmtskill_v2.core.config import MaintenanceConfig
from src1.pmtskill_v2.core.io import load_primitives
from src1.pmtskill_v2.core.models import (
    ExecutionTrace,
    SkillRecord,
    SkillStatus,
    SkillTopology,
    TraceEvent,
)
from src1.pmtskill_v2.skills.identity import skill_version_id
from src1.pmtskill_v2.skills.importer import (
    import_skill_cluster,
    scan_skill_cluster,
)
from src1.pmtskill_v2.skills.maintenance import SkillMaintainer
from src1.pmtskill_v2.skills.store import SkillStore


class SkillClusterImportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SkillStore(self.root / "library.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    def _make_package(self) -> Path:
        package = self.root / "ceo-advisor"
        (package / "references").mkdir(parents=True)
        (package / "scripts").mkdir()
        (package / "SKILL.md").write_text(
            "---\n"
            "name: CEO Advisor\n"
            "description: >\n"
            "  Open an Android mobile app and\n"
            "  complete the requested workflow.\n"
            "metadata:\n"
            "  owner: test\n"
            "---\n"
            "Use the reference, tap the visible target, and verify completion.\n",
            encoding="utf-8",
        )
        (package / "references" / "guide.md").write_text(
            "Reference marker: inspect the current Android screen first.",
            encoding="utf-8",
        )
        # 如果 importer 错误执行脚本就会产生 sentinel；正确行为只能读取文本。
        (package / "scripts" / "helper.py").write_text(
            "from pathlib import Path\nPath('SHOULD_NOT_EXIST').write_text('bad')\n",
            encoding="utf-8",
        )
        return package

    def test_single_package_root_keeps_resources_and_updates_on_reference_change(self):
        package = self._make_package()
        scanned = scan_skill_cluster(package, namespace="external:ceo-advisor")
        self.assertEqual(len(scanned), 1)
        self.assertEqual(scanned[0].skill_id, "external:ceo-advisor")
        self.assertIn("complete the requested workflow", scanned[0].description)
        self.assertIn("Reference marker", scanned[0].metadata["teacher_source_context"])
        self.assertIn("helper.py", scanned[0].metadata["teacher_source_context"])
        self.assertFalse((self.root / "SHOULD_NOT_EXIST").exists())

        first = import_skill_cluster(
            package, self.store, namespace="external:ceo-advisor"
        )
        second = import_skill_cluster(
            package, self.store, namespace="external:ceo-advisor"
        )
        self.assertEqual(first.inserted, 1)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(second.skill_ids, ["external:ceo-advisor"])

        before = self.store.get_skill("external:ceo-advisor")
        (package / "references" / "guide.md").write_text(
            "Reference marker changed: inspect, tap, then verify.", encoding="utf-8"
        )
        changed = import_skill_cluster(
            package, self.store, namespace="external:ceo-advisor"
        )
        after = self.store.get_skill("external:ceo-advisor")
        self.assertEqual(changed.updated, 1)
        self.assertEqual(after.version, before.version + 1)
        self.assertNotEqual(after.source_hash, before.source_hash)

    def test_cluster_returns_only_exact_batch_ids(self):
        cluster = self.root / "cluster"
        for relative in ("alpha", "nested/beta"):
            package = cluster / relative
            package.mkdir(parents=True)
            (package / "SKILL.md").write_text(
                f"---\nname: {relative}\n---\nAndroid mobile UI tap workflow",
                encoding="utf-8",
            )
        result = import_skill_cluster(
            cluster, self.store, namespace="external:test-cluster"
        )
        self.assertEqual(result.scanned, 2)
        self.assertEqual(
            set(result.skill_ids),
            {"external:test-cluster:alpha", "external:test-cluster:nested-beta"},
        )


class LibrarySchemaTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SkillStore(self.root / "library.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _trace(successful: bool, *, empty: bool = False) -> ExecutionTrace:
        events = () if empty else (
            TraceEvent(
                0,
                "teacher",
                None,
                ("reason.intent", "action.click"),
                successful,
                1.0,
                metadata={"action_committed": True},
            ),
        )
        return ExecutionTrace.new(
            "goal",
            "Task",
            successful,
            events,
            metadata={
                "source": "android_world_m3a",
                "episode_data_valid": True,
                "had_exception": False,
                "collection_source": {"kind": "task", "skill_id": None},
            },
        )

    def test_primitive_catalog_and_trajectory_quality_are_queryable(self):
        sync = self.store.sync_primitive_catalog(load_primitives())
        self.assertEqual(sync["total"], 26)
        self.assertEqual(len(self.store.list_primitive_catalog()), 26)

        excellent = self._trace(True)
        failed = self._trace(False)
        rejected = self._trace(False, empty=True)
        for trace in (excellent, failed, rejected, excellent):
            self.store.append_trace(trace, update_skill_metrics=False)
        self.assertEqual(
            len(self.store.list_trajectory_library(quality_status="excellent")), 1
        )
        self.assertEqual(
            len(self.store.list_trajectory_library(quality_status="failed")), 1
        )
        self.assertEqual(
            len(self.store.list_trajectory_library(quality_status="rejected")), 1
        )

    def test_initialize_backfills_old_trace_database_without_altering_trace(self):
        database = self.root / "legacy.sqlite3"
        trace = self._trace(True)
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                """
                CREATE TABLE traces (
                    trace_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    successful INTEGER NOT NULL,
                    trace_json TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO traces VALUES(?, ?, ?, ?, 0, ?)",
                (
                    trace.trace_id,
                    trace.task_name,
                    int(trace.successful),
                    json.dumps(trace.to_dict()),
                    trace.created_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        migrated = SkillStore(database)
        migrated.initialize()
        rows = migrated.list_trajectory_library()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quality_status"], "excellent")
        self.assertNotIn(
            "trajectory_quality", migrated.list_traces()[0].metadata
        )

    def test_maintain_records_raw_and_polished_lineage_idempotently(self):
        raw = SkillRecord(
            skill_id="raw:input",
            name="raw input",
            description="input",
            kind="raw",
            status=SkillStatus.IMPORTED,
            level=1,
            topology=SkillTopology.from_sequence(("action.click", "action.type")),
        )
        component = SkillRecord(
            skill_id="polished:component:v1",
            name="component",
            description="component",
            kind="polished",
            status=SkillStatus.ACTIVE,
            level=2,
            topology=SkillTopology.from_sequence(("action.click",)),
        )
        self.store.upsert_skill(raw)
        self.store.upsert_skill(component)
        for index in range(2):
            trace = ExecutionTrace.new(
                "goal",
                f"Task{index}",
                True,
                (
                    TraceEvent(
                        0,
                        "teacher",
                        component.skill_id,
                        ("action.click", "action.type"),
                        True,
                        1.0,
                        metadata={
                            "action_committed": True,
                            "skill_version_id": skill_version_id(component),
                        },
                    ),
                ),
                metadata={
                    "source": "android_world_m3a",
                    "episode_data_valid": True,
                    "collection_source": {
                        "kind": "database_skill",
                        "skill_id": raw.skill_id,
                        "skill_version": raw.version,
                        "skill_source_hash": "raw-hash",
                    },
                },
            )
            self.store.append_trace(trace, update_skill_metrics=False)

        maintainer = SkillMaintainer(
            self.store,
            MaintenanceConfig(
                minimum_support=2,
                minimum_subsequence_length=2,
                maximum_subsequence_length=2,
            ),
        )
        first = maintainer.run_cycle()
        self.assertEqual(len(first.candidates_created), 1)
        candidate_id = first.candidates_created[0]
        relations = self.store.list_skill_relations(
            candidate_id, direction="children"
        )
        self.assertEqual(
            {row["relation_type"] for row in relations},
            {"derived_from_raw", "composes_polished"},
        )
        self.assertTrue(all(row["evidence_count"] == 2 for row in relations))

        maintainer.run_cycle()
        repeated = self.store.list_skill_relations(
            candidate_id, direction="children"
        )
        self.assertTrue(all(row["evidence_count"] == 2 for row in repeated))


class ThreeInputCliTest(unittest.TestCase):
    def test_parser_exposes_database_and_cluster_sources(self):
        parser = build_parser()
        database = parser.parse_args(
            ["collect-optimization-traces", "--source", "database-skills"]
        )
        self.assertEqual(database.source, "database-skills")
        cluster = parser.parse_args(
            [
                "collect-optimization-traces",
                "--source",
                "skill-cluster",
                "--skill-root",
                "/tmp/skills",
            ]
        )
        self.assertEqual(cluster.source, "skill-cluster")
        self.assertEqual(cluster.skill_root, "/tmp/skills")
        imported = parser.parse_args(
            ["import-skills", "--skill-root", "/tmp/skills", "--dry-run"]
        )
        self.assertTrue(imported.dry_run)

    def test_database_source_dry_run_executes_handler_without_emulator(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "collect-optimization-traces",
                "--source",
                "database-skills",
                "--dry-run",
                "--output-dir",
                str(Path(tempfile.gettempdir()) / "guided-dry-run"),
            ]
        )
        skill = SkillRecord(
            skill_id="raw:test",
            name="test",
            description="Android test",
            kind="raw",
            status=SkillStatus.IMPORTED,
            level=1,
            topology=SkillTopology.from_sequence(("action.click",)),
        )

        class FakeStore:
            database = Path(tempfile.gettempdir()) / "library.sqlite3"

            @staticmethod
            def list_skills(*, kind=None):
                return [] if kind == "polished" else [skill]

        config = SimpleNamespace(
            offline=SimpleNamespace(teacher_model_id="teacher"),
            model=lambda model_id: SimpleNamespace(model_id=model_id),
        )
        emitted = []
        with patch(
            "src1.pmtskill_v2.cli._open", return_value=(config, FakeStore())
        ), patch("src1.pmtskill_v2.cli._print", side_effect=emitted.append):
            self.assertEqual(command_collect_optimization_traces(args), 0)
        self.assertEqual(emitted[0]["source_kind"], "database_skill")
        self.assertEqual(emitted[0]["source_skills"], ["raw:test"])


if __name__ == "__main__":
    unittest.main()
