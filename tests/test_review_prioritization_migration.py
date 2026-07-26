"""Tests for the one-way Expert Review Prioritization migration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shinka.database.review_priority_migration import (
    ReviewPrioritizationMigrationError,
    migrate_review_prioritization,
)


def _create_legacy_database(
    db_path: Path,
    *,
    include_metrics: bool = True,
    include_settings: bool = True,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE programs (
                id TEXT PRIMARY KEY,
                novelty_level TEXT DEFAULT 'none',
                novelty_data TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO programs (id, novelty_level, novelty_data) VALUES (?, ?, ?)",
            ("p1", "high", json.dumps({"reason": "large gain"})),
        )
        conn.execute(
            """
            CREATE TABLE interactive_status (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO interactive_status (key, value, updated_at)
            VALUES ('novelty_settings', ?, 10.0)
            """,
            (json.dumps({"mode": "dissimilarity", "source": "status"}),),
        )
        conn.execute("CREATE TABLE metadata_store (key TEXT PRIMARY KEY, value TEXT)")
        if include_metrics:
            conn.execute(
                """
                CREATE TABLE novelty_cache (
                    program_id TEXT PRIMARY KEY,
                    score_change REAL,
                    dissimilarity_code REAL,
                    dissimilarity_reasoning REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO novelty_cache
                (program_id, score_change, dissimilarity_code, dissimilarity_reasoning)
                VALUES ('p1', 0.42, 0.25, 0.75)
                """
            )
        if include_settings:
            conn.execute(
                "CREATE TABLE novelty_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO novelty_settings (key, value) VALUES (?, ?)",
                ("mode", json.dumps("score_change")),
            )
            conn.execute(
                "INSERT INTO novelty_settings (key, value) VALUES (?, ?)",
                ("score_change_high", json.dumps(0.4)),
            )
        conn.commit()
    finally:
        conn.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table_name}")')}


def test_complete_legacy_schema_is_migrated_and_preserved(tmp_path: Path) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)
    (tmp_path / "novelty_error.json").write_text('{"error": "old"}')

    report = migrate_review_prioritization(db_path)

    assert report.changed
    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert not (tmp_path / "novelty_error.json").exists()
    assert (tmp_path / "review_prioritization_error.json").exists()

    conn = sqlite3.connect(db_path)
    try:
        assert "novelty_level" not in _columns(conn, "programs")
        assert {
            "review_priority_level",
            "review_priority_data",
        } <= _columns(conn, "programs")
        row = conn.execute(
            """
            SELECT review_priority_level, review_priority_data
            FROM programs WHERE id = 'p1'
            """
        ).fetchone()
        assert row == ("high", json.dumps({"reason": "large gain"}))

        assert "novelty_cache" not in _table_names(conn)
        assert "review_priority_metrics" in _table_names(conn)
        metrics = conn.execute(
            """
            SELECT score_change, dissimilarity_code, dissimilarity_reasoning
            FROM review_priority_metrics WHERE program_id = 'p1'
            """
        ).fetchone()
        assert metrics == (0.42, 0.25, 0.75)
        assert "novelty_settings" not in _table_names(conn)
        assert (
            conn.execute(
                "SELECT value FROM metadata_store "
                "WHERE key = 'review_prioritization_schema_version'"
            ).fetchone()[0]
            == "1"
        )
    finally:
        conn.close()


def test_standalone_settings_take_precedence(tmp_path: Path) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)

    migrate_review_prioritization(db_path, create_backup=False)

    conn = sqlite3.connect(db_path)
    try:
        raw_settings = conn.execute(
            """
            SELECT value FROM interactive_status
            WHERE key = 'review_prioritization_settings'
            """
        ).fetchone()[0]
        assert json.loads(raw_settings) == {
            "mode": "score_change",
            "score_change_high": 0.4,
        }
        assert (
            conn.execute(
                "SELECT 1 FROM interactive_status WHERE key = 'novelty_settings'"
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_missing_legacy_metrics_and_settings_create_canonical_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(
        db_path,
        include_metrics=False,
        include_settings=False,
    )

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM interactive_status WHERE key = 'novelty_settings'")
    conn.commit()
    conn.close()

    migrate_review_prioritization(db_path, create_backup=False)

    conn = sqlite3.connect(db_path)
    try:
        assert "review_priority_metrics" in _table_names(conn)
        assert (
            conn.execute(
                """
                SELECT 1 FROM interactive_status
                WHERE key = 'review_prioritization_settings'
                """
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_legacy_interactive_settings_are_moved_when_table_is_absent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path, include_settings=False)

    migrate_review_prioritization(db_path, create_backup=False)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT value, updated_at FROM interactive_status
            WHERE key = 'review_prioritization_settings'
            """
        ).fetchone()
        assert json.loads(row[0]) == {
            "mode": "dissimilarity",
            "source": "status",
        }
        assert row[1] == 10.0
    finally:
        conn.close()


def test_dry_run_does_not_modify_or_back_up_database(tmp_path: Path) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)

    report = migrate_review_prioritization(db_path, dry_run=True)

    assert report.changed
    assert report.backup_path is None
    assert not list(tmp_path.glob("programs.sqlite.bak*"))
    conn = sqlite3.connect(db_path)
    try:
        assert "novelty_level" in _columns(conn, "programs")
        assert "novelty_cache" in _table_names(conn)
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)
    migrate_review_prioritization(db_path, create_backup=False)

    report = migrate_review_prioritization(db_path, create_backup=False)

    assert not report.changed
    assert report.actions == []


@pytest.mark.parametrize(
    ("conflict_sql", "expected"),
    [
        (
            "ALTER TABLE programs ADD COLUMN review_priority_level TEXT",
            "columns coexist",
        ),
        (
            """
            CREATE TABLE review_priority_metrics (
                program_id TEXT PRIMARY KEY,
                score_change REAL,
                dissimilarity_code REAL,
                dissimilarity_reasoning REAL
            )
            """,
            "tables coexist",
        ),
    ],
)
def test_conflicting_legacy_and_canonical_schema_aborts(
    tmp_path: Path,
    conflict_sql: str,
    expected: str,
) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(conflict_sql)
    conn.commit()
    conn.close()

    with pytest.raises(ReviewPrioritizationMigrationError, match=expected):
        migrate_review_prioritization(db_path)

    assert not list(tmp_path.glob("programs.sqlite.bak*"))


def test_backup_contains_pre_migration_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_legacy_database(db_path)

    report = migrate_review_prioritization(db_path)

    assert report.backup_path is not None
    backup = sqlite3.connect(report.backup_path)
    try:
        assert "novelty_level" in _columns(backup, "programs")
        assert "novelty_cache" in _table_names(backup)
    finally:
        backup.close()
