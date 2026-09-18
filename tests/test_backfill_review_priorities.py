"""Tests for canonical expert-review-priority backfilling."""

from __future__ import annotations

import json
import hashlib
import socket
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from shinka.tools.compat import backfill_review_priorities as tool
from shinka.tools.compat.backfill_review_priorities import backfill, main


@pytest.fixture(autouse=True)
def offline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache reconstruction must not contact or construct an embedding provider."""
    from shinka.embed import embedding

    blocked = Mock(side_effect=AssertionError("Provider/network access is forbidden"))
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(embedding, "EmbeddingClient", blocked)
    monkeypatch.setattr(embedding, "AsyncEmbeddingClient", blocked)


def _create_canonical_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE programs (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                archive_inspiration_ids TEXT,
                top_k_inspiration_ids TEXT,
                generation INTEGER,
                timestamp REAL,
                combined_score REAL,
                correct INTEGER,
                public_metrics TEXT,
                private_metrics TEXT,
                embedding TEXT,
                reasoning_embedding TEXT,
                code_diff TEXT,
                metadata TEXT,
                review_priority_level TEXT DEFAULT 'none',
                review_priority_data TEXT
            )
            """
        )
        rows = [
            (
                "root",
                None,
                0,
                1.0,
                100.0,
                json.dumps([1.0, 0.0]),
                json.dumps([1.0, 0.0]),
                "none",
                None,
            ),
            (
                "child",
                "root",
                1,
                2.0,
                120.0,
                json.dumps([0.0, 1.0]),
                json.dumps([0.8, 0.2]),
                "none",
                None,
            ),
            (
                "preserved",
                "child",
                2,
                3.0,
                121.0,
                json.dumps([-1.0, 0.0]),
                json.dumps([0.0, 1.0]),
                "high",
                json.dumps({"reason": "Existing custom assignment"}),
            ),
        ]
        conn.executemany(
            """
            INSERT INTO programs (
                id, parent_id, archive_inspiration_ids, top_k_inspiration_ids,
                generation, timestamp, combined_score, correct, public_metrics,
                private_metrics, embedding, reasoning_embedding, code_diff,
                metadata, review_priority_level, review_priority_data
            )
            VALUES (?, ?, '[]', '[]', ?, ?, ?, 1, '{}', '{}', ?, ?, NULL, '{}', ?, ?)
            """,
            rows,
        )
        conn.execute(
            "UPDATE programs SET metadata = ?",
            (json.dumps({"patch_description": "Reuse the parent search strategy"}),),
        )
        conn.execute(
            """
            CREATE TABLE review_priority_metrics (
                program_id TEXT PRIMARY KEY,
                score_change REAL,
                dissimilarity_code REAL,
                dissimilarity_reasoning REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _contents(db_path: Path) -> tuple[list[tuple], list[tuple]]:
    """Read complete program rows and cache rows for conservation checks."""
    with closing(sqlite3.connect(db_path)) as conn:
        return (
            conn.execute("SELECT * FROM programs ORDER BY id").fetchall(),
            conn.execute(
                "SELECT * FROM review_priority_metrics ORDER BY program_id"
            ).fetchall(),
        )


def test_backfill_caches_metrics_and_preserves_existing_priorities(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)

    dry_run = backfill(db_path, dry_run=True)
    assert dry_run == {
        "total": 3,
        "metrics": 3,
        "skipped": 1,
        "none": 1,
        "moderate": 1,
        "high": 0,
    }
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM review_priority_metrics").fetchone()[0]
            == 0
        )

    applied = backfill(db_path)
    assert applied == dry_run

    conn = sqlite3.connect(db_path)
    try:
        metrics = conn.execute(
            """
            SELECT program_id, score_change, dissimilarity_code
            FROM review_priority_metrics ORDER BY program_id
            """
        ).fetchall()
        assert len(metrics) == 3
        assert metrics[0] == ("child", 0.2, 1.0)

        priorities = dict(
            conn.execute(
                "SELECT id, review_priority_level FROM programs ORDER BY id"
            ).fetchall()
        )
        assert priorities == {
            "child": "moderate",
            "preserved": "high",
            "root": "none",
        }
    finally:
        conn.close()


def test_metrics_only_preserves_every_program_field_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair missing/stale cache entries without reclassifying any history."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "UPDATE programs SET review_priority_data = ? WHERE id = 'child'",
            ('{"reason": "Keep a custom none decision", "detail": [1, 2]}',),
        )
        conn.execute("INSERT INTO review_priority_metrics VALUES ('child', 99, 99, 99)")
        conn.execute("CREATE TABLE settings (key TEXT, value TEXT)")
        conn.execute("INSERT INTO settings VALUES ('threshold', '0.5')")
    programs_before, _ = _contents(db_path)
    monkeypatch.setattr(
        tool,
        "default_prioritize_for_review",
        Mock(side_effect=AssertionError("Historical programs must not be classified")),
    )

    stats = backfill(db_path, metrics_only=True)

    programs_after, metrics = _contents(db_path)
    assert programs_after == programs_before
    assert stats == {
        "total": 3,
        "metrics": 3,
        "skipped": 3,
        "none": 0,
        "moderate": 0,
        "high": 0,
    }
    assert metrics[0] == pytest.approx(("child", 0.2, 1.0, 1 - 0.8 / 0.68**0.5))
    assert metrics[1] == pytest.approx(("preserved", 1 / 120, 1.0, 1 - 0.2 / 0.68**0.5))
    assert metrics[2] == ("root", None, None, None)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT * FROM settings").fetchall() == [
            ("threshold", "0.5")
        ]
    assert backfill(db_path, metrics_only=True) == stats
    assert _contents(db_path) == (programs_after, metrics)


def test_metrics_only_uses_strict_runtime_predecessors(tmp_path: Path) -> None:
    """Tied peers do not compare; earlier generations outrank timestamp order."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "UPDATE programs SET generation = 0, timestamp = 1, "
            "embedding = '[1, 0]', reasoning_embedding = '[1, 0]' "
            "WHERE id IN ('root', 'child')"
        )
        conn.execute(
            "UPDATE programs SET generation = 1, timestamp = 0.5, "
            "embedding = '[-1, 0]', reasoning_embedding = '[-1, 0]' "
            "WHERE id = 'preserved'"
        )
    backfill(db_path, metrics_only=True)
    metrics = {row[0]: row[1:] for row in _contents(db_path)[1]}
    assert metrics["root"] == (None, None, None)
    assert metrics["child"] == (0.2, None, None)
    assert metrics["preserved"] == pytest.approx((1 / 120, 1.0, 2.0))


@pytest.mark.parametrize(
    ("code", "reasoning", "metadata", "expected_code"),
    [
        ([], [], {"patch_description": "Cache paths"}, None),
        ([1, 0, 0], [1, 0, 0], {"patch_description": "Cache paths"}, 1.0),
        ([0, 1], [0, 0], {"patch_description": "Cache paths"}, 1.0),
        ([0, 1], [0, 1], {"patch_description": "initial program"}, 1.0),
    ],
)
def test_metrics_only_preserves_missing_and_incompatible_metric_semantics(
    tmp_path: Path,
    code: list[float],
    reasoning: list[float],
    metadata: dict[str, str],
    expected_code: float | None,
) -> None:
    """Reasoning needs valid comparable vectors; retain the existing code policy."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "UPDATE programs SET embedding = ?, reasoning_embedding = ?, metadata = ? "
            "WHERE id = 'child'",
            (json.dumps(code), json.dumps(reasoning), json.dumps(metadata)),
        )
    before, _ = _contents(db_path)
    backfill(db_path, metrics_only=True)
    after, metrics = _contents(db_path)
    assert after == before
    assert metrics[0] == ("child", 0.2, expected_code, None)


def test_metrics_only_cli_dry_run_is_read_only(tmp_path: Path) -> None:
    """Dry run keeps database bytes, journal mode, and directory entries intact."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert main([str(db_path), "--metrics-only", "--dry-run"]) == 0
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [db_path.name]
    assert _contents(db_path)[1] == []
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_dry_run_does_not_create_missing_database(tmp_path: Path) -> None:
    """Both programmatic and CLI use reject a nonexistent database."""
    db_path = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        backfill(db_path, dry_run=True, metrics_only=True)
    assert main([str(db_path), "--metrics-only", "--dry-run"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_metrics_only_rejects_force_before_backup_or_write(tmp_path: Path) -> None:
    """Conflicting preservation/reclassification requests cannot touch the source."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    before = db_path.read_bytes()
    with pytest.raises(ValueError, match="cannot be combined"):
        backfill(db_path, metrics_only=True, force=True)
    with pytest.raises(SystemExit) as error:
        main([str(db_path), "--metrics-only", "--force"])
    assert error.value.code == 2
    assert db_path.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [db_path.name]


def test_metrics_only_rolls_back_partial_writes_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later insertion failure restores existing cache rows and releases locks."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("INSERT INTO review_priority_metrics VALUES ('root', 99, 99, 99)")
        conn.execute(
            "CREATE TRIGGER fail_child BEFORE INSERT ON review_priority_metrics "
            "WHEN NEW.program_id = 'child' BEGIN "
            "SELECT RAISE(ABORT, 'simulated cache failure'); END"
        )
    before = _contents(db_path)
    connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def track_connection(*args: object, **kwargs: object) -> sqlite3.Connection:
        """Retain the tool connection to check deterministic cleanup on error."""
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(tool.sqlite3, "connect", track_connection)
        with pytest.raises(sqlite3.IntegrityError, match="simulated cache failure"):
            backfill(db_path, metrics_only=True)
    assert _contents(db_path) == before
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")
    with closing(sqlite3.connect(db_path, timeout=0)) as conn:
        conn.execute("BEGIN IMMEDIATE")


def test_force_retains_existing_reclassification_behavior(tmp_path: Path) -> None:
    """Force still resets historical assignments that no longer meet defaults."""
    db_path = tmp_path / "programs.sqlite"
    _create_canonical_database(db_path)
    stats = backfill(db_path, force=True)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute(
            "SELECT review_priority_level, review_priority_data FROM programs "
            "WHERE id = 'preserved'"
        ).fetchone() == ("none", None)
    assert stats["metrics"] == 3
    assert stats["skipped"] == 0
