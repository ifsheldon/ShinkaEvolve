"""Offline release-copy repair, conservation, and failure-path regressions."""

from __future__ import annotations

import json
import socket
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from shinka.reasoning import minimum_reasoning_distance
from shinka.tools.compat import repair_reasoning_embeddings as repair
from shinka.tools.compat.backfill_reasoning_embeddings import backfill, _backup_db


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every maintenance scenario is local and must stay that way."""

    def blocked(*args: object, **kwargs: object) -> None:
        """Reject every attempted network connection in this test."""
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _database(path: Path, *, wal: bool = False) -> sqlite3.Connection:
    """Create historic data with bad vectors, tied births, and unrelated state."""
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.executescript("""
        CREATE TABLE programs (
            id TEXT PRIMARY KEY, parent_id TEXT, generation INTEGER, timestamp REAL,
            code TEXT, combined_score REAL, metadata TEXT, embedding TEXT,
            reasoning_embedding TEXT, reasoning_embedding_pca_2d TEXT,
            reasoning_embedding_cluster_id INTEGER,
            review_priority_level TEXT, review_priority_data TEXT
        );
        CREATE TABLE review_priority_metrics (
            program_id TEXT PRIMARY KEY REFERENCES programs(id),
            score_change REAL, dissimilarity_code REAL, dissimilarity_reasoning REAL
        );
        CREATE TABLE interactive_status (key TEXT PRIMARY KEY, value TEXT, updated_at REAL);
        INSERT INTO interactive_status VALUES ('review_prioritization_settings', '{"mode":"score_change"}', 123);
    """)
    rows = [
        (
            "placeholder",
            0,
            0.0,
            {"patch_description": "Initial program setup"},
            [2.0, 1.0],
        ),
        ("a", 1, 1.0, {"patch_description": "Cache paths"}, [1.0, 0.0]),
        ("b", 2, 2.0, {"patch_description": "Reuse edges"}, [0.0, 1.0]),
        ("c", 3, 3.0, {"patch_description": "Search backwards"}, [-1.0, 0.0]),
        (
            "d",
            4,
            4.0,
            {"patch_description": "Explore residual capacities"},
            [0.0, -1.0],
        ),
        ("missing", 5, 5.0, {"patch_description": "Use the best path"}, []),
        (
            "mixed",
            6,
            6.0,
            {
                "patch_description": "none",
                "llm_result": {"thought": "Combine cached partial paths"},
            },
            [0.2, 0.8],
        ),
        ("tie-a", 7, 7.0, {"patch_description": "Try a balanced search"}, [0.5, 0.5]),
        (
            "tie-b",
            7,
            7.0,
            {"patch_description": "Adjust the search balance"},
            [0.4, 0.6],
        ),
    ]
    for program_id, generation, timestamp, metadata, vector in rows:
        conn.execute(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                program_id,
                None if generation == 0 else "placeholder",
                generation,
                timestamp,
                f"print({generation})",
                float(generation),
                json.dumps(metadata),
                "[0.1,0.9]",
                json.dumps(vector),
                "[9,9]",
                3,
                "high",
                '{"mode":"score_change","reason":"historical"}',
            ),
        )
        conn.execute(
            "INSERT INTO review_priority_metrics VALUES (?, 0.25, 0.75, 0.99)",
            (program_id,),
        )
    conn.commit()
    return conn


def test_read_only_audit_does_not_change_files_or_construct_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both audit commands leave a DELETE-mode source and its directory intact."""
    from shinka.embed import embedding

    monkeypatch.setattr(
        embedding,
        "EmbeddingClient",
        Mock(side_effect=AssertionError("provider constructed")),
    )
    source = tmp_path / "source.sqlite"
    _database(source).close()
    before = repair.source_hashes(source)
    files = sorted(path.name for path in tmp_path.iterdir())
    report = repair.repair_database(source)
    old_backfill_report = backfill(source, dry_run=True)
    assert report["before"]["invalid_vector_count"] == 1
    assert report["before"]["mixed_text_retained_ids"] == ["mixed"]
    assert old_backfill_report["no_text"] == 1
    assert repair.source_hashes(source) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == files
    with sqlite3.connect(source) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_wal_snapshot_repairs_only_reasoning_and_is_idempotent(tmp_path: Path) -> None:
    """Committed WAL rows survive copying, and historical priorities stay exact."""
    source, output = tmp_path / "source.sqlite", tmp_path / "release.sqlite"
    writer = _database(source, wal=True)
    before = repair.source_hashes(source)
    assert Path(str(source) + "-wal").stat().st_size > 0
    try:
        invariant = repair.invariant_digest(writer)
        original_vectors = dict(
            writer.execute("SELECT id, reasoning_embedding FROM programs")
        )
        report = repair.repair_database(source, output)
        assert repair.source_hashes(source) == before
        assert report["source_unchanged"] and report["idempotent"]
        assert report["projections_rebuilt"] == 7
        assert not report["after"]["needs_repair"]
        assert report["after"]["retained_vectors"] == 7
        assert report["after"]["valid_text_missing_vector_ids"] == ["missing"]
        with sqlite3.connect(output) as conn:
            assert repair.invariant_digest(conn) == invariant
            assert conn.execute(
                "SELECT reasoning_embedding, reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs WHERE id='placeholder'"
            ).fetchone() == ("[]", "[]", None)
            for program_id, raw in conn.execute(
                "SELECT id, reasoning_embedding FROM programs WHERE id != 'placeholder'"
            ):
                assert raw == original_vectors[program_id]
            for program_id, vector in [("tie-a", [0.5, 0.5]), ("tie-b", [0.4, 0.6])]:
                expected = minimum_reasoning_distance(
                    vector,
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.2, 0.8]],
                )
                actual = conn.execute(
                    "SELECT dissimilarity_reasoning FROM review_priority_metrics WHERE program_id=?",
                    (program_id,),
                ).fetchone()[0]
                assert actual == pytest.approx(expected)
        repeated = repair.repair_database(output, tmp_path / "second.sqlite")
        assert not repeated["before"]["needs_repair"]
        assert repeated["projections_rebuilt"] == 0
        with sqlite3.connect(tmp_path / "second.sqlite") as conn:
            assert repair.invariant_digest(conn) == invariant
    finally:
        writer.close()


@pytest.mark.parametrize("output_kind", ["source", "existing", "symlink"])
def test_output_protection(tmp_path: Path, output_kind: str) -> None:
    """Neither explicit overwrite nor a symlink can replace an existing file."""
    source = tmp_path / "source.sqlite"
    _database(source).close()
    output = source if output_kind == "source" else tmp_path / "existing.sqlite"
    if output_kind == "existing":
        output.write_bytes(b"preserve me")
    elif output_kind == "symlink":
        output.symlink_to(source)
    with pytest.raises(repair.ReasoningRepairError):
        repair.repair_database(source, output)
    if output_kind == "existing":
        assert output.read_bytes() == b"preserve me"


@pytest.mark.parametrize("corruption", ["metadata", "dimensions"])
def test_ambiguous_data_aborts_without_output(tmp_path: Path, corruption: str) -> None:
    """The tool must not guess how to classify corrupt metadata or model spaces."""
    source, output = tmp_path / "source.sqlite", tmp_path / "release.sqlite"
    conn = _database(source)
    if corruption == "metadata":
        conn.execute("UPDATE programs SET metadata = 'not json' WHERE id='a'")
    else:
        conn.execute("UPDATE programs SET reasoning_embedding = '[1,2,3]' WHERE id='a'")
    conn.commit()
    conn.close()
    before = repair.source_hashes(source)
    with pytest.raises(repair.ReasoningRepairError):
        repair.repair_database(source, output)
    assert not output.exists()
    assert repair.source_hashes(source) == before
    assert not list(tmp_path.glob(".reasoning-repair-*"))


def test_computation_failure_never_publishes_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection failure leaves no usable-looking partial release database."""
    source, output = tmp_path / "source.sqlite", tmp_path / "release.sqlite"
    _database(source).close()
    monkeypatch.setattr(
        repair,
        "compute_reasoning_features",
        Mock(side_effect=ValueError("projection failed")),
    )
    with pytest.raises(ValueError, match="projection failed"):
        repair.repair_database(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".reasoning-repair-*"))


def test_invariant_failure_rejects_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future regression that edits program code must be caught before publish."""
    source, output = tmp_path / "source.sqlite", tmp_path / "release.sqlite"
    _database(source).close()
    original = repair.apply_repair

    def corrupt(conn: sqlite3.Connection, audit: repair.ReasoningAudit) -> int:
        """Inject an unrelated change to verify publication is rejected."""
        result = original(conn, audit)
        conn.execute("UPDATE programs SET code = 'changed' WHERE id='a'")
        conn.commit()
        return result

    monkeypatch.setattr(repair, "apply_repair", corrupt)
    with pytest.raises(repair.ReasoningRepairError, match="unrelated data"):
        repair.repair_database(source, output)
    assert not output.exists()


def test_old_backfill_cleans_before_skipping_and_preserves_billed_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even force mode cannot retain placeholder vectors or accept bad API data."""
    from shinka.embed import embedding

    source = tmp_path / "source.sqlite"
    _database(source).close()
    client = Mock()
    client.get_embedding.return_value = ([[0.0, 0.0]], 0.5)
    monkeypatch.setattr(embedding, "EmbeddingClient", Mock(return_value=client))
    stats = backfill(source)
    assert stats["skipped"] == 7
    assert stats["errors"] == 1 and stats["total_cost"] == 0.5
    assert client.get_embedding.call_args.args == (["Use the best path"],)
    with sqlite3.connect(source) as conn:
        assert (
            conn.execute(
                "SELECT reasoning_embedding FROM programs WHERE id='placeholder'"
            ).fetchone()[0]
            == "[]"
        )
        assert (
            conn.execute(
                "SELECT reasoning_embedding FROM programs WHERE id='missing'"
            ).fetchone()[0]
            == "[]"
        )
        assert not repair.audit_connection(conn).report()["needs_repair"]


def test_existing_backfill_backup_includes_wal(tmp_path: Path) -> None:
    """The API-based backfill's backup is also a SQLite snapshot, not copy2."""
    source = tmp_path / "source.sqlite"
    writer = _database(source, wal=True)
    try:
        backup = _backup_db(source)
        with sqlite3.connect(backup) as conn:
            assert conn.execute("SELECT count(*) FROM programs").fetchone()[0] == 9
    finally:
        writer.close()
