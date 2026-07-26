"""Tests for canonical expert-review-priority backfilling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from shinka.tools.compat.backfill_review_priorities import backfill


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
