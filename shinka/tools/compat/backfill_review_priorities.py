#!/usr/bin/env python3
"""Backfill expert review prioritization for existing ShinkaEvolve databases.

Runs the default review-prioritization function (score gain over the parent)
on every program that does not already have a review-priority assignment,
processing them in generation order so parent scores are always available.

Usage::

    python -m shinka.tools.compat.backfill_review_priorities path/to/shinka.db
    python -m shinka.tools.compat.backfill_review_priorities path/to/shinka.db --dry-run

The original database is backed up to ``<name>.db.bak`` (or
``<name>.db.bak.1``, ``.bak.2``, … if earlier backups exist) before any
modifications are made.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import directly from the module file to avoid pulling in the full
# shinka.core.__init__ (which transitively requires yaml, litellm, etc.).
from shinka.core.review_prioritizer import (
    ProgramData,
    ReviewPrioritizer,
    ReviewPriorityLevel,
    default_prioritize_for_review,
)
from shinka.database.review_priority_migration import (
    assert_canonical_review_prioritization_schema,
    create_sqlite_backup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backup_db(db_path: Path) -> Path:
    """Create a numbered backup of *db_path* and return the backup path."""
    return create_sqlite_backup(db_path)


def _assert_backfill_schema(conn: sqlite3.Connection) -> None:
    """Require the canonical schema produced by the one-way migration."""
    assert_canonical_review_prioritization_schema(conn)
    program_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(programs)")
    }
    required_columns = {"review_priority_level", "review_priority_data"}
    missing_columns = required_columns - program_columns
    if missing_columns:
        raise RuntimeError(
            "Missing canonical program columns "
            f"{sorted(missing_columns)}. Run the review-prioritization migration first."
        )
    metrics_table = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'review_priority_metrics'
        """
    ).fetchone()
    if metrics_table is None:
        raise RuntimeError(
            "Missing review_priority_metrics. "
            "Run the review-prioritization migration first."
        )


def _json_or_default(raw: Optional[str], default: Any = None) -> Any:
    """Parse a JSON string, returning *default* on failure or None input."""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _row_to_program_data(row: sqlite3.Row) -> ProgramData:
    """Convert a raw DB row into a lightweight ProgramData."""
    combined_score = row["combined_score"] or 0.0
    correct_raw = row["correct"]
    correct = correct_raw in (True, 1, "true", "True", "1")

    keys = row.keys()
    return ProgramData(
        id=row["id"],
        generation=row["generation"],
        combined_score=combined_score,
        correct=correct,
        public_metrics=_json_or_default(row["public_metrics"], {}),
        private_metrics=_json_or_default(row["private_metrics"], {}),
        embedding=_json_or_default(row["embedding"], []),
        code_diff=row["code_diff"] if "code_diff" in keys else None,
        metadata=_json_or_default(row["metadata"], {}),
        reasoning_embedding=_json_or_default(row["reasoning_embedding"], [])
        if "reasoning_embedding" in keys
        else [],
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def backfill(
    db_path: Path, *, dry_run: bool = False, force: bool = False
) -> Dict[str, int]:
    """Run default expert review prioritization on all unclassified programs.

    When *force* is True, re-classify every program (overwriting existing
    review-priority data). Otherwise programs with
    ``review_priority_level != 'none'`` are skipped.

    Cached prioritization metrics are populated for every program, including
    programs whose existing priority is preserved.

    Returns a dict with counts: ``total``, ``metrics``, ``skipped``,
    ``none``, ``moderate``, ``high``.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    _assert_backfill_schema(conn)

    # Load all programs ordered by generation then timestamp
    rows = conn.execute(
        "SELECT * FROM programs ORDER BY generation ASC, timestamp ASC"
    ).fetchall()

    # Build a lookup for fast parent access
    programs_by_id: Dict[str, sqlite3.Row] = {row["id"]: row for row in rows}

    stats = {
        "total": len(rows),
        "metrics": 0,
        "skipped": 0,
        "none": 0,
        "moderate": 0,
        "high": 0,
    }
    updates: List[tuple] = []
    metric_updates: List[tuple] = []
    previous_code_embeddings: List[List[float]] = []
    previous_reasoning_embeddings: List[List[float]] = []
    prioritizer = ReviewPrioritizer()

    for row in rows:
        program_data = _row_to_program_data(row)

        # Look up parent
        parent_data: Optional[ProgramData] = None
        parent_id = row["parent_id"]
        if parent_id and parent_id in programs_by_id:
            parent_data = _row_to_program_data(programs_by_id[parent_id])

        # Look up inspirations (archive + top_k)
        inspiration_data: List[ProgramData] = []
        archive_ids = _json_or_default(row["archive_inspiration_ids"], [])
        top_k_ids = _json_or_default(row["top_k_inspiration_ids"], [])
        seen: set = set()
        for insp_id in list(archive_ids) + list(top_k_ids):
            if insp_id not in seen and insp_id in programs_by_id:
                seen.add(insp_id)
                inspiration_data.append(_row_to_program_data(programs_by_id[insp_id]))

        metrics = prioritizer.compute_priority_metrics(
            program_data,
            parent_data,
            previous_code_embeddings,
            previous_reasoning_embeddings,
        )
        metric_updates.append(
            (
                row["id"],
                metrics["score_change"],
                metrics["dissimilarity_code"],
                metrics["dissimilarity_reasoning"],
            )
        )
        stats["metrics"] += 1
        if program_data.embedding:
            previous_code_embeddings.append(program_data.embedding)
        if program_data.reasoning_embedding:
            previous_reasoning_embeddings.append(program_data.reasoning_embedding)

        # Preserve existing assignments unless force mode was requested.
        existing_level = row["review_priority_level"]
        if not force and existing_level and existing_level != "none":
            stats["skipped"] += 1
            continue

        level, display_data = default_prioritize_for_review(
            program_data, parent_data, inspiration_data
        )

        if level != ReviewPriorityLevel.NONE:
            updates.append(
                (
                    level.value,
                    json.dumps(display_data) if display_data else None,
                    row["id"],
                )
            )
            stats[level.value] += 1
        else:
            # In force mode, explicitly reset earlier priorities to "none".
            if force:
                updates.append(("none", None, row["id"]))
            stats["none"] += 1

    if not dry_run:
        conn.executemany(
            """
            INSERT OR REPLACE INTO review_priority_metrics
            (program_id, score_change, dissimilarity_code, dissimilarity_reasoning)
            VALUES (?, ?, ?, ?)
            """,
            metric_updates,
        )
        if updates:
            conn.executemany(
                """
                UPDATE programs
                SET review_priority_level = ?, review_priority_data = ?
                WHERE id = ?
                """,
                updates,
            )
        conn.commit()

    conn.close()
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shinka.tools.compat.backfill_review_priorities",
        description=(
            "Backfill expert review priorities and cached prioritization metrics "
            "for a canonical ShinkaEvolve database. Uses the default "
            "score-improvement signal and backs up the DB first."
        ),
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the ShinkaEvolve .db file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse and print results without modifying the database.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-prioritize all programs, overwriting existing priority data.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path: Path = args.db_path.resolve()
    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        return 1

    # Back up before any modification
    if not args.dry_run:
        backup_path = _backup_db(db_path)
        print(f"Backed up {db_path.name} -> {backup_path.name}")
    else:
        print("[DRY RUN] No backup created, database will not be modified.")

    print(f"Processing {db_path} ...")
    if args.force:
        print("Force mode: all programs will be re-classified.")
    stats = backfill(db_path, dry_run=args.dry_run, force=args.force)

    print(f"\nResults ({'DRY RUN' if args.dry_run else 'APPLIED'}):")
    print(f"  Total programs : {stats['total']}")
    print(f"  Metrics cached : {stats['metrics']}")
    print(f"  Skipped (exist): {stats['skipped']}")
    print(f"  None           : {stats['none']}")
    print(f"  Moderate       : {stats['moderate']}")
    print(f"  High           : {stats['high']}")

    prioritized = stats["moderate"] + stats["high"]
    if prioritized > 0 and not args.dry_run:
        print(f"\n  {prioritized} program(s) prioritized for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
