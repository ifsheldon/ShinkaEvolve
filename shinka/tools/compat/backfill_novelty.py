#!/usr/bin/env python3
"""Backfill novelty detection for existing ShinkaEvolve databases.

Runs the default novelty detector (score-gain vs. parent) on every program
that doesn't already have a novelty classification, processing them in
generation order so parent scores are always available.

Usage::

    python -m shinka.tools.compat.backfill_novelty path/to/shinka.db
    python -m shinka.tools.compat.backfill_novelty path/to/shinka.db --dry-run

The original database is backed up to ``<name>.db.bak`` (or
``<name>.db.bak.1``, ``.bak.2``, … if earlier backups exist) before any
modifications are made.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import directly from the module file to avoid pulling in the full
# shinka.core.__init__ (which transitively requires yaml, litellm, etc.).
from shinka.core.novelty_detector import (
    NoveltyLevel,
    ProgramData,
    default_detect_novelty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backup_db(db_path: Path) -> Path:
    """Create a numbered backup of *db_path* and return the backup path."""
    candidate = db_path.with_suffix(db_path.suffix + ".bak")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = db_path.with_suffix(f"{db_path.suffix}.bak.{counter}")
    shutil.copy2(db_path, candidate)
    return candidate


def _ensure_novelty_columns(conn: sqlite3.Connection) -> None:
    """Add novelty_level / novelty_data columns if they don't exist."""
    cur = conn.execute("PRAGMA table_info(programs)")
    columns = {row[1] for row in cur.fetchall()}

    if "novelty_level" not in columns:
        conn.execute(
            "ALTER TABLE programs ADD COLUMN novelty_level TEXT DEFAULT 'none'"
        )
    if "novelty_data" not in columns:
        conn.execute("ALTER TABLE programs ADD COLUMN novelty_data TEXT")
    conn.commit()


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

    return ProgramData(
        id=row["id"],
        generation=row["generation"],
        combined_score=combined_score,
        correct=correct,
        public_metrics=_json_or_default(row["public_metrics"], {}),
        private_metrics=_json_or_default(row["private_metrics"], {}),
        embedding=_json_or_default(row["embedding"], []),
        code_diff=row["code_diff"] if "code_diff" in row.keys() else None,
        metadata=_json_or_default(row["metadata"], {}),
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def backfill(
    db_path: Path, *, dry_run: bool = False, force: bool = False
) -> Dict[str, int]:
    """Run default novelty detection on all un-classified programs.

    When *force* is True, re-classify every program (overwriting existing
    novelty data).  Otherwise programs with ``novelty_level != 'none'``
    are skipped.

    Returns a dict with counts: ``total``, ``skipped``, ``none``,
    ``moderate``, ``high``.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    # Check if novelty columns exist (don't create them in dry-run)
    cur = conn.execute("PRAGMA table_info(programs)")
    columns = {row[1] for row in cur.fetchall()}
    has_novelty_cols = "novelty_level" in columns

    if not dry_run:
        _ensure_novelty_columns(conn)

    # Load all programs ordered by generation then timestamp
    rows = conn.execute(
        "SELECT * FROM programs ORDER BY generation ASC, timestamp ASC"
    ).fetchall()

    # Build a lookup for fast parent access
    programs_by_id: Dict[str, sqlite3.Row] = {row["id"]: row for row in rows}

    stats = {"total": len(rows), "skipped": 0, "none": 0, "moderate": 0, "high": 0}
    updates: List[tuple] = []

    for row in rows:
        # Skip programs that already have novelty classification
        if not force:
            existing_level = row["novelty_level"] if has_novelty_cols else None
            if existing_level and existing_level != "none":
                stats["skipped"] += 1
                continue

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

        level, display_data = default_detect_novelty(
            program_data, parent_data, inspiration_data
        )

        if level != NoveltyLevel.NONE:
            updates.append(
                (
                    level.value,
                    json.dumps(display_data) if display_data else None,
                    row["id"],
                )
            )
            stats[level.value] += 1
        else:
            # In force mode, explicitly reset previously-novel programs to "none"
            if force:
                updates.append(("none", None, row["id"]))
            stats["none"] += 1

    if not dry_run and updates:
        conn.executemany(
            "UPDATE programs SET novelty_level = ?, novelty_data = ? WHERE id = ?",
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
        prog="python -m shinka.tools.compat.backfill_novelty",
        description=(
            "Backfill novelty detection for an existing ShinkaEvolve database. "
            "Uses the default score-gain detector. Backs up the DB first."
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
        help="Re-classify all programs, overwriting existing novelty data.",
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
    print(f"  Skipped (exist): {stats['skipped']}")
    print(f"  None           : {stats['none']}")
    print(f"  Moderate       : {stats['moderate']}")
    print(f"  High           : {stats['high']}")

    novel = stats["moderate"] + stats["high"]
    if novel > 0 and not args.dry_run:
        print(f"\n  {novel} program(s) classified as novel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
