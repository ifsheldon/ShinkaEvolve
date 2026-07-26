"""One-way migration for expert review prioritization storage."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "1"
SCHEMA_VERSION_KEY = "review_prioritization_schema_version"
LEGACY_SETTINGS_KEY = "novelty_settings"
SETTINGS_KEY = "review_prioritization_settings"


class ReviewPrioritizationMigrationError(RuntimeError):
    """Raised when a database cannot be migrated without ambiguity."""


@dataclass
class MigrationReport:
    """Summary of a review-prioritization schema migration."""

    changed: bool
    actions: list[str] = field(default_factory=list)
    backup_path: Optional[Path] = None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table_name}")')}


def _status_value(conn: sqlite3.Connection, key: str) -> Optional[tuple[str, float]]:
    if not _table_exists(conn, "interactive_status"):
        return None
    row = conn.execute(
        "SELECT value, updated_at FROM interactive_status WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), float(row[1])


def _legacy_settings_table_value(
    conn: sqlite3.Connection,
) -> Optional[dict[str, Any]]:
    if not _table_exists(conn, LEGACY_SETTINGS_KEY):
        return None

    settings: dict[str, Any] = {}
    for key, raw_value in conn.execute(
        f'SELECT key, value FROM "{LEGACY_SETTINGS_KEY}"'
    ):
        try:
            settings[str(key)] = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError):
            settings[str(key)] = raw_value
    return settings or None


def _parse_settings(raw_value: str, *, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ReviewPrioritizationMigrationError(
            f"Cannot parse settings stored in {source}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ReviewPrioritizationMigrationError(
            f"Settings stored in {source} must be a JSON object."
        )
    return parsed


def _numbered_backup_path(db_path: Path) -> Path:
    candidate = db_path.with_suffix(db_path.suffix + ".bak")
    index = 0
    while candidate.exists():
        index += 1
        candidate = db_path.with_suffix(f"{db_path.suffix}.bak.{index}")
    return candidate


def create_sqlite_backup(db_path: Path) -> Path:
    """Create a numbered, transactionally consistent SQLite backup."""
    backup_path = _numbered_backup_path(db_path)
    source = sqlite3.connect(str(db_path), timeout=30)
    destination = sqlite3.connect(str(backup_path))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return backup_path


def _validate_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    program_columns = _table_columns(conn, "programs")
    if not program_columns:
        raise ReviewPrioritizationMigrationError(
            "The database does not contain a programs table."
        )

    legacy_program_columns = {
        "novelty_level",
        "novelty_data",
    } & program_columns
    canonical_program_columns = {
        "review_priority_level",
        "review_priority_data",
    } & program_columns
    if legacy_program_columns and canonical_program_columns:
        raise ReviewPrioritizationMigrationError(
            "Legacy and canonical review-priority columns coexist in programs: "
            f"{sorted(legacy_program_columns | canonical_program_columns)}"
        )

    if _table_exists(conn, "novelty_cache") and _table_exists(
        conn, "review_priority_metrics"
    ):
        raise ReviewPrioritizationMigrationError(
            "Legacy novelty_cache and canonical review_priority_metrics tables coexist."
        )

    legacy_status = _status_value(conn, LEGACY_SETTINGS_KEY)
    canonical_status = _status_value(conn, SETTINGS_KEY)
    if canonical_status and (legacy_status or _table_exists(conn, LEGACY_SETTINGS_KEY)):
        raise ReviewPrioritizationMigrationError(
            "Legacy and canonical review-prioritization settings coexist."
        )

    legacy_error = db_path.parent / "novelty_error.json"
    canonical_error = db_path.parent / "review_prioritization_error.json"
    if legacy_error.exists() and canonical_error.exists():
        raise ReviewPrioritizationMigrationError(
            "Legacy and canonical review-prioritization error files coexist."
        )


def legacy_schema_items(conn: sqlite3.Connection) -> list[str]:
    """Return legacy post-evaluation schema items present in a database."""
    items: list[str] = []
    columns = _table_columns(conn, "programs")
    for column in ("novelty_level", "novelty_data"):
        if column in columns:
            items.append(f"programs.{column}")
    if _table_exists(conn, "novelty_cache"):
        items.append("novelty_cache")
    if _table_exists(conn, LEGACY_SETTINGS_KEY):
        items.append(LEGACY_SETTINGS_KEY)
    if _status_value(conn, LEGACY_SETTINGS_KEY):
        items.append(f"interactive_status[{LEGACY_SETTINGS_KEY}]")
    return items


def assert_canonical_review_prioritization_schema(
    conn: sqlite3.Connection,
) -> None:
    """Reject legacy post-evaluation schemas before normal runtime access."""
    legacy_items = legacy_schema_items(conn)
    if legacy_items:
        joined = ", ".join(legacy_items)
        raise ReviewPrioritizationMigrationError(
            "Legacy expert-review-prioritization schema detected "
            f"({joined}). Run `uv run python -m "
            "shinka.tools.compat.migrate_review_prioritization <database>` first."
        )


def _planned_actions(conn: sqlite3.Connection, db_path: Path) -> list[str]:
    actions: list[str] = []
    columns = _table_columns(conn, "programs")

    for legacy, canonical in (
        ("novelty_level", "review_priority_level"),
        ("novelty_data", "review_priority_data"),
    ):
        if legacy in columns:
            actions.append(f"rename programs.{legacy} to programs.{canonical}")
        elif canonical not in columns:
            actions.append(f"add programs.{canonical}")

    if _table_exists(conn, "novelty_cache"):
        actions.append("rename novelty_cache to review_priority_metrics")
    elif not _table_exists(conn, "review_priority_metrics"):
        actions.append("create review_priority_metrics")

    standalone_settings = _legacy_settings_table_value(conn)
    legacy_status = _status_value(conn, LEGACY_SETTINGS_KEY)
    if standalone_settings is not None:
        actions.append(
            "move standalone novelty_settings into "
            "interactive_status[review_prioritization_settings]"
        )
    elif legacy_status is not None:
        actions.append(
            "rename interactive_status[novelty_settings] to "
            "interactive_status[review_prioritization_settings]"
        )
    if _table_exists(conn, LEGACY_SETTINGS_KEY):
        actions.append("drop novelty_settings")
    if legacy_status is not None:
        actions.append("delete interactive_status[novelty_settings]")

    legacy_error = db_path.parent / "novelty_error.json"
    if legacy_error.exists():
        actions.append("rename novelty_error.json to review_prioritization_error.json")

    version_row = None
    if _table_exists(conn, "metadata_store"):
        version_row = conn.execute(
            "SELECT value FROM metadata_store WHERE key = ?",
            (SCHEMA_VERSION_KEY,),
        ).fetchone()
    if version_row is None or str(version_row[0]) != SCHEMA_VERSION:
        actions.append(f"record schema version {SCHEMA_VERSION}")

    return actions


def migrate_review_prioritization(
    db_path: Path,
    *,
    dry_run: bool = False,
    create_backup: bool = True,
) -> MigrationReport:
    """Migrate one database to the canonical expert-review-prioritization schema."""
    resolved_path = db_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Database not found: {resolved_path}")

    conn = sqlite3.connect(str(resolved_path), timeout=30)
    backup_path: Optional[Path] = None
    try:
        _validate_schema(conn, resolved_path)
        actions = _planned_actions(conn, resolved_path)
        if dry_run or not actions:
            return MigrationReport(changed=bool(actions), actions=actions)

        if create_backup:
            conn.close()
            backup_path = create_sqlite_backup(resolved_path)
            conn = sqlite3.connect(str(resolved_path), timeout=30)
            _validate_schema(conn, resolved_path)

        standalone_settings = _legacy_settings_table_value(conn)
        legacy_status = _status_value(conn, LEGACY_SETTINGS_KEY)
        settings: Optional[dict[str, Any]] = None
        settings_updated_at = time.time()
        if standalone_settings is not None:
            settings = standalone_settings
        elif legacy_status is not None:
            settings = _parse_settings(
                legacy_status[0],
                source=f"interactive_status[{LEGACY_SETTINGS_KEY}]",
            )
            settings_updated_at = legacy_status[1]

        conn.execute("BEGIN IMMEDIATE")
        columns = _table_columns(conn, "programs")
        for legacy, canonical, definition in (
            (
                "novelty_level",
                "review_priority_level",
                "TEXT DEFAULT 'none'",
            ),
            ("novelty_data", "review_priority_data", "TEXT"),
        ):
            if legacy in columns:
                conn.execute(
                    f'ALTER TABLE programs RENAME COLUMN "{legacy}" TO "{canonical}"'
                )
            elif canonical not in columns:
                conn.execute(
                    f'ALTER TABLE programs ADD COLUMN "{canonical}" {definition}'
                )

        if _table_exists(conn, "novelty_cache"):
            conn.execute("ALTER TABLE novelty_cache RENAME TO review_priority_metrics")
        elif not _table_exists(conn, "review_priority_metrics"):
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

        if settings is not None:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactive_status (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO interactive_status (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (SETTINGS_KEY, json.dumps(settings), settings_updated_at),
            )

        if _table_exists(conn, LEGACY_SETTINGS_KEY):
            conn.execute(f'DROP TABLE "{LEGACY_SETTINGS_KEY}"')
        if _table_exists(conn, "interactive_status"):
            conn.execute(
                "DELETE FROM interactive_status WHERE key = ?",
                (LEGACY_SETTINGS_KEY,),
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_store (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            (SCHEMA_VERSION_KEY, SCHEMA_VERSION),
        )
        conn.commit()

        legacy_error = resolved_path.parent / "novelty_error.json"
        canonical_error = resolved_path.parent / "review_prioritization_error.json"
        if legacy_error.exists():
            legacy_error.rename(canonical_error)

        return MigrationReport(
            changed=True,
            actions=actions,
            backup_path=backup_path,
        )
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()
