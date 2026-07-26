"""Migrate legacy post-evaluation novelty fields to review prioritization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from shinka.database.review_priority_migration import (
    ReviewPrioritizationMigrationError,
    migrate_review_prioritization,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shinka.tools.compat.migrate_review_prioritization",
        description=(
            "Apply the one-way Expert Review Prioritization schema migration "
            "to a ShinkaEvolve database."
        ),
    )
    parser.add_argument("database", type=Path, help="Path to programs.sqlite.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned changes without modifying or backing up the database.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = migrate_review_prioritization(
            args.database,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ReviewPrioritizationMigrationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    if not report.actions:
        print("Database already uses the canonical review-prioritization schema.")
        return 0

    print(f"{mode}:")
    for action in report.actions:
        print(f"  - {action}")
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
