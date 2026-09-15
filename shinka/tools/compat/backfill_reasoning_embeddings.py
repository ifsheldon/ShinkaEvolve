#!/usr/bin/env python3
"""Backfill reasoning embeddings for existing ShinkaEvolve databases.

Generates reasoning embeddings from program metadata (patch_description +
thought) for programs that don't already have them, then recomputes PCA 2D
projections and cluster IDs.

Usage::

    python -m shinka.tools.compat.backfill_reasoning_embeddings path/to/shinka.db
    python -m shinka.tools.compat.backfill_reasoning_embeddings path/to/shinka.db --dry-run
    python -m shinka.tools.compat.backfill_reasoning_embeddings path/to/shinka.db --model text-embedding-3-small

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
from typing import Dict, List, Optional

from shinka.reasoning import extract_reasoning_text, reasoning_vector, validated_vector
from shinka.reasoning_features import (
    recompute_reasoning_features,
    refresh_reasoning_metrics,
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
    source = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    destination = sqlite3.connect(candidate)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return candidate


def _ensure_reasoning_columns(conn: sqlite3.Connection) -> None:
    """Add reasoning embedding columns if they don't exist."""
    cur = conn.execute("PRAGMA table_info(programs)")
    columns = {row[1] for row in cur.fetchall()}

    if "reasoning_embedding" not in columns:
        conn.execute("ALTER TABLE programs ADD COLUMN reasoning_embedding TEXT")
    if "reasoning_embedding_pca_2d" not in columns:
        conn.execute("ALTER TABLE programs ADD COLUMN reasoning_embedding_pca_2d TEXT")
    if "reasoning_embedding_cluster_id" not in columns:
        conn.execute(
            "ALTER TABLE programs ADD COLUMN reasoning_embedding_cluster_id INTEGER"
        )
    conn.commit()


def _json_or_default(raw: Optional[str], default=None):
    """Parse a JSON string, returning *default* on failure or None input."""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _recompute_pca_and_clusters(conn: sqlite3.Connection) -> int:
    """Refresh shared local projections and existing reasoning-distance cache."""
    count = recompute_reasoning_features(conn)
    refresh_reasoning_metrics(conn)
    return count


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def backfill(
    db_path: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    model_name: str = "text-embedding-3-small",
    batch_size: int = 20,
    max_chars: int = 10000,
) -> Dict[str, object]:
    """Generate reasoning embeddings for programs missing them.

    When *force* is True, regenerate for all programs (overwriting existing).
    Otherwise programs with existing reasoning_embedding are skipped.

    Returns a dict with counts: ``total``, ``skipped``, ``embedded``,
    ``no_text``, ``errors``, ``total_cost``, ``pca_updated``.
    """
    if batch_size < 1 or max_chars < 1:
        raise ValueError("batch_size and max_chars must be positive.")
    conn = sqlite3.connect(
        db_path.resolve().as_uri() + ("?mode=ro" if dry_run else "?mode=rw"),
        uri=True,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    try:
        if not dry_run:
            _ensure_reasoning_columns(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(programs)")}
        embedding_column = (
            "reasoning_embedding"
            if "reasoning_embedding" in columns
            else "NULL AS reasoning_embedding"
        )
        rows = conn.execute(
            f"SELECT id, metadata, {embedding_column} FROM programs ORDER BY generation, timestamp, id"
        ).fetchall()
        stats: Dict[str, object] = {
            "total": len(rows),
            "skipped": 0,
            "embedded": 0,
            "no_text": 0,
            "errors": 0,
            "total_cost": 0.0,
            "pca_updated": 0,
            "cleared": 0,
        }
        to_embed: List[tuple[str, str]] = []
        to_clear: list[tuple[str]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Unclassifiable metadata for program {row['id']}."
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata must be an object for program {row['id']}.")
            existing = _json_or_default(row["reasoning_embedding"], [])
            vector = reasoning_vector(metadata, existing)
            text = extract_reasoning_text(metadata)
            if vector is None:
                to_clear.append((row["id"],))
                if existing or row["reasoning_embedding"] not in (
                    None,
                    "",
                    "[]",
                    "null",
                ):
                    stats["cleared"] = int(stats["cleared"]) + 1
            if text is None:
                stats["no_text"] = int(stats["no_text"]) + 1
            elif vector is not None and not force:
                stats["skipped"] = int(stats["skipped"]) + 1
            else:
                to_embed.append((row["id"], text[:max_chars]))
        if dry_run:
            stats["embedded"] = len(to_embed)
            return stats
        with conn:
            conn.executemany(
                "UPDATE programs SET reasoning_embedding = '[]', reasoning_embedding_pca_2d = '[]', "
                "reasoning_embedding_cluster_id = NULL WHERE id = ?",
                to_clear,
            )
        if to_embed:
            # Provider import and construction happen only after the dry-run return.
            from shinka.embed.embedding import EmbeddingClient

            client = EmbeddingClient(model_name=model_name)
            for batch_start in range(0, len(to_embed), batch_size):
                batch = to_embed[batch_start : batch_start + batch_size]
                ids, texts = zip(*batch)
                try:
                    embeddings, cost = client.get_embedding(list(texts))
                    stats["total_cost"] = float(stats["total_cost"]) + cost
                    if len(batch) == 1 and validated_vector(embeddings) is not None:
                        embeddings = [embeddings]
                    if not isinstance(embeddings, list) or len(embeddings) != len(
                        batch
                    ):
                        raise ValueError(
                            "Embedding response does not match the requested batch size."
                        )
                    if any(validated_vector(vector) is None for vector in embeddings):
                        raise ValueError(
                            "Embedding response contains an unusable vector."
                        )
                    with conn:
                        conn.executemany(
                            "UPDATE programs SET reasoning_embedding = ? WHERE id = ?",
                            [
                                (json.dumps(vector), program_id)
                                for program_id, vector in zip(ids, embeddings)
                            ],
                        )
                    stats["embedded"] = int(stats["embedded"]) + len(batch)
                except Exception as exc:
                    print(
                        f"Embedding batch at {batch_start} failed ({type(exc).__name__}).",
                        file=sys.stderr,
                    )
                    stats["errors"] = int(stats["errors"]) + len(batch)
        stats["pca_updated"] = _recompute_pca_and_clusters(conn)
        return stats
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shinka.tools.compat.backfill_reasoning_embeddings",
        description=(
            "Backfill reasoning embeddings for an existing ShinkaEvolve database. "
            "Generates embeddings from patch_description + thought metadata. "
            "Backs up the DB first."
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
        help="Regenerate all reasoning embeddings, overwriting existing ones.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of texts to embed per API call (default: 20).",
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
        print("Force mode: all programs will be re-embedded.")
    print(f"Model: {args.model}")

    stats = backfill(
        db_path,
        dry_run=args.dry_run,
        force=args.force,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print(f"\nResults ({'DRY RUN' if args.dry_run else 'APPLIED'}):")
    print(f"  Total programs   : {stats['total']}")
    print(f"  Skipped (exist)  : {stats['skipped']}")
    print(f"  Embedded         : {stats['embedded']}")
    print(f"  No reasoning text: {stats['no_text']}")
    print(f"  Errors           : {stats['errors']}")
    if not args.dry_run:
        print(f"  Embedding cost   : ${stats['total_cost']:.4f}")
        print(f"  PCA/clusters     : {stats['pca_updated']} programs updated")

    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
