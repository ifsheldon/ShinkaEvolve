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
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from shinka.edit.async_apply import extract_reasoning_text
from shinka.embed.embedding import EmbeddingClient


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


def _ensure_reasoning_columns(conn: sqlite3.Connection) -> None:
    """Add reasoning embedding columns if they don't exist."""
    cur = conn.execute("PRAGMA table_info(programs)")
    columns = {row[1] for row in cur.fetchall()}

    if "reasoning_embedding" not in columns:
        conn.execute("ALTER TABLE programs ADD COLUMN reasoning_embedding TEXT")
    if "reasoning_embedding_pca_2d" not in columns:
        conn.execute(
            "ALTER TABLE programs ADD COLUMN reasoning_embedding_pca_2d TEXT"
        )
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
    """Recompute PCA 2D and GMM clusters for all reasoning embeddings."""
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print(
            "Warning: scikit-learn not installed. "
            "Skipping PCA/cluster recomputation.",
            file=sys.stderr,
        )
        return 0

    rows = conn.execute(
        "SELECT id, reasoning_embedding FROM programs "
        "WHERE reasoning_embedding IS NOT NULL"
    ).fetchall()

    ids: List[str] = []
    embeddings: List[List[float]] = []
    for row in rows:
        emb = _json_or_default(row["reasoning_embedding"], [])
        if isinstance(emb, list) and len(emb) > 0:
            ids.append(row["id"])
            embeddings.append(emb)

    if len(embeddings) < 2:
        return 0

    matrix = np.array(embeddings)

    # PCA 2D
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    pca_2d = pca.fit_transform(matrix)

    # GMM clustering
    n_clusters = min(max(2, len(embeddings) // 5), 10)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_ids = gmm.fit_predict(matrix)

    updates = []
    for i, prog_id in enumerate(ids):
        coords = pca_2d[i].tolist()
        updates.append((json.dumps(coords), int(cluster_ids[i]), prog_id))

    conn.executemany(
        "UPDATE programs SET reasoning_embedding_pca_2d = ?, "
        "reasoning_embedding_cluster_id = ? WHERE id = ?",
        updates,
    )
    conn.commit()
    return len(updates)


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
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    if not dry_run:
        _ensure_reasoning_columns(conn)

    # Check if reasoning_embedding column exists (may not in dry-run on old DBs)
    cur = conn.execute("PRAGMA table_info(programs)")
    columns = {row[1] for row in cur.fetchall()}
    has_reasoning_col = "reasoning_embedding" in columns

    if has_reasoning_col:
        rows = conn.execute(
            "SELECT id, metadata, reasoning_embedding FROM programs "
            "ORDER BY generation ASC, timestamp ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, metadata FROM programs "
            "ORDER BY generation ASC, timestamp ASC"
        ).fetchall()

    stats: Dict[str, object] = {
        "total": len(rows),
        "skipped": 0,
        "embedded": 0,
        "no_text": 0,
        "errors": 0,
        "total_cost": 0.0,
        "pca_updated": 0,
    }

    # Collect programs that need embedding
    to_embed: List[tuple] = []  # (id, text)
    for row in rows:
        if not force and has_reasoning_col:
            existing = _json_or_default(row["reasoning_embedding"], [])
            if isinstance(existing, list) and len(existing) > 0:
                stats["skipped"] = int(stats["skipped"]) + 1
                continue

        metadata = _json_or_default(row["metadata"], {})
        text = extract_reasoning_text(metadata)
        if not text:
            stats["no_text"] = int(stats["no_text"]) + 1
            continue

        if len(text) > max_chars:
            text = text[:max_chars]

        to_embed.append((row["id"], text))

    if dry_run:
        stats["embedded"] = len(to_embed)
        conn.close()
        return stats

    if not to_embed:
        # Still recompute PCA if forced
        if force:
            stats["pca_updated"] = _recompute_pca_and_clusters(conn)
        conn.close()
        return stats

    # Initialize embedding client
    client = EmbeddingClient(model_name=model_name)
    total_cost = 0.0

    # Process in batches
    for batch_start in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_start : batch_start + batch_size]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        try:
            embeddings, cost = client.get_embedding(texts)
            total_cost += cost

            if not isinstance(embeddings[0], list):
                # Single result returned as flat list
                embeddings = [embeddings]

            for prog_id, emb in zip(ids, embeddings):
                conn.execute(
                    "UPDATE programs SET reasoning_embedding = ? WHERE id = ?",
                    (json.dumps(emb), prog_id),
                )
            stats["embedded"] = int(stats["embedded"]) + len(batch)

        except Exception as e:
            print(
                f"Error embedding batch starting at {batch_start}: {e}",
                file=sys.stderr,
            )
            stats["errors"] = int(stats["errors"]) + len(batch)

        conn.commit()

        done = min(batch_start + batch_size, len(to_embed))
        print(f"  Progress: {done}/{len(to_embed)} programs embedded", end="\r")

    print()  # newline after progress

    stats["total_cost"] = total_cost

    # Recompute PCA and clusters
    stats["pca_updated"] = _recompute_pca_and_clusters(conn)

    conn.close()
    return stats


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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
