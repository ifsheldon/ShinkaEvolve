"""Audit reasoning embeddings read-only, or repair a separate SQLite snapshot.

This command is entirely offline. It never generates embeddings or changes
historical review-priority assignments. The input must have the canonical
review-prioritization schema and one metrics row per program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from shinka.reasoning import (
    eligible_reasoning_text,
    extract_reasoning_text,
    finite_coordinates,
    validated_vector,
)
from shinka.reasoning_features import (
    compute_reasoning_features,
    historical_reasoning_distances,
)


class ReasoningRepairError(ValueError):
    """An input cannot be repaired without guessing or changing unrelated data."""


@dataclass(frozen=True)
class ReasoningRecord:
    """Parsed SQLite boundary retaining the original embedding payload."""

    program_id: str
    generation: int
    timestamp: float
    raw_vector: str | None
    vector: list[float] | None
    has_text: bool
    mixed_text: bool
    invalid_reason: str | None
    pca: object
    cluster: object


@dataclass
class ReasoningAudit:
    """Validated population and its proposed reasoning-only changes."""

    records: list[ReasoningRecord]
    invalid: dict[str, list[str]]
    stale_ids: list[str]
    missing_feature_ids: list[str]
    metrics: dict[str, float | None]
    metric_updates: list[tuple[float | None, str]]
    reproject: bool

    def report(self) -> dict[str, Any]:
        """Produce an audit without including prose, code, or raw metadata."""
        valid = [record for record in self.records if record.vector is not None]
        return {
            "programs": len(self.records),
            "retained_vectors": len(valid),
            "dimensions": sorted({len(record.vector) for record in valid}),
            "invalid_vectors": self.invalid,
            "invalid_vector_count": sum(map(len, self.invalid.values())),
            "valid_text_missing_vector_ids": [
                record.program_id
                for record in self.records
                if record.has_text
                and record.vector is None
                and record.invalid_reason is None
            ],
            "mixed_text_retained_ids": [
                record.program_id for record in valid if record.mixed_text
            ],
            "stale_feature_ids": self.stale_ids,
            "missing_feature_ids": self.missing_feature_ids,
            "rebuild_projections": self.reproject,
            "reasoning_metric_update_ids": [
                program_id for _, program_id in self.metric_updates
            ],
            "non_null_reasoning_metrics": sum(
                value is not None for value in self.metrics.values()
            ),
            "needs_repair": bool(self.invalid or self.reproject or self.metric_updates),
        }


def _decode(raw: str | None, default: object) -> object:
    """Decode an optional JSON column without silently accepting corruption."""
    return json.loads(raw) if raw else default


def _assert_schema(conn: sqlite3.Connection) -> None:
    """Require canonical columns and complete cached-metric membership."""
    required = {
        "programs": {
            "id",
            "generation",
            "timestamp",
            "metadata",
            "reasoning_embedding",
            "reasoning_embedding_pca_2d",
            "reasoning_embedding_cluster_id",
            "review_priority_level",
            "review_priority_data",
        },
        "review_priority_metrics": {
            "program_id",
            "score_change",
            "dissimilarity_code",
            "dissimilarity_reasoning",
        },
    }
    for table, columns in required.items():
        actual = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if not columns <= actual:
            raise ReasoningRepairError(
                f"Missing canonical {table} columns: {sorted(columns - actual)}. "
                "Run the review-prioritization migration/backfill first."
            )
        if {"novelty_level", "novelty_data"} & actual:
            raise ReasoningRepairError(
                "Legacy novelty columns coexist with the canonical schema."
            )
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'novelty_cache'"
    ).fetchone():
        raise ReasoningRepairError("Legacy novelty_cache must be migrated first.")
    program_ids = {row[0] for row in conn.execute("SELECT id FROM programs")}
    metric_ids = {
        row[0] for row in conn.execute("SELECT program_id FROM review_priority_metrics")
    }
    if program_ids != metric_ids:
        raise ReasoningRepairError(
            "Expected exactly one review_priority_metrics row per program."
        )


def audit_connection(conn: sqlite3.Connection) -> ReasoningAudit:
    """Classify vectors and calculate the complete proposed repair in memory."""
    _assert_schema(conn)
    rows = conn.execute(
        "SELECT id, generation, timestamp, metadata, reasoning_embedding, "
        "reasoning_embedding_pca_2d, reasoning_embedding_cluster_id "
        "FROM programs ORDER BY generation, timestamp, id"
    ).fetchall()
    records = []
    invalid: dict[str, list[str]] = {}
    for (
        program_id,
        generation,
        timestamp,
        raw_metadata,
        raw_vector,
        raw_pca,
        cluster,
    ) in rows:
        try:
            metadata = _decode(raw_metadata, {})
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReasoningRepairError(
                f"Unclassifiable metadata for program {program_id}."
            ) from exc
        if not isinstance(metadata, dict):
            raise ReasoningRepairError(
                f"Metadata must be an object for program {program_id}."
            )
        if (
            not isinstance(generation, int)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
        ):
            raise ReasoningRepairError(
                f"Invalid historical ordering for program {program_id}."
            )
        has_text = extract_reasoning_text(metadata) is not None
        llm_result = metadata.get("llm_result")
        thought = llm_result.get("thought") if isinstance(llm_result, dict) else None
        mixed = (
            eligible_reasoning_text(metadata.get("patch_description")) is None
            and eligible_reasoning_text(thought) is not None
        )
        reason = None
        try:
            value = _decode(raw_vector, [])
            present = value is not None and value != []
        except (json.JSONDecodeError, TypeError):
            value, present, reason = None, True, "malformed_vector"
        vector = validated_vector(value) if has_text else None
        if present and vector is None:
            reason = reason or (
                "missing_or_placeholder_text" if not has_text else "invalid_vector"
            )
            invalid.setdefault(reason, []).append(program_id)
        try:
            pca = _decode(raw_pca, [])
        except (json.JSONDecodeError, TypeError):
            pca = "malformed"
        records.append(
            ReasoningRecord(
                program_id,
                generation,
                timestamp,
                raw_vector,
                vector,
                has_text,
                mixed,
                reason,
                pca,
                cluster,
            )
        )
    vectors = {
        record.program_id: record.vector
        for record in records
        if record.vector is not None
    }
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) > 1:
        raise ReasoningRepairError(
            f"Incompatible reasoning vector dimensions: {sorted(dimensions)}."
        )
    projectable = len({tuple(vector) for vector in vectors.values()}) >= 4 and all(
        d >= 2 for d in dimensions
    )
    stale_ids, missing_ids = [], []
    for record in records:
        has_features = record.pca not in (None, []) or record.cluster is not None
        valid_features = (
            finite_coordinates(record.pca, 2)
            and isinstance(record.cluster, int)
            and not isinstance(record.cluster, bool)
            and 0 <= record.cluster < 4
        )
        if has_features and (
            record.vector is None or not projectable or not valid_features
        ):
            stale_ids.append(record.program_id)
        if record.vector is not None and projectable and not valid_features:
            missing_ids.append(record.program_id)
    metrics = historical_reasoning_distances(
        [
            (record.program_id, record.generation, record.timestamp, record.vector)
            for record in records
        ]
    )
    cached = dict(
        conn.execute(
            "SELECT program_id, dissimilarity_reasoning FROM review_priority_metrics"
        )
    )
    updates = []
    for program_id, expected in metrics.items():
        current = cached[program_id]
        same = current is None and expected is None
        if current is not None and expected is not None:
            same = (
                isinstance(current, (int, float))
                and math.isfinite(current)
                and abs(current - expected) <= 1e-12
            )
        if not same:
            updates.append((expected, program_id))
    return ReasoningAudit(
        records,
        invalid,
        stale_ids,
        missing_ids,
        metrics,
        updates,
        bool(invalid or stale_ids or missing_ids),
    )


def _quote_identifier(identifier: str) -> str:
    """Quote schema-owned SQLite identifiers, including embedded quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def invariant_digest(conn: sqlite3.Connection) -> str:
    """Hash the schema and every value outside permitted reasoning changes."""
    excluded = {
        "programs": {
            "reasoning_embedding",
            "reasoning_embedding_pca_2d",
            "reasoning_embedding_cluster_id",
        },
        "review_priority_metrics": {"dissimilarity_reasoning"},
    }
    digest = hashlib.sha256()
    schema = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    digest.update(json.dumps([tuple(row) for row in schema]).encode())
    tables = [row[1] for row in schema if row[0] == "table"]
    for table in tables:
        quoted = _quote_identifier(table)
        columns = [
            row[1]
            for row in conn.execute(f"PRAGMA table_info({quoted})")
            if row[1] not in excluded.get(table, set())
        ]
        selected = ", ".join(map(_quote_identifier, columns))
        digest.update(table.encode())
        for row in conn.execute(f"SELECT {selected} FROM {quoted} ORDER BY {selected}"):
            digest.update(
                json.dumps(
                    tuple(row),
                    ensure_ascii=False,
                    default=lambda value: {"blob": value.hex()},
                ).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Hash a file using bounded reads."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hashes(path: Path) -> dict[str, str | None]:
    """Capture persistent SQLite files; the shared-memory lock file is transient."""
    return {
        suffix or "database": file_hash(candidate) if candidate.exists() else None
        for suffix in ("", "-wal")
        for candidate in (Path(str(path) + suffix),)
    }


def _integrity_check(conn: sqlite3.Connection) -> None:
    """Reject structurally corrupt databases before reporting a usable artifact."""
    if [row[0] for row in conn.execute("PRAGMA integrity_check")] != ["ok"]:
        raise ReasoningRepairError("SQLite integrity_check failed.")
    if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ReasoningRepairError("SQLite foreign_key_check failed.")


def apply_repair(conn: sqlite3.Connection, audit: ReasoningAudit) -> int:
    """Apply the audited changes atomically, preserving vector payloads retained."""
    vectors = {
        record.program_id: record.vector
        for record in audit.records
        if record.vector is not None
    }
    features = compute_reasoning_features(vectors) if audit.reproject else {}
    with conn:
        conn.executemany(
            "UPDATE programs SET reasoning_embedding = '[]', reasoning_embedding_pca_2d = '[]', "
            "reasoning_embedding_cluster_id = NULL WHERE id = ?",
            [(program_id,) for ids in audit.invalid.values() for program_id in ids],
        )
        if audit.reproject:
            conn.execute(
                "UPDATE programs SET reasoning_embedding_pca_2d = '[]', reasoning_embedding_cluster_id = NULL"
            )
            conn.executemany(
                "UPDATE programs SET reasoning_embedding_pca_2d = ?, reasoning_embedding_cluster_id = ? WHERE id = ?",
                [
                    (json.dumps(coords), cluster, program_id)
                    for program_id, (coords, cluster) in features.items()
                ],
            )
        conn.executemany(
            "UPDATE review_priority_metrics SET dissimilarity_reasoning = ? WHERE program_id = ?",
            audit.metric_updates,
        )
    return len(features)


def repair_database(source: Path, output: Path | None = None) -> dict[str, Any]:
    """Audit a read-only source, optionally publishing a verified separate copy."""
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ReasoningRepairError("Source must be a SQLite file.")
    if output is not None:
        if output.is_symlink():
            raise ReasoningRepairError("Output must not be a symbolic link.")
        output = output.resolve()
        if output == source or output.exists():
            raise ReasoningRepairError(
                "Output must be a new file distinct from the source."
            )
    before_hashes = source_hashes(source)
    conn = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    temporary: Path | None = None
    report: dict[str, Any] = {
        "source_name": source.name,
        "source_hashes_before": before_hashes,
        "embedding_api_calls": 0,
        "algorithms": {
            "pca": "StandardScaler + PCA(full, 2)",
            "gmm": "full covariance, 4 components",
            "seed": 42,
            "ordering": "program ID for projections; generation/timestamp groups for history",
            "numpy_version": version("numpy"),
            "scikit_learn_version": version("scikit-learn"),
        },
    }
    try:
        _integrity_check(conn)
        audit = audit_connection(conn)
        report["before"] = audit.report()
        if output is None:
            report["mode"] = "dry-run"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=".reasoning-repair-", suffix=".sqlite", dir=output.parent
            )
            os.close(fd)
            temporary = Path(name)
            destination = sqlite3.connect(temporary)
            try:
                conn.backup(destination)
                destination.execute("PRAGMA journal_mode=DELETE")
                snapshot_audit = audit_connection(destination)
                report["before"] = snapshot_audit.report()
                invariant_before = invariant_digest(destination)
                retained = {
                    record.program_id: record.raw_vector
                    for record in snapshot_audit.records
                    if record.vector is not None
                }
                report["projections_rebuilt"] = apply_repair(
                    destination, snapshot_audit
                )
                _integrity_check(destination)
                after = audit_connection(destination)
                invariant_after = invariant_digest(destination)
                retained_after = dict(
                    destination.execute("SELECT id, reasoning_embedding FROM programs")
                )
                if invariant_before != invariant_after or any(
                    retained_after[program_id] != raw
                    for program_id, raw in retained.items()
                ):
                    raise ReasoningRepairError(
                        "Repair changed unrelated data or a retained vector payload."
                    )
                if after.report()["needs_repair"]:
                    raise ReasoningRepairError(
                        "Repaired snapshot still contains outstanding reasoning changes."
                    )
                report.update(
                    {
                        "mode": "repair-copy",
                        "after": after.report(),
                        "invariant_sha256": invariant_after,
                        "unrelated_data_unchanged": True,
                        "retained_vectors_unchanged": True,
                        "integrity_check": "ok",
                        "idempotent": True,
                    }
                )
            finally:
                destination.close()
        after_hashes = source_hashes(source)
        if before_hashes != after_hashes:
            raise ReasoningRepairError(
                "Source changed during the operation; no output was published."
            )
        report["source_hashes_after"] = after_hashes
        report["source_unchanged"] = True
        if output is not None and temporary is not None:
            report["output_sha256"] = file_hash(temporary)
            # Same-filesystem link publishes atomically and cannot overwrite a racing writer.
            os.link(temporary, output)
        return report
    finally:
        conn.close()
        if temporary is not None:
            for suffix in ("", "-wal", "-shm", "-journal"):
                Path(str(temporary) + suffix).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Run one explicit offline audit or copy-repair operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = repair_database(args.source, args.output)
    except (ReasoningRepairError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Reasoning repair failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
