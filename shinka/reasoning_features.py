"""Offline, deterministic projections and clustering of reasoning vectors."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from itertools import groupby

from shinka.reasoning import reasoning_vector


def historical_reasoning_distances(
    records: list[tuple[str, int, float, list[float] | None]],
) -> dict[str, float | None]:
    """Compare against strictly earlier generations/timestamps, never tied peers."""
    import numpy as np

    results: dict[str, float | None] = {}
    previous: dict[int, list[np.ndarray]] = {}
    records = sorted(records, key=lambda row: (row[1], row[2], row[0]))
    for _, group in groupby(records, key=lambda row: (row[1], row[2])):
        additions = []
        for program_id, _, _, vector in group:
            results[program_id] = None
            if vector is None:
                continue
            array = np.asarray(vector, dtype=np.float64)
            # Scaling first avoids overflow when squaring large finite values.
            array = array / np.max(np.abs(array))
            array = array / np.linalg.norm(array)
            compatible = previous.get(len(vector), [])
            if compatible:
                similarities = np.asarray(compatible) @ array
                results[program_id] = 1.0 - float(np.clip(similarities.max(), -1, 1))
            additions.append(array)
        for array in additions:
            previous.setdefault(len(array), []).append(array)
    return results


def compute_reasoning_features(
    vectors: Mapping[str, list[float]], num_clusters: int = 4
) -> dict[str, tuple[list[float], int]]:
    """Compute runtime PCA/GMM features without constructing a provider client."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if num_clusters < 1:
        raise ValueError("Reasoning cluster count must be positive.")
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) > 1:
        raise ValueError("Reasoning embeddings have incompatible dimensions.")
    ids = sorted(vectors)
    if len(ids) < num_clusters:
        return {}
    matrix = np.asarray([vectors[program_id] for program_id in ids], dtype=np.float64)
    if matrix.shape[1] < 2 or len(np.unique(matrix, axis=0)) < num_clusters:
        return {}
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(matrix)
    )
    model = GaussianMixture(n_components=num_clusters, random_state=42)
    clusters = model.fit_predict(matrix)
    if not model.converged_ or not np.isfinite(coordinates).all():
        raise ValueError(
            "Reasoning PCA/GMM computation did not converge to finite features."
        )
    return {
        program_id: (coordinates[index].tolist(), int(clusters[index]))
        for index, program_id in enumerate(ids)
    }


def recompute_reasoning_features(
    conn: sqlite3.Connection, num_clusters: int = 4
) -> int:
    """Replace all reasoning derivatives atomically, including stale exclusions."""
    rows = conn.execute(
        "SELECT id, metadata, reasoning_embedding FROM programs"
    ).fetchall()
    vectors = {}
    for program_id, metadata_raw, vector_raw in rows:
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            value = json.loads(vector_raw) if vector_raw else []
        except (json.JSONDecodeError, TypeError):
            continue
        vector = reasoning_vector(metadata, value)
        if vector is not None:
            vectors[program_id] = vector
    features = compute_reasoning_features(vectors, num_clusters)
    with conn:
        conn.execute(
            "UPDATE programs SET reasoning_embedding_pca_2d = '[]', "
            "reasoning_embedding_cluster_id = NULL"
        )
        conn.executemany(
            "UPDATE programs SET reasoning_embedding_pca_2d = ?, "
            "reasoning_embedding_cluster_id = ? WHERE id = ?",
            [
                (json.dumps(coords), cluster, program_id)
                for program_id, (coords, cluster) in features.items()
            ],
        )
    return len(features)


def refresh_reasoning_metrics(conn: sqlite3.Connection) -> None:
    """Refresh the reasoning component of existing canonical cache rows only."""
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'review_priority_metrics'"
        ).fetchone()
        is None
    ):
        return
    records = []
    for program_id, generation, timestamp, metadata_raw, vector_raw in conn.execute(
        "SELECT id, generation, timestamp, metadata, reasoning_embedding FROM programs"
    ):
        try:
            vector = reasoning_vector(
                json.loads(metadata_raw or "{}"), json.loads(vector_raw or "[]")
            )
        except (json.JSONDecodeError, TypeError):
            vector = None
        records.append((program_id, generation, timestamp, vector))
    distances = historical_reasoning_distances(records)
    with conn:
        conn.executemany(
            "UPDATE review_priority_metrics SET dissimilarity_reasoning = ? WHERE program_id = ?",
            [(distance, program_id) for program_id, distance in distances.items()],
        )
