"""Provider-independent eligibility rules for reasoning text and embeddings."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Real
from typing import Any, cast


PLACEHOLDERS = frozenset(
    {
        "none",
        "null",
        "n/a",
        "na",
        "undefined",
        "unknown",
        "no description",
        "no reasoning",
        "initial program",
        "initial program setup",
        "initial program setup (fallback)",
    }
)
_QUOTES = "\"'`“”‘’«»"
_SENTENCE_ENDINGS = ".!?。！？…"


def eligible_reasoning_text(value: object) -> str | None:
    """Return original trimmed prose unless it is an exact placeholder."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    comparison = text
    while True:
        stripped = comparison.strip().strip(_QUOTES).rstrip(_SENTENCE_ENDINGS)
        if stripped == comparison:
            break
        comparison = stripped
    comparison = " ".join(comparison.split()).casefold()
    if comparison in PLACEHOLDERS or not any(c.isalnum() for c in comparison):
        return None
    return text


def extract_reasoning_text(metadata: object) -> str | None:
    """Filter description and thought independently, then join eligible prose."""
    if not isinstance(metadata, Mapping):
        return None
    description = eligible_reasoning_text(metadata.get("patch_description"))
    llm_result = metadata.get("llm_result")
    thought = eligible_reasoning_text(
        llm_result.get("thought") if isinstance(llm_result, Mapping) else None
    )
    parts = [part for part in (description, thought) if part is not None]
    return "\n\n".join(parts)[:10000] if parts else None


def finite_coordinates(value: object, dimensions: int | None = None) -> bool:
    """Check finite numeric arrays, allowing an origin in projected space."""
    if not isinstance(value, list) or not value:
        return False
    if dimensions is not None and len(value) != dimensions:
        return False
    try:
        return all(
            isinstance(item, Real)
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
    except (OverflowError, TypeError, ValueError):
        return False


def validated_vector(value: object) -> list[float] | None:
    """Accept finite, nonzero vectors without changing their numeric payload."""
    if not finite_coordinates(value):
        return None
    vector = cast(list[float], value)
    norm = math.hypot(*vector)
    return vector if math.isfinite(norm) and norm > 0 else None


def reasoning_vector(metadata: object, value: object) -> list[float] | None:
    """Require both eligible source metadata and a structurally usable vector."""
    return validated_vector(value) if extract_reasoning_text(metadata) else None


def normalize_reasoning_fields(data: MutableMapping[str, Any]) -> None:
    """Normalize a program boundary, clearing derivatives of missing vectors."""
    vector = reasoning_vector(data.get("metadata"), data.get("reasoning_embedding"))
    data["reasoning_embedding"] = vector if vector is not None else []
    pca = data.get("reasoning_embedding_pca_2d")
    cluster = data.get("reasoning_embedding_cluster_id")
    if vector is None or not finite_coordinates(pca, 2):
        data["reasoning_embedding_pca_2d"] = []
        data["reasoning_embedding_cluster_id"] = None
    elif not isinstance(cluster, int) or isinstance(cluster, bool) or cluster < 0:
        data["reasoning_embedding_cluster_id"] = None


def serialized_reasoning_fields(
    metadata: object, vector: object, pca: object, cluster: object
) -> tuple[str, str, int | None]:
    """Validate parsed or SQLite JSON fields before a direct SQL copy/migration."""
    data: dict[str, Any] = {"reasoning_embedding_cluster_id": cluster}
    for key, value, default in (
        ("metadata", metadata, {}),
        ("reasoning_embedding", vector, []),
        ("reasoning_embedding_pca_2d", pca, []),
    ):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = default
        data[key] = value
    normalize_reasoning_fields(data)
    return (
        json.dumps(data["reasoning_embedding"]),
        json.dumps(data["reasoning_embedding_pca_2d"]),
        data["reasoning_embedding_cluster_id"],
    )


def cosine_similarity(left: object, right: object) -> float | None:
    """Return a finite cosine similarity, or None for an unavailable comparison."""
    a, b = validated_vector(left), validated_vector(right)
    if a is None or b is None or len(a) != len(b):
        return None
    norm_a, norm_b = math.hypot(*a), math.hypot(*b)
    similarity = math.fsum((x / norm_a) * (y / norm_b) for x, y in zip(a, b))
    return max(-1.0, min(1.0, similarity))


def minimum_reasoning_distance(
    vector: object, previous: Sequence[list[float]]
) -> float | None:
    """Return minimum cosine distance only when a usable predecessor exists."""
    similarities = [
        similarity
        for prior in previous
        if (similarity := cosine_similarity(vector, prior)) is not None
    ]
    return 1.0 - max(similarities) if similarities else None
