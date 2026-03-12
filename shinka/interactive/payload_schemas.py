"""Pydantic validation schemas for interactive command payloads.

Each payload-bearing command type (SET_TARGET, SUGGEST, MERGE) has a
dedicated model that enforces types, bounds, and basic sanitization.
Validation errors are raised as ``pydantic.ValidationError`` and caught
by the ``process_commands`` error handler, which marks the command FAILED.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_TARGET_GENERATIONS: int = 100_000
"""Upper bound on target_generations to prevent runaway loops."""

MAX_PARENT_IDS: int = 10
"""Maximum number of parent IDs accepted in a single merge command."""

MAX_PROMPT_CHARS: int = 8_000
"""Prompts longer than this are silently truncated before being forwarded."""

VALID_PATCH_TYPES: frozenset[str] = frozenset({"full", "cross", "diff"})
"""Allowed values for the patch_type field."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SetTargetPayload(BaseModel):
    """Payload for the SET_TARGET command."""

    target_generations: int

    @field_validator("target_generations")
    @classmethod
    def must_be_in_bounds(cls, v: int) -> int:
        """Enforce 1 ≤ target_generations ≤ MAX_TARGET_GENERATIONS."""
        if v < 1:
            raise ValueError(
                f"target_generations must be >= 1, got {v}"
            )
        if v > MAX_TARGET_GENERATIONS:
            raise ValueError(
                f"target_generations must be <= {MAX_TARGET_GENERATIONS}, got {v}"
            )
        return v


class SuggestPayload(BaseModel):
    """Payload for the SUGGEST command."""

    parent_id: str
    prompt: str = ""
    patch_type: str = "full"

    @field_validator("parent_id")
    @classmethod
    def parent_id_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("parent_id must not be empty")
        return v

    @field_validator("prompt")
    @classmethod
    def truncate_prompt(cls, v: str) -> str:
        """Silently truncate over-long prompts rather than rejecting them."""
        return v[:MAX_PROMPT_CHARS]

    @field_validator("patch_type")
    @classmethod
    def valid_patch_type(cls, v: str) -> str:
        if v not in VALID_PATCH_TYPES:
            raise ValueError(
                f"patch_type must be one of {sorted(VALID_PATCH_TYPES)}, got {v!r}"
            )
        return v


class MergePayload(BaseModel):
    """Payload for the MERGE command."""

    parent_ids: List[str]
    prompt: str = ""
    patch_type: str = "cross"

    @field_validator("parent_ids")
    @classmethod
    def validate_parent_ids(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError(
                f"merge requires at least 2 parent_ids, got {len(v)}"
            )
        if len(v) > MAX_PARENT_IDS:
            raise ValueError(
                f"merge accepts at most {MAX_PARENT_IDS} parent_ids, got {len(v)}"
            )
        stripped = [s.strip() for s in v]
        if any(not s for s in stripped):
            raise ValueError("all parent_ids must be non-empty strings")
        return stripped

    @field_validator("prompt")
    @classmethod
    def truncate_prompt(cls, v: str) -> str:
        """Silently truncate over-long prompts rather than rejecting them."""
        return v[:MAX_PROMPT_CHARS]

    @field_validator("patch_type")
    @classmethod
    def valid_patch_type(cls, v: str) -> str:
        if v not in VALID_PATCH_TYPES:
            raise ValueError(
                f"patch_type must be one of {sorted(VALID_PATCH_TYPES)}, got {v!r}"
            )
        return v
