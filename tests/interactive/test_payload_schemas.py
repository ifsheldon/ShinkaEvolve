"""Tests for interactive command payload validation schemas."""

import pytest
from pydantic import ValidationError

from shinka.interactive.payload_schemas import (
    MAX_PARENT_IDS,
    MAX_PROMPT_CHARS,
    MAX_TARGET_GENERATIONS,
    VALID_PATCH_TYPES,
    MergePayload,
    SetTargetPayload,
    SuggestPayload,
)


# ---------------------------------------------------------------------------
# SetTargetPayload
# ---------------------------------------------------------------------------


class TestSetTargetPayload:
    def test_valid_target(self):
        p = SetTargetPayload(target_generations=50)
        assert p.target_generations == 50

    def test_boundary_min(self):
        p = SetTargetPayload(target_generations=1)
        assert p.target_generations == 1

    def test_boundary_max(self):
        p = SetTargetPayload(target_generations=MAX_TARGET_GENERATIONS)
        assert p.target_generations == MAX_TARGET_GENERATIONS

    def test_rejects_zero(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            SetTargetPayload(target_generations=0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            SetTargetPayload(target_generations=-5)

    def test_rejects_above_max(self):
        with pytest.raises(ValidationError, match=f"must be <= {MAX_TARGET_GENERATIONS}"):
            SetTargetPayload(target_generations=MAX_TARGET_GENERATIONS + 1)


# ---------------------------------------------------------------------------
# SuggestPayload
# ---------------------------------------------------------------------------


class TestSuggestPayload:
    def test_valid_suggest(self):
        p = SuggestPayload(parent_id="abc-123", prompt="try something", patch_type="full")
        assert p.parent_id == "abc-123"
        assert p.prompt == "try something"
        assert p.patch_type == "full"

    def test_defaults(self):
        p = SuggestPayload(parent_id="x")
        assert p.prompt == ""
        assert p.patch_type == "full"

    def test_rejects_empty_parent_id(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            SuggestPayload(parent_id="")

    def test_rejects_whitespace_only_parent_id(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            SuggestPayload(parent_id="   ")

    def test_strips_parent_id(self):
        p = SuggestPayload(parent_id="  abc  ")
        assert p.parent_id == "abc"

    def test_prompt_truncated_at_max(self):
        long_prompt = "x" * (MAX_PROMPT_CHARS + 500)
        p = SuggestPayload(parent_id="p", prompt=long_prompt)
        assert len(p.prompt) == MAX_PROMPT_CHARS

    def test_prompt_at_exact_max_not_truncated(self):
        exact_prompt = "y" * MAX_PROMPT_CHARS
        p = SuggestPayload(parent_id="p", prompt=exact_prompt)
        assert len(p.prompt) == MAX_PROMPT_CHARS

    def test_all_valid_patch_types(self):
        for pt in VALID_PATCH_TYPES:
            p = SuggestPayload(parent_id="p", patch_type=pt)
            assert p.patch_type == pt

    def test_rejects_invalid_patch_type(self):
        with pytest.raises(ValidationError, match="patch_type must be one of"):
            SuggestPayload(parent_id="p", patch_type="invalid")


# ---------------------------------------------------------------------------
# MergePayload
# ---------------------------------------------------------------------------


class TestMergePayload:
    def test_valid_merge(self):
        p = MergePayload(parent_ids=["a", "b"], prompt="combine", patch_type="cross")
        assert p.parent_ids == ["a", "b"]
        assert p.prompt == "combine"
        assert p.patch_type == "cross"

    def test_defaults(self):
        p = MergePayload(parent_ids=["a", "b"])
        assert p.prompt == ""
        assert p.patch_type == "cross"

    def test_rejects_single_parent(self):
        with pytest.raises(ValidationError, match="at least 2"):
            MergePayload(parent_ids=["a"])

    def test_rejects_empty_list(self):
        with pytest.raises(ValidationError, match="at least 2"):
            MergePayload(parent_ids=[])

    def test_rejects_too_many_parents(self):
        ids = [f"p{i}" for i in range(MAX_PARENT_IDS + 1)]
        with pytest.raises(ValidationError, match=f"at most {MAX_PARENT_IDS}"):
            MergePayload(parent_ids=ids)

    def test_boundary_max_parents(self):
        ids = [f"p{i}" for i in range(MAX_PARENT_IDS)]
        p = MergePayload(parent_ids=ids)
        assert len(p.parent_ids) == MAX_PARENT_IDS

    def test_rejects_empty_string_parent_id(self):
        with pytest.raises(ValidationError, match="non-empty"):
            MergePayload(parent_ids=["a", ""])

    def test_rejects_whitespace_only_parent_id(self):
        with pytest.raises(ValidationError, match="non-empty"):
            MergePayload(parent_ids=["a", "   "])

    def test_strips_parent_ids(self):
        p = MergePayload(parent_ids=["  a  ", "  b  "])
        assert p.parent_ids == ["a", "b"]

    def test_prompt_truncated(self):
        long_prompt = "z" * (MAX_PROMPT_CHARS + 100)
        p = MergePayload(parent_ids=["a", "b"], prompt=long_prompt)
        assert len(p.prompt) == MAX_PROMPT_CHARS

    def test_rejects_invalid_patch_type(self):
        with pytest.raises(ValidationError, match="patch_type must be one of"):
            MergePayload(parent_ids=["a", "b"], patch_type="banana")

    def test_all_valid_patch_types(self):
        for pt in VALID_PATCH_TYPES:
            p = MergePayload(parent_ids=["a", "b"], patch_type=pt)
            assert p.patch_type == pt
