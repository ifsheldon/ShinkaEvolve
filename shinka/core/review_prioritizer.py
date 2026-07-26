"""Post-evaluation expert review prioritization for ShinkaEvolve.

This module assigns review priorities to newly evaluated programs using
performance change, embedding dissimilarity, or a user-defined function. It is
distinct from ``NoveltyJudge``, which is a pre-evaluation rejection-sampling
mechanism based on embedding similarity.

Users can supply a custom ``prioritize_for_review`` function in a Python file
(configured via ``evo_config.review_prioritization_function_path``).  The file is
hot-reloaded on each invocation so edits take effect without restarting
the evolution run.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from shinka.database.dbase import Program

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ReviewPriorityLevel(str, enum.Enum):
    """Priority assigned to a newly evaluated program for expert review."""

    NONE = "none"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class ProgramData:
    """Lightweight view passed to user review-prioritization functions.

    This avoids exposing full ORM internals to user code.
    """

    id: str
    generation: int
    combined_score: float
    correct: bool
    public_metrics: Dict[str, Any]
    private_metrics: Dict[str, Any]
    embedding: List[float]
    reasoning_embedding: List[float]
    code_diff: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ReviewPriorityResult:
    """Return value from expert review prioritization."""

    level: ReviewPriorityLevel
    display_data: Optional[Dict[str, Any]] = None


# Type alias for user-defined review-prioritization functions
ReviewPrioritizationFunction = Callable[
    [ProgramData, Optional[ProgramData], List[ProgramData]],
    Tuple["ReviewPriorityLevel", Optional[Dict[str, Any]]],
]


# ---------------------------------------------------------------------------
# Default review-prioritization function
# ---------------------------------------------------------------------------


def default_prioritize_for_review(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[ReviewPriorityLevel, Optional[Dict[str, Any]]]:
    """Assign review priority from performance gain over the parent.

    Thresholds:
        * ≥ 30 % gain  →  HIGH
        * ≥ 15 % gain  →  MODERATE
        * otherwise     →  NONE

    Edge cases:
        * Gen-0 programs (no parent) → NONE
        * Incorrect programs → NONE
        * Parent score == 0 and program score > 0 → HIGH
    """
    if parent is None:
        return ReviewPriorityLevel.NONE, None

    if not program.correct:
        return ReviewPriorityLevel.NONE, None

    parent_score = parent.combined_score
    program_score = program.combined_score

    # Handle zero parent score
    if parent_score == 0.0:
        if program_score > 0.0:
            return ReviewPriorityLevel.HIGH, {
                "reason": "First correct solution from parent with zero score",
                "parent_score": parent_score,
                "program_score": program_score,
                "gain_pct": None,
            }
        return ReviewPriorityLevel.NONE, None

    gain_pct = (program_score - parent_score) / abs(parent_score)

    display: Dict[str, Any] = {
        "parent_score": round(parent_score, 4),
        "program_score": round(program_score, 4),
        "gain_pct": round(gain_pct * 100, 2),
    }

    if gain_pct >= 0.30:
        return ReviewPriorityLevel.HIGH, {
            **display,
            "reason": f"{gain_pct * 100:.1f}% gain (>=30%)",
        }
    if gain_pct >= 0.15:
        return ReviewPriorityLevel.MODERATE, {
            **display,
            "reason": f"{gain_pct * 100:.1f}% gain (>=15%)",
        }
    return ReviewPriorityLevel.NONE, display


# ---------------------------------------------------------------------------
# ReviewPrioritizer – orchestration + hot-reload
# ---------------------------------------------------------------------------


class ReviewPrioritizer:
    """Manage review-prioritization function loading and execution.

    Parameters
    ----------
    review_prioritization_function_path
        Optional path to a Python file containing a user-defined
        ``prioritize_for_review`` function.  When *None* the built-in
        :func:`default_prioritize_for_review` is used.
    function_name
        Name of the callable to import from the user file.
    results_dir
        If given, prioritization load errors are written as
        ``<results_dir>/review_prioritization_error.json`` so the frontend can display them.
    """

    def __init__(
        self,
        review_prioritization_function_path: Optional[str] = None,
        function_name: str = "prioritize_for_review",
        results_dir: Optional[str] = None,
    ) -> None:
        self._function_path = review_prioritization_function_path
        self._function_name = function_name
        self._results_dir = results_dir
        self._cached_module = None
        self._cached_mtime: float = 0.0
        self._load_error: Optional[str] = None
        self._prioritization_fn: ReviewPrioritizationFunction = (
            default_prioritize_for_review
        )

        if review_prioritization_function_path:
            self._try_load_function()

    # -- public API ----------------------------------------------------------

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def uses_custom_function(self) -> bool:
        """Return whether a custom prioritization function is configured."""
        return self._function_path is not None

    def prioritize(
        self,
        program: Program,
        parent: Optional[Program],
        inspirations: List[Program],
    ) -> ReviewPriorityResult:
        """Assign a review priority.

        Attempts a hot-reload of the user function first, then runs
        prioritization. Falls back to the default function on any error.
        """
        # Hot-reload check
        if self._function_path:
            self._try_load_function()

        # Convert to lightweight data objects
        prog_data = self._to_program_data(program)
        parent_data = self._to_program_data(parent) if parent else None
        insp_data = [self._to_program_data(i) for i in inspirations]

        try:
            level, display_data = self._prioritization_fn(
                prog_data, parent_data, insp_data
            )
            return ReviewPriorityResult(level=level, display_data=display_data)
        except Exception:
            # User prioritization hooks are plugin boundaries; preserve the run and
            # fall back to the built-in function, but keep the full traceback.
            logger.exception("Review-prioritization function raised an exception")
            level, display_data = default_prioritize_for_review(
                prog_data, parent_data, insp_data
            )
            return ReviewPriorityResult(level=level, display_data=display_data)

    def compute_dissimilarity(
        self,
        embedding: List[float],
        all_previous_embeddings: List[List[float]],
    ) -> float:
        """Compute minimum cosine distance between embedding and all previous embeddings.

        Returns the minimum distance (1 - cosine_similarity).
        Returns 1.0 if no previous embeddings exist.
        """
        if not embedding or not all_previous_embeddings:
            return 1.0

        query = np.asarray(embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return 1.0

        min_distance = 1.0
        for prev in all_previous_embeddings:
            prev_arr = np.asarray(prev, dtype=np.float64)
            if prev_arr.shape != query.shape:
                continue
            prev_norm = np.linalg.norm(prev_arr)
            if prev_norm == 0.0:
                continue
            cosine_sim = float(np.dot(query, prev_arr) / (query_norm * prev_norm))
            # Clamp to [-1, 1] to handle floating-point errors
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            distance = 1.0 - cosine_sim
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def compute_priority_metrics(
        self,
        program: Program,
        parent: Optional[Program],
        all_previous_code_embeddings: List[List[float]],
        all_previous_reasoning_embeddings: List[List[float]],
    ) -> Dict[str, Optional[float]]:
        """Compute the three signals available for review prioritization.

        Args:
            program: The newly evaluated program.
            parent: The parent program (None for gen-0).
            all_previous_code_embeddings: Code embeddings of all earlier programs.
            all_previous_reasoning_embeddings: Reasoning embeddings of all earlier programs.

        Returns:
            {"score_change": float|None,
             "dissimilarity_code": float|None,
             "dissimilarity_reasoning": float|None}
        """
        # --- score_change ---
        score_change: Optional[float] = None
        if parent is not None and program.correct:
            parent_score = parent.combined_score
            program_score = program.combined_score
            if parent_score == 0.0:
                score_change = 1.0 if program_score > 0.0 else 0.0
            else:
                score_change = (program_score - parent_score) / abs(parent_score)

        # --- dissimilarity_code ---
        dissimilarity_code: Optional[float] = None
        prog_embedding = program.embedding or []
        if prog_embedding and all_previous_code_embeddings:
            dissimilarity_code = self.compute_dissimilarity(
                prog_embedding, all_previous_code_embeddings
            )

        # --- dissimilarity_reasoning ---
        dissimilarity_reasoning: Optional[float] = None
        prog_reasoning_embedding = program.reasoning_embedding or []
        if prog_reasoning_embedding and all_previous_reasoning_embeddings:
            dissimilarity_reasoning = self.compute_dissimilarity(
                prog_reasoning_embedding, all_previous_reasoning_embeddings
            )

        return {
            "score_change": score_change,
            "dissimilarity_code": dissimilarity_code,
            "dissimilarity_reasoning": dissimilarity_reasoning,
        }

    # -- internal ------------------------------------------------------------

    def _try_load_function(self) -> bool:
        """Attempt to (re)load the user review-prioritization function.

        Returns *True* if the custom function is active, *False* if
        we fell back to the default.
        """
        if not self._function_path:
            return False

        path = Path(self._function_path)
        if not path.exists():
            self._set_error(f"Review-prioritization function file not found: {path}")
            self._prioritization_fn = default_prioritize_for_review
            return False

        try:
            current_mtime = path.stat().st_mtime

            # Skip reload if file hasn't changed
            if self._cached_module is not None and current_mtime == self._cached_mtime:
                return True

            spec = importlib.util.spec_from_file_location(
                "user_review_prioritization", str(path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec from {path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            fn = getattr(module, self._function_name, None)
            if fn is None:
                raise AttributeError(
                    f"Function '{self._function_name}' not found in {path}"
                )
            if not callable(fn):
                raise TypeError(f"'{self._function_name}' in {path} is not callable")

            # Validate with dummy data
            self._validate_function(fn)

            self._cached_module = module
            self._cached_mtime = current_mtime
            self._prioritization_fn = fn
            self._clear_error()
            logger.info(f"Loaded review-prioritization function from {path}")
            return True

        except Exception as exc:
            logger.exception(
                "Failed to load review-prioritization function from %s", path
            )
            self._set_error(
                f"Failed to load review-prioritization function from {path}: {exc}"
            )
            self._prioritization_fn = default_prioritize_for_review
            return False

    def _validate_function(self, fn: Callable) -> None:
        """Call *fn* with dummy data and verify the return shape."""
        dummy = ProgramData(
            id="__validate__",
            generation=1,
            combined_score=1.0,
            correct=True,
            public_metrics={},
            private_metrics={},
            embedding=[],
            reasoning_embedding=[],
            code_diff=None,
            metadata={},
        )
        dummy_parent = ProgramData(
            id="__validate_parent__",
            generation=0,
            combined_score=0.5,
            correct=True,
            public_metrics={},
            private_metrics={},
            embedding=[],
            reasoning_embedding=[],
            code_diff=None,
            metadata={},
        )

        result = fn(dummy, dummy_parent, [])

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "Review-prioritization function must return "
                f"(ReviewPriorityLevel, Optional[dict]), "
                f"got {type(result)}"
            )
        level, data = result
        if not isinstance(level, ReviewPriorityLevel):
            raise TypeError(
                f"First return value must be ReviewPriorityLevel, got {type(level)}"
            )
        if data is not None and not isinstance(data, dict):
            raise TypeError(
                f"Second return value must be None or dict, got {type(data)}"
            )

    def _set_error(self, msg: str) -> None:
        self._load_error = msg
        logger.warning(msg)
        self._write_error_file(msg)

    def _clear_error(self) -> None:
        self._load_error = None
        self._remove_error_file()

    def _write_error_file(self, msg: str) -> None:
        if not self._results_dir:
            return
        try:
            err_path = os.path.join(
                self._results_dir, "review_prioritization_error.json"
            )
            with open(err_path, "w") as f:
                json.dump({"error": msg}, f)
        except OSError as exc:
            logger.warning(
                "Could not write review-prioritization error file %s: %s",
                os.path.join(self._results_dir, "review_prioritization_error.json"),
                exc,
            )

    def _remove_error_file(self) -> None:
        if not self._results_dir:
            return
        try:
            err_path = os.path.join(
                self._results_dir, "review_prioritization_error.json"
            )
            if os.path.exists(err_path):
                os.remove(err_path)
        except OSError as exc:
            logger.warning(
                "Could not remove review-prioritization error file %s: %s",
                os.path.join(self._results_dir, "review_prioritization_error.json"),
                exc,
            )

    @staticmethod
    def _to_program_data(program: Program) -> ProgramData:
        return ProgramData(
            id=program.id,
            generation=program.generation,
            combined_score=program.combined_score,
            correct=program.correct,
            public_metrics=program.public_metrics or {},
            private_metrics=program.private_metrics or {},
            embedding=program.embedding or [],
            reasoning_embedding=program.reasoning_embedding or [],
            code_diff=program.code_diff,
            metadata=program.metadata or {},
        )
