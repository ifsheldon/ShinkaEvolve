"""Post-evaluation novelty detection for ShinkaEvolve.

This module classifies newly evaluated programs by how much they improve
over their parent(s).  It is distinct from ``NoveltyJudge`` which is a
*pre-evaluation* rejection-sampling mechanism based on embedding similarity.

Users can supply a custom ``detect_novelty`` function in a Python file
(configured via ``evo_config.novelty_function_path``).  The file is
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

from shinka.database.dbase import Program

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class NoveltyLevel(str, enum.Enum):
    """Classification of how novel a newly evaluated program is."""

    NONE = "none"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class ProgramData:
    """Lightweight view of a :class:`Program` passed to user novelty functions.

    This avoids exposing full ORM internals to user code.
    """

    id: str
    generation: int
    combined_score: float
    correct: bool
    public_metrics: Dict[str, Any]
    private_metrics: Dict[str, Any]
    embedding: List[float]
    code_diff: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class NoveltyResult:
    """Return value from novelty detection."""

    level: NoveltyLevel
    display_data: Optional[Dict[str, Any]] = None


# Type alias for user-defined novelty functions
NoveltyFunction = Callable[
    [ProgramData, Optional[ProgramData], List[ProgramData]],
    Tuple["NoveltyLevel", Optional[Dict[str, Any]]],
]


# ---------------------------------------------------------------------------
# Default novelty function
# ---------------------------------------------------------------------------


def default_detect_novelty(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[NoveltyLevel, Optional[Dict[str, Any]]]:
    """Default novelty detection based on performance gain vs best parent.

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
        return NoveltyLevel.NONE, None

    if not program.correct:
        return NoveltyLevel.NONE, None

    parent_score = parent.combined_score
    program_score = program.combined_score

    # Handle zero parent score
    if parent_score == 0.0:
        if program_score > 0.0:
            return NoveltyLevel.HIGH, {
                "reason": "First correct solution from parent with zero score",
                "parent_score": parent_score,
                "program_score": program_score,
                "gain_pct": None,
            }
        return NoveltyLevel.NONE, None

    gain_pct = (program_score - parent_score) / abs(parent_score)

    display: Dict[str, Any] = {
        "parent_score": round(parent_score, 4),
        "program_score": round(program_score, 4),
        "gain_pct": round(gain_pct * 100, 2),
    }

    if gain_pct >= 0.30:
        return NoveltyLevel.HIGH, {
            **display,
            "reason": f"{gain_pct * 100:.1f}% gain (>=30%)",
        }
    if gain_pct >= 0.15:
        return NoveltyLevel.MODERATE, {
            **display,
            "reason": f"{gain_pct * 100:.1f}% gain (>=15%)",
        }
    return NoveltyLevel.NONE, display


# ---------------------------------------------------------------------------
# NoveltyDetector – orchestration + hot-reload
# ---------------------------------------------------------------------------


class NoveltyDetector:
    """Manages novelty function loading, hot-reload, and execution.

    Parameters
    ----------
    novelty_function_path
        Optional path to a Python file containing a user-defined
        ``detect_novelty`` function.  When *None* the built-in
        :func:`default_detect_novelty` is used.
    function_name
        Name of the callable to import from the user file.
    results_dir
        If given, novelty load errors are written as
        ``<results_dir>/novelty_error.json`` so the frontend can display them.
    """

    def __init__(
        self,
        novelty_function_path: Optional[str] = None,
        function_name: str = "detect_novelty",
        results_dir: Optional[str] = None,
    ) -> None:
        self._function_path = novelty_function_path
        self._function_name = function_name
        self._results_dir = results_dir
        self._cached_module = None
        self._cached_mtime: float = 0.0
        self._load_error: Optional[str] = None
        self._novelty_fn: NoveltyFunction = default_detect_novelty

        if novelty_function_path:
            self._try_load_function()

    # -- public API ----------------------------------------------------------

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(
        self,
        program: Program,
        parent: Optional[Program],
        inspirations: List[Program],
    ) -> NoveltyResult:
        """Run novelty detection.

        Attempts a hot-reload of the user function first, then runs
        detection.  Falls back to the default function on any error.
        """
        # Hot-reload check
        if self._function_path:
            self._try_load_function()

        # Convert to lightweight data objects
        prog_data = self._to_program_data(program)
        parent_data = self._to_program_data(parent) if parent else None
        insp_data = [self._to_program_data(i) for i in inspirations]

        try:
            level, display_data = self._novelty_fn(prog_data, parent_data, insp_data)
            return NoveltyResult(level=level, display_data=display_data)
        except Exception as e:
            logger.error(f"Novelty function raised exception: {e}")
            # Fallback to default
            try:
                level, display_data = default_detect_novelty(
                    prog_data, parent_data, insp_data
                )
                return NoveltyResult(level=level, display_data=display_data)
            except Exception:
                return NoveltyResult(level=NoveltyLevel.NONE)

    # -- internal ------------------------------------------------------------

    def _try_load_function(self) -> bool:
        """Attempt to (re)load the user novelty function.

        Returns *True* if the custom function is active, *False* if
        we fell back to the default.
        """
        if not self._function_path:
            return False

        path = Path(self._function_path)
        if not path.exists():
            self._set_error(f"Novelty function file not found: {path}")
            self._novelty_fn = default_detect_novelty
            return False

        try:
            current_mtime = path.stat().st_mtime

            # Skip reload if file hasn't changed
            if self._cached_module is not None and current_mtime == self._cached_mtime:
                return True

            spec = importlib.util.spec_from_file_location("user_novelty", str(path))
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
            self._novelty_fn = fn
            self._clear_error()
            logger.info(f"Loaded novelty function from {path}")
            return True

        except Exception as e:
            self._set_error(f"Failed to load novelty function from {path}: {e}")
            self._novelty_fn = default_detect_novelty
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
            code_diff=None,
            metadata={},
        )

        result = fn(dummy, dummy_parent, [])

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"Novelty function must return (NoveltyLevel, Optional[dict]), "
                f"got {type(result)}"
            )
        level, data = result
        if not isinstance(level, NoveltyLevel):
            raise TypeError(
                f"First return value must be NoveltyLevel, got {type(level)}"
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
            err_path = os.path.join(self._results_dir, "novelty_error.json")
            with open(err_path, "w") as f:
                json.dump({"error": msg}, f)
        except OSError as exc:
            logger.warning(
                "Could not write novelty error file %s: %s",
                os.path.join(self._results_dir, "novelty_error.json"),
                exc,
            )

    def _remove_error_file(self) -> None:
        if not self._results_dir:
            return
        try:
            err_path = os.path.join(self._results_dir, "novelty_error.json")
            if os.path.exists(err_path):
                os.remove(err_path)
        except OSError as exc:
            logger.warning(
                "Could not remove novelty error file %s: %s",
                os.path.join(self._results_dir, "novelty_error.json"),
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
            code_diff=program.code_diff,
            metadata=program.metadata or {},
        )
