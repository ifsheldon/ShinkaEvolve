"""Custom expert review prioritization for circle packing.

This is an example showing how to write a custom review-prioritization function.
Place a ``review_prioritization.py`` file in your example directory and set
``evo_config.review_prioritization_function_path`` in the task configuration
to point to it.

The function must be named ``prioritize_for_review`` and have the signature::

    def prioritize_for_review(
        program: ProgramData,
        parent: ProgramData | None,
        inspirations: list[ProgramData],
    ) -> tuple[ReviewPriorityLevel, dict | None]:
        ...

``ProgramData`` is a lightweight dataclass with fields including the program
ID, generation, score, metrics, embeddings, code diff, and metadata.
"""

from typing import Dict, List, Optional, Tuple

from shinka.core.review_prioritizer import ProgramData, ReviewPriorityLevel


def prioritize_for_review(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[ReviewPriorityLevel, Optional[Dict]]:
    """Assign review priority based on performance gain from parent.

    Thresholds (customizable per-task):
        * >= 30% gain  ->  HIGH
        * >= 15% gain  ->  MODERATE
        * otherwise    ->  NONE
    """
    if parent is None or not program.correct:
        return ReviewPriorityLevel.NONE, None

    parent_score = parent.combined_score
    program_score = program.combined_score

    if parent_score == 0.0:
        if program_score > 0.0:
            return ReviewPriorityLevel.HIGH, {
                "reason": "First correct solution from zero-score parent",
                "parent_score": 0.0,
                "program_score": round(program_score, 4),
                "gain_pct": None,
            }
        return ReviewPriorityLevel.NONE, None

    gain_pct = (program_score - parent_score) / abs(parent_score)

    display = {
        "parent_score": round(parent_score, 4),
        "program_score": round(program_score, 4),
        "gain_pct": round(gain_pct * 100, 2),
    }

    if gain_pct >= 0.30:
        return ReviewPriorityLevel.HIGH, {
            **display,
            "reason": f"Major score breakthrough ({gain_pct * 100:.1f}% gain)",
        }
    if gain_pct >= 0.15:
        return ReviewPriorityLevel.MODERATE, {
            **display,
            "reason": f"Significant improvement ({gain_pct * 100:.1f}% gain)",
        }
    return ReviewPriorityLevel.NONE, display
