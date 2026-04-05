"""Custom novelty detection for circle packing.

This is an example showing how to write a custom novelty function.
Place a ``novelty.py`` file in your example directory and set
``evo_config.novelty_function_path`` in the task YAML config to point to it.

The function must be named ``detect_novelty`` (or whatever is configured
via ``novelty_function_name``) and have the signature::

    def detect_novelty(
        program: ProgramData,
        parent: ProgramData | None,
        inspirations: list[ProgramData],
    ) -> tuple[NoveltyLevel, dict | None]:
        ...

``ProgramData`` is a lightweight dataclass with fields: id, generation,
combined_score, correct, public_metrics, private_metrics, embedding,
code_diff, metadata.
"""

from shinka.core.novelty_detector import NoveltyLevel, ProgramData
from typing import Dict, List, Optional, Tuple


def detect_novelty(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[NoveltyLevel, Optional[Dict]]:
    """Detect novelty based on performance gain from parent.

    Thresholds (customizable per-task):
        * >= 30% gain  ->  HIGH
        * >= 15% gain  ->  MODERATE
        * otherwise    ->  NONE
    """
    if parent is None or not program.correct:
        return NoveltyLevel.NONE, None

    parent_score = parent.combined_score
    program_score = program.combined_score

    if parent_score == 0.0:
        if program_score > 0.0:
            return NoveltyLevel.HIGH, {
                "reason": "First correct solution from zero-score parent",
                "parent_score": 0.0,
                "program_score": round(program_score, 4),
                "gain_pct": None,
            }
        return NoveltyLevel.NONE, None

    gain_pct = (program_score - parent_score) / abs(parent_score)

    display = {
        "parent_score": round(parent_score, 4),
        "program_score": round(program_score, 4),
        "gain_pct": round(gain_pct * 100, 2),
    }

    if gain_pct >= 0.30:
        return NoveltyLevel.HIGH, {
            **display,
            "reason": f"Major score breakthrough ({gain_pct * 100:.1f}% gain)",
        }
    if gain_pct >= 0.15:
        return NoveltyLevel.MODERATE, {
            **display,
            "reason": f"Significant improvement ({gain_pct * 100:.1f}% gain)",
        }
    return NoveltyLevel.NONE, display
