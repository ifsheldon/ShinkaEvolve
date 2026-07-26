"""Mock review prioritization for interactive sandbox testing.

Randomly assigns a review priority level so you can test the UI without waiting
for real performance breakthroughs.
"""

import random
from typing import Dict, List, Optional, Tuple

from shinka.core.review_prioritizer import ProgramData, ReviewPriorityLevel


def prioritize_for_review(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[ReviewPriorityLevel, Optional[Dict]]:
    level = random.choice(
        [
            ReviewPriorityLevel.NONE,
            ReviewPriorityLevel.MODERATE,
            ReviewPriorityLevel.HIGH,
        ]
    )

    if level == ReviewPriorityLevel.NONE:
        return ReviewPriorityLevel.NONE, None

    return level, {
        "reason": f"Mock {level.value} review priority (random)",
        "program_score": round(program.combined_score, 4),
        "parent_score": round(parent.combined_score, 4) if parent else None,
        "gain_pct": round(random.uniform(5, 60), 2),
    }
