"""Mock novelty detection for interactive sandbox testing.

Randomly assigns a novelty level so you can test the UI without waiting
for real performance breakthroughs.
"""

import random
from typing import Dict, List, Optional, Tuple

from shinka.core.novelty_detector import NoveltyLevel, ProgramData


def detect_novelty(
    program: ProgramData,
    parent: Optional[ProgramData],
    inspirations: List[ProgramData],
) -> Tuple[NoveltyLevel, Optional[Dict]]:
    level = random.choice([NoveltyLevel.NONE, NoveltyLevel.MODERATE, NoveltyLevel.HIGH])

    if level == NoveltyLevel.NONE:
        return NoveltyLevel.NONE, None

    return level, {
        "reason": f"Mock {level.value} novelty (random)",
        "program_score": round(program.combined_score, 4),
        "parent_score": round(parent.combined_score, 4) if parent else None,
        "gain_pct": round(random.uniform(5, 60), 2),
    }
