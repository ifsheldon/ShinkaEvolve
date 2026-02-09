"""
Minimal evaluator for the interactive sandbox example.

Returns a random score between 0 and 10, plus a 'correct' flag that is True
~80 % of the time.  This lets us exercise the full evolution pipeline (parent
selection, scoring, archive, etc.) without any real computation.
"""

import os
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval


def validate_result(result: Any) -> Tuple[bool, Optional[str]]:
    """Accept any numeric result; reject non-numeric garbage."""
    if isinstance(result, (int, float)):
        return True, None
    return False, f"Expected a number, got {type(result).__name__}"


def aggregate_metrics(
    results: List[Any],
    results_dir: str,
) -> Dict[str, Any]:
    """
    Return a random combined_score in [0, 10].
    Also echo back the raw result so we can inspect it in the UI.
    """
    raw_value = float(results[0]) if results else 0.0
    # The score is random so we don't need a real LLM to see the tree grow.
    score = random.uniform(0, 10)
    return {
        "combined_score": round(score, 4),
        "public": {
            "raw_value": round(raw_value, 4),
            "random_score": round(score, 4),
        },
        "private": {},
        "text_feedback": f"Raw output was {raw_value:.4f}. Random score: {score:.4f}.",
    }


def main(program_path: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    def _agg(r: list) -> dict:
        return aggregate_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,
        validate_fn=validate_result,
        aggregate_metrics_fn=_agg,
    )

    tag = "OK" if correct else "FAIL"
    print(f"[evaluate] {tag}  score={metrics.get('combined_score', '?')}  "
          f"error={error_msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive sandbox evaluator")
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
