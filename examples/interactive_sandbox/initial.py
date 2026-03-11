# EVOLVE-BLOCK-START
import random


def compute(seed: int = 42) -> float:
    """Compute a value. The goal is to maximize this."""
    random.seed(seed)
    x = sum(random.gauss(0, 1) for _ in range(10))
    return abs(x)


def run_experiment(seed: int = 1) -> float:
    """Entry point called by the evaluator."""
    return compute(seed)


# EVOLVE-BLOCK-END
