"""Offline pricing and seed-only initialization for the interactive sandbox."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from shinka.core import ShinkaEvolveInteractiveRunner
from shinka.interactive.interactive_db import (
    InteractiveDatabase,
    InteractiveStatus,
    RunState,
)
from shinka.pricing import (
    CatalogSnapshot,
    ModelPrice,
    PricingCatalog,
    load_run_pricing_snapshot,
    write_run_pricing_snapshot,
)


def prepare_mock_pricing(results_dir: Path) -> None:
    """Install a validated, zero-cost catalog before runner construction."""
    entries = (
        ModelPrice("mock-llm", "mock-llm", "mock", "llm", 0.0, 0.0),
        ModelPrice("mock-embedding", "mock-embedding", "mock", "embedding", 0.0),
    )
    digest = hashlib.sha256(
        json.dumps([asdict(entry) for entry in entries], sort_keys=True).encode()
    ).hexdigest()
    results_dir.mkdir(parents=True, exist_ok=True)
    write_run_pricing_snapshot(
        CatalogSnapshot(PricingCatalog(entries), "bundled", None, None, digest),
        results_dir,
    )
    if load_run_pricing_snapshot(results_dir) is None:
        raise RuntimeError("Could not load the sandbox pricing snapshot")


async def initialize_only(runner: ShinkaEvolveInteractiveRunner) -> None:
    """Evaluate and copy initial programs without submitting evolution jobs."""
    runner.start_time = time.time()
    runner.last_progress_time = runner.start_time
    try:
        await runner._setup_async()
        programs = runner.db.get_all_programs()
        if len(programs) != runner.db_config.num_islands or any(
            program.generation != 0 for program in programs
        ):
            raise RuntimeError(
                "Seed initialization did not produce one node per island"
            )
        InteractiveDatabase(
            str(Path(runner.results_dir) / "programs.sqlite")
        ).write_status(
            InteractiveStatus(
                run_state=RunState.WAITING_FOR_START,
                generation=0,
                best_score=max(program.combined_score or 0 for program in programs),
                total_programs=len(programs),
                target_generations=runner.evo_config.num_generations,
            )
        )
    finally:
        await runner._cleanup_async()
