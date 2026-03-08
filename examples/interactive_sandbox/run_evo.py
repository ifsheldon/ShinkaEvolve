#!/usr/bin/env python3
"""
Interactive Sandbox — a minimal evolution example with a **mock LLM**.

Purpose
-------
Run this to exercise the full ShinkaEvolve + interactive pipeline without
needing real API keys.  The mock LLM prints every prompt it receives,
returns a trivially modified program, and the evaluator assigns a random
score in [0, 10].

Usage
-----
    # From the ShinkaEvolve root:
    cd examples/interactive_sandbox
    python run_evo.py

    # Then in another terminal, start evolve-shell to see the UI:
    cd ../../evolve-shell && npm run dev

    # The evolve-shell backend should point at the same DB:
    cd ../../evolve-shell/backend
    SHINKA_SEARCH_ROOT=../../examples/interactive_sandbox python -m uvicorn main:app --reload
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

# Set dummy API key so the embedding client doesn't crash on init.
# We never actually call the embedding API (embedding_model=None disables it).
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-sandbox-testing")

from shinka.core import EvolutionRunner, EvolutionConfig, AsyncInteractiveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.llm.providers.result import QueryResult


IS_RESUME_MODE = False

# ── Mock LLM ────────────────────────────────────────────────────────────────

_CALL_COUNTER = 0


def _mock_query(
    self,
    msg: str,
    system_msg: str,
    msg_history: List[Dict] = [],
    llm_kwargs: Optional[Dict] = None,
    model_sample_probs: Optional[List[float]] = None,
    model_posterior: Optional[List[float]] = None,
) -> QueryResult:
    """
    Drop-in replacement for ``LLMClient.query``.

    * Prints the full system + user prompt so you can inspect them.
    * Returns a trivially modified Python program (changes a constant).
    * Cost is always $0.
    """
    global _CALL_COUNTER
    _CALL_COUNTER += 1

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  MOCK LLM — call #{_CALL_COUNTER}")
    print(sep)
    print(
        f"[SYSTEM MESSAGE]\n{textwrap.shorten(system_msg, width=600, placeholder=' ...')}"
    )
    print(f"\n[USER MESSAGE]\n{textwrap.shorten(msg, width=1200, placeholder=' ...')}")
    print(sep)

    # Produce a simple "full" replacement program.
    # We randomise one constant so each generation is slightly different.
    rand_val = random.randint(5, 50)
    rand_mul = round(random.uniform(0.5, 3.0), 2)

    fake_code = textwrap.dedent(f"""\
        import random

        def compute(seed: int = 42) -> float:
            random.seed(seed)
            x = sum(random.gauss(0, 1) for _ in range({rand_val}))
            return abs(x) * {rand_mul}

        def run_experiment(seed: int = 1) -> float:
            return compute(seed)
    """)

    fake_name = f"variant_{_CALL_COUNTER}"
    fake_desc = f"Changed loop count to {rand_val} and multiplier to {rand_mul}."

    content = (
        f"<NAME>{fake_name}</NAME>\n"
        f"<DESCRIPTION>{fake_desc}</DESCRIPTION>\n\n"
        f"```python\n{fake_code}```\n"
    )

    print(f"[MOCK RESPONSE]  name={fake_name}  (loop={rand_val}, mul={rand_mul})")
    print()

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=[],
        model_name="mock-llm",
        kwargs=llm_kwargs or {},
        input_tokens=len(msg) // 4,
        output_tokens=len(content) // 4,
        cost=0.0,
        model_posteriors={"mock-llm": 1.0},
    )


# ── Mock Embedding ──────────────────────────────────────────────────────────

_EMBED_DIM = 256
_EMBED_COUNTER = 0


def _mock_get_embedding(
    self,
    code,
):
    """
    Drop-in replacement for ``EmbeddingClient.get_embedding``.

    Returns a deterministic random embedding vector seeded from the hash of
    the input code, so identical code always produces the same vector.
    """
    global _EMBED_COUNTER
    _EMBED_COUNTER += 1

    if isinstance(code, list):
        embeddings = []
        for c in code:
            seed = int(hashlib.sha256(c.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(_EMBED_DIM)]
            embeddings.append(vec)
        print(f"[MOCK EMBEDDING #{_EMBED_COUNTER}]  batch={len(code)} dim={_EMBED_DIM}")
        return embeddings, 0.0

    seed = int(hashlib.sha256(code.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(_EMBED_DIM)]
    print(f"[MOCK EMBEDDING #{_EMBED_COUNTER}]  len(code)={len(code)} dim={_EMBED_DIM}")
    return vec, 0.0


def _mock_embed_init(self, model_name="mock-embedding", verbose=False):
    """Skip real API client creation."""
    self.client = None
    self.model = model_name
    self.model_name = model_name
    self.verbose = verbose


def _mock_async_embed_init(self, model_name="mock-embedding", verbose=False):
    """Skip real async API client creation."""
    self.async_client = None
    self.model = model_name
    self.model_name = model_name
    self.provider = None
    self.verbose = verbose


async def _mock_embed_async(self, code):
    """Async drop-in for ``AsyncEmbeddingClient.embed_async``."""
    global _EMBED_COUNTER
    _EMBED_COUNTER += 1

    if isinstance(code, str):
        code = [code]
        single = True
    else:
        single = False

    embeddings = []
    for c in code:
        seed = int(hashlib.sha256(c.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        embeddings.append([rng.gauss(0, 1) for _ in range(_EMBED_DIM)])

    print(
        f"[MOCK ASYNC EMBEDDING #{_EMBED_COUNTER}]  batch={len(code)} dim={_EMBED_DIM}"
    )
    if single:
        return embeddings[0], 0.0
    return embeddings, 0.0


def _mock_get_kwargs(self, model_sample_probs=None):
    """Return mock kwargs without resolving model backend via pricing.csv."""
    return {
        "model_name": "mock-llm",
        "temperature": 0.7,
        "max_output_tokens": 2048,
    }


# ── Monkey-patch the LLM + Embedding clients ───────────────────────────────

from shinka.llm.llm import LLMClient, AsyncLLMClient  # noqa: E402
from shinka.embed.embedding import EmbeddingClient, AsyncEmbeddingClient  # noqa: E402


async def _mock_async_query(
    self,
    msg: str,
    system_msg: str,
    msg_history: List[Dict] = [],
    llm_kwargs: Optional[Dict] = None,
    model_sample_probs: Optional[List[float]] = None,
    model_posterior: Optional[List[float]] = None,
) -> QueryResult:
    """Async drop-in replacement for ``AsyncLLMClient.query``."""
    # Delegate to the sync mock — the logic is identical.
    return _mock_query(
        self,
        msg,
        system_msg,
        msg_history,
        llm_kwargs,
        model_sample_probs,
        model_posterior,
    )


LLMClient.query = _mock_query
LLMClient.get_kwargs = _mock_get_kwargs
AsyncLLMClient.query = _mock_async_query
AsyncLLMClient.get_kwargs = _mock_get_kwargs
EmbeddingClient.__init__ = _mock_embed_init
EmbeddingClient.get_embedding = _mock_get_embedding
AsyncEmbeddingClient.__init__ = _mock_async_embed_init
AsyncEmbeddingClient.embed_async = _mock_embed_async


# ── Configuration ───────────────────────────────────────────────────────────

job_config = LocalJobConfig(eval_program_path="evaluate.py")

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=5,
    archive_size=10,
    num_archive_inspirations=2,
    num_top_k_inspirations=1,
    parent_selection_strategy="power_law",
    exploitation_alpha=1.0,
    exploitation_ratio=0.3,
    migration_interval=10,  # enable migration across islands
)

evo_config = EvolutionConfig(
    task_sys_msg=(
        "You are evolving a simple Python function that returns a number. "
        "The goal is to maximise the returned value. "
        "Be creative — change constants, restructure the logic, try new ideas."
    ),
    patch_types=["full"],  # only full rewrites (easiest to mock)
    patch_type_probs=[1.0],
    num_generations=100,
    max_parallel_jobs=2,
    max_patch_resamples=1,
    max_patch_attempts=1,
    language="python",
    llm_models=["mock-llm"],  # not used — we monkeypatch query()
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=2048,
    ),
    embedding_model="mock-embedding",  # use mocked embedding
    code_embed_sim_threshold=0.95,  # enable novelty rejection
    init_program_path="initial.py",
    results_dir="results_sandbox",
    interactive_mode=True,  # ← enable interactive tables
    # for resume mode
    interaction_mode="manual" if IS_RESUME_MODE else "auto",
)


# ── Main ────────────────────────────────────────────────────────────────────


def _clean_previous_run() -> None:
    """Remove stale artefacts from a previous run so we always start fresh."""
    here = Path(__file__).resolve().parent
    db_file = here / "evolution_db.sqlite"
    results = here / "results_sandbox"

    removed = []
    for p in (db_file, results):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(str(p.name))

    if removed:
        print(f"[clean] Removed stale artefacts: {', '.join(removed)}")


def main():
    if not IS_RESUME_MODE:
        _clean_previous_run()

    print("=" * 72)
    print("  Interactive Sandbox — Mock Evolution")
    print("  DB:  evolution_db.sqlite")
    print("  Interactive mode: ON")
    print("=" * 72)
    print()
    print("Tip: start the evolve-shell UI in another terminal to interact.")
    print("     You can pause/resume/suggest/merge from the web interface.\n")

    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    runner.run()


async def main_async():
    """Async version using AsyncInteractiveRunner for 5-10x faster evolution."""
    if not IS_RESUME_MODE:
        _clean_previous_run()

    print("=" * 72)
    print("  Interactive Sandbox — Mock Evolution (ASYNC)")
    print("  DB:  evolution_db.sqlite")
    print("  Interactive mode: ON")
    print("=" * 72)
    print()
    print("Tip: start the evolve-shell UI in another terminal to interact.")
    print("     You can pause/resume/suggest/merge from the web interface.\n")

    runner = AsyncInteractiveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=2,
        max_proposal_jobs=4,
        verbose=True,
    )
    await runner.run()


if __name__ == "__main__":
    import sys

    if "--async" in sys.argv:
        import asyncio

        asyncio.run(main_async())
    else:
        main()
