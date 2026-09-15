"""Regression coverage for reasoning eligibility and provider-free DB consumers."""

from __future__ import annotations

import asyncio
import json
import socket
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.async_interactive_runner import ShinkaEvolveInteractiveRunner
from shinka.core.review_prioritizer import ReviewPrioritizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.edit.async_apply import get_reasoning_embedding_async
from shinka.reasoning import extract_reasoning_text, minimum_reasoning_distance
from shinka.reasoning_features import compute_reasoning_features


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression must never turn these synthetic scenarios into paid calls."""

    def blocked(*args: object, **kwargs: object) -> None:
        """Reject every attempted network connection in this test."""
        raise AssertionError("Network access is forbidden in reasoning tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        [],
        {},
        "",
        " \n\t",
        "...",
        "—",
        "NONE",
        '"None."',
        "‘ n/A ’!",
        "NULL",
        "na",
        "undefined",
        "unknown",
        "no description",
        "no reasoning",
        "initial program",
        " Initial   program SETUP. ",
        "Initial program setup (fallback)",
    ],
)
def test_placeholder_does_not_call_embedding_client(value: object) -> None:
    """Both metadata fields must be filtered before an API request is made."""
    client = SimpleNamespace(
        embed_async=AsyncMock(side_effect=AssertionError("called"))
    )
    result = asyncio.run(
        get_reasoning_embedding_async(
            {"patch_description": value, "llm_result": {"thought": value}}, client
        )
    )
    assert result == (None, 0.0)
    client.embed_async.assert_not_awaited()


@pytest.mark.parametrize(
    ("description", "thought", "expected"),
    [
        ("Greedy", None, "Greedy"),
        ("使用动态规划减少重复计算", "none", "使用动态规划减少重复计算"),
        ("none", "Reuse the residual graph", "Reuse the residual graph"),
        (
            "Initial program setup",
            "Use shortest augmenting paths",
            "Use shortest augmenting paths",
        ),
        (
            "None of the previous approaches handles congestion",
            None,
            "None of the previous approaches handles congestion",
        ),
        (
            "  Cache paths  ",
            " Reuse unchanged edges ",
            "Cache paths\n\nReuse unchanged edges",
        ),
    ],
)
def test_generation_embeds_only_eligible_parts(
    description: str | None, thought: str | None, expected: str | None
) -> None:
    """Valid thoughts rescue placeholder descriptions without embedding boilerplate."""
    client = SimpleNamespace(embed_async=AsyncMock(return_value=([1.0, 2.0], 0.125)))
    result = asyncio.run(
        get_reasoning_embedding_async(
            {"patch_description": description, "llm_result": {"thought": thought}},
            client,
        )
    )
    assert result == ([1.0, 2.0], 0.125)
    client.embed_async.assert_awaited_once_with(expected)


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [float("nan"), 1], [float("inf")], [True, 1], ["1", 2]]
)
def test_invalid_provider_vector_preserves_cost(vector: object) -> None:
    """A billed but unusable response must not become a vector or lose its cost."""
    client = SimpleNamespace(embed_async=AsyncMock(return_value=(vector, 0.25)))
    assert asyncio.run(
        get_reasoning_embedding_async({"patch_description": "Cache paths"}, client)
    ) == (None, 0.25)


def test_sync_provider_and_truncation_after_filtering() -> None:
    """The synchronous provider adapter obeys the same content and length contract."""
    client = SimpleNamespace(get_embedding=Mock(return_value=([1.0, 0.0], 0.1)))
    asyncio.run(
        get_reasoning_embedding_async(
            {
                "patch_description": "none",
                "llm_result": {"thought": "Cache paths across iterations"},
            },
            client,
            max_chars=11,
        )
    )
    client.get_embedding.assert_called_once_with("Cache paths")


@pytest.mark.parametrize("failed_evaluation", [False, True])
@pytest.mark.parametrize(
    ("description", "llm_metadata", "expected"),
    [
        ("Initial program setup", None, None),
        ("Use a greedy seed", None, "Use a greedy seed"),
        (
            "Initial program setup",
            {
                "patch_description": "none",
                "llm_result": {"thought": "Seed the residual graph"},
            },
            "Seed the residual graph",
        ),
    ],
)
def test_initialization_persists_only_real_reasoning(
    tmp_path: Path,
    failed_evaluation: bool,
    description: str | None,
    llm_metadata: dict[str, Any] | None,
    expected: str | None,
) -> None:
    """Exercise seed evaluation, fallback, embedding, persistence, and cost accounting."""
    runner = object.__new__(ShinkaEvolveRunner)
    runner.results_dir = str(tmp_path)
    runner.lang_ext = "py"
    runner.verbose = False
    runner.max_proposal_jobs = runner.max_evaluation_jobs = runner.max_db_workers = 1
    runner.db_config = SimpleNamespace(max_stdout_log_chars=None)
    runner.scheduler = SimpleNamespace(
        run=Mock(
            side_effect=RuntimeError("evaluation failed")
            if failed_evaluation
            else None,
            return_value=(
                {"correct": {"correct": True}, "metrics": {"combined_score": 1.0}},
                0.01,
            ),
        )
    )
    runner._get_code_embedding_async = AsyncMock(return_value=([], 0.0))
    client = SimpleNamespace(embed_async=AsyncMock(return_value=([1.0, 2.0], 0.25)))

    async def embed(metadata: dict[str, Any]) -> tuple[list[float] | None, float]:
        """Apply the real reasoning policy through the stubbed provider."""
        return await get_reasoning_embedding_async(metadata, client)

    runner._get_reasoning_embedding_async = embed
    runner.async_db = SimpleNamespace(
        add_program_async=AsyncMock(return_value=[]),
        update_program_metadata_async=AsyncMock(),
    )
    runner.total_api_cost = 0.0
    runner.meta_summarizer = None
    runner.llm_selection = None
    runner.stuck_detection_count = 0
    asyncio.run(
        runner._setup_initial_program_with_metadata(
            "print(1)", "seed", description, 0.0, llm_metadata
        )
    )
    stored = runner.async_db.add_program_async.call_args.args[0]
    assert stored.reasoning_embedding == ([1.0, 2.0] if expected else [])
    assert runner.total_api_cost == (0.25 if expected else 0.0)
    if expected:
        client.embed_async.assert_awaited_once_with(expected)
        assert extract_reasoning_text(stored.metadata) == expected
    else:
        client.embed_async.assert_not_awaited()


def test_database_readers_and_novelty_skip_legacy_invalid_vectors(
    tmp_path: Path,
) -> None:
    """Raw historic corruption cannot bypass regular or thread-safe readers."""
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "db.sqlite"), num_islands=1),
        embedding_model="",
    )
    try:
        for index, program_id in enumerate(["placeholder", "valid", "zero", "query"]):
            db.add(
                Program(
                    id=program_id,
                    code="print(1)",
                    correct=True,
                    complexity=1,
                    generation=index,
                    timestamp=float(index),
                    island_idx=0,
                    combined_score=float(index),
                    embedding=[1.0, 0.0],
                    reasoning_embedding=[1.0, 0.0],
                    metadata={"patch_description": "Cache paths"},
                ),
                defer_maintenance=True,
            )
        db.conn.execute(
            "UPDATE programs SET metadata = ?, reasoning_embedding_pca_2d = '[9,9]', reasoning_embedding_cluster_id = 3 WHERE id = 'placeholder'",
            (json.dumps({"patch_description": "none"}),),
        )
        db.conn.execute(
            "UPDATE programs SET reasoning_embedding = '[0,0]' WHERE id = 'zero'"
        )
        db.conn.commit()
        for program in [
            db.get("placeholder"),
            db.get_programs_by_generation_thread_safe(0)[0],
        ]:
            assert program.reasoning_embedding == []
            assert program.reasoning_embedding_pca_2d == []
            assert program.reasoning_embedding_cluster_id is None
        top = {program.id: program for program in db.get_top_programs_thread_safe(10)}
        assert top["valid"].reasoning_embedding == [1.0, 0.0]
        assert top["zero"].reasoning_embedding == []
        assert db.get_all_embeddings_before("query")[1] == [[1.0, 0.0]]
        assert db.compute_reasoning_similarity([1.0, 0.0], 0) == [1.0, 1.0]
        assert db.compute_reasoning_similarity_thread_safe([1.0, 0.0], 0) == [1.0, 1.0]
        assert db.compute_reasoning_similarity([1.0, 0.0, 0.0], 0) == []
        program = db.get("valid")
        program.metadata["patch_description"] = "none"
        program.id = "mutated"
        db.add(program, defer_maintenance=True)
        assert (
            db.conn.execute(
                "SELECT reasoning_embedding FROM programs WHERE id='mutated'"
            ).fetchone()[0]
            == "[]"
        )
    finally:
        db.close()


def test_reasoning_metrics_do_not_invent_novelty_for_missing_comparisons() -> None:
    """Zero, nonfinite, and mismatched predecessors cannot become novelty scores."""
    program = Program(
        id="query",
        code="",
        metadata={"patch_description": "Cache paths"},
        reasoning_embedding=[1.0, 0.0],
    )
    metrics = ReviewPrioritizer().compute_priority_metrics(
        program, None, [], [[0.0, 0.0], [1.0], [float("nan"), 1.0]]
    )
    assert metrics["dissimilarity_reasoning"] is None
    assert minimum_reasoning_distance([1.0, 0.0], [[-1.0, 0.0]]) == 2.0
    from shinka.reasoning_features import historical_reasoning_distances

    distances = historical_reasoning_distances(
        [
            ("short", 0, 0.0, [1.0]),
            ("first", 1, 1.0, [1.0, 0.0]),
            ("tied", 1, 1.0, [-1.0, 0.0]),
            ("later", 1, 2.0, [0.0, 1.0]),
        ]
    )
    assert distances == {"short": None, "first": None, "tied": None, "later": 1.0}


def test_projection_refresh_needs_no_code_vectors_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both runtime refresh paths work offline and clear obsolete cluster data."""
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "db.sqlite"), num_islands=1),
        embedding_model="",
    )
    monkeypatch.setattr(
        db,
        "_ensure_embedding_client",
        Mock(side_effect=AssertionError("provider constructed")),
    )
    try:
        for index, vector in enumerate(
            ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0])
        ):
            db.add(
                Program(
                    id=str(index),
                    code="",
                    complexity=1,
                    island_idx=0,
                    generation=index,
                    reasoning_embedding=vector,
                    metadata={"patch_description": "Explore a different path"},
                ),
                defer_maintenance=True,
            )
        db._recompute_embeddings_and_clusters_thread_safe()
        assert all(
            len(db.get(str(index)).reasoning_embedding_pca_2d) == 2
            for index in range(4)
        )
        first = [
            tuple(row)
            for row in db.conn.execute(
                "SELECT reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs ORDER BY id"
            )
        ]
        db._recompute_embeddings_and_clusters()
        assert first == [
            tuple(row)
            for row in db.conn.execute(
                "SELECT reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs ORDER BY id"
            )
        ]
        db.conn.execute(
            "UPDATE programs SET metadata = '{\"patch_description\":\"none\"}' WHERE id='0'"
        )
        db.conn.commit()
        db._recompute_embeddings_and_clusters_thread_safe()
        assert all(
            tuple(row) == ("[]", None)
            for row in db.conn.execute(
                "SELECT reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs"
            )
        )
    finally:
        db.close()
    assert (
        compute_reasoning_features({str(index): [1.0, 1.0] for index in range(4)}) == {}
    )


@pytest.mark.parametrize("action", ["normal", "suggest", "merge"])
@pytest.mark.parametrize("thought", [None, "Reuse the residual graph"])
def test_proposal_paths_store_filtered_reasoning_and_account_for_cost(
    tmp_path: Path, action: str, thought: str | None
) -> None:
    """Exercise the real normal/Suggest/Merge orchestration with provider stubs."""
    runner = object.__new__(ShinkaEvolveInteractiveRunner)
    runner.results_dir = str(tmp_path)
    runner.lang_ext = "py"
    runner.verbose = False
    runner.meta_summarizer = runner.llm_selection = runner.novelty_judge = None
    runner.running_jobs, runner.submitted_jobs = [], {}
    runner.active_proposal_tasks = {}
    runner.max_evaluation_jobs = 4
    runner.total_api_cost = 0.0
    runner.slot_available = asyncio.Event()
    runner._update_avg_proposal_cost = Mock()
    runner._read_file_async = AsyncMock(return_value="print(1)")
    runner.event_notifier = SimpleNamespace(notify_queued=AsyncMock())
    runner._get_banned_ids = Mock(return_value=set())
    parent = Program(id="parent", code="print(1)", correct=True, island_idx=0)
    runner.async_db = SimpleNamespace(
        sample_with_fix_mode_async=AsyncMock(return_value=(parent, [], [], False))
    )
    runner.db_config = SimpleNamespace(parent_selection_strategy="weighted")
    runner.evo_config = SimpleNamespace(max_novelty_attempts=1, max_patch_resamples=1)
    runner.scheduler = SimpleNamespace(
        submit_async_nonblocking=AsyncMock(return_value="job")
    )
    runner._submit_evaluation_job_with_slot = AsyncMock(
        return_value=("job", 0, 1.0, 1.0, 0)
    )
    runner._run_patch_async = AsyncMock(
        return_value=(
            "diff",
            {
                "patch_description": "none",
                "llm_result": {"thought": thought},
                "api_costs": 0.1,
            },
            True,
        )
    )
    runner._get_code_embedding_async = AsyncMock(return_value=(None, 0.2))
    client = SimpleNamespace(embed_async=AsyncMock(return_value=([1.0, 2.0], 0.3)))

    async def embed(metadata: dict[str, Any]) -> tuple[list[float] | None, float]:
        """Apply the real reasoning policy through the stubbed provider."""
        return await get_reasoning_embedding_async(metadata, client)

    runner._get_reasoning_embedding_async = embed
    if action == "normal":
        job = asyncio.run(
            runner._generate_evolved_proposal(
                1,
                "task",
                str(tmp_path / "main.py"),
                str(tmp_path),
                None,
                None,
                None,
                1.0,
                None,
                0,
            )
        )
    else:
        job = asyncio.run(
            runner._generate_interactive_proposal_async(
                1,
                "task",
                parent,
                [],
                [],
                "cross" if action == "merge" else "diff",
                "Consider cached paths",
                action,
            )
        )
    assert job is not None
    assert job.reasoning_embedding == ([1.0, 2.0] if thought else None)
    assert job.embed_cost == pytest.approx(0.5 if thought else 0.2)
    assert runner.total_api_cost == pytest.approx(0.6 if thought else 0.3)
    runner.event_notifier.notify_queued.assert_awaited_once()
    if thought:
        client.embed_async.assert_awaited_once_with(thought)
    else:
        client.embed_async.assert_not_awaited()


def test_island_sql_copies_and_migration_clear_invalid_reasoning(
    tmp_path: Path,
) -> None:
    """Direct SQL island operations cannot propagate old invalid vector fields."""
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "db.sqlite"), num_islands=2),
        embedding_model="",
    )
    try:
        program = Program(
            id="original",
            code="print(1)",
            correct=True,
            complexity=1,
            island_idx=0,
            metadata={"patch_description": "Cache paths"},
            reasoning_embedding=[1.0, 2.0],
        )
        db.add(program, defer_maintenance=True)
        program.metadata["patch_description"] = "none"
        program.reasoning_embedding_pca_2d = [9.0, 9.0]
        program.reasoning_embedding_cluster_id = 3
        ids = db.island_manager.copy_program_to_islands(program)
        assert ids
        for program_id in ids:
            assert tuple(
                db.conn.execute(
                    "SELECT reasoning_embedding, reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs WHERE id=?",
                    (program_id,),
                ).fetchone()
            ) == ("[]", "[]", None)
        db.conn.execute(
            "UPDATE programs SET metadata = ?, reasoning_embedding_pca_2d = '[9,9]', reasoning_embedding_cluster_id=3 WHERE id='original'",
            (json.dumps(program.metadata),),
        )
        db.conn.commit()
        raw = dict(
            db.conn.execute("SELECT * FROM programs WHERE id='original'").fetchone()
        )
        spawned = db.island_manager._copy_program_to_island(
            raw, 2, None, "initial", is_root=True
        )
        assert tuple(
            db.conn.execute(
                "SELECT reasoning_embedding, reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs WHERE id=?",
                (spawned,),
            ).fetchone()
        ) == ("[]", "[]", None)
        db.island_manager.migration_strategy._migrate_program("original", 0, 1, 1)
        assert tuple(
            db.conn.execute(
                "SELECT reasoning_embedding, reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs WHERE id='original'"
            ).fetchone()
        ) == ("[]", "[]", None)
    finally:
        db.close()


def test_metadata_writer_keeps_vectors_consistent_with_source_text(
    tmp_path: Path,
) -> None:
    """Async metadata updates preserve valid vectors and clear newly invalid ones."""
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "db.sqlite"), num_islands=1),
        embedding_model="",
    )
    db.add(
        Program(
            id="program",
            code="print(1)",
            complexity=1,
            metadata={"patch_description": "Cache paths"},
            reasoning_embedding=[1.0, 2.0],
            reasoning_embedding_pca_2d=[0.0, 0.0],
            reasoning_embedding_cluster_id=0,
        ),
        defer_maintenance=True,
    )

    async def exercise() -> None:
        """Exercise metadata updates through the asynchronous database writer."""
        async_db = AsyncProgramDatabase(sync_db=db)
        try:
            await async_db.update_program_metadata_async(
                "program", {"patch_description": "Cache paths", "note": "reviewed"}
            )
            assert db.get("program").reasoning_embedding == [1.0, 2.0]
            await async_db.update_program_metadata_async(
                "program", {"patch_description": "none"}
            )
            assert tuple(
                db.conn.execute(
                    "SELECT reasoning_embedding, reasoning_embedding_pca_2d, reasoning_embedding_cluster_id FROM programs WHERE id='program'"
                ).fetchone()
            ) == ("[]", "[]", None)
        finally:
            await async_db.close_async()

    try:
        asyncio.run(exercise())
    finally:
        db.close()
