"""Offline regression coverage for persistent meta memory and runner resume."""

import asyncio
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any provider/network connection while allowing asyncio local sockets."""
    original_connect = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network access is forbidden in meta resume tests")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


def program(program_id="pending", generation=2):
    """Build a candidate that can participate in real meta analysis."""
    return Program(
        id=program_id,
        code="print(1)",
        language="python",
        generation=generation,
        correct=True,
        combined_score=1.0,
        metadata={"patch_name": "strategy"},
    )


def summarizer():
    """Return an async summarizer with no provider client."""
    return AsyncMetaSummarizer(MetaSummarizer(async_mode=True))


def test_pending_programs_and_previous_insights_survive_resume(tmp_path):
    async def run():
        original = summarizer()
        original.configure_persistence(tmp_path, resuming=False)
        original.add_evaluated_program(program())
        original.sync_summarizer.meta_summary = "Previous summaries"
        original.sync_summarizer.meta_scratch_pad = "Previous insights"
        original.sync_summarizer.meta_recommendations = "1. Previous recommendation"
        original.sync_summarizer.meta_recommendations_history = ["1. Older advice"]
        original.sync_summarizer.total_programs_processed = 4
        await original.checkpoint_async()

        resumed = summarizer()
        resumed.configure_persistence(tmp_path, resuming=True)
        assert resumed.get_current() == original.get_current()
        assert resumed.get_recommendations_history_count() == 1
        assert resumed.get_total_programs_processed() == 4
        assert not resumed.should_update_meta(2)
        resumed.add_evaluated_program(program("new", generation=3))
        assert resumed.should_update_meta(2)
        assert [item.id for item in resumed.evaluated_since_last_meta] == [
            "pending",
            "new",
        ]

    asyncio.run(run())


def test_historical_text_is_not_a_checkpoint(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    historical = meta / "meta_15.txt"
    historical.write_text("# GLOBAL INSIGHTS SCRATCHPAD\nHistorical text")
    before = historical.read_bytes()
    resumed = summarizer()
    resumed.configure_persistence(tmp_path, resuming=True)
    assert resumed.get_current() == (None, None, None)
    assert resumed.get_unprocessed_program_count() == 0
    assert historical.read_bytes() == before
    assert not (meta / "state.json").exists()


@pytest.mark.parametrize(
    "invalid", [{}, {"meta_summary": 12}, {"unprocessed_programs": [None]}]
)
def test_invalid_checkpoint_never_partially_restores_or_gets_overwritten(
    tmp_path, invalid
):
    original = summarizer()
    original.configure_persistence(tmp_path, resuming=False)
    original.sync_summarizer.meta_summary = "Original summaries"
    original.sync_summarizer.save_meta_state(tmp_path / "meta" / "state.json")
    path = tmp_path / "meta" / "state.json"
    data = json.loads(path.read_text())
    data = invalid if invalid == {} else data | invalid
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    resumed = summarizer()
    resumed.sync_summarizer.meta_summary = "Unchanged state"
    with pytest.raises(ValueError, match="invalid meta checkpoint"):
        resumed.configure_persistence(tmp_path, resuming=True)
    asyncio.run(resumed.checkpoint_async())
    assert resumed.meta_summary == "Unchanged state"
    assert path.read_bytes() == before


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    state = MetaSummarizer()
    path = tmp_path / "state.json"
    state.meta_summary = "Saved summary"
    state.save_meta_state(path)
    before = path.read_bytes()
    state.meta_summary = "New summary"

    def fail_replace(self, target):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        state.save_meta_state(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "field,value",
    [("metadata", "broken"), ("combined_score", "bad"), ("archive_inspiration_ids", 7)],
)
def test_corrupt_pending_program_is_not_silently_normalized(tmp_path, field, value):
    state = summarizer()
    state.configure_persistence(tmp_path, resuming=False)
    state.add_evaluated_program(program())
    path = tmp_path / "meta" / "state.json"
    state.save_meta_state(path)
    checkpoint = json.loads(path.read_text())
    checkpoint["unprocessed_programs"][0][field] = value
    path.write_text(json.dumps(checkpoint))
    before = path.read_bytes()
    resumed = summarizer()
    with pytest.raises(ValueError, match="invalid meta checkpoint"):
        resumed.configure_persistence(tmp_path, resuming=True)
    asyncio.run(resumed.checkpoint_async())
    assert resumed.get_unprocessed_program_count() == 0
    assert path.read_bytes() == before


def test_cancelled_checkpoint_finishes_before_another_write(tmp_path, monkeypatch):
    async def run():
        state = summarizer()
        state.configure_persistence(tmp_path, resuming=False)
        entered = threading.Event()
        release = threading.Event()
        writes = []
        save = state.sync_summarizer.save_meta_state

        def delayed_save(path):
            writes.append("start")
            if len(writes) == 1:
                entered.set()
                assert release.wait(5)
            save(path)
            writes.append("finish")

        monkeypatch.setattr(state.sync_summarizer, "save_meta_state", delayed_save)
        first = asyncio.create_task(state.checkpoint_async())
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        second = asyncio.create_task(state.checkpoint_async())
        await asyncio.sleep(0)
        assert not first.done()
        assert writes == ["start"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert writes == ["start", "finish", "start", "finish"]
        assert summarizer().load_meta_state(tmp_path / "meta" / "state.json")

    asyncio.run(run())


def test_runner_setup_restores_meta_before_proposals(tmp_path):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"), num_islands=1
        )
        db = ProgramDatabase(config, embedding_model="")
        db.add(program("initial", 0))
        db.close()
        original = summarizer()
        original.configure_persistence(tmp_path, resuming=False)
        original.sync_summarizer.meta_recommendations = "1. Persisted advice"
        original.add_evaluated_program(program())
        await original.checkpoint_async()

        runner = object.__new__(ShinkaEvolveRunner)
        runner.results_dir = tmp_path
        runner.db_config = config
        runner.event_notifier = SimpleNamespace(db_path=None)
        runner.evo_config = SimpleNamespace(
            embedding_model="", evolve_prompts=False, num_generations=10
        )
        runner.max_db_workers = 1
        runner.enable_deadlock_debugging = False
        runner.console = None
        runner.wandb_logger = SimpleNamespace(enabled=False)
        runner.meta_summarizer = summarizer()
        runner._get_total_api_costs = AsyncMock(return_value=0.0)
        runner._load_bandit_state = lambda: None
        runner._log_wandb_population_progress = lambda: None
        await runner._setup_async()
        try:
            assert runner.meta_summarizer.get_current()[0] == "1. Persisted advice"
            assert runner.meta_summarizer.get_unprocessed_program_count() == 1
            assert runner.completed_generations == 1
        finally:
            await runner.async_db.close_async()

    asyncio.run(run())


@pytest.mark.parametrize("succeeded", [False, True])
def test_final_summary_keeps_reported_cost_and_checkpoints_pending_state(
    tmp_path, succeeded
):
    async def run():
        state = summarizer()
        state.async_llm_client = object()
        state.configure_persistence(tmp_path, resuming=False)
        state.add_evaluated_program(program())

        async def update(_best):
            if succeeded:
                state.sync_summarizer.meta_recommendations = "1. Advice"
                state.sync_summarizer.evaluated_since_last_meta = []
                state.sync_summarizer.total_programs_processed = 1
            return ("1. Advice" if succeeded else None), 0.25

        state.update_meta_memory_async = update
        success, cost = await state.perform_final_summary_async(str(tmp_path))
        assert success is succeeded
        assert cost == 0.25
        resumed = summarizer()
        resumed.configure_persistence(tmp_path, resuming=True)
        assert resumed.get_unprocessed_program_count() == (0 if succeeded else 1)
        assert resumed.get_current()[0] == ("1. Advice" if succeeded else None)

    asyncio.run(run())


def test_resumed_analysis_uses_previous_insights_and_retains_cost_on_output_error(
    tmp_path,
):
    async def run():
        original = summarizer()
        original.configure_persistence(tmp_path, resuming=False)
        original.sync_summarizer.meta_summary = "Existing individual summaries"
        original.sync_summarizer.meta_scratch_pad = "Existing global insights"
        original.sync_summarizer.meta_recommendations = "1. Existing advice"
        original.sync_summarizer.meta_recommendations_history = ["1. Existing advice"]
        original.add_evaluated_program(program())
        await original.checkpoint_async()

        resumed = summarizer()
        resumed.configure_persistence(tmp_path, resuming=True)
        provider = SimpleNamespace(
            batch_kwargs_query=AsyncMock(
                return_value=[
                    SimpleNamespace(content="New individual summary", cost=0.1)
                ]
            ),
            query=AsyncMock(
                side_effect=[
                    SimpleNamespace(content="New global insights", cost=0.2),
                    SimpleNamespace(content="1. New advice", cost=0.3),
                ]
            ),
        )
        resumed.async_llm_client = provider
        resumed.write_meta_output_async = AsyncMock(side_effect=OSError("disk full"))
        success, cost = await resumed.perform_final_summary_async(str(tmp_path))
        assert not success
        assert cost == pytest.approx(0.6)
        assert (
            "Existing global insights"
            in provider.query.await_args_list[0].kwargs["msg"]
        )
        assert "Existing advice" in provider.query.await_args_list[1].kwargs["msg"]
        loaded = summarizer()
        loaded.configure_persistence(tmp_path, resuming=True)
        assert "Existing individual summaries" in loaded.meta_summary
        assert "New individual summary" in loaded.meta_summary
        assert loaded.get_current()[0] == "1. New advice"
        assert loaded.get_unprocessed_program_count() == 0
        assert loaded.get_recommendations_history_count() == 2

    asyncio.run(run())


@pytest.mark.parametrize("failed_stage", [1, 2, 3])
@pytest.mark.parametrize("empty_content", [None, ""])
def test_empty_provider_response_retains_reported_cost_and_pending_program(
    tmp_path, failed_stage, empty_content
):
    async def run():
        state = summarizer()
        state.configure_persistence(tmp_path, resuming=False)
        state.add_evaluated_program(program())
        responses = [
            SimpleNamespace(
                content=empty_content if stage == failed_stage else "Summary", cost=0.25
            )
            for stage in [1, 2, 3]
        ]
        state.async_llm_client = SimpleNamespace(
            batch_kwargs_query=AsyncMock(return_value=[responses[0]]),
            query=AsyncMock(side_effect=responses[1:]),
        )
        success, cost = await state.perform_final_summary_async(str(tmp_path))
        assert not success
        assert cost == pytest.approx(0.25 * failed_stage)
        restored = summarizer()
        restored.configure_persistence(tmp_path, resuming=True)
        assert [item.id for item in restored.evaluated_since_last_meta] == ["pending"]
        assert restored.get_total_programs_processed() == 0

    asyncio.run(run())


def test_partial_batch_survives_checkpoint_and_complete_retry(tmp_path):
    async def run():
        state = summarizer()
        state.configure_persistence(tmp_path, resuming=False)
        state.add_evaluated_program(program("first", 2))
        state.add_evaluated_program(program("second", 3))
        failed_provider = SimpleNamespace(
            batch_kwargs_query=AsyncMock(
                return_value=[None, SimpleNamespace(content="Second summary", cost=0.2)]
            ),
            query=AsyncMock(),
        )
        state.async_llm_client = failed_provider
        success, cost = await state.perform_final_summary_async(str(tmp_path))
        assert not success
        assert cost == 0.2
        failed_provider.query.assert_not_awaited()

        resumed = summarizer()
        resumed.configure_persistence(tmp_path, resuming=True)
        assert [item.id for item in resumed.evaluated_since_last_meta] == [
            "first",
            "second",
        ]
        assert resumed.get_total_programs_processed() == 0
        provider = SimpleNamespace(
            batch_kwargs_query=AsyncMock(
                return_value=[
                    SimpleNamespace(content="First summary", cost=0.1),
                    SimpleNamespace(content="Second summary", cost=0.1),
                ]
            ),
            query=AsyncMock(
                side_effect=[
                    SimpleNamespace(content="Global insights", cost=0.1),
                    SimpleNamespace(content="1. Advice", cost=0.1),
                ]
            ),
        )
        resumed.async_llm_client = provider
        success, cost = await resumed.perform_final_summary_async(str(tmp_path))
        assert success
        assert cost == pytest.approx(0.4)
        assert provider.batch_kwargs_query.await_args.kwargs["num_samples"] == 2
        loaded = summarizer()
        loaded.configure_persistence(tmp_path, resuming=True)
        assert loaded.get_unprocessed_program_count() == 0
        assert loaded.get_total_programs_processed() == 2
        assert "Generation 2" in loaded.meta_summary
        assert "Generation 3" in loaded.meta_summary

    asyncio.run(run())


def test_failed_final_summary_persists_reported_cost_to_database(tmp_path):
    async def run():
        config = DatabaseConfig(
            db_path=str(tmp_path / "programs.sqlite"), num_islands=1
        )
        db = ProgramDatabase(config, embedding_model="")
        best = program("best", 0)
        best.metadata["meta_cost"] = 0.1
        db.add(best)
        state = summarizer()
        state.configure_persistence(tmp_path, resuming=False)
        state.add_evaluated_program(best)
        state.async_llm_client = SimpleNamespace(
            batch_kwargs_query=AsyncMock(
                return_value=[SimpleNamespace(content="", cost=0.25)]
            ),
            query=AsyncMock(),
        )
        try:
            success, cost = await state.perform_final_summary_async(
                str(tmp_path), best, config
            )
            assert not success
            assert cost == 0.25
            assert db.get("best").metadata["meta_cost"] == pytest.approx(0.35)
            loaded = summarizer()
            loaded.configure_persistence(tmp_path, resuming=True)
            assert loaded.evaluated_since_last_meta[0].metadata[
                "meta_cost"
            ] == pytest.approx(0.35)
        finally:
            db.close()

    asyncio.run(run())
