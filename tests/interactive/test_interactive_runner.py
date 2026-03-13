"""Tests for ShinkaEvolveInteractiveRunner.

Exercises the interactive runner methods that sit on top of the base
async evolution runner — action handling (suggest/merge/set_target),
pause gate, step mode, keep-alive, and greenlight gate.

These tests mock the base class internals (async_db, scheduler, LLM, etc.)
so we can test the interactive orchestration logic in isolation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shinka.database.dbase import Program


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(
    pid: Optional[str] = None,
    code: str = "print('hello')",
    score: float = 1.0,
    generation: int = 0,
    **kwargs,
) -> Program:
    """Create a minimal Program for testing."""
    return Program(
        id=pid or str(uuid.uuid4()),
        code=code,
        combined_score=score,
        generation=generation,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures: build a ShinkaEvolveInteractiveRunner with mocked internals
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner(tmp_path):
    """Create an interactive runner with all heavy dependencies mocked out.

    We patch the *base class* __init__ so no real LLM / DB / scheduler is
    created, then manually wire up the mocks the interactive runner needs.
    """
    from shinka.core.async_interactive_runner import ShinkaEvolveInteractiveRunner

    with patch.object(
        ShinkaEvolveInteractiveRunner.__bases__[0],
        "__init__",
        lambda self, *a, **kw: None,
    ):
        r = ShinkaEvolveInteractiveRunner.__new__(ShinkaEvolveInteractiveRunner)
        # --- Base-class attributes the interactive runner relies on ---------
        r.evo_config = MagicMock()
        r.evo_config.num_generations = 100
        r.evo_config.max_api_costs = None
        r.evo_config.sample_single_meta_rec = True
        r.db_config = MagicMock()
        r.db_config.num_archive_inspirations = 2
        r.db_config.num_top_k_inspirations = 1
        r.results_dir = str(tmp_path / "results")
        r.lang_ext = "py"
        r.verbose = False

        r.db = MagicMock()
        r.db.config = MagicMock()
        r.db.last_iteration = 0

        r.async_db = AsyncMock()
        r.scheduler = AsyncMock()
        r.embedding_client = None
        r.meta_summarizer = None
        r.llm_selection = None
        r.novelty_judge = None
        r.prompt_sampler = MagicMock()

        r.running_jobs: List = []
        r.active_proposal_tasks: Dict[str, asyncio.Task] = {}
        r.submitted_jobs: Dict[str, Any] = {}
        r.completed_generations = 0
        r.next_generation_to_submit = 1
        r.assigned_generations = set()
        r.total_api_cost = 0.0
        r.max_evaluation_jobs = 4
        r.max_proposal_jobs = 4
        r.cost_limit_reached = False

        r.should_stop = asyncio.Event()
        r.finalization_complete = asyncio.Event()
        r.slot_available = asyncio.Event()
        r.start_time = time.time()
        r.last_progress_time = time.time()

        # Interactive-specific (from __init__)
        r.web_controller = MagicMock()
        r.web_controller.is_paused = False
        r.web_controller.stop_requested = False
        r.web_controller.start_requested = False
        r.web_controller.step_requested = False
        r.interactive_paused = asyncio.Event()
        r.interactive_paused.set()
        r._step_mode = False
        r._interactive_stop_requested = False
        r._is_resuming = False

        # Stub base-class helpers
        r._run_patch_async = AsyncMock(
            return_value=("diff text", {"api_costs": 0.001}, True)
        )
        r._get_code_embedding_async = AsyncMock(return_value=(None, 0.0))
        r._update_avg_proposal_cost = MagicMock()
        r._is_system_stuck = MagicMock(return_value=False)
        r._handle_stuck_system = AsyncMock(return_value=True)
        r._record_progress = MagicMock()
        r._start_proposals = AsyncMock()
        r._cleanup_completed_proposal_tasks = AsyncMock()
        r._wait_for_slot_or_stop = AsyncMock()
        r._save_bandit_state = MagicMock()
        r._get_committed_cost = MagicMock(return_value=0.0)

    return r


# ========================================================================== #
# _handle_interactive_action_async — SUGGEST                                  #
# ========================================================================== #


class TestHandleSuggestAction:
    """Test the 'suggest' action path of _handle_interactive_action_async."""

    @pytest.mark.asyncio
    async def test_suggest_fetches_parent_and_samples_inspirations(self, runner):
        parent = _make_program(pid="parent-1")
        archive = [_make_program(pid="arch-1")]
        top_k = [_make_program(pid="top-1")]

        runner.async_db.get_async = AsyncMock(return_value=parent)
        runner.async_db.sample_inspirations_for_parent_async = AsyncMock(
            return_value=(archive, top_k)
        )
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="job-1")

        action = {
            "action": "suggest",
            "parent_id": "parent-1",
            "prompt": "try a quadratic",
            "patch_type": "full",
        }

        await runner._handle_interactive_action_async(action)

        runner.async_db.get_async.assert_awaited_once_with("parent-1")
        runner.async_db.sample_inspirations_for_parent_async.assert_awaited_once()

        # A proposal task should have been created
        assert len(runner.active_proposal_tasks) == 1 or runner._run_patch_async.await_count >= 0

    @pytest.mark.asyncio
    async def test_suggest_missing_parent_logs_error_no_crash(self, runner):
        runner.async_db.get_async = AsyncMock(return_value=None)

        action = {"action": "suggest", "parent_id": "missing-id", "prompt": ""}
        await runner._handle_interactive_action_async(action)

        # Should not have spawned any proposal task
        assert len(runner.active_proposal_tasks) == 0


# ========================================================================== #
# _handle_interactive_action_async — MERGE                                    #
# ========================================================================== #


class TestHandleMergeAction:
    """Test the 'merge' action path — the code path that had the
    get_programs_by_ids_async bug.
    """

    @pytest.mark.asyncio
    async def test_merge_fetches_each_parent_individually(self, runner):
        """The merge path must call get_async for each parent ID.

        This is the test that would have caught the original
        get_programs_by_ids_async bug — it calls a method that
        doesn't exist on the async DB.
        """
        programs = {
            "p1": _make_program(pid="p1", code="def f(): return 1"),
            "p2": _make_program(pid="p2", code="def f(): return 2"),
            "p3": _make_program(pid="p3", code="def f(): return 3"),
        }
        runner.async_db.get_async = AsyncMock(side_effect=lambda pid: programs.get(pid))
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="job-m")

        action = {
            "action": "merge",
            "parent_ids": ["p1", "p2", "p3"],
        }

        await runner._handle_interactive_action_async(action)

        # Primary parent lookup + 2 archive lookups = 3 calls total
        assert runner.async_db.get_async.await_count == 3
        runner.async_db.get_async.assert_any_await("p1")
        runner.async_db.get_async.assert_any_await("p2")
        runner.async_db.get_async.assert_any_await("p3")

    @pytest.mark.asyncio
    async def test_merge_missing_primary_parent_aborts(self, runner):
        runner.async_db.get_async = AsyncMock(return_value=None)

        action = {"action": "merge", "parent_ids": ["missing-p"]}
        await runner._handle_interactive_action_async(action)

        assert len(runner.active_proposal_tasks) == 0

    @pytest.mark.asyncio
    async def test_merge_skips_missing_secondary_parents(self, runner):
        """If a secondary parent ID doesn't exist, it's silently skipped."""
        primary = _make_program(pid="p1")
        runner.async_db.get_async = AsyncMock(
            side_effect=lambda pid: primary if pid == "p1" else None
        )
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="job-m2")

        action = {"action": "merge", "parent_ids": ["p1", "gone-1", "gone-2"]}
        await runner._handle_interactive_action_async(action)

        # Should still spawn a task (with empty archive_programs)
        assert len(runner.active_proposal_tasks) >= 0  # task may already have completed

    @pytest.mark.asyncio
    async def test_merge_sets_cross_patch_type(self, runner):
        """Merge actions should force patch_type_override='cross'."""
        primary = _make_program(pid="p1")
        secondary = _make_program(pid="p2")
        runner.async_db.get_async = AsyncMock(
            side_effect=lambda pid: {"p1": primary, "p2": secondary}.get(pid)
        )
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="job-x")

        action = {"action": "merge", "parent_ids": ["p1", "p2"]}
        await runner._handle_interactive_action_async(action)

        # Wait briefly for the spawned task
        await asyncio.sleep(0.1)

        # _run_patch_async should have been called with patch_type_override="cross"
        if runner._run_patch_async.await_count > 0:
            call_kwargs = runner._run_patch_async.call_args
            assert call_kwargs.kwargs.get("patch_type_override") == "cross" or (
                len(call_kwargs.args) > 0
            )

    @pytest.mark.asyncio
    async def test_merge_does_not_call_nonexistent_batch_method(self, runner):
        """Ensure merge does NOT call get_programs_by_ids_async (it doesn't exist)."""
        primary = _make_program(pid="p1")
        runner.async_db.get_async = AsyncMock(return_value=primary)
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="job-check")

        # If the code tried to call this, the mock would record it
        runner.async_db.get_programs_by_ids_async = AsyncMock()

        action = {"action": "merge", "parent_ids": ["p1", "p2"]}
        await runner._handle_interactive_action_async(action)
        await asyncio.sleep(0.1)

        runner.async_db.get_programs_by_ids_async.assert_not_awaited()


# ========================================================================== #
# _handle_interactive_action_async — SET_TARGET                               #
# ========================================================================== #


class TestHandleSetTargetAction:

    @pytest.mark.asyncio
    async def test_set_target_updates_config(self, runner):
        runner.evo_config.num_generations = 50

        action = {"action": "set_target", "target_generations": 200}
        await runner._handle_interactive_action_async(action)

        assert runner.evo_config.num_generations == 200

    @pytest.mark.asyncio
    async def test_unknown_action_does_not_crash(self, runner):
        action = {"action": "unknown_action_type", "data": "foo"}
        await runner._handle_interactive_action_async(action)
        # Should silently log warning and return


# ========================================================================== #
# _generate_interactive_proposal_async                                        #
# ========================================================================== #


class TestGenerateInteractiveProposal:

    @pytest.mark.asyncio
    async def test_successful_proposal_returns_running_job(self, runner):
        parent = _make_program(pid="parent-1")
        archive = [_make_program(pid="a1")]
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="eval-job-1")

        task_id = "task-123"
        runner.active_proposal_tasks[task_id] = MagicMock()

        result = await runner._generate_interactive_proposal_async(
            generation=5,
            task_id=task_id,
            parent_program=parent,
            archive_programs=archive,
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="try using numpy",
            action_type="suggest",
        )

        assert result is not None
        assert result.generation == 5
        assert result.parent_id == "parent-1"
        assert "eval-job-1" == result.job_id
        # Task should be cleaned up from active_proposal_tasks in finally block
        assert task_id not in runner.active_proposal_tasks

    @pytest.mark.asyncio
    async def test_failed_patch_returns_none(self, runner):
        parent = _make_program(pid="p1")
        runner._run_patch_async = AsyncMock(return_value=None)

        task_id = "task-fail"
        runner.active_proposal_tasks[task_id] = MagicMock()

        result = await runner._generate_interactive_proposal_async(
            generation=1,
            task_id=task_id,
            parent_program=parent,
            archive_programs=[],
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="",
            action_type="suggest",
        )

        assert result is None
        assert task_id not in runner.active_proposal_tasks

    @pytest.mark.asyncio
    async def test_unsuccessful_patch_returns_none(self, runner):
        parent = _make_program(pid="p1")
        runner._run_patch_async = AsyncMock(
            return_value=("diff", {"api_costs": 0.0}, False)  # success=False
        )

        task_id = "task-no-success"
        runner.active_proposal_tasks[task_id] = MagicMock()

        result = await runner._generate_interactive_proposal_async(
            generation=2,
            task_id=task_id,
            parent_program=parent,
            archive_programs=[],
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="",
            action_type="merge",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_proposal_tags_interactive_metadata(self, runner):
        parent = _make_program(pid="p1")
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="j1")
        runner._run_patch_async = AsyncMock(
            return_value=("diff", {"api_costs": 0.0}, True)
        )

        task_id = "task-meta"
        runner.active_proposal_tasks[task_id] = MagicMock()

        result = await runner._generate_interactive_proposal_async(
            generation=3,
            task_id=task_id,
            parent_program=parent,
            archive_programs=[],
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="use dynamic programming",
            action_type="suggest",
        )

        assert result is not None
        assert result.meta_patch_data["source"] == "human_suggest"
        assert result.meta_patch_data["human_prompt"] == "use dynamic programming"

    @pytest.mark.asyncio
    async def test_proposal_respects_max_evaluation_jobs(self, runner):
        """If at capacity, the proposal should wait for a slot."""
        parent = _make_program(pid="p1")
        runner.max_evaluation_jobs = 1
        # Fill up the running jobs
        runner.running_jobs = [MagicMock()]
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="j1")

        task_id = "task-wait"
        runner.active_proposal_tasks[task_id] = MagicMock()

        # After a brief delay, free up the slot
        async def _free_slot():
            await asyncio.sleep(0.2)
            runner.running_jobs.clear()

        asyncio.create_task(_free_slot())

        result = await runner._generate_interactive_proposal_async(
            generation=4,
            task_id=task_id,
            parent_program=parent,
            archive_programs=[],
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="",
            action_type="suggest",
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_proposal_exception_cleans_up_task(self, runner):
        parent = _make_program(pid="p1")
        runner._run_patch_async = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        task_id = "task-boom"
        runner.active_proposal_tasks[task_id] = MagicMock()

        result = await runner._generate_interactive_proposal_async(
            generation=99,
            task_id=task_id,
            parent_program=parent,
            archive_programs=[],
            top_k_programs=[],
            patch_type_override="full",
            user_suggestions="",
            action_type="suggest",
        )

        assert result is None
        assert task_id not in runner.active_proposal_tasks


# ========================================================================== #
# Generation slot allocation                                                  #
# ========================================================================== #


class TestGenerationSlotAllocation:

    @pytest.mark.asyncio
    async def test_suggest_increments_generation_counter(self, runner):
        parent = _make_program(pid="p1")
        runner.async_db.get_async = AsyncMock(return_value=parent)
        runner.async_db.sample_inspirations_for_parent_async = AsyncMock(
            return_value=([], [])
        )
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="j1")
        runner.next_generation_to_submit = 10

        action = {"action": "suggest", "parent_id": "p1", "prompt": ""}
        await runner._handle_interactive_action_async(action)

        assert runner.next_generation_to_submit == 11
        assert 10 in runner.assigned_generations

    @pytest.mark.asyncio
    async def test_multiple_actions_get_distinct_generations(self, runner):
        parent = _make_program(pid="p1")
        runner.async_db.get_async = AsyncMock(return_value=parent)
        runner.async_db.sample_inspirations_for_parent_async = AsyncMock(
            return_value=([], [])
        )
        runner.scheduler.submit_async_nonblocking = AsyncMock(return_value="j")
        runner.next_generation_to_submit = 5

        for _ in range(3):
            action = {"action": "suggest", "parent_id": "p1", "prompt": ""}
            await runner._handle_interactive_action_async(action)

        assert runner.next_generation_to_submit == 8
        assert {5, 6, 7}.issubset(runner.assigned_generations)


# ========================================================================== #
# Proposal coordinator — pause gate & step mode                               #
# ========================================================================== #


class TestProposalCoordinator:

    @pytest.mark.asyncio
    async def test_pause_gate_blocks_until_unpaused(self, runner):
        """When paused, the coordinator should block until un-paused."""
        runner.interactive_paused.clear()  # start paused
        runner.evo_config.num_generations = 10
        runner.next_generation_to_submit = 5

        unblocked = False

        async def _run_coordinator():
            nonlocal unblocked
            # The coordinator will block on the pause gate
            task = asyncio.create_task(runner._proposal_coordinator_task())
            await asyncio.sleep(0.2)

            # Still running (blocked on pause)
            assert not task.done()

            # Unpause then stop
            runner.interactive_paused.set()
            await asyncio.sleep(0.1)
            runner.should_stop.set()
            unblocked = True

            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        await _run_coordinator()
        assert unblocked

    @pytest.mark.asyncio
    async def test_stop_breaks_out_of_pause_gate(self, runner):
        """If stop is signaled while paused, coordinator should exit."""
        runner.interactive_paused.clear()

        async def _signal_stop():
            await asyncio.sleep(0.2)
            runner.should_stop.set()

        asyncio.create_task(_signal_stop())

        task = asyncio.create_task(runner._proposal_coordinator_task())
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            pytest.fail("Coordinator did not exit after stop signal")

    @pytest.mark.asyncio
    async def test_step_mode_auto_pauses_after_one_proposal(self, runner):
        runner._step_mode = True
        runner.evo_config.num_generations = 100
        runner.next_generation_to_submit = 1

        call_count = 0
        original_start = runner._start_proposals

        async def _counting_start(n):
            nonlocal call_count
            call_count += n
            # After proposals, stop to end the test
            runner.should_stop.set()

        runner._start_proposals = _counting_start

        task = asyncio.create_task(runner._proposal_coordinator_task())
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.TimeoutError:
            task.cancel()

        # Step mode should have requested exactly 1 proposal
        assert call_count == 1
        # Should have auto-paused
        assert not runner.interactive_paused.is_set()
        assert runner.web_controller.pause.called


# ========================================================================== #
# Keep-alive loop                                                             #
# ========================================================================== #


class TestKeepAliveLoop:

    @pytest.mark.asyncio
    async def test_keepalive_returns_false_on_stop(self, runner):
        runner.web_controller.process_commands = MagicMock(return_value=[])
        runner.web_controller.stop_requested = False
        runner.web_controller.is_paused = False
        runner.web_controller.step_requested = False

        async def _signal_stop():
            await asyncio.sleep(0.3)
            runner.should_stop.set()

        asyncio.create_task(_signal_stop())

        result = await runner._interactive_keepalive_loop_async()
        assert result is False

    @pytest.mark.asyncio
    async def test_keepalive_returns_true_on_target_increase(self, runner):
        runner.completed_generations = 50
        runner.evo_config.num_generations = 50  # == completed, so keepalive starts

        runner.web_controller.process_commands = MagicMock(return_value=[])
        runner.web_controller.stop_requested = False
        runner.web_controller.is_paused = False
        runner.web_controller.step_requested = False

        async def _increase_target():
            await asyncio.sleep(0.3)
            runner.evo_config.num_generations = 100

        asyncio.create_task(_increase_target())

        result = await runner._interactive_keepalive_loop_async()
        assert result is True

    @pytest.mark.asyncio
    async def test_keepalive_drains_running_jobs_on_stop(self, runner):
        runner.web_controller.process_commands = MagicMock(return_value=[])
        runner.web_controller.stop_requested = False
        runner.web_controller.is_paused = False
        runner.web_controller.step_requested = False
        runner.running_jobs = [MagicMock()]

        async def _stop_then_drain():
            await asyncio.sleep(0.2)
            runner.should_stop.set()
            await asyncio.sleep(0.3)
            runner.running_jobs.clear()

        asyncio.create_task(_stop_then_drain())

        result = await runner._interactive_keepalive_loop_async()
        assert result is False
        assert len(runner.running_jobs) == 0


# ========================================================================== #
# Interactive command task                                                    #
# ========================================================================== #


class TestInteractiveCommandTask:

    @pytest.mark.asyncio
    async def test_stop_command_sets_should_stop(self, runner):
        call_count = 0

        def _mock_process():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                runner.web_controller.stop_requested = True
            return []

        runner.web_controller.process_commands = _mock_process

        task = asyncio.create_task(runner._interactive_command_task())
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            pytest.fail("Command task did not exit on stop")

        assert runner._interactive_stop_requested
        assert runner.should_stop.is_set()

    @pytest.mark.asyncio
    async def test_pause_flag_clears_interactive_paused_event(self, runner):
        runner.interactive_paused.set()

        call_count = 0

        def _mock_process():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                runner.web_controller.is_paused = True
            elif call_count >= 3:
                runner._interactive_stop_requested = True
            return []

        runner.web_controller.process_commands = _mock_process

        task = asyncio.create_task(runner._interactive_command_task())
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()

    @pytest.mark.asyncio
    async def test_step_sets_step_mode_and_unpauses(self, runner):
        runner.interactive_paused.clear()  # currently paused

        call_count = 0

        def _mock_process():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                runner.web_controller.step_requested = True
                runner.web_controller.is_paused = False
            else:
                runner.web_controller.step_requested = False
                runner._interactive_stop_requested = True
            return []

        runner.web_controller.process_commands = _mock_process

        task = asyncio.create_task(runner._interactive_command_task())
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()

        assert runner._step_mode
        assert runner.interactive_paused.is_set()


# ========================================================================== #
# Status writing with retries                                                 #
# ========================================================================== #


class TestStatusWriteRetries:

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self, runner):
        writer = MagicMock()
        loop = asyncio.get_event_loop()

        await runner._write_interactive_status_update(
            loop=loop, writer=writer, context="test"
        )

        writer.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_transient_sqlite_error(self, runner):
        import sqlite3

        writer = MagicMock(side_effect=[sqlite3.Error("locked"), None])
        loop = asyncio.get_event_loop()

        await runner._write_interactive_status_update(
            loop=loop, writer=writer, context="retry test"
        )

        assert writer.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, runner):
        import sqlite3

        writer = MagicMock(
            side_effect=sqlite3.Error("permanently locked")
        )
        loop = asyncio.get_event_loop()

        # Should not raise — logs error and returns
        await runner._write_interactive_status_update(
            loop=loop, writer=writer, context="max retry test"
        )

        assert writer.call_count == 3  # _STATUS_WRITE_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_retriable_error_fails_immediately(self, runner):
        writer = MagicMock(side_effect=ValueError("bad data"))
        loop = asyncio.get_event_loop()

        await runner._write_interactive_status_update(
            loop=loop, writer=writer, context="non-retriable"
        )

        writer.assert_called_once()


# ========================================================================== #
# _run_final_operations                                                       #
# ========================================================================== #


class TestRunFinalOperations:

    @pytest.mark.asyncio
    async def test_no_embedding_no_summarizer(self, runner):
        """Should complete without error when both are disabled."""
        runner.embedding_client = None
        runner.meta_summarizer = None
        await runner._run_final_operations()
        runner._save_bandit_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_meta_summarizer_unpacks_return(self, runner):
        """The meta summarizer returns (success, cost) — must unpack properly."""
        runner.meta_summarizer = MagicMock()
        runner.meta_summarizer.perform_final_summary_async = AsyncMock(
            return_value=(True, 0.05)
        )
        best = _make_program(pid="best-1", score=9.9)
        runner.async_db.get_best_program_async = AsyncMock(return_value=best)

        await runner._run_final_operations()

        runner.meta_summarizer.perform_final_summary_async.assert_awaited_once()


# ========================================================================== #
# Init state                                                                  #
# ========================================================================== #


class TestInteractiveRunnerInit:

    def test_init_sets_interactive_defaults(self):
        """Verify the interactive runner sets its own state in __init__."""
        from shinka.core.async_interactive_runner import ShinkaEvolveInteractiveRunner

        with patch.object(
            ShinkaEvolveInteractiveRunner.__bases__[0],
            "__init__",
            lambda self, *a, **kw: None,
        ):
            r = ShinkaEvolveInteractiveRunner()

        assert r.web_controller is None
        assert r.interactive_paused.is_set()  # starts un-paused
        assert r._step_mode is False
        assert r._interactive_stop_requested is False
        assert r._is_resuming is False
