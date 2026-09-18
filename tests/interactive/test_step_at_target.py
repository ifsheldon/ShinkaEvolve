"""Offline coverage for single-step generation at and beyond the run target."""

import asyncio
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shinka.core.async_interactive_runner import ShinkaEvolveInteractiveRunner
from shinka.interactive import WebController
from shinka.interactive.interactive_db import CommandType


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject provider connections while allowing the event loop's local sockets."""
    original_connect = socket.socket.connect

    def connect(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network access is forbidden in step tests")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def runner(tmp_path: Path) -> ShinkaEvolveInteractiveRunner:
    """Build real interactive orchestration with provider-free base dependencies."""
    with patch.object(ShinkaEvolveInteractiveRunner.__bases__[0], "__init__"):
        instance = ShinkaEvolveInteractiveRunner()
    instance.results_dir = str(tmp_path)
    instance.evo_config = SimpleNamespace(num_generations=10, max_api_costs=None)
    instance.completed_generations = 10
    instance.next_generation_to_submit = 10
    instance.running_jobs = []
    instance.active_proposal_tasks = {}
    instance.max_evaluation_jobs = 4
    instance.max_proposal_jobs = 4
    instance.cost_limit_reached = False
    instance.verbose = False
    instance.should_stop = asyncio.Event()
    instance.finalization_complete = asyncio.Event()
    instance.slot_available = asyncio.Event()
    instance.meta_summarizer = None
    instance.db = SimpleNamespace(last_iteration=0)
    instance.async_db = SimpleNamespace(
        get_best_program_async=AsyncMock(return_value=None),
        get_total_program_count_async=AsyncMock(return_value=10),
    )
    instance.web_controller = WebController(str(tmp_path / "programs.sqlite"))
    instance.web_controller.pause()
    instance.interactive_paused.clear()
    instance._is_system_stuck = MagicMock(return_value=False)
    instance._record_progress = MagicMock()
    instance._cleanup_completed_proposal_tasks = AsyncMock()
    instance._wait_for_slot_or_stop = AsyncMock()
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "completed", "next_generation", "in_flight", "expected_target"),
    [(10, 10, 10, 0, 11), (10, 8, 10, 2, 11), (20, 8, 10, 2, 20)],
)
async def test_step_reserves_one_unassigned_slot_and_pauses(
    runner: ShinkaEvolveInteractiveRunner,
    target: int,
    completed: int,
    next_generation: int,
    in_flight: int,
    expected_target: int,
) -> None:
    """One Step adds one proposal even when prior assignments are still running."""
    runner.evo_config.num_generations = target
    runner.completed_generations = completed
    runner.next_generation_to_submit = next_generation
    runner.running_jobs = [object() for _ in range(in_flight)]
    existing_jobs = list(runner.running_jobs)
    submitted = []

    async def submit(count: int) -> None:
        assert runner.next_generation_to_submit < runner.evo_config.num_generations
        submitted.extend(range(next_generation, next_generation + count))
        runner.next_generation_to_submit += count
        runner.should_stop.set()

    runner._start_proposals = submit
    runner._request_step()
    await asyncio.wait_for(runner._proposal_coordinator_task(), timeout=2)

    assert submitted == [next_generation]
    assert runner.running_jobs == existing_jobs
    assert runner.evo_config.num_generations == expected_target
    assert runner.web_controller.is_paused
    assert not runner.interactive_paused.is_set()
    assert not runner._step_mode


@pytest.mark.asyncio
async def test_step_waits_for_capacity_without_extending_again(
    runner: ShinkaEvolveInteractiveRunner,
) -> None:
    """A full pipeline retains one pending step until an evaluation slot opens."""
    runner.completed_generations = 6
    runner.running_jobs = [object() for _ in range(4)]
    submitted = []

    async def free_one_slot(timeout: float) -> None:
        assert runner.evo_config.num_generations == 11
        runner.running_jobs.pop()

    async def submit(count: int) -> None:
        submitted.append(count)
        runner.should_stop.set()

    runner._wait_for_slot_or_stop = free_one_slot
    runner._start_proposals = submit
    runner._request_step()
    await asyncio.wait_for(runner._proposal_coordinator_task(), timeout=2)

    assert submitted == [1]
    assert runner.evo_config.num_generations == 11
    assert runner.web_controller.is_paused


@pytest.mark.asyncio
@pytest.mark.parametrize("already_limited", [False, True])
async def test_step_after_idle_respects_committed_cost_limit(
    runner: ShinkaEvolveInteractiveRunner,
    already_limited: bool,
) -> None:
    """Step cannot use idle recovery to bypass an existing or newly reached cap."""
    runner.evo_config.max_api_costs = 1.0
    runner.cost_limit_reached = already_limited
    runner.total_api_cost = 0.75
    runner._get_committed_cost = MagicMock(return_value=1.0)
    runner._is_system_stuck = MagicMock(return_value=True)
    runner._start_proposals = AsyncMock()

    async def stop_after_check(timeout: float) -> None:
        runner.should_stop.set()

    runner._wait_for_slot_or_stop = stop_after_check
    runner._request_step()
    await asyncio.wait_for(runner._proposal_coordinator_task(), timeout=2)

    runner._start_proposals.assert_not_awaited()
    assert runner.cost_limit_reached
    assert runner.next_generation_to_submit == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("after_long_idle", [False, True])
async def test_step_command_reenters_exhausted_run_without_second_continue(
    runner: ShinkaEvolveInteractiveRunner,
    after_long_idle: bool,
) -> None:
    """A real queued Step exits keep-alive and reaches the proposal coordinator."""
    submitted = []
    runner._setup_async = AsyncMock()
    runner._verify_database_ready = AsyncMock()
    runner._wait_for_greenlight = AsyncMock()
    runner._run_final_operations = AsyncMock()
    runner._cleanup_async = AsyncMock()
    runner._print_final_summary = AsyncMock()
    if after_long_idle:
        # Exercise the inherited recovery path, which also starts proposals.
        # An idle timeout must not leave Step pending after that submission.
        runner._is_system_stuck = MagicMock(return_value=True)
        runner.stuck_detection_timeout = 60
        runner.stuck_detection_count = 0
        runner.max_stuck_detections = 3
        runner.failed_jobs_for_retry = {}
        runner._get_completed_job_work_count = MagicMock(return_value=0)

    async def monitor() -> None:
        if runner.completed_generations >= runner.evo_config.num_generations:
            runner.finalization_complete.set()
        await asyncio.Event().wait()

    async def submit(count: int) -> None:
        submitted.append((count, runner.next_generation_to_submit))
        runner.next_generation_to_submit += count
        runner.completed_generations += count
        runner.should_stop.set()
        runner._interactive_stop_requested = True
        runner.finalization_complete.set()

    real_keepalive = runner._interactive_keepalive_loop_async

    async def keepalive() -> bool:
        if after_long_idle:
            runner.last_progress_time = time.time() - 120
        runner.web_controller.interactive_db.push_command(CommandType.STEP)
        return await real_keepalive()

    runner._job_monitor_task = monitor
    runner._start_proposals = submit
    runner._interactive_keepalive_loop_async = keepalive
    with patch(
        "shinka.core.async_interactive_runner.WebController",
        return_value=runner.web_controller,
    ):
        await asyncio.wait_for(runner._run_async(), timeout=5)

    assert submitted == [(1, 10)]
    assert runner.completed_generations == 11
    assert runner.evo_config.num_generations == 11
    assert runner.web_controller.is_paused
    assert not runner.interactive_paused.is_set()
    assert not runner._step_mode

    await runner._handle_interactive_action_async(
        {"action": "set_target", "target_generations": 13}
    )
    assert not runner._step_mode
    assert not runner.interactive_paused.is_set()
