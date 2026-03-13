"""Tests for WebController interactive command processing and status management."""

import tempfile
from pathlib import Path

import pytest

from shinka.interactive.interactive_db import (
    CommandStatus,
    CommandType,
    InteractiveDatabase,
    RunState,
)
from shinka.interactive.payload_schemas import (
    MergePayload,
    SetTargetPayload,
    SuggestPayload,
)
from shinka.interactive.web_controller import WebController


@pytest.fixture
def controller(tmp_path):
    """Provide a fresh WebController backed by a temp DB."""
    db_path = str(tmp_path / "test.db")
    return WebController(db_path)


# ---------------------------------------------------------------------------
# Flag commands (handled internally, no actions returned)
# ---------------------------------------------------------------------------


class TestFlagCommands:
    def test_pause_sets_flag(self, controller):
        assert not controller.is_paused
        controller.interactive_db.push_command(CommandType.PAUSE)
        actions = controller.process_commands()
        assert actions == []
        assert controller.is_paused

    def test_resume_clears_flag(self, controller):
        controller.pause()
        assert controller.is_paused
        controller.interactive_db.push_command(CommandType.RESUME)
        controller.process_commands()
        assert not controller.is_paused

    def test_stop_sets_flag(self, controller):
        assert not controller.stop_requested
        controller.interactive_db.push_command(CommandType.STOP)
        controller.process_commands()
        assert controller.stop_requested

    def test_continue_sets_flag(self, controller):
        assert not controller.continue_requested
        controller.interactive_db.push_command(CommandType.CONTINUE)
        controller.process_commands()
        assert controller.continue_requested

    def test_clear_continue(self, controller):
        controller.interactive_db.push_command(CommandType.CONTINUE)
        controller.process_commands()
        assert controller.continue_requested
        controller.clear_continue()
        assert not controller.continue_requested

    def test_start_sets_flag(self, controller):
        assert not controller.start_requested
        controller.interactive_db.push_command(CommandType.START)
        controller.process_commands()
        assert controller.start_requested

    def test_step_is_read_and_clear(self, controller):
        controller.interactive_db.push_command(CommandType.STEP)
        controller.process_commands()
        assert controller.step_requested is True  # first read
        assert controller.step_requested is False  # second read clears

    def test_pause_resume_methods(self, controller):
        controller.pause()
        assert controller.is_paused
        controller.resume()
        assert not controller.is_paused


# ---------------------------------------------------------------------------
# Action commands (return action dicts)
# ---------------------------------------------------------------------------


class TestActionCommands:
    def test_suggest_returns_action(self, controller):
        payload = SuggestPayload(parent_id="prog-1", prompt="try X", patch_type="diff")
        controller.interactive_db.push_command(CommandType.SUGGEST, payload)
        actions = controller.process_commands()
        assert len(actions) == 1
        a = actions[0]
        assert a["action"] == "suggest"
        assert a["parent_id"] == "prog-1"
        assert a["prompt"] == "try X"
        assert a["patch_type"] == "diff"
        assert "command_id" in a

    def test_merge_returns_action(self, controller):
        payload = MergePayload(parent_ids=["a", "b"], prompt="combine")
        controller.interactive_db.push_command(CommandType.MERGE, payload)
        actions = controller.process_commands()
        assert len(actions) == 1
        a = actions[0]
        assert a["action"] == "merge"
        assert a["parent_ids"] == ["a", "b"]
        assert a["prompt"] == "combine"

    def test_set_target_returns_action(self, controller):
        payload = SetTargetPayload(target_generations=200)
        controller.interactive_db.push_command(CommandType.SET_TARGET, payload)
        actions = controller.process_commands()
        assert len(actions) == 1
        a = actions[0]
        assert a["action"] == "set_target"
        assert a["target_generations"] == 200

    def test_multiple_commands_in_one_drain(self, controller):
        controller.interactive_db.push_command(CommandType.PAUSE)
        controller.interactive_db.push_command(
            CommandType.SUGGEST,
            SuggestPayload(parent_id="p1"),
        )
        controller.interactive_db.push_command(
            CommandType.SET_TARGET,
            SetTargetPayload(target_generations=50),
        )
        actions = controller.process_commands()
        # PAUSE is internal (no action), SUGGEST and SET_TARGET produce actions
        assert len(actions) == 2
        assert controller.is_paused


# ---------------------------------------------------------------------------
# Command status tracking
# ---------------------------------------------------------------------------


class TestCommandStatusTracking:
    def test_successful_command_marked_completed(self, controller):
        cid = controller.interactive_db.push_command(CommandType.PAUSE)
        controller.process_commands()
        cmd = controller.interactive_db.get_command(cid)
        assert cmd.status == CommandStatus.COMPLETED

    def test_action_command_marked_completed(self, controller):
        payload = SetTargetPayload(target_generations=10)
        cid = controller.interactive_db.push_command(CommandType.SET_TARGET, payload)
        controller.process_commands()
        cmd = controller.interactive_db.get_command(cid)
        assert cmd.status == CommandStatus.COMPLETED

    def test_commands_not_reprocessed(self, controller):
        controller.interactive_db.push_command(CommandType.PAUSE)
        controller.process_commands()
        controller.resume()
        # Second drain should find nothing pending
        actions = controller.process_commands()
        assert actions == []
        assert not controller.is_paused  # stayed resumed


# ---------------------------------------------------------------------------
# Status writing
# ---------------------------------------------------------------------------


class TestStatusWriting:
    def test_write_status_running(self, controller):
        controller.write_status(
            generation=5,
            best_score=10.0,
            queued_jobs=3,
            total_programs=20,
            target_generations=100,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.RUNNING.value
        assert status.generation == 5

    def test_write_status_paused(self, controller):
        controller.pause()
        controller.write_status(
            generation=5,
            best_score=10.0,
            queued_jobs=0,
            total_programs=20,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.PAUSED.value

    def test_write_status_stopped_overrides_paused(self, controller):
        """Stopped takes priority over paused in state hierarchy."""
        controller.pause()
        controller.interactive_db.push_command(CommandType.STOP)
        controller.process_commands()
        controller.write_status(
            generation=5,
            best_score=10.0,
            queued_jobs=0,
            total_programs=20,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.STOPPED.value

    def test_write_status_waiting_for_start(self, controller):
        controller.write_status(
            generation=0,
            best_score=0.0,
            queued_jobs=0,
            total_programs=0,
            waiting_for_start=True,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.WAITING_FOR_START.value

    def test_write_status_idle(self, controller):
        controller.write_status(
            generation=10,
            best_score=50.0,
            queued_jobs=0,
            total_programs=100,
            idle=True,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.IDLE.value

    def test_paused_overrides_idle(self, controller):
        """Paused should take priority over idle (bug fix a06e6fd)."""
        controller.pause()
        controller.write_status(
            generation=10,
            best_score=50.0,
            queued_jobs=0,
            total_programs=100,
            idle=True,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.PAUSED.value

    def test_mark_idle(self, controller):
        controller.mark_idle()
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.IDLE.value

    def test_mark_completed(self, controller):
        controller.mark_completed()
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.COMPLETED.value

    def test_heartbeat_written_on_status(self, controller):
        controller.write_status(
            generation=1,
            best_score=0.0,
            queued_jobs=0,
            total_programs=0,
        )
        hb = controller.interactive_db.read_heartbeat()
        assert hb is not None

    def test_write_generation_heartbeat(self, controller):
        controller.write_generation_heartbeat()
        hb = controller.interactive_db.read_heartbeat()
        assert hb is not None

    def test_is_resuming_propagated(self, controller):
        controller.write_status(
            generation=5,
            best_score=10.0,
            queued_jobs=0,
            total_programs=20,
            is_resuming=True,
        )
        status = controller.interactive_db.read_status()
        assert status.is_resuming is True


# ---------------------------------------------------------------------------
# State priority
# ---------------------------------------------------------------------------


class TestStatePriority:
    """Verify the documented state priority hierarchy."""

    def test_waiting_for_start_beats_everything(self, controller):
        """waiting_for_start > stopped > paused > idle."""
        controller.pause()
        controller.interactive_db.push_command(CommandType.STOP)
        controller.process_commands()
        controller.write_status(
            generation=0,
            best_score=0,
            queued_jobs=0,
            total_programs=0,
            idle=True,
            waiting_for_start=True,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.WAITING_FOR_START.value

    def test_waiting_overrides_paused(self, controller):
        """The 'waiting' flag (manual continue mode) overrides paused."""
        controller.pause()
        controller.write_status(
            generation=0,
            best_score=0,
            queued_jobs=0,
            total_programs=0,
            waiting=True,
        )
        status = controller.interactive_db.read_status()
        assert status.run_state == RunState.WAITING.value
