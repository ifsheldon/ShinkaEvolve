"""Extended tests for InteractiveDatabase command/status round-trips."""

import time

import pytest

from shinka.interactive.interactive_db import (
    CommandStatus,
    CommandType,
    InteractiveDatabase,
    InteractiveStatus,
    RunState,
    _normalize_command_payload,
    _serialize_command_payload,
    _deserialize_command_payload,
)
from shinka.interactive.payload_schemas import (
    MergePayload,
    SetTargetPayload,
    SuggestPayload,
)


@pytest.fixture
def db(tmp_path):
    """Provide a fresh InteractiveDatabase per test."""
    return InteractiveDatabase(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Command CRUD
# ---------------------------------------------------------------------------


class TestCommandCRUD:
    def test_push_and_get(self, db):
        cid = db.push_command(CommandType.PAUSE)
        cmd = db.get_command(cid)
        assert cmd is not None
        assert cmd.command_type == CommandType.PAUSE
        assert cmd.status == CommandStatus.PENDING

    def test_push_returns_incrementing_ids(self, db):
        id1 = db.push_command(CommandType.PAUSE)
        id2 = db.push_command(CommandType.RESUME)
        assert id2 > id1

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_command(99999) is None

    def test_status_transitions(self, db):
        cid = db.push_command(CommandType.STOP)

        db.update_command_status(cid, CommandStatus.PROCESSING)
        assert db.get_command(cid).status == CommandStatus.PROCESSING

        db.update_command_status(cid, CommandStatus.COMPLETED, result="done")
        cmd = db.get_command(cid)
        assert cmd.status == CommandStatus.COMPLETED
        assert cmd.result == "done"
        assert cmd.processed_at is not None

    def test_failed_status_preserves_error(self, db):
        cid = db.push_command(CommandType.STOP)
        db.update_command_status(cid, CommandStatus.FAILED, result="oops")
        cmd = db.get_command(cid)
        assert cmd.status == CommandStatus.FAILED
        assert cmd.result == "oops"


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPolling:
    def test_poll_returns_only_pending(self, db):
        id1 = db.push_command(CommandType.PAUSE)
        id2 = db.push_command(CommandType.RESUME)
        db.update_command_status(id1, CommandStatus.COMPLETED)

        pending = db.poll_pending_commands()
        assert len(pending) == 1
        assert pending[0].id == id2

    def test_poll_returns_oldest_first(self, db):
        id1 = db.push_command(CommandType.PAUSE)
        id2 = db.push_command(CommandType.RESUME)
        id3 = db.push_command(CommandType.STOP)

        pending = db.poll_pending_commands()
        assert [c.id for c in pending] == [id1, id2, id3]

    def test_poll_empty_db(self, db):
        assert db.poll_pending_commands() == []


# ---------------------------------------------------------------------------
# Recent commands
# ---------------------------------------------------------------------------


class TestRecentCommands:
    def test_recent_respects_limit(self, db):
        for _ in range(5):
            db.push_command(CommandType.PAUSE)
        recent = db.get_recent_commands(limit=3)
        assert len(recent) == 3

    def test_recent_returns_newest_first(self, db):
        id1 = db.push_command(CommandType.PAUSE)
        id2 = db.push_command(CommandType.RESUME)
        recent = db.get_recent_commands(limit=10)
        assert recent[0].id == id2
        assert recent[1].id == id1


# ---------------------------------------------------------------------------
# Payload round-trips
# ---------------------------------------------------------------------------


class TestPayloadRoundTrips:
    def test_suggest_payload_round_trip(self, db):
        payload = SuggestPayload(parent_id="p1", prompt="try X", patch_type="diff")
        cid = db.push_command(CommandType.SUGGEST, payload)
        cmd = db.get_command(cid)
        assert isinstance(cmd.payload, SuggestPayload)
        assert cmd.payload.parent_id == "p1"
        assert cmd.payload.prompt == "try X"
        assert cmd.payload.patch_type == "diff"

    def test_merge_payload_round_trip(self, db):
        payload = MergePayload(parent_ids=["a", "b", "c"], prompt="merge them")
        cid = db.push_command(CommandType.MERGE, payload)
        cmd = db.get_command(cid)
        assert isinstance(cmd.payload, MergePayload)
        assert cmd.payload.parent_ids == ["a", "b", "c"]

    def test_set_target_payload_round_trip(self, db):
        payload = SetTargetPayload(target_generations=42)
        cid = db.push_command(CommandType.SET_TARGET, payload)
        cmd = db.get_command(cid)
        assert isinstance(cmd.payload, SetTargetPayload)
        assert cmd.payload.target_generations == 42

    def test_simple_command_has_empty_payload(self, db):
        cid = db.push_command(CommandType.PAUSE)
        cmd = db.get_command(cid)
        assert cmd.payload == {}

    def test_rejects_payload_on_simple_command(self, db):
        with pytest.raises(ValueError, match="does not accept a payload"):
            db.push_command(CommandType.PAUSE, {"extra": "data"})

    def test_dict_payload_validated_on_push(self, db):
        """Passing a raw dict for a payload-bearing command should validate."""
        cid = db.push_command(
            CommandType.SET_TARGET,
            {"target_generations": 10},
        )
        cmd = db.get_command(cid)
        assert isinstance(cmd.payload, SetTargetPayload)
        assert cmd.payload.target_generations == 10


# ---------------------------------------------------------------------------
# Status read/write
# ---------------------------------------------------------------------------


class TestStatus:
    def test_write_and_read_status(self, db):
        status = InteractiveStatus(
            run_state=RunState.RUNNING.value,
            generation=5,
            best_score=42.5,
            queued_jobs=3,
            total_programs=20,
            target_generations=100,
            is_resuming=False,
        )
        db.write_status(status)
        read = db.read_status()
        assert read is not None
        assert read.run_state == RunState.RUNNING.value
        assert read.generation == 5
        assert read.best_score == 42.5
        assert read.queued_jobs == 3
        assert read.total_programs == 20
        assert read.target_generations == 100
        assert read.is_resuming is False
        assert read.updated_at > 0

    def test_status_upsert_overwrites(self, db):
        db.write_status(InteractiveStatus(generation=1))
        db.write_status(InteractiveStatus(generation=2))
        read = db.read_status()
        assert read.generation == 2

    def test_read_status_empty(self, db):
        assert db.read_status() is None

    def test_status_preserves_is_resuming(self, db):
        db.write_status(InteractiveStatus(is_resuming=True))
        assert db.read_status().is_resuming is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_write_and_read_heartbeat(self, db):
        db.write_heartbeat()
        ts = db.read_heartbeat()
        assert ts is not None
        assert ts > 0

    def test_heartbeat_updates_timestamp(self, db):
        db.write_heartbeat()
        ts1 = db.read_heartbeat()
        time.sleep(0.05)
        db.write_heartbeat()
        ts2 = db.read_heartbeat()
        assert ts2 > ts1

    def test_read_heartbeat_empty(self, db):
        assert db.read_heartbeat() is None

    def test_custom_heartbeat_key(self, db):
        db.write_heartbeat(key="custom_hb")
        assert db.read_heartbeat(key="custom_hb") is not None
        assert db.read_heartbeat(key="other_key") is None


# ---------------------------------------------------------------------------
# Payload normalization helpers
# ---------------------------------------------------------------------------


class TestPayloadNormalization:
    def test_normalize_none_for_simple_command(self):
        result = _normalize_command_payload(CommandType.PAUSE, None)
        assert result == {}

    def test_normalize_dict_for_suggest(self):
        result = _normalize_command_payload(
            CommandType.SUGGEST,
            {"parent_id": "p1", "prompt": "hello"},
        )
        assert isinstance(result, SuggestPayload)
        assert result.parent_id == "p1"

    def test_normalize_pydantic_model_passthrough(self):
        payload = SuggestPayload(parent_id="p1")
        result = _normalize_command_payload(CommandType.SUGGEST, payload)
        assert result is payload

    def test_serialize_pydantic_payload(self):
        payload = SetTargetPayload(target_generations=10)
        serialized = _serialize_command_payload(payload)
        assert serialized == {"target_generations": 10}

    def test_serialize_empty_payload(self):
        assert _serialize_command_payload({}) == {}

    def test_deserialize_suggest(self):
        import json

        raw = json.dumps({"parent_id": "p1", "prompt": "test", "patch_type": "full"})
        result = _deserialize_command_payload(CommandType.SUGGEST, raw)
        assert isinstance(result, SuggestPayload)
        assert result.parent_id == "p1"

    def test_deserialize_empty_for_simple_command(self):
        result = _deserialize_command_payload(CommandType.PAUSE, None)
        assert result == {}
