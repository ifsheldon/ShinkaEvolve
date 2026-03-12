import tempfile
from pathlib import Path

from shinka.interactive.interactive_db import (
    CommandStatus,
    CommandType,
    InteractiveDatabase,
)
from shinka.interactive.payload_schemas import SuggestPayload


def test_push_command_returns_int_and_status_updates_use_enums():
    """Interactive DB commands should round-trip through typed enum APIs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "interactive.db"
        db = InteractiveDatabase(str(db_path))

        command_id = db.push_command(
            CommandType.SUGGEST,
            SuggestPayload(parent_id="parent-1", prompt="try a new heuristic"),
        )

        assert isinstance(command_id, int)

        pending = db.get_command(command_id)
        assert pending is not None
        assert pending.command_type == CommandType.SUGGEST
        assert pending.status == CommandStatus.PENDING

        db.update_command_status(
            command_id,
            CommandStatus.COMPLETED,
            result="ok",
        )

        completed = db.get_command(command_id)
        assert completed is not None
        assert completed.status == CommandStatus.COMPLETED
        assert completed.result == "ok"
