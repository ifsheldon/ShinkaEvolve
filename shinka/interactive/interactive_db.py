"""Interactive (Human-in-the-Loop) database layer.

Uses two SQLite tables in the same evolution database for IPC between
the EvolutionRunner process and the evolve-shell FastAPI backend:

- ``interactive_commands``  — Web UI → Runner  (pause, resume, stop, suggest, merge)
- ``interactive_status``    — Runner → Web UI  (run state, generation, best score)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Never, Optional, TypeAlias, cast

from shinka.interactive.payload_schemas import (
    MergePayload,
    SetTargetPayload,
    SuggestPayload,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class CommandType(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    SUGGEST = "suggest"
    MERGE = "merge"
    CONTINUE = "continue"
    SET_TARGET = "set_target"
    STEP = "step"
    START = "start"


class CommandStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RunState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    IDLE = "idle"  # generations done, still accepting interactive commands
    COMPLETED = "completed"
    STOPPED = "stopped"
    WAITING = "waiting"  # manual mode: waiting for human to click Continue
    WAITING_FOR_START = "waiting_for_start"  # runner ready, awaiting user greenlight
    ERROR = "error"


EmptyPayload: TypeAlias = dict[str, Never]
InteractiveCommandPayload: TypeAlias = (
    EmptyPayload | SetTargetPayload | SuggestPayload | MergePayload
)
InteractiveCommandPayloadInput: TypeAlias = (
    InteractiveCommandPayload | Mapping[str, Any] | None
)


def _empty_payload() -> EmptyPayload:
    """Return an empty payload value for commands that carry no data."""
    return {}


def _payload_model_for_command(
    command_type: CommandType,
) -> type[SetTargetPayload] | type[SuggestPayload] | type[MergePayload] | None:
    """Return the payload schema for a command, if it has one."""
    if command_type == CommandType.SET_TARGET:
        return SetTargetPayload
    if command_type == CommandType.SUGGEST:
        return SuggestPayload
    if command_type == CommandType.MERGE:
        return MergePayload
    return None


def _serialize_command_payload(payload: InteractiveCommandPayload) -> Dict[str, Any]:
    """Convert a typed payload into a JSON-serializable mapping."""
    if isinstance(payload, (SetTargetPayload, SuggestPayload, MergePayload)):
        return payload.model_dump()
    return {}


def _normalize_command_payload(
    command_type: CommandType,
    payload: InteractiveCommandPayloadInput,
) -> InteractiveCommandPayload:
    """Validate and normalize command payloads before writing them to SQLite."""
    payload_model = _payload_model_for_command(command_type)
    if payload_model is None:
        if payload is None:
            return _empty_payload()
        payload_data = (
            payload.model_dump()
            if isinstance(payload, (SetTargetPayload, SuggestPayload, MergePayload))
            else dict(payload)
        )
        if payload_data:
            raise ValueError(f"{command_type.value} does not accept a payload")
        return _empty_payload()

    if isinstance(payload, payload_model):
        return payload

    payload_data = (
        {}
        if payload is None
        else payload.model_dump()
        if isinstance(payload, (SetTargetPayload, SuggestPayload, MergePayload))
        else dict(payload)
    )
    return cast(InteractiveCommandPayload, payload_model.model_validate(payload_data))


def _deserialize_command_payload(
    command_type: CommandType,
    raw_payload: str | None,
) -> InteractiveCommandPayload:
    """Parse and validate a payload loaded from SQLite."""
    payload_data = json.loads(raw_payload) if raw_payload else {}
    return _normalize_command_payload(command_type, payload_data)


@dataclass
class InteractiveCommand:
    id: Optional[int] = None
    command_type: CommandType = CommandType.PAUSE
    payload: InteractiveCommandPayload = field(default_factory=_empty_payload)
    status: CommandStatus = CommandStatus.PENDING
    created_at: float = 0.0
    processed_at: Optional[float] = None
    result: Optional[str] = None


@dataclass
class InteractiveStatus:
    run_state: str = RunState.RUNNING.value
    generation: int = 0
    best_score: float = 0.0
    queued_jobs: int = 0
    total_programs: int = 0
    target_generations: int = 0
    is_resuming: bool = False
    updated_at: float = 0.0


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SQLite connection constants
# ---------------------------------------------------------------------------

#: Timeout passed to ``sqlite3.connect()`` — measured in **seconds**.
_SQLITE_CONNECT_TIMEOUT_S: int = 30

#: SQLite ``PRAGMA busy_timeout`` — measured in **milliseconds**.
_SQLITE_BUSY_TIMEOUT_MS: int = 10_000


class InteractiveDatabase:
    """Manages the interactive command/status tables within the evolution SQLite DB."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_SQLITE_CONNECT_TIMEOUT_S)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactive_commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_type TEXT NOT NULL,
                    payload TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    processed_at REAL,
                    result TEXT
                );

                CREATE TABLE IF NOT EXISTS interactive_status (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ---- Commands: written by backend, read by runner -------------------

    def push_command(
        self,
        command_type: CommandType,
        payload: InteractiveCommandPayloadInput = None,
    ) -> int:
        """Insert a new interactive command (called by the web backend)."""
        normalized_payload = _normalize_command_payload(command_type, payload)
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO interactive_commands (command_type, payload, status, created_at) "
                "VALUES (?, ?, 'pending', ?)",
                (
                    command_type.value,
                    json.dumps(_serialize_command_payload(normalized_payload)),
                    time.time(),
                ),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()

    def poll_pending_commands(self) -> List[InteractiveCommand]:
        """Return all pending interactive commands, oldest first (called by runner)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM interactive_commands WHERE status = 'pending' "
                "ORDER BY created_at ASC"
            ).fetchall()
            cmds: List[InteractiveCommand] = []
            for r in rows:
                command_type = CommandType(r["command_type"])
                cmds.append(
                    InteractiveCommand(
                        id=r["id"],
                        command_type=command_type,
                        payload=_deserialize_command_payload(
                            command_type, r["payload"]
                        ),
                        status=CommandStatus(r["status"]),
                        created_at=r["created_at"],
                        processed_at=r["processed_at"],
                        result=r["result"],
                    )
                )
            return cmds
        finally:
            conn.close()

    def update_command_status(
        self,
        cmd_id: int,
        status: CommandStatus,
        result: Optional[str] = None,
    ) -> None:
        """Mark a command as processing/completed/failed (called by runner)."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE interactive_commands SET status = ?, processed_at = ?, result = ? "
                "WHERE id = ?",
                (status.value, time.time(), result, cmd_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_command(self, cmd_id: int) -> Optional[InteractiveCommand]:
        """Get a single command by ID."""
        conn = self._connect()
        try:
            r = conn.execute(
                "SELECT * FROM interactive_commands WHERE id = ?", (cmd_id,)
            ).fetchone()
            if not r:
                return None
            command_type = CommandType(r["command_type"])
            return InteractiveCommand(
                id=r["id"],
                command_type=command_type,
                payload=_deserialize_command_payload(command_type, r["payload"]),
                status=CommandStatus(r["status"]),
                created_at=r["created_at"],
                processed_at=r["processed_at"],
                result=r["result"],
            )
        finally:
            conn.close()

    def get_recent_commands(self, limit: int = 20) -> List[InteractiveCommand]:
        """Return the most recent commands regardless of status."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM interactive_commands ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                InteractiveCommand(
                    id=r["id"],
                    command_type=command_type,
                    payload=_deserialize_command_payload(command_type, r["payload"]),
                    status=CommandStatus(r["status"]),
                    created_at=r["created_at"],
                    processed_at=r["processed_at"],
                    result=r["result"],
                )
                for r in rows
                for command_type in [CommandType(r["command_type"])]
            ]
        finally:
            conn.close()

    # ---- Status: written by runner, read by backend ---------------------

    def write_status(self, status: InteractiveStatus) -> None:
        """Upsert the current run status (called by runner)."""
        conn = self._connect()
        now = time.time()
        try:
            data = {
                "run_state": status.run_state,
                "generation": status.generation,
                "best_score": status.best_score,
                "queued_jobs": status.queued_jobs,
                "total_programs": status.total_programs,
                "target_generations": status.target_generations,
                "is_resuming": status.is_resuming,
            }
            conn.execute(
                "INSERT OR REPLACE INTO interactive_status (key, value, updated_at) "
                "VALUES ('run_status', ?, ?)",
                (json.dumps(data), now),
            )
            conn.commit()
        finally:
            conn.close()

    def write_heartbeat(self, key: str = "generation_backend_heartbeat") -> None:
        """Upsert a liveness heartbeat timestamp for the generation backend."""
        conn = self._connect()
        now = time.time()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO interactive_status (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                (key, "{}", now),
            )
            conn.commit()
        finally:
            conn.close()

    def read_status(self) -> Optional[InteractiveStatus]:
        """Read the current run status (called by backend)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value, updated_at FROM interactive_status WHERE key = 'run_status'"
            ).fetchone()
            if not row:
                return None
            data = json.loads(row["value"])
            return InteractiveStatus(
                run_state=data.get("run_state", RunState.RUNNING.value),
                generation=data.get("generation", 0),
                best_score=data.get("best_score", 0.0),
                queued_jobs=data.get("queued_jobs", 0),
                total_programs=data.get("total_programs", 0),
                target_generations=data.get("target_generations", 0),
                is_resuming=data.get("is_resuming", False),
                updated_at=row["updated_at"],
            )
        finally:
            conn.close()

    def read_heartbeat(
        self, key: str = "generation_backend_heartbeat"
    ) -> Optional[float]:
        """Read a liveness heartbeat timestamp."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT updated_at FROM interactive_status WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            return float(row["updated_at"])
        finally:
            conn.close()
