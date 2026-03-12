"""Web-based interactive controller for EvolutionRunner.

Replaces the CLI InteractiveController with a non-blocking controller that
reads commands from the ``interactive_commands`` SQLite table and writes status
back to ``interactive_status``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from shinka.interactive.payload_schemas import (
    MergePayload,
    SetTargetPayload,
    SuggestPayload,
)

from shinka.interactive.interactive_db import (
    CommandStatus,
    CommandType,
    InteractiveCommand,
    InteractiveDatabase,
    InteractiveStatus,
    RunState,
)

logger = logging.getLogger(__name__)


class WebController:
    """Non-blocking interactive controller driven by the ``interactive_commands`` table.

    The runner calls :meth:`process_commands` once per iteration (in the
    main ``while`` loop).  The method drains all pending commands and
    returns structured actions for the runner to execute.
    """

    def __init__(self, db_path: str) -> None:
        self.interactive_db = InteractiveDatabase(db_path)
        self._paused = False
        self._stop_requested = False
        self._continue_requested = False
        self._step_requested = False
        self._start_requested = False

    # ------------------------------------------------------------------
    # Public API used by EvolutionRunner
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def continue_requested(self) -> bool:
        return self._continue_requested

    @property
    def step_requested(self) -> bool:
        """Read-and-clear: returns True once, then resets."""
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    @property
    def start_requested(self) -> bool:
        return self._start_requested

    def clear_continue(self) -> None:
        """Reset the continue flag after a job has been submitted."""
        self._continue_requested = False

    def pause(self) -> None:
        """Pause the runner (block new job submissions)."""
        self._paused = True

    def resume(self) -> None:
        """Resume the runner (allow new job submissions)."""
        self._paused = False

    def process_commands(self) -> List[dict]:
        """Drain pending commands and return actions for the runner.

        Returns a list of action dicts. Currently supported actions:

        * ``{"action": "suggest", "parent_id": str, "prompt": str,
              "patch_type": str, "command_id": int}``
        * ``{"action": "merge", "parent_ids": [str, ...], "prompt": str,
              "patch_type": str, "command_id": int}``

        Pause / resume / stop are handled internally (they flip flags).

        .. note::

           Parent-existence validation is intentionally deferred to the
           caller (the runner's action handler) so that this method never
           touches the main ``ProgramDatabase`` connection — which may
           belong to a different thread in the async runner.
        """
        commands = self.interactive_db.poll_pending_commands()
        actions: List[dict] = []

        for cmd in commands:
            self.interactive_db.update_command_status(
                cmd.id,
                CommandStatus.PROCESSING.value,  # type: ignore[arg-type]
            )
            try:
                action = self._handle_command(cmd)
                if action is not None:
                    actions.append(action)
                self.interactive_db.update_command_status(
                    cmd.id,
                    CommandStatus.COMPLETED.value,  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.error("Interactive command %s failed: %s", cmd.id, exc)
                self.interactive_db.update_command_status(
                    cmd.id,  # type: ignore[arg-type]
                    CommandStatus.FAILED.value,
                    result=str(exc),
                )

        return actions

    def write_status(
        self,
        generation: int,
        best_score: float,
        queued_jobs: int,
        total_programs: int,
        target_generations: int = 0,
        *,
        idle: bool = False,
        waiting: bool = False,
        waiting_for_start: bool = False,
        is_resuming: bool = False,
    ) -> None:
        """Persist current run status so the web backend can read it.

        State priority (highest first):
        1. waiting_for_start — runner ready, awaiting user greenlight
        2. stopped — user requested stop
        3. paused — user paused (takes priority over idle, per bug fix a06e6fd)
        4. idle — no jobs in flight, target reached
        5. running — default
        """
        if waiting_for_start:
            state = RunState.WAITING_FOR_START
        elif self._stop_requested:
            state = RunState.STOPPED
        elif waiting:
            state = RunState.WAITING
        elif self._paused:
            state = RunState.PAUSED
        elif idle:
            state = RunState.IDLE
        else:
            state = RunState.RUNNING
        self.interactive_db.write_heartbeat()
        self.interactive_db.write_status(
            InteractiveStatus(
                run_state=state.value,
                generation=generation,
                best_score=best_score,
                queued_jobs=queued_jobs,
                total_programs=total_programs,
                target_generations=target_generations,
                is_resuming=is_resuming,
            )
        )

    def mark_idle(self) -> None:
        """Mark the run as idle (generations done, still accepting commands)."""
        self.interactive_db.write_heartbeat()
        self.interactive_db.write_status(
            InteractiveStatus(run_state=RunState.IDLE.value)
        )

    def mark_completed(self) -> None:
        """Mark the run as completed in the status table."""
        self.interactive_db.write_heartbeat()
        self.interactive_db.write_status(
            InteractiveStatus(run_state=RunState.COMPLETED.value)
        )

    def write_generation_heartbeat(self) -> None:
        """Refresh generation-backend liveness without changing run status."""
        self.interactive_db.write_heartbeat()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_command(self, cmd: InteractiveCommand) -> Optional[dict]:
        ct = cmd.command_type

        if ct == CommandType.PAUSE.value:
            logger.info("Interactive: pause requested")
            self._paused = True
            return None

        if ct == CommandType.RESUME.value:
            logger.info("Interactive: resume requested")
            self._paused = False
            return None

        if ct == CommandType.STOP.value:
            logger.info("Interactive: stop requested")
            self._stop_requested = True
            return None

        if ct == CommandType.CONTINUE.value:
            logger.info("Interactive: continue requested")
            self._continue_requested = True
            return None

        if ct == CommandType.START.value:
            logger.info("Interactive: start requested (greenlight)")
            self._start_requested = True
            return None

        if ct == CommandType.SET_TARGET.value:
            p = SetTargetPayload.model_validate(cmd.payload)
            logger.info(
                "Interactive: set_target — target_generations=%d",
                p.target_generations,
            )
            return {
                "action": "set_target",
                "target_generations": p.target_generations,
                "command_id": cmd.id,
            }

        if ct == CommandType.STEP.value:
            logger.info("Interactive: step requested")
            self._step_requested = True
            return None

        if ct == CommandType.SUGGEST.value:
            p = SuggestPayload.model_validate(cmd.payload)
            logger.info(
                "Interactive: suggest — parent=%s patch_type=%s prompt=%.60s…",
                p.parent_id,
                p.patch_type,
                p.prompt,
            )
            return {
                "action": "suggest",
                "parent_id": p.parent_id,
                "prompt": p.prompt,
                "patch_type": p.patch_type,
                "command_id": cmd.id,
            }

        if ct == CommandType.MERGE.value:
            p = MergePayload.model_validate(cmd.payload)
            logger.info(
                "Interactive: merge — parents=%s patch_type=%s prompt=%.60s…",
                p.parent_ids,
                p.patch_type,
                p.prompt,
            )
            return {
                "action": "merge",
                "parent_ids": p.parent_ids,
                "prompt": p.prompt,
                "patch_type": p.patch_type,
                "command_id": cmd.id,
            }

        logger.warning("Interactive: unknown command type %s", ct)
        return None
