"""Async evolution runner with interactive expert-in-the-loop steering.

Subclasses :class:`AsyncEvolutionRunner` to add a 4th concurrent asyncio
task that polls the SQLite ``interactive_commands`` table via
:class:`WebController`.  Supports the same command vocabulary as the sync
interactive runner (PAUSE / RESUME / STOP / CONTINUE / SUGGEST / MERGE)
while retaining the full async concurrency pipeline for 5-10x speedup.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from shinka.core.async_runner import AsyncEvolutionRunner, AsyncRunningJob
from shinka.core.runner import FOLDER_PREFIX
from shinka.database.dbase import Program
from shinka.interactive import WebController

logger = logging.getLogger(__name__)


class AsyncInteractiveRunner(AsyncEvolutionRunner):
    """Async evolution runner with interactive steering support.

    Adds three capabilities on top of :class:`AsyncEvolutionRunner`:

    1. A 4th concurrent task (``_interactive_command_task``) that polls the
       SQLite command queue and dispatches interactive actions.
    2. Interaction-mode gating in ``_proposal_coordinator_task`` (auto /
       manual / wait) that controls when automatic proposals are generated.
    3. A keep-alive loop that keeps the runner alive after scheduled
       generations complete so experts can continue submitting suggestions
       and merges.
    """

    # --------------------------------------------------------------------- #
    # Initialization                                                         #
    # --------------------------------------------------------------------- #

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Interactive state ------------------------------------------------
        self.web_controller: Optional[WebController] = None

        # Pause gate: SET = not paused (coordinator can run),
        #             CLEAR = paused (coordinator blocks).
        self.interactive_paused = asyncio.Event()
        self.interactive_paused.set()  # start un-paused

        # Continue signal for manual mode: coordinator waits for this.
        self.continue_signal = asyncio.Event()

        # Step mode: generate one proposal then auto-pause.
        self._step_mode = False

        # Distinct from should_stop so we can distinguish "user clicked
        # stop" from "generations exhausted".
        self._interactive_stop_requested = False

    # --------------------------------------------------------------------- #
    # Main run() override                                                    #
    # --------------------------------------------------------------------- #

    _MAX_REENTRIES = 100  # safety guard against infinite re-entry

    async def run(self):
        """Main async evolution loop with interactive steering."""
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        tasks: list[asyncio.Task] = []

        try:
            # --- Setup (inherited) ----------------------------------------
            await self._setup_async()

            # --- Always init interactive controller -----------------------
            db_path = str(Path(self.results_dir) / "programs.sqlite")
            self.web_controller = WebController(db_path=db_path)
            logger.info("Interactive async web controller enabled")

            await self._verify_database_ready()

            # --- Greenlight gate: wait for user confirmation --------------
            await self._wait_for_greenlight()
            if self._interactive_stop_requested:
                return

            # On resume, start paused so user can review / suggest / merge
            if self._is_resuming:
                self.web_controller.pause()
                self.interactive_paused.clear()
                logger.info(
                    "Resumed run — starting paused for review. "
                    "Use Continue or Step in the UI to proceed."
                )

            # --- Generation loop (iterative re-entry on target increase) --
            for reentry_count in range(self._MAX_REENTRIES):
                if reentry_count > 0:
                    logger.info(
                        "Re-entering generation loop (attempt %d)",
                        reentry_count + 1,
                    )
                    self.cost_limit_reached = False
                    # Reset signals for the new generation cycle
                    self.should_stop.clear()
                    self.finalization_complete.clear()
                    self._interactive_stop_requested = False

                # --- Spawn concurrent tasks -------------------------------
                tasks = [
                    asyncio.create_task(self._job_monitor_task(), name="job_monitor"),
                    asyncio.create_task(
                        self._proposal_coordinator_task(),
                        name="proposal_coordinator",
                    ),
                ]

                if self.meta_summarizer:
                    tasks.append(
                        asyncio.create_task(
                            self._meta_summarizer_task(), name="meta_summarizer"
                        )
                    )

                tasks.append(
                    asyncio.create_task(
                        self._interactive_command_task(),
                        name="interactive_commands",
                    )
                )

                # --- Wait for scheduled generations to finish -------------
                await self.finalization_complete.wait()

                # --- Final embedding / meta (same as base class) ----------
                await self._run_final_operations()

                # --- Keep-alive loop (always active) ----------------------
                if self._interactive_stop_requested:
                    break

                self.web_controller.mark_idle()
                logger.info(
                    "Interactive: scheduled generations done — entering "
                    "async keep-alive mode.  Submit suggestions / merges "
                    "from the UI, increase the target, or send a stop "
                    "command to exit."
                )
                # Cancel the old coordinator (no more auto-proposals needed
                # unless the keep-alive restarts it)
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []  # reset so finally-block doesn't double-cancel

                should_reenter = await self._interactive_keepalive_loop_async()
                if not should_reenter:
                    break
                # Loop continues: target was increased
            else:
                logger.error(
                    "Interactive runner hit max re-entry limit (%d). "
                    "Stopping to prevent runaway loop.",
                    self._MAX_REENTRIES,
                )

        except Exception as e:
            logger.error(f"Error in async interactive run: {e}")
            raise
        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._cleanup_async()
            if self.web_controller:
                self.web_controller.mark_completed()

        await self._print_final_summary()

    # --------------------------------------------------------------------- #
    # Greenlight gate                                                        #
    # --------------------------------------------------------------------- #

    async def _wait_for_greenlight(self) -> None:
        """Wait for user to give greenlight via CLI or Web UI."""
        import threading

        target_gens = self.evo_config.num_generations
        best = await self.async_db.get_best_program_async()
        total_programs = await self.async_db.get_total_program_count_async()
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(
            None,
            lambda: self.web_controller.write_status(
                generation=self.completed_generations,
                best_score=best.combined_score if best and best.combined_score else 0.0,
                queued_jobs=0,
                total_programs=total_programs,
                target_generations=target_gens,
                waiting_for_start=True,
                is_resuming=self._is_resuming,
            ),
        )

        greenlight = threading.Event()
        started_from = "cli"

        def _cli_prompt() -> None:
            nonlocal started_from
            try:
                if self._is_resuming:
                    input(
                        f"\n>>> Resuming from generation {self.completed_generations}"
                        f"/{target_gens}. "
                        "Press Enter to continue (starts paused for review), "
                        "or connect via Web UI...\n"
                    )
                else:
                    input(
                        "\n>>> Press Enter to start generation "
                        "(or connect via Web UI and click Start)...\n"
                    )
                if not greenlight.is_set():
                    started_from = "cli"
                    greenlight.set()
            except EOFError:
                pass

        cli_thread = threading.Thread(target=_cli_prompt, daemon=True)
        cli_thread.start()

        while not greenlight.is_set():
            actions = await loop.run_in_executor(
                None, self.web_controller.process_commands
            )
            for action in actions:
                await self._handle_interactive_action_async(action)

            if self.web_controller.start_requested:
                started_from = "web"
                greenlight.set()
                break
            if self.web_controller.stop_requested:
                self._interactive_stop_requested = True
                greenlight.set()
                break
            await asyncio.sleep(0.5)

        if started_from == "web":
            logger.info("Started via Web UI")
        elif self._interactive_stop_requested:
            logger.info("Stop requested before start")
        else:
            logger.info("Started via CLI")

    # --------------------------------------------------------------------- #
    # Final operations helper (extracted from base run())                    #
    # --------------------------------------------------------------------- #

    async def _run_final_operations(self):
        """Embedding recomputation + final meta summary."""
        if self.verbose:
            logger.info("Performing final embedding recomputation and meta summary...")

        # Final embedding recomputation
        if self.embedding_client:
            try:
                await asyncio.wait_for(
                    self.async_db.force_recompute_embeddings_async(),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Final embedding recomputation timed out after 2 min")
            except Exception as e:
                logger.error(f"Error in final embedding recomputation: {e}")

        # Final meta summary
        if self.meta_summarizer:
            try:
                best_program = await asyncio.wait_for(
                    self.async_db.get_best_program_async(), timeout=30.0
                )
                if best_program:
                    await asyncio.wait_for(
                        self.meta_summarizer.perform_final_summary_async(
                            str(self.results_dir),
                            best_program,
                            self.db.config,
                        ),
                        timeout=600.0,
                    )
            except asyncio.TimeoutError:
                logger.warning("Final meta summary timed out")
            except Exception as e:
                logger.error(f"Error in final meta summary: {e}")

        await asyncio.sleep(0.5)
        self._save_bandit_state()

    # --------------------------------------------------------------------- #
    # 4th concurrent task: interactive command polling                       #
    # --------------------------------------------------------------------- #

    async def _interactive_command_task(self):
        """Poll the SQLite interactive_commands table and dispatch actions.

        Runs blocking ``WebController`` I/O in a thread-pool executor so
        the asyncio event loop is never blocked.
        """
        logger.info("Interactive command task started")
        loop = asyncio.get_event_loop()

        while not self._interactive_stop_requested:
            try:
                # --- Poll commands (blocking I/O → executor) --------------
                actions = await loop.run_in_executor(
                    None,
                    self.web_controller.process_commands,
                )

                for action in actions:
                    await self._handle_interactive_action_async(action)

                # --- React to flag changes --------------------------------
                if self.web_controller.stop_requested:
                    self._interactive_stop_requested = True
                    logger.info("Interactive: stop requested")
                    self.should_stop.set()
                    self.slot_available.set()
                    break

                if self.web_controller.continue_requested:
                    self.continue_signal.set()
                    self.web_controller.clear_continue()

                # Step mode: unpause + allow one submission
                if self.web_controller.step_requested:
                    self._step_mode = True
                    self.web_controller.resume()
                    self.interactive_paused.set()
                    logger.info(
                        "Interactive: step mode — will generate 1 node then pause"
                    )

                if self.web_controller.is_paused:
                    self.interactive_paused.clear()
                else:
                    self.interactive_paused.set()

                # --- Write status (blocking I/O → executor) ---------------
                await self._write_interactive_status(loop)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in interactive command task: {e}")

            await asyncio.sleep(1.0)

    async def _write_interactive_status(self, loop: asyncio.AbstractEventLoop):
        """Push current run status to the interactive_status table."""
        try:
            best = await self.async_db.get_best_program_async()
            total_programs = await self.async_db.get_total_program_count_async()
            is_idle = (
                len(self.running_jobs) == 0 and len(self.active_proposal_tasks) == 0
            )

            await loop.run_in_executor(
                None,
                lambda: self.web_controller.write_status(
                    generation=self.completed_generations,
                    best_score=(
                        best.combined_score if best and best.combined_score else 0.0
                    ),
                    queued_jobs=len(self.running_jobs)
                    + len(self.active_proposal_tasks),
                    total_programs=total_programs,
                    target_generations=self.evo_config.num_generations,
                    idle=is_idle,
                ),
            )
        except Exception as e:
            logger.debug(f"Error writing interactive status: {e}")

    # --------------------------------------------------------------------- #
    # Interactive action handler                                             #
    # --------------------------------------------------------------------- #

    async def _handle_interactive_action_async(self, action: dict):
        """Process a SUGGEST or MERGE action by spawning an async proposal.

        The proposal runs concurrently alongside any automatic proposals
        managed by the coordinator.  It reuses ``_run_patch_async`` with
        the interactive ``patch_type_override`` and ``user_suggestions``
        parameters and **skips novelty checking** (expert intent is
        respected).
        """
        action_type = action["action"]
        prompt = action.get("prompt", "")
        patch_type_override = action.get("patch_type", "full")

        if action_type == "set_target":
            new_target = action["target_generations"]
            old_target = self.evo_config.num_generations
            self.evo_config.num_generations = new_target
            logger.info(
                "Interactive: target generations changed %d → %d",
                old_target,
                new_target,
            )
            return

        if action_type == "suggest":
            parent_id = action["parent_id"]
            parent_program = await self.async_db.get_async(parent_id)
            if parent_program is None:
                logger.error("Interactive suggest: parent %s not found", parent_id)
                return
            (
                archive_programs,
                top_k_programs,
            ) = await self.async_db.sample_inspirations_for_parent_async(
                parent_program,
                self.db_config.num_archive_inspirations,
                self.db_config.num_top_k_inspirations,
            )

        elif action_type == "merge":
            parent_ids = action["parent_ids"]
            parent_program = await self.async_db.get_async(parent_ids[0])
            if parent_program is None:
                logger.error(
                    "Interactive merge: primary parent %s not found",
                    parent_ids[0],
                )
                return
            archive_programs = await self.async_db.get_programs_by_ids_async(
                parent_ids[1:]
            )
            top_k_programs: List[Program] = []
            patch_type_override = "cross"
        else:
            logger.warning("Unknown interactive action: %s", action_type)
            return

        # Allocate a generation slot.
        # Safe without a lock: asyncio is single-threaded, and we only
        # yield (await) AFTER the increment.
        generation = self.next_generation_to_submit
        self.assigned_generations.add(generation)
        self.next_generation_to_submit += 1

        task_id = str(uuid.uuid4())
        task = asyncio.create_task(
            self._generate_interactive_proposal_async(
                generation=generation,
                task_id=task_id,
                parent_program=parent_program,
                archive_programs=archive_programs,
                top_k_programs=top_k_programs,
                patch_type_override=patch_type_override,
                user_suggestions=prompt,
                action_type=action_type,
            ),
            name=f"interactive_{action_type}_{generation}",
        )
        self.active_proposal_tasks[task_id] = task
        logger.info(
            "Interactive %s: spawned proposal for gen %d (parent=%s)",
            action_type,
            generation,
            parent_program.id,
        )

    # --------------------------------------------------------------------- #
    # Interactive proposal pipeline                                          #
    # --------------------------------------------------------------------- #

    async def _generate_interactive_proposal_async(
        self,
        generation: int,
        task_id: str,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        patch_type_override: str,
        user_suggestions: str,
        action_type: str,
    ) -> Optional[AsyncRunningJob]:
        """Generate a proposal from an interactive command.

        Reuses the base class ``_run_patch_async`` with interactive
        overrides and skips novelty checking (expert intent is respected).
        """
        try:
            # --- Directory setup ------------------------------------------
            gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"
            exec_fname = f"{gen_dir}/main.{self.lang_ext}"
            results_dir = f"{gen_dir}/results"

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: (
                    Path(gen_dir).mkdir(parents=True, exist_ok=True),
                    Path(results_dir).mkdir(parents=True, exist_ok=True),
                ),
            )

            # --- Write the parent code to the generation directory --------
            if parent_program.code:
                await loop.run_in_executor(
                    None,
                    lambda: Path(exec_fname).write_text(
                        parent_program.code, encoding="utf-8"
                    ),
                )

            # --- Meta recommendations (best-effort) ----------------------
            meta_recs = None
            if self.meta_summarizer:
                if self.evo_config.sample_single_meta_rec:
                    meta_recs = self.meta_summarizer.get_sampled_recommendation()
                else:
                    meta_recs, _, _ = self.meta_summarizer.get_current()

            # --- LLM model selection (transparent pass-through) -----------
            model_sample_probs, model_posterior = None, None
            if self.llm_selection is not None:
                model_sample_probs, model_posterior = self.llm_selection.select_llm()

            # --- Run patch with interactive overrides ---------------------
            patch_result = await self._run_patch_async(
                parent_program,
                archive_programs,
                top_k_programs,
                generation,
                meta_recs,
                patch_type_override=patch_type_override,
                user_suggestions=user_suggestions,
                model_sample_probs=model_sample_probs,
                model_posterior=model_posterior,
            )

            if not patch_result:
                logger.warning(
                    "Interactive %s: patch generation failed for gen %d",
                    action_type,
                    generation,
                )
                return None

            code_diff, meta_patch_data, success = patch_result
            if not success:
                logger.warning(
                    "Interactive %s: patch not successful for gen %d",
                    action_type,
                    generation,
                )
                return None

            # --- Tag with interactive metadata ----------------------------
            meta_patch_data["source"] = f"human_{action_type}"
            if user_suggestions:
                meta_patch_data["human_prompt"] = user_suggestions

            # --- Get code embedding ---------------------------------------
            code_embedding, embed_cost = await self._get_code_embedding_async(
                exec_fname
            )

            # --- Submit for evaluation (skip novelty check) ---------------
            job_id = await self.scheduler.submit_async_nonblocking(
                exec_fname, results_dir
            )

            api_costs = meta_patch_data.get("api_costs", 0.0)
            running_job = AsyncRunningJob(
                job_id=job_id,
                exec_fname=exec_fname,
                results_dir=results_dir,
                start_time=time.time(),
                generation=generation,
                parent_id=parent_program.id,
                archive_insp_ids=[p.id for p in archive_programs],
                top_k_insp_ids=[p.id for p in top_k_programs],
                code_diff=code_diff,
                meta_patch_data=meta_patch_data,
                code_embedding=code_embedding,
                embed_cost=embed_cost,
                novelty_cost=0.0,
                proposal_task_id=task_id,
            )

            # --- Update cost tracking -------------------------------------
            proposal_total_cost = api_costs + embed_cost
            self.total_api_cost += proposal_total_cost
            self._update_avg_proposal_cost(proposal_total_cost)

            # --- Wait for evaluation slot if at capacity ------------------
            while len(self.running_jobs) >= self.max_evaluation_jobs:
                if self.should_stop.is_set():
                    try:
                        await self.scheduler.cancel_job_async(job_id)
                    except Exception:
                        pass
                    return None
                await asyncio.sleep(0.5)

            # --- Track the job --------------------------------------------
            self.running_jobs.append(running_job)
            self.submitted_jobs[str(job_id)] = running_job
            self.slot_available.set()

            logger.info(
                "Interactive %s: submitted gen %d for eval (parent=%s, cost=$%.4f)",
                action_type,
                generation,
                parent_program.id,
                proposal_total_cost,
            )
            return running_job

        except Exception as e:
            logger.error("Error in interactive proposal gen %d: %s", generation, e)
            return None
        finally:
            # Remove from active proposals regardless of outcome
            self.active_proposal_tasks.pop(task_id, None)

    # --------------------------------------------------------------------- #
    # Proposal coordinator override (interaction mode gating)                #
    # --------------------------------------------------------------------- #

    async def _proposal_coordinator_task(self):
        """Coordinate proposal generation with pause gate.

        Blocks when the user pauses via the UI.  Step mode allows one
        proposal then auto-pauses.
        """
        while not self.should_stop.is_set():
            try:
                # --- Gate: Pause ------------------------------------------
                if not self.interactive_paused.is_set():
                    logger.debug("Proposal coordinator paused by interactive command")
                    # Wait for either un-pause or stop
                    unpause_task = asyncio.create_task(self.interactive_paused.wait())
                    stop_task = asyncio.create_task(self.should_stop.wait())
                    done, pending = await asyncio.wait(
                        [unpause_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()
                    if self.should_stop.is_set():
                        break
                    continue

                # --- Base class coordinator logic -------------------------
                if self._is_system_stuck():
                    recovery_success = await self._handle_stuck_system()
                    if not recovery_success:
                        break

                proposals_remaining = max(
                    0,
                    self.evo_config.num_generations - self.next_generation_to_submit,
                )

                should_generate_proposals = not self.cost_limit_reached
                if (
                    not self.cost_limit_reached
                    and self.evo_config.max_api_costs is not None
                ):
                    committed_cost = self._get_committed_cost()
                    if committed_cost >= self.evo_config.max_api_costs:
                        should_generate_proposals = False
                        self.cost_limit_reached = True
                        if self.verbose:
                            in_flight_cost = committed_cost - self.total_api_cost
                            logger.info(
                                f"Cost budget reached: "
                                f"actual=${self.total_api_cost:.4f} + "
                                f"in-flight=${in_flight_cost:.4f} = "
                                f"${committed_cost:.4f} >= "
                                f"${self.evo_config.max_api_costs:.2f}"
                            )

                pipeline_capacity = len(self.running_jobs) + len(
                    self.active_proposal_tasks
                )
                pipeline_target = self.max_evaluation_jobs
                proposals_needed = min(
                    max(0, pipeline_target - pipeline_capacity),
                    proposals_remaining,
                    self.max_proposal_jobs - len(self.active_proposal_tasks),
                )

                if proposals_needed > 0 and should_generate_proposals:
                    # Step mode: only submit one proposal then auto-pause
                    if self._step_mode:
                        proposals_needed = 1

                    if self.verbose:
                        logger.info(
                            f"Starting {proposals_needed} new proposals. "
                            f"Pipeline: {pipeline_capacity}/{pipeline_target} "
                            f"(running={len(self.running_jobs)}, "
                            f"proposals={len(self.active_proposal_tasks)}"
                            f"/{self.max_proposal_jobs}), "
                            f"Remaining: {proposals_remaining}"
                        )
                    await self._start_proposals(proposals_needed)
                    self._record_progress()

                    # Step mode: auto-pause after submission
                    if self._step_mode:
                        self._step_mode = False
                        self.interactive_paused.clear()
                        self.web_controller.pause()
                        logger.info("Interactive: step complete, auto-pausing")

                await self._cleanup_completed_proposal_tasks()
                await self._wait_for_slot_or_stop(timeout=5.0)

            except Exception as e:
                logger.error(f"Error in proposal coordinator: {e}")
                await asyncio.sleep(1)

    # --------------------------------------------------------------------- #
    # Keep-alive loop                                                        #
    # --------------------------------------------------------------------- #

    async def _interactive_keepalive_loop_async(self) -> bool:
        """Keep the runner alive after scheduled generations for interactive use.

        Restarts the command polling and job monitoring tasks, then waits
        until the expert sends a STOP command or increases the target.

        Returns True if the target was increased (caller should re-enter
        the main generation loop), False otherwise.
        """
        logger.info("Interactive: async keep-alive started")

        # Reset stop signals so tasks can run
        self.should_stop.clear()
        self.finalization_complete.clear()
        self._interactive_stop_requested = False
        should_reenter = False

        cmd_task = asyncio.create_task(
            self._interactive_command_task(), name="keepalive_commands"
        )
        monitor_task = asyncio.create_task(
            self._job_monitor_task(), name="keepalive_monitor"
        )

        try:
            # Poll until stop is requested or target increases
            while not self.should_stop.is_set():
                # Check if target was increased beyond completed
                if self.evo_config.num_generations > self.completed_generations:
                    logger.info(
                        "Interactive keep-alive: target increased to %d "
                        "(completed %d), re-entering generation loop",
                        self.evo_config.num_generations,
                        self.completed_generations,
                    )
                    should_reenter = True
                    break
                await asyncio.sleep(1.0)

            if not should_reenter:
                # Drain remaining running jobs
                logger.info(
                    "Interactive keep-alive: stop requested, draining %d running jobs...",
                    len(self.running_jobs),
                )
                drain_timeout = 300.0  # 5 min max wait
                drain_start = time.time()
                while self.running_jobs:
                    if time.time() - drain_start > drain_timeout:
                        logger.warning(
                            "Keep-alive drain timed out, %d jobs still running",
                            len(self.running_jobs),
                        )
                        break
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Interactive keep-alive cancelled")
        finally:
            cmd_task.cancel()
            monitor_task.cancel()
            await asyncio.gather(cmd_task, monitor_task, return_exceptions=True)
            logger.info("Interactive: async keep-alive ended")

        return should_reenter
