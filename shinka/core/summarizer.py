from typing import List, Optional, Tuple
import logging
import json
import os
import tempfile
from dataclasses import fields
import random
import re
from pathlib import Path
from shinka.database import Program
from shinka.llm import LLMClient
from shinka.prompts import (
    construct_individual_program_msg,
    META_STEP1_SYSTEM_MSG,
    META_STEP1_USER_MSG,
    META_STEP2_SYSTEM_MSG,
    META_STEP2_USER_MSG,
    META_STEP3_SYSTEM_MSG,
    META_STEP3_USER_MSG,
)

logger = logging.getLogger(__name__)


class MetaSummarizer:
    """Handles meta-level summarization and recommendation generation."""

    def __init__(
        self,
        meta_llm_client: Optional[LLMClient] = None,
        language: str = "python",
        use_text_feedback: bool = False,
        max_recommendations: int = 5,
        async_mode: bool = False,
    ):
        self.meta_llm_client = meta_llm_client
        self.language = language
        self.use_text_feedback = use_text_feedback
        self.max_recommendations = max_recommendations
        self.async_mode = async_mode  # Flag for async mode

        # Meta state
        self.meta_summary = None
        self.meta_scratch_pad = None  # New: Global insights scratchpad
        self.meta_recommendations = None
        self.meta_recommendations_history = []

        # Track programs evaluated since last meta query for persistent memory
        self.evaluated_since_last_meta: List[Program] = []

        # Track the accumulated count of programs processed in meta updates
        self.total_programs_processed = 0

    def add_evaluated_program(self, program: Program) -> None:
        """Add newly evaluated program to the tracking list."""
        logger.debug(
            f"Evaluating program {program.id} for meta memory: "
            f"correct={program.correct}"
        )

        # Track ALL evaluated programs (both correct and incorrect)
        # for meta learning
        self.evaluated_since_last_meta.append(program)
        logger.info(
            f"Added program {program.id} to meta memory tracking "
            f"(correct={program.correct}, "
            f"total: {len(self.evaluated_since_last_meta)})"
        )

        # Log when we're getting close to the meta update threshold
        if hasattr(self, "_last_logged_count"):
            if len(self.evaluated_since_last_meta) != self._last_logged_count:
                logger.debug(
                    f"Meta memory: {len(self.evaluated_since_last_meta)} "
                    f"programs tracked"
                )
        self._last_logged_count = len(self.evaluated_since_last_meta)

    def should_update_meta(self, meta_rec_interval: Optional[int]) -> bool:
        """Check if meta update should be performed based on interval.

        Now triggers based on the number of unprocessed programs rather than
        generation intervals for better timing with parallel jobs.
        """
        if meta_rec_interval is None:
            return False

        # In async mode, the async wrapper handles LLM client, so don't check it
        # In sync mode, we need a meta_llm_client
        if not self.async_mode and not self.meta_llm_client:
            return False

        # Use number of unprocessed programs instead of generation count
        unprocessed_count = len(self.evaluated_since_last_meta)
        return unprocessed_count >= meta_rec_interval

    def update_meta_memory(
        self, best_program: Optional[Program] = None
    ) -> Tuple[Optional[str], float]:
        """
        Perform 3-step meta-analysis and update internal state.
        Returns tuple of (updated_recommendations, total_cost) or
        (None, 0.0) if no update occurred.
        """
        if not self.meta_llm_client:
            logger.warning("No meta LLM client configured")
            return None, 0.0

        # Use recently evaluated programs for memory scratchpad
        # Make a copy to avoid issues if the list is modified during processing
        programs_to_analyze = (
            self.evaluated_since_last_meta.copy()
            if self.evaluated_since_last_meta
            else []
        )

        if len(programs_to_analyze) == 0:
            logger.info("No programs evaluated since last meta query, skipping")
            return None, 0.0

        total_meta_cost = 0.0

        try:
            # Step 1: Create individual program summaries
            individual_summaries, step1_cost = self._step1_individual_summaries(
                programs_to_analyze
            )
            total_meta_cost += step1_cost
            if not individual_summaries:
                logger.error("Step 1 failed - no individual summaries generated")
                return None, total_meta_cost

            # Step 2: Generate global insights scratchpad
            global_insights, step2_cost = self._step2_global_insights(
                individual_summaries, best_program
            )
            total_meta_cost += step2_cost
            if not global_insights:
                logger.error("Step 2 failed - no global insights generated")
                return None, total_meta_cost

            # Step 3: Generate recommendations based on insights
            recommendations, step3_cost = self._step3_generate_recommendations(
                global_insights, best_program
            )
            total_meta_cost += step3_cost
            if not recommendations:
                logger.error("Step 3 failed - no recommendations generated")
                return None, total_meta_cost

            # Update internal state
            # Concatenate new individual summaries to existing ones
            if self.meta_summary:
                self.meta_summary += "\n\n" + individual_summaries
            else:
                self.meta_summary = individual_summaries

            self.meta_scratch_pad = global_insights
            self.meta_recommendations = recommendations

            # Store the newly generated recommendations in history immediately
            if recommendations and isinstance(recommendations, str):
                self.meta_recommendations_history.append(recommendations)
                logger.debug(
                    f"Added new recommendations to history "
                    f"(total: {len(self.meta_recommendations_history)})"
                )

            logger.info(
                f"==> Meta-analysis completed successfully with 3-step process (total cost: ${total_meta_cost:.4f})"
            )
        except Exception as e:
            logger.error(f"Failed to complete 3-step meta-analysis: {e}")
            return None, total_meta_cost

        # Clear the evaluated programs list immediately after processing
        # This ensures that only programs added AFTER this meta update
        # will be saved as "unprocessed" programs
        num_processed = len(self.evaluated_since_last_meta)
        self.total_programs_processed += num_processed
        self.evaluated_since_last_meta = []
        logger.info(
            f"Processed and cleared {num_processed} programs from meta memory "
            f"(total processed: {self.total_programs_processed})"
        )

        return (
            (
                self.meta_recommendations
                if isinstance(self.meta_recommendations, str)
                else None
            ),
            total_meta_cost,
        )

    def get_unprocessed_program_count(self) -> int:
        """Get the count of unprocessed programs awaiting meta analysis."""
        return len(self.evaluated_since_last_meta)

    def get_recommendations_history_count(self) -> int:
        """Get the count of previous recommendations stored in history."""
        return len(self.meta_recommendations_history)

    def get_total_programs_processed(self) -> int:
        """Get the total count of programs processed in meta updates."""
        return self.total_programs_processed

    def perform_final_summary(
        self, results_dir: str, best_program: Optional[Program] = None
    ) -> bool:
        """Perform a final meta summary if there are unprocessed programs."""
        if not self.meta_llm_client:
            logger.info("No meta LLM client configured, skipping final summary")
            return False

        unprocessed_count = len(self.evaluated_since_last_meta)
        if unprocessed_count == 0:
            logger.info("No unprocessed programs for final summary")
            return False

        logger.info(
            f"Performing final meta summary for {unprocessed_count} "
            f"remaining programs..."
        )

        updated_recs, meta_cost = self.update_meta_memory(best_program)
        if updated_recs:
            self.write_meta_output(results_dir)
            logger.info(f"Final meta summary completed (cost: ${meta_cost:.4f})")
            return True
        else:
            logger.warning("Final meta summary failed to generate recommendations")
            return False

    def _step1_individual_summaries(
        self, programs_to_analyze: List[Program]
    ) -> Tuple[Optional[str], float]:
        """Step 1: Create individual summaries for each program using batch queries."""
        if not programs_to_analyze:
            logger.warning("No programs to analyze in Step 1")
            return None, 0.0

        # Create individual program messages for batch processing
        user_messages, generation_ids, patch_names, correct_programs = [], [], [], []
        for program in programs_to_analyze:
            individual_program_msg = construct_individual_program_msg(
                program,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
            generation_ids.append(program.generation)
            patch_names.append(program.metadata["patch_name"])
            correct_programs.append(program.correct)
            user_msg = META_STEP1_USER_MSG.replace(
                "{individual_program_msg}", individual_program_msg
            )
            user_messages.append(user_msg)

        # Use batch query to process all programs
        num_programs = len(programs_to_analyze)
        logger.info(f"==> Step 1 - Processing {num_programs} programs with batch query")
        responses = self.meta_llm_client.batch_kwargs_query(
            num_samples=num_programs,
            msg=user_messages,
            system_msg=META_STEP1_SYSTEM_MSG,
        )

        if not responses:
            logger.error("Step 1: Failed to get responses from meta LLM client")
            return None, 0.0

        # Filter out None responses and combine summaries
        valid_responses = [r for r in responses if r is not None]
        if not valid_responses:
            logger.error("Step 1: All batch responses were None")
            return None, 0.0

        # Combine all individual summaries
        combined_summaries = []
        total_cost = 0.0
        for i, response in enumerate(valid_responses):
            if response and response.content:
                program_summary = response.content.strip()
                program_summary += "\n**Program Identifier:** "
                program_summary += f"Generation {generation_ids[i]} - Patch Name {patch_names[i]} - Correct Program: {correct_programs[i]}"
                combined_summaries.append(program_summary)
                total_cost += response.cost or 0.0
            else:
                logger.warning(f"Step 1: Empty response for program {i}")

        # Sort combined_summaries by generation (using generation_ids)
        # Zip together summaries and their generation, sort, then extract summaries
        summaries_with_gen = list(zip(generation_ids, combined_summaries))
        summaries_with_gen.sort(key=lambda x: x[0])
        combined_summaries = [summary for _, summary in summaries_with_gen]

        if not combined_summaries:
            logger.error("Step 1: No valid summaries generated")
            return None, total_cost

        # Join all summaries with double newlines
        final_summary = "\n\n".join(combined_summaries)
        logger.info(
            f"==> Step 1 - {len(combined_summaries)}/{num_programs} "
            f"individual summaries generated (cost: ${total_cost:.4f})"
        )
        return final_summary, total_cost

    def _step2_global_insights(
        self, individual_summaries: str, best_program: Optional[Program] = None
    ) -> Tuple[Optional[str], float]:
        """Step 2: Generate global insights from individual summaries."""
        previous_insights = self.meta_scratch_pad or "*No previous insights available.*"

        # Format best program information
        if best_program:
            from shinka.prompts import construct_individual_program_msg

            best_program_info = construct_individual_program_msg(
                best_program,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
        else:
            best_program_info = "*No best program information available.*"

        user_msg = (
            META_STEP2_USER_MSG.replace("{individual_summaries}", individual_summaries)
            .replace("{previous_insights}", previous_insights)
            .replace("{best_program_info}", best_program_info)
        )
        llm_params = self.meta_llm_client.get_kwargs()
        response = self.meta_llm_client.query(
            msg=user_msg,
            system_msg=META_STEP2_SYSTEM_MSG,
            llm_kwargs=llm_params,
        )

        if response is None:
            logger.error("Step 2: Failed to get response from meta LLM client")
            return None, 0.0

        cost = response.cost or 0.0
        logger.info(f"==> Step 2 - Global insights generated (cost: ${cost:.4f})")
        return response.content.strip(), cost

    def _step3_generate_recommendations(
        self, global_insights: str, best_program: Optional[Program] = None
    ) -> Tuple[Optional[str], float]:
        """Step 3: Generate recommendations based on global insights."""
        previous_recommendations = (
            self.meta_recommendations or "*No previous recommendations available.*"
        )

        # Format best program information
        if best_program:
            from shinka.prompts import construct_individual_program_msg

            best_program_info = construct_individual_program_msg(
                best_program,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
        else:
            best_program_info = "*No best program information available.*"

        user_msg = (
            META_STEP3_USER_MSG.replace("{global_insights}", global_insights)
            .replace("{previous_recommendations}", previous_recommendations)
            .replace("{max_recommendations}", str(self.max_recommendations))
            .replace("{best_program_info}", best_program_info)
        )

        llm_params = self.meta_llm_client.get_kwargs()
        response = self.meta_llm_client.query(
            msg=user_msg,
            system_msg=META_STEP3_SYSTEM_MSG,
            llm_kwargs=llm_params,
        )

        if response is None:
            logger.error("Step 3: Failed to get response from meta LLM client")
            return None, 0.0

        cost = response.cost or 0.0
        logger.info(f"==> Step 3 - Recommendations generated (cost: ${cost:.4f})")
        return response.content.strip(), cost

    def get_current(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get current meta recommendations without updating."""
        recommendations = (
            self.meta_recommendations
            if isinstance(self.meta_recommendations, str)
            else None
        )
        summary = self.meta_summary if isinstance(self.meta_summary, str) else None
        scratch_pad = (
            self.meta_scratch_pad if isinstance(self.meta_scratch_pad, str) else None
        )

        # Debug logging
        logger.debug(
            f"get_current() returning: "
            f"recommendations={'Yes' if recommendations else 'No'}, "
            f"summary={'Yes' if summary else 'No'}, "
            f"scratch_pad={'Yes' if scratch_pad else 'No'}"
        )
        if recommendations:
            rec_preview = (
                recommendations[:100] + "..."
                if len(recommendations) > 100
                else recommendations
            )
            logger.debug(f"Current recommendations preview: {rec_preview}")

        return (recommendations, summary, scratch_pad)

    def get_sampled_recommendation(self) -> Optional[str]:
        """Sample a single recommendation from the current recommendations.

        Parses the numbered list of recommendations and returns one randomly.
        This can be used to provide diversity across parallel generations.

        Returns:
            A single recommendation string, or None if no recommendations exist.
        """
        logger.info(
            f"get_sampled_recommendation called, "
            f"meta_recommendations exists: {bool(self.meta_recommendations)}"
        )
        if not self.meta_recommendations or self.meta_recommendations == "none":
            logger.info("No meta recommendations available to sample from")
            return None

        # Parse numbered recommendations (format: "1. ...\n2. ...\n3. ...")
        pattern = r"^\d+\.\s+"
        lines = self.meta_recommendations.strip().split("\n")

        # Group lines into recommendations (handle multi-line recommendations)
        recommendations = []
        current_rec = []
        for line in lines:
            if re.match(pattern, line):
                if current_rec:
                    recommendations.append("\n".join(current_rec))
                current_rec = [line]
            elif current_rec:
                current_rec.append(line)
        if current_rec:
            recommendations.append("\n".join(current_rec))

        if not recommendations:
            logger.debug("No parseable recommendations found")
            return None

        # Sample one randomly and remove the leading number (e.g., "1. ")
        sampled = random.choice(recommendations)
        sampled = re.sub(r"^\d+\.\s*", "", sampled)
        logger.info(
            f"Sampled 1 recommendation from {len(recommendations)} total: "
            f"{sampled[:80]}..."
        )
        return sampled

    def _build_previous_context(self) -> str:
        """Build context string from previous meta state."""
        context_parts = []

        if self.meta_summary and self.meta_summary != "none":
            context_parts.append("## Previous Summary")
            context_parts.append(str(self.meta_summary))

        if self.meta_recommendations and self.meta_recommendations != "none":
            rec_count = self._count_recommendations(self.meta_recommendations)
            context_parts.append(
                f"\n## Previous Recommendations "
                f"({rec_count}/{self.max_recommendations} items)"
            )
            context_parts.append(str(self.meta_recommendations))

        if not context_parts:
            return "*No previous memory state - this is the first meta update.*"

        return "\n".join(context_parts)

    def _count_recommendations(self, text: str) -> int:
        """Count recommendation items (lines starting with •)."""
        if not text:
            return 0
        return len([line for line in text.split("\n") if line.strip().startswith("•")])

    def save_meta_state(self, filepath: str | Path) -> None:
        """Atomically checkpoint complete meta state, raising on write failure."""
        state = {
            "unprocessed_programs": [
                program.to_dict() for program in self.evaluated_since_last_meta
            ],
            "meta_summary": self.meta_summary,
            "meta_scratch_pad": self.meta_scratch_pad,
            "meta_recommendations": self.meta_recommendations,
            "meta_recommendations_history": self.meta_recommendations_history,
            "total_programs_meta_processed": self.total_programs_processed,
        }
        # Serialize before creating a temporary file; never publish partial state.
        serialized = json.dumps(state, indent=2, allow_nan=False)
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_meta_state(self, filepath: str | Path) -> bool:
        """Restore a complete checkpoint, leaving current state intact on failure."""
        path = Path(filepath)
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("checkpoint must contain an object")
            text_fields = ("meta_summary", "meta_scratch_pad", "meta_recommendations")
            for field in text_fields:
                value = state[field]
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{field} must be text or null")
            history = state["meta_recommendations_history"]
            if not isinstance(history, list) or any(
                not isinstance(item, str) for item in history
            ):
                raise ValueError("recommendation history must contain strings")
            processed = state["total_programs_meta_processed"]
            if type(processed) is not int or processed < 0:
                raise ValueError(
                    "processed program count must be a nonnegative integer"
                )
            pending = state["unprocessed_programs"]
            if not isinstance(pending, list):
                raise ValueError("unprocessed programs must contain a list")
            programs = []
            program_ids = set()
            for item in pending:
                if not isinstance(item, dict):
                    raise ValueError("unprocessed program must contain an object")
                if set(item) != {field.name for field in fields(Program)}:
                    raise ValueError(
                        "pending program must contain the complete program schema"
                    )
                if any(
                    not isinstance(item[field], str)
                    for field in ("id", "code", "language")
                ):
                    raise ValueError(
                        "program identity, code and language must be strings"
                    )
                if type(item["generation"]) is not int or item["generation"] < 0:
                    raise ValueError("program generation must be a nonnegative integer")
                for field in (
                    "metadata",
                    "public_metrics",
                    "private_metrics",
                    "review_priority_data",
                ):
                    if not isinstance(item[field], dict):
                        raise ValueError(f"program {field} must be an object")
                for field in ("combined_score", "complexity", "timestamp"):
                    if item[field] is not None and type(item[field]) not in (
                        int,
                        float,
                    ):
                        raise ValueError(f"program {field} must be numeric or null")
                for field in ("correct", "in_archive"):
                    if type(item[field]) is not bool:
                        raise ValueError(f"program {field} must be a boolean")
                for field in ("archive_inspiration_ids", "top_k_inspiration_ids"):
                    if not isinstance(item[field], list) or any(
                        not isinstance(value, str) for value in item[field]
                    ):
                        raise ValueError(f"program {field} must contain strings")
                if not item["id"] or item["id"] in program_ids:
                    raise ValueError(
                        "pending program identities must be nonempty and unique"
                    )
                program_ids.add(item["id"])
                restored = Program(**item)
                if restored.to_dict() != item:
                    raise ValueError(
                        "pending program contains noncanonical derived data"
                    )
                programs.append(restored)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            logger.error("Cannot restore meta checkpoint %s: %s", path, exc)
            return False

        # Commit only after every field and pending program has been validated.
        self.meta_summary = state["meta_summary"]
        self.meta_scratch_pad = state["meta_scratch_pad"]
        self.meta_recommendations = state["meta_recommendations"]
        self.meta_recommendations_history = history
        self.total_programs_processed = processed
        self.evaluated_since_last_meta = programs
        return True

    def write_meta_output(self, results_dir: str) -> None:
        """Write meta summary, scratchpad, and recommendations to a file."""
        output_str = ""

        if self.meta_summary:
            output_str += "# INDIVIDUAL PROGRAM SUMMARIES\n\n"
            output_str += (
                "The following are summaries of individual programs "
                "evaluated since the last meta update:\n\n"
            )
            output_str += str(self.meta_summary)
            output_str += "\n\n"

        if self.meta_scratch_pad:
            output_str += "# GLOBAL INSIGHTS SCRATCHPAD\n\n"
            output_str += (
                "The following are global insights about optimization "
                "approaches and their effectiveness:\n\n"
            )
            output_str += str(self.meta_scratch_pad)
            output_str += "\n\n"

        if self.meta_recommendations:
            output_str += "# META RECOMMENDATIONS\n\n"
            output_str += (
                "The following are actionable recommendations for the next "
                "program generations:\n\n"
            )
            output_str += str(self.meta_recommendations)

        if output_str:
            # Create meta subdirectory if it doesn't exist
            meta_dir = Path(results_dir) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)

            meta_path = meta_dir / f"meta_{self.total_programs_processed}.txt"
            with meta_path.open("w", encoding="utf-8") as f:
                f.write(output_str)
            logger.info(f"Wrote meta output to {meta_path}")
