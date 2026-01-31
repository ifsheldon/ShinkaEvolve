from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from shinka.database import ProgramDatabase, Program


@dataclass
class InteractiveSelection:
    """Represents user choices for interactive evolution."""

    use_auto_parent: bool = True
    use_auto_patch_type: bool = True
    use_auto_inspirations: bool = True
    parent_id: Optional[str] = None
    patch_type: Optional[str] = None
    archive_inspiration_ids: List[str] = field(default_factory=list)
    top_k_inspiration_ids: List[str] = field(default_factory=list)
    user_suggestions: Optional[str] = None


class InteractiveController:
    """CLI-based controller for interactive evolution choices."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def prompt_selection(
        self,
        db: ProgramDatabase,
        generation: int,
        allowed_patch_types: Sequence[str],
    ) -> InteractiveSelection:
        """Prompt the user for parent, patch type, and inspirations.

        Args:
            db: Program database for listing candidate programs.
            generation: Target generation to create next.
            allowed_patch_types: Patch types allowed by config.

        Returns:
            InteractiveSelection with user choices or auto flags.
        """
        selection = InteractiveSelection()
        self._output("\n=== Interactive Selection ===")
        self._output(f"Target generation: {generation}")

        prev_gen = max(generation - 1, 0)
        prev_summary = db.get_program_summaries(generation=prev_gen, limit=10)
        if prev_summary:
            self._output(f"\nLast generation ({prev_gen}) summary (top 10):")
            for row in prev_summary:
                self._output(
                    "  - "
                    f"id={row['id']} score={row['combined_score']:.4f} "
                    f"correct={row['correct']} island={row['island_idx']}"
                )

        top_summary = db.get_program_summaries(limit=10, correct_only=True)
        if top_summary:
            self._output("\nTop programs (correct, top 10):")
            for row in top_summary:
                self._output(
                    "  - "
                    f"id={row['id']} score={row['combined_score']:.4f} "
                    f"gen={row['generation']} island={row['island_idx']}"
                )

        # Parent selection
        parent_choice = self._prompt(
            "\nParent program (enter program id or 'auto'): "
        )
        if parent_choice.lower() != "auto":
            selection.use_auto_parent = False
            selection.parent_id = parent_choice

        # Patch type selection
        patch_type_prompt = (
            "\nPatch type (diff/full/cross or 'auto') "
            f"[allowed={','.join(allowed_patch_types)}]: "
        )
        patch_choice = self._prompt(patch_type_prompt)
        if patch_choice.lower() != "auto":
            selection.use_auto_patch_type = False
            selection.patch_type = patch_choice

        # Inspirations selection
        insp_choice = self._prompt(
            "\nInspirations (enter 'auto', 'none', or comma-separated ids): "
        )
        if insp_choice.lower() == "none":
            selection.use_auto_inspirations = False
            selection.archive_inspiration_ids = []
            selection.top_k_inspiration_ids = []
        elif insp_choice.lower() != "auto":
            selection.use_auto_inspirations = False
            ids = [item.strip() for item in insp_choice.split(",") if item.strip()]
            selection.archive_inspiration_ids = ids
            selection.top_k_inspiration_ids = []

        suggestions = self._prompt_optional(
            "\nOptional suggestions/ideas to guide the edit (press Enter to skip): "
        )
        if suggestions:
            selection.user_suggestions = suggestions

        return selection

    def _prompt(self, message: str) -> str:
        """Prompt the user until a non-empty response is provided."""
        while True:
            response = self._input(message).strip()
            if response:
                return response
            self._output("Please enter a value.")

    def _prompt_optional(self, message: str) -> Optional[str]:
        """Prompt the user once, allowing empty input."""
        response = self._input(message).strip()
        return response or None

    @staticmethod
    def resolve_programs(
        db: ProgramDatabase, program_ids: Sequence[str]
    ) -> List[Program]:
        """Resolve program IDs into Program objects.

        Args:
            db: Program database for lookup.
            program_ids: Program IDs to resolve.

        Returns:
            List of Program objects (missing IDs are skipped).
        """
        return db.get_programs_by_ids(list(program_ids))
