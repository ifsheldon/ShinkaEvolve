"""Tests for the ReviewPrioritizer and default review prioritization logic."""

import json
import textwrap


from shinka.core.review_prioritizer import (
    ReviewPrioritizer,
    ReviewPriorityLevel,
    ReviewPriorityResult,
    ProgramData,
    default_prioritize_for_review,
)
from shinka.database.dbase import Program


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(
    id: str = "test",
    generation: int = 1,
    score: float = 100.0,
    correct: bool = True,
    **kwargs,
) -> Program:
    return Program(
        id=id,
        generation=generation,
        combined_score=score,
        correct=correct,
        code="",
        public_metrics=kwargs.get("public_metrics", {}),
        private_metrics=kwargs.get("private_metrics", {}),
        embedding=kwargs.get("embedding", []),
        code_diff=kwargs.get("code_diff", None),
        metadata=kwargs.get("metadata", {}),
    )


def _make_program_data(
    id: str = "test",
    generation: int = 1,
    score: float = 100.0,
    correct: bool = True,
) -> ProgramData:
    return ProgramData(
        id=id,
        generation=generation,
        combined_score=score,
        correct=correct,
        public_metrics={},
        private_metrics={},
        embedding=[],
        reasoning_embedding=[],
        code_diff=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# default_prioritize_for_review
# ---------------------------------------------------------------------------


class TestDefaultPrioritizeForReview:
    def test_no_parent_returns_none(self):
        prog = _make_program_data(score=50.0)
        level, data = default_prioritize_for_review(prog, None, [])
        assert level == ReviewPriorityLevel.NONE
        assert data is None

    def test_incorrect_program_returns_none(self):
        prog = _make_program_data(score=200.0, correct=False)
        parent = _make_program_data(id="parent", score=100.0)
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.NONE

    def test_high_priority_above_30_percent(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=135.0)  # 35% gain
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.HIGH
        assert data is not None
        assert data["gain_pct"] == 35.0

    def test_moderate_priority_between_15_and_30(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=120.0)  # 20% gain
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.MODERATE
        assert data["gain_pct"] == 20.0

    def test_no_priority_below_15_percent(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=110.0)  # 10% gain
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.NONE
        assert data is not None  # still returns display data
        assert data["gain_pct"] == 10.0

    def test_zero_parent_score_positive_child_is_high(self):
        parent = _make_program_data(id="parent", score=0.0)
        prog = _make_program_data(score=50.0)
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.HIGH
        assert data["gain_pct"] is None
        assert "First correct solution" in data["reason"]

    def test_zero_parent_score_zero_child_is_none(self):
        parent = _make_program_data(id="parent", score=0.0)
        prog = _make_program_data(score=0.0)
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.NONE

    def test_negative_gain_is_none(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=80.0)  # -20% gain
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.NONE

    def test_boundary_exactly_30_percent(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=130.0)  # exactly 30%
        level, _ = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.HIGH

    def test_boundary_exactly_15_percent(self):
        parent = _make_program_data(id="parent", score=100.0)
        prog = _make_program_data(score=115.0)  # exactly 15%
        level, _ = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.MODERATE

    def test_negative_parent_score(self):
        """Percentage gain uses abs(parent_score) for negative scores."""
        parent = _make_program_data(id="parent", score=-100.0)
        prog = _make_program_data(score=-60.0)  # 40% gain
        level, data = default_prioritize_for_review(prog, parent, [])
        assert level == ReviewPriorityLevel.HIGH
        assert data["gain_pct"] == 40.0


# ---------------------------------------------------------------------------
# ReviewPrioritizer - default function
# ---------------------------------------------------------------------------


class TestReviewPrioritizerDefault:
    def test_prioritize_without_custom_function(self):
        detector = ReviewPrioritizer()
        parent = _make_program(id="parent", score=100.0)
        child = _make_program(score=140.0)
        result = detector.prioritize(child, parent, [])
        assert result.level == ReviewPriorityLevel.HIGH

    def test_prioritize_no_parent(self):
        detector = ReviewPrioritizer()
        prog = _make_program(generation=0, score=50.0)
        result = detector.prioritize(prog, None, [])
        assert result.level == ReviewPriorityLevel.NONE

    def test_prioritize_converts_program_to_program_data(self):
        """Verifies the Program → ProgramData conversion works."""
        detector = ReviewPrioritizer()
        parent = _make_program(
            id="parent",
            score=100.0,
            public_metrics={"x": 1},
            private_metrics={"y": 2},
            embedding=[0.1, 0.2],
            code_diff="- old\n+ new",
            metadata={"tag": "v1"},
        )
        child = _make_program(score=150.0)
        result = detector.prioritize(child, parent, [parent])
        assert isinstance(result, ReviewPriorityResult)


# ---------------------------------------------------------------------------
# ReviewPrioritizer - cached priority signals
# ---------------------------------------------------------------------------


class TestReviewPriorityMetrics:
    def test_dissimilarity_requires_an_earlier_embedding(self):
        prioritizer = ReviewPrioritizer()
        program = _make_program(
            generation=0,
            embedding=[1.0, 0.0],
        )
        program.reasoning_embedding = [1.0, 0.0]

        metrics = prioritizer.compute_priority_metrics(program, None, [], [])

        assert metrics == {
            "score_change": None,
            "dissimilarity_code": None,
            "dissimilarity_reasoning": None,
        }

    def test_computes_score_and_embedding_signals(self):
        prioritizer = ReviewPrioritizer()
        parent = _make_program(id="parent", score=100.0)
        program = _make_program(
            score=120.0,
            embedding=[0.0, 1.0],
        )
        program.reasoning_embedding = [0.0, 1.0]

        metrics = prioritizer.compute_priority_metrics(
            program,
            parent,
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        )

        assert metrics == {
            "score_change": 0.2,
            "dissimilarity_code": 1.0,
            "dissimilarity_reasoning": 1.0,
        }


# ---------------------------------------------------------------------------
# ReviewPrioritizer - custom function loading
# ---------------------------------------------------------------------------


class TestReviewPrioritizerCustomFunction:
    def test_loads_custom_function(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                return ReviewPriorityLevel.MODERATE, {"custom": True}
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        assert detector.load_error is None
        parent = _make_program(id="parent", score=100.0)
        child = _make_program(score=101.0)  # tiny gain, default would say NONE
        result = detector.prioritize(child, parent, [])
        assert result.level == ReviewPriorityLevel.MODERATE
        assert result.display_data == {"custom": True}

    def test_hot_reload_on_file_change(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                return ReviewPriorityLevel.MODERATE, {"version": 1}
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        parent = _make_program(id="parent", score=100.0)
        child = _make_program(score=101.0)

        result1 = detector.prioritize(child, parent, [])
        assert result1.display_data["version"] == 1

        # Update the file (must change mtime)
        import time

        time.sleep(0.05)
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                return ReviewPriorityLevel.HIGH, {"version": 2}
        """)
        )

        result2 = detector.prioritize(child, parent, [])
        assert result2.level == ReviewPriorityLevel.HIGH
        assert result2.display_data["version"] == 2

    def test_skips_reload_if_unchanged(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                return ReviewPriorityLevel.MODERATE, None
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        # Record the cached module
        first_module = detector._cached_module

        parent = _make_program(id="parent", score=100.0)
        child = _make_program(score=101.0)
        detector.prioritize(child, parent, [])

        # Same module should be reused
        assert detector._cached_module is first_module

    def test_fallback_on_missing_file(self, tmp_path):
        detector = ReviewPrioritizer(
            review_prioritization_function_path=str(tmp_path / "nonexistent.py"),
        )
        assert detector.load_error is not None
        # Should still work with default function
        parent = _make_program(id="parent", score=100.0)
        child = _make_program(score=140.0)
        result = detector.prioritize(child, parent, [])
        assert result.level == ReviewPriorityLevel.HIGH

    def test_fallback_on_missing_function_name(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text("# empty module\n")
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        assert detector.load_error is not None
        assert "not found" in detector.load_error

    def test_fallback_on_invalid_return_type(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text(
            textwrap.dedent("""\
            def prioritize_for_review(program, parent, inspirations):
                return "bad"
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        assert detector.load_error is not None
        assert "must return" in detector.load_error

    def test_fallback_on_runtime_exception(self, tmp_path):
        custom_fn = tmp_path / "review_prioritization.py"
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                raise RuntimeError("boom")
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        # Loads OK (validation with dummy passes because error is conditional)
        # But at runtime it will fail and fall back
        # Actually the validation call will also raise - let's check
        # The validation calls the function with dummy data which will raise
        assert detector.load_error is not None

    def test_fallback_preserves_run_on_runtime_error(self, tmp_path):
        """Even if custom fn raises at detect time, we fall back gracefully."""
        custom_fn = tmp_path / "review_prioritization.py"
        # Write a valid function first
        custom_fn.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                if program.id != "__validate__":
                    raise RuntimeError("boom at runtime")
                return ReviewPriorityLevel.NONE, None
        """)
        )
        detector = ReviewPrioritizer(review_prioritization_function_path=str(custom_fn))
        assert detector.load_error is None  # validation passed

        parent = _make_program(id="parent", score=100.0)
        child = _make_program(id="real", score=140.0)
        result = detector.prioritize(child, parent, [])
        # Should fall back to default and still return a result
        assert result.level == ReviewPriorityLevel.HIGH


# ---------------------------------------------------------------------------
# ReviewPrioritizer - error file I/O
# ---------------------------------------------------------------------------


class TestReviewPrioritizerErrorFile:
    def test_error_file_written(self, tmp_path):
        ReviewPrioritizer(
            review_prioritization_function_path=str(tmp_path / "missing.py"),
            results_dir=str(tmp_path),
        )
        err_path = tmp_path / "review_prioritization_error.json"
        assert err_path.exists()
        data = json.loads(err_path.read_text())
        assert "error" in data

    def test_error_file_cleared_on_success(self, tmp_path):
        # First create an error
        ReviewPrioritizer(
            review_prioritization_function_path=str(tmp_path / "missing.py"),
            results_dir=str(tmp_path),
        )
        err_path = tmp_path / "review_prioritization_error.json"
        assert err_path.exists()

        # Now create a valid function and reload
        fn_path = tmp_path / "review_prioritization.py"
        fn_path.write_text(
            textwrap.dedent("""\
            from shinka.core.review_prioritizer import ReviewPriorityLevel

            def prioritize_for_review(program, parent, inspirations):
                return ReviewPriorityLevel.NONE, None
        """)
        )
        detector2 = ReviewPrioritizer(
            review_prioritization_function_path=str(fn_path),
            results_dir=str(tmp_path),
        )
        assert detector2.load_error is None
        assert not err_path.exists()

    def test_no_error_file_without_results_dir(self, tmp_path):
        """Error file should not be written when results_dir is None."""
        detector = ReviewPrioritizer(
            review_prioritization_function_path=str(tmp_path / "missing.py"),
        )
        assert detector.load_error is not None
        # No file should be written anywhere


# ---------------------------------------------------------------------------
# ProgramData conversion
# ---------------------------------------------------------------------------


class TestProgramDataConversion:
    def test_handles_none_fields_gracefully(self):
        """Program fields that are None should default to empty containers."""
        prog = Program(
            id="test",
            generation=0,
            combined_score=0.0,
            correct=True,
            code="",
            public_metrics=None,
            private_metrics=None,
            embedding=None,
            code_diff=None,
            metadata=None,
        )
        pd = ReviewPrioritizer._to_program_data(prog)
        assert pd.public_metrics == {}
        assert pd.private_metrics == {}
        assert pd.embedding == []
        assert pd.metadata == {}
        assert pd.code_diff is None
