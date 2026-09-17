"""Exercise sandbox summary routing without leaking its global client patches."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_mock_summary_pipeline_is_offline_and_does_not_generate_patches(
    tmp_path: Path,
) -> None:
    """Use real summary prompts and both client paths in an isolated process."""
    script = textwrap.dedent(
        """
        import asyncio
        from pathlib import Path
        import runpy
        import sys

        def block_network(event, args):
            if event in {"socket.connect", "socket.getaddrinfo"}:
                raise AssertionError("Sandbox tests must not access the network")

        sys.addaudithook(block_network)
        runpy.run_path("examples/interactive_sandbox/run_evo.py")

        from shinka.core.async_summarizer import AsyncMetaSummarizer
        from shinka.core.summarizer import MetaSummarizer
        from shinka.database import Program
        from shinka.llm import AsyncLLMClient, LLMClient
        from shinka.prompts import (
            META_STEP1_SYSTEM_MSG, META_STEP2_SYSTEM_MSG, META_STEP3_SYSTEM_MSG,
        )

        client = LLMClient(model_names=["mock-llm"])
        for prompt, expected in (
            (META_STEP1_SYSTEM_MSG, "**Summary:**"),
            (META_STEP2_SYSTEM_MSG, "**Illustrative mock summary.**"),
            (META_STEP3_SYSTEM_MSG, "1. Increase the loop count"),
        ):
            response = client.query(msg="Analyze this program.", system_msg=prompt)
            assert expected in response.content
            assert "<NAME>" not in response.content
            assert "```python" not in response.content
            assert response.cost == 0.0

        # A generation prompt can mention insights without becoming a summary call.
        response = client.query(
            msg="Improve the program.",
            system_msg="Use global insights and actionable recommendations to evolve code.",
        )
        assert "<NAME>variant_" in response.content
        assert "```python" in response.content

        async def check_pipeline():
            state = MetaSummarizer(async_mode=True)
            state.add_evaluated_program(Program(
                id="example", code="def run_experiment(): return 1",
                language="python", generation=1, correct=True,
                combined_score=1.0, metadata={"patch_name": "example"},
            ))
            summarizer = AsyncMetaSummarizer(
                state, AsyncLLMClient(model_names=["mock-llm"]),
            )
            recommendations, cost = await summarizer.update_meta_memory_async()
            assert recommendations.startswith("1. Increase the loop count")
            assert cost == 0.0
            assert "**Summary:**" in state.meta_summary
            assert "Illustrative mock summary" in state.meta_scratch_pad
            assert "Scores are random" in state.meta_scratch_pad
            assert "<NAME>" not in state.meta_scratch_pad
            await summarizer.write_meta_output_async(sys.argv[1])
            output = (Path(sys.argv[1]) / "meta/meta_1.txt").read_text()
            assert "# GLOBAL INSIGHTS SCRATCHPAD" in output
            assert "# META RECOMMENDATIONS" in output
            assert "Illustrative mock summary" in output
            assert "```python" not in output

        asyncio.run(check_pipeline())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "SHINKA_PRICING_MODE": "offline"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
