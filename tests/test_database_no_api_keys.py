import asyncio
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase


def _program(program_id: str) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=0,
    )


def test_program_database_init_without_openai_key(monkeypatch):
    """DB construction should not require API credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "no_key_init.db"
        db = ProgramDatabase(config=DatabaseConfig(db_path=str(db_path), num_islands=1))
        try:
            db.add(_program("p0"))
            assert db.get("p0") is not None
        finally:
            db.close()


def test_async_db_add_without_openai_key_when_embeddings_disabled(monkeypatch):
    """Async wrapper should preserve disabled embedding mode in worker DBs."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "no_key_async.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                await async_db.add_program_async(_program("async-p0"))
                assert sync_db.get("async-p0") is not None
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_add_maps_write_fields_to_typed_program(monkeypatch):
    """Async DB writes should persist supplemental fields via real Program attributes."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "typed_async_write.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                program = Program(
                    id="async-typed-p0",
                    code="def f():\n    return 1\n",
                    correct=True,
                    combined_score=1.0,
                    generation=1,
                    metadata={"existing": "value"},
                )

                await async_db.add_program_async(
                    program=program,
                    parent_id="parent-1",
                    archive_insp_ids=["archive-1"],
                    top_k_insp_ids=["topk-1"],
                    code_diff="@@ -1 +1 @@",
                    meta_patch_data={"patch_type": "diff", "api_costs": 1.25},
                    code_embedding=[0.1, 0.2],
                    embed_cost=0.5,
                )

                stored = sync_db.get(program.id)
                assert stored is not None
                assert stored.parent_id == "parent-1"
                assert stored.archive_inspiration_ids == ["archive-1"]
                assert stored.top_k_inspiration_ids == ["topk-1"]
                assert stored.code_diff == "@@ -1 +1 @@"
                assert stored.embedding == [0.1, 0.2]
                assert stored.metadata["existing"] == "value"
                assert stored.metadata["patch_type"] == "diff"
                assert stored.metadata["api_costs"] == 1.25
                assert stored.metadata["embed_cost"] == 0.5

                assert program.embedding == [0.1, 0.2]
                assert program.metadata["embed_cost"] == 0.5
                assert not hasattr(program, "code_embedding")
                assert not hasattr(program, "meta_patch_data")
                assert not hasattr(program, "embed_cost")
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())


def test_async_db_named_row_access_helpers(monkeypatch):
    """Async DB helper queries should work via named sqlite rows."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "named_rows_async.db"
            sync_db = ProgramDatabase(
                config=DatabaseConfig(db_path=str(db_path), num_islands=1),
                embedding_model="",
            )
            async_db = AsyncProgramDatabase(sync_db=sync_db)
            try:
                sync_db.add(
                    Program(
                        id="p-low",
                        code="def f():\n    return 1\n",
                        generation=0,
                        correct=True,
                        combined_score=1.0,
                    )
                )
                sync_db.add(
                    Program(
                        id="p-high",
                        code="def g():\n    return 2\n",
                        generation=1,
                        correct=True,
                        combined_score=3.0,
                    )
                )
                sync_db.add(
                    Program(
                        id="p-wrong",
                        code="def h():\n    return 0\n",
                        generation=1,
                        correct=False,
                        combined_score=10.0,
                    )
                )

                assert await async_db.get_total_program_count_async() == 3
                assert (
                    await async_db.compute_percentile_async(2.0, correct_only=True)
                    == 0.5
                )
                assert await async_db.compute_percentile_async(
                    5.0, correct_only=False
                ) == (2 / 3)
            finally:
                await async_db.close_async()
                sync_db.close()

    asyncio.run(_run())
