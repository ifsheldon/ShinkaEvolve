"""Integration tests for the evolve-shell /api/callback endpoint.

These tests verify that the backend correctly receives runner lifecycle
events and broadcasts them to connected WebSocket clients.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The backend lives in evolve-shell/backend/main.py.  Add its parent to
# sys.path so that ``import main`` resolves.
_BACKEND_DIR = (
    Path(__file__).resolve().parents[2]  # ShinkaEvolve/
    / ".."
    / "evolve-shell"
    / "backend"
)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture(autouse=True)
def _set_search_root(tmp_path, monkeypatch):
    """Point the backend at a temp directory so it doesn't touch real DBs."""
    monkeypatch.setenv("SHINKA_SEARCH_ROOT", str(tmp_path))


@pytest.fixture
def client():
    """Provide a FastAPI TestClient for the evolve-shell backend."""
    # Re-import with a fresh SEARCH_ROOT each time.
    import importlib

    import main as backend_main

    importlib.reload(backend_main)

    from fastapi.testclient import TestClient

    return TestClient(backend_main.app)


@pytest.fixture
def ws_manager():
    """Return the live ConnectionManager from the backend module."""
    import main as backend_main

    return backend_main.ws_manager


@pytest.fixture
def program_count_tracker():
    """Return the module-level _last_program_count dict."""
    import main as backend_main

    return backend_main._last_program_count


@pytest.fixture
def db_cache_ref():
    """Return the module-level db_cache dict."""
    import main as backend_main

    return backend_main.db_cache


# ---------------------------------------------------------------------------
# Basic callback acceptance
# ---------------------------------------------------------------------------


class TestCallbackEndpoint:
    def test_queued_event_accepted(self, client):
        resp = client.post(
            "/api/callback",
            json={
                "event": "program.queued",
                "db_path": "results/shinka.db",
                "program_id": "queued-001",
                "parent_id": "parent-001",
                "generation": 3,
                "code": "def solve(): pass",
                "timestamp": 1710300000.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("broadcast", "no_subscribers")

    def test_generated_event_accepted(self, client):
        resp = client.post(
            "/api/callback",
            json={
                "event": "program.generated",
                "db_path": "results/shinka.db",
                "program": {
                    "id": "gen-001",
                    "code": "def solve(): return 42",
                    "generation": 3,
                    "combined_score": 0.85,
                    "parent_id": "parent-001",
                    "timestamp": 1710300001.0,
                },
            },
        )
        assert resp.status_code == 200

    def test_unknown_event_ignored(self, client):
        resp = client.post(
            "/api/callback",
            json={"event": "unknown.event", "db_path": "test.db"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_no_subscribers_returns_status(self, client):
        resp = client.post(
            "/api/callback",
            json={
                "event": "program.queued",
                "db_path": "nonexistent/path.db",
                "program_id": "id-1",
                "generation": 1,
                "code": "pass",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_subscribers"


# ---------------------------------------------------------------------------
# Cache invalidation on program.generated
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_generated_invalidates_db_cache(self, client, db_cache_ref):
        # Pre-populate cache with a matching key
        db_cache_ref["results/shinka.db"] = (0.0, [{"id": "old"}])

        client.post(
            "/api/callback",
            json={
                "event": "program.generated",
                "db_path": "results/shinka.db",
                "program": {
                    "id": "gen-001",
                    "code": "pass",
                    "generation": 1,
                },
            },
        )

        assert "results/shinka.db" not in db_cache_ref

    def test_generated_increments_program_count_when_subscribed(
        self, client, program_count_tracker
    ):
        """Program count tracker is only incremented when there are WS
        subscribers (to prevent fallback polling from re-broadcasting)."""
        db_path = "test_count.db"
        program_count_tracker[db_path] = 10

        with client.websocket_connect(f"/ws/{db_path}"):
            client.post(
                "/api/callback",
                json={
                    "event": "program.generated",
                    "db_path": db_path,
                    "program": {
                        "id": "gen-002",
                        "code": "pass",
                        "generation": 2,
                    },
                },
            )

        assert program_count_tracker[db_path] == 11

    def test_queued_does_not_invalidate_cache(self, client, db_cache_ref):
        db_cache_ref["results/shinka.db"] = (0.0, [{"id": "existing"}])

        client.post(
            "/api/callback",
            json={
                "event": "program.queued",
                "db_path": "results/shinka.db",
                "program_id": "q-1",
                "generation": 5,
                "code": "pass",
            },
        )

        # Cache should NOT be invalidated for queued events
        assert "results/shinka.db" in db_cache_ref


# ---------------------------------------------------------------------------
# WebSocket broadcast via callback
# ---------------------------------------------------------------------------


class TestWebSocketBroadcast:
    def test_queued_broadcast_to_subscriber(self, client):
        """Connect a WS client, then POST a queued event — the client
        should receive a program.queued message."""
        # Use a simple db_path that doesn't need file existence
        db_path = "test_ws.db"

        with client.websocket_connect(f"/ws/{db_path}") as ws:
            # POST the callback
            resp = client.post(
                "/api/callback",
                json={
                    "event": "program.queued",
                    "db_path": db_path,
                    "program_id": "ws-queued-001",
                    "parent_id": "ws-parent",
                    "generation": 7,
                    "code": "def run(): ...",
                    "code_diff": "@@ -1 +1 @@",
                    "timestamp": 1710300000.0,
                    "metadata": {"patch_type": "diff"},
                },
            )
            assert resp.json()["status"] == "broadcast"

            # Read the message from the WS
            msg = ws.receive_json()
            assert msg["type"] == "program.queued"
            assert msg["program_id"] == "ws-queued-001"
            assert msg["parent_id"] == "ws-parent"
            assert msg["generation"] == 7
            assert msg["code"] == "def run(): ..."
            assert msg["metadata"]["patch_type"] == "diff"

    def test_generated_broadcast_to_subscriber(self, client):
        db_path = "test_ws.db"

        with client.websocket_connect(f"/ws/{db_path}") as ws:
            program_data = {
                "id": "ws-gen-001",
                "code": "def solve(): return 42",
                "generation": 3,
                "combined_score": 0.95,
                "parent_id": "ws-parent",
                "timestamp": 1710300001.0,
            }
            resp = client.post(
                "/api/callback",
                json={
                    "event": "program.generated",
                    "db_path": db_path,
                    "program": program_data,
                },
            )
            assert resp.json()["status"] == "broadcast"

            msg = ws.receive_json()
            assert msg["type"] == "program.generated"
            assert msg["program"]["id"] == "ws-gen-001"
            assert msg["program"]["combined_score"] == 0.95

    def test_broadcast_with_task_prefix_db_path(self, client, tmp_path):
        """The runner sends a bare db_path but the WS client subscribes
        with a task-name prefix.  The callback should still match."""
        task_name = os.path.basename(str(tmp_path))
        bare_path = "results/shinka.db"
        prefixed_path = f"{task_name}/{bare_path}"

        with client.websocket_connect(f"/ws/{prefixed_path}") as ws:
            resp = client.post(
                "/api/callback",
                json={
                    "event": "program.queued",
                    "db_path": bare_path,
                    "program_id": "prefix-001",
                    "generation": 1,
                    "code": "pass",
                },
            )
            assert resp.json()["status"] == "broadcast"

            msg = ws.receive_json()
            assert msg["type"] == "program.queued"
            assert msg["program_id"] == "prefix-001"
