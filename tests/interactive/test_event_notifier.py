"""Tests for EventNotifier fire-and-forget HTTP callback client."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from shinka.core.event_notifier import EventNotifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeProgram:
    """Minimal stand-in for shinka.database.Program."""

    id: str = "prog-001"
    code: str = "print('hello')"
    generation: int = 1
    parent_id: Optional[str] = "prog-000"
    combined_score: float = 0.75
    correct: bool = True
    timestamp: float = 1710300000.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _RequestLog:
    """Collects POST requests made via the mock client."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def mock_post(self, status_code: int = 200):
        """Return an AsyncMock whose side_effect records each call."""
        resp = AsyncMock()
        resp.status_code = status_code
        resp.text = "ok"

        async def _post(url: str, json: Any = None):
            self.requests.append({"url": url, "json": json})
            return resp

        return _post


# ---------------------------------------------------------------------------
# No-op when callback_url is None
# ---------------------------------------------------------------------------


class TestNoopWithoutCallbackUrl:
    @pytest.mark.asyncio
    async def test_notify_queued_is_noop(self):
        notifier = EventNotifier(callback_url=None, db_path="test.db")
        # Should return immediately without error
        await notifier.notify_queued(
            program_id="id-1",
            parent_id=None,
            generation=1,
            code="pass",
        )

    @pytest.mark.asyncio
    async def test_notify_generated_is_noop(self):
        notifier = EventNotifier(callback_url=None, db_path="test.db")
        await notifier.notify_generated(_FakeProgram())


# ---------------------------------------------------------------------------
# Payload correctness
# ---------------------------------------------------------------------------


class TestQueuedPayload:
    @pytest.mark.asyncio
    async def test_payload_shape(self):
        log = _RequestLog()
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="my/shinka.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=log.mock_post())
            mock_get.return_value = client

            await notifier.notify_queued(
                program_id="queued-uuid",
                parent_id="parent-uuid",
                generation=5,
                code="def solve(): ...",
                code_diff="@@ -1 +1 @@",
                island_idx=2,
                metadata={"patch_type": "diff", "model_name": "gpt-5"},
                archive_inspiration_ids=["a1", "a2"],
                top_k_inspiration_ids=["t1"],
            )

            # Fire-and-forget task needs a tick to execute
            await asyncio.sleep(0.05)

        assert len(log.requests) == 1
        req = log.requests[0]
        assert req["url"] == "http://localhost:8000/api/callback"

        payload = req["json"]
        assert payload["event"] == "program.queued"
        assert payload["db_path"] == "my/shinka.db"
        assert payload["program_id"] == "queued-uuid"
        assert payload["parent_id"] == "parent-uuid"
        assert payload["generation"] == 5
        assert payload["code"] == "def solve(): ..."
        assert payload["code_diff"] == "@@ -1 +1 @@"
        assert payload["island_idx"] == 2
        assert payload["metadata"]["patch_type"] == "diff"
        assert payload["archive_inspiration_ids"] == ["a1", "a2"]
        assert payload["top_k_inspiration_ids"] == ["t1"]
        assert isinstance(payload["timestamp"], float)

    @pytest.mark.asyncio
    async def test_defaults_for_optional_fields(self):
        log = _RequestLog()
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="test.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=log.mock_post())
            mock_get.return_value = client

            await notifier.notify_queued(
                program_id="id-1",
                parent_id=None,
                generation=0,
                code="pass",
            )
            await asyncio.sleep(0.05)

        payload = log.requests[0]["json"]
        assert payload["parent_id"] is None
        assert payload["code_diff"] is None
        assert payload["island_idx"] is None
        assert payload["metadata"] == {}
        assert payload["archive_inspiration_ids"] == []
        assert payload["top_k_inspiration_ids"] == []


class TestGeneratedPayload:
    @pytest.mark.asyncio
    async def test_payload_contains_full_program(self):
        log = _RequestLog()
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="results/shinka.db"
        )

        program = _FakeProgram(
            id="gen-uuid",
            code="def solve(): return 42",
            generation=3,
            combined_score=0.95,
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=log.mock_post())
            mock_get.return_value = client

            await notifier.notify_generated(program)
            await asyncio.sleep(0.05)

        assert len(log.requests) == 1
        payload = log.requests[0]["json"]
        assert payload["event"] == "program.generated"
        assert payload["db_path"] == "results/shinka.db"
        assert payload["program"]["id"] == "gen-uuid"
        assert payload["program"]["combined_score"] == 0.95
        assert payload["program"]["code"] == "def solve(): return 42"


# ---------------------------------------------------------------------------
# Error resilience (fire-and-forget must never raise)
# ---------------------------------------------------------------------------


class TestErrorResilience:
    @pytest.mark.asyncio
    async def test_http_error_does_not_raise(self):
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="test.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            resp = AsyncMock()
            resp.status_code = 500
            resp.text = "Internal Server Error"
            client.post = AsyncMock(return_value=resp)
            mock_get.return_value = client

            # Should not raise
            await notifier.notify_queued(
                program_id="id-1", parent_id=None, generation=1, code="pass"
            )
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_connection_error_does_not_raise(self):
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="test.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=ConnectionError("refused"))
            mock_get.return_value = client

            await notifier.notify_queued(
                program_id="id-1", parent_id=None, generation=1, code="pass"
            )
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_generated_connection_error_does_not_raise(self):
        notifier = EventNotifier(
            callback_url="http://localhost:8000", db_path="test.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=TimeoutError("timed out"))
            mock_get.return_value = client

            await notifier.notify_generated(_FakeProgram())
            await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


class TestUrlNormalization:
    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        log = _RequestLog()
        notifier = EventNotifier(
            callback_url="http://localhost:8000/", db_path="test.db"
        )

        with patch(
            "shinka.core.event_notifier._get_client", new_callable=AsyncMock
        ) as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=log.mock_post())
            mock_get.return_value = client

            await notifier.notify_queued(
                program_id="id-1", parent_id=None, generation=1, code="pass"
            )
            await asyncio.sleep(0.05)

        assert log.requests[0]["url"] == "http://localhost:8000/api/callback"


@pytest.mark.asyncio
async def test_private_callback_credential_is_only_an_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public-demo callbacks authenticate without putting credentials in payloads or URLs."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("EVOMAESTRO_CALLBACK_TOKEN", token)
    notifier = EventNotifier("http://127.0.0.1:8000", "run/programs.sqlite")
    client = AsyncMock()
    client.post.return_value.status_code = 200
    with patch("shinka.core.event_notifier._get_client", return_value=client):
        await notifier._post({"event": "program.generated"})
    args, kwargs = client.post.call_args
    assert args == ("http://127.0.0.1:8000/api/callback",)
    assert kwargs["json"] == {"event": "program.generated"}
    assert kwargs["headers"] == {"Authorization": f"Bearer {token}"}
    assert token not in repr(notifier._callback_token)


@pytest.mark.parametrize(
    "url",
    [
        "https://external.example",
        "http://localhost.example",
        "http://user:pass@localhost",
    ],
)
def test_private_callback_credential_cannot_be_sent_to_remote_hosts(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """A private runner credential is only valid for the local callback service."""
    monkeypatch.setenv("EVOMAESTRO_CALLBACK_TOKEN", secrets.token_urlsafe(32))
    with pytest.raises(ValueError, match="loopback"):
        EventNotifier(url, "run/programs.sqlite")


@pytest.mark.asyncio
async def test_callback_client_ignores_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loopback bearer request cannot be forwarded through an environment proxy."""
    from shinka.core import event_notifier

    monkeypatch.setenv("HTTP_PROXY", "http://external.example:8080")
    monkeypatch.setenv("ALL_PROXY", "http://external.example:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(event_notifier, "_client", None)
    client = await event_notifier._get_client()
    try:
        assert client.trust_env is False
    finally:
        await client.aclose()
