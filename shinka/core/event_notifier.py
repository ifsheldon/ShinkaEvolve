"""Fire-and-forget HTTP callbacks for program lifecycle events.

When ``callback_url`` is configured, the runner POSTs lightweight JSON
payloads to the evolve-shell backend so it can push real-time updates
to connected frontends via WebSocket — replacing the SQLite polling loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

if TYPE_CHECKING:
    from shinka.database import Program

logger = logging.getLogger(__name__)

# Shared client — reused across the runner lifetime to benefit from
# connection pooling.  Created lazily on first use.
_client: Optional[httpx.AsyncClient] = None

_TIMEOUT = httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=3.0)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


class EventNotifier:
    """Async, fire-and-forget notifier that POSTs lifecycle events.

    If *callback_url* is ``None`` every method is a silent no-op so
    callers don't need any conditional logic.
    """

    def __init__(self, callback_url: Optional[str], db_path: str) -> None:
        self.callback_url = callback_url.rstrip("/") if callback_url else None
        self.db_path = db_path
        if self.callback_url:
            logger.info("EventNotifier: push callbacks → %s", self.callback_url)
        else:
            logger.info("EventNotifier: no callback_url — push updates disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def notify_queued(
        self,
        *,
        program_id: str,
        parent_id: Optional[str],
        generation: int,
        code: str,
        code_diff: Optional[str] = None,
        island_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        archive_inspiration_ids: Optional[list] = None,
        top_k_inspiration_ids: Optional[list] = None,
    ) -> None:
        """Notify that a program has been queued for evaluation."""
        if self.callback_url is None:
            return
        payload = {
            "event": "program.queued",
            "db_path": self.db_path,
            "program_id": program_id,
            "parent_id": parent_id,
            "generation": generation,
            "code": code,
            "code_diff": code_diff,
            "timestamp": time.time(),
            "island_idx": island_idx,
            "metadata": metadata or {},
            "archive_inspiration_ids": archive_inspiration_ids or [],
            "top_k_inspiration_ids": top_k_inspiration_ids or [],
        }
        asyncio.create_task(self._post(payload))

    async def notify_generated(self, program: "Program") -> None:
        """Notify that a program has completed evaluation and is in the DB."""
        if self.callback_url is None:
            return
        payload = {
            "event": "program.generated",
            "db_path": self.db_path,
            "program": program.to_dict(),
        }
        asyncio.create_task(self._post(payload))

    async def close(self) -> None:
        """Shut down the shared HTTP client (call at runner teardown)."""
        global _client
        if _client is not None and not _client.is_closed:
            await _client.aclose()
            _client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post(self, payload: Dict[str, Any]) -> None:
        """POST *payload* to the callback endpoint.  Never raises."""
        url = f"{self.callback_url}/api/callback"
        try:
            client = await _get_client()
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    "Callback POST %s returned %d: %s",
                    url,
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            # Fire-and-forget: log and move on.
            logger.warning("Callback POST to %s failed: %s", url, exc)
