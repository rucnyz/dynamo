# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async HTTP client for a remote standalone ``sglang.launch_server``."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class SglangHttpClient:
    """Thin httpx wrapper around the SGLang HTTP surface we need.

    Endpoints used:
      * ``POST /generate`` (SSE when ``stream=true``)
      * ``POST /aginfer/session_end``
      * ``PUT  /aginfer/events``
      * ``GET  /server_info``
      * ``POST /abort_request``
    """

    def __init__(
        self,
        base_url: str,
        *,
        generate_timeout_s: float = 3600.0,
        control_timeout_s: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generate_timeout_s = generate_timeout_s
        self.control_timeout_s = control_timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                control_timeout_s,
                read=generate_timeout_s,
                write=control_timeout_s,
                connect=control_timeout_s,
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_server_info(self) -> Dict[str, Any]:
        resp = await self._client.get(
            "/server_info", timeout=self.control_timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(
                f"/server_info returned non-object JSON: {type(data).__name__}"
            )
        return data

    async def put_aginfer_events(self, events: list) -> Dict[str, Any]:
        resp = await self._client.put(
            "/aginfer/events",
            json={"events": events},
            timeout=self.control_timeout_s,
        )
        # Soft-fail: the in-process path also swallows errors. Return the
        # body when present so callers can log applied/skipped counts.
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"ok": False, "status_code": resp.status_code, "text": resp.text}
        if resp.status_code >= 400:
            body.setdefault("ok", False)
            body.setdefault("status_code", resp.status_code)
        return body if isinstance(body, dict) else {"ok": False, "body": body}

    async def session_end(
        self, program_id: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"program_id": program_id}
        if session_id is not None:
            payload["session_id"] = session_id
        resp = await self._client.post(
            "/aginfer/session_end",
            json=payload,
            timeout=self.control_timeout_s,
        )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"/aginfer/session_end returned non-JSON "
                f"(status={resp.status_code}): {resp.text[:200]}"
            ) from exc
        if not isinstance(body, dict):
            raise RuntimeError(
                f"/aginfer/session_end returned non-object: {type(body).__name__}"
            )
        # 409 = deferred/incomplete on some ranks; still return the body so
        # the handler can decide whether to treat it as success.
        if resp.status_code not in (200, 409):
            raise RuntimeError(
                f"/aginfer/session_end failed status={resp.status_code}: {body}"
            )
        return body

    async def abort_request(
        self, rid: Optional[str] = None, *, abort_all: bool = False
    ) -> None:
        payload: Dict[str, Any] = {"abort_all": bool(abort_all)}
        if rid is not None:
            payload["rid"] = rid
        try:
            resp = await self._client.post(
                "/abort_request",
                json=payload,
                timeout=self.control_timeout_s,
            )
            if resp.status_code >= 400:
                logger.debug(
                    "abort_request failed status=%s body=%s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:  # noqa: BLE001 — cancel path must never raise
            logger.debug("abort_request raised (ignored)", exc_info=True)

    async def generate_stream(
        self, body: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """POST /generate with ``stream=true`` and yield parsed SSE JSON chunks.

        Terminates on ``data: [DONE]``. Non-JSON data lines are skipped with a
        debug log; a non-2xx response raises ``httpx.HTTPStatusError``.
        """
        payload = dict(body)
        payload["stream"] = True
        async with self._client.stream(
            "POST",
            "/generate",
            json=payload,
            timeout=httpx.Timeout(
                self.control_timeout_s,
                read=self.generate_timeout_s,
                write=self.control_timeout_s,
                connect=self.control_timeout_s,
            ),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("skipping non-JSON SSE chunk: %r", data[:120])
                    continue
                if isinstance(chunk, dict):
                    yield chunk
