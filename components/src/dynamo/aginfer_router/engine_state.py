# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aginfer_router engine-state client (#251 step 5: admission → router).

The verified #251 split puts MIGRATION in the engine (done, increments 1-4) and ADMISSION
(program pause/resume) at the ROUTER — because admission enforcement is a request-INGRESS
blocking gate and the pause DECISION needs the cross-rank fleet view a per-rank engine plugin
lacks. For the router to make a VALUE-AWARE pause decision (not its current token-size proxy)
it must read the engine's value state.

This module is the bridge. It GETs the engine's ``/aginfer/state`` dump and builds the engine's
REAL ``SchedulerState`` by REUSING the in-engine ``build_paper_state`` (single source — no copy
of the value/state math; the router runs with the sglang fork on its PYTHONPATH and imports it
directly). The router then feeds that ``SchedulerState`` to the in-engine ``admission_controller``
(also imported, single source) to get value-aware pause/resume candidates.

DO-NO-HARM: when no state URL is configured, or the engine is unreachable, or the backend is not
an sglang/aginfer worker (no ``/aginfer/state``), this returns ``None`` and the router falls back
to its existing token-size pause proxy. So enabling the value model can never make admission worse
than the ThunderAgent baseline — it can only refine it when real engine state is available.

Hot-path note (#160/F3): the engine dump is 5-50ms under load, so this is driven by the router's
periodic SCHEDULER TICK (a cached snapshot), NEVER a per-request fetch on the ingress critical path.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("dynamo.aginfer_router.engine_state")


class EngineStateClient:
    """Fetches ``/aginfer/state`` and builds the engine's ``SchedulerState`` (in-engine reuse).

    Stateless across fetches except for an empty ``ProgramTracker`` + an unknown-tier log set
    that ``build_paper_state`` needs (the router holds the authoritative program lifecycle in its
    own ``ProgramStatus`` table; the tracker here is the build's required arg, kept empty — program
    state for the value math comes from the dump's ``per_program_usage`` the same way the in-engine
    tick uses it). The synthetic ``MEMORY_PRESSURE`` event makes ``build_paper_state`` produce the
    pressure decision-set (top-k), matching the daemon's pressure decides.
    """

    def __init__(self, state_url: Optional[str]) -> None:
        # e.g. "http://127.0.0.1:9200/aginfer/state" (the dev/dynamo aginfer_bridge) or a direct
        # sglang HTTP server. None / "" ⇒ disabled ⇒ router uses its size proxy (do-no-harm).
        self._url = state_url or None
        self._tracker = None              # lazily-built sglang ProgramTracker (build_paper_state arg)
        self._unknown_tier_log: set = set()

    @property
    def enabled(self) -> bool:
        return self._url is not None

    async def fetch(self, http_client) -> Optional[Any]:
        """GET the dump via the injected async ``http_client`` and build a ``SchedulerState``.
        Returns ``None`` on ANY failure (disabled / non-200 / network / parse / build) so the
        caller degrades to the size proxy. ``http_client`` is injected for testability."""
        if self._url is None:
            return None
        try:
            resp = await http_client.get(self._url, timeout=5.0)
            status = getattr(resp, "status_code", None)
            if status != 200:
                logger.warning("aginfer engine-state fetch %s -> %s", self._url, status)
                return None
            dump = resp.json()
        except Exception as exc:  # noqa: BLE001 — never let a state-fetch failure break routing
            logger.warning("aginfer engine-state fetch failed (%s): %r", self._url, exc)
            return None
        return self.build(dump)

    def build(self, dump: Any) -> Optional[Any]:
        """Parsed ``/aginfer/state`` dump → engine ``SchedulerState`` via the in-engine
        ``build_paper_state`` (single source). Returns ``None`` on a non-dict / unsupported-cache
        dump or if the sglang import / build fails (degrade-to-proxy)."""
        if not isinstance(dump, dict) or "unsupported_tree_cache" in dump:
            return None
        try:
            from sglang.srt.mem_cache.aginfer.state_builder import build_paper_state
            from sglang.srt.mem_cache.aginfer.events import Event, EventKind
            from sglang.srt.mem_cache.aginfer.program_tracker import ProgramTracker
            if self._tracker is None:
                self._tracker = ProgramTracker()
            return build_paper_state(
                dump,
                event=Event(kind=EventKind.MEMORY_PRESSURE),
                tracker=self._tracker,
                unknown_tier_log=self._unknown_tier_log,
            )
        except Exception as exc:  # noqa: BLE001 — sglang not on path / build raised ⇒ degrade
            logger.warning("aginfer engine-state build failed (degrade to size proxy): %r", exc)
            return None
