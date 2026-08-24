# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aginfer_router engine-state client (#251 step 5: admission → router).

The verified #251 split puts MIGRATION in the engine (done, increments 1-4) and ADMISSION
(program pause/resume) at the ROUTER — because admission enforcement is a request-INGRESS
blocking gate and the pause DECISION needs the cross-rank fleet view a per-rank engine plugin
lacks. For the router to make a VALUE-AWARE pause decision (not its current token-size proxy)
it must read the engine's value state.

This module is the bridge. It fetches the engine's aginfer dump and builds the engine's
REAL ``SchedulerState`` by REUSING the in-engine ``build_paper_state`` (single source — no copy
of the value/state math; the router runs with the sglang fork on its PYTHONPATH and imports it
directly). The router then feeds that ``SchedulerState`` to the in-engine ``admission_controller``
(also imported, single source) to get value-aware pause/resume candidates.

TWO TRANSPORTS, one dump. ``python -m dynamo.sglang`` does not serve sglang's own HTTP surface,
so ``GET /aginfer/state`` does not exist in a Dynamo stack. What does exist there is the worker's
system-status server (``DYN_SYSTEM_PORT``) and its generic ``POST /engine/call_tokenizer_manager``
passthrough (registered by ``--enable-rl``), which reaches the same
``tokenizer_manager.get_aginfer_state()`` the HTTP route calls. The transport is inferred from the
configured URL's path rather than a second flag, and the payload is normalised to one dump either
way — sglang's ``dump_aginfer_state_bytes`` returns JSON *text* precisely so the one serialisation
survives both hops.

DO-NO-HARM: when no state URL is configured, or the engine is unreachable, or the backend is not
an sglang/aginfer worker (no ``/aginfer/state``), this returns ``None`` and the router falls back
to its existing token-size pause proxy. So enabling the value model can never make admission worse
than the ThunderAgent baseline — it can only refine it when real engine state is available.

Hot-path note (#160/F3): the engine dump is 5-50ms under load, so this is driven by the router's
periodic SCHEDULER TICK (a cached snapshot), NEVER a per-request fetch on the ingress critical path.
"""
from __future__ import annotations

import json
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
        # Either a direct sglang HTTP server ("http://host:30000/aginfer/state") or a Dynamo
        # worker's system-status passthrough ("http://host:9090/engine/call_tokenizer_manager").
        # None / "" ⇒ disabled ⇒ router uses its size proxy (do-no-harm).
        self._url = state_url or None
        # The passthrough is a POST with a method name in the body; the native route is a plain
        # GET. Inferred from the path so the operator configures one URL, not a URL and a mode.
        self._via_engine_route = bool(self._url) and "/engine/" in self._url
        self._tracker = None              # lazily-built sglang ProgramTracker (build_paper_state arg)
        self._unknown_tier_log: set = set()
        self._multi_rank_warned = False

    @property
    def enabled(self) -> bool:
        return self._url is not None

    @property
    def via_engine_route(self) -> bool:
        return self._via_engine_route

    async def fetch(self, http_client) -> Optional[Any]:
        """Fetch the dump via the injected async ``http_client`` and build a ``SchedulerState``.
        Returns ``None`` on ANY failure (disabled / non-200 / network / parse / build) so the
        caller degrades to the size proxy. ``http_client`` is injected for testability."""
        if self._url is None:
            return None
        try:
            if self._via_engine_route:
                resp = await http_client.post(
                    self._url, json={"method": "get_aginfer_state"}, timeout=5.0
                )
            else:
                resp = await http_client.get(self._url, timeout=5.0)
            status = getattr(resp, "status_code", None)
            if status != 200:
                logger.warning("aginfer engine-state fetch %s -> %s", self._url, status)
                return None
            dump = self._unwrap(resp.json())
        except Exception as exc:  # noqa: BLE001 — never let a state-fetch failure break routing
            logger.warning("aginfer engine-state fetch failed (%s): %r", self._url, exc)
            return None
        return self.build(dump)

    def _unwrap(self, payload: Any) -> Any:
        """Any of the three envelopes the dump arrives in → the dump itself.

        * ``{"result": [{"state_bytes": "<json text>" | None, "state": {...}}, ...]}`` — the
          Dynamo ``call_tokenizer_manager`` passthrough, which dataclass-asdicts one
          ``GetAginferStateReqOutput`` per DP rank.
        * ``{"per_rank": [dump, ...]}`` — sglang's own HTTP route with more than one DP rank.
        * the bare dump — sglang's HTTP route, single rank.

        Multi-rank takes rank 0: the daemon owns the same hash space across replicas in this
        deployment, so rank 0's value ordering is representative, and the router only needs a
        relative order over programs. Logged once so a real multi-rank deployment is not silent.
        """
        if not isinstance(payload, dict):
            return payload
        ranks = payload.get("result")
        if isinstance(ranks, list) and ranks:
            first = ranks[0]
            if isinstance(first, dict) and ("state_bytes" in first or "state" in first):
                self._note_multi_rank(len(ranks))
                text = first.get("state_bytes")
                if isinstance(text, str):
                    return json.loads(text)
                return first.get("state")
        per_rank = payload.get("per_rank")
        if isinstance(per_rank, list) and per_rank:
            self._note_multi_rank(len(per_rank))
            return per_rank[0]
        return payload

    def _note_multi_rank(self, n: int) -> None:
        if n > 1 and not self._multi_rank_warned:
            self._multi_rank_warned = True
            logger.warning(
                "aginfer engine-state: %d DP ranks in the dump; scoring on rank 0 only", n
            )

    def program_scores(self, state: Any) -> Optional[dict]:
        """``SchedulerState`` → per-program ``V_u`` via the in-engine
        ``shared_aware_prog_scores`` (single source — the holder-split value the
        engine's own admission math uses, so the router orders programs on the same
        number the engine would). ``None`` on a missing state / import failure / a
        non-dict return, which sends the caller back to its proxy."""
        if state is None:
            return None
        try:
            from sglang.srt.mem_cache.aginfer.admission_controller import (
                shared_aware_prog_scores,
            )
            scores = shared_aware_prog_scores(state)
        except Exception as exc:  # noqa: BLE001 — sglang absent / math raised ⇒ degrade
            logger.warning("aginfer program_scores failed (degrade to proxy): %r", exc)
            return None
        if not isinstance(scores, dict) or not scores:
            return None
        return {str(k): float(v) for k, v in scores.items()}

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
