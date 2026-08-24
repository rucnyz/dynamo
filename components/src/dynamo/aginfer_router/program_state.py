# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program lifecycle data model. Mirrors ``ThunderAgent/program/state.py``.

v0 difference: ``token_total`` is real ``prompt_tokens + completion_tokens``
from chat-completions ``usage``, not upstream's ``chars / 5`` heuristic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProgramStatus(Enum):
    REASONING = "reasoning"
    ACTING = "acting"


class ProgramLifecycle(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    program_id: str

    status: ProgramStatus = ProgramStatus.REASONING
    lifecycle: ProgramLifecycle = ProgramLifecycle.ACTIVE

    assigned_worker_id: Optional[int] = None

    # Every aggregated worker that may still hold KV for this program.  The
    # active ``assigned_worker_id`` is cleared when a program is paused/moved,
    # but a terminal notification still has to reach the old worker so it can
    # drop the program's cache.
    kv_worker_ids: set[int] = field(default_factory=set)

    # The current control-plane client is scoped to one Dynamo component.
    # Distinct prefill/decode workers therefore cannot be cleaned safely yet.
    disaggregated_workers_observed: bool = False

    token_total: int = 0

    step_count: int = 0
    marked_for_pause: bool = False
    # monotonic seconds; >0 means priority demotion active
    soft_demoted_until: float = 0.0
    waiting: Optional[asyncio.Event] = field(default=None, repr=False)

    # Requests that passed admission but have not yet run ``after_request``.
    # A terminal notification waits for this count to reach zero before it
    # snapshots the KV-worker set, otherwise a concurrently streaming request
    # can publish a new worker after cleanup has already been acknowledged.
    inflight_requests: int = 0
    drained: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    # monotonic seconds; used to compute resume-side decay
    acting_since: float = 0.0


@dataclass
class ProgramTable:
    programs: dict[str, Program] = field(default_factory=dict)
    # Insertion-ordered: ties in `_greedy_resume`'s sort resolve oldest-paused
    # first, mirroring upstream TA. Values are unused.
    paused: dict[str, None] = field(default_factory=dict)

    def begin_request(
        self, program_id: str, estimated_prompt_tokens: int = 0
    ) -> Program:
        program = self.programs.get(program_id)
        if program is None:
            program = Program(program_id=program_id)
            self.programs[program_id] = program
        program.step_count += 1
        if estimated_prompt_tokens > 0:
            program.token_total = estimated_prompt_tokens
        program.status = ProgramStatus.REASONING
        program.acting_since = 0.0
        program.inflight_requests += 1
        program.drained.clear()
        return program

    def end_request(
        self, program_id: str, prompt_tokens: int, completion_tokens: int
    ) -> Optional[Program]:
        program = self.programs.get(program_id)
        if program is None:
            return None
        if program.inflight_requests > 0:
            program.inflight_requests -= 1
        if program.inflight_requests == 0:
            program.drained.set()
        program.token_total = prompt_tokens + completion_tokens
        program.status = ProgramStatus.ACTING
        program.acting_since = time.monotonic()
        return program

    def cancel_request(self, program_id: str) -> Optional[Program]:
        """Undo an admission that is rejected by a concurrent terminal event."""
        program = self.programs.get(program_id)
        if program is None:
            return None
        if program.inflight_requests > 0:
            program.inflight_requests -= 1
        if program.inflight_requests == 0:
            program.drained.set()
        return program

    def release(self, program_id: str) -> Optional[Program]:
        """Remove a finished program from the table (and the paused set).

        Mirrors upstream TA's ``release_program`` deletion. Returns the removed
        Program (or None if it was already gone).
        """
        self.paused.pop(program_id, None)
        return self.programs.pop(program_id, None)
