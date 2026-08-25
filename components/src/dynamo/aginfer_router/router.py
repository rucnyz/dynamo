# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent program scheduler, with value (not size) as the pause/resume gate.

Pause-least-valuable-ACTING-first; BFD restore; exponential decay on the resume
side. v0 reads real token counts from chat-completions ``usage`` instead of
upstream's ``chars / 5`` proxy estimator.

ThunderAgent schedules on working-set SIZE -- it pauses the smallest ACTING
program and restores smallest-first, because size is all it knows. This
scheduler keeps that machinery (the watermarks, the soft-demote band, the BFD
placement pass, the forced-resume timeout, SESSION_END/Dead-KV cleanup) and
overrides exactly two ordering keys with a per-program value ``V_u``:

    * which program to pause under pressure -> LOWEST value, not smallest
    * which paused program to resume        -> smallest first by default,
      unchanged from ThunderAgent; HIGHEST value first is opt-in via
      ``value_ordered_resume`` (off by default: the ceiling admits a fixed
      token budget per tick, and value-ordering it stalls the backlog --
      see ``_resume_selection_key``)

``V_u`` comes from whichever signal is available, best first:

    1. the engine's own value state (``--aginfer-state-url``), whose
       ``shared_aware_prog_scores`` splits each cache block's value across the
       sessions holding it -- the real shared-prefix signal;
    2. a router-local proxy over what the request plane already carries:
       working set x turns taken x live sub-agents.

The fallback is not a placeholder for the first: an engine dump is only
available from an sglang worker with the unified radix tree enabled, so the
proxy is the operating point for every other backend. Both feed the same
comparison, so a missing dump degrades the ordering's precision, never its
correctness -- it can be no worse than the pause-smallest baseline it replaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from dynamo.aginfer_router.engine_state import EngineStateClient
from dynamo.aginfer_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)

if TYPE_CHECKING:
    from dynamo.aginfer_router.capacity import WorkerCapacityProvider

logger = logging.getLogger(__name__)

_TERMINATED_TOMBSTONE_TTL_SECONDS = 3600.0
_MAX_TERMINATED_TOMBSTONES = 100_000


class ProgramTerminatedError(RuntimeError):
    """Raised when work arrives after a program's terminal notification."""


class ProgramNotFoundError(RuntimeError):
    """Raised when a terminal notification has no routing history."""


class DisaggregatedCleanupUnsupportedError(RuntimeError):
    """Raised when terminal cleanup would cross Dynamo worker components."""


@dataclass
class PauseDecision:
    program_id: str
    priority_jump: float = 0.0
    waited_seconds: float = 0.0
    was_paused: bool = False
    was_soft_demoted: bool = False
    assigned_worker_hint: Optional[int] = None


@dataclass
class ThunderAgentConfig:
    pause_threshold: float = 0.95
    soft_demote_threshold: float = 0.80
    soft_demote_priority_jump: float = -2.0
    resume_priority_boost: float = 1.0
    resume_timeout_seconds: float = 1800.0
    scheduler_interval_seconds: float = 5.0
    resume_hysteresis: float = 0.10
    pause_target: float = 0.80
    acting_token_weight: float = 1.0
    acting_decay_tau_seconds: float = 1.0
    buffer_per_program: int = 100

    #: Weight on the live-sub-agent holder count in the proxy value. 0 drops
    #: the holder term entirely (ablation: value = working set x turns).
    holder_weight: float = 1.0
    #: ``/aginfer/state`` URL. Unset => proxy value only.
    state_url: Optional[str] = None
    #: Order resumes by value too (most valuable admitted first) instead of
    #: the default smallest-first. Off by default because it costs makespan:
    #: see ``_resume_selection_key``. Kept as an opt-in so the cost stays
    #: measurable.
    value_ordered_resume: bool = False
    #: How much a victim's size counts against picking it, relative to its
    #: value. 0 is pure value; large approaches the size-ordered baseline.
    #: See ``_pause_victim_key`` for why pure value strands its own victims.
    victim_size_weight: float = 0.0


def _normalised_ranks(
    programs: list[Program], score: Callable[[Program], float]
) -> dict[str, float]:
    """Map each program to its rank under ``score``, scaled to [0, 1].

    Equal scores share a rank, so two identical programs cannot be separated by
    whichever happened to be inserted first.
    """
    if not programs:
        return {}
    if len(programs) == 1:
        return {programs[0].program_id: 0.0}

    ordered = sorted(programs, key=score)
    span = float(len(ordered) - 1)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and score(ordered[stop]) == score(ordered[index]):
            stop += 1
        shared = (index + stop - 1) / 2.0 / span
        for program in ordered[index:stop]:
            ranks[program.program_id] = shared
        index = stop
    return ranks


class ThunderAgentScheduler:
    def __init__(
        self,
        capacity: WorkerCapacityProvider,
        config: ThunderAgentConfig,
    ) -> None:
        self._capacity = capacity
        self._cfg = config
        self._table = ProgramTable()
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_forced_resumes = 0
        self._terminated: OrderedDict[str, float] = OrderedDict()
        self._last_logged_util = -1.0

        self._engine_state = EngineStateClient(config.state_url)
        self._http: Optional[Any] = None
        # pid -> the program that spawned it, and the reverse index. Holder
        # counts are read through the program table so a child that ended
        # stops counting even before _forget_program runs for it.
        self._parent: dict[str, str] = {}
        self._children: dict[str, set[str]] = {}
        # Per-program V_u from the last engine dump; None => no dump.
        self._engine_scores: Optional[dict[str, float]] = None
        self._engine_scores_logged: Optional[bool] = None
        self._id_mismatch_warned = False
        # Bumped whenever the engine scores are replaced, so the memoised
        # victim ranks below expire with them and not only when the table
        # changes shape.
        self._scores_generation = 0
        self._victim_keys: dict[str, float] = {}
        self._victim_keys_signature: Optional[tuple[int, int, int, int]] = None

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "ThunderAgent scheduler started (interval=%ss, pause=%.2f, soft=%.2f)",
            self._cfg.scheduler_interval_seconds,
            self._cfg.pause_threshold,
            self._cfg.soft_demote_threshold,
        )
        if self._engine_state.enabled and self._http is None:
            import httpx

            self._http = httpx.AsyncClient()
        logger.info(
            "aginfer value gate active (holder_weight=%.2f, victim_size_weight=%.2f, "
            "value_ordered_resume=%s, engine_state=%s)",
            self._cfg.holder_weight,
            self._cfg.victim_size_weight,
            self._cfg.value_ordered_resume,
            self._cfg.state_url or "off (proxy value)",
        )

    async def stop(self) -> None:
        if self._scheduler_task is None:
            return
        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            pass
        self._scheduler_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def before_request(
        self,
        program_id: str,
        estimated_prompt_tokens: int = 0,
        parent_program_id: Optional[str] = None,
    ) -> PauseDecision:
        wait_started = time.monotonic()
        async with self._lock:
            self._observe_program(program_id, parent_program_id)
            wait_event, was_paused = self._admit_locked(
                program_id, estimated_prompt_tokens
            )

        try:
            if wait_event is not None:
                await asyncio.wait_for(
                    wait_event.wait(), timeout=self._cfg.resume_timeout_seconds
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Forced resume for %s after %.1fs",
                program_id,
                self._cfg.resume_timeout_seconds,
            )
            try:
                async with self._lock:
                    program = self._table.programs.get(program_id)
                    if (
                        program is not None
                        and program.lifecycle == ProgramLifecycle.PAUSED
                    ):
                        worker_id = self._least_loaded_worker_locked(
                            self._capacity.snapshot()
                        )
                        self._resume_program(program, worker_id)
                        self._stat_forced_resumes += 1
            except BaseException:
                await self._cancel_admission(program_id)
                raise
        except BaseException:
            await self._cancel_admission(program_id)
            raise

        try:
            waited = time.monotonic() - wait_started

            async with self._lock:
                program = self._table.programs.get(program_id)
                if (
                    program is None
                    or program.lifecycle == ProgramLifecycle.TERMINATED
                    or self._is_terminated_locked(program_id)
                ):
                    raise ProgramTerminatedError(
                        f"program {program_id!r} has already been terminated"
                    )

                priority_jump = self._cfg.resume_priority_boost if was_paused else 0.0
                soft_demoted = program.soft_demoted_until > time.monotonic()
                if soft_demoted:
                    priority_jump += self._cfg.soft_demote_priority_jump

                return PauseDecision(
                    program_id=program_id,
                    priority_jump=priority_jump,
                    waited_seconds=waited,
                    was_paused=was_paused,
                    was_soft_demoted=soft_demoted,
                    assigned_worker_hint=program.assigned_worker_id,
                )
        except BaseException:
            await self._cancel_admission(program_id)
            raise

    async def _cancel_admission(self, program_id: str) -> None:
        async with self._lock:
            self._table.cancel_request(program_id)

    def _admit_locked(
        self,
        program_id: str,
        estimated_prompt_tokens: int,
    ) -> tuple[Optional[asyncio.Event], bool]:
        # Caller holds self._lock.
        existing = self._table.programs.get(program_id)
        if self._is_terminated_locked(program_id) or (
            existing is not None and existing.lifecycle == ProgramLifecycle.TERMINATED
        ):
            raise ProgramTerminatedError(
                f"program {program_id!r} has already been terminated"
            )
        was_new = program_id not in self._table.programs
        program = self._table.begin_request(program_id, estimated_prompt_tokens)
        if program.lifecycle == ProgramLifecycle.PAUSED:
            program.waiting = program.waiting or asyncio.Event()
            return program.waiting, True

        if not (was_new and program.assigned_worker_id is None):
            return None, False

        capacities = self._capacity.snapshot()
        if not capacities:
            # Cold start: MDC hasn't published yet. Let the request flow
            # through with no pin; the chunk-loop callback will populate
            # ``assigned_worker_id`` once the engine picks a worker, and
            # subsequent turns get the sticky pin.
            return None, False
        worker_id = self._select_worker_for_new_program_locked(
            capacities, program.token_total
        )
        if worker_id is not None:
            program.assigned_worker_id = worker_id
            program.kv_worker_ids.add(worker_id)
            return None, False

        # All workers full: queue until the scheduler tick resumes us.
        program.waiting = program.waiting or asyncio.Event()
        program.lifecycle = ProgramLifecycle.PAUSED
        self._table.paused[program_id] = None
        logger.debug(
            "Queued new program %s (tokens=%d)",
            program_id,
            program.token_total,
        )
        return program.waiting, True

    def record_output_tokens(self, program_id: str, delta_tokens: int) -> None:
        # No-await fast path on the streaming chunk loop. Safe because the
        # event loop is single-task; the scheduler tick tolerates a stale
        # token_total by one tick.
        program = self._table.programs.get(program_id)
        if program is not None and program.status == ProgramStatus.REASONING:
            program.token_total += delta_tokens

    async def after_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        do_pause = False
        async with self._lock:
            program = self._table.end_request(
                program_id, prompt_tokens, completion_tokens
            )
            if program is None:
                return
            if (
                program.lifecycle != ProgramLifecycle.TERMINATED
                and program.marked_for_pause
            ):
                program.marked_for_pause = False
                do_pause = True

        if do_pause:
            await self._pause_acting(program_id)

    async def assign_worker(self, program_id: str, worker_id: int) -> None:
        await self.assign_workers(program_id, decode_worker_id=worker_id)

    async def assign_workers(
        self,
        program_id: str,
        *,
        decode_worker_id: Optional[int] = None,
        prefill_worker_id: Optional[int] = None,
    ) -> None:
        """Record worker attribution needed by terminal cleanup."""
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is not None:
                if (
                    prefill_worker_id is not None
                    and prefill_worker_id != decode_worker_id
                ):
                    program.disaggregated_workers_observed = True
                if decode_worker_id is not None:
                    program.assigned_worker_id = decode_worker_id
                    program.kv_worker_ids.add(decode_worker_id)
                elif prefill_worker_id is not None:
                    # Aggregated responses normally populate both fields with
                    # the same id. Retain this fallback for older bindings that
                    # report only the active worker in the prefill slot.
                    program.assigned_worker_id = prefill_worker_id
                    program.kv_worker_ids.add(prefill_worker_id)

    async def worker_ids_for_program(
        self, program_id: str
    ) -> Optional[tuple[int, ...]]:
        """Return workers that may retain this program's KV.

        ``None`` means the program is already absent (making a duplicate end
        notification idempotent).  An empty tuple means the program is known,
        but no worker attribution was ever observed; callers must not release
        the local mapping in that case because doing so would orphan KV.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return None
            if program.disaggregated_workers_observed:
                raise DisaggregatedCleanupUnsupportedError(
                    "Dead-KV cleanup currently supports aggregated SGLang only; "
                    f"program {program_id!r} used distinct prefill/decode workers"
                )
            worker_ids = set(program.kv_worker_ids)
            if program.assigned_worker_id is not None:
                worker_ids.add(program.assigned_worker_id)
            return tuple(sorted(worker_ids))

    async def begin_end_program(self, program_id: str) -> Optional[tuple[int, ...]]:
        """Close admission, drain active requests, then snapshot KV workers.

        The program remains in the table until :meth:`end_program` is called,
        so a failed worker cleanup can be retried without losing its routing
        history. New work and paused waiters fail closed as soon as this method
        marks the lifecycle terminal.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                if self._is_terminated_locked(program_id):
                    return None
                raise ProgramNotFoundError(f"cannot end unknown program {program_id!r}")
            self._mark_terminated_locked(program_id)
            program.lifecycle = ProgramLifecycle.TERMINATED
            self._table.paused.pop(program_id, None)
            if program.waiting is not None:
                program.waiting.set()
                program.waiting = None
            drained = program.drained

        await drained.wait()

        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return None
            if program.disaggregated_workers_observed:
                raise DisaggregatedCleanupUnsupportedError(
                    "Dead-KV cleanup currently supports aggregated SGLang only; "
                    f"program {program_id!r} used distinct prefill/decode workers"
                )
            worker_ids = set(program.kv_worker_ids)
            if program.assigned_worker_id is not None:
                worker_ids.add(program.assigned_worker_id)
            return tuple(sorted(worker_ids))

    async def _scheduler_loop(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                await asyncio.sleep(self._cfg.scheduler_interval_seconds)
                try:
                    await self._scheduler_tick()
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.exception("ThunderAgent scheduler tick error")
                    if consecutive_failures >= 10:
                        logger.error(
                            "Scheduler tick failed %d times in a row; halting loop",
                            consecutive_failures,
                        )
                        return
        except asyncio.CancelledError:
            return

    async def _scheduler_tick(self) -> None:
        await self._refresh_engine_scores()
        capacities = self._capacity.snapshot()
        if not capacities:
            return
        self._log_utilization(capacities)
        # Upstream TA ordering: resume first, then pause -- a program paused
        # this tick can't resume until the next.
        self._apply_soft_demotes(capacities)
        await self._greedy_resume(capacities)
        await self._pause_until_safe(capacities)

    def _log_utilization(self, capacities: dict[int, int]) -> None:
        """Report headroom whenever it moves materially.

        Same contract as ``thunderagent_router``'s: ``run_ab_router.sh``'s
        ``isolate_arm`` greps this exact literal to print what an arm actually
        started from, and a run with no pauses is otherwise indistinguishable
        from one that never came near ``pause_threshold``.
        """
        busiest = max(
            capacities,
            key=lambda w: self._worker_used(w) / max(1, capacities[w]),
        )
        capacity = max(1, capacities[busiest])
        used = self._worker_used(busiest)
        util = used / capacity
        if abs(util - self._last_logged_util) < 0.05:
            return
        self._last_logged_util = util
        logger.info(
            "scheduler.util worker=%s used=%d/%d util=%.2f programs=%d "
            "paused=%d (pause at %.2f)",
            busiest,
            used,
            capacity,
            util,
            len(self._active_programs_for_worker(busiest)),
            len(self._table.paused),
            self._cfg.pause_threshold,
        )

    def _program_tokens(self, program: Program, *, decayed: bool = False) -> int:
        if program.status != ProgramStatus.ACTING:
            return program.token_total
        if not decayed:
            return int(program.token_total * self._cfg.acting_token_weight)
        tau = max(self._cfg.acting_decay_tau_seconds, 1e-3)
        idle = (
            max(0.0, time.monotonic() - program.acting_since)
            if program.acting_since > 0
            else 0.0
        )
        return int(program.token_total * (2.0 ** (-(idle / tau))))

    # ---- program structure: the holder signal -------------------------------

    def _observe_program(
        self, program_id: str, parent_program_id: Optional[str]
    ) -> None:
        """Record ``parent_session_id`` as a holder edge.

        A program with live sub-agents is worth more than its own bytes: it is
        blocked on them, so it WILL be re-entered when they return, and they
        were forked from its context, so its prefix is shared rather than
        private. Pause-smallest-first cannot see either -- the parent is often
        the larger program, so size-ordered restore puts it last.

        Called under ``self._lock`` on every request, before admission.
        """
        if not parent_program_id or parent_program_id == program_id:
            return
        if self._parent.get(program_id) == parent_program_id:
            return
        self._parent[program_id] = parent_program_id
        self._children.setdefault(parent_program_id, set()).add(program_id)

    def _forget_program(self, program_id: str) -> None:
        """Drop whatever ``_observe_program`` recorded.

        Called after the program leaves the table, so bookkeeping cannot
        outlive the program it describes.
        """
        parent = self._parent.pop(program_id, None)
        if parent is not None:
            siblings = self._children.get(parent)
            if siblings is not None:
                siblings.discard(program_id)
                if not siblings:
                    del self._children[parent]
        self._children.pop(program_id, None)

    def _n_holders(self, program_id: str) -> float:
        children = self._children.get(program_id)
        if not children:
            return 1.0
        live = sum(1 for child in children if child in self._table.programs)
        return 1.0 + self._cfg.holder_weight * live

    # ---- the value itself -----------------------------------------------

    def _program_value(self, program: Program) -> float:
        """``V_u`` for a program's resident KV. Higher = keep / resume first.

        Prefers the engine's own value state (``shared_aware_prog_scores``,
        the real shared-prefix signal); falls back to a router-local proxy
        when no engine dump is available (e.g. non-sglang backends, or the
        unified radix tree is off).
        """
        if self._engine_scores is not None:
            score = self._engine_scores.get(program.program_id)
            if score is not None:
                return float(score)
        return self._proxy_value(program)

    def _proxy_value(self, program: Program) -> float:
        """Value from what the request plane alone carries.

        ``working set x turns taken x holders``:

        * working set -- the re-prefill cost if this KV is dropped;
        * turns taken -- a program on its tenth turn has already re-read its
          prefix nine times, so the same bytes have earned more than a fresh
          program's;
        * holders -- live sub-agents forked from this context (see
          ``_observe_program``).
        """
        tokens = float(max(0, program.token_total))
        turns = 1.0 + float(max(0, program.step_count))
        return tokens * turns * self._n_holders(program.program_id)

    def _active_programs_for_worker(self, worker_id: int) -> list[Program]:
        return [
            p
            for p in self._table.programs.values()
            if p.lifecycle == ProgramLifecycle.ACTIVE
            and p.assigned_worker_id == worker_id
        ]

    def _worker_used(self, worker_id: int, *, decayed: bool = False) -> int:
        programs = self._active_programs_for_worker(worker_id)
        tokens = sum(self._program_tokens(p, decayed=decayed) for p in programs)
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _least_loaded_worker_locked(self, capacities: dict[int, int]) -> Optional[int]:
        if not capacities:
            return None
        return max(
            capacities,
            key=lambda w: capacities[w] - self._worker_used(w, decayed=True),
        )

    def _select_worker_for_new_program_locked(
        self,
        capacities: dict[int, int],
        estimated_tokens: int,
    ) -> Optional[int]:
        # Fairness: new programs queue behind any existing paused program.
        if self._table.paused:
            return None
        buffer = self._cfg.buffer_per_program
        required = estimated_tokens + buffer
        candidates = [
            (w, self._worker_used(w))
            for w, c in capacities.items()
            if c - self._worker_used(w) >= required
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])[0]

    def _apply_soft_demotes(self, capacities: dict[int, int]) -> None:
        soft_until = time.monotonic() + self._cfg.scheduler_interval_seconds * 1.5
        for worker_id, capacity in capacities.items():
            util = self._worker_used(worker_id) / capacity
            if not (
                self._cfg.soft_demote_threshold <= util < self._cfg.pause_threshold
            ):
                continue
            for program in self._active_programs_for_worker(worker_id):
                if (
                    not program.marked_for_pause
                    and program.soft_demoted_until < soft_until
                ):
                    program.soft_demoted_until = soft_until

    async def _pause_until_safe(self, capacities: dict[int, int]) -> None:
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for worker_id, capacity in capacities.items():
            # Hold the lock for the entire per-worker decision so the snapshot
            # of program state used by _smallest_candidates / _worker_used
            # cannot race with concurrent before_request admissions.
            async with self._lock:
                base_used = self._worker_used(worker_id)
                if base_used <= capacity * threshold:
                    continue

                target_limit = capacity * pause_target
                paused_this_tick = 0
                marked_this_tick = 0
                # Bound the inner loop by total program count so a candidate
                # transitioning out from under us can't spin the tick.
                for _ in range(len(self._table.programs) + 1):
                    if self._worker_used(worker_id) <= target_limit:
                        break
                    acting, reasoning = self._smallest_candidates(worker_id)
                    if acting is not None:
                        if self._pause_acting_locked(acting.program_id):
                            paused_this_tick += 1
                        continue
                    if reasoning is not None:
                        if (
                            not reasoning.marked_for_pause
                            and reasoning.lifecycle == ProgramLifecycle.ACTIVE
                            and reasoning.status == ProgramStatus.REASONING
                        ):
                            reasoning.marked_for_pause = True
                            marked_this_tick += 1
                        continue
                    break

                final_used = self._worker_used(worker_id)

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "scheduler.tick worker=%s paused=%d marked=%d util=%.4f -> %.4f",
                    worker_id,
                    paused_this_tick,
                    marked_this_tick,
                    base_used / capacity,
                    final_used / capacity,
                )

    # ---- the overridden ordering keys ---------------------------------------

    def _pause_victim_key(self, program: Program) -> float:
        """Least valuable first, optionally discounted by how big the victim is.

        Pure value (``victim_size_weight = 0``) loses to the size-ordered
        baseline, and measurably so: it pauses 4x as often, each pause lasts
        3.8x longer and 42 of 100 resumes come from the forced-resume timer
        rather than from freed capacity (``benchmark/dynamo/README.md``). The
        reason is that the engine scores a large low-reuse working set very
        negative, so pure value systematically picks the *largest* programs,
        and a large victim does not fit back under the resume ceiling. It
        strands until the timer releases it, is still the least valuable, and
        is paused again.

        ``V(u) = p_reuse x reload_cost - holding_cost`` answers "what is worth
        least" and says nothing about "can this be taken back", so this weight
        supplies the missing half. Value and size are blended as normalised
        ranks rather than raw numbers because they share no unit -- engine
        scores land in the tens while the proxy lands in the millions, and any
        fixed coefficient between them would be meaningless in one regime or
        the other.

        Ranks also make the two arms endpoints of one knob: 0 is the value
        gate, and a large enough weight reproduces the baseline's
        pause-smallest.

        Avoid ``w = 1`` exactly. Equal weight on the two ranks means a program
        that is last by value and first by size scores the same as its mirror
        image, and on this workload that cancellation is the common case
        rather than a corner: the engine scores large working sets most
        negative, so value is anti-correlated with size by construction.
        Sweep either side of 1.
        """
        if self._cfg.victim_size_weight <= 0.0:
            return self._program_value(program)
        blended = self._blended_victim_keys().get(program.program_id)
        # A program not yet in the table (or a single-program table) has no
        # rank to speak of; value alone still orders it consistently.
        return self._program_value(program) if blended is None else blended

    def _blended_victim_keys(self) -> dict[str, float]:
        """Per-program ``value_rank + w * size_rank``, recomputed when stale.

        Memoised because the key is asked for pairwise -- once per candidate
        per comparison -- and a rank needs the whole table, which would
        otherwise be re-sorted on every request.
        """
        signature = (
            len(self._table.programs),
            sum(p.token_total for p in self._table.programs.values()),
            sum(p.step_count for p in self._table.programs.values()),
            self._scores_generation,
        )
        if signature != self._victim_keys_signature:
            programs = list(self._table.programs.values())
            value_rank = _normalised_ranks(programs, self._program_value)
            size_rank = _normalised_ranks(programs, lambda p: float(p.token_total))
            weight = self._cfg.victim_size_weight
            self._victim_keys = {
                pid: value_rank[pid] + weight * size_rank[pid] for pid in value_rank
            }
            self._victim_keys_signature = signature
        return self._victim_keys

    def _resume_selection_key(self, program: Program) -> float:
        """Size order (smallest first) unless ``value_ordered_resume``.

        Pause and resume are not the same decision, so they do not take the
        same key. Pause is a sacrifice: give up the KV worth least. Resume is
        a queue: the ceiling admits a fixed number of tokens per tick, so
        smallest-first lets the most programs back per tick and drains the
        backlog fastest.

        Ordering resumes by value inverts that -- the most valuable program is
        usually also the largest (working set is a factor of the value), so
        each tick readmits fewer programs, the backlog stops moving, and the
        forced-resume timeout becomes the main way out. Measured on the
        600-request Claude Code slice under manufactured pressure,
        value-ordered resume ran 56.7 tok/s against the baseline's 144.7 with
        11 of 21 resumes fired by the timer rather than by capacity; see
        ``benchmark/dynamo/README.md``.
        """
        if self._cfg.value_ordered_resume:
            # Ascending sort, so negate: most valuable is admitted first.
            return -self._program_value(program)
        return float(program.token_total)

    # ---- engine state refresh (tick-cached, never per-request) --------------

    async def _refresh_engine_scores(self) -> None:
        """Pull the engine's value state once per tick.

        The dump costs 5-50ms under load, so it stays on the background tick
        and never touches request ingress. Any failure leaves
        ``_engine_scores`` as None and the proxy takes over on the next
        comparison.
        """
        if self._http is None or not self._engine_state.enabled:
            return
        state = await self._engine_state.fetch(self._http)
        scores = self._engine_state.program_scores(state)
        available = scores is not None
        if available != self._engine_scores_logged:
            self._engine_scores_logged = available
            logger.info(
                "aginfer value source: %s",
                "engine state (%d programs scored)" % len(scores)
                if scores is not None
                else "proxy (engine state unavailable)",
            )
        self._engine_scores = scores
        self._scores_generation += 1
        self._warn_if_id_spaces_disjoint(scores)

    def _warn_if_id_spaces_disjoint(self, scores: Optional[dict[str, float]]) -> None:
        """A dump full of scores that match no program is an ID-space mismatch.

        The engine keys value by ITS program id; the router keys by the
        ``session_id`` on the wire. If those diverge -- salting, sanitising, a
        different namespace -- every lookup misses and the per-program
        fallback quietly serves the proxy for the whole fleet: the value path
        would look enabled while never being used.
        """
        if self._id_mismatch_warned or not scores or not self._table.programs:
            return
        if any(pid in scores for pid in self._table.programs):
            return
        self._id_mismatch_warned = True
        logger.warning(
            "aginfer engine state scores %d programs but none match the "
            "router's session ids (e.g. engine=%r router=%r); the value gate "
            "is running on the proxy. Check that the worker keys programs by "
            "the request's session_id.",
            len(scores),
            next(iter(scores)),
            next(iter(self._table.programs)),
        )

    def _smallest_candidates(
        self, worker_id: int
    ) -> tuple[Optional[Program], Optional[Program]]:
        smallest_acting: Optional[Program] = None
        smallest_reasoning: Optional[Program] = None
        for program in self._table.programs.values():
            if program.assigned_worker_id != worker_id:
                continue
            if program.lifecycle != ProgramLifecycle.ACTIVE:
                continue
            if program.marked_for_pause:
                continue
            if program.status == ProgramStatus.ACTING:
                if smallest_acting is None or self._pause_victim_key(
                    program
                ) < self._pause_victim_key(smallest_acting):
                    smallest_acting = program
            elif program.status == ProgramStatus.REASONING:
                if smallest_reasoning is None or self._pause_victim_key(
                    program
                ) < self._pause_victim_key(smallest_reasoning):
                    smallest_reasoning = program
        return smallest_acting, smallest_reasoning

    async def _pause_acting(self, program_id: str) -> bool:
        async with self._lock:
            return self._pause_acting_locked(program_id)

    def _pause_acting_locked(self, program_id: str) -> bool:
        # Caller holds self._lock.
        program = self._table.programs.get(program_id)
        if program is None:
            return False
        if program.lifecycle == ProgramLifecycle.PAUSED:
            return False
        if program.status != ProgramStatus.ACTING:
            return False
        program.lifecycle = ProgramLifecycle.PAUSED
        program.assigned_worker_id = None
        if program.waiting is None:
            program.waiting = asyncio.Event()
        else:
            program.waiting.clear()
        self._table.paused[program_id] = None
        # INFO, with the key that selected it: comparing two schedulers means
        # comparing the victims they pick, and a count of pauses cannot show
        # that they picked differently. run_ab_router.sh / analyze_ab.py grep
        # this exact literal ("scheduler.pause") to attribute pause events to
        # an arm, so keep it in parity with thunderagent_router's.
        logger.info(
            "scheduler.pause program=%s tokens=%d key=%.6g",
            program_id,
            program.token_total,
            self._pause_victim_key(program),
        )
        return True

    async def end_program(self, program_id: str) -> bool:
        """Release a finished program.

        Deletes it from the program table + paused set and wakes any waiter,
        so its tokens stop counting against worker utilization. Mirrors
        upstream TA's ``release_program``. Idempotent: returns False if unknown.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return False
            self._mark_terminated_locked(program_id)
            program.lifecycle = ProgramLifecycle.TERMINATED
            if program.waiting is not None:
                program.waiting.set()  # unblock any coroutine paused on this program
                program.waiting = None
            self._table.release(program_id)
            self._forget_program(program_id)
            logger.info(
                "Released program %s (%d remaining)",
                program_id,
                len(self._table.programs),
            )
            return True

    def _is_terminated_locked(self, program_id: str) -> bool:
        # Caller holds self._lock.
        deadline = self._terminated.get(program_id)
        if deadline is None:
            return False
        if deadline <= time.monotonic():
            self._terminated.pop(program_id, None)
            return False
        self._terminated.move_to_end(program_id)
        return True

    def _mark_terminated_locked(self, program_id: str) -> None:
        # Caller holds self._lock. Bound memory while retaining enough history
        # to reject delayed/retried requests that reuse a closed session id.
        self._terminated[program_id] = (
            time.monotonic() + _TERMINATED_TOMBSTONE_TTL_SECONDS
        )
        self._terminated.move_to_end(program_id)
        while len(self._terminated) > _MAX_TERMINATED_TOMBSTONES:
            self._terminated.popitem(last=False)

    async def _greedy_resume(self, capacities: dict[int, int]) -> None:
        if not self._table.paused:
            return

        async with self._lock:
            paused_programs = [
                self._table.programs[pid]
                for pid in self._table.paused
                if pid in self._table.programs
            ]
            if not paused_programs:
                return

            def group_key(program: Program) -> int:
                if program.step_count <= 1:
                    return 1
                if program.status == ProgramStatus.REASONING:
                    return 0
                return 2

            # Within the structural group, order by ``_resume_selection_key``
            # (size ascending by default; value-descending opt-in).
            paused_programs.sort(
                key=lambda p: (group_key(p), self._resume_selection_key(p))
            )

            resume_ceiling = max(
                0.0, self._cfg.pause_threshold - self._cfg.resume_hysteresis
            )
            backend_caps = [
                (w, int(c * resume_ceiling) - self._worker_used(w, decayed=False))
                for w, c in capacities.items()
            ]
            backend_caps = [
                (w, r) for w, r in backend_caps if r > self._cfg.buffer_per_program
            ]
            if not backend_caps:
                return

            backend_caps.sort(key=lambda x: -x[1])

            total_capacity = sum(r for _, r in backend_caps)
            resumable_programs: list[Program] = []
            cumulative = 0
            for program in paused_programs:
                required = program.token_total + self._cfg.buffer_per_program
                if cumulative + required <= total_capacity:
                    resumable_programs.append(program)
                    cumulative += required

            if not resumable_programs:
                return

            resumable_programs.sort(key=self._resume_selection_key)
            min_required = (
                min(p.token_total for p in resumable_programs)
                + self._cfg.buffer_per_program
            )

            resumed_this_tick = 0
            for program in resumable_programs:
                if not backend_caps:
                    break
                worker_id, remaining = backend_caps[0]
                if min_required > remaining:
                    break
                required = program.token_total + self._cfg.buffer_per_program
                if required > remaining:
                    continue
                self._resume_program(program, worker_id)
                resumed_this_tick += 1
                updated_remaining = remaining - required
                if updated_remaining > self._cfg.buffer_per_program:
                    backend_caps[0] = (worker_id, updated_remaining)
                    backend_caps.sort(key=lambda x: -x[1])
                else:
                    backend_caps.pop(0)

            if resumed_this_tick:
                logger.info(
                    "scheduler.tick resumed=%d still_paused=%d",
                    resumed_this_tick,
                    len(self._table.paused),
                )

    def _resume_program(
        self, program: Program, target_worker_id: Optional[int]
    ) -> None:
        # Caller holds self._lock.
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.assigned_worker_id = target_worker_id
        if target_worker_id is not None:
            program.kv_worker_ids.add(target_worker_id)
        notify = program.waiting
        program.waiting = None
        self._table.paused.pop(program.program_id, None)
        if notify is not None:
            notify.set()
        # INFO, same literal contract as scheduler.pause above -- see there.
        logger.info(
            "scheduler.resume program=%s worker=%s tokens=%d key=%.6g",
            program.program_id,
            target_worker_id,
            program.token_total,
            self._resume_selection_key(program),
        )
