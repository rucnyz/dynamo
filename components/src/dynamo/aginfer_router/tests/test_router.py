# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgentScheduler that don't need a Dynamo runtime.

The pause/resume/BFD/SESSION_END machinery is covered in the first half. The
second half (below ``# ---- value gate`` ) pins the value-gate delta on top of
it -- that value, not size, decides who gets paused and (opt-in) who gets
resumed first -- so each of those tests is built so the size-ordered baseline
would make the OPPOSITE choice.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.aginfer_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.aginfer_router.router import (
    DisaggregatedCleanupUnsupportedError,
    ProgramNotFoundError,
    ProgramTerminatedError,
    ThunderAgentConfig,
    ThunderAgentScheduler,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeCapacity:
    """Stand-in for WorkerCapacityProvider that returns a fixed snapshot."""

    workers: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[int, int]:
        return dict(self.workers)


def make_router(
    capacity_workers: Optional[dict[int, int]] = None,
    config: Optional[ThunderAgentConfig] = None,
) -> tuple[ThunderAgentScheduler, FakeCapacity]:
    capacity = FakeCapacity(workers=capacity_workers or {})
    cfg = config or ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
        pause_threshold=0.95,
        soft_demote_threshold=0.80,
    )
    return ThunderAgentScheduler(capacity, cfg), capacity  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_first_turn_no_admission_block():
    router, _ = make_router()
    decision = await router.before_request("p1")
    assert decision.was_paused is False
    assert decision.priority_jump == 0.0


@pytest.mark.asyncio
async def test_after_request_records_real_tokens():
    router, _ = make_router()
    await router.before_request("p1")
    await router.after_request("p1", prompt_tokens=120, completion_tokens=30)
    program = router._table.programs["p1"]
    assert program.token_total == 150
    assert program.status == ProgramStatus.ACTING


@pytest.mark.asyncio
async def test_before_request_records_exact_prompt_estimate_before_admission():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=1234)
    program = router._table.programs["p1"]
    assert program.token_total == 1234
    assert program.status == ProgramStatus.REASONING


@pytest.mark.asyncio
async def test_assigned_worker_hint_reflects_sticky_assignment():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.assign_worker("p1", 3)
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 3


@pytest.mark.asyncio
async def test_worker_ids_for_program_tracks_aggregated_and_old_placements():
    router, _ = make_router()
    await router.before_request("p1")
    await router.assign_workers("p1", decode_worker_id=3, prefill_worker_id=3)

    # A resume/migration changes the active sticky route but must not discard
    # workers that can still retain older KV for terminal cleanup.
    async with router._lock:
        router._table.programs["p1"].lifecycle = ProgramLifecycle.PAUSED
        router._table.programs["p1"].assigned_worker_id = None
        router._resume_program(router._table.programs["p1"], target_worker_id=5)

    assert await router.worker_ids_for_program("p1") == (3, 5)


@pytest.mark.asyncio
async def test_disaggregated_cleanup_fails_closed_and_retains_mapping():
    router, _ = make_router()
    await router.before_request("p1")
    await router.assign_workers("p1", decode_worker_id=3, prefill_worker_id=7)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=1)

    with pytest.raises(DisaggregatedCleanupUnsupportedError, match="aggregated"):
        await router.begin_end_program("p1")

    assert "p1" in router._table.programs
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.TERMINATED


@pytest.mark.asyncio
async def test_worker_ids_for_program_distinguishes_unknown_from_unattributed():
    router, _ = make_router()

    assert await router.worker_ids_for_program("missing") is None
    await router.before_request("known")
    assert await router.worker_ids_for_program("known") == ()


@pytest.mark.asyncio
async def test_pause_acting_then_before_request_blocks_until_resume():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
    )
    router, _ = make_router(config=cfg)

    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.PAUSED

    waiter = asyncio.create_task(router.before_request("p1"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)

    async with router._lock:
        router._resume_program(router._table.programs["p1"], target_worker_id=1)

    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True
    assert decision.priority_jump == cfg.resume_priority_boost
    assert decision.assigned_worker_hint == 1


@pytest.mark.asyncio
async def test_end_program_wakes_paused_waiter_and_rejects_it():
    router, _ = make_router()
    await router.before_request("p1")
    await router.after_request("p1", prompt_tokens=100, completion_tokens=0)
    await router._pause_acting("p1")

    waiter = asyncio.create_task(router.before_request("p1"))
    await asyncio.sleep(0)
    worker_ids = await router.begin_end_program("p1")
    assert worker_ids == ()

    with pytest.raises(ProgramTerminatedError):
        await waiter
    assert router._table.programs["p1"].inflight_requests == 0

    assert await router.end_program("p1") is True
    with pytest.raises(ProgramTerminatedError):
        await router.before_request("p1")


@pytest.mark.asyncio
async def test_begin_end_waits_for_inflight_then_snapshots_latest_worker():
    router, _ = make_router()
    await router.before_request("p1")

    closing = asyncio.create_task(router.begin_end_program("p1"))
    await asyncio.sleep(0)
    assert not closing.done()

    # This attribution can arrive from the already-admitted streaming request
    # after the terminal notification started. It must be in the cleanup set.
    await router.assign_worker("p1", 17)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=1)
    assert await asyncio.wait_for(closing, timeout=1.0) == (17,)

    with pytest.raises(ProgramTerminatedError):
        await router.before_request("p1")


@pytest.mark.asyncio
async def test_cancelled_paused_admission_does_not_block_terminal_drain():
    router, _ = make_router()
    await router.before_request("p1")
    await router.after_request("p1", prompt_tokens=100, completion_tokens=0)
    await router._pause_acting("p1")

    waiter = asyncio.create_task(router.before_request("p1"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert router._table.programs["p1"].inflight_requests == 0
    assert await asyncio.wait_for(router.begin_end_program("p1"), timeout=1.0) == ()


@pytest.mark.asyncio
async def test_unknown_final_fails_closed_but_tombstoned_duplicate_is_noop():
    router, _ = make_router()
    with pytest.raises(ProgramNotFoundError):
        await router.begin_end_program("never-seen")

    await router.before_request("known")
    await router.after_request("known", prompt_tokens=1, completion_tokens=0)
    assert await router.begin_end_program("known") == ()
    assert await router.end_program("known") is True
    assert await router.begin_end_program("known") is None


@pytest.mark.asyncio
async def test_forced_resume_after_timeout():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=0.05,
    )
    router, _ = make_router(config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    decision = await router.before_request("p1")
    assert decision.was_paused is True
    assert router._stat_forced_resumes >= 1
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_new_program_queues_before_first_request_when_capacity_full():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=2.0,
        pause_threshold=1.0,
        resume_hysteresis=0.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("existing", estimated_prompt_tokens=950)
    await router.assign_worker("existing", 1)

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=100)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert router._table.programs["new"].lifecycle == ProgramLifecycle.PAUSED

    async with router._lock:
        router._resume_program(router._table.programs["new"], target_worker_id=1)
    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True


@pytest.mark.asyncio
async def test_cold_start_admits_without_sticky_pin():
    """No MDC visible yet: don't park, let the request through; the
    chunk-loop callback will populate ``assigned_worker_id`` once the
    engine picks a worker."""
    router, _ = make_router(capacity_workers={})
    decision = await router.before_request("cold_start")
    assert decision.was_paused is False
    assert decision.assigned_worker_hint is None
    program = router._table.programs["cold_start"]
    assert program.lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_soft_demote_marks_borderline_workers():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        soft_demote_threshold=0.80,
        pause_threshold=0.95,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=750, completion_tokens=0)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)

    router._apply_soft_demotes(router._capacity.snapshot())
    program = router._table.programs["p1"]
    assert program.soft_demoted_until > time.monotonic()

    await router.after_request("p1", prompt_tokens=860, completion_tokens=2)
    decision = await router.before_request("p1")
    assert decision.priority_jump == cfg.soft_demote_priority_jump
    assert decision.was_soft_demoted is True


@pytest.mark.asyncio
async def test_pause_until_safe_pauses_smallest_acting_first():
    cfg = ThunderAgentConfig(
        pause_threshold=0.80,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    # Used = 600 + 100 + 2*100 = 900; pausing small leaves 700 <= target.
    for pid, prompt_tokens in [("big", 600), ("small", 100)]:
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_until_safe_is_scoped_to_overloaded_worker():
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
        2: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    for pid, worker_id, prompt_tokens in [
        ("hot_big", 1, 700),
        ("hot_small", 1, 200),
        ("cold", 2, 700),
    ]:
        await router.before_request(pid)
        await router.assign_worker(pid, worker_id)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["hot_small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["hot_big"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["cold"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_drives_util_to_pause_target_not_threshold():
    """Each pause cycle drains util down to pause_target, not just below threshold."""
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1_000_000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100_000, completion_tokens=0)

    await router._pause_until_safe(router._capacity.snapshot())

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    # 10 programs * (100k tokens + 100 buffer) = 1.0010M; target 0.80M.
    # Each pause releases (100k + 100). Pause 2 -> 0.8008M (still over),
    # pause 3 -> 0.7007M (under). Anything else means over- or under-shoot.
    assert paused == 3, f"paused={paused}"


@pytest.mark.asyncio
async def test_scheduler_tick_resumes_before_pausing_new_overload():
    """Upstream TA ordering: resume old paused work, then pause overload."""
    cfg = ThunderAgentConfig(
        pause_threshold=1.0,
        pause_target=0.80,
        resume_hysteresis=0.0,
        acting_token_weight=1.0,
        acting_decay_tau_seconds=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, capacity = make_router(config=cfg)

    # Capacity is attached after setup so first-turn admission gating does not
    # queue the synthetic programs before the scheduler tick.
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100, completion_tokens=0)
        router._table.programs[pid].acting_since = time.monotonic() - 10.0

    capacity.workers = workers
    await router._scheduler_tick()

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    assert paused == 6


# ---- value gate: pause picks least valuable, not smallest -----------------


async def add_program(
    router: ThunderAgentScheduler,
    pid: str,
    *,
    tokens: int,
    worker_id: Optional[int] = None,
    turns: int = 1,
    parent: Optional[str] = None,
) -> None:
    """Drive a program to ACTING with ``turns`` completed requests.

    Programs are built against an EMPTY capacity snapshot (the provider is
    populated, or the snapshot passed in, only once the table is set up):
    admission would otherwise queue the later ones on a worker the earlier ones
    have already filled, and ``before_request`` would block on the resume
    timeout instead of returning.
    """
    for _ in range(turns):
        await router.before_request(pid, parent_program_id=parent)
        if worker_id is not None:
            await router.assign_worker(pid, worker_id)
        await router.after_request(pid, prompt_tokens=tokens, completion_tokens=0)


def _pause_cfg(**kw) -> ThunderAgentConfig:
    return ThunderAgentConfig(
        pause_threshold=0.80,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
        **kw,
    )


@pytest.mark.asyncio
async def test_value_gate_picks_least_valuable_not_smallest():
    """``small`` is a third of ``big`` but has ten turns behind it, so its
    bytes have been re-read nine more times: pause-smallest would evict it
    precisely because it is small; the value gate evicts ``big`` instead."""
    workers = {1: 1000}

    router, _ = make_router(config=_pause_cfg())
    await add_program(router, "big", tokens=600, worker_id=1, turns=1)
    await add_program(router, "small", tokens=300, worker_id=1, turns=10)
    await router._pause_until_safe(workers)
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["small"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_spares_the_parent_of_live_sub_agents():
    """A blocked parent is the worst thing to pause: it will be re-entered the
    moment its children return, and they were forked from its context."""
    workers = {1: 1000}
    router, _ = make_router(config=_pause_cfg())

    await add_program(router, "big", tokens=600, worker_id=1, turns=1)
    await add_program(router, "parent", tokens=300, worker_id=1, turns=1)
    for i in range(4):
        await add_program(router, f"child{i}", tokens=10, parent="parent")

    await router._pause_until_safe(workers)

    assert router._table.programs["big"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["parent"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_holder_weight_zero_reverts_to_size_times_turns():
    """Ablation knob: with the holder term off, the same fan-out no longer
    protects the parent, so the choice falls back to the smaller program."""
    workers = {1: 1000}
    router, _ = make_router(config=_pause_cfg(holder_weight=0.0))

    await add_program(router, "big", tokens=600, worker_id=1, turns=1)
    await add_program(router, "parent", tokens=300, worker_id=1, turns=1)
    for i in range(4):
        await add_program(router, f"child{i}", tokens=10, parent="parent")

    await router._pause_until_safe(workers)

    assert router._table.programs["parent"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


# ---- victim size weight: recoverability -----------------------------------


@pytest.mark.asyncio
async def test_size_weight_stops_value_from_picking_the_largest_victim():
    """The pure-value gate pauses ``big`` and then cannot fit it back.

    Value is deliberately inverted against size here, which is what the engine
    actually reports: it scores a large low-reuse working set most negative. With
    the size weight on, the victim ordering flips to the small program, which is
    the one admission control can actually re-grant.
    """
    workers = {1: 1000}

    pure, _ = make_router(config=_pause_cfg())
    await add_program(pure, "big", tokens=600, worker_id=1, turns=1)
    await add_program(pure, "small", tokens=300, worker_id=1, turns=10)
    await pure._pause_until_safe(workers)
    assert pure._table.programs["big"].lifecycle == ProgramLifecycle.PAUSED

    weighted, _ = make_router(config=_pause_cfg(victim_size_weight=4.0))
    await add_program(weighted, "big", tokens=600, worker_id=1, turns=1)
    await add_program(weighted, "small", tokens=300, worker_id=1, turns=10)
    await weighted._pause_until_safe(workers)
    assert weighted._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert weighted._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_size_weight_zero_is_the_pure_value_key():
    """The knob's low end has to be the policy that was measured, exactly."""
    router, _ = make_router(config=_pause_cfg())
    await add_program(router, "big", tokens=600, turns=1)
    await add_program(router, "small", tokens=300, turns=10)

    for pid in ("big", "small"):
        program = router._table.programs[pid]
        assert router._pause_victim_key(program) == router._program_value(program)


@pytest.mark.asyncio
async def test_equal_programs_share_a_rank():
    """Otherwise insertion order silently becomes a tiebreaker in the key."""
    router, _ = make_router(config=_pause_cfg(victim_size_weight=1.0))
    await add_program(router, "first", tokens=300, turns=2)
    await add_program(router, "second", tokens=300, turns=2)

    keys = router._blended_victim_keys()
    assert keys["first"] == keys["second"]


@pytest.mark.asyncio
async def test_ranks_expire_when_the_engine_scores_change():
    """The ranks are memoised on the table's shape, which a new dump does not
    change -- so a score refresh has to invalidate them on its own or the gate
    keeps ordering by a stale dump."""
    # Not 1.0: at w=1 value and size carry equal weight, so a pair whose value
    # order is the exact reverse of its size order cancels to a tie (see
    # test_weight_one_cancels_perfectly_anti_correlated_inputs).
    router, _ = make_router(config=_pause_cfg(victim_size_weight=0.5))
    await add_program(router, "a", tokens=600, turns=1)
    await add_program(router, "b", tokens=300, turns=1)

    router._engine_scores = {"a": 10.0, "b": -10.0}
    router._scores_generation += 1
    before = dict(router._blended_victim_keys())
    assert before["a"] > before["b"]

    router._engine_scores = {"a": -10.0, "b": 10.0}
    router._scores_generation += 1
    after = router._blended_victim_keys()
    assert after["a"] < after["b"]


@pytest.mark.asyncio
async def test_weight_one_cancels_perfectly_anti_correlated_inputs():
    """w=1 is a degenerate point, and on this workload it is the likely one.

    Equal weight on both ranks means a program that is last by value and first by
    size scores the same as its mirror image, so the key carries no information
    and the ordering falls back to whatever the pairwise scan saw first. That is
    not hypothetical here: the engine scores large working sets most negative, so
    value is anti-correlated with size by construction. Pick a weight either side
    of 1.
    """
    router, _ = make_router(config=_pause_cfg(victim_size_weight=1.0))
    await add_program(router, "a", tokens=600, turns=1)
    await add_program(router, "b", tokens=300, turns=1)
    router._engine_scores = {"a": -10.0, "b": 10.0}
    router._scores_generation += 1

    keys = router._blended_victim_keys()
    assert keys["a"] == keys["b"]


# ---- recoverability weight: penalise against the ceiling, not the table ----


@pytest.mark.asyncio
async def test_recoverability_weight_ignores_victims_that_fit_regardless_of_size():
    """Size-rank compares a victim against whoever else happens to be in the
    table, so two programs that both comfortably fit under the resume ceiling
    still get spread 0..1 apart purely because one is 3x the other. That
    penalises a victim for nothing: it was never going to strand either way.
    Recoverability compares against the ceiling itself, so both score zero --
    and reserves the penalty for the one that actually cannot come back.
    """
    router, _ = make_router(config=_pause_cfg(victim_recoverability_weight=2.0))
    await add_program(router, "a", tokens=100, turns=1)
    await add_program(router, "b", tokens=300, turns=1)
    await add_program(router, "huge", tokens=5000, turns=1)
    # Equalise value so only the size/recoverability term can separate them.
    router._engine_scores = {"a": 0.0, "b": 0.0, "huge": 0.0}
    router._scores_generation += 1
    # As a real tick would: capacity=1000, pause_target=0.80 -> ceiling=800.
    router._pause_target_ceiling = 800.0

    keys = router._blended_victim_keys()
    assert keys["a"] == keys["b"]
    assert keys["huge"] > keys["a"]

    # The same table under the size-rank term alone WOULD separate a and b,
    # which is the behaviour recoverability is meant to not have.
    size_only, _ = make_router(config=_pause_cfg(victim_size_weight=2.0))
    await add_program(size_only, "a", tokens=100, turns=1)
    await add_program(size_only, "b", tokens=300, turns=1)
    await add_program(size_only, "huge", tokens=5000, turns=1)
    size_only._engine_scores = {"a": 0.0, "b": 0.0, "huge": 0.0}
    size_only._scores_generation += 1
    size_keys = size_only._blended_victim_keys()
    assert size_keys["a"] != size_keys["b"]


@pytest.mark.asyncio
async def test_recoverability_weight_zero_is_unaffected():
    """Default off: identical to the pure-value key, same as size_weight=0."""
    router, _ = make_router(config=_pause_cfg())
    await add_program(router, "a", tokens=100, turns=1)
    await add_program(router, "huge", tokens=5000, turns=1)
    router._pause_target_ceiling = 800.0

    for pid in ("a", "huge"):
        program = router._table.programs[pid]
        assert router._pause_victim_key(program) == router._program_value(program)


@pytest.mark.asyncio
async def test_recoverability_weight_before_any_tick_is_a_no_op():
    """No capacity snapshot has been seen yet -> the ceiling is unknown, so
    the term must not fire on a made-up ceiling (e.g. zero, which would flag
    everything as unrecoverable)."""
    router, _ = make_router(config=_pause_cfg(victim_recoverability_weight=2.0))
    await add_program(router, "a", tokens=100, turns=1)
    await add_program(router, "huge", tokens=5000, turns=1)
    assert router._pause_target_ceiling is None

    huge = router._table.programs["huge"]
    assert router._recoverability_penalty(huge) == 0.0


@pytest.mark.asyncio
async def test_recoverability_and_size_weight_combine():
    """The two terms are additive, not either/or: a victim can be penalised
    for being merely the biggest program present (size) AND for exceeding the
    ceiling outright (recoverability) at the same time."""
    router, _ = make_router(
        config=_pause_cfg(victim_size_weight=1.0, victim_recoverability_weight=1.0)
    )
    await add_program(router, "a", tokens=100, turns=1)
    await add_program(router, "huge", tokens=5000, turns=1)
    router._pause_target_ceiling = 800.0

    keys = router._blended_victim_keys()
    size_only, _ = make_router(config=_pause_cfg(victim_size_weight=1.0))
    await add_program(size_only, "a", tokens=100, turns=1)
    await add_program(size_only, "huge", tokens=5000, turns=1)
    size_only_keys = size_only._blended_victim_keys()

    # Adding the recoverability term on top can only push "huge" further
    # away from "a" (both blends already agree "huge" is worse).
    assert keys["huge"] - keys["a"] > size_only_keys["huge"] - size_only_keys["a"]


@pytest.mark.asyncio
async def test_pause_until_safe_populates_the_ceiling_from_live_capacity():
    """``_pause_target_ceiling`` has to come from a real tick's capacity
    snapshot (``_pause_until_safe`` is also how ``_scheduler_tick`` reaches
    it), not just be settable by a test writing to the private field."""
    router, _ = make_router(config=_pause_cfg())
    assert router._pause_target_ceiling is None
    await router._pause_until_safe({1: 1000, 2: 2000})
    # Biggest worker's ceiling, not the average -- a paused program can
    # resume onto whichever worker BFD picks.
    assert router._pause_target_ceiling == 2000 * 0.80


# ---- resume: default smallest-first, value-ordered opt-in ------------------


def _resume_cfg(**kw) -> ThunderAgentConfig:
    return ThunderAgentConfig(
        pause_threshold=1.0,
        pause_target=0.80,
        resume_hysteresis=0.0,
        scheduler_interval_seconds=10.0,
        **kw,
    )


async def _two_paused_programs(router: ThunderAgentScheduler) -> None:
    """``cheap`` and ``rich`` are the same size; only ``rich`` has sub-agents.

    Same structural group on purpose (same turn count, both ACTING): the
    inherited group pre-sort outranks the ordering key, so the key only reorders
    within a group. ``cheap`` is inserted first, which is what a stable size sort
    picks between equals.
    """
    await add_program(router, "cheap", tokens=600, worker_id=1, turns=2)
    await add_program(router, "rich", tokens=600, worker_id=1, turns=2)
    for i in range(3):
        await add_program(router, f"child{i}", tokens=10, parent="rich")
    for pid in ("cheap", "rich"):
        assert await router._pause_acting(pid)


@pytest.mark.asyncio
async def test_resume_is_not_value_ordered_by_default():
    """Pause and resume are different decisions and must not share a key.

    Resume is a queue: the ceiling admits a fixed token budget per tick, so
    smallest-first readmits the most programs and drains the backlog fastest.
    Value-ordering it readmits the largest (value scales with working set), the
    backlog stalls, and the forced-resume timer becomes the way out -- measured
    at 56.7 tok/s against the baseline's 144.7. So the default stays size-ordered
    even though the pause victim is still value-chosen.
    """
    router, _ = make_router(config=_resume_cfg())
    await _two_paused_programs(router)

    for pid in ("cheap", "rich"):
        program = router._table.programs[pid]
        assert router._resume_selection_key(program) == float(program.token_total)
    # The pause side is still ours: rich outscores cheap at identical size.
    rich = router._table.programs["rich"]
    cheap = router._table.programs["cheap"]
    assert router._pause_victim_key(rich) > router._pause_victim_key(cheap)


@pytest.mark.asyncio
async def test_value_ordered_resume_admits_most_valuable_first():
    """The opt-in ablation, kept so its cost stays measurable."""
    router, _ = make_router(config=_resume_cfg(value_ordered_resume=True))
    await _two_paused_programs(router)

    # 1000 admits one program (600 + 100 buffer), never both.
    await router._greedy_resume({1: 1000})

    assert router._table.programs["rich"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["cheap"].lifecycle == ProgramLifecycle.PAUSED


# ---- holder bookkeeping -----------------------------------------------------


@pytest.mark.asyncio
async def test_holders_count_only_live_children():
    router, _ = make_router()
    await add_program(router, "parent", tokens=100)
    await add_program(router, "a", tokens=10, parent="parent")
    await add_program(router, "b", tokens=10, parent="parent")
    assert router._n_holders("parent") == 3.0

    await router.end_program("a")
    assert router._n_holders("parent") == 2.0

    await router.end_program("b")
    assert router._n_holders("parent") == 1.0
    # The reverse index is dropped with the last child, not left to grow.
    assert "parent" not in router._children


@pytest.mark.asyncio
async def test_repeated_turns_do_not_inflate_holder_count():
    """Every turn of a sub-agent re-sends parent_session_id; the edge is a set."""
    router, _ = make_router()
    await add_program(router, "parent", tokens=100)
    await add_program(router, "a", tokens=10, parent="parent", turns=5)
    assert router._n_holders("parent") == 2.0


@pytest.mark.asyncio
async def test_self_parent_edge_ignored():
    router, _ = make_router()
    await add_program(router, "p", tokens=100, parent="p")
    assert router._n_holders("p") == 1.0


# ---- engine state as the value source ---------------------------------------


@pytest.mark.asyncio
async def test_engine_scores_override_proxy_per_program():
    router, _ = make_router()
    await add_program(router, "a", tokens=100, turns=3)
    await add_program(router, "b", tokens=100, turns=3)
    proxy_b = router._program_value(router._table.programs["b"])

    router._engine_scores = {"a": 42.0}

    assert router._program_value(router._table.programs["a"]) == 42.0
    # A program the dump does not mention keeps the proxy rather than 0.
    assert router._program_value(router._table.programs["b"]) == proxy_b


@pytest.mark.asyncio
async def test_refresh_is_noop_without_state_url():
    """Do-no-harm: no URL means no client, no fetch, and the proxy stands."""
    router, _ = make_router()
    assert router._engine_state.enabled is False
    router.start()
    try:
        assert router._http is None
        await router._refresh_engine_scores()
        assert router._engine_scores is None
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_refresh_picks_up_engine_scores_then_degrades():
    router, _ = make_router(
        config=ThunderAgentConfig(
            scheduler_interval_seconds=10.0,
            state_url="http://127.0.0.1:1/aginfer/state",
        )
    )
    router._http = object()  # only truthiness matters; fetch is stubbed below

    scores: Optional[dict] = {"a": 7.0}

    async def fake_fetch(_client):
        return "state" if scores is not None else None

    router._engine_state.fetch = fake_fetch  # type: ignore[method-assign]
    router._engine_state.program_scores = lambda _state: scores  # type: ignore[method-assign]

    await router._refresh_engine_scores()
    assert router._engine_scores == {"a": 7.0}

    scores = None
    await router._refresh_engine_scores()
    assert router._engine_scores is None


@pytest.mark.asyncio
async def test_disjoint_id_spaces_are_reported_once(caplog):
    """Scores that match no session id mean the value gate is silently on the
    proxy for the whole fleet -- that must not be invisible."""
    router, _ = make_router(
        config=ThunderAgentConfig(
            scheduler_interval_seconds=10.0,
            state_url="http://127.0.0.1:1/aginfer/state",
        )
    )
    router._http = object()
    await add_program(router, "sess#salt", tokens=100)

    async def fake_fetch(_client):
        return "state"

    router._engine_state.fetch = fake_fetch  # type: ignore[method-assign]
    router._engine_state.program_scores = lambda _s: {"unsalted-pid": 1.0}  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        await router._refresh_engine_scores()
        await router._refresh_engine_scores()

    warnings = [r for r in caplog.records if "none match the" in r.getMessage()]
    assert len(warnings) == 1, "expected exactly one mismatch warning, got %d" % len(
        warnings
    )
    # Still serving values, just not the engine's.
    assert router._program_value(router._table.programs["sess#salt"]) == 200.0
