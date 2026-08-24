# aginfer_router — value-gated, program-aware KV router

A drop-in **peer** of `dynamo.thunderagent_router`. It keeps ThunderAgent's exact
trajectory-grouped pause / resume / BFD-restore machinery, and changes **one thing**:
the ordering key for pause/resume/priority is a per-program **value** `V_u` instead of
ThunderAgent's token-working-set **size**.

This makes it a clean A/B ablation against the (unchanged) ThunderAgent baseline:
same skeleton, only the gate differs.

## What changes vs ThunderAgent

| decision | ThunderAgent (baseline) | aginfer (this) |
|---|---|---|
| which program to **pause** under pressure | smallest `token_total` first | **lowest `V_u` first** |
| which paused program to **resume** | BFD by `token_total` | **highest `V_u` first** |
| `priority_jump` (sticky pin + downstream eviction priority) | resume boost / soft-demote | scaled by value |

`_program_value` (see `router.py`):

```
V_u(program) = token_total * reuse_weight * n_holders
  token_total  — KV at stake (re-prefill cost if dropped)
  reuse_weight — 1 + step_count  (an established multi-turn agent has reused its
                 prefix repeatedly -> worth more than a fresh program of equal size;
                 the program-aware signal ThunderAgent lacks)
  n_holders    — shared-prefix holder count (a prefix shared by N trajectories is
                 worth Nx keeping; v0 default 1)
```

**v1 TODO** (the full DESIGN-7 V_u): fold in ETA reuse-imminence (a program *parked in
a tool-call gap* is will-resume = high value, not low-priority) and KV-event-derived
`n_holders`. These are the levers where program-awareness beats recency/size.

## Usage

Identical CLI to `thunderagent_router` (it shares `args.py`):

```bash
python -m dynamo.aginfer_router \
    --endpoint dynamo.sglang.generate \
    --model-name <model> \
    --router-block-size 64
```

Pause/resume is opt-in per request via `agent_context.session_id` (requests
without it route through a plain `KvRouter`).  Legacy replay producers using
`trajectory_id` / `trajectory_final` are accepted at the aginfer boundary and
normalized to Dynamo's canonical `session_id` / `session_final` fields.

Terminal requests (`session_final=true` or
`kv_hints.evict_session=true`) are sent to the dedicated sibling worker
endpoint `end_program`. In aggregated serving, the router directly targets
every SGLang worker previously attributed to the program and releases its
local mapping only after all workers acknowledge cleanup. The required
SGLang contract is an idempotent `Engine.async_end_program(program_id)` method.
The Dynamo worker also supports a legacy synchronous
`Engine.end_program(program_id)` fallback, executed outside the active asyncio
loop.

This explicit Dead-KV reclamation path currently requires an aggregated,
aginfer-enabled SGLang backend. Other backends can still use the value-aware
scheduling path, but a terminal request fails closed if their worker does not
expose `end_program`. Disaggregated prefill/decode cleanup is also fail-closed:
the router retains the mapping when it observes distinct worker components,
because Dynamo direct dispatch is endpoint-scoped.

## Dead-KV lifecycle

The full lifecycle is split across Dynamo and SGLang:

1. Dynamo normalizes legacy trajectory metadata to the canonical session fields and
   forwards `session_id` to every SGLang generate path as `program_id`.
2. The router records every aggregated worker that may retain KV for the program.
   Historical workers remain in the cleanup set after migration.
3. A terminal request closes admission and waits for already-admitted requests to
   finish before taking the worker snapshot.
4. The router directly invokes each worker's `end_program` endpoint. Each worker waits
   for all SGLang ranks to report completed reclamation.
5. Only after every worker acknowledges cleanup does the router remove the sticky
   mapping. Partial failure retains the mapping so the terminal request can be retried.

Completed sessions are kept in a bounded, expiring tombstone table. Duplicate terminal
requests are idempotent, while new inference requests that reuse a recently ended
session ID fail closed.

## Why this lives here

Dynamo owns request/session identity and worker placement, so it is the only layer that
can identify every worker that may hold a program's KV. SGLang owns the physical HBM and
host-cache allocations, so it performs the actual leaf-to-root reclamation. Keeping the
two responsibilities explicit avoids treating a fake inference request as a lifecycle
signal and gives the router a concrete acknowledgement before it forgets the mapping.

The value-aware pause/resume policy remains an additive scheduling layer on top of
Dynamo's native KV router. The engine-side cache policy remains in the aginfer SGLang
fork and receives the stable `program_id` forwarded by this component.

---

Derived from `dynamo.thunderagent_router` (Apache-2.0, NVIDIA) — the pause/resume/BFD
machinery is theirs; the value gate (`_program_value` + the ordering swaps) is aginfer's.
