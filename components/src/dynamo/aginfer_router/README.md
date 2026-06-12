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

Pause/resume is opt-in per request via `nvext.agent_context.trajectory_id`
(requests without it route through a plain `KvRouter`).

## Why this lives here

aginfer is a program-aware, value-driven KV **scheduler** that currently runs as an
external daemon + in-engine eviction scorer on sglang HiCache + mooncake. Dynamo is the
one stack that natively carries program/session identity (`nvext.agent_context`) at the
orchestrator **and** owns a multi-tier KV substrate -- so aginfer's orchestration-side
levers (value-gated pause/resume, worker placement, predictive promote) attach here as an
**additive component**, exactly the pattern ThunderAgent established. The in-cache
value-eviction lever stays engine-side (sglang scorer), forwarded via the priority scalar.

---

Derived from `dynamo.thunderagent_router` (Apache-2.0, NVIDIA) — the pause/resume/BFD
machinery is theirs; the value gate (`_program_value` + the ordering swaps) is aginfer's.
