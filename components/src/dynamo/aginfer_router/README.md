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

---

## #251 step 5 — value-aware admission (engine-state ingestion)

The verified #251 split puts MIGRATION in the engine and ADMISSION (pause/resume) HERE at the
router. Today this router's pause decision uses a token-size proxy (`_program_value`); step 5
makes it value-aware by reading the engine's real value state — **reusing the in-engine
`build_paper_state` + `admission_controller` directly** (the router runs with the sglang fork on
its PYTHONPATH; single source, no copy). Confirmed feasible: the router CAN import
`sglang.srt.mem_cache.aginfer.{state_builder,admission_controller}`.

Ordered plan (foundation → live-path; the high-risk steps need a stabilized sglang-backed stack):
1. **`engine_state.py` — DONE (increment 1, 10 server-free tests).** Async GET `/aginfer/state`
   → the engine's real `SchedulerState` via in-engine `build_paper_state` (synthetic
   MEMORY_PRESSURE event). DO-NO-HARM: `None` when no `--aginfer-state-url` / unreachable /
   non-sglang backend ⇒ router falls back to the size proxy. Tick-cached (NOT per-request — the
   dump is 5-50ms, #160).
2. wire a `--aginfer-state-url` arg (`args.py`) + a tick-driven snapshot refresh in the scheduler
   loop (the cached `SchedulerState`).
3. swap `_program_value` → in-engine `admission_controller.shared_aware_prog_scores` (fleet,
   holder-split) when state present; size proxy when absent. Replace watermark pause-victim
   selection with `pause_candidates` (cost vs shadow-price `pause_relief`). **high risk.**
4. gate the ingress `asyncio.Event` on the engine's `per_program_usage[pid].state == PAUSED`
   (not just router-local lifecycle); port `_gated_count` + resume-in-flight dedup (#215) +
   ended-while-gated 499 (#183). **high risk — distributed state authority / lag.**
5. SESSION_END / disconnect on the gate; expose theta_hi/lo/heartbeat for `capacity_fits`/`forecast`.

Hard problems (see the step-5 map): backend must be sglang (vLLM has no `/aginfer/state`);
router↔engine state-lag authority (avoid double-pause/strand); `pause_relief` needs the engine's
D_t (conservative `exclude={}` until plumbed); pause-thrash on stale state (regime-dependent win,
do-no-harm floor is not) ⇒ the value path must be provably no-worse-than the size baseline when
state is stale/absent. Live goodput-vs-ThunderAgent A/B (moderate concurrency, N≥3 paired) runs
only after the unit layer is green AND on a stabilized stack (V4 is crash-prone under flood).
