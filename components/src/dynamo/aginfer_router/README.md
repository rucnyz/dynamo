# aginfer_router — value-gated, program-aware KV router

A drop-in **peer** of `dynamo.thunderagent_router`. It keeps ThunderAgent's exact
trajectory-grouped pause / resume / BFD-restore machinery (plus SESSION_END /
Dead-KV cleanup) and changes **two ordering keys**: which program is sacrificed
under pressure, and (opt-in) which paused program comes back first, are chosen
by per-program **value** `V_u` instead of ThunderAgent's token working-set
**size**.

| decision | ThunderAgent (baseline) | aginfer (this) |
|---|---|---|
| which program to **pause** under pressure | smallest `token_total` first | **lowest `V_u` first** (`victim_size_weight`-blended, see below) |
| which paused program to **resume** | smallest `token_total` first | same by default; `--aginfer-value-ordered-resume` opts into highest-`V_u`-first |
| `priority_jump` (sticky pin + downstream eviction priority) | resume boost / soft-demote | unchanged |

Nothing else differs, and that is enforced structurally: the value gate lives
entirely behind `_pause_victim_key` / `_resume_selection_key` /
`_program_value` in `router.py`. The watermarks, the soft-demote band, the BFD
placement pass, and the forced-resume timeout are untouched, so a difference
in a paired run against `thunderagent_router` is the scheduling policy and
cannot be the skeleton.

### Why resume is *not* value-ordered by default

Pause and resume look symmetric and are not, so they do not share a key:

- **Pause is a sacrifice.** Something has to go; give up the KV worth least.
  Value belongs here.
- **Resume is a queue.** The ceiling admits a fixed token budget per tick, so
  smallest-first readmits the most programs per tick and drains the backlog
  fastest. Value belongs nowhere near it: `V_u` has working set as a factor, so
  the most valuable program is usually also the largest. Ordering resumes by
  value readmits the biggest first, each tick clears fewer programs, the
  backlog stops moving, and the forced-resume timeout ends up being the main
  way out.

That is measured, not assumed. On the 600-request Claude Code slice under
manufactured pressure, value-ordered resume ran **56.7 tok/s against the
baseline's 144.7** (and a no-admission control's 285.5), with 11 of its 21
resumes fired by the timer rather than by capacity — see
`benchmark/dynamo/README.md`. `--aginfer-value-ordered-resume` turns it back
on, purely to keep that cost reproducible.

> The placement pass likewise stays size-descending: it is a bin-packing
> heuristic, not a policy. Value decides *who* gets sacrificed; packing decides
> *where* the survivors land.

### Value alone picks victims it cannot get back

Pure value (`--aginfer-victim-size-weight 0`, the default) loses to the
size-ordered baseline by 2.35x on the Claude Code slice: it pauses 4x as
often, each pause lasts 3.8x longer, and 42 of 100 resumes come from the
forced-resume timer rather than from freed capacity. The engine scores a large
low-reuse working set most negative, so pure value systematically picks the
*largest* programs — and a large victim does not fit back under the resume
ceiling. It strands until the timer releases it, is still the least valuable,
and is paused again.

`V(u) = p_reuse x reload_cost - holding_cost` answers "what is worth least" and
says nothing about "can this be taken back". `--aginfer-victim-size-weight w`
supplies the missing half:

```text
victim key = value_rank + w * size_rank      (lowest is paused first)
```

Normalised ranks, not raw numbers, because value and size share no unit —
engine scores land in the tens while the proxy lands in the millions, so any
fixed coefficient between them would be meaningless in one regime or the
other. Ranks also make the two arms endpoints of a single knob: `w = 0` is the
pure value gate, a large `w` reproduces the baseline's pause-smallest.

**Do not use `w = 1` exactly.** Equal weight on the two ranks means a program
that is last by value and first by size scores the same as its mirror image,
and here that cancellation is the common case rather than a corner, precisely
because the engine's scores are anti-correlated with size. Sweep either side
of 1.

## Where `V_u` comes from

Best signal available wins, per program:

1. **Engine value state** — `--aginfer-state-url` points either at an sglang
   worker's `/aginfer/state` or, in a Dynamo stack where sglang's HTTP surface
   is not served, at the worker's `POST /engine/call_tokenizer_manager`
   passthrough (`DYN_SYSTEM_PORT` + `--enable-rl`); the transport is inferred
   from the path. The dump is turned into the engine's own `SchedulerState` and
   scored with the in-engine `shared_aware_prog_scores`, which splits each
   cache block's value across the sessions holding it — the real
   shared-prefix signal, and the engine's own number rather than a
   re-derivation of it.

   This path depends on the worker forwarding `agent_context.session_id` to
   `async_generate(program_id=...)` — that is what tags radix nodes with their
   holders. Without it the dump's `per_program_usage` is empty and the client
   correctly reports no engine state.
2. **Router-local proxy** — `working set x turns taken x holders`:
   * *working set* — the re-prefill cost if this KV is dropped. What
     ThunderAgent uses, and all it has.
   * *turns taken* — `1 + step_count`. A program on its tenth turn has already
     re-read its prefix nine times; the same bytes have earned more than a
     fresh program's.
   * *holders* — `1 + w x live sub-agents`, from `agent_context.parent_session_id`.
     A parent blocked on live children is the worst thing to pause: it will be
     re-entered the moment they return, and they were forked from its context.
     Because the parent is usually the *larger* program, size-ordered restore
     puts it last. `w` is `--aginfer-holder-weight`; `0` removes the term
     (ablation).

The proxy is not a placeholder for the dump: the aginfer state surface only
answers on a worker with `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, so the proxy is
the operating point for every other backend. The dump is fetched once per
scheduler tick (5-50ms under load, never on request ingress) and any failure —
no URL, unreachable, non-sglang backend, unsupported tree cache — silently
leaves the proxy in charge. The value path can therefore be no worse than the
ThunderAgent baseline; it can only refine the ordering when real state is
there.

## Usage

Same flag set as `thunderagent_router` (its own arg group, kept in parity by
hand rather than imported — see `args.py`), plus the value-gate flags above:

```bash
python -m dynamo.aginfer_router \
    --endpoint dynamo.sglang.generate \
    --model-name <model> \
    --router-block-size 64 \
    [--aginfer-state-url http://127.0.0.1:30000/aginfer/state] \
    [--aginfer-holder-weight 1.0] \
    [--aginfer-victim-size-weight 0.0] \
    [--aginfer-value-ordered-resume]
```

Serves `{namespace}.aginfer_router.generate`, so it can run alongside the
ThunderAgent baseline in one namespace for an A/B. Pause/resume is opt-in per
request via `agent_context.session_id`; requests without it are routed via
plain `KvRouter` with no lifecycle. Legacy replay producers using
`trajectory_id` / `trajectory_final` are accepted at the boundary and
normalized to Dynamo's canonical `session_id` / `session_final` fields.

Terminal requests (`session_final=true` or `kv_hints.evict_session=true`) are
sent to the dedicated sibling worker endpoint `end_program`. In aggregated
serving, the router directly targets every SGLang worker previously
attributed to the program and releases its local mapping only after all
workers acknowledge cleanup. The required SGLang contract is an idempotent
`Engine.async_end_program(program_id)` method. The Dynamo worker also supports
a legacy synchronous `Engine.end_program(program_id)` fallback, executed
outside the active asyncio loop.

This explicit Dead-KV reclamation path currently requires an aggregated,
aginfer-enabled SGLang backend. Other backends can still use the value-aware
scheduling path, but a terminal request fails closed if their worker does not
expose `end_program`. Disaggregated prefill/decode cleanup is also
fail-closed: the router retains the mapping when it observes distinct worker
components, because Dynamo direct dispatch is endpoint-scoped.

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

## Tests

```bash
cd components/src/dynamo
python -m pytest aginfer_router/tests thunderagent_router/tests -q
```

Each value-gate test is built so the **baseline would make the opposite
choice**.

## Why this lives here

Dynamo owns request/session identity and worker placement, so it is the only
layer that can identify every worker that may hold a program's KV, and it is
the one stack that natively carries program/session identity (`AgentContext`)
at the orchestrator. SGLang owns the physical HBM and host-cache allocations,
so it performs the actual leaf-to-root reclamation and the in-cache
value-eviction lever. Keeping the two responsibilities explicit avoids
treating a fake inference request as a lifecycle signal and gives the router
a concrete acknowledgement before it forgets the mapping.

The value-aware pause/resume policy remains an additive scheduling layer on
top of Dynamo's native KV router. The engine-side cache policy remains in the
aginfer SGLang fork and receives the stable `program_id` forwarded by this
component.

---

Derived from `dynamo.thunderagent_router` (Apache-2.0, NVIDIA) — the pause/resume/BFD
machinery is theirs; the value gate (`_program_value`, `_pause_victim_key`,
`_resume_selection_key` and the holder graph) is aginfer's.
