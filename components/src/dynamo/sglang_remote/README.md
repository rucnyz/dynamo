# dynamo.sglang_remote

Thin Dynamo worker that proxies to a **remote standalone**
`python -m sglang.launch_server` over HTTP (+ ZMQ KV events).

Registers the same endpoints as in-process `dynamo.sglang`:

- `{namespace}.backend.generate`
- `{namespace}.backend.end_program`
- `{namespace}.backend.clear_kv_blocks` (stub: returns error)

so `aginfer_router` / `thunderagent_router` and `agentreplay replay-dynamo`
need no code changes.

## Topology

```
agentreplay → {ns}.ROUTER.generate          (local)
           → {ns}.backend.generate          (local, this package)
           → HTTP REMOTE:30000/generate     (standalone sglang)
```

Control plane (`/aginfer/state`, `/aginfer/metrics`, `/aginfer/session_end`,
`/aginfer/events`) is reached **directly** by the router / replay harness via
`STATE_URL` / `METRICS_URL` — not through this proxy.

## Quick start

On the GPU host:

```bash
MODEL=/path/to/model TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AGINFER_IN_ENGINE=1 \
  bash benchmark/dynamo/run_sglang_server.sh
```

On the Dynamo host:

```bash
SGLANG_URL=http://<gpu-host>:30000 MODEL=/path/to/model \
  bash benchmark/dynamo/run_stack_remote.sh
```

Replay:

```bash
SGLANG_URL=http://<gpu-host>:30000 \
STATE_URL=$SGLANG_URL/aginfer/state \
METRICS_URL=$SGLANG_URL/aginfer/metrics \
SYSTEM_PORT= \
ROUTER=aginfer_router bash benchmark/dynamo/run_replay.sh
```

## Preserved capabilities

| Capability | How |
|---|---|
| Teacher forcing | `extra_args.forced_output_ids` → `sampling_params.custom_params` |
| `program_id` | `agent_context.session_id` → `/generate` top-level field |
| Belief events | fire-and-forget `PUT /aginfer/events` |
| SESSION_END | `{ns}.backend.end_program` → `POST /aginfer/session_end` |
| KvRouter prefix hits | ZMQ SUB to remote `--kv-events-config` |

## Out of scope

Disaggregated prefill/decode, LoRA load, weight update — refuse loudly.

## Tests

```bash
PYTHONPATH=dynamo/components/src \
  python -m pytest dynamo/components/src/dynamo/sglang_remote/tests \
  --noconftest -v
```
