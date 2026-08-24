<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dead-KV full-pipeline verifier

`verify_dead_kv_dynamo.py` checks the aggregated Dynamo-to-SGLang lifecycle path
against a live deployment. It uses only the Python standard library.

> [!WARNING]
> The verifier calls `flush_cache` before sending requests. Run it only against a
> dedicated test deployment; it destroys all cache contents on the selected worker.

## Prerequisites

- An aggregated SGLang worker built from the matching aginfer-enabled SGLang source.
- The worker system server must be available and the worker must start with
  `--enable-rl`, which exposes `call_tokenizer_manager` for state inspection.
- A Dynamo frontend and `dynamo.aginfer_router` using the same worker.
- No concurrent production or benchmark traffic on that deployment.

Disaggregated prefill/decode cleanup is not supported by this verifier or the current
router implementation. The router fails closed and retains its mapping if it observes
distinct prefill and decode workers.

## Run

```bash
python components/src/dynamo/aginfer_router/verification/verify_dead_kv_dynamo.py \
  --frontend-url http://127.0.0.1:8000 \
  --worker-url http://127.0.0.1:8081 \
  --model deadkv-e2e \
  --artifact-dir /tmp/deadkv-dynamo-e2e/artifacts
```

If the frontend requires authentication, set `DYNAMO_API_KEY` or pass `--api-key`.
The key is sent to the frontend but is not written to the artifacts.

The verifier creates two sessions with shared and exclusive prompt regions, then
checks that:

- ordinary requests retain both sessions' KV;
- ending session A removes its exclusive physical KV while preserving B and their
  shared prefix;
- a duplicate terminal request is idempotent;
- the surviving session still records a cache hit;
- ending session B removes its remaining KV; and
- the frontend and worker remain live after cleanup.

Each run writes raw HTTP metadata, parsed per-rank state snapshots, poll history, and
`summary.json` to a unique subdirectory of `--artifact-dir`. No generated artifacts or
mock data are checked into the repository.
