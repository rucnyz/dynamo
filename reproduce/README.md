# aginfer reproduce — experiment packages

Each subdirectory is a self-contained experiment. All use the same 5-arm design:

| Arm | Router | Engine eviction | How enabled |
|---|---|---|---|
| **B** | default | LRU | nothing extra |
| **TA** | ThunderAgent | LRU | `dynamo.thunderagent_router` |
| **Ours-evict** | default | value | sglang: `SGLANG_AGINFER_IN_ENGINE=1`; vllm: `VLLM_AGINFER_VALUE_EVICTION=1` |
| **TA+Ours-evict** | ThunderAgent | value | TA router + value env |
| **Ours-full** | our router | value | our router + value env |

## Experiments

| Dir | Backend | Tiers | What |
|---|---|---|---|
| `E1-hicache-nixl/` | sglang + HiCache + nixl | HBM→DRAM→SSD→DROP | Headline win (4-tier) |
| `E2-hicache-2tier/` | sglang + HiCache (no backend) | HBM→DRAM→DROP | Tier ablation |
| `E3-hicache-mooncake/` | sglang + HiCache + mooncake | HBM→DRAM→SSD→DROP | Transport backend |
| `F1-kvbm-vllm/` | vLLM + KVBM | HBM→CPU→DISK→DROP | Engine-agnostic (Dynamo-only PR) |

## Data

Each experiment's `data/` contains agentreplay traces generated via:
```bash
python -m agentreplay convert \
  --tokenizer <model> --max-turns 4 --min-turns 2 --max-prompt-tokens 32000 \
  --out data/<trace>.jsonl
```
Traces are real Claude-Code trajectories, not synthetic.

## How to run

```bash
# 1. Start Dynamo stack (see each experiment's README.md)
# 2. For each arm: set env, restart worker, run replay
python -m agentreplay replay-dynamo --trace data/<trace>.jsonl --label <arm> --out results/<arm>.json
# 3. Compare
python -m agentreplay report --ours results/ours*.json --base results/baseline*.json
```

N ≥ 3 per arm. Do-no-harm = ours ≤ baseline in every metric.
