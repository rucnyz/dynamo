# F1: vLLM + KVBM (engine-agnostic, Dynamo-only PR path)

## Config
- **Backend**: vLLM fork (`rucnyz/vllm@aginfer`, 7 commits)
- **Model**: DeepSeek-V4-Flash, tp=2
- **Tiers**: HBM (vLLM block_pool) + CPU/DISK (KVBM)
- **Value eviction**: HBM layer via `VLLM_AGINFER_VALUE_EVICTION=1` + KVBM offload layer (TODO: KVBM plugin)

## Launch
```bash
# vLLM with KVBM (aggregated)
DYN_KVBM_CPU_CACHE_GB=20 \
python -m dynamo.vllm \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}'
```

## Arms
| Arm | Extra env/config |
|---|---|
| B | (nothing) |
| TA | ThunderAgent router |
| Ours-evict | `VLLM_AGINFER_VALUE_EVICTION=1` |
| TA+Ours-evict | TA router + `VLLM_AGINFER_VALUE_EVICTION=1` |
| Ours-full | our router + `VLLM_AGINFER_VALUE_EVICTION=1` |

## Replay
```bash
python -m agentreplay replay-dynamo --trace data/cc_deepseek.jsonl --label <arm>
```

## Results
(pending)
