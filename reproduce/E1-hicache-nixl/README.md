# E1: sglang + HiCache + nixl (4-tier headline)

## Config
- **Backend**: sglang fork (`rucnyz/sglang@aginfer-synced`)
- **Model**: DeepSeek-V4-Flash, tp=2
- **Tiers**: HBM → DRAM → SSD → DROP (full 4-tier)

## Launch
```bash
python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --trust-remote-code \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-storage-backend nixl
```

## Arms
| Arm | Extra env/config |
|---|---|
| B | (nothing) |
| TA | ThunderAgent router |
| Ours-evict | `SGLANG_AGINFER_IN_ENGINE=1` |
| TA+Ours-evict | TA router + `SGLANG_AGINFER_IN_ENGINE=1` |
| Ours-full | our router + `SGLANG_AGINFER_IN_ENGINE=1` |

## Replay
```bash
python -m agentreplay replay-dynamo --trace data/cc_deepseek.jsonl --label <arm>
```

## Results
(pending)
