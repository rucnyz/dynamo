# E2: sglang + HiCache 2-tier (no SSD, tier ablation)

Same as E1 but without `--hicache-storage-backend`. Tiers: HBM → DRAM → DROP.
Must show no-regression vs E1 baseline.

```bash
python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --trust-remote-code \
  --enable-hierarchical-cache --hicache-ratio 2
```

See E1 README for arms and replay instructions. Results: (pending)
