# E3: sglang + HiCache + mooncake (transport backend extensibility)

Same as E1 but with `--hicache-storage-backend mooncake`. Proves value-eviction
works across different transport backends.

```bash
python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --trust-remote-code \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-storage-backend mooncake
```

See E1 README for arms and replay instructions. Results: (pending)
