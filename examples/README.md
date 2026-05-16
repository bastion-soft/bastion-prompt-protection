# Examples — four ways to use Bastion Prompt Protection

Each pattern is self-contained: its own folder with a tutorial
`README.md` and runnable code. Pick the one that matches where you are
in the adoption journey.

| Pattern | Best for | Tutorial |
|---|---|---|
| **1. Raw ONNX, no SDK** | Skeptics, compliance reviewers, non-Python ports | [`01_raw_onnx/`](01_raw_onnx/README.md) |
| **2. SDK** | Standard server-side LLM apps; fastest integration | [`02_sdk/`](02_sdk/README.md) |
| **3. Verify the leaderboard** | "Don't trust the numbers, re-run the suite" | [`03_eval/`](03_eval/README.md) |
| **4. FastAPI + Docker** | Production sidecar, multi-language clients | [`04_server/`](04_server/README.md) |

All four reach the same risk number for the same prompt. They differ in
*how much you trust the vendor* — Pattern 1 minimises the trust surface
(no library, raw weights); Pattern 4 maximises convenience (pre-built
image, ready to deploy).

## Advanced

| Recipe | When |
|---|---|
| [`05_local_cache/`](05_local_cache/README.md) | Offline / air-gapped operation using the SDK with a pre-downloaded model cache |

## Reference behaviour

A typical run on a consumer CPU:

- **Cold start** (first inference after process boot): ~1500 ms — ONNX
  session init + first inference.
- **Warm steady-state**: ~5 ms p50, ~7 ms p95 per prompt.
- **Throughput**: ~180 prompts / second single-threaded; ~700 prompts /
  second on a 4-worker FastAPI deployment.

## Adding a new pattern

If you contribute a new pattern (e.g. LangChain integration, streaming,
multi-language), follow the existing shape:

```
examples/<NN>_<short_name>/
├── README.md       # 5-section tutorial: what / prerequisites / run / how / when-to-use-vs-other
├── main.py         # runnable, ~50-150 lines
└── (...)           # any extra files the pattern needs
```

Then add a row to the table above.
