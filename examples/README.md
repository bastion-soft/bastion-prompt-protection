# Examples — various ways to use Bastion Prompt Protection

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

## Integrations

| Recipe | When |
|---|---|
| [`06_langchain/`](06_langchain/README.md) | LangChain apps — `BastionGuardrailMiddleware` for `create_agent` agents, or `BastionGuardrail` as an LCEL input guardrail |
| [`07_llamaindex/`](07_llamaindex/README.md) | LlamaIndex RAG — screen the query and retrieved nodes, catching indirect injection in retrieved documents |
| [`08_openai_agents/`](08_openai_agents/README.md) | OpenAI Agents SDK — `make_input_guardrail()` as an `input_guardrail` that trips a tripwire before the agent's model call |
| [`09_litellm/`](09_litellm/README.md) | LiteLLM Proxy — one `config.yaml` stanza protects every model in the gateway, zero application-code changes |
| [`10_prompt_injection/`](10_prompt_injection/README.md) | End-to-end attack-and-defense demo — indirect prompt injection inside a structured invoice field, caught by a raw-ONNX guard before the LLM call |

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
