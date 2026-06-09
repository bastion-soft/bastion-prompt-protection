# Example 7 — LlamaIndex RAG guardrail

Screen a LlamaIndex RAG pipeline for prompt injection — including **indirect
injection**, where the malicious instruction is hidden in a *retrieved document*
rather than the user's question. That's the threat unique to RAG, and it's why a
node postprocessor (which sees the retrieved content) catches things a
front-of-chain input guard can't.

## Prerequisites

```bash
pip install "bastion-prompt-protection[llamaindex]"
```

## Run

```bash
python examples/07_llamaindex/main.py
```

Runs offline (hand-built nodes, no LLM/embeddings/API key). Shows a poisoned
retrieved node being dropped (`block=False`) and the query being blocked
(`block=True`).

## How it works

`BastionGuardrailPostprocessor` is a LlamaIndex `BaseNodePostprocessor`. It runs
**after retrieval, before response synthesis**, and screens both the query and
the retrieved nodes:

```python
from bastion_prompt_protection.integrations.llamaindex import BastionGuardrailPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[BastionGuardrailPostprocessor()],
)
query_engine.query("…")   # raises PromptInjectionError if the query or a
                          # retrieved node is flagged
```

- **`block=True`** (default): raise `PromptInjectionError` (carrying the
  `GuardResult` on `.result`) if the query or any retrieved node is flagged.
- **`block=False`**: don't raise — **drop** flagged retrieved nodes so poisoned
  documents never reach the LLM, and answer from the clean ones.
- **`screen_query` / `screen_nodes`**: toggle each surface independently.
- Bring your own `Guard` (different `preset`/threshold, or the commercial
  multilingual model) via `BastionGuardrailPostprocessor(guard=...)` / `preset=` /
  `config=`.

By default it uses the free `tiny` model — see the repo
[Editions](../../README.md#editions) for the multilingual option.
