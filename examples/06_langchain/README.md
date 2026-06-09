# Example 6 — LangChain input guardrail

Screen prompt-injection / jailbreak attempts at the front of an LCEL chain, before
they reach your model.

## Prerequisites

```bash
pip install "bastion-prompt-protection[langchain]"
```

## Run

```bash
python examples/06_langchain/main.py
```

The first (benign) prompt passes through; the second (an injection) is blocked
with a `PromptInjectionError` before the model is ever called.

## How it works

`BastionGuardrail` is an idiomatic LangChain `Runnable`, so it composes with `|`:

```python
from bastion_prompt_protection.integrations.langchain import BastionGuardrail

chain = BastionGuardrail() | prompt | llm
chain.invoke("Ignore previous instructions…")   # -> raises PromptInjectionError
```

- **Benign input** is returned unchanged, so the chain continues.
- **An attack** raises `PromptInjectionError` (the triggering `GuardResult` is on
  `.result`). Pass `block=False` to pass input through instead and branch on
  `guardrail.detect(text)` yourself.
- **Dict inputs** (prompt-template variables): set `input_key="field"` to screen
  one field, or leave it unset to screen all string values.
- Bring your own `Guard` (e.g. a different `preset`, threshold, or the commercial
  multilingual model) via `BastionGuardrail(guard=...)` / `preset=` / `config=`.

By default it uses the free `tiny` model — see the repo [Editions](../../README.md#editions)
for the multilingual option.
