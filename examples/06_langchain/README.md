# Example 6 — LangChain integration

Screen prompt-injection / jailbreak attempts in a LangChain app, before they reach
your model. The package ships two entry points:

| File | Entry point | Use it for |
|---|---|---|
| [`agent_middleware.py`](agent_middleware.py) | `BastionGuardrailMiddleware` | Agents built with `create_agent` |
| [`main.py`](main.py) | `BastionGuardrail` | LCEL chains and RAG pipelines |

## Prerequisites

```bash
pip install "bastion-prompt-protection[langchain]"
```

## Run

```bash
python examples/06_langchain/agent_middleware.py   # agent middleware
python examples/06_langchain/main.py               # LCEL guardrail
```

In both, the first (benign) prompt passes through; the second (an injection) is
blocked before the model is called.

## Agent middleware (recommended for agents)

`BastionGuardrailMiddleware` screens in the `before_model` hook, which runs both
for the incoming user turn and after tools return:

```python
from langchain.agents import create_agent
from bastion_prompt_protection.integrations.langchain import BastionGuardrailMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[...],
    middleware=[BastionGuardrailMiddleware()],
)
```

- **User input and tool results** are screened by default (`check_input`,
  `check_tool_results`), so it also catches *indirect* injection carried in
  retrieved documents or tool output.
- **A flagged turn** ends the run with a refusal message (`exit_behavior="end"`,
  the default), raises `PromptInjectionError` (`"error"`), or is replaced in
  place so the run continues (`"replace"`).
- Customize with `violation_message`, `threshold`, `preset`, `config`, or a
  pre-built `guard=`.

## LCEL guardrail (for chains)

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

Both entry points use the free `tiny` model by default — see the repo
[Editions](../../README.md#editions) for the multilingual option, passed via
`preset=` / `config=` / `guard=`.
