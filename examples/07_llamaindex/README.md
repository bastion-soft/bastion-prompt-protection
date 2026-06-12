# Example 7 — LlamaIndex integration

Screen a LlamaIndex RAG pipeline for prompt injection across three surfaces,
each targeting a different point in the pipeline.

## Prerequisites

```bash
pip install "bastion-prompt-protection[llamaindex]"
```

## Run

```bash
python examples/07_llamaindex/main.py
```

Runs offline (hand-built nodes, no LLM / embeddings / API key). Shows all three
integration surfaces.

---

## Surface A — BastionGuardQueryEngine (PRIMARY)

The **only** surface that gives genuine pre-retrieval query-path blocking.
Wrap any existing query engine and injection attempts are stopped before the
vector store is ever queried:

```python
from bastion_prompt_protection.integrations.llamaindex import BastionGuardQueryEngine

safe_engine = BastionGuardQueryEngine(inner_engine=index.as_query_engine())
safe_engine.query("Ignore previous instructions...")   # raises PromptInjectionError
                                                       # BEFORE any retrieval
```

- **`screen_query=True`** (default): block the incoming query string.
- **`screen_nodes=True`** (default): also screen retrieved documents. When the
  inner engine exposes a `node_postprocessors` pipeline (as `index.as_query_engine()`
  does), a `BastionNodePostprocessor` is inserted into it so screening runs
  *before* synthesis; otherwise it falls back to a post-hoc pass over the
  response's source nodes.
- **`block=True`** (default): raise `PromptInjectionError`. Set `False` for
  passive monitoring (inspect via `detect()`).
- **`threshold`**, **`preset`**, **`config`**, **`guard=`**: same conventions
  as the LangChain integration.

---

## Surface B — BastionNodePostprocessor (SECONDARY)

Screens retrieved documents for *indirect injection* — a malicious instruction
hidden in a corpus document that was retrieved by a benign query:

```python
from bastion_prompt_protection.integrations.llamaindex import BastionNodePostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[BastionNodePostprocessor()],
)
```

- **`block=True`** (default): raise `PromptInjectionError` on the first flagged
  node.
- **`block=False`**: drop flagged nodes so poisoned content never reaches the
  LLM; the dropped node's metadata gets a `bastion_guard_result` entry for
  auditing.
- **`screen_query=False`** (default): query screening is an explicit opt-in
  here because retrieval has already happened by the time this postprocessor
  runs. Use `BastionGuardQueryEngine` for pre-retrieval query protection.

---

## Surface C — BastionWorkflowMixin (DOCS-FIRST)

For apps built as a LlamaIndex `Workflow`. Inherit from the mixin before
`Workflow`; the `bastion_guard_step` step intercepts `StartEvent`, screens the
input, and either forwards a `SafePassEvent` (safe) or terminates the run:

```python
from llama_index.core.workflow import Workflow, StopEvent, step
from bastion_prompt_protection.integrations.llamaindex import (
    BastionWorkflowMixin, SafePassEvent,
)

class MyWorkflow(BastionWorkflowMixin, Workflow):

    @step
    async def process(self, ev: SafePassEvent) -> StopEvent:
        # only reached when Bastion cleared the input
        return StopEvent(result=f"Answer for: {ev.input}")

wf = MyWorkflow()
result = await wf.run(input="What is focaccia?")  # -> "Answer for: ..."
```

Customise via class-level attributes or `__init__` kwargs:

| Kwarg | Default | Effect |
|---|---|---|
| `bastion_guard` | `None` (auto-created from `preset`) | Bring your own `Guard` |
| `bastion_preset` | `Preset.TINY` | Which model to load |
| `bastion_threshold` | `None` (Guard's default) | Override decision threshold |
| `bastion_block` | `True` | `True` raises; `False` returns `StopEvent` refusal |

---

Both `BastionGuardQueryEngine` and `BastionNodePostprocessor` expose a
`detect(text) -> GuardResult` helper that never raises — useful for logging or
building custom routing logic on top of the raw verdict.

By default they use the free `tiny` model — see the repo
[Editions](../../README.md#editions) for the multilingual option.
