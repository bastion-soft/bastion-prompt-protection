"""LangSmith sink (E6 follow-on) — log Bastion detections as LangSmith runs so
LangChain users see guardrail verdicts alongside their traces.

Built behind the Sink interface: it's just another channel in the reporter
pipeline. The ``langsmith`` package is an optional dependency
(``bastion-prompt-protection[langsmith]``), lazy-imported; a client can be
injected (for tests / to attach to an existing run tree). The record→run mapping
is a pure function so it's testable without the dependency or a network call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def langsmith_run_payload(record: dict[str, Any], *, project: str | None = None) -> dict[str, Any]:
    """Map a Bastion record onto LangSmith ``create_run`` kwargs (pure)."""
    label = record.get("label")
    payload: dict[str, Any] = {
        "name": "bastion.guardrail",
        "run_type": "tool",
        "inputs": {"vector": record.get("vector"), "origin": record.get("origin")},
        "outputs": {
            "risk": record.get("risk"),
            "label": label,
            "stage": record.get("stage"),
        },
        "tags": ["bastion", f"label:{label}", f"origin:{record.get('origin')}"],
        "extra": {"metadata": {k: record.get(k) for k in ("source", "model_version", "preset")}},
    }
    if project is not None:
        payload["project_name"] = project
    return payload


def make_langsmith_sink(
    client: Any | None = None, *, api_key: str | None = None, project: str | None = None
) -> Callable[[list[dict[str, Any]]], None]:
    if client is None:
        from langsmith import Client  # optional dep; lazy

        client = Client(api_key=api_key) if api_key else Client()

    def sink(batch: list[dict[str, Any]]) -> None:
        for rec in batch:
            client.create_run(**langsmith_run_payload(rec, project=project))

    return sink
