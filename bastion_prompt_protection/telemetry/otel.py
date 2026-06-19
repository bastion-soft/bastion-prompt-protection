"""OTel sink (plan §17.3 ch.2, E6.4) — emit guardrail spans to the customer's
own OTLP collector. Optional dependency (``bastion-prompt-protection[otel]``);
lazy-imported so the core install stays lean.

Attributes match the gateway's spec-corrected mapping (semconv v1.37.0):
``gen_ai.operation.name=guardrail``, ``gen_ai.provider.name=bastion``,
``gen_ai.request.model``, plus ``bastion.*``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_BASTION_KEYS = ("risk", "label", "stage", "vector", "origin", "source", "preset")


def make_otel_sink(
    endpoint: str, *, service_name: str = "bastion-sdk"
) -> Callable[[list[dict[str, Any]]], None]:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
    )
    tracer = provider.get_tracer("bastion_prompt_protection")

    def sink(batch: list[dict[str, Any]]) -> None:
        for rec in batch:
            model = rec.get("model_version")
            name = f"guardrail {model}" if model else "guardrail"
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("gen_ai.operation.name", "guardrail")
                span.set_attribute("gen_ai.provider.name", "bastion")
                if model:
                    span.set_attribute("gen_ai.request.model", model)
                for k in _BASTION_KEYS:
                    v = rec.get(k)
                    if v is not None:
                        span.set_attribute(f"bastion.{k}", v)

    return sink
