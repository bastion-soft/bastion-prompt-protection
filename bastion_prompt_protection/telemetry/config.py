"""Telemetry config surface (plan §17.6, E6.5) — ALL DEFAULT OFF.

With no configuration nothing is emitted: pure in-process detection, zero egress.
Each channel (native HTTP → gateway, OTLP → collector, LangSmith) is enabled
independently; the reporter pipeline fans to whichever are configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TelemetryConfig:
    # Native HTTP channel → Bastion gateway console
    endpoint: str | None = None
    api_key: str | None = None
    # OTLP channel → customer's own collector (E6.4)
    otel_endpoint: str | None = None
    # LangSmith channel (LangChain users)
    langsmith: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    # shared
    sample_rate: float = 1.0
    client_id: str | None = None
    source: str = "sdk"
    environment: str | None = None

    @property
    def http_enabled(self) -> bool:
        return bool(self.endpoint and self.api_key)

    @property
    def otel_enabled(self) -> bool:
        return bool(self.otel_endpoint)

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith)

    @property
    def enabled(self) -> bool:
        return self.http_enabled or self.otel_enabled or self.langsmith_enabled

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        return cls(
            endpoint=os.environ.get("BASTION_TELEMETRY_ENDPOINT"),
            api_key=os.environ.get("BASTION_TELEMETRY_KEY"),
            otel_endpoint=os.environ.get("BASTION_OTEL_ENDPOINT"),
            langsmith=_env_bool("BASTION_LANGSMITH"),
            langsmith_api_key=os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT"),
            sample_rate=_env_float("BASTION_TELEMETRY_SAMPLE_RATE", 1.0),
            client_id=os.environ.get("BASTION_CLIENT_ID"),
            source=os.environ.get("BASTION_SOURCE", "sdk"),
            environment=os.environ.get("BASTION_ENVIRONMENT"),
        )
