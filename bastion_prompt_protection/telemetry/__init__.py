"""Telemetry — the composable reporter pipeline that fans in-process detections
to a Bastion console / the customer's observability (plan §17).

Decoupled from ``Guard`` by design: build a reporter, hand it to an integration
(or wrap a guard in :class:`ReportingGuard`). All off by default.
"""

from __future__ import annotations

from functools import lru_cache

from bastion_prompt_protection.telemetry.config import TelemetryConfig
from bastion_prompt_protection.telemetry.reporter import (
    BackgroundReporter,
    MultiReporter,
    NoopReporter,
    ReportContext,
    Reporter,
    make_record,
)
from bastion_prompt_protection.telemetry.reporting import ReportingGuard


def build_reporter(config: TelemetryConfig | None) -> Reporter:
    """Compose the reporter pipeline from config. Each configured channel becomes
    its own independent (queued, retrying, isolated) reporter; returns a no-op
    when nothing is configured (default ⇒ zero egress, no background thread)."""
    if config is None or not config.enabled:
        return NoopReporter()

    reporters: list[Reporter] = []
    if config.http_enabled:
        from bastion_prompt_protection.telemetry.http import make_http_sink

        reporters.append(BackgroundReporter(
            make_http_sink(config.endpoint, config.api_key),  # type: ignore[arg-type]
            sample_rate=config.sample_rate))
    if config.otel_enabled:
        from bastion_prompt_protection.telemetry.otel import make_otel_sink

        reporters.append(BackgroundReporter(
            make_otel_sink(config.otel_endpoint), sample_rate=config.sample_rate))  # type: ignore[arg-type]
    if config.langsmith_enabled:
        from bastion_prompt_protection.telemetry.langsmith import make_langsmith_sink

        reporters.append(BackgroundReporter(
            make_langsmith_sink(api_key=config.langsmith_api_key, project=config.langsmith_project),
            sample_rate=config.sample_rate))

    if not reporters:
        return NoopReporter()
    return reporters[0] if len(reporters) == 1 else MultiReporter(reporters)


@lru_cache(maxsize=1)
def default_reporter() -> Reporter:
    """Cached reporter built from the environment — the fallback an integration
    uses when no reporter is injected. No-op unless ``BASTION_TELEMETRY_*`` etc.
    are set, so it preserves the zero-config-but-env-activatable UX."""
    return build_reporter(TelemetryConfig.from_env())


__all__ = [
    "BackgroundReporter",
    "MultiReporter",
    "NoopReporter",
    "ReportContext",
    "Reporter",
    "ReportingGuard",
    "TelemetryConfig",
    "build_reporter",
    "default_reporter",
    "make_record",
]
