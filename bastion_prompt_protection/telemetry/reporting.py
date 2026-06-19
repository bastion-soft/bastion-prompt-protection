"""Composition helpers (E6.1) — wiring a reporter to detection *without* coupling
it into :class:`Guard`.

``Guard`` stays a pure detector. Reporting is layered on by composition: the
integrations hold a reporter and call ``reporter.report(make_record(...))``;
direct users wrap a guard in :class:`ReportingGuard`.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

from bastion_prompt_protection.telemetry.reporter import ReportContext, Reporter, make_record


class ReportingGuard:
    """Wrap a :class:`Guard` so each ``protect`` also reports — by composition.

        guard = Guard()
        reporter = build_reporter(TelemetryConfig.from_env())
        safe = ReportingGuard(guard, reporter)
        safe.protect("…")   # detects, then fire-and-forget reports (direct/user_prompt)
    """

    def __init__(self, guard: Any, reporter: Reporter, *, context: ReportContext | None = None) -> None:
        self._guard = guard
        self._reporter = reporter
        self._context = context or ReportContext()

    def protect(self, prompt: str) -> Any:
        result = self._guard.protect(prompt)
        ctx = replace(self._context, content=prompt) if self._context.content is None else self._context
        with contextlib.suppress(Exception):  # telemetry must never break detection
            self._reporter.report(make_record(result, ctx, self._guard))
        return result

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (model_version, license_status, …) to the guard.
        return getattr(self._guard, name)
