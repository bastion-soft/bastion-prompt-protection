"""Telemetry reporter (plan §17.2/§17.4, E6.1/E6.2).

One pluggable reporter that every integration feeds via ``Guard.report`` — not
N "X→console" integrations. The reliability rules are non-negotiable (§17.4):
async/background, batched, fire-and-forget, bounded queue with drop-on-overflow,
retry+backoff, flush on shutdown, optional sampling that always keeps
flagged/blocked. **It never adds latency to, or raises into, the request path.**

Dependency-free (stdlib threading/queue) so the core SDK stays lean and the
reporter works in any sync or async app.
"""

from __future__ import annotations

import atexit
import logging
import queue
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Where a detection was caught — mirrors the gateway ScanContext (plan §6/§17).
VECTOR_DIRECT = "direct"
VECTOR_INDIRECT = "indirect"
ORIGIN_USER_PROMPT = "user_prompt"
ORIGIN_RAG_DOCUMENT = "rag_document"
ORIGIN_TOOL_RESULT = "tool_result"
ORIGIN_AGENT_STEP = "agent_step"


@dataclass
class ReportContext:
    """Provenance for one detection, supplied by the integration that caught it."""

    vector: str = VECTOR_DIRECT
    origin: str = ORIGIN_USER_PROMPT
    direction: str = "input"  # input | output (output-side screening)
    source: str | None = None  # integration tag: litellm/langchain/llamaindex/…
    request_id: str | None = None
    client_id: str | None = None
    content: str | None = None  # screened text; the gateway applies its snippet policy


class Reporter(ABC):
    @abstractmethod
    def report(self, record: dict[str, Any]) -> None: ...

    def flush(self) -> None:  # noqa: B027 - optional
        pass

    def shutdown(self) -> None:  # noqa: B027 - optional
        pass


class NoopReporter(Reporter):
    """Default — telemetry off. Zero work, zero egress."""

    def report(self, record: dict[str, Any]) -> None:
        return None


Sink = Callable[[list[dict[str, Any]]], None]  # delivers a batch; raises on failure


class BackgroundReporter(Reporter):
    """Bounded queue + daemon worker. Drops on overflow, retries with backoff,
    flushes on shutdown, swallows every error. ``report`` is non-blocking and
    never raises."""

    def __init__(
        self,
        sink: Sink,
        *,
        sample_rate: float = 1.0,
        max_queue: int = 10_000,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        self._sink = sink
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.dropped = 0
        self.sent = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="bastion-reporter", daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def report(self, record: dict[str, Any]) -> None:
        try:
            # Sampling: always keep non-safe (flagged/blocked); sample the rest.
            if (
                self._sample_rate < 1.0
                and record.get("label") == "safe"
                and random.random() > self._sample_rate
            ):
                return
            self._q.put_nowait(record)
        except queue.Full:
            self.dropped += 1
        except Exception:  # the request path must never see a telemetry error
            logger.debug("reporter.report swallowed an error", exc_info=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._flush(batch)
        remaining = self._drain()
        if remaining:
            self._flush(remaining)

    def _collect(self) -> list[dict[str, Any]]:
        try:
            first = self._q.get(timeout=self._flush_interval)
        except queue.Empty:
            return []
        batch = [first]
        while len(batch) < self._batch_size:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _drain(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while True:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                return batch

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        for attempt in range(self._max_retries):
            try:
                self._sink(batch)
                self.sent += len(batch)
                return
            except Exception:
                backoff = min(0.5 * 2**attempt, 5.0) * (1 + random.random() * 0.2)
                if not self._stop.wait(backoff):
                    continue
                break
        self.failed += len(batch)
        logger.warning("reporter dropped %d events after retries", len(batch))

    def flush(self) -> None:
        remaining = self._drain()
        if remaining:
            self._flush(remaining)

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        remaining = self._drain()
        if remaining:
            self._flush(remaining)


class MultiReporter(Reporter):
    """A pipeline of reporters — fans each record to every child (HTTP, OTel,
    LangSmith, …). Children are independent: each has its own queue/retry/drop,
    so one slow or failing channel can't affect the others. Never raises."""

    def __init__(self, reporters: list[Reporter]) -> None:
        self._reporters = reporters

    def report(self, record: dict[str, Any]) -> None:
        for r in self._reporters:
            try:
                r.report(record)
            except Exception:  # a channel can never break the request path
                logger.debug("child reporter raised on report", exc_info=True)

    def flush(self) -> None:
        for r in self._reporters:
            try:
                r.flush()
            except Exception:
                logger.debug("child reporter raised on flush", exc_info=True)

    def shutdown(self) -> None:
        for r in self._reporters:
            try:
                r.shutdown()
            except Exception:
                logger.debug("child reporter raised on shutdown", exc_info=True)


def make_record(result: Any, context: ReportContext, guard: Any) -> dict[str, Any]:
    """Build the wire record from a detection + its provenance + the guard's
    metadata. Pure helper — ``guard`` is only read for version/preset, so neither
    Guard nor the integrations depend on the reporter to produce a record."""
    record: dict[str, Any] = {
        "risk": result.risk,
        "label": result.label,
        "stage": result.stage_reached,
        "vector": context.vector,
        "origin": context.origin,
        "direction": context.direction,
        "source": context.source,
        "request_id": context.request_id,
        "client_id": context.client_id,
        "model_version": getattr(guard, "model_version", None),
        "sdk_version": getattr(guard, "sdk_version", None),
        "preset": getattr(getattr(guard, "config", None), "preset", None)
        and guard.config.preset.value,
        "latency_ms": result.latency_ms,
    }
    if context.content is not None:
        record["prompt"] = context.content
    return record
