"""Native batched HTTP ingest channel (plan §17.3 ch.1, E6.3).

Posts the rich audit record to the gateway's ``POST /v1/events:batch``
(ingest-scoped key). Stdlib ``urllib`` only — no requests/httpx dependency. The
sink raises on failure so :class:`BackgroundReporter` retries with backoff.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any


def make_http_sink(
    endpoint: str, api_key: str, *, timeout: float = 5.0
) -> Callable[[list[dict[str, Any]]], None]:
    url = endpoint.rstrip("/") + "/v1/events:batch"

    def sink(batch: list[dict[str, Any]]) -> None:
        data = json.dumps({"events": batch}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"ingest failed: HTTP {resp.status}")

    return sink
