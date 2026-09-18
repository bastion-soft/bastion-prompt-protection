from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single ASCII space and trim."""
    return _WS_RE.sub(" ", text).strip()


def round_to(x: float, n: int) -> float:
    """Round *x* to *n* decimal places."""
    return round(x, n)
