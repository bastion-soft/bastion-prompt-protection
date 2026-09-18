from __future__ import annotations

MODEL_TOKEN_WINDOW = 512
"""Total tokens the classifier reads, including [CLS] and [SEP]."""

SPECIAL_TOKEN_BUDGET = 2
"""[CLS] and [SEP] reserved by the post-processor."""

CONTENT_TOKEN_WINDOW = MODEL_TOKEN_WINDOW - SPECIAL_TOKEN_BUDGET
"""Content tokens per window after special tokens are attached (510)."""

DEFAULT_TOKEN_OVERLAP = 64
"""Default overlap between consecutive windows, in content tokens."""

DEFAULT_SLAB_CHARS = 512
"""Maximum characters per tokenization slab."""

DEFAULT_MAX_INPUT_CHARS = 262_144
"""Total input length bound before token windowing (characters, not tokens)."""

LABEL_ATTACK = "attack"
LABEL_SAFE = "safe"

STAGE_CLASSIFIER = "classifier"
STAGE_HEURISTICS = "heuristics"


def estimate_window_count(
    token_count: int,
    content_window: int = CONTENT_TOKEN_WINDOW,
    overlap_tokens: int = DEFAULT_TOKEN_OVERLAP,
) -> int:
    """Estimate how many windows cover a token stream of the given length."""
    if token_count <= 0:
        return 0
    if token_count <= content_window:
        return 1
    step = max(1, content_window - overlap_tokens)
    return -(-( token_count - content_window) // step) + 1
