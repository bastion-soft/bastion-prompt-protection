from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeuristicRule:
    pattern: re.Pattern[str]
    confidence: float


# v1.2.0 — pure-vocabulary regex rules removed. The v1.1 binary classifier
# already handles those patterns at higher precision; the regex layer was
# duplicating work and producing false positives (e.g. "Show me how to write
# a system prompt for my own chatbot" was flagged by the old
# system_prompt_leak regex). Only rules left in this tuple are structural —
# they detect attacks that don't survive tokenization (chat-template control
# tokens) or use formatting cues the model wasn't trained on (fake
# end-of-prompt delimiters).
RULES: tuple[HeuristicRule, ...] = (
    # Chat template control tokens injected as user input.
    HeuristicRule(
        pattern=re.compile(
            r"<\|(im_start|im_end|system|user|assistant|endoftext)\|>"
            r"|\[/?(INST|SYS)\]"
            r"|<<SYS>>",
        ),
        confidence=0.97,
    ),
    # Fake delimiters claiming the system prompt has ended.
    HeuristicRule(
        pattern=re.compile(
            r"-{3,}\s*end\s+of\s+(prompt|instructions?|system)\s*-{3,}"
            r"|###\s*end\s+of\s+(prompt|instructions?|system)\s*###",
            re.IGNORECASE,
        ),
        confidence=0.90,
    ),
)


_ZERO_WIDTH_CHARS = "​‌‍⁠﻿᠎"
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")
# Base64 payloads worth flagging are long, mixed-case, contain digits, and
# end with padding `=`. These three constraints cut math/identifier strings.
_BASE64_PAYLOAD_RE = re.compile(
    r"\b(?=[A-Za-z0-9+/]*[A-Z])(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*\d)"
    r"[A-Za-z0-9+/]{60,}={1,2}\b"
)
# Spaced-letter obfuscation: at least 8 single letters separated by spaces.
# 5 was too short and matched legitimate single-letter spelling-out in math
# and language problems.
_SPACED_LETTERS_RE = re.compile(r"(?:\b[A-Za-z]\s){8,}[A-Za-z]\b")


def _structural_score(text: str) -> float:
    """Confidence score for structural obfuscation signals; 0.0 if none."""
    # Zero-width characters in user prompts are almost always adversarial.
    # Threshold of 3 catches obfuscation while tolerating one stray pasted char.
    if len(_ZERO_WIDTH_RE.findall(text)) >= 3:
        return 0.96
    if _SPACED_LETTERS_RE.search(text):
        return 0.80
    if _BASE64_PAYLOAD_RE.search(text):
        return 0.55
    return 0.0


@dataclass
class HeuristicsStage:
    rules: tuple[HeuristicRule, ...] = field(default_factory=lambda: RULES)

    def run(self, text: str) -> float:
        """Return the highest-confidence match score (0.0 if no match)."""
        if not text:
            return 0.0
        best = max(
            (rule.confidence for rule in self.rules if rule.pattern.search(text)),
            default=0.0,
        )
        return max(best, _structural_score(text))
