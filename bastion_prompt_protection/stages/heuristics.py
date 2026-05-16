from __future__ import annotations

import re
from dataclasses import dataclass, field

# Injection type taxonomy. Mirror of training labels so heuristic outputs and
# model outputs share a vocabulary.
TYPE_BENIGN = "benign"
TYPE_JAILBREAK = "jailbreak"
TYPE_DIRECT_INJECTION = "direct_injection"
TYPE_INDIRECT_INJECTION = "indirect_injection"
TYPE_SYSTEM_PROMPT_LEAK = "system_prompt_leak"
TYPE_DATA_EXFILTRATION = "data_exfiltration"
TYPE_OBFUSCATION = "obfuscation"
TYPE_HARMFUL_INTENT = "harmful_intent"

ALL_TYPES = (
    TYPE_BENIGN,
    TYPE_JAILBREAK,
    TYPE_DIRECT_INJECTION,
    TYPE_INDIRECT_INJECTION,
    TYPE_SYSTEM_PROMPT_LEAK,
    TYPE_DATA_EXFILTRATION,
    TYPE_OBFUSCATION,
    TYPE_HARMFUL_INTENT,
)


@dataclass(frozen=True)
class HeuristicRule:
    id: str
    pattern: re.Pattern[str]
    injection_type: str
    confidence: float
    description: str


@dataclass(frozen=True)
class HeuristicMatch:
    rule_id: str
    injection_type: str
    confidence: float
    span: tuple[int, int]
    matched_text: str


@dataclass(frozen=True)
class HeuristicResult:
    matches: tuple[HeuristicMatch, ...] = ()
    score: float = 0.0
    inferred_type: str | None = None

    @property
    def matched_rule_ids(self) -> list[str]:
        return [m.rule_id for m in self.matches]


# Confidence values are calibrated precision estimates: probability that a
# match on this pattern alone indicates a real attack on a typical benign
# corpus. Anything >= 0.95 short-circuits the pipeline at Stage 1.
#
# Numbers here are starting estimates; they will be re-calibrated against the
# benchmark in eval/calibrate_heuristics.py once that exists.
RULES: tuple[HeuristicRule, ...] = (
    HeuristicRule(
        id="ignore_previous",
        pattern=re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,60}?\b"
            r"(all\s+)?(previous|prior|above|earlier|the\s+last|preceding|former)\b"
            r"[^.\n]{0,40}?\b(instructions?|prompts?|rules?|messages?|directives?|orders?|context)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_DIRECT_INJECTION,
        confidence=0.97,
        description="Classic ignore-previous-instructions attack family.",
    ),
    HeuristicRule(
        id="dan_persona",
        pattern=re.compile(
            r"\b(you\s+are\s+(now\s+)?|act\s+as|pretend\s+to\s+be|"
            r"from\s+now\s+on\s+you\s+are)\s*"
            r"(DAN|STAN|DUDE|AIM|developer\s+mode|jailbroken|"
            r"do\s+anything\s+now|an?\s+unfiltered\s+ai)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_JAILBREAK,
        confidence=0.98,
        description="Known jailbreak persona (DAN, STAN, AIM, developer mode, etc.).",
    ),
    HeuristicRule(
        id="do_anything_now",
        pattern=re.compile(
            r"\bdo\s+anything\s+now\b|\bDAN\s+mode\b|\bdeveloper\s+mode\s+enabled\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_JAILBREAK,
        confidence=0.96,
        description="DAN mode activation phrasing.",
    ),
    HeuristicRule(
        id="no_restrictions",
        pattern=re.compile(
            # Requires an AI-addressing subject within 80 chars, otherwise
            # benign roleplay phrases like "without copyright restrictions"
            # match. The subject anchors the rule to model-directed requests.
            r"\b(you|AI|assistant|model|chatbot|GPT|language\s+model)\b"
            r"[^.\n]{0,80}?\b(without|with\s+no|free\s+from|bypass(?:ing)?)\b"
            r"[^.\n]{0,30}?\b(content\s+filters?|guardrails?|"
            r"ethical\s+guidelines?|safety\s+(constraints?|measures?|guidelines?|"
            r"protocols?)|usage\s+polic(?:y|ies)|moral\s+guidelines?)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_JAILBREAK,
        confidence=0.88,
        description="Requests for an unrestricted model directed at the AI.",
    ),
    HeuristicRule(
        id="system_prompt_leak",
        pattern=re.compile(
            r"\b(reveal|print|repeat|output|display|show|tell\s+me|give\s+me)\b"
            r"[^.\n]{0,40}?\b(your\s+)?"
            r"(system\s+prompt|initial\s+instructions?|original\s+prompt|"
            r"hidden\s+instructions?|prior\s+instructions?|system\s+message)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_SYSTEM_PROMPT_LEAK,
        confidence=0.97,
        description="Direct request to leak the system prompt.",
    ),
    HeuristicRule(
        id="repeat_above",
        pattern=re.compile(
            r"\b(repeat|print|echo|say)\b[^.\n]{0,30}?\b"
            r"(everything|all|the\s+text)\s+(above|preceding|before|prior\s+to)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_SYSTEM_PROMPT_LEAK,
        confidence=0.85,
        description="Indirect attempt to dump prior context.",
    ),
    HeuristicRule(
        id="role_tag_inject",
        pattern=re.compile(
            r"<\|(im_start|im_end|system|user|assistant|endoftext)\|>"
            r"|\[/?(INST|SYS)\]"
            r"|<<SYS>>",
        ),
        injection_type=TYPE_DIRECT_INJECTION,
        confidence=0.97,
        description="Chat template control tokens injected as user input.",
    ),
    HeuristicRule(
        id="exfiltration_url",
        pattern=re.compile(
            r"\b(send|post|upload|exfiltrate|forward|leak)\b[^.\n]{0,40}?\b"
            r"(to|via)\b[^.\n]{0,30}?(https?://|attacker\.com|webhook)",
            re.IGNORECASE,
        ),
        injection_type=TYPE_DATA_EXFILTRATION,
        confidence=0.92,
        description="Data exfiltration to a URL.",
    ),
    HeuristicRule(
        id="end_of_prompt_marker",
        pattern=re.compile(
            r"-{3,}\s*end\s+of\s+(prompt|instructions?|system)\s*-{3,}"
            r"|###\s*end\s+of\s+(prompt|instructions?|system)\s*###",
            re.IGNORECASE,
        ),
        injection_type=TYPE_DIRECT_INJECTION,
        confidence=0.90,
        description="Fake delimiters claiming the system prompt has ended.",
    ),
    HeuristicRule(
        id="new_instructions",
        pattern=re.compile(
            # Drops "following" (benign in task headers like "### Following rules:")
            # and requires explicit override framing.
            r"\b(here\s+are\s+your|the\s+real|your\s+new|updated|revised)\s+"
            r"(instructions?|rules?|directives?|guidelines?|prompts?)\s*"
            r"(are|:)",
            re.IGNORECASE,
        ),
        injection_type=TYPE_DIRECT_INJECTION,
        confidence=0.78,
        description="Explicit override of existing instructions.",
    ),
    HeuristicRule(
        id="grandma_exploit",
        pattern=re.compile(
            r"\b(my\s+)?(grandma|grandmother|late\s+grandmother)\b[^.\n]{0,80}?"
            r"\b(used\s+to|would)\b[^.\n]{0,80}?\b(read|tell|recite|sing)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_JAILBREAK,
        confidence=0.93,
        description="Grandma roleplay social-engineering exploit.",
    ),
    HeuristicRule(
        id="hypothetical_unrestricted",
        pattern=re.compile(
            r"\b(in\s+a\s+hypothetical|imagine\s+a\s+world|in\s+a\s+fictional)\b"
            r"[^.\n]{0,80}?\b(no\s+rules|no\s+restrictions|anything\s+is\s+allowed)\b",
            re.IGNORECASE,
        ),
        injection_type=TYPE_JAILBREAK,
        confidence=0.82,
        description="Hypothetical-world framing to bypass restrictions.",
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


@dataclass(frozen=True)
class StructuralSignals:
    zero_width_count: int = 0
    has_long_base64: bool = False
    has_spaced_letters: bool = False
    non_ascii_ratio: float = 0.0


def _structural_scan(text: str) -> StructuralSignals:
    if not text:
        return StructuralSignals()

    zero_width = len(_ZERO_WIDTH_RE.findall(text))
    base64_hit = bool(_BASE64_PAYLOAD_RE.search(text))
    spaced_hit = bool(_SPACED_LETTERS_RE.search(text))
    non_ascii = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)

    return StructuralSignals(
        zero_width_count=zero_width,
        has_long_base64=base64_hit,
        has_spaced_letters=spaced_hit,
        non_ascii_ratio=non_ascii,
    )


def _structural_to_match(signals: StructuralSignals, text_len: int) -> HeuristicMatch | None:
    # Zero-width characters in user prompts are almost always adversarial.
    # Threshold of 3 catches obfuscation while tolerating one stray pasted char.
    if signals.zero_width_count >= 3:
        return HeuristicMatch(
            rule_id="zero_width_obfuscation",
            injection_type=TYPE_OBFUSCATION,
            confidence=0.96,
            span=(0, text_len),
            matched_text=f"<{signals.zero_width_count} zero-width chars>",
        )
    if signals.has_spaced_letters:
        return HeuristicMatch(
            rule_id="spaced_letter_obfuscation",
            injection_type=TYPE_OBFUSCATION,
            confidence=0.80,
            span=(0, text_len),
            matched_text="<spaced letters>",
        )
    if signals.has_long_base64:
        return HeuristicMatch(
            rule_id="base64_payload",
            injection_type=TYPE_OBFUSCATION,
            confidence=0.55,
            span=(0, text_len),
            matched_text="<base64 payload>",
        )
    return None


@dataclass
class HeuristicsStage:
    rules: tuple[HeuristicRule, ...] = field(default_factory=lambda: RULES)

    def run(self, text: str) -> HeuristicResult:
        if not text:
            return HeuristicResult()

        matches: list[HeuristicMatch] = []

        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                matches.append(
                    HeuristicMatch(
                        rule_id=rule.id,
                        injection_type=rule.injection_type,
                        confidence=rule.confidence,
                        span=m.span(),
                        matched_text=m.group(0),
                    )
                )

        structural = _structural_to_match(_structural_scan(text), len(text))
        if structural is not None:
            matches.append(structural)

        if not matches:
            return HeuristicResult()

        best = max(matches, key=lambda m: m.confidence)
        return HeuristicResult(
            matches=tuple(matches),
            score=best.confidence,
            inferred_type=best.injection_type,
        )
