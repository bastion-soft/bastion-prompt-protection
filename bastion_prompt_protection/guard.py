from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from bastion_prompt_protection.config import GuardConfig, Preset
from bastion_prompt_protection.stages.binary import BinaryStage
from bastion_prompt_protection.stages.heuristics import HeuristicsStage
from bastion_prompt_protection.version import __version__

LABEL_SAFE = "safe"
LABEL_ATTACK = "attack"

STAGE_HEURISTICS = "heuristics"
STAGE_BINARY = "binary"


@dataclass
class GuardResult:
    risk: float
    label: str
    stage_reached: str = STAGE_HEURISTICS
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_attack(self) -> bool:
        return self.label == LABEL_ATTACK


class Guard:
    def __init__(
        self,
        preset: str | Preset = Preset.TINY,
        config: GuardConfig | None = None,
    ) -> None:
        self.config = config or GuardConfig.from_preset(preset)

        self._heuristics = HeuristicsStage() if self.config.enable_heuristics else None
        self._binary = (
            BinaryStage(self.config.model_id("binary"), cache_dir=self.config.cache_dir)
            if self.config.enable_binary
            else None
        )

    @property
    def sdk_version(self) -> str:
        """The bastion-prompt-protection package version."""
        return __version__

    @property
    def model_version(self) -> str | None:
        """Identifier for the currently loaded model build (7-char commit
        SHA of the HuggingFace snapshot under the hood). Returns `None`
        if the model hasn't been loaded yet — lazy load triggers on the
        first `protect()` call — or if the binary stage is disabled.
        Useful for audit logs and bug reports."""
        if self._binary is None:
            return None
        return self._binary.model_version

    def protect(self, prompt: str) -> GuardResult:
        start = time.perf_counter()
        text = (prompt or "")[: self.config.max_input_chars]

        heuristic_score = self._heuristics.run(text) if self._heuristics is not None else 0.0

        if heuristic_score >= self.config.thresholds.heuristic_short_circuit:
            return self._finalize(
                risk=heuristic_score,
                stage_reached=STAGE_HEURISTICS,
                start=start,
            )

        binary_risk = 0.0
        binary_available = False
        if self._binary is not None:
            pred = self._binary.predict(text)
            if pred.available:
                binary_risk = pred.risk
                binary_available = True

        risk = max(heuristic_score, binary_risk)
        stage_reached = STAGE_BINARY if binary_available else STAGE_HEURISTICS

        return self._finalize(
            risk=risk,
            stage_reached=stage_reached,
            start=start,
        )

    def _finalize(
        self,
        *,
        risk: float,
        stage_reached: str,
        start: float,
    ) -> GuardResult:
        label = LABEL_ATTACK if risk >= self.config.thresholds.attack_above else LABEL_SAFE
        latency_ms = (time.perf_counter() - start) * 1000.0
        return GuardResult(
            risk=round(risk, 4),
            label=label,
            stage_reached=stage_reached,
            latency_ms=round(latency_ms, 3),
        )
