from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from bastion_prompt_protection.config import GuardConfig, Preset
from bastion_prompt_protection.stages.binary import BinaryStage
from bastion_prompt_protection.stages.heuristics import HeuristicResult, HeuristicsStage
from bastion_prompt_protection.stages.multiclass import MulticlassStage
from bastion_prompt_protection.version import __version__

LABEL_SAFE = "safe"
LABEL_ATTACK = "attack"

STAGE_HEURISTICS = "heuristics"
STAGE_BINARY = "binary"
STAGE_MULTICLASS = "multiclass"


@dataclass
class GuardResult:
    risk: float
    label: str
    injection_type: str | None = None
    type_scores: dict[str, float] = field(default_factory=dict)
    matched_rules: list[str] = field(default_factory=list)
    stage_reached: str = STAGE_HEURISTICS
    latency_ms: float = 0.0
    model_version: str = __version__

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
        self._multiclass = (
            MulticlassStage(self.config.model_id("multiclass"), cache_dir=self.config.cache_dir)
            if self.config.enable_multiclass
            else None
        )

    def protect(self, prompt: str) -> GuardResult:
        start = time.perf_counter()
        text = (prompt or "")[: self.config.max_input_chars]

        heuristic_result = (
            self._heuristics.run(text) if self._heuristics is not None else HeuristicResult()
        )

        if heuristic_result.score >= self.config.thresholds.heuristic_short_circuit:
            return self._finalize(
                risk=heuristic_result.score,
                heuristic=heuristic_result,
                stage_reached=STAGE_HEURISTICS,
                start=start,
                inferred_type=heuristic_result.inferred_type,
            )

        binary_risk = 0.0
        binary_available = False
        if self._binary is not None:
            pred = self._binary.predict(text)
            if pred.available:
                binary_risk = pred.risk
                binary_available = True

        risk = max(heuristic_result.score, binary_risk)
        stage_reached = STAGE_BINARY if binary_available else STAGE_HEURISTICS

        if risk < self.config.thresholds.safe_below:
            return self._finalize(
                risk=risk,
                heuristic=heuristic_result,
                stage_reached=stage_reached,
                start=start,
                inferred_type=heuristic_result.inferred_type,
            )

        type_scores: dict[str, float] = {}
        inferred_type: str | None = heuristic_result.inferred_type

        should_type = self._multiclass is not None and risk >= self.config.thresholds.safe_below
        if should_type:
            mc = self._multiclass.predict(text)  # type: ignore[union-attr]
            if mc.available:
                type_scores = mc.type_scores
                inferred_type = mc.inferred_type or inferred_type
                stage_reached = STAGE_MULTICLASS

        return self._finalize(
            risk=risk,
            heuristic=heuristic_result,
            stage_reached=stage_reached,
            start=start,
            inferred_type=inferred_type,
            type_scores=type_scores,
        )

    def _finalize(
        self,
        *,
        risk: float,
        heuristic: HeuristicResult,
        stage_reached: str,
        start: float,
        inferred_type: str | None = None,
        type_scores: dict[str, float] | None = None,
    ) -> GuardResult:
        label = LABEL_ATTACK if risk >= self.config.thresholds.attack_above else LABEL_SAFE
        if label == LABEL_SAFE and risk < self.config.thresholds.safe_below:
            inferred_type = None

        latency_ms = (time.perf_counter() - start) * 1000.0
        return GuardResult(
            risk=round(risk, 4),
            label=label,
            injection_type=inferred_type if label == LABEL_ATTACK else None,
            type_scores=type_scores or {},
            matched_rules=heuristic.matched_rule_ids,
            stage_reached=stage_reached,
            latency_ms=round(latency_ms, 3),
        )
