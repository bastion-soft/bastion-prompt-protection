from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from bastion_prompt_protection.config import GuardOptions, Preset, resolve_config
from bastion_prompt_protection.constants import (
    LABEL_ATTACK,
    LABEL_SAFE,
    STAGE_CLASSIFIER,
    STAGE_HEURISTICS,
)
from bastion_prompt_protection.stages.classifier import ClassifierStage
from bastion_prompt_protection.stages.heuristics import HeuristicsStage
from bastion_prompt_protection.utils import collapse_whitespace, round_to
from bastion_prompt_protection.version import __version__

if TYPE_CHECKING:
    from bastion_prompt_protection.license import LicenseStatus


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


@dataclass
class WindowedGuardResult(GuardResult):
    windows_scanned: int = 0
    windows_total: int = 0
    windows_total_exact: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Guard:
    """Two-stage prompt-injection detector.

    Note on sync vs async: Python's ``protect()`` is synchronous. TypeScript's
    is async because Node I/O is unavoidably async. Everything else — thresholds,
    ordering, rounding — matches the TypeScript pipeline exactly.
    """

    def __init__(self, config: GuardOptions | Preset | str | None = None) -> None:
        self.config: GuardOptions = resolve_config(config)

        if self.config.require_license:
            status = self.license_status
            if not status.valid:
                raise RuntimeError(
                    "Bastion: require_license=True but no valid commercial license was "
                    f"found ({status.reason}). Obtain one at https://bastionsoft.com, "
                    "or set require_license=False."
                )

        self._heuristics = HeuristicsStage() if self.config.enable_heuristics else None
        self._classifier = (
            ClassifierStage(
                self.config.classifier_repo(),
                on_model_unavailable=self.config.on_model_unavailable,
                cache_dir=self.config.cache_dir,
            )
            if self.config.enable_classifier
            else None
        )

    @property
    def sdk_version(self) -> str:
        """The bastion-prompt-protection package version."""
        return __version__

    @property
    def model_version(self) -> str | None:
        """7-char commit SHA of the loaded HuggingFace snapshot, or None if not yet loaded."""
        if self._classifier is None:
            return None
        return self._classifier.model_version

    @property
    def license_status(self) -> LicenseStatus:
        """Offline status of the commercial license from ``config.license_path``."""
        from bastion_prompt_protection.license import verify_license

        return verify_license(self.config.license_path)

    def protect(
        self,
        prompt: str,
        *,
        max_windows: int | None = None,
        overlap_tokens: int | None = None,
        window_tokens: int | None = None,
        slab_chars: int | None = None,
        normalize_whitespace: bool | None = None,
    ) -> WindowedGuardResult:
        """Scan *prompt* for prompt injection.

        By default the input is split into overlapping token windows and every
        window is scanned, stopping early once a window reaches
        ``thresholds.attack_above``. Pass ``max_windows=1`` to scan only the
        first model window — single-window mode equivalent to old truncation
        behaviour.

        Raises ``ModelUnavailableError`` when the classifier cannot be loaded.
        The heuristic short-circuit (``score >= thresholds.heuristic_short_circuit``)
        runs before the model is consulted and returns an attack without the model.
        """
        start = time.perf_counter()

        should_normalize = normalize_whitespace if normalize_whitespace is not None else self.config.normalize_whitespace
        full_text = collapse_whitespace(prompt) if should_normalize else prompt
        text_for_model = full_text[: self.config.max_input_chars]

        if self._heuristics is not None:
            heuristic_score = self._heuristics.score(full_text)
            if heuristic_score >= self.config.thresholds.heuristic_short_circuit:
                return self._with_windows(
                    self._finalize(heuristic_score, STAGE_HEURISTICS, start),
                    scanned=0,
                    total=0,
                    exact=True,
                )

        if self._classifier is None or not text_for_model:
            return self._with_windows(
                self._finalize(0.0, STAGE_HEURISTICS, start),
                scanned=0,
                total=0,
                exact=True,
            )

        if max_windows == 1:
            risk = self._classifier.score(text_for_model)
            return self._with_windows(
                self._finalize(risk, STAGE_CLASSIFIER, start),
                scanned=1,
                total=1,
                exact=True,
            )

        _overlap = overlap_tokens if overlap_tokens is not None else self.config.overlap_tokens
        limit = max_windows if max_windows is not None else float("inf")

        worst_result: GuardResult | None = None
        scanned = 0
        windows_total = 1
        windows_total_exact = False

        window_kwargs: dict = {"overlap_tokens": _overlap}
        if window_tokens is not None:
            window_kwargs["window_tokens"] = window_tokens
        if slab_chars is not None:
            window_kwargs["slab_chars"] = slab_chars

        for window in self._classifier.windows(text_for_model, **window_kwargs):
            if scanned >= limit:
                break
            scanned += 1
            windows_total = window.windows_total
            windows_total_exact = window.windows_total_exact

            risk = self._classifier.score_encoded(window.encoding)
            result = self._finalize(risk, STAGE_CLASSIFIER, start)
            if worst_result is None or result.risk > worst_result.risk:
                worst_result = result
            if worst_result.risk >= self.config.thresholds.attack_above:
                break

        if scanned == 0:
            risk = self._classifier.score(text_for_model)
            return self._with_windows(
                self._finalize(risk, STAGE_CLASSIFIER, start),
                scanned=0,
                total=0,
                exact=True,
            )

        return WindowedGuardResult(
            risk=worst_result.risk,  # type: ignore[union-attr]
            label=worst_result.label,  # type: ignore[union-attr]
            stage_reached=worst_result.stage_reached,  # type: ignore[union-attr]
            latency_ms=round_to((time.perf_counter() - start) * 1000.0, 3),
            windows_scanned=scanned,
            windows_total=windows_total,
            windows_total_exact=windows_total_exact,
        )

    def _finalize(self, risk: float, stage_reached: str, start: float) -> GuardResult:
        label = LABEL_ATTACK if risk >= self.config.thresholds.attack_above else LABEL_SAFE
        latency_ms = (time.perf_counter() - start) * 1000.0
        return GuardResult(
            risk=round_to(risk, 4),
            label=label,
            stage_reached=stage_reached,
            latency_ms=round_to(latency_ms, 3),
        )

    def _with_windows(
        self,
        base: GuardResult,
        *,
        scanned: int,
        total: int,
        exact: bool,
    ) -> WindowedGuardResult:
        return WindowedGuardResult(
            risk=base.risk,
            label=base.label,
            stage_reached=base.stage_reached,
            latency_ms=base.latency_ms,
            windows_scanned=scanned,
            windows_total=total,
            windows_total_exact=exact,
        )
