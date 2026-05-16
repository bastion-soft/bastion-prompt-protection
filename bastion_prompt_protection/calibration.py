from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaling for calibrating classifier logits.

    Fit on a held-out validation set: minimize NLL by scaling logits by 1/T.
    Loaded alongside model weights at inference time.
    """

    temperature: float = 1.0

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> TemperatureScaler:
        from scipy.optimize import minimize  # type: ignore[import-not-found]

        def nll(t: np.ndarray) -> float:
            scaled = logits / max(t[0], 1e-3)
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return float(-log_probs[np.arange(len(labels)), labels].mean())

        result = minimize(
            nll,
            x0=np.array([1.0]),
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": lr},
        )
        return cls(temperature=float(result.x[0]))
