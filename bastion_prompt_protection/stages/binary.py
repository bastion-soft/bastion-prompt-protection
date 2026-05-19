from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from bastion_prompt_protection.calibration import TemperatureScaler
from bastion_prompt_protection.models.loader import OnnxModelLoader

logger = logging.getLogger(__name__)


# Returned when model weights are not yet available. Sits exactly between
# safe_below and attack_above so it routes to whichever next stage is enabled
# without falsely classifying anything.
NEUTRAL_RISK = 0.5


@dataclass
class BinaryPrediction:
    risk: float
    available: bool


class BinaryStage:
    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self._loader = OnnxModelLoader(model_id, cache_dir=cache_dir)
        # Default to identity scaling (T=1.0); replaced with the fitted
        # value the first time the model loads successfully.
        self._scaler: TemperatureScaler = TemperatureScaler(temperature=1.0)
        self._calibration_loaded: bool = False

    def is_available(self) -> bool:
        return self._loader.is_available()

    @property
    def model_version(self) -> str | None:
        """Identifier for the currently loaded model build (7-char prefix
        of the HuggingFace snapshot commit SHA). Returns `None` if the
        model hasn't been loaded yet; does not trigger loading."""
        sha = self._loader.revision
        if sha is None:
            return None
        return sha[:7]

    def predict(self, text: str) -> BinaryPrediction:
        if not self.is_available():
            return BinaryPrediction(risk=NEUTRAL_RISK, available=False)

        artifact = self._loader.artifact

        # Lazy-load the temperature.json on the first available() ping.
        # Done here (not in __init__) because the snapshot directory only
        # exists after the loader has fetched the model.
        if not self._calibration_loaded:
            self._scaler = _load_temperature(artifact.model_dir)
            self._calibration_loaded = True

        encoding = artifact.tokenizer.encode(text)

        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in {i.name for i in artifact.session.get_inputs()}:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = artifact.session.run(None, feed)
        # Apply temperature calibration to the raw logits before softmax.
        # If temperature.json was missing, _scaler is identity (T=1.0).
        logits = self._scaler.transform(outputs[0][0])
        probs = _softmax(logits)
        # Convention: index 1 is the attack class.
        attack_prob = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])

        return BinaryPrediction(risk=attack_prob, available=True)


def _load_temperature(model_dir) -> TemperatureScaler:
    """Read temperature.json from the model snapshot, or fall back to T=1.0.

    Older model snapshots without a calibration file load with identity
    scaling so the SDK remains backward-compatible.
    """
    temp_file = model_dir / "temperature.json"
    if not temp_file.exists():
        logger.info(
            "bastion_prompt_protection: temperature.json not found in %s; "
            "skipping calibration (scores will be uncalibrated)",
            model_dir,
        )
        return TemperatureScaler(temperature=1.0)
    try:
        payload = json.loads(temp_file.read_text())
        temperature = float(payload["temperature"])
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        return TemperatureScaler(temperature=temperature)
    except Exception as exc:
        logger.warning(
            "bastion_prompt_protection: could not load temperature.json (%s); "
            "falling back to identity scaling",
            exc,
        )
        return TemperatureScaler(temperature=1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
