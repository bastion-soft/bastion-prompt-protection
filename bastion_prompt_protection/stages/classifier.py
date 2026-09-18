from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Generator

import numpy as np

from bastion_prompt_protection.calibration import TemperatureScaler
from bastion_prompt_protection.models.loader import OnnxModelLoader
from bastion_prompt_protection.models.tokenizer import Encoding, TokenWindow

if TYPE_CHECKING:
    from bastion_prompt_protection.config import ModelUnavailableMode

logger = logging.getLogger(__name__)


class ClassifierStage:
    def __init__(
        self,
        model_id: str,
        on_model_unavailable: ModelUnavailableMode = "try-download-then-throw",
        cache_dir: str | None = None,
    ) -> None:
        self._loader = OnnxModelLoader(
            model_id,
            on_model_unavailable=on_model_unavailable,
            cache_dir=cache_dir,
        )
        self._scaler: TemperatureScaler = TemperatureScaler(temperature=1.0)
        self._calibration_loaded: bool = False

    @property
    def model_version(self) -> str | None:
        """7-char prefix of the HuggingFace snapshot commit SHA, or None if not loaded."""
        sha = self._loader.revision
        if sha is None:
            return None
        return sha[:7]

    def score(self, text: str) -> float:
        """Single-window path: encode *text* with tokenizer truncation and run ONNX."""
        artifact = self._loader.load()
        self._maybe_load_calibration(artifact)
        encoding = artifact.tokenizer.encode(text)
        return self._run_onnx(artifact, encoding)

    def score_encoded(self, encoding: Encoding) -> float:
        """Window path: run ONNX on a pre-built encoding (already wrapped with CLS/SEP)."""
        artifact = self._loader.load()
        self._maybe_load_calibration(artifact)
        return self._run_onnx(artifact, encoding)

    def windows(
        self,
        text: str,
        *,
        window_tokens: int | None = None,
        overlap_tokens: int | None = None,
        slab_chars: int | None = None,
    ) -> Generator[TokenWindow, None, None]:
        """Delegate window generation to the tokenizer (requires model to be loaded)."""
        artifact = self._loader.load()
        self._maybe_load_calibration(artifact)
        kwargs: dict = {}
        if window_tokens is not None:
            kwargs["window_tokens"] = window_tokens
        if overlap_tokens is not None:
            kwargs["overlap_tokens"] = overlap_tokens
        if slab_chars is not None:
            kwargs["slab_chars"] = slab_chars
        return artifact.tokenizer.windows(text, **kwargs)

    def _maybe_load_calibration(self, artifact) -> None:
        if not self._calibration_loaded:
            self._scaler = _load_temperature(artifact.model_dir)
            self._calibration_loaded = True

    def _run_onnx(self, artifact, encoding: Encoding) -> float:
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in {i.name for i in artifact.session.get_inputs()}:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = artifact.session.run(None, feed)
        logits = self._scaler.transform(outputs[0][0])
        probs = _softmax(logits)
        return float(probs[1]) if probs.shape[0] > 1 else float(probs[0])


def _load_temperature(model_dir) -> TemperatureScaler:
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
