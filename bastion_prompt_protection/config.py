from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Preset(str, Enum):
    # The only published preset. DeBERTa-v3-xsmall fine-tune, 70M params,
    # ONNX-INT8 quantized — see model card on Hugging Face.
    TINY = "tiny"


# Model registry. Keys map to HuggingFace repos that we publish; the SDK
# downloads weights on first use and caches them.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    Preset.TINY.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
    },
}


@dataclass(frozen=True)
class Thresholds:
    safe_below: float = 0.20
    attack_above: float = 0.50
    heuristic_short_circuit: float = 0.95


@dataclass
class GuardConfig:
    preset: Preset = Preset.TINY
    thresholds: Thresholds = field(default_factory=Thresholds)

    enable_heuristics: bool = True
    enable_binary: bool = True
    enable_llm_judge: bool = False

    max_input_chars: int = 8000

    cache_dir: str | None = None

    @classmethod
    def from_preset(cls, preset: str | Preset) -> GuardConfig:
        if isinstance(preset, str):
            preset = Preset(preset)
        return cls(preset=preset)

    def model_id(self, stage: str) -> str:
        return MODEL_REGISTRY[self.preset.value][stage]
