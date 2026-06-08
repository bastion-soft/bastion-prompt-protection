from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Preset(str, Enum):
    # Free, AGPL. DeBERTa-v3-xsmall fine-tune, 70M params, ONNX-INT8 quantized.
    TINY = "tiny"
    # Commercial, gated on the HF Hub. mdeberta-v3-base, 280M, multilingual.
    # Requires a license + granted HF access (https://bastionsoft.com) — the
    # weights simply won't download without it. See bastion_prompt_protection.license.
    MULTILINGUAL = "multilingual"


# Model registry. Keys map to HuggingFace repos; the SDK downloads weights on
# first use and caches them. Presets are just named shortcuts — you don't have
# to use one: pass any repo id via GuardConfig(model=...) to point the
# detector at your own (or a self-hosted) model.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    Preset.TINY.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
    },
    Preset.MULTILINGUAL.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-mdeberta-v3-base-v1",
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

    # Point the detector at any HF repo id, bypassing the preset registry.
    # When set, this wins over `preset` — lets you run your own (or a
    # self-hosted) model without registering a preset.
    model: str | None = None

    # Commercial license (optional, verified offline). Path to the signed
    # license JSON emailed on purchase; defaults to $BASTION_LICENSE or
    # ~/.bastion/license.json. `require_license=True` makes Guard() refuse to
    # start without a valid one. Default is non-blocking — status is exposed
    # via Guard.license_status for audit/logging.
    license_path: str | None = None
    require_license: bool = False

    @classmethod
    def from_preset(cls, preset: str | Preset) -> GuardConfig:
        if isinstance(preset, str):
            preset = Preset(preset)
        return cls(preset=preset)

    def model_id(self, stage: str) -> str:
        if stage == "binary" and self.model:
            return self.model
        return MODEL_REGISTRY[self.preset.value][stage]
