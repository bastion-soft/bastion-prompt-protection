from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Preset(str, Enum):
    TINY = "tiny"  # DeBERTa-v3-xsmall (22M) — published as v1.0
    FAST = "fast"  # DeBERTa-v3-small (44M) — coming in v1.1
    ACCURATE = "accurate"  # DeBERTa-v3-base (180M) — coming in v1.1


# Model registry. Keys map to HuggingFace repos that we publish; the SDK
# downloads weights on first use and caches them. Only the TINY repo is
# populated at v1.0; FAST / ACCURATE follow once the larger backbones train.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    Preset.TINY.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
        "multiclass": "bastionsoft/multiclass-bastion-prompt-protection-deberta-v3-xsmall-v1",
    },
    Preset.FAST.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-small-v1",
        "multiclass": "bastionsoft/multiclass-bastion-prompt-protection-deberta-v3-small-v1",
    },
    Preset.ACCURATE.value: {
        "binary": "bastionsoft/binary-bastion-prompt-protection-deberta-v3-base-v1",
        "multiclass": "bastionsoft/multiclass-bastion-prompt-protection-deberta-v3-base-v1",
    },
}


@dataclass(frozen=True)
class Thresholds:
    safe_below: float = 0.20
    attack_above: float = 0.85
    heuristic_short_circuit: float = 0.95


@dataclass
class GuardConfig:
    preset: Preset = Preset.TINY
    thresholds: Thresholds = field(default_factory=Thresholds)

    enable_heuristics: bool = True
    enable_binary: bool = True
    enable_multiclass: bool = False  # multiclass typer ships in v2.0
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
