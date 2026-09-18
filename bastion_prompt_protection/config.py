from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from bastion_prompt_protection.constants import DEFAULT_MAX_INPUT_CHARS, DEFAULT_TOKEN_OVERLAP


class Preset(str, Enum):
    # Free, AGPL. DeBERTa-v3-xsmall fine-tune, 70M params, ONNX-INT8 quantized.
    TINY = "tiny"
    # Commercial, gated on the HF Hub. mdeberta-v3-base, 280M, multilingual.
    # Requires a license + granted HF access (https://bastionsoft.com) — the
    # weights simply won't download without it. See bastion_prompt_protection.license.
    MULTILINGUAL = "multilingual"


# Model registry. Keys map to HuggingFace repos; the SDK downloads weights on
# first use and caches them. Presets are just named shortcuts — you don't have
# to use one: pass any repo id via GuardOptions(model=...) to point the
# detector at your own (or a self-hosted) model.
PRESET_HF_REPOS: dict[str, str] = {
    Preset.TINY.value: "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
    Preset.MULTILINGUAL.value: "bastionsoft/binary-bastion-prompt-protection-mdeberta-v3-base-v1",
}

ModelUnavailableMode = Literal["throw", "try-download-then-throw"]


@dataclass(frozen=True)
class Thresholds:
    attack_above: float = 0.50
    heuristic_short_circuit: float = 0.95


DEFAULT_THRESHOLDS: Thresholds = Thresholds()


@dataclass(frozen=True)
class GuardOptions:
    preset: Preset = Preset.TINY
    thresholds: Thresholds = field(default_factory=Thresholds)

    enable_heuristics: bool = True
    enable_classifier: bool = True

    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    overlap_tokens: int = DEFAULT_TOKEN_OVERLAP
    normalize_whitespace: bool = True

    on_model_unavailable: ModelUnavailableMode = "try-download-then-throw"

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

    def classifier_repo(self) -> str:
        """Resolve the HuggingFace repo id for the classifier stage."""
        if self.model:
            return self.model
        repo = PRESET_HF_REPOS.get(self.preset.value)
        if repo is None:
            raise ValueError(f"Unknown preset: {self.preset!r}")
        return repo


def resolve_config(config: GuardOptions | Preset | str | None) -> GuardOptions:
    """Normalise any of the accepted constructor forms into a ``GuardOptions``."""
    if config is None:
        return GuardOptions()
    if isinstance(config, GuardOptions):
        return config
    if isinstance(config, Preset):
        return GuardOptions(preset=config)
    if isinstance(config, str):
        return GuardOptions(preset=Preset(config))
    raise TypeError(f"Expected GuardOptions | Preset | str | None, got {type(config)!r}")
