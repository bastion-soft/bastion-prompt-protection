from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bastion_prompt_protection.models.loader import OnnxModelLoader
from bastion_prompt_protection.stages.heuristics import ALL_TYPES


@dataclass
class MulticlassPrediction:
    type_scores: dict[str, float] = field(default_factory=dict)
    inferred_type: str | None = None
    available: bool = False


class MulticlassStage:
    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self._loader = OnnxModelLoader(model_id, cache_dir=cache_dir)

    def is_available(self) -> bool:
        return self._loader.is_available()

    def predict(self, text: str) -> MulticlassPrediction:
        if not self.is_available():
            return MulticlassPrediction(available=False)

        artifact = self._loader.artifact
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
        logits = outputs[0][0]
        probs = _softmax(logits)

        labels = artifact.labels or list(ALL_TYPES[: len(probs)])
        type_scores = {label: float(p) for label, p in zip(labels, probs, strict=False)}
        inferred = max(type_scores, key=type_scores.get) if type_scores else None

        return MulticlassPrediction(
            type_scores=type_scores,
            inferred_type=inferred,
            available=True,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
