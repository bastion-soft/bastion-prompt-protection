from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    session: Any
    tokenizer: Any
    labels: list[str]
    model_dir: Path


class OnnxModelLoader:
    """Lazy loader for ONNX classifier artifacts published on the HF Hub.

    Loading is deferred until the first inference call so importing the
    library is fast and possible without inference deps installed.
    """

    def __init__(self, model_id: str, cache_dir: str | Path | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._artifact: ModelArtifact | None = None
        self._load_error: Exception | None = None

    def is_available(self) -> bool:
        if self._artifact is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            self._artifact = self._load()
            return True
        except Exception as exc:
            self._load_error = exc
            logger.warning(
                "bastion_prompt_protection: model %s unavailable (%s). "
                "Stage will return a neutral score until weights are published.",
                self.model_id,
                exc,
            )
            return False

    @property
    def artifact(self) -> ModelArtifact:
        if not self.is_available():
            raise RuntimeError(f"Model {self.model_id} is not available: {self._load_error}")
        assert self._artifact is not None
        return self._artifact

    def _load(self) -> ModelArtifact:
        try:
            import onnxruntime  # type: ignore[import-not-found]
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Required runtime dependency missing — "
                "reinstall bastion-prompt-protection to repair: pip install --force-reinstall bastion-prompt-protection"
            ) from exc

        local_dir = snapshot_download(
            repo_id=self.model_id,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        local_path = Path(local_dir)

        # Look for the ONNX file in the HF/Optimum-conventional locations,
        # preferring the quantized build for fastest CPU inference.
        onnx_candidates = [
            local_path / "onnx" / "model_quantized.onnx",
            local_path / "onnx" / "model.onnx",
            local_path / "model.onnx",
        ]
        onnx_path = next((p for p in onnx_candidates if p.exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(
                f"No ONNX weights found in {local_path}. "
                f"Looked for: {[str(p.relative_to(local_path)) for p in onnx_candidates]}"
            )

        tokenizer_path = local_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {local_path}")

        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        labels = self._load_labels(local_path)

        return ModelArtifact(
            session=session,
            tokenizer=tokenizer,
            labels=labels,
            model_dir=local_path,
        )

    @staticmethod
    def _load_labels(model_dir: Path) -> list[str]:
        labels_file = model_dir / "labels.txt"
        if labels_file.exists():
            return [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
        return []
