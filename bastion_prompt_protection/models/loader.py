from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bastion_prompt_protection.errors import ModelUnavailableError

if TYPE_CHECKING:
    from bastion_prompt_protection.config import ModelUnavailableMode
    from bastion_prompt_protection.models.tokenizer import BastionTokenizer

RETRY_COOLDOWN_S = 5.0

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_ONNX_CANDIDATES = (
    "onnx/model_quantized.onnx",
    "onnx/model.onnx",
    "model.onnx",
)

_SIDECARS = ["*.json", "*.txt", "*.model"]


@dataclass
class ModelArtifact:
    session: Any
    tokenizer: BastionTokenizer
    labels: list[str]
    model_dir: Path


def _repo_folder_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _cache_root(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore[import-not-found]

    return Path(HF_HUB_CACHE)


def _find_cached_snapshot(cache_dir: Path | None, repo_id: str) -> Path | None:
    """Return the newest cached snapshot dir for *repo_id*, or None if absent."""
    snapshots_dir = _cache_root(cache_dir) / _repo_folder_name(repo_id) / "snapshots"
    try:
        entries = list(snapshots_dir.iterdir())
    except OSError:
        return None

    sha_dirs = [p for p in entries if p.is_dir() and _COMMIT_SHA_RE.match(p.name)]
    if not sha_dirs:
        return None
    if len(sha_dirs) == 1:
        return sha_dirs[0]

    best: Path | None = None
    best_mtime = -1.0
    for sha_dir in sha_dirs:
        mtime = sha_dir.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = sha_dir
    return best


def _snapshot_is_loadable(snapshot: Path) -> bool:
    if not (snapshot / "tokenizer.json").is_file():
        return False
    return any((snapshot / rel).is_file() for rel in _ONNX_CANDIDATES)


class OnnxModelLoader:
    """Lazy, fail-closed loader for ONNX classifier artifacts on the HF Hub.

    Loading is deferred until the first ``load()`` call so importing the
    library is fast. Failure behaviour is controlled by ``on_model_unavailable``:

    - ``"try-download-then-throw"``: on failure, raises ``ModelUnavailableError``
      but clears cached state after a 5-second cooldown so the next call retries.
      Once a retry succeeds the loader self-heals and operates normally.

    - ``"throw"``: the first failure is cached permanently; every subsequent
      ``load()`` call raises immediately with no retry.

    When a complete snapshot already exists in the HuggingFace cache, ``load()``
    uses it directly without contacting the Hub. Hub access happens only when
    the model is missing or the cached snapshot is incomplete.
    """

    def __init__(
        self,
        model_id: str,
        on_model_unavailable: ModelUnavailableMode = "try-download-then-throw",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self._on_model_unavailable = on_model_unavailable
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._artifact: ModelArtifact | None = None
        self._failure: tuple[Exception, float] | None = None
        self._lock = threading.Lock()

    @property
    def revision(self) -> str | None:
        """HF commit SHA the loaded snapshot was resolved to, or None if not yet loaded."""
        if self._artifact is None:
            return None
        return self._artifact.model_dir.name

    def load(self) -> ModelArtifact:
        """Load the model artifact. Thread-safe; concurrent callers share one download.

        Raises ``ModelUnavailableError`` when the model is not available.
        """
        if self._artifact is not None:
            return self._artifact

        with self._lock:
            # Re-check after acquiring the lock (another thread may have loaded).
            if self._artifact is not None:
                return self._artifact

            if self._failure is not None:
                err, failed_at = self._failure
                if self._on_model_unavailable == "throw":
                    raise ModelUnavailableError(self.model_id, err)
                elapsed = time.monotonic() - failed_at
                if elapsed < RETRY_COOLDOWN_S:
                    remaining = int(RETRY_COOLDOWN_S - elapsed) + 1
                    raise ModelUnavailableError(
                        self.model_id, err, hint=f"retry in {remaining}s"
                    )
                self._failure = None

            try:
                artifact = self._do_load()
                self._artifact = artifact
                return artifact
            except ModelUnavailableError:
                raise
            except Exception as exc:
                self._failure = (exc, time.monotonic())
                raise ModelUnavailableError(self.model_id, exc) from exc

    def _do_load(self) -> ModelArtifact:
        cached = _find_cached_snapshot(self.cache_dir, self.model_id)
        if cached is not None and _snapshot_is_loadable(cached):
            local_path = cached
        else:
            local_path = self._download_snapshot(cached)

        return self._build_artifact(local_path)

    def _download_snapshot(self, cached: Path | None) -> Path:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Required runtime dependency missing — "
                "reinstall bastion-prompt-protection to repair: "
                "pip install --force-reinstall bastion-prompt-protection"
            ) from exc

        cache = str(self.cache_dir) if self.cache_dir else None
        revision = cached.name if cached is not None else None
        download_kwargs: dict[str, Any] = {
            "repo_id": self.model_id,
            "cache_dir": cache,
        }
        if revision is not None:
            download_kwargs["revision"] = revision

        local_path = Path(
            snapshot_download(
                **download_kwargs,
                allow_patterns=["onnx/model_quantized.onnx", *_SIDECARS],
            )
        )
        if not (local_path / "onnx" / "model_quantized.onnx").exists():
            local_path = Path(
                snapshot_download(
                    **download_kwargs,
                    allow_patterns=["*.onnx", "onnx/*.onnx", *_SIDECARS],
                )
            )
        return local_path

    def _build_artifact(self, local_path: Path) -> ModelArtifact:
        try:
            import onnxruntime  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Required runtime dependency missing — "
                "reinstall bastion-prompt-protection to repair: "
                "pip install --force-reinstall bastion-prompt-protection"
            ) from exc

        onnx_candidates = [local_path / rel for rel in _ONNX_CANDIDATES]
        onnx_path = next((p for p in onnx_candidates if p.exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(
                f"No ONNX weights found in {local_path}. "
                f"Looked for: {[str(p.relative_to(local_path)) for p in onnx_candidates]}"
            )

        tokenizer_path = local_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {local_path}")

        from bastion_prompt_protection.models.tokenizer import BastionTokenizer

        tokenizer = BastionTokenizer(str(tokenizer_path))

        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        labels = _load_labels(local_path)

        return ModelArtifact(
            session=session,
            tokenizer=tokenizer,
            labels=labels,
            model_dir=local_path,
        )


def _load_labels(model_dir: Path) -> list[str]:
    labels_file = model_dir / "labels.txt"
    if labels_file.exists():
        return [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
    return []
