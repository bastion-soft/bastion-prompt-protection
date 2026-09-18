"""Unit tests for cache-first model loading (no Hub calls when cache is warm).

These tests use filesystem fixtures and mocks — no network, no real ONNX.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bastion_prompt_protection.models.loader import (
    OnnxModelLoader,
    _find_cached_snapshot,
    _repo_folder_name,
    _snapshot_is_loadable,
)

_REPO = "owner/model"
_SHA_OLD = "a" * 40
_SHA_NEW = "b" * 40


def _repo_cache_root(cache_dir: Path) -> Path:
    return cache_dir / _repo_folder_name(_REPO)


def _snapshot_dir(cache_dir: Path, sha: str) -> Path:
    return _repo_cache_root(cache_dir) / "snapshots" / sha


def _write_loadable_snapshot(snapshot: Path) -> None:
    (snapshot / "onnx").mkdir(parents=True, exist_ok=True)
    (snapshot / "onnx" / "model_quantized.onnx").write_bytes(b"onnx")
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}')


def _patch_runtime(snapshot_download: MagicMock | None = None):
    """Patch heavy deps so _do_load can run against fake files."""
    mock_sd = snapshot_download if snapshot_download is not None else MagicMock()
    mock_session = MagicMock()
    mock_tokenizer = MagicMock()
    return (
        patch("onnxruntime.InferenceSession", return_value=mock_session),
        patch(
            "bastion_prompt_protection.models.tokenizer.BastionTokenizer",
            return_value=mock_tokenizer,
        ),
        patch("huggingface_hub.snapshot_download", mock_sd),
    )


class TestCacheHelpers:
    def test_repo_folder_name(self) -> None:
        assert _repo_folder_name("owner/model") == "models--owner--model"

    def test_snapshot_is_loadable_requires_tokenizer_and_onnx(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap"
        snap.mkdir()
        assert not _snapshot_is_loadable(snap)

        (snap / "tokenizer.json").write_text("{}")
        assert not _snapshot_is_loadable(snap)

        (snap / "onnx").mkdir()
        (snap / "onnx" / "model.onnx").write_bytes(b"x")
        assert _snapshot_is_loadable(snap)

    def test_find_cached_snapshot_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _find_cached_snapshot(tmp_path, _REPO) is None

    def test_find_cached_snapshot_picks_newest_mtime(self, tmp_path: Path) -> None:
        old = _snapshot_dir(tmp_path, _SHA_OLD)
        new = _snapshot_dir(tmp_path, _SHA_NEW)
        old.mkdir(parents=True)
        new.mkdir(parents=True)

        # Ensure distinct mtimes.
        os.utime(old, (time.time() - 100, time.time() - 100))
        os.utime(new, (time.time(), time.time()))

        assert _find_cached_snapshot(tmp_path, _REPO) == new


class TestWarmCacheSkipsHub:
    def test_complete_snapshot_never_calls_snapshot_download(self, tmp_path: Path) -> None:
        snap = _snapshot_dir(tmp_path, _SHA_OLD)
        snap.mkdir(parents=True)
        _write_loadable_snapshot(snap)

        mock_sd = MagicMock()
        patches = _patch_runtime(mock_sd)
        loader = OnnxModelLoader(_REPO, cache_dir=tmp_path)  # type: ignore[arg-type]

        with patches[0], patches[1], patches[2]:
            artifact = loader.load()

        mock_sd.assert_not_called()
        assert artifact.model_dir == snap
        assert loader.revision == _SHA_OLD

    def test_uses_custom_cache_dir_not_default(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom-cache"
        other = tmp_path / "other-cache"
        snap = _snapshot_dir(custom, _SHA_OLD)
        snap.mkdir(parents=True)
        _write_loadable_snapshot(snap)

        # Same SHA under a different cache root must not be picked up.
        wrong = _snapshot_dir(other, _SHA_NEW)
        wrong.mkdir(parents=True)
        _write_loadable_snapshot(wrong)

        mock_sd = MagicMock()
        patches = _patch_runtime(mock_sd)
        loader = OnnxModelLoader(_REPO, cache_dir=custom)  # type: ignore[arg-type]

        with patches[0], patches[1], patches[2]:
            artifact = loader.load()

        mock_sd.assert_not_called()
        assert artifact.model_dir == snap


class TestEmptyCacheDownloads:
    def test_no_snapshot_triggers_snapshot_download(self, tmp_path: Path) -> None:
        # No snapshots/ dir yet — download must run. Files appear only in the
        # path snapshot_download returns (simulating a fresh Hub fetch).
        snap = tmp_path / "downloaded-snapshot"
        snap.mkdir()
        _write_loadable_snapshot(snap)

        mock_sd = MagicMock(return_value=str(snap))
        patches = _patch_runtime(mock_sd)
        loader = OnnxModelLoader(_REPO, cache_dir=tmp_path)  # type: ignore[arg-type]

        with patches[0], patches[1], patches[2]:
            loader.load()

        assert mock_sd.call_count >= 1
        first_kwargs = mock_sd.call_args_list[0].kwargs
        assert first_kwargs["repo_id"] == _REPO
        assert first_kwargs.get("cache_dir") == str(tmp_path)
        assert "revision" not in first_kwargs


class TestIncompleteSnapshotRepair:
    def test_incomplete_snapshot_pins_revision(self, tmp_path: Path) -> None:
        snap = _snapshot_dir(tmp_path, _SHA_OLD)
        snap.mkdir(parents=True)
        # ONNX present but tokenizer missing — not loadable.
        (snap / "onnx").mkdir()
        (snap / "onnx" / "model_quantized.onnx").write_bytes(b"onnx")

        def simulate_repair(**kwargs: object) -> str:
            _write_loadable_snapshot(snap)
            return str(snap)

        mock_sd = MagicMock(side_effect=simulate_repair)
        patches = _patch_runtime(mock_sd)
        loader = OnnxModelLoader(_REPO, cache_dir=tmp_path)  # type: ignore[arg-type]

        with patches[0], patches[1], patches[2]:
            loader.load()

        assert mock_sd.call_count >= 1
        first_kwargs = mock_sd.call_args_list[0].kwargs
        assert first_kwargs.get("revision") == _SHA_OLD
