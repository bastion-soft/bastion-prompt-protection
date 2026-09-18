"""Unit tests for the two ``on_model_unavailable`` modes.

- ``"throw"``: first download failure is permanent; every subsequent call
  throws immediately without retrying.
- ``"try-download-then-throw"`` (default): throws on failure but retries the
  download after a 5-second cooldown; self-heals once a download succeeds.

These tests mock ``OnnxModelLoader._do_load`` so no network or HF cache is needed.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bastion_prompt_protection.errors import ModelUnavailableError
from bastion_prompt_protection.models.loader import ModelArtifact, OnnxModelLoader

_FAKE_ARTIFACT = ModelArtifact(
    session=MagicMock(),
    tokenizer=MagicMock(),
    labels=[],
    model_dir=Path("/cache/test-model/snapshots/abc1234def5678"),
)


def make_loader(mode: str) -> OnnxModelLoader:
    return OnnxModelLoader("owner/model", on_model_unavailable=mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mode: "throw"
# ---------------------------------------------------------------------------


class TestThrowMode:
    def test_raises_on_first_failure(self) -> None:
        loader = make_loader("throw")
        with patch.object(loader, "_do_load", side_effect=OSError("connection refused")):
            with pytest.raises(ModelUnavailableError):
                loader.load()

    def test_carries_original_cause(self) -> None:
        loader = make_loader("throw")
        cause = OSError("connection refused")
        with patch.object(loader, "_do_load", side_effect=cause):
            with pytest.raises(ModelUnavailableError) as exc_info:
                loader.load()
        assert exc_info.value.cause is cause

    def test_caches_failure_permanently(self) -> None:
        loader = make_loader("throw")
        call_count = 0

        def fail():
            nonlocal call_count
            call_count += 1
            raise OSError("network error")

        with patch.object(loader, "_do_load", side_effect=fail):
            with pytest.raises(ModelUnavailableError):
                loader.load()

        # Even after a long wait, it should not retry.
        for _ in range(3):
            with pytest.raises(ModelUnavailableError):
                loader.load()

        assert call_count == 1, "loader should not retry after permanent failure"

    def test_resolves_normally_on_success(self) -> None:
        loader = make_loader("throw")
        with patch.object(loader, "_do_load", return_value=_FAKE_ARTIFACT):
            result = loader.load()
        assert result is _FAKE_ARTIFACT


# ---------------------------------------------------------------------------
# mode: "try-download-then-throw" (default)
# ---------------------------------------------------------------------------


class TestTryDownloadThenThrow:
    def test_raises_on_first_failure(self) -> None:
        loader = make_loader("try-download-then-throw")
        with patch.object(loader, "_do_load", side_effect=OSError("connection refused")):
            with pytest.raises(ModelUnavailableError):
                loader.load()

    def test_throws_during_cooldown_without_retrying(self) -> None:
        loader = make_loader("try-download-then-throw")
        call_count = 0

        def fail():
            nonlocal call_count
            call_count += 1
            raise OSError("network error")

        with patch.object(loader, "_do_load", side_effect=fail):
            with pytest.raises(ModelUnavailableError):
                loader.load()

        # Simulate being inside the 5s cooldown window.
        with patch("bastion_prompt_protection.models.loader.time") as mock_time:
            mock_time.monotonic.return_value = loader._failure[1] + 1.0  # type: ignore[index]
            with pytest.raises(ModelUnavailableError):
                loader.load()

        assert call_count == 1, "should not retry during cooldown"

    def test_hint_message_during_cooldown(self) -> None:
        loader = make_loader("try-download-then-throw")
        with patch.object(loader, "_do_load", side_effect=OSError("network error")):
            with pytest.raises(ModelUnavailableError):
                loader.load()

        with patch("bastion_prompt_protection.models.loader.time") as mock_time:
            mock_time.monotonic.return_value = loader._failure[1] + 2.0  # type: ignore[index]
            with pytest.raises(ModelUnavailableError, match=r"retry in \d+s"):
                loader.load()

    def test_retries_after_cooldown_and_succeeds(self) -> None:
        loader = make_loader("try-download-then-throw")
        call_count = 0

        def succeed_on_second():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network error")
            return _FAKE_ARTIFACT

        with patch.object(loader, "_do_load", side_effect=succeed_on_second):
            with pytest.raises(ModelUnavailableError):
                loader.load()

            # Move past the cooldown.
            failed_at = loader._failure[1]  # type: ignore[index]
            with patch("bastion_prompt_protection.models.loader.time") as mock_time:
                mock_time.monotonic.return_value = failed_at + 6.0
                result = loader.load()

        assert result is _FAKE_ARTIFACT
        assert call_count == 2

    def test_self_heals_after_success(self) -> None:
        loader = make_loader("try-download-then-throw")
        call_count = 0

        def succeed_on_second():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("transient")
            return _FAKE_ARTIFACT

        with patch.object(loader, "_do_load", side_effect=succeed_on_second):
            with pytest.raises(ModelUnavailableError):
                loader.load()

            failed_at = loader._failure[1]  # type: ignore[index]
            with patch("bastion_prompt_protection.models.loader.time") as mock_time:
                mock_time.monotonic.return_value = failed_at + 6.0
                loader.load()

        # After self-heal, further calls should not invoke _do_load again.
        result = loader.load()
        assert result is _FAKE_ARTIFACT
        assert call_count == 2

    def test_persistent_failure_across_cooldown_cycles(self) -> None:
        loader = make_loader("try-download-then-throw")
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise OSError("persistent error")

        with patch.object(loader, "_do_load", side_effect=always_fail):
            # First failure.
            with pytest.raises(ModelUnavailableError):
                loader.load()

            # After cooldown — retry and fail again.
            with patch("bastion_prompt_protection.models.loader.time") as mock_time:
                mock_time.monotonic.return_value = loader._failure[1] + 6.0  # type: ignore[index]
                with pytest.raises(ModelUnavailableError):
                    loader.load()

            # After second cooldown — retry and fail again.
            with patch("bastion_prompt_protection.models.loader.time") as mock_time:
                mock_time.monotonic.return_value = loader._failure[1] + 6.0  # type: ignore[index]
                with pytest.raises(ModelUnavailableError):
                    loader.load()

        assert call_count == 3
