from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Sequence

from bastion_prompt_protection.constants import (
    CONTENT_TOKEN_WINDOW,
    DEFAULT_SLAB_CHARS,
    DEFAULT_TOKEN_OVERLAP,
    MODEL_TOKEN_WINDOW,
    SPECIAL_TOKEN_BUDGET,
    estimate_window_count,
)


@dataclass
class Encoding:
    ids: list[int]
    attention_mask: list[int]


@dataclass
class TokenWindow:
    encoding: Encoding
    windows_total: int
    windows_total_exact: bool


def slabify(text: str, slab_chars: int = DEFAULT_SLAB_CHARS) -> Generator[str, None, None]:
    """Split *text* into contiguous character slabs for bounded tokenization.

    Slabs are non-overlapping and lossless: ``"".join(slabify(text)) == text``.
    Each slab is at most *slab_chars* characters. When possible the cut lands on
    whitespace so the concatenated token stream matches a single ``encode()``
    call. Whitespace-free runs (base64, minified JSON, HTML) are hard-cut at
    the limit.
    """
    limit = max(1, slab_chars)
    if not text:
        return

    pos = 0
    while pos < len(text):
        end = min(pos + limit, len(text))
        if end < len(text):
            ws_cut = -1
            for i in range(end, pos, -1):
                if text[i - 1].isspace():
                    ws_cut = i
                    break
            if ws_cut > pos:
                end = ws_cut
        yield text[pos:end]
        pos = end


class BastionTokenizer:
    """Wraps the ``tokenizers`` library to provide token-exact sliding windows.

    Python's ``tokenizers.Tokenizer.encode()`` already honours the ``truncation``
    block embedded in ``tokenizer.json``, so the JavaScript truncation shim is not
    needed here.
    """

    def __init__(self, tokenizer_path: str) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]

        self._tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

        # Discover the special token ids by encoding an empty string.
        special = self._tokenizer.encode("")
        special_ids: list[int] = list(special.ids)
        if len(special_ids) < 2:
            raise RuntimeError(
                "tokenizer.encode('') must yield at least two special token ids"
            )
        self._cls_id: int = special_ids[0]
        self._sep_id: int = special_ids[-1]

    def encode(self, text: str) -> Encoding:
        """Full encode with special tokens (honours tokenizer.json truncation)."""
        enc = self._tokenizer.encode(text)
        return Encoding(ids=list(enc.ids), attention_mask=list(enc.attention_mask))

    def encode_content(self, text: str) -> list[int]:
        """Tokenize content without ``[CLS]`` / ``[SEP]``."""
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return list(enc.ids)

    def windows(
        self,
        text: str,
        *,
        window_tokens: int = MODEL_TOKEN_WINDOW,
        overlap_tokens: int = DEFAULT_TOKEN_OVERLAP,
        slab_chars: int = DEFAULT_SLAB_CHARS,
    ) -> Generator[TokenWindow, None, None]:
        """Lazily yield token-exact sliding windows over *text*.

        Character slabs keep tokenization linear; overlap and window size are
        measured in content tokens, not characters.
        """
        content_window = window_tokens - SPECIAL_TOKEN_BUDGET
        if content_window <= 0:
            raise ValueError(
                f"window_tokens must exceed SPECIAL_TOKEN_BUDGET ({SPECIAL_TOKEN_BUDGET})"
            )
        step = max(1, content_window - overlap_tokens)

        stream: list[int] = []
        chars_consumed = 0
        input_length = len(text)
        pos = 0

        def project_total(exact: bool) -> int:
            if exact:
                return estimate_window_count(len(stream), content_window, overlap_tokens)
            if not stream or chars_consumed == 0:
                return 1
            density = len(stream) / chars_consumed
            projected_tokens = int(density * input_length)
            return estimate_window_count(projected_tokens, content_window, overlap_tokens)

        def wrap(content: list[int]) -> Encoding:
            ids = [self._cls_id, *content, self._sep_id]
            return Encoding(ids=ids, attention_mask=[1] * len(ids))

        def emit_at(start: int, exact: bool) -> TokenWindow:
            return TokenWindow(
                encoding=wrap(stream[start : start + content_window]),
                windows_total=project_total(exact),
                windows_total_exact=exact,
            )

        for slab in slabify(text, slab_chars):
            stream.extend(self.encode_content(slab))
            chars_consumed += len(slab)

            while len(stream) >= pos + content_window + step:
                yield emit_at(pos, False)
                pos += step

        if not stream:
            return

        if len(stream) <= content_window:
            yield emit_at(0, True)
            return

        while True:
            start = (
                len(stream) - content_window
                if pos + content_window >= len(stream)
                else pos
            )
            is_last = start + content_window >= len(stream)
            yield emit_at(start, is_last)
            if is_last:
                break
            pos += step

    def slabs(self, text: str, slab_chars: int = DEFAULT_SLAB_CHARS) -> list[str]:
        """Return the slab list for *text*. Exposed for tests."""
        return list(slabify(text, slab_chars))
