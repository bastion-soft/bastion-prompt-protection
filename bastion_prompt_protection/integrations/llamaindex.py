"""LlamaIndex integration for Bastion Prompt Protection.

Install::

    pip install "bastion-prompt-protection[llamaindex]"

Screen a RAG pipeline for prompt injection as a **node postprocessor** — it runs
after retrieval and before response synthesis, so it can check both the user's
query (direct injection) and the *retrieved* content (indirect injection — a
malicious instruction hidden in a document your retriever pulled in)::

    from bastion_prompt_protection.integrations.llamaindex import BastionGuardrailPostprocessor

    query_engine = index.as_query_engine(
        node_postprocessors=[BastionGuardrailPostprocessor()],
    )
    query_engine.query("…")   # raises PromptInjectionError if the query or a
                              # retrieved node is flagged

By default a detected injection raises ``PromptInjectionError``. Set
``block=False`` to instead **drop** flagged retrieved nodes (so poisoned
documents never reach the LLM) without aborting the query.
"""

from __future__ import annotations

from typing import Any

from bastion_prompt_protection import Guard, GuardConfig, GuardResult, Preset
from bastion_prompt_protection.exceptions import PromptInjectionError

try:
    from llama_index.core import QueryBundle
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "LlamaIndex is required for this integration. Install it with: "
        'pip install "bastion-prompt-protection[llamaindex]"'
    ) from exc

__all__ = ["BastionGuardrailPostprocessor", "PromptInjectionError"]


class BastionGuardrailPostprocessor(BaseNodePostprocessor):
    """A LlamaIndex node postprocessor that screens a RAG pipeline for injection.

    Drop it into a query engine's ``node_postprocessors``; it runs after
    retrieval and before response synthesis, screening both the query and the
    retrieved nodes::

        index.as_query_engine(node_postprocessors=[BastionGuardrailPostprocessor()])

    - ``block=True`` (default): raise :class:`PromptInjectionError` if the query
      or any retrieved node is flagged.
    - ``block=False``: don't raise — drop flagged retrieved nodes so poisoned
      content can't reach the LLM (the query is still screened, but only via
      ``detect()`` / not enforced, since a postprocessor can't abort the query).
    """

    block: bool = True
    threshold: float | None = None
    screen_query: bool = True
    screen_nodes: bool = True

    _guard: Guard = PrivateAttr()

    def __init__(
        self,
        guard: Guard | None = None,
        *,
        block: bool = True,
        threshold: float | None = None,
        screen_query: bool = True,
        screen_nodes: bool = True,
        preset: str | Preset = Preset.TINY,
        config: GuardConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            guard: A pre-built :class:`Guard`. If omitted, one is created from
                ``preset`` / ``config``.
            block: Raise :class:`PromptInjectionError` on a flagged query or
                node (default). Set ``False`` to drop flagged nodes instead.
            threshold: Override the attack decision threshold (risk >= threshold
                ⇒ attack). Defaults to the Guard's own ``attack_above``.
            screen_query: Screen the incoming query string (direct injection).
            screen_nodes: Screen the retrieved nodes (indirect injection).
            preset / config: Forwarded to :class:`Guard` when ``guard`` is None.
        """
        super().__init__(
            block=block,
            threshold=threshold,
            screen_query=screen_query,
            screen_nodes=screen_nodes,
            **kwargs,
        )
        self._guard = guard or Guard(preset=preset, config=config)

    @classmethod
    def class_name(cls) -> str:
        return "BastionGuardrailPostprocessor"

    def detect(self, text: str) -> GuardResult:
        """Run Bastion on ``text`` and return the raw verdict (never raises)."""
        return self._guard.protect(text)

    def _is_attack(self, result: GuardResult) -> bool:
        if self.threshold is not None:
            return result.risk >= self.threshold
        return result.is_attack

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        # 1) Direct injection — screen the user's query.
        if self.screen_query and query_bundle is not None:
            q = self._guard.protect(query_bundle.query_str)
            if self._is_attack(q) and self.block:
                raise PromptInjectionError(q)

        # 2) Indirect injection — screen the retrieved content.
        if not self.screen_nodes:
            return nodes
        kept: list[NodeWithScore] = []
        for nws in nodes:
            result = self._guard.protect(nws.node.get_content())
            if self._is_attack(result):
                if self.block:
                    raise PromptInjectionError(result)
                continue  # block=False → drop the poisoned node
            kept.append(nws)
        return kept
