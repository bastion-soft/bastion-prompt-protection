"""OpenAI Agents SDK integration for Bastion Prompt Protection.

Install::

    pip install "bastion-prompt-protection[openai-agents]"

This module provides an input guardrail that screens user messages for prompt
injection and jailbreak attempts before the OpenAI Agents SDK runs your agent.

Quick start
-----------
``make_input_guardrail`` (factory -- recommended)::

    from agents import Agent
    from bastion_prompt_protection.integrations.openai_agents import make_input_guardrail

    guardrail = make_input_guardrail()          # free tiny model, auto-download
    agent = Agent(
        name="my-agent",
        instructions="You are a helpful assistant.",
        input_guardrails=[guardrail],
    )

``BastionInputGuardrail`` (class -- for full control)::

    from agents import Agent
    from bastion_prompt_protection import GuardOptions, Preset
    from bastion_prompt_protection.integrations.openai_agents import BastionInputGuardrail

    bg = BastionInputGuardrail(config=Preset.MULTILINGUAL, threshold=0.6)
    agent = Agent(
        name="my-agent",
        instructions="You are a helpful assistant.",
        input_guardrails=[bg.as_guardrail()],
    )
"""

from __future__ import annotations

from typing import Any

from bastion_prompt_protection import Guard, GuardOptions, GuardResult, Preset
from bastion_prompt_protection.errors import PromptInjectionError

try:
    from agents import Agent
    from agents.exceptions import InputGuardrailTripwireTriggered  # noqa: F401 (re-exported)
    from agents.guardrail import GuardrailFunctionOutput, InputGuardrail
    from agents.items import TResponseInputItem
    from agents.run_context import RunContextWrapper
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The OpenAI Agents SDK is required for this integration.  Install it with: "
        'pip install "bastion-prompt-protection[openai-agents]"'
    ) from exc

__all__ = ["BastionInputGuardrail", "make_input_guardrail", "PromptInjectionError"]


class BastionInputGuardrail:
    """A Bastion guardrail wrapper for the OpenAI Agents SDK."""

    def __init__(
        self,
        guard: Guard | None = None,
        *,
        threshold: float | None = None,
        config: GuardOptions | Preset | str | None = None,
        name: str = "bastion_input_guardrail",
        run_in_parallel: bool = True,
    ) -> None:
        """
        Args:
            guard: A pre-built :class:`Guard`. If omitted, one is created from
                ``config``.
            threshold: Override the attack decision threshold.
            config: Forwarded to :class:`Guard` when ``guard`` is None.
            name: Guardrail name surfaced in OpenAI Agents SDK traces.
            run_in_parallel: Whether the guardrail runs concurrently with the
                agent (``True``, default) or strictly before it (``False``).
        """
        self._guard = guard or Guard(config)
        self._threshold = threshold
        self._name = name
        self._run_in_parallel = run_in_parallel

    # -- public helpers -------------------------------------------------------

    def detect(self, text: str) -> GuardResult:
        """Run Bastion on ``text`` and return the raw verdict (never raises)."""
        return self._guard.protect(text)

    def as_guardrail(self) -> InputGuardrail[Any]:
        """Return an :class:`agents.InputGuardrail` for ``Agent(input_guardrails=[...])``."""
        _self = self

        async def _bastion_guardrail_fn(
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            input: str | list[TResponseInputItem],
        ) -> GuardrailFunctionOutput:
            text = _extract_text(input)
            result = _self._guard.protect(text)
            triggered = _self._is_attack(result)
            return GuardrailFunctionOutput(
                tripwire_triggered=triggered,
                output_info=result,
            )

        return InputGuardrail(
            guardrail_function=_bastion_guardrail_fn,
            name=self._name,
            run_in_parallel=self._run_in_parallel,
        )

    # -- internals ------------------------------------------------------------

    def _is_attack(self, result: GuardResult) -> bool:
        if self._threshold is not None:
            return result.risk >= self._threshold
        return result.is_attack


def make_input_guardrail(
    guard: Guard | None = None,
    *,
    threshold: float | None = None,
    config: GuardOptions | Preset | str | None = None,
    name: str = "bastion_input_guardrail",
    run_in_parallel: bool = True,
) -> InputGuardrail[Any]:
    """Create and return an :class:`agents.InputGuardrail` ready for ``Agent(input_guardrails=[...])``.

    This is the recommended one-liner API::

        from agents import Agent
        from bastion_prompt_protection.integrations.openai_agents import make_input_guardrail

        agent = Agent(
            name="my-agent",
            instructions="You are a helpful assistant.",
            input_guardrails=[make_input_guardrail()],
        )
    """
    return BastionInputGuardrail(
        guard=guard,
        threshold=threshold,
        config=config,
        name=name,
        run_in_parallel=run_in_parallel,
    ).as_guardrail()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_text(input: str | list[TResponseInputItem]) -> str:
    if isinstance(input, str):
        return input

    if isinstance(input, list):
        for item in reversed(input):
            if _item_role(item) in ("user", "human"):
                text = _item_text(item)
                if text:
                    return text
        for item in reversed(input):
            text = _item_text(item)
            if text:
                return text
        parts: list[str] = []
        for item in input:
            t = _item_text(item)
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(input)


def _item_role(item: Any) -> str:
    role = item.get("role", "") if isinstance(item, dict) else getattr(item, "role", "")
    return role.lower() if isinstance(role, str) else ""


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content", item.get("text", ""))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p)
    content = getattr(item, "content", None) or getattr(item, "text", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                t = getattr(block, "text", None)
                if t:
                    parts.append(str(t))
        return "\n".join(p for p in parts if p)
    return ""
