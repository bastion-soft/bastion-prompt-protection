"""OpenAI Agents SDK integration for Bastion Prompt Protection.

Install::

    pip install "bastion-prompt-protection[openai-agents]"

This module provides an input guardrail that screens user messages for prompt
injection and jailbreak attempts before the OpenAI Agents SDK runs your agent.

Licensing note
--------------
This integration runs Bastion **in-process** inside your application.  Under the
AGPL-3.0 licence, distributing or providing access to a modified version of the
library requires publishing the source.  Commercial / closed-source deployments
should obtain a commercial licence from BastionSoft — see
https://github.com/bastion-soft/bastion-prompt-protection for details.

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
    from bastion_prompt_protection import GuardConfig, Preset
    from bastion_prompt_protection.integrations.openai_agents import BastionInputGuardrail

    bg = BastionInputGuardrail(preset=Preset.MULTILINGUAL, threshold=0.6)
    agent = Agent(
        name="my-agent",
        instructions="You are a helpful assistant.",
        input_guardrails=[bg.as_guardrail()],
    )

    # Or call directly (useful in tests / standalone screening):
    result = bg.detect("What is the capital of France?")
    print(result.is_attack)   # False

Phase 2 (not yet implemented): an output guardrail variant (``BastionOutputGuardrail``)
for post-generation screening of agent replies.  Track progress at
https://github.com/bastion-soft/bastion-prompt-protection/issues.
"""

from __future__ import annotations

from typing import Any

from bastion_prompt_protection import Guard, GuardConfig, GuardResult, Preset, ReportContext
from bastion_prompt_protection.exceptions import PromptInjectionError
from bastion_prompt_protection.telemetry import Reporter, default_reporter, make_record

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
    """A Bastion guardrail wrapper for the OpenAI Agents SDK.

    Holds a :class:`~bastion_prompt_protection.Guard` and exposes:

    * :meth:`detect` -- run Bastion on arbitrary text; never raises.
    * :meth:`as_guardrail` -- return an :class:`agents.InputGuardrail` for use in
      ``Agent(input_guardrails=[...])``.

    Use :func:`make_input_guardrail` for the simple one-liner API, or instantiate
    this class when you need to share one ``Guard`` across several agents or call
    :meth:`detect` outside the SDK lifecycle.
    """

    def __init__(
        self,
        guard: Guard | None = None,
        *,
        threshold: float | None = None,
        preset: str | Preset = Preset.TINY,
        config: GuardConfig | None = None,
        name: str = "bastion_input_guardrail",
        run_in_parallel: bool = True,
        reporter: Reporter | None = None,
    ) -> None:
        """
        Args:
            guard: A pre-built :class:`Guard`. If omitted, one is created from
                ``preset`` / ``config``.
            threshold: Override the attack decision threshold (``risk >= threshold``
                implies attack). When ``None`` the Guard's own ``attack_above``
                setting is used.
            preset / config: Forwarded to :class:`Guard` when ``guard`` is None.
            name: Guardrail name surfaced in OpenAI Agents SDK traces.
            run_in_parallel: Whether the guardrail runs concurrently with the
                agent (``True``, default) or strictly before it (``False``).
            reporter: Telemetry reporter (composed in, not coupled to Guard).
                Defaults to the env-configured reporter (no-op unless set).
        """
        self._guard = guard or Guard(preset=preset, config=config)
        self._threshold = threshold
        self._name = name
        self._run_in_parallel = run_in_parallel
        self._reporter = reporter or default_reporter()

    # -- public helpers -------------------------------------------------------

    def detect(self, text: str) -> GuardResult:
        """Run Bastion on ``text`` and return the raw verdict (never raises)."""
        return self._guard.protect(text)

    def as_guardrail(self) -> InputGuardrail[Any]:
        """Return an :class:`agents.InputGuardrail` for use in ``Agent(input_guardrails=[...])``.

        Example::

            agent = Agent(
                name="my-agent",
                instructions="You are a helpful assistant.",
                input_guardrails=[bg.as_guardrail()],
            )

        When a prompt-injection attempt is detected the SDK raises
        :class:`agents.InputGuardrailTripwireTriggered`; the triggering
        :class:`~bastion_prompt_protection.GuardResult` is available as
        ``exc.guardrail_result.output.output_info``.
        """
        # Capture self explicitly to avoid stale-closure surprises in
        # multi-instance scenarios.
        _self = self

        async def _bastion_guardrail_fn(
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            input: str | list[TResponseInputItem],
        ) -> GuardrailFunctionOutput:
            text = _extract_text(input)
            result = _self._guard.protect(text)
            _self._reporter.report(
                make_record(
                    result,
                    ReportContext(
                        vector="direct", origin="user_prompt", source="openai-agents", content=text
                    ),
                    _self._guard,
                )
            )
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
    preset: str | Preset = Preset.TINY,
    config: GuardConfig | None = None,
    name: str = "bastion_input_guardrail",
    run_in_parallel: bool = True,
    reporter: Reporter | None = None,
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

    When a prompt-injection attempt is detected the OpenAI Agents SDK raises
    :class:`agents.InputGuardrailTripwireTriggered`; the triggering
    :class:`~bastion_prompt_protection.GuardResult` is available as
    ``exc.guardrail_result.output.output_info``.

    Args:
        guard: A pre-built :class:`Guard`. If omitted, one is created from
            ``preset`` / ``config``.
        threshold: Override the attack decision threshold (``risk >= threshold``
            implies attack). When ``None`` the Guard's own ``attack_above``
            setting is used.
        preset / config: Forwarded to :class:`Guard` when ``guard`` is None.
        name: Guardrail name surfaced in OpenAI Agents SDK traces.
        run_in_parallel: Whether the guardrail runs concurrently with the
            agent (``True``, default) or strictly before it (``False``).

    Returns:
        An :class:`agents.InputGuardrail` instance.
    """
    return BastionInputGuardrail(
        guard=guard,
        threshold=threshold,
        preset=preset,
        config=config,
        name=name,
        run_in_parallel=run_in_parallel,
        reporter=reporter,
    ).as_guardrail()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_text(input: str | list[TResponseInputItem]) -> str:
    """Extract screenable text from a guardrail input.

    The OpenAI Agents SDK passes either a plain string (simple ``Runner.run``
    call) or a list of response-input items (multi-turn / structured input).
    We prefer the *last* user/human message (the turn being guarded); if no
    role-tagged user message is present we fall back to the most-recent item
    with text, then to joining all text blocks so nothing slips through.
    """
    if isinstance(input, str):
        return input

    if isinstance(input, list):
        # Prefer the most-recent user/human message — that is the turn the
        # input guardrail is meant to screen.
        for item in reversed(input):
            if _item_role(item) in ("user", "human"):
                text = _item_text(item)
                if text:
                    return text
        # No role-tagged user message: screen the most-recent item with text.
        for item in reversed(input):
            text = _item_text(item)
            if text:
                return text
        # Fallback: join all text from all items.
        parts: list[str] = []
        for item in input:
            t = _item_text(item)
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(input)


def _item_role(item: Any) -> str:
    """Return the lower-cased ``role`` of a TResponseInputItem (dict or object)."""
    role = item.get("role", "") if isinstance(item, dict) else getattr(item, "role", "")
    return role.lower() if isinstance(role, str) else ""


def _item_text(item: Any) -> str:
    """Pull text out of a single TResponseInputItem (dict or object)."""
    # Items are typically TypedDicts with a ``type`` field and ``content`` or ``text``.
    if isinstance(item, dict):
        content = item.get("content", item.get("text", ""))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Content blocks: [{"type": "input_text", "text": "..."}, ...]
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p)
    # Pydantic / dataclass objects.
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
