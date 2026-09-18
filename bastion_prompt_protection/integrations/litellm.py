"""LiteLLM Proxy integration for Bastion Prompt Protection.

Install::

    pip install "bastion-prompt-protection[litellm]"

This module ships ``BastionGuardrailPlugin``, a
``litellm.integrations.custom_guardrail.CustomGuardrail`` subclass that plugs
into the LiteLLM Proxy via a single ``config.yaml`` stanza.

Loading the plugin
------------------
LiteLLM resolves a custom guardrail's dotted path as a **file relative to the
config directory** — it does not import installed packages by dotted path. So
drop a one-line shim next to your ``config.yaml`` and reference *that*::

    # bastion_guardrail.py (next to config.yaml)
    from bastion_prompt_protection.integrations.litellm import BastionGuardrailPlugin

Quick start — ``config.yaml``::

    model_list:
      - model_name: gpt-4o-mini
        litellm_params:
          model: openai/gpt-4o-mini
          api_key: os.environ/OPENAI_API_KEY

    guardrails:
      - guardrail_name: bastion-injection-guard
        litellm_params:
          guardrail: bastion_guardrail.BastionGuardrailPlugin
          mode: pre_call
          default_on: true
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bastion_prompt_protection import Guard, GuardOptions, GuardResult, Preset
from bastion_prompt_protection.errors import ModelUnavailableError, PromptInjectionError

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "LiteLLM is required for this integration. Install it with: "
        'pip install "bastion-prompt-protection[litellm]"'
    ) from exc

__all__ = ["BastionGuardrailPlugin", "PromptInjectionError"]

_DEFAULT_VIOLATION_MESSAGE = (
    "I can't help with that request: it was flagged as a potential "
    "prompt-injection attempt and blocked."
)


class BastionGuardrailPlugin(CustomGuardrail):
    """LiteLLM Proxy guardrail plugin that screens requests for prompt injection."""

    def __init__(
        self,
        guard: Guard | None = None,
        *,
        config: GuardOptions | Preset | str | None = None,
        threshold: float | None = None,
        block: bool = True,
        screen_tool_results: bool = True,
        screen_output: bool = False,
        violation_message: str = _DEFAULT_VIOLATION_MESSAGE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._guard = guard or Guard(config)
        self._threshold = threshold
        self._block = block
        self._screen_tool_results = screen_tool_results
        self._screen_output = screen_output
        self._violation_message = violation_message

    # -- public helper -------------------------------------------------------

    def detect(self, text: str) -> GuardResult:
        """Run Bastion on ``text`` and return the raw verdict (never raises)."""
        return self._guard.protect(text)

    # -- pre-call hook -------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):
            return data

        messages: list[dict] = data.get("messages") or []

        try:
            for text, _origin, _vector in _screenable_texts(messages, self._screen_tool_results):
                result = self._guard.protect(text)
                if self._is_attack(result):
                    if self._block:
                        _raise_rejected(self._format(result), data)
                    break
        except ModelUnavailableError:
            _raise_service_unavailable()

        return data

    # -- post-call hook (opt-in) ---------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        if not self._screen_output:
            return

        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call):
            return

        try:
            import litellm

            if isinstance(response, litellm.ModelResponse):
                for choice in response.choices:
                    if isinstance(choice, litellm.Choices):
                        content = choice.message.content
                        if content and isinstance(content, str):
                            result = self._guard.protect(content)
                            if self._is_attack(result):
                                raise ValueError(self._format(result))
        except ImportError:  # pragma: no cover
            pass

    # -- internals -----------------------------------------------------------

    def _is_attack(self, result: GuardResult) -> bool:
        if self._threshold is not None:
            return result.risk >= self._threshold
        return result.is_attack

    def _format(self, result: GuardResult) -> str:
        try:
            return self._violation_message.format(risk=result.risk, stage=result.stage_reached)
        except (KeyError, IndexError):
            return self._violation_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text_val = block.get("text") or block.get("content", "")
                if text_val and isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


def _screenable_texts(
    messages: list[dict],
    screen_tool_results: bool,
) -> Iterator[tuple[str, str, str]]:
    last_user_text: str | None = None
    for msg in reversed(messages):
        role = (msg.get("role") or "").lower()
        if role in ("user", "human"):
            text = _message_text(msg)
            if text:
                last_user_text = text
            break

    if last_user_text is not None:
        yield last_user_text, "user_prompt", "direct"

    if screen_tool_results:
        for msg in messages:
            role = (msg.get("role") or "").lower()
            if role in ("tool", "function"):
                text = _message_text(msg)
                if text:
                    yield text, "tool_result", "indirect"


def _raise_rejected(message: str, data: dict) -> None:
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail={"error": message})


def _raise_service_unavailable() -> None:
    from fastapi import HTTPException

    raise HTTPException(status_code=503, detail={"error": "model unavailable"})
