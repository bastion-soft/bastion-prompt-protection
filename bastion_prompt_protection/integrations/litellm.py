"""LiteLLM Proxy integration for Bastion Prompt Protection.

Install::

    pip install "bastion-prompt-protection[litellm]"

This module ships ``BastionGuardrailPlugin``, a
``litellm.integrations.custom_guardrail.CustomGuardrail`` subclass that plugs
into the LiteLLM Proxy via a single ``config.yaml`` stanza.  Because the proxy
runs as a standalone process, the AGPL license of ``bastion-prompt-protection``
does **not** propagate to your application code — the guardrail runs across a
network boundary.

Loading the plugin
------------------
LiteLLM resolves a custom guardrail's dotted path as a **file relative to the
config directory** — it does not import installed packages by dotted path. So
drop a one-line shim next to your ``config.yaml`` and reference *that*::

    # bastion_guardrail.py (next to config.yaml)
    from bastion_prompt_protection.integrations.litellm import BastionGuardrailPlugin

Running the proxy also needs the proxy server itself
(``pip install "litellm[proxy]"``) in addition to
``pip install "bastion-prompt-protection[litellm]"``.

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
          mode: pre_call          # screen input before the LLM call
          default_on: true        # protect every request automatically

Then start the proxy::

    litellm --config config.yaml

Any request whose last human/user message (or any tool-result message) is
flagged as a prompt-injection or jailbreak attempt is rejected with HTTP 400
before the LLM is ever called.

Advanced — screen model output too::

    guardrails:
      - guardrail_name: bastion-injection-guard
        litellm_params:
          guardrail: bastion_guardrail.BastionGuardrailPlugin
          mode: pre_call
          default_on: true
          screen_output: true   # also screen the LLM's response (default: false)

Advanced — custom threshold / preset / pass-through mode::

    guardrails:
      - guardrail_name: bastion-injection-guard
        litellm_params:
          guardrail: bastion_guardrail.BastionGuardrailPlugin
          mode: pre_call
          default_on: true
          threshold: 0.7           # tighter than the default 0.50 attack_above
          screen_tool_results: false  # skip screening tool/function messages
          block: false             # log-only mode: detect but don't block
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bastion_prompt_protection import Guard, GuardConfig, GuardResult, Preset, ReportContext
from bastion_prompt_protection.exceptions import PromptInjectionError
from bastion_prompt_protection.telemetry import Reporter, default_reporter, make_record

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
    """LiteLLM Proxy guardrail plugin that screens requests for prompt injection.

    Register it in your proxy ``config.yaml``::

        guardrails:
          - guardrail_name: bastion-injection-guard
            litellm_params:
              guardrail: bastion_guardrail.BastionGuardrailPlugin
              mode: pre_call
              default_on: true

    The plugin implements two hooks:

    ``async_pre_call_hook`` (mode: ``pre_call``, default)
        Runs **before** the LLM call.  Screens the last human/user message and
        (optionally) any tool/function-result messages to catch indirect
        injection smuggled through tool output.  A flagged message raises a
        ``fastapi.HTTPException`` (HTTP 400) so the LLM is never called.

    ``async_post_call_success_hook`` (mode: ``post_call``, opt-in)
        Runs **after** the LLM returns.  Screens the model's reply for injected
        content.  Enable it by passing ``screen_output=True`` **and** using
        ``mode: post_call`` (or ``["pre_call", "post_call"]``) in config.
        Disabled by default because the primary value of this guardrail is
        stopping malicious *input* before it reaches the model.

    Licensing note: the LiteLLM proxy runs as a separate process; your
    application code calls it over HTTP and is therefore **not** subject to
    AGPL licence propagation.  See https://www.gnu.org/licenses/agpl-3.0.html
    for the full licence text.
    """

    def __init__(
        self,
        guard: Guard | None = None,
        *,
        preset: str | Preset = Preset.TINY,
        config: GuardConfig | None = None,
        threshold: float | None = None,
        block: bool = True,
        screen_tool_results: bool = True,
        screen_output: bool = False,
        violation_message: str = _DEFAULT_VIOLATION_MESSAGE,
        reporter: Reporter | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            guard: A pre-built :class:`~bastion_prompt_protection.Guard`.  If
                omitted, one is created from ``preset`` / ``config``.
            preset / config: Forwarded to :class:`~bastion_prompt_protection.Guard`
                when ``guard`` is ``None``.
            threshold: Override the attack decision threshold (``risk >=
                threshold`` ⇒ attack).  Defaults to the Guard's own
                ``attack_above`` (0.50 for the TINY preset).
            block: When ``True`` (default) a flagged input raises
                ``fastapi.HTTPException`` (HTTP 400) and the LLM is never
                called.  Set ``False`` for log-only / observe mode — the
                request passes through but the detection result is still
                accessible via :meth:`detect`.
            screen_tool_results: Also screen ``tool`` / ``function`` role
                messages for indirect injection (default ``True``).  Disable
                this if tool results are already sanitised upstream.
            screen_output: Screen the LLM's reply in
                ``async_post_call_success_hook`` (default ``False``).  Enable
                if you want defence-in-depth on the *output* side, e.g. to
                catch jailbreak completions.
            violation_message: The error message returned to the caller on a
                blocked request.  Supports ``{risk}`` and ``{stage}``
                template fields.
            **kwargs: Forwarded verbatim to
                :class:`~litellm.integrations.custom_guardrail.CustomGuardrail`
                (e.g. ``guardrail_name``, ``default_on``, …).
        """
        super().__init__(**kwargs)
        self._guard = guard or Guard(preset=preset, config=config)
        self._threshold = threshold
        self._block = block
        self._screen_tool_results = screen_tool_results
        self._screen_output = screen_output
        self._violation_message = violation_message
        self._reporter = reporter or default_reporter()

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
        """Screen input messages before the LLM call.

        Returns the (possibly unmodified) ``data`` dict on a clean request, or
        raises ``fastapi.HTTPException`` (HTTP 400) when ``block=True`` and a
        prompt-injection attempt is detected.
        """
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):
            return data

        messages: list[dict] = data.get("messages") or []

        for text, origin, vector in _screenable_texts(messages, self._screen_tool_results):
            result = self._guard.protect(text)
            self._reporter.report(
                make_record(
                    result,
                    ReportContext(vector=vector, origin=origin, source="litellm", content=text),
                    self._guard,
                )
            )
            if self._is_attack(result):
                if self._block:
                    _raise_rejected(self._format(result), data)
                # block=False: let it through (observe/log mode)
                break

        return data

    # -- post-call hook (opt-in) ---------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """Screen the LLM's reply for injected content (opt-in).

        Only active when the plugin was constructed with ``screen_output=True``.
        On a flagged response, raises ``ValueError`` which the proxy converts to
        an HTTP 500 error (output-side violation after the LLM has responded).
        """
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
                            self._reporter.report(
                                make_record(
                                    result,
                                    ReportContext(
                                        direction="output", source="litellm", content=content
                                    ),
                                    self._guard,
                                )
                            )
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
    """Extract screenable text from an OpenAI-style message dict.

    Handles both plain string content and the content-block list format used
    by multimodal / tool-use messages.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # {"type": "text", "text": "..."} or tool-result blocks
                text_val = block.get("text") or block.get("content", "")
                if text_val and isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


def _screenable_texts(
    messages: list[dict],
    screen_tool_results: bool,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(text, origin, vector)`` for each message to screen.

    Screens:
    - The **last** ``user`` / ``human`` message → ``(user_prompt, direct)``.
    - Optionally all ``tool`` / ``function`` messages → ``(tool_result, indirect)``
      — the indirect-injection surface — controlled by ``screen_tool_results``.

    Messages before the last user turn were already screened in the previous
    round — re-screening the full history would be redundant.
    """
    # Find the last user message (direct injection surface)
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

    # Tool/function results (indirect injection surface)
    if screen_tool_results:
        for msg in messages:
            role = (msg.get("role") or "").lower()
            if role in ("tool", "function"):
                text = _message_text(msg)
                if text:
                    yield text, "tool_result", "indirect"


def _raise_rejected(message: str, data: dict) -> None:
    """Block the request with a hard HTTP 400 before the LLM is called.

    A ``fastapi.HTTPException`` raised inside ``async_pre_call_hook`` propagates
    through the proxy as a real ``400 Bad Request`` (vs. ``RejectedRequestError``,
    which the proxy converts into a *graceful* 200 response whose content is the
    refusal message). FastAPI is always present in the proxy runtime, so the
    import is done lazily here to keep the base ``[litellm]`` install lean.
    """
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail={"error": message})
