"""Exceptions shared across Bastion integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bastion_prompt_protection.guard import GuardResult


class PromptInjectionError(ValueError):
    """Raised when Bastion flags input (or retrieved content) as a prompt
    injection / jailbreak.

    The triggering :class:`~bastion_prompt_protection.guard.GuardResult` is
    available on ``.result`` for logging or custom handling. Subclasses
    ``ValueError`` so existing ``except ValueError`` handlers still catch it.
    """

    def __init__(self, result: GuardResult) -> None:
        self.result = result
        super().__init__(
            f"Prompt injection detected (risk={result.risk:.3f}, stage={result.stage_reached})."
        )
