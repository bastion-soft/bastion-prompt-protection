"""Shim so the LiteLLM proxy can load the Bastion guardrail.

LiteLLM resolves a custom guardrail's dotted path as a **file relative to the
config directory** (it does not import installed packages by dotted path). So
to use the pip-installed plugin, drop this one-line shim next to your
``config.yaml`` and point the ``guardrail:`` field at ``bastion_guardrail.BastionGuardrailPlugin``.
"""

from bastion_prompt_protection.integrations.litellm import (  # noqa: F401
    BastionGuardrailPlugin,
)
