from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bastion_prompt_protection.guard import GuardResult


class PromptInjectionError(Exception):
    """Raised by integrations that choose to fail closed on a detection.

    The core ``Guard.protect()`` never raises this — it returns a result.
    This exists so integrations and application code have one shared error type.
    """

    def __init__(self, result: GuardResult) -> None:
        super().__init__(
            f"Prompt injection detected (risk={result.risk:.3f}, stage={result.stage_reached})."
        )
        self.result = result


class ModelUnavailableError(Exception):
    """Raised by ``Guard.protect()`` when the ONNX classifier cannot be loaded.

    When ``on_model_unavailable`` is ``"throw"``, this is raised permanently
    after the first failed download. When set to ``"try-download-then-throw"``
    (the default), each ``protect()`` call re-attempts the download after a
    5-second cooldown; this error is raised while the model is unavailable and
    the cooldown has not elapsed.

    ``cause`` carries the original download or parse error.
    """

    def __init__(self, model_id: str, cause: Exception, hint: str | None = None) -> None:
        msg = f"bastion-prompt-protection: model {model_id} is unavailable"
        if hint:
            msg += f" — {hint}"
        msg += f". Underlying cause: {cause}"
        super().__init__(msg)
        self.cause = cause
