"""Optional framework integrations for Bastion Prompt Protection.

Each submodule has its own optional dependency — import the one you need
(this package does not import them eagerly, so the core install stays lean):

    from bastion_prompt_protection.integrations.langchain import BastionGuardrail
    # pip install "bastion-prompt-protection[langchain]"

    from bastion_prompt_protection.integrations.llamaindex import BastionGuardrailPostprocessor
    # pip install "bastion-prompt-protection[llamaindex]"

Both raise the shared bastion_prompt_protection.exceptions.PromptInjectionError.
"""
