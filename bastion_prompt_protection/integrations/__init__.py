"""Optional framework integrations for Bastion Prompt Protection.

Each submodule has its own optional dependency -- import the one you need
(this package does not import them eagerly, so the core install stays lean):

    from bastion_prompt_protection.integrations.langchain import BastionGuardrail
    # pip install "bastion-prompt-protection[langchain]"

    from bastion_prompt_protection.integrations.llamaindex import BastionGuardQueryEngine
    # pip install "bastion-prompt-protection[llamaindex]"

    from bastion_prompt_protection.integrations.openai_agents import make_input_guardrail
    # pip install "bastion-prompt-protection[openai-agents]"
    # agent = Agent(name="...", input_guardrails=[make_input_guardrail()])

    from bastion_prompt_protection.integrations.litellm import BastionGuardrailPlugin
    # pip install "bastion-prompt-protection[litellm]"

All integrations raise (or surface via the SDK's own exception chain) the shared
bastion_prompt_protection.exceptions.PromptInjectionError.
"""
