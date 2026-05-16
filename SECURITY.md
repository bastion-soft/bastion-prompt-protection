# Security

## Reporting a vulnerability

If you find a way to bypass Bastion Prompt Protection's detection in production
deployments — e.g., a novel jailbreak template, an obfuscation technique we
miss, or a structural weakness in the pipeline — please report it privately
via the **Report a vulnerability** button on the repository's
[Security tab](https://github.com/bastionsoft/bastion-prompt-protection/security) instead of
filing a public issue.

We will:

- Acknowledge receipt within 48 hours.
- Confirm whether it's a real bypass and assign a severity.
- Aim to ship a patched detector within 14 days for high-severity bypasses.

## Scope

In scope:

- Detection bypasses (false negatives): inputs that should be flagged as
  attacks but receive risk < 0.5.
- High-volume false positives on benign content.
- Vulnerabilities in `bastion-prompt-protection` package itself (e.g., regex denial-of-service,
  arbitrary code execution via crafted input).

Out of scope:

- Asking the model to roleplay or hypothetically discuss harmful topics —
  that's content moderation, not prompt injection.
- Performance reports on attacks we already know about and have flagged
  publicly.

## Responsible disclosure

We don't publish bypass details until a patched detector ships. If you
report a bypass, we'll credit you in the release notes unless you prefer to
remain anonymous.
