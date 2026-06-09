"""Offline commercial-license verification.

A Bastion commercial license is an Ed25519-signed JSON document (emailed on
purchase, alongside a human-readable PDF). Verification is fully offline: the
public key ships in this module and the signature proves the license is
authentic and untampered — no network call, so it works in air-gapped and
container deployments.

This is an *assurance / audit* layer, not DRM. Model access itself is gated at
download time (the commercial weights are gated on the HF Hub). Use this to
record and prove license validity in your own logs, or set
``GuardConfig(require_license=True)`` if your compliance wants the process to
refuse to start without a valid license.

Needs the ``license`` extra::

    pip install "bastion-prompt-protection[license]"
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Bastion Soft Ed25519 public verification key — safe to publish; pairs with the
# Secret-Manager-held signing key used by the licensing backend. Base64 of the
# raw 32-byte key (matches keys/bastion-licensing-public.pem). If we ever rotate
# the signing key, this constant changes in a new SDK release.
_PUBLIC_KEY_B64 = "BSMpA1IBRWo671jTEp6ZZ96vrjRANPgvOM1g4PwAUkk="


def _default_paths() -> list[str]:
    return [
        p
        for p in (
            os.environ.get("BASTION_LICENSE"),
            str(Path.home() / ".bastion" / "license.json"),
        )
        if p
    ]


@dataclass
class LicenseStatus:
    """Result of an offline license check. Truthy iff `valid`."""

    valid: bool
    reason: str
    license_id: str | None = None
    tier: str | None = None
    company: str | None = None
    valid_until: str | None = None
    expired: bool = False

    def __bool__(self) -> bool:
        return self.valid


def _canonical_json(obj: dict) -> bytes:
    # MUST match the licensing minter byte-for-byte or signatures won't verify:
    # sorted keys, no whitespace, non-ASCII preserved, UTF-8.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _load(source: dict | str | Path | None) -> dict | None:
    if isinstance(source, dict):
        return source
    candidates = [str(source)] if source else _default_paths()
    for c in candidates:
        p = Path(c)
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def verify_license(source: dict | str | Path | None = None) -> LicenseStatus:
    """Verify a signed license offline.

    `source` may be the license dict, a path to the license JSON, or ``None``
    to auto-discover it ($BASTION_LICENSE, then ~/.bastion/license.json).
    Checks the Ed25519 signature first, then the `valid_until` expiry.
    """
    data = _load(source)
    if data is None:
        return LicenseStatus(valid=False, reason="no license file found")

    meta = dict(
        license_id=data.get("license_id"),
        tier=data.get("tier"),
        company=(data.get("customer") or {}).get("company_name"),
        valid_until=data.get("valid_until"),
    )

    sig_b64 = data.get("signature")
    if not sig_b64:
        return LicenseStatus(valid=False, reason="license has no signature", **meta)

    try:
        import nacl.signing  # type: ignore[import-not-found]
    except ImportError:
        return LicenseStatus(
            valid=False,
            reason="verification needs the license extra: "
            "pip install 'bastion-prompt-protection[license]'",
            **meta,
        )

    body = {k: v for k, v in data.items() if k != "signature"}
    try:
        verify_key = nacl.signing.VerifyKey(base64.b64decode(_PUBLIC_KEY_B64))
        verify_key.verify(_canonical_json(body), base64.b64decode(sig_b64))
    except Exception:
        return LicenseStatus(valid=False, reason="signature verification failed", **meta)

    valid_until = data.get("valid_until")
    if valid_until:
        try:
            exp = datetime.fromisoformat(valid_until)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp:
                return LicenseStatus(valid=False, reason="license expired", expired=True, **meta)
        except ValueError:
            pass  # unparseable date → don't fail closed on signature-valid license

    return LicenseStatus(valid=True, reason="valid", **meta)
