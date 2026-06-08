"""Tests for offline license verification.

The signature cases need the `license` extra (pynacl); they skip cleanly when
it isn't installed. The no-file / no-signature cases need no crypto and always
run. Mirrors the sign/verify contract in the licensing repo — the canonical
JSON here must stay byte-identical to the minter's.
"""

from __future__ import annotations

import base64
import json

import pytest

from bastion_prompt_protection import LicenseStatus, verify_license
from bastion_prompt_protection import license as lic


# --- No crypto needed -------------------------------------------------------

def test_missing_file_is_invalid(tmp_path) -> None:
    status = verify_license(str(tmp_path / "nope.json"))
    assert isinstance(status, LicenseStatus)
    assert not status.valid
    assert not status  # __bool__
    assert "no license" in status.reason.lower()


def test_license_without_signature_is_invalid() -> None:
    status = verify_license({"license_id": "BPP-2026-X", "tier": "product"})
    assert not status.valid
    assert "signature" in status.reason.lower()


# --- Signature cases (need the `license` extra) -----------------------------

@pytest.fixture
def signing_key(monkeypatch):
    nacl_signing = pytest.importorskip("nacl.signing")
    sk = nacl_signing.SigningKey.generate()
    # Point the verifier's embedded public key at this throwaway test key.
    monkeypatch.setattr(
        lic, "_PUBLIC_KEY_B64", base64.b64encode(bytes(sk.verify_key)).decode()
    )
    return sk


def _sign(body: dict, sk) -> dict:
    sig = sk.sign(lic._canonical_json(body)).signature
    return {**body, "signature": base64.b64encode(sig).decode()}


def test_valid_license_verifies(signing_key) -> None:
    body = {
        "license_id": "BPP-2026-TEST",
        "tier": "enterprise",
        "valid_until": "2099-01-01T00:00:00+00:00",
        "customer": {"company_name": "ACME GmbH"},
    }
    status = verify_license(_sign(body, signing_key))
    assert status.valid
    assert status.tier == "enterprise"
    assert status.company == "ACME GmbH"
    assert status.license_id == "BPP-2026-TEST"


def test_tampered_license_fails(signing_key) -> None:
    body = {"license_id": "X", "tier": "enterprise", "valid_until": "2099-01-01T00:00:00+00:00"}
    signed = _sign(body, signing_key)
    signed["tier"] = "product"  # mutate a field after signing
    status = verify_license(signed)
    assert not status.valid
    assert "verification failed" in status.reason.lower()


def test_wrong_key_fails(signing_key, monkeypatch) -> None:
    nacl_signing = pytest.importorskip("nacl.signing")
    body = {"license_id": "X", "valid_until": "2099-01-01T00:00:00+00:00"}
    signed = _sign(body, signing_key)
    # Swap the embedded public key to a different one → must not verify.
    other = nacl_signing.SigningKey.generate().verify_key
    monkeypatch.setattr(lic, "_PUBLIC_KEY_B64", base64.b64encode(bytes(other)).decode())
    assert not verify_license(signed).valid


def test_expired_license_flagged(signing_key) -> None:
    body = {"license_id": "X", "valid_until": "2000-01-01T00:00:00+00:00"}
    status = verify_license(_sign(body, signing_key))
    assert not status.valid
    assert status.expired
    assert "expired" in status.reason.lower()


def test_verify_from_file(signing_key, tmp_path) -> None:
    body = {"license_id": "X", "valid_until": "2099-01-01T00:00:00+00:00"}
    path = tmp_path / "license.json"
    path.write_text(json.dumps(_sign(body, signing_key)))
    assert verify_license(str(path)).valid
