"""Shared consultation helpers for HTTP and MCP; no route-handler dependency."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import List

from fastapi import HTTPException
from mnemos.domain.graeae.engine import _REGISTRY_MAP


_AUDIT_GENESIS_LABEL = b"MNEMOS_AUDIT_GENESIS_v3"


_FIELD_SEP = "\x1f"


def _audit_hmac_key() -> bytes:
    """Return the server-held audit signing key, or fail closed.

    Fail-closed (review theme #3): we refuse to read or write an UNSIGNED audit
    chain. If the key is unset the audit endpoints return 503 rather than
    silently degrading to a forgeable bare hash.
    """
    key = os.environ.get("MNEMOS_GRAEAE_AUDIT_HMAC_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "audit chain signing key (MNEMOS_GRAEAE_AUDIT_HMAC_KEY) is not "
                "configured; refusing to operate an unsigned audit chain"
            ),
        )
    return key.encode()


def audit_genesis_hash() -> str:
    """Keyed genesis so the chain root cannot be reproduced without the key."""
    return hmac.new(_audit_hmac_key(), _AUDIT_GENESIS_LABEL, hashlib.sha256).hexdigest()


def compute_audit_chain_hash(
    *,
    prev_chain_hash: str,
    prompt_hash: str,
    response_hash: str,
    sequence_num: object = None,
    consultation_id: object = None,
    task_type: object = None,
    provider: object = None,
    quality_score: object = None,
) -> str:
    """HMAC each link over the full, ordered metadata tuple (review #10/#11)."""
    payload = _FIELD_SEP.join(
        [
            prev_chain_hash or "",
            "" if sequence_num is None else str(sequence_num),
            "" if consultation_id is None else str(consultation_id),
            "" if task_type is None else str(task_type),
            "" if provider is None else str(provider),
            "" if quality_score is None else f"{float(quality_score):.6f}",
            prompt_hash or "",
            response_hash or "",
        ]
    )
    return hmac.new(_audit_hmac_key(), payload.encode(), hashlib.sha256).hexdigest()


_REGISTRY_TO_GRAEAE = {
    cfg["registry_provider"]: name for name, cfg in _REGISTRY_MAP.items()
}


def _to_graeae_provider(registry_name: str) -> str:
    return _REGISTRY_TO_GRAEAE.get(registry_name, registry_name)


def _extract_memory_ids(result: dict) -> List[str]:
    """Collect injected/reference memory IDs from known result shapes."""
    raw_ids = (
        result.get("memory_ids")
        or result.get("injected_memory_ids")
        or result.get("citations")
        or []
    )
    memory_ids: list[str] = []
    for raw_id in raw_ids:
        memory_id = str(raw_id).strip()
        if memory_id and memory_id not in memory_ids:
            memory_ids.append(memory_id)
    return memory_ids


def _require_non_empty_consultation_result(result: object, mode: str) -> dict:
    """Fail loudly instead of letting an empty engine result serialize as success."""
    if not isinstance(result, dict) or not result:
        raise HTTPException(
            status_code=502,
            detail=(
                "GRAEAE consultation returned an empty result "
                f"for mode={mode!r}; refusing to return HTTP 200 with an empty body."
            ),
        )
    return result
