"""GRAEAE multi-provider consultation endpoints — v3.0.0 unified service.

/v1/consultations — GRAEAE reasoning domain with hash-chained audit log and memory refs.

"""

import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from mnemos.api.dependencies import UserContext, get_current_user
from mnemos.api.persistence_helpers import require_consultations_backend
from mnemos.core.rate_limit import limiter
from mnemos.core.security import is_root, scope_namespace
from mnemos.domain.graeae.engine import _REGISTRY_MAP
from mnemos.domain.models import (
    AuditLogEntry,
    AuditVerifyResponse,
    ConsultationArtifact,
    ConsultationRequest,
    ConsultationResponse,
    SUPPORTED_CONSULTATION_MODES,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["consultations"])

# ── Audit-chain signing (review #10/#11) ──────────────────────────────────────
# Canonical, single-source-of-truth chain math for the GRAEAE audit log.
#
# #10 — the chain was a BARE SHA-256 over a PUBLIC genesis constant. Any actor
#       with DB write access could recompute every chain_hash from public
#       inputs, and verify() re-derived them identically, so tampering was
#       undetectable. We now sign each link with HMAC-SHA256 under a server-held
#       secret; a DB-only attacker can no longer forge valid links.
# #11 — the chain covered only prev+prompt_hash+response_hash, so metadata
#       (provider / task_type / quality_score / consultation_id) could be
#       rewritten and the chain still verified, and a sequence_num was never
#       bound so trailing rows could be truncated undetected. We now bind all
#       of that metadata (and the sequence_num) into the MAC.
#
# ROLLOUT NOTE: the LIVE write path is mnemos-core's
# `backend.consultations.create_consultation_with_audit`, which must import and
# use `compute_audit_chain_hash` / `audit_genesis_hash` below, and a one-time
# re-anchor migration must re-sign existing rows. Full tail-truncation defense
# additionally needs a signed tip-anchor row (schema follow-up in core).
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


# ── Per-principal consultation quota (review #12) ─────────────────────────────
# The only throttle was @limiter.limit("60/minute"); a single principal could
# still fire 60 mode=all / mode=debate consultations per minute, each fanning
# out to every muse — a financial DoS. Enforce a server-side, per-USER,
# cost-WEIGHTED sliding-window quota so expensive modes drain budget faster.
_MODE_COST_WEIGHT = {
    "single": 1.0,
    "local": 1.0,
    "auto": 2.0,
    "external": 3.0,
    "majority": 3.0,
    "all": 5.0,
    "debate": 8.0,
}
_QUOTA_WINDOW_SECONDS = 60.0
# user_id -> deque[(monotonic_ts, weight)]. Bounded by active principals;
# empty buckets are pruned opportunistically below.
_consult_usage: dict[str, deque] = defaultdict(deque)


def _consult_quota_per_min() -> float:
    try:
        return float(os.environ.get("MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN", "20"))
    except ValueError:
        return 20.0


def _enforce_consult_quota(user_id: str, mode: str) -> None:
    """Cost-weighted per-principal quota. Raises 429 when the window is full.

    Synchronous (no awaits) so the read-modify-write of the bucket is atomic
    within the event loop; keying is the SERVER-derived user_id, never a
    client-supplied value.
    """
    quota = _consult_quota_per_min()
    if quota <= 0:
        return
    weight = _MODE_COST_WEIGHT.get((mode or "auto").lower(), 2.0)
    now = time.monotonic()
    cutoff = now - _QUOTA_WINDOW_SECONDS
    bucket = _consult_usage[str(user_id or "anonymous")]
    while bucket and bucket[0][0] < cutoff:
        bucket.popleft()
    used = sum(w for _, w in bucket)
    if used + weight > quota:
        retry = max(1, int(bucket[0][0] + _QUOTA_WINDOW_SECONDS - now)) if bucket else int(_QUOTA_WINDOW_SECONDS)
        raise HTTPException(
            status_code=429,
            detail=(
                f"GRAEAE consultation quota exceeded for this principal "
                f"({used:.0f}+{weight:.0f} > {quota:.0f} cost-units/min)"
            ),
            headers={"Retry-After": str(retry)},
        )
    bucket.append((now, weight))
    if not bucket:  # pragma: no cover - defensive prune
        _consult_usage.pop(str(user_id or "anonymous"), None)


def _schedule_outbox_deliveries(delivery_ids: list[str]) -> None:
    if not delivery_ids:
        return
    from mnemos.api.routes.memories import _schedule_outbox_deliveries as _schedule

    _schedule(delivery_ids)


# ── Custom Query selection (v3.2) ─────────────────────────────────────────────

_VALID_TIERS = {"frontier", "premium", "budget"}

# Translate a model_registry.provider value (e.g. "anthropic") back into
# the GRAEAE engine provider key (e.g. "claude") so consult()'s selection
# filter doesn't silently drop entries. Only `anthropic→claude` flips
# today but the map is built from _REGISTRY_MAP so future renames
# propagate automatically.
_REGISTRY_TO_GRAEAE = {cfg["registry_provider"]: name for name, cfg in _REGISTRY_MAP.items()}


def _to_graeae_provider(registry_name: str) -> str:
    return _REGISTRY_TO_GRAEAE.get(registry_name, registry_name)


async def _tier_lineup(tier: str) -> dict:
    """Resolve a tier name to {provider_name: model_id} using model_registry.

    Tier definitions (aligned with the v3.1.2 /v1/models registry work):

      * frontier  — arena_rank <= 5 OR graeae_weight >= 0.95
      * premium   — arena_rank BETWEEN 6 AND 15 OR graeae_weight in [0.85, 0.95)
      * budget    — cheapest available models at graeae_weight >= 0.75

    The caller reflects a tier into a concrete dict that consult()
    consumes as a selection. Empty registry -> empty dict; handler
    treats that as a hard error (otherwise we'd silently fall back
    to auto, which violates the caller's intent).
    """
    if tier not in _VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail=(f"unknown tier {tier!r}; " f"expected one of {sorted(_VALID_TIERS)}"),
        )
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        rows = await backend.consultations.resolve_tier_lineup(tx, tier)
    # Translate registry provider → GRAEAE engine provider key
    # (`anthropic` → `claude` etc.) so consult()'s selection filter
    # at engine.py:_candidate_providers doesn't silently drop muses.
    return {_to_graeae_provider(r["provider"]): r["model_id"] for r in rows}


async def _resolve_models(model_ids: List[str]) -> dict:
    """Resolve each explicit model_id to its provider via model_registry.

    Returns {provider_name: model_id}. Raises 400 on the first
    unrecognized model_id — fail-loudly beats silently narrowing a
    deliberately-chosen lineup.
    """
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        rows = await backend.consultations.resolve_models(tx, model_ids)
    found = {r["model_id"]: r["provider"] for r in rows}
    missing = [m for m in model_ids if m not in found]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model_id(s): {missing}",
        )
    return {_to_graeae_provider(found[m]): m for m in model_ids}


async def _resolve_selection(
    engine,
    models: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    tier: Optional[str] = None,
) -> Optional[dict]:
    """Resolve a caller's Custom Query selectors to a
    {provider_name: model_id_or_None} dict consult() understands.

    Precedence: models > providers > tier > None (auto lineup).
    Raises HTTPException(400) for unknown providers, unknown tiers,
    unknown model_ids, or empty tier result sets.
    """
    # Mutual exclusion — at most one selector. Prevents a caller from
    # passing both `tier=frontier` and `providers=[...]` and then
    # wondering which won. If a caller wants combined semantics (e.g.
    # "frontier models FROM these providers"), that's a follow-up
    # design; reject the combination today for clarity.
    set_fields = [
        n
        for n in (
            "models" if models else None,
            "providers" if providers else None,
            "tier" if tier else None,
        )
        if n
    ]
    if len(set_fields) > 1:
        raise HTTPException(
            status_code=400,
            detail=(f"Custom Query accepts at most one of " f"{{'models', 'providers', 'tier'}}; got {set_fields}"),
        )

    if models:
        return await _resolve_models(models)

    if providers:
        # Accept either GRAEAE name ("claude") or registry name
        # ("anthropic") and normalise to the GRAEAE key that consult()
        # filters against. Without this, providers=["anthropic"] would
        # 400 because engine.providers is keyed by "claude".
        normalised = [_to_graeae_provider(p) for p in providers]
        unknown = [p for p in normalised if p not in engine.providers]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider(s): {unknown}",
            )
        # De-duplicate while preserving caller order (two callers could
        # legitimately pass both "anthropic" and "claude" — they map to
        # the same engine slot).
        seen: set[str] = set()
        out: dict = {}
        for p in normalised:
            if p not in seen:
                out[p] = None
                seen.add(p)
        return out

    if tier:
        lineup = await _tier_lineup(tier)
        if not lineup:
            raise HTTPException(
                status_code=404,
                detail=f"tier {tier!r} has no matching rows in model_registry",
            )
        return lineup

    # None set -> auto lineup (existing behavior).
    return None


# ── Audit helpers ─────────────────────────────────────────────────────────────


async def _write_audit_entry_on_conn(
    conn,
    consultation_id,
    prompt: str,
    response: str,
    task_type: str,
    provider: str,
    quality_score: float,
) -> None:
    """Append a hash-chained entry to graeae_audit_log on an existing connection.

    Expects to be called inside an open transaction on `conn`. Raises on
    failure — callers must let the exception propagate so the surrounding
    consultation transaction aborts (tamper-evidence requires the audit row
    and the consultation row to commit atomically).
    """
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    response_hash = hashlib.sha256(response.encode()).hexdigest()

    # Advisory lock serializes concurrent inserts.
    # SELECT FOR UPDATE alone has a TOCTOU race: T2 reads the "last row"
    # before blocking, then computes the chain against that stale row after
    # T1 has already inserted a newer one.
    # Advisory lock (magic key = 0x4772616561 = "Graea") ensures only
    # one writer holds the chain tip at a time.
    await conn.execute("SELECT pg_advisory_xact_lock(285734657)")
    # Audit-chain continuity is internal tamper-evidence, not a
    # user-content read path: the chain tip must include soft-deleted
    # rows so later writes keep validating across GDPR restore windows.
    prev_row = await conn.fetchrow(
        "SELECT id, chain_hash, sequence_num FROM graeae_audit_log ORDER BY sequence_num DESC LIMIT 1"
    )
    if prev_row:
        prev_chain = prev_row["chain_hash"]
        prev_id = prev_row["id"]
        sequence_num = int(prev_row["sequence_num"]) + 1
    else:
        prev_chain = audit_genesis_hash()
        prev_id = None
        sequence_num = 1

    # SECURITY (review #10/#11): keyed MAC over prev + the full metadata tuple
    # (sequence_num/consultation_id/task_type/provider/quality_score) plus the
    # prompt/response hashes — see compute_audit_chain_hash. The advisory lock
    # above guarantees sequence_num = prev + 1 is the value this row will land.
    chain_hash = compute_audit_chain_hash(
        prev_chain_hash=prev_chain,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        sequence_num=sequence_num,
        consultation_id=consultation_id,
        task_type=task_type,
        provider=provider,
        quality_score=quality_score,
    )

    await conn.execute(
        "INSERT INTO graeae_audit_log "
        "(consultation_id, prompt, prompt_hash, provider, response_text, "
        "response_hash, chain_hash, prev_id, prev_chain_hash, "
        "task_type, quality_score) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        consultation_id,
        prompt,
        prompt_hash,
        provider,
        response,
        response_hash,
        chain_hash,
        prev_id,
        prev_chain,
        task_type,
        quality_score,
    )


async def _write_memory_refs_on_conn(
    conn,
    consultation_id: str,
    memory_ids: List[str],
) -> None:
    """Record which memories were injected into this consultation, on an open conn.

    Raises on failure; caller's transaction aborts so memory-ref bookkeeping
    stays consistent with the consultation row.
    """
    if not memory_ids:
        return
    for memory_id in memory_ids:
        await conn.execute(
            "INSERT INTO consultation_memory_refs "
            "(consultation_id, memory_id, injected_at) "
            "VALUES ($1, $2, NOW()) "
            "ON CONFLICT DO NOTHING",
            consultation_id,
            memory_id,
        )


def _extract_memory_ids(result: dict) -> List[str]:
    """Collect injected/reference memory IDs from known result shapes."""
    raw_ids = result.get("memory_ids") or result.get("injected_memory_ids") or result.get("citations") or []
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


# ── Consultation endpoint ─────────────────────────────────────────────────────


@router.post("/consultations", response_model=ConsultationResponse)
@limiter.limit("60/minute")
async def consult_graeae(request: Request, body: ConsultationRequest, user: UserContext = Depends(get_current_user)):
    """Consult GRAEAE multi-provider consensus engine.

    Creates a hash-chained audit entry and records any injected memories.
    Returns raw provider responses (full, best, or truncated per format param).
    """
    logger.info(
        f"[CONSULTATION] {user.user_id}: {body.task_type} " f"(limit_chars={body.limit_chars}, format={body.format})"
    )
    # SECURITY (review #12): cost-weighted per-principal quota, BEFORE any
    # provider fan-out, so a single caller cannot turn 60 req/min of
    # mode=all/debate into a financial DoS.
    _enforce_consult_quota(user.user_id, body.mode)
    try:
        backend = require_consultations_backend()
        from mnemos.domain.graeae.engine import get_graeae_engine

        engine = get_graeae_engine()

        # v3.2 Custom Query mode: resolve the caller's lineup from the
        # three optional selectors on the request body. Precedence:
        # models > providers > tier > auto. `_resolve_selection` is
        # HTTPException-raising on bad input (unknown provider, unknown
        # model_id, unknown tier, empty tier result set).
        selection = await _resolve_selection(
            engine=engine,
            models=body.models,
            providers=body.providers,
            tier=body.tier,
        )

        result = await engine.consult(
            body.prompt,
            body.task_type,
            selection=selection,
            mode=body.mode,
        )
        result = _require_non_empty_consultation_result(result, body.mode)

        if body.limit_chars and result.get("all_responses"):
            for provider, resp in result["all_responses"].items():
                if isinstance(resp.get("response_text"), str):
                    original_len = len(resp["response_text"])
                    resp["response_text"] = resp["response_text"][: body.limit_chars]
                    resp["truncated"] = original_len > body.limit_chars

        if body.format == "best" and result.get("all_responses"):
            best = max(result["all_responses"].items(), key=lambda x: x[1].get("final_score", 0))
            result["all_responses"] = {best[0]: best[1]}

        consultation_id = None
        memory_ids = _extract_memory_ids(result)
        delivery_ids: list[str] = []
        if result.get("all_responses"):
            # Persistence reads consensus fields FROM THE ENGINE return
            # dict instead of re-deriving them locally. The engine's
            # _compute_consensus is the single source of truth for
            # winning_muse / consensus_response / consensus_score /
            # cost / latency_ms; previously this block ran its own
            # max() over all_responses, which diverged from the engine
            # whenever scoring rules changed and produced nonsense
            # rows on all-failure (max() of a dict with only errored
            # entries picks an arbitrary error).
            #
            # On all-failure _compute_consensus returns
            # consensus_response="" / consensus_score=0.0 /
            # winning_muse=None / cost=0.0 / latency_ms=0, all safe to
            # persist. The response row still lands so the caller
            # has a stable consultation_id and the audit chain is
            # unbroken.
            consensus_response = result.get("consensus_response", "") or ""
            consensus_score = float(result.get("consensus_score", 0.0) or 0.0)
            winning_muse = result.get("winning_muse")
            engine_cost = float(result.get("cost", 0.0) or 0.0)
            engine_latency_ms = int(result.get("latency_ms", 0) or 0)

            # All three writes — consultation row, audit entry, memory refs —
            # must commit as a single unit. If the audit write fails we MUST
            # abort the consultation row: tamper-evidence requires that a
            # committed consultation implies a committed audit chain link.
            try:
                async with backend.transactional() as tx:
                    consultation_id = await backend.consultations.create_consultation_with_audit(
                        tx,
                        prompt=body.prompt,
                        task_type=body.task_type,
                        consensus_response=consensus_response,
                        consensus_score=consensus_score,
                        winning_muse=winning_muse,
                        cost=engine_cost,
                        latency_ms=engine_latency_ms,
                        mode=body.mode,
                        owner_id=user.user_id,
                        namespace=user.namespace,
                        memory_ids=memory_ids,
                        # SECURITY (review #10): keyed genesis. NOTE: mnemos-core's
                        # create_consultation_with_audit must adopt
                        # compute_audit_chain_hash for the per-link MAC too, else
                        # verify() below will (correctly) report the chain unsigned.
                        genesis_hash=audit_genesis_hash(),
                    )
                    if backend.supports_webhooks:
                        delivery_ids = await backend.webhooks.dispatch_event(
                            tx,
                            "consultation.completed",
                            {
                                "consultation_id": str(consultation_id),
                                "task_type": body.task_type,
                                "winning_muse": result.get("winning_muse"),
                                "consensus_score": result.get("consensus_score"),
                                "owner_id": user.user_id,
                                "namespace": user.namespace,
                            },
                            owner_id=user.user_id,
                            namespace=user.namespace,
                        )
                    else:
                        delivery_ids = []
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[CONSULTATION] persist failed — aborting: {e}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail="Consultation persistence failed; audit trail is required.",
                )

        _schedule_outbox_deliveries(delivery_ids)
        if consultation_id is not None:
            from mnemos.nats import publish_event as _nats_publish_event
            from mnemos.nats.client import get_node_name as _nats_get_node_name

            safe_ns = (user.namespace or "default").replace(".", "_")
            await _nats_publish_event(
                f"mnemos.consultation.completed.{safe_ns}",
                {
                    "consultation_id": str(consultation_id),
                    "task_type": body.task_type,
                    "mode": body.mode,
                    "winning_muse": result.get("winning_muse"),
                    "consensus_score": result.get("consensus_score"),
                    "namespace": user.namespace,
                    "user_id": user.user_id,
                    "source_node": _nats_get_node_name(),
                },
                msg_id=f"{consultation_id}.completed",
            )

        return ConsultationResponse(
            # asyncpg returns UUID columns as uuid.UUID objects, not strings.
            # ConsultationResponse.consultation_id is typed str, so coerce.
            consultation_id=str(consultation_id) if consultation_id is not None else None,
            all_responses=result.get("all_responses", {}),
            consensus_response=result.get("consensus_response"),
            consensus_score=result.get("consensus_score"),
            winning_muse=result.get("winning_muse"),
            cost=result.get("cost"),
            latency_ms=result.get("latency_ms"),
            mode=body.mode,
            timestamp=result.get("timestamp", ""),
            round_1=result.get("round_1"),
            round_2=result.get("round_2"),
            quorum_reached=result.get("quorum_reached"),
            quorum_threshold=result.get("quorum_threshold"),
            similarity_pairs=result.get("similarity_pairs"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CONSULTATION] Error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Consultation failed — see server logs for details")


# ── Audit log endpoints (declared before dynamic /{consultation_id} to prevent
#    'audit' string being matched as a UUID path param) ───────────────────────


@router.get("/consultations/audit", response_model=List[AuditLogEntry])
@limiter.limit("30/minute")
async def list_audit_log(
    request: Request,
    limit: int = Query(20, le=100),
    offset: int = 0,
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
):
    """List GRAEAE audit log entries (newest first).

    Non-root callers only see audit rows for their own consultations. Root
    callers keep the operational global view.
    """
    target_ns = scope_namespace(user, namespace)
    root = is_root(user)
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        rows = await backend.consultations.list_audit_log(
            tx,
            root=root,
            user_id=user.user_id,
            namespace=target_ns if (namespace is not None or not root) else None,
            limit=limit,
            offset=offset,
        )
    return [
        AuditLogEntry(
            id=str(r["id"]),
            sequence_num=r["sequence_num"],
            consultation_id=str(r["consultation_id"]) if r["consultation_id"] else None,
            prompt_hash=r["prompt_hash"],
            response_hash=r["response_hash"],
            chain_hash=r["chain_hash"] if root else None,
            prev_id=str(r["prev_id"]) if r["prev_id"] else None,
            task_type=r.get("task_type"),
            provider=r.get("provider"),
            quality_score=r.get("quality_score"),
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ]


@router.get("/consultations/audit/verify", response_model=AuditVerifyResponse)
@limiter.limit("5/minute")
async def verify_audit_chain(
    request: Request,
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
):
    """Verify the integrity of the hash chain in the GRAEAE audit log.

    Root walks the entire chain from genesis, verifying each link. Non-root
    callers verify only rows attached to their own consultations while deriving
    each predecessor from the immediate previous global sequence row, not from
    the row's tamperable prev_id. Rate-limited because the cost grows linearly
    with audit-log size. Returns details of any broken sequences.
    """
    target_ns = scope_namespace(user, namespace)
    verify_global_chain = is_root(user) and namespace is None
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        rows = await backend.consultations.fetch_audit_chain(
            tx,
            root=is_root(user),
            user_id=user.user_id,
            namespace=None if verify_global_chain else target_ns,
        )

    if not rows:
        return AuditVerifyResponse(
            valid=True,
            entries_checked=0,
            message=("Audit log is empty" if verify_global_chain else "No audit entries visible for caller"),
        )

    if verify_global_chain:
        prev_chain = audit_genesis_hash()
        failures: dict[int, str] = {}
        for row in rows:
            expected = compute_audit_chain_hash(
                prev_chain_hash=prev_chain,
                prompt_hash=row["prompt_hash"],
                response_hash=row["response_hash"],
                sequence_num=row.get("sequence_num"),
                consultation_id=row.get("consultation_id"),
                task_type=row.get("task_type"),
                provider=row.get("provider"),
                quality_score=row.get("quality_score"),
            )
            if expected != row["chain_hash"]:
                failures.setdefault(
                    row["sequence_num"],
                    f"Chain broken at sequence {row['sequence_num']}: "
                    f"expected {expected[:16]}…, stored {row['chain_hash'][:16]}…",
                )
            prev_chain = row["chain_hash"]

        if failures:
            entries_failed = sorted(failures)
            first_broken_sequence = entries_failed[0]
            message = failures[first_broken_sequence]
            if len(entries_failed) > 1:
                message = f"{message}; {len(entries_failed)} entries failed verification"
            return AuditVerifyResponse(
                valid=False,
                entries_checked=len(rows),
                first_broken_sequence=first_broken_sequence,
                entries_failed=entries_failed,
                message=message,
            )

        return AuditVerifyResponse(
            valid=True,
            entries_checked=len(rows),
            message=f"All {len(rows)} entries verified — chain intact",
        )

    failures: dict[int, str] = {}
    for row in rows:
        scoped_sequence_num = row["scoped_sequence_num"]
        stored_prev_chain = row["prev_chain_hash"]
        prev_chain = row["expected_prev_hash"] or audit_genesis_hash()
        if stored_prev_chain and stored_prev_chain != prev_chain:
            logger.warning(
                "Scoped audit predecessor mismatch for user=%s scoped_row=%s "
                "global_sequence=%s expected_prev_hash=%s stored_prev_hash=%s",
                user.user_id,
                scoped_sequence_num,
                row["sequence_num"],
                prev_chain,
                stored_prev_chain,
            )
            failures.setdefault(
                scoped_sequence_num,
                f"Scoped chain broken at row {scoped_sequence_num}: "
                "stored previous hash does not match actual previous row",
            )
        expected = compute_audit_chain_hash(
            prev_chain_hash=prev_chain,
            prompt_hash=row["prompt_hash"],
            response_hash=row["response_hash"],
            sequence_num=row.get("sequence_num"),
            consultation_id=row.get("consultation_id"),
            task_type=row.get("task_type"),
            provider=row.get("provider"),
            quality_score=row.get("quality_score"),
        )
        if expected != row["chain_hash"]:
            logger.warning(
                "Scoped audit hash mismatch for user=%s scoped_row=%s "
                "global_sequence=%s expected_hash=%s stored_hash=%s",
                user.user_id,
                scoped_sequence_num,
                row["sequence_num"],
                expected,
                row["chain_hash"],
            )
            failures.setdefault(
                scoped_sequence_num,
                f"Hash mismatch at row {scoped_sequence_num}",
            )

    if failures:
        entries_failed = sorted(failures)
        first_broken_sequence = entries_failed[0]
        message = failures[first_broken_sequence]
        if len(entries_failed) > 1:
            message = f"{message}; {len(entries_failed)} scoped entries failed verification"
        return AuditVerifyResponse(
            valid=False,
            entries_checked=len(rows),
            first_broken_sequence=first_broken_sequence,
            entries_failed=entries_failed,
            message=message,
        )

    return AuditVerifyResponse(
        valid=True,
        entries_checked=len(rows),
        message=f"All {len(rows)} scoped entries verified - chain intact",
    )


# ── Static muse / mode listings ────────────────────────────────────────────────
# MUST be declared BEFORE the parametric /{consultation_id} route below so
# FastAPI doesn't try to parse "muses" or "modes" as a UUID and 500 in asyncpg.


@router.get("/consultations/muses")
async def list_muses(_: UserContext = Depends(get_current_user)):
    """List the live GRAEAE muse manifest.

    Pulls model + weight + api shape from the engine's in-memory provider
    map, which is auto-refreshed from model_registry at startup and on
    /admin/graeae/reload-providers. Operators can use this to confirm the
    daily provider sync rotated to current model_ids.
    """
    from mnemos.domain.graeae.engine import get_graeae_engine

    engine = get_graeae_engine()
    muses = []
    for name, cfg in engine.providers.items():
        muses.append(
            {
                "name": name,
                "model": cfg.get("model"),
                "weight": cfg.get("weight"),
                "api": cfg.get("api"),
                "key_name": cfg.get("key_name"),
            }
        )
    return {"count": len(muses), "muses": muses}


@router.get("/consultations/modes")
async def list_modes(_: UserContext = Depends(get_current_user)):
    """List supported consultation routing and reasoning-shape modes."""
    return {
        "modes": [
            {
                "name": "auto",
                "description": "engine picks the default routing strategy based on task_type",
            },
            {
                "name": "local",
                "description": "force local-only muses where configured (no commercial APIs)",
            },
            {
                "name": "external",
                "description": "force external commercial muses where configured",
            },
            {
                "name": "all",
                "description": "fan out to every available muse",
            },
            {
                "name": "single",
                "description": "pick exactly one highest-weighted muse; fastest and lowest-cost path",
            },
            {
                "name": "debate",
                "description": "run a two-round cross-muse debate and return the refined round",
            },
            {
                "name": "majority",
                "description": "fan out to up to three muses and report whether quorum agreement was reached",
            },
        ],
        "validation": {
            "supported": list(SUPPORTED_CONSULTATION_MODES),
            "unknown_mode_status": 422,
            "unknown_mode_detail": (
                "The request schema validates mode with a Pydantic Literal; "
                "unknown modes are rejected before business logic runs."
            ),
        },
    }


# ── Dynamic /{consultation_id} routes (declared after static /audit above) ────


@router.get("/consultations/{consultation_id}")
async def get_consultation(
    consultation_id: str,
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
):
    """Retrieve a consultation by ID.

    Scoped to the calling user: non-root callers only see their own
    consultations. Not-yours and not-exists both return 404 so we don't
    leak which consultation IDs are in use across users.
    """
    target_ns = scope_namespace(user, namespace)
    root = is_root(user)
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        row = await backend.consultations.get_consultation(
            tx,
            consultation_id=consultation_id,
            root=root,
            user_id=user.user_id,
            namespace=target_ns if (namespace is not None or not root) else None,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Consultation not found")

    return {
        "id": str(row["id"]),
        "prompt": row["prompt"],
        "task_type": row["task_type"],
        "consensus_response": row["consensus_response"],
        "consensus_score": row["consensus_score"],
        "winning_muse": row["winning_muse"],
        "cost": row["cost"],
        "latency_ms": row["latency_ms"],
        "mode": row["mode"],
        "created_at": row["created"].isoformat() if hasattr(row["created"], "isoformat") else str(row["created"]),
    }


@router.get("/consultations/{consultation_id}/artifacts")
async def get_consultation_artifacts(
    consultation_id: str,
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
):
    """Retrieve structured outputs and citations from a consultation."""
    target_ns = scope_namespace(user, namespace)
    root = is_root(user)
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        consultation, memory_refs = await backend.consultations.get_consultation_artifacts(
            tx,
            consultation_id=consultation_id,
            root=root,
            user_id=user.user_id,
            namespace=target_ns if (namespace is not None or not root) else None,
        )
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    return ConsultationArtifact(
        consultation_id=str(consultation["id"]),
        citations=[str(ref["memory_id"]) for ref in memory_refs],
        memory_refs=[
            {
                "memory_id": str(ref["memory_id"]),
                "injected_at": ref["injected_at"].isoformat()
                if hasattr(ref["injected_at"], "isoformat")
                else str(ref["injected_at"]),
            }
            for ref in memory_refs
        ],
        created_at=consultation["created"].isoformat()
        if hasattr(consultation["created"], "isoformat")
        else str(consultation["created"]),
    )


# ── /v1/consultations/{consultation_id}/full — verbatim classified recall ──
#
# The MCP tool response can be truncated by the client; this endpoint makes
# the full verbatim consultation recoverable, classified into four record
# types (source / quorum / synthesis / muses) and PAGED so an arbitrarily
# long result is retrievable page by page. Owner-scoped: a non-root caller
# only recalls its own consultations (enforced in the persistence layer;
# an invisible/unknown id returns 404). No schema change — assembled at read
# time from graeae_consultations + graeae_audit_log.
_VALID_FULL_SECTIONS = ("all", "source", "quorum", "synthesis", "muses")


def _serialise_full_part(part: dict[str, Any]) -> tuple[str, int]:
    """Return ``(rendered_json, char_len)`` for a part, for paging math."""
    rendered = json.dumps(part, ensure_ascii=False, default=str)
    return rendered, len(rendered)


def _select_full_parts(full: dict[str, Any], section: str) -> list[dict[str, Any]]:
    """Ordered list of classified parts for the requested section."""
    parts: list[dict[str, Any]] = []
    if section in ("all", "source"):
        parts.append({"type": "source", **(full.get("source") or {})})
    if section in ("all", "quorum"):
        parts.append({"type": "quorum", **(full.get("quorum") or {})})
    if section in ("all", "synthesis"):
        parts.append({"type": "synthesis", **(full.get("synthesis") or {})})
    if section in ("all", "muses"):
        muses = full.get("muses") or []
        for idx, muse in enumerate(muses, start=1):
            parts.append({"type": f"muse:{idx}/{len(muses)}", **muse})
    return parts


def _paginate_full_parts(
    parts: list[dict[str, Any]], page_size: int, page: int
) -> tuple[list[dict[str, Any]], int]:
    """Part-level paging; parts longer than ``page_size`` split into numbered
    sub-pages (never mid-part, never mid-codepoint). Returns (page_parts,
    total_pages)."""
    rendered: list[dict[str, Any]] = []
    for part in parts:
        text, length = _serialise_full_part(part)
        if length <= page_size:
            rendered.append(part)
            continue
        chunks = [text[i : i + page_size] for i in range(0, length, page_size)]
        base_type = part.get("type", "part")
        for idx, chunk in enumerate(chunks, start=1):
            rendered.append(
                {"type": f"{base_type}#{idx}/{len(chunks)}", "_chunk": idx, "_chunks": len(chunks), "text": chunk}
            )
    total_pages = len(rendered)
    if page < 1 or page > total_pages:
        return [], total_pages
    return [rendered[page - 1]], total_pages


@router.get("/consultations/{consultation_id}/full")
@limiter.limit("30/minute")
async def get_consultation_full(
    request: Request,
    consultation_id: str,
    section: str = Query("all", description="all | source | quorum | synthesis | muses"),
    page: int = Query(1, ge=1),
    page_size: int = Query(6000, ge=500, le=50000),
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
):
    """Recall one GRAEAE consultation, verbatim and classified, paged.

    Non-root callers only see their own consultations. One record per page:
    ``total_pages`` reports how many pages the requested ``section`` spans so
    a caller can walk the full untruncated result.
    """
    if section not in _VALID_FULL_SECTIONS:
        raise HTTPException(status_code=422, detail=f"section must be one of {_VALID_FULL_SECTIONS}")
    root = is_root(user)
    target_ns = scope_namespace(user, namespace)
    backend = require_consultations_backend()
    async with backend.transactional() as tx:
        full = await backend.consultations.fetch_consultation_full(
            tx,
            consultation_id,
            root=root,
            user_id=user.user_id,
            namespace=target_ns if (namespace is not None or not root) else None,
        )
    if full is None:
        raise HTTPException(status_code=404, detail="consultation not found")
    parts = _select_full_parts(full, section)
    page_parts, total_pages = _paginate_full_parts(parts, page_size, page)
    return {
        "consultation_id": consultation_id,
        "section": section,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "parts": page_parts,
    }
