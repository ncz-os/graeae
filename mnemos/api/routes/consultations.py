"""GRAEAE multi-provider consultation endpoints — v3.0.0 unified service.

/v1/consultations — GRAEAE reasoning domain with hash-chained audit log and memory refs.

"""

import hashlib
import hmac
import inspect
import json
import logging
import os
import time
from typing import Any, List, Optional

import mnemos.core.lifecycle as _lc
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from mnemos.api.dependencies import UserContext, get_current_user
from mnemos.api.persistence_helpers import backend_or_503, require_consultations_backend
from mnemos.core import audit_chain
from mnemos.core.rate_limit import limiter
from mnemos.core.security import is_root, scope_namespace
from mnemos.domain.graeae.consultation_support import (
    _audit_hmac_key,
    audit_genesis_hash,
    compute_audit_chain_hash,
    _to_graeae_provider,
    _extract_memory_ids,
    _require_non_empty_consultation_result,
)
from mnemos.domain.models import (
    SUPPORTED_CONSULTATION_MODES,
    AuditLogEntry,
    AuditVerifyResponse,
    ConsultationArtifact,
    ConsultationRequest,
    ConsultationResponse,
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


# ── Per-principal consultation quota (review #12) ─────────────────────────────
# The only throttle was @limiter.limit("60/minute"); a single principal could
# still fire 60 mode=all / mode=debate consultations per minute, each fanning
# out to every muse — a financial DoS. Enforce a server-side, per-USER,
# cost-WEIGHTED sliding-window quota so expensive modes drain budget faster.
#
# Storage: a single ``state_kv`` row per principal (a JSON array of recent
# {ts_ms, weight} entries), kept in a dedicated namespace so quota rows never
# share keys with other state_kv callers. The persistence backend is the same
# fleet-shared coordination point the rest of mnemos-core already uses for
# webhook lease/claim, journal, etc.; no external cache dependency. Every
# read + prune + write happens inside ONE backend transaction so two
# concurrent requests for the same principal can't both succeed past the
# quota.
#
# Locking: ``state_kv.get()`` is a plain non-locking SELECT on every
# backend (postgres / oracle / mysql / mariadb / db2 / sqlite), so the
# naive read-modify-write is racy. We take a per-principal lock first.
#   * Postgres  : pg_advisory_xact_lock — proven pattern (review #10 audit
#                 chain in this same file uses the same keying scheme).
#   * Oracle/DB2/MySQL/MariaDB: SELECT ... FOR UPDATE on the row via the
#                 backend's underlying conn. SQLite serializes writers
#                 via BEGIN IMMEDIATE, so the implicit tx lock is enough.
#
# Multiple instances pointing at the same backend are safe by the same
# reasoning as the audit chain: the backend's row lock is fleet-wide.
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
_QUOTA_NAMESPACE = "graeae-consult-quota"
# A single sentinel "default" owner_id keeps the quota row per-principal
# (the principal hash IS the key). Cross-tenant isolation is the SHA-256
# over user_id; see _quota_principal_key below.
_QUOTA_OWNER_ID = "default"
# Bound the JSON blob: 256 entries is 256 * 256B-ish events — enough for
# any plausible per-minute workload without an unbounded row.
_QUOTA_MAX_ENTRIES = 256
# Backend class names for which the advisory-lock path is available.
# Same detection shape as mnemos/webhooks/dispatcher.py:70.
_POSTGRES_BACKEND_NAME = "PostgresBackend"


def _consult_quota_per_min() -> float:
    try:
        return float(os.environ.get("MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN", "20"))
    except ValueError:
        return 20.0


def _quota_principal_key(user_id: str) -> str:
    """Stable per-principal key for the state_kv quota row.

    SHA-256 over the raw user_id keeps PII out of the row key and gives a
    fixed-length identifier; the actual read/append happens against this
    string only inside the backend.
    """
    return hashlib.sha256(str(user_id or "anonymous").encode()).hexdigest()


def _advisory_lock_key(principal_key: str) -> int:
    """Stable signed-int64 Postgres advisory lock key for one principal.

    Same shape as _branch_advisory_lock_key in mnemos/api/routes/dag.py:
    take the first 8 bytes of the SHA-256 digest of a namespaced string,
    reinterpret as signed int64 so pg_advisory_xact_lock accepts it.
    """
    digest = hashlib.sha256(f"graeae-consult-quota:{principal_key}".encode()).digest()[
        :8
    ]
    key = int.from_bytes(digest, "big", signed=False)
    if key >= 2**63:
        key -= 2**64
    return key


# Dialect-aware identifier quoting for the inline row-lock SQL we issue
# against MySQL/MariaDB. ``key`` (and several other state_kv column names)
# are RESERVED WORDS on MySQL/MariaDB — an unquoted ``SELECT key FROM state``
# raises ``1064 You have an error in your SQL syntax`` on those backends.
# Postgres/Oracle/Db2/SQLite accept ``key`` either way in practice (Oracle
# needs lower-case quoted identifiers in real DBs but our state table was
# created with bare ``key`` historically and Oracle treats it as the literal
# name with no quoting required).
#
# Backticks are the MySQL/MariaDB convention (ANSI uses ``""`` for
# identifiers, which Postgres also accepts but Python DB-API 2.0 does NOT
# massage — and our Oracle/Db2 bound-parameter form below already handles
# that family). We do not introduce a new shared utility — there is no
# ``mnemos/persistence/`` module in this checkout, and the only consumer
# of this inline SQL is the row-lock here — so a local helper keeps the
# surface tight. (If a future change needs dialect-aware identifier
# quoting elsewhere, lift this next to a `mnemos/persistence/_sql.py`
# module and consume from there.)
_MYSQL_QUOTED_COLUMNS = "`owner_id`, `namespace`, `key`"


def _mysql_lock_sql() -> str:
    """Build the dialect-correct SELECT ... FOR UPDATE statement.

    Returning the rendered SQL (not the parameters) keeps this helper
    unit-testable from a static SQL-shape assertion, which is the form
    the fix-instruction asked for given we have no live MySQL/MariaDB
    fixture in this repo's CI.
    """
    return (
        f"SELECT {_MYSQL_QUOTED_COLUMNS} FROM state "
        "WHERE owner_id = %s AND namespace = %s AND `key` = %s "
        "AND deleted_at IS NULL FOR UPDATE"
    )


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` only if it's actually awaitable.

    Oracle/Db2 cursor methods have an inconsistent sync/async surface
    across drivers and wrappers: real python-oracledb's ``AsyncCursor
    .close()`` is a plain synchronous call returning ``None`` (closing
    a cursor needs no DB round trip), while this repo's own
    ``Db2BackendNative`` cursor wrapper makes ``close()`` a coroutine.
    Awaiting unconditionally 503'd every real Oracle consultation with
    ``TypeError: object NoneType can't be used in 'await' expression``
    even after the initial ``conn.cursor()``-await bug was fixed
    (2026-09-14) -- this normalizes both shapes instead of assuming one.
    """
    return await value if inspect.isawaitable(value) else value


async def _lock_quota_row_for_update(tx: Any, principal_key: str) -> None:
    """Take a row-level lock on the (owner_id, namespace, key) quota row.

    Backend dispatch — all paths are equivalent to "serialize concurrent
    read-modify-write on this row inside the open transaction":

      * Postgres: pg_advisory_xact_lock(int64) keyed on a stable hash of
        the principal. Same proven pattern as the audit-chain advisory
        lock earlier in this file.
      * Oracle / Db2: ``conn.cursor()`` (no await — cursor creation is
        not an I/O operation in either driver) then SELECT ... FOR
        UPDATE on the ``state`` row. ``execute``/``fetchone`` are
        coroutines on both drivers, but ``close()`` is NOT consistently
        one — real python-oracledb's is a plain sync call, while this
        repo's Db2BackendNative wrapper's is a coroutine — so ``close()``
        goes through ``_maybe_await`` rather than a bare ``await``.
      * MySQL / MariaDB: ``async with conn.cursor() as cursor`` then
        SELECT ... FOR UPDATE. aiomysql returns a context-manager cursor
        from ``conn.cursor()`` directly.
      * Sqlite: ``BEGIN IMMEDIATE`` already holds a RESERVED write lock
        on the only writer connection; no extra statement needed.

    Lock/driver errors propagate to fail-closed admission. Oracle creates
    the first quota row before locking; SELECT FOR UPDATE alone cannot
    serialize requests when that row does not yet exist.
    """
    backend = _lc._persistence_backend
    backend_name = type(backend).__name__ if backend is not None else ""
    conn = getattr(tx, "conn", None)
    if conn is None:
        return

    if backend_name == _POSTGRES_BACKEND_NAME:
        # asyncpg: ``conn.execute`` with a bind parameter, no commit
        # needed — pg_advisory_xact_lock is tx-scoped. Same template as
        # the audit-chain advisory lock earlier in this file.
        await conn.execute(
            "SELECT pg_advisory_xact_lock($1)", _advisory_lock_key(principal_key)
        )
        return

    if backend_name in {"Db2Backend", "Db2BackendNative"}:
        cursor = conn.cursor()
        params = (_QUOTA_OWNER_ID, _QUOTA_NAMESPACE, principal_key)
        try:
            from mnemos.persistence.oracle import _is_unique_violation

            try:
                await cursor.execute(
                    "INSERT INTO state (owner_id, namespace, key, value) "
                    "VALUES (?, ?, ?, '[]')",
                    params,
                )
            except Exception as exc:  # noqa: BLE001 - only duplicate rows are expected
                if not _is_unique_violation(exc):
                    raise
            await cursor.execute(
                "SELECT key FROM state WHERE owner_id = ? "
                "AND namespace = ? AND key = ? FOR UPDATE WITH RS",
                params,
            )
            await cursor.fetchone()
        finally:
            await _maybe_await(cursor.close())
        return

    if backend_name == "OracleBackend":
        # cursor() is synchronous on both python-oracledb's async
        # connection and this repo's own Db2BackendNative wrapper
        # (mnemos/persistence/db2.py::_Db2NativeAsyncConnection.cursor) --
        # only execute/fetch are coroutines. Awaiting the plain cursor
        # object raises TypeError: object AsyncCursor can't be used in
        # 'await' expression, which took every consultation down with a
        # 503 in production (2026-09-14).
        cursor = conn.cursor()
        try:
            # Oracle does not gap-lock a missing row. Insert first and
            # ignore only the expected unique-key race; the winner's
            # transaction must finish before the loser can lock/read it.
            await cursor.execute(
                "BEGIN INSERT INTO state (owner_id, namespace, key, value) "
                "VALUES (:owner_id, :namespace, :key, '[]'); "
                "EXCEPTION WHEN DUP_VAL_ON_INDEX THEN NULL; END;",
                {
                    "owner_id": _QUOTA_OWNER_ID,
                    "namespace": _QUOTA_NAMESPACE,
                    "key": principal_key,
                },
            )
            await cursor.execute(
                "SELECT key FROM state "
                "WHERE owner_id = :owner_id AND namespace = :namespace "
                "AND key = :key FOR UPDATE",
                {
                    "owner_id": _QUOTA_OWNER_ID,
                    "namespace": _QUOTA_NAMESPACE,
                    "key": principal_key,
                },
            )
            # Fetch failures must abort admission; they are not evidence of
            # an empty row and must not silently bypass quota serialization.
            await cursor.fetchone()
        finally:
            await _maybe_await(cursor.close())
        return

    if backend_name in {"MysqlBackend", "MariadbBackend"}:
        # aiomysql: ``conn.cursor()`` returns an async context manager
        # directly (no await).
        #
        # ``key`` is a RESERVED WORD in MySQL/MariaDB and the unquoted
        # SELECT that previously lived here failed with ``1064 You have
        # an error in your SQL syntax`` on those backends, which the
        # caller mapped to a generic 503. Build the dialect-correct
        # SQL via _mysql_lock_sql() so the identifier is backtick-
        # quoted on MySQL/MariaDB specifically.
        async with conn.cursor() as cursor:
            # Create/lock the row before a gap-locking SELECT can deadlock
            # competing first-use inserts. A duplicate is an unchanged row.
            await cursor.execute(
                "INSERT INTO state (`owner_id`, `namespace`, `key`, value) "
                "VALUES (%s, %s, %s, '[]') ON DUPLICATE KEY UPDATE `key` = `key`",
                (_QUOTA_OWNER_ID, _QUOTA_NAMESPACE, principal_key),
            )
            await cursor.execute(
                _mysql_lock_sql(),
                (_QUOTA_OWNER_ID, _QUOTA_NAMESPACE, principal_key),
            )
            await cursor.fetchone()
        return

    # Sqlite + any other backend: BEGIN IMMEDIATE / transaction-level
    # serialization is already in effect. No extra statement needed.


def _prune_quota_entries(
    entries: list[dict[str, Any]], *, now_ms: int
) -> list[dict[str, Any]]:
    """Drop entries older than the sliding window and bound total size."""
    cutoff = now_ms - int(_QUOTA_WINDOW_SECONDS * 1000)
    pruned = [entry for entry in entries if int(entry.get("ts_ms", 0)) > cutoff]
    if len(pruned) > _QUOTA_MAX_ENTRIES:
        # Keep the most recent MAX_ENTRIES — older ones are either in the
        # window (and thus already counted) or stale. We can't have
        # overflowed past MAX_ENTRIES legitimately inside one minute
        # without already being throttled, so this is a defensive trim.
        pruned = pruned[-_QUOTA_MAX_ENTRIES:]
    return pruned


async def _enforce_consult_quota(user_id: str, mode: str) -> None:
    """Atomically enforce the shared, cost-weighted per-principal quota.

    The whole read-prune-write happens inside one ``backend.transactional()``
    block so the row lock taken by ``_lock_quota_row_for_update`` actually
    spans the read-modify-write window. Two concurrent requests for the
    same principal queue at the row lock; the second sees the first's
    appended entry and is rejected if it pushes the principal over quota.
    """
    quota = _consult_quota_per_min()
    if quota <= 0:
        return

    weight = _MODE_COST_WEIGHT.get((mode or "auto").lower(), 2.0)
    principal_key = _quota_principal_key(user_id)

    backend = backend_or_503()
    state_kv = getattr(backend, "state_kv", None)
    if state_kv is None:
        # Backend doesn't implement the StateRepository ABC. We have no
        # shared storage path — fail closed with the same shape the prior
        # external-cache path used.
        raise HTTPException(
            status_code=503,
            detail="Shared consultation quota storage is unavailable.",
        )

    now_ms = int(time.time() * 1000)

    try:
        async with backend.transactional() as tx:
            await _lock_quota_row_for_update(tx, principal_key)
            existing = await state_kv.get(
                tx,
                principal_key,
                owner_id=_QUOTA_OWNER_ID,
                namespace=_QUOTA_NAMESPACE,
            )
            try:
                raw_value = existing["value"] if existing else None
                entries = json.loads(raw_value) if raw_value else []
                if not isinstance(entries, list):
                    entries = []
            except (TypeError, ValueError, json.JSONDecodeError):
                entries = []

            entries = _prune_quota_entries(entries, now_ms=now_ms)
            used = sum(float(entry.get("weight", 0.0)) for entry in entries)

            if used + weight > quota:
                # 429 with Retry-After. Match the prior external-cache path's
                # response shape exactly so existing clients see the same
                # contract.
                if entries:
                    oldest_ts_ms = int(entries[0].get("ts_ms", now_ms))
                    retry = max(
                        1,
                        (
                            (oldest_ts_ms + int(_QUOTA_WINDOW_SECONDS * 1000) - now_ms)
                            + 999
                        )
                        // 1000,
                    )
                else:
                    retry = 1
                # Persist the pruned list so we don't leak stale entries
                # that survived the cutoff but were rejected this turn.
                await state_kv.set(
                    tx,
                    principal_key,
                    json.dumps(entries, separators=(",", ":")),
                    owner_id=_QUOTA_OWNER_ID,
                    namespace=_QUOTA_NAMESPACE,
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"GRAEAE consultation quota exceeded for this principal "
                        f"({used:.0f}+{weight:.0f} > {quota:.0f} cost-units/min)"
                    ),
                    headers={"Retry-After": str(retry)},
                )

            entries.append({"ts_ms": now_ms, "weight": weight})
            await state_kv.set(
                tx,
                principal_key,
                json.dumps(entries, separators=(",", ":")),
                owner_id=_QUOTA_OWNER_ID,
                namespace=_QUOTA_NAMESPACE,
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("[CONSULTATION] shared quota check failed")
        raise HTTPException(
            status_code=503,
            detail="Shared consultation quota storage is unavailable.",
        )


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
            detail=(f"unknown tier {tier!r}; expected one of {sorted(_VALID_TIERS)}"),
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
            detail=(
                f"Custom Query accepts at most one of "
                f"{{'models', 'providers', 'tier'}}; got {set_fields}"
            ),
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


# ── Consultation endpoint ─────────────────────────────────────────────────────


@router.post("/consultations", response_model=ConsultationResponse)
@limiter.limit("60/minute")
async def consult_graeae(
    request: Request,
    body: ConsultationRequest,
    user: UserContext = Depends(get_current_user),
):
    """Consult GRAEAE multi-provider consensus engine.

    Creates a hash-chained audit entry and records any injected memories.
    Returns raw provider responses (full, best, or truncated per format param).
    """
    logger.info(
        f"[CONSULTATION] {user.user_id}: {body.task_type} "
        f"(limit_chars={body.limit_chars}, format={body.format})"
    )
    # SECURITY (review #12): cost-weighted per-principal quota, BEFORE any
    # provider fan-out, so a single caller cannot turn 60 req/min of
    # mode=all/debate into a financial DoS.
    await _enforce_consult_quota(user.user_id, body.mode)
    # Validate the signing key before provider fan-out incurs cost. The same
    # precomputed genesis is passed to persistence after the provider returns.
    genesis_hash = audit_genesis_hash()
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
            successful = [
                item
                for item in result["all_responses"].items()
                if item[1].get("status") == "success"
            ]
            best = max(
                successful,
                key=lambda item: item[1].get("final_score", 0),
                default=None,
            )
            result["all_responses"] = {best[0]: best[1]} if best else {}

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
            engine_cost = (
                float(result["cost"]) if result.get("cost") is not None else None
            )
            engine_latency_ms = int(result.get("latency_ms", 0) or 0)

            # All three writes — consultation row, audit entry, memory refs —
            # must commit as a single unit. If the audit write fails we MUST
            # abort the consultation row: tamper-evidence requires that a
            # committed consultation implies a committed audit chain link.
            try:
                async with backend.transactional() as tx:
                    consultation_id = (
                        await backend.consultations.create_consultation_with_audit(
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
                            genesis_hash=genesis_hash,
                            # Core signs each link with the same key the verifier
                            # below uses. Without this the writer fell back to an
                            # unkeyed 3-field SHA-256 and verify() reported the
                            # chain unsigned -- the failure the comment above
                            # predicted.
                            audit_key=_audit_hmac_key(),
                        )
                    )
                    if getattr(backend, "supports_webhooks", True):
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
                logger.error(
                    f"[CONSULTATION] persist failed — aborting: {e}", exc_info=True
                )
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
            consultation_id=str(consultation_id)
            if consultation_id is not None
            else None,
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
            contradiction_pairs=result.get("contradiction_pairs"),
            agreement_metric=result.get("agreement_metric"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CONSULTATION] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Consultation failed — see server logs for details"
        )


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
            message=(
                "Audit log is empty"
                if verify_global_chain
                else "No audit entries visible for caller"
            ),
        )

    if verify_global_chain:
        prev_chain = audit_genesis_hash()
        failures: dict[int, str] = {}
        for row in rows:
            # Verify each row under the algorithm that SIGNED it. Rows written
            # before the chain was versioned carry NULL/sha256-v1 and are
            # verified with the legacy unkeyed hash; re-signing them under v2
            # would make them verify at the cost of destroying the tamper
            # evidence the log exists to provide.
            expected = audit_chain.compute_for_algo(
                row.get("chain_algo"),
                key=_audit_hmac_key(),
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
                message = (
                    f"{message}; {len(entries_failed)} entries failed verification"
                )
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
            message = (
                f"{message}; {len(entries_failed)} scoped entries failed verification"
            )
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
                "description": "fan out to up to five muses and report whether quorum agreement was reached",
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
        "created_at": row["created"].isoformat()
        if hasattr(row["created"], "isoformat")
        else str(row["created"]),
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
        (
            consultation,
            memory_refs,
        ) = await backend.consultations.get_consultation_artifacts(
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


# Verbatim classified recall — paging/section helpers live in core
# (mnemos.domain.consultation_recall) so this route and the
# graeae_get_consultation MCP tool share identical semantics.
from mnemos.domain.consultation_recall import (  # noqa: E402
    VALID_FULL_SECTIONS as _VALID_FULL_SECTIONS,
)
from mnemos.domain.consultation_recall import (  # noqa: E402
    paginate_full_parts as _paginate_full_parts,
)
from mnemos.domain.consultation_recall import (  # noqa: E402
    select_full_parts as _select_full_parts,
)


@router.get("/consultations/{consultation_id}/full")
@limiter.limit("30/minute")
async def get_consultation_full(
    request: Request,
    consultation_id: str,
    section: str = Query(
        "all", description="all | source | quorum | synthesis | muses"
    ),
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
        raise HTTPException(
            status_code=422, detail=f"section must be one of {_VALID_FULL_SECTIONS}"
        )
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
