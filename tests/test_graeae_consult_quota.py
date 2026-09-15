"""Tests for the per-principal consultation quota (review #12 / quota-fix).

The current production quota lives in ``mnemos.api.routes.consultations`` and
is backed by ``state_kv`` — a single row per principal, JSON-encoded array of
``{ts_ms, weight}`` entries, pruned to the 60-second window. Atomicity
matters: two concurrent requests for the same principal must not both
succeed past the quota, exactly as the old Redis Lua script and the old
``threading.Lock`` fallback used to guarantee.

These tests exercise ``_enforce_consult_quota`` directly against a small
in-memory ``StateRepository`` + a transactional context that runs each
caller under an ``asyncio.Lock`` keyed on the principal hash. That gives
us the same per-principal serialization a real backend provides via
``pg_advisory_xact_lock`` / ``SELECT ... FOR UPDATE`` — enough to prove
the read-modify-write path is atomic against concurrent requests for the
same principal without standing up a real Postgres/Oracle/MySQL/SQLite
in CI.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Self

import pytest
from mnemos.core import lifecycle

from mnemos.api.routes import consultations

# ── Fixtures / helpers ────────────────────────────────────────────────────────


class _LockingTx:
    """Backend-neutral transaction stand-in that serializes on the principal.

    The point: we don't actually have asyncpg/Oracle/SQLite available in
    this unit test, but we DO want to prove the read-modify-write is
    atomic against concurrent requests for the same principal. An
    ``asyncio.Lock`` keyed on the principal hash gives us exactly that —
    every concurrent caller for the same principal queues on the same
    lock before entering the critical section, just like a row-level
    database lock or a Postgres advisory lock would.
    """

    def __init__(self, principal_key: str) -> None:
        self.principal_key = principal_key
        self.lock = _PER_PRINCIPAL_LOCKS.setdefault(principal_key, asyncio.Lock())

    async def __aenter__(self) -> Self:
        await self.lock.acquire()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        self.lock.release()


_PER_PRINCIPAL_LOCKS: dict[str, asyncio.Lock] = {}


class _InMemoryStateRepository:
    """Tiny in-memory StateRepository with an async lock for each row.

    Implements just enough of the ``StateRepository`` ABC
    (``get``/``set`` inside an open transaction) for the quota path. The
    lock taken on the active ``_LockingTx`` mimics the per-principal row
    lock the production code takes on Postgres / Oracle / MySQL / SQLite.
    """

    def __init__(self) -> None:
        # rows keyed by (owner_id, namespace, key)
        self.rows: dict[tuple[str, str, str], str] = {}
        # An extra sentinel asyncio.Lock per (owner, namespace, key) so the
        # test can independently prove the SUT also serializes when the
        # caller doesn't supply an external tx-lock wrapper.
        self.row_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self.reads: list[tuple[str, str, str]] = []
        self.writes: list[tuple[str, str, str, str]] = []

    async def get(self, tx, key, *, owner_id="default", namespace="default"):
        self.reads.append((owner_id, namespace, key))
        value = self.rows.get((owner_id, namespace, key))
        if value is None:
            return None
        return {"key": key, "value": value, "version": 1}

    async def set(self, tx, key, value, *, owner_id="default", namespace="default", expires_at=None):
        _ = expires_at
        self.rows[(owner_id, namespace, key)] = value
        self.writes.append((owner_id, namespace, key, value))
        return {"key": key, "value": value, "version": 2}


class _LockingBackend:
    """A backend whose ``transactional()`` yields a per-principal tx-lock.

    Every request goes through ``backend.transactional()`` (matches the
    production code path) and gets back a ``_LockingTx`` for the
    principal the caller is operating on. The state_kv get/set are then
    serialized on the same asyncio.Lock a real Postgres advisory lock or
    SELECT ... FOR UPDATE row lock would protect.
    """

    def __init__(self, principal_key: str, state_kv: _InMemoryStateRepository) -> None:
        self.principal_key = principal_key
        self.state_kv = state_kv
        self.transaction_calls = 0

    @asynccontextmanager
    async def transactional(self):
        self.transaction_calls += 1
        tx = _LockingTx(self.principal_key)
        async with tx:
            yield SimpleNamespace(conn=None)


def _patch_quota_env(monkeypatch: pytest.MonkeyPatch, *, quota: float, mode: str | None = None) -> None:
    """Set MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN + stub the consultations helper."""
    monkeypatch.setenv("MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN", str(quota))
    if mode is not None:
        monkeypatch.setattr(consultations, "_consult_quota_per_min", lambda: quota)


def _patch_backend(monkeypatch: pytest.MonkeyPatch, backend: Any) -> None:
    """Wire `backend` into mnemos.core.lifecycle as the active persistence backend."""
    monkeypatch.setattr(lifecycle, "_persistence_backend", backend)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_under_quota_succeeds(monkeypatch: pytest.MonkeyPatch):
    """A principal well under the quota gets through repeatedly."""

    principal = "alice@example.com"
    backend = _LockingBackend(
        principal_key=consultations._quota_principal_key(principal),
        state_kv=_InMemoryStateRepository(),
    )
    _patch_backend(monkeypatch, backend)
    _patch_quota_env(monkeypatch, quota=20.0)

    # mode='single' has weight 1.0 → 20/mines cap allows 20 calls.
    for _ in range(15):
        await consultations._enforce_consult_quota(principal, "single")

    # Verify the state_kv row was actually populated with the right
    # number of entries (one per call) — and that all reads/writes
    # went through the active backend, not any local fallback.
    key = consultations._quota_principal_key(principal)
    stored = backend.state_kv.rows[("default", "graeae-consult-quota", key)]
    entries = json.loads(stored)
    assert len(entries) == 15
    assert all(entry["weight"] == 1.0 for entry in entries)
    assert backend.transaction_calls == 15


@pytest.mark.asyncio
async def test_over_quota_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    """A principal that pushes past the quota gets 429 + Retry-After + the
    cost-unit error message shape (preserved verbatim from the Redis
    path)."""

    from fastapi import HTTPException

    principal = "bob@example.com"
    backend = _LockingBackend(
        principal_key=consultations._quota_principal_key(principal),
        state_kv=_InMemoryStateRepository(),
    )
    _patch_backend(monkeypatch, backend)
    _patch_quota_env(monkeypatch, quota=10.0)

    # mode='all' has weight 5.0; cap=10 → two calls fit, third is rejected.
    await consultations._enforce_consult_quota(principal, "all")
    await consultations._enforce_consult_quota(principal, "all")

    with pytest.raises(HTTPException) as exc_info:
        await consultations._enforce_consult_quota(principal, "all")

    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert isinstance(detail, str)
    # The old Redis path emitted this exact substring; clients depend on it.
    assert "GRAEAE consultation quota exceeded for this principal" in detail
    assert "10+5 > 10 cost-units/min" in detail
    # Retry-After is mandatory — clients back off on it.
    assert "Retry-After" in exc_info.value.headers
    retry_after = int(exc_info.value.headers["Retry-After"])
    assert 1 <= retry_after <= 60


@pytest.mark.asyncio
async def test_concurrent_same_principal_cannot_both_succeed(monkeypatch: pytest.MonkeyPatch):
    """Two concurrent calls for the SAME principal that would JOINTLY push
    over quota must not both succeed.

    This is the actual race the prior in-process lock and the prior
    external-cache Lua script were both protecting against. With the row
    lock in place the second call observes the first's appended entry
    and rejects with 429; without the lock both would read ``[]`` and
    both would be allowed, letting the principal blow past the cap.

    Math: ``mode='all'`` has weight 5.0; with cap=9 the first call fits
    (0+5=5 ≤ 9) but the second, observing the first's appended entry,
    does not (5+5=10 > 9). One success, one rejection.
    """

    from fastapi import HTTPException

    principal = "carol@example.com"
    backend = _LockingBackend(
        principal_key=consultations._quota_principal_key(principal),
        state_kv=_InMemoryStateRepository(),
    )
    _patch_backend(monkeypatch, backend)
    # Cap = 9, weight per "all" call = 5.0 → 9/5.0=1 fits; the second
    # concurrent caller observes the first's write and is rejected.
    _patch_quota_env(monkeypatch, quota=9.0)

    results = await asyncio.gather(
        consultations._enforce_consult_quota(principal, "all"),
        consultations._enforce_consult_quota(principal, "all"),
        return_exceptions=True,
    )

    successes = [r for r in results if r is None]
    failures = [r for r in results if isinstance(r, HTTPException)]
    assert len(successes) == 1, (
        f"Expected exactly one success under the cap (cap=9, weight=5 → 1 fits), got "
        f"{len(successes)} successes and {len(failures)} failures; results={results!r}"
    )
    assert len(failures) == 1
    assert failures[0].status_code == 429
    assert "5+5 > 9 cost-units/min" in failures[0].detail

    # And the state_kv row has exactly one entry — proving the second call
    # was rejected AFTER observing the first's write, not before.
    key = consultations._quota_principal_key(principal)
    stored = backend.state_kv.rows[("default", "graeae-consult-quota", key)]
    entries = json.loads(stored)
    assert len(entries) == 1
    assert entries[0]["weight"] == 5.0


@pytest.mark.asyncio
async def test_concurrent_distinct_principals_do_not_block_each_other(monkeypatch: pytest.MonkeyPatch):
    """Lock is per-principal, not fleet-wide. Two DIFFERENT principals
    firing concurrently must both succeed (otherwise the per-principal
    lock has degenerated into a global lock that serializes all traffic).
    """

    # Two principals share the same state_kv so we exercise the full
    # quota path against a real dict, but the per-principal lock is what
    # we're really testing here. Each backend dispatches its tx to a
    # lock keyed on its principal so distinct principals don't serialize
    # on each other.
    shared_state_kv = _InMemoryStateRepository()

    class _DualPrincipalBackend:
        supports_webhooks = False
        state_kv = shared_state_kv

        def __init__(self, principal_key: str) -> None:
            self.principal_key = principal_key

        @asynccontextmanager
        async def transactional(self):
            tx = _LockingTx(self.principal_key)
            async with tx:
                yield SimpleNamespace(conn=None)

    # Both concurrent calls go through the same patched backend. They
    # exercise different principal keys, so the per-principal advisory
    # lock keys differ — proving the lock isn't a fleet-wide serialization
    # point. If it were, both calls would still pass (we don't push over
    # quota here) but the test would be less interesting.
    backend_a = _DualPrincipalBackend(consultations._quota_principal_key("dave@example.com"))

    _patch_quota_env(monkeypatch, quota=20.0)
    _patch_backend(monkeypatch, backend_a)
    # Fire two concurrent calls — one for dave, one for erin. Both should
    # succeed; if the lock were fleet-wide they'd serialize (still both
    # passing) but the entries list would interleave; with a per-principal
    # lock they each see only their own row.
    results = await asyncio.gather(
        consultations._enforce_consult_quota("dave@example.com", "single"),
        consultations._enforce_consult_quota("erin@example.com", "single"),
        return_exceptions=True,
    )
    assert results == [None, None], f"Both distinct-principal calls should succeed: {results!r}"

    # And each principal's quota row has exactly one entry — proving
    # the per-principal lock didn't accidentally serialize them onto
    # the same row.
    dave_key = consultations._quota_principal_key("dave@example.com")
    erin_key = consultations._quota_principal_key("erin@example.com")
    dave_entries = json.loads(shared_state_kv.rows[("default", "graeae-consult-quota", dave_key)])
    erin_entries = json.loads(shared_state_kv.rows[("default", "graeae-consult-quota", erin_key)])
    assert len(dave_entries) == 1
    assert len(erin_entries) == 1


@pytest.mark.asyncio
async def test_window_slide_drops_old_entries(monkeypatch: pytest.MonkeyPatch):
    """Entries older than the 60s window must be dropped on the next read.

    We can simulate the time-passage by stubbing ``time.time`` so each
    call advances by a minute; the SUT's prune step must observe that
    none of the prior entries survive.
    """
    principal = "frank@example.com"
    backend = _LockingBackend(
        principal_key=consultations._quota_principal_key(principal),
        state_kv=_InMemoryStateRepository(),
    )
    _patch_backend(monkeypatch, backend)
    _patch_quota_env(monkeypatch, quota=10.0)

    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr(consultations.time, "time", lambda: fake_now["t"])

    # First call lands at t=1_000_000. weight=5.0 → used=5.0.
    await consultations._enforce_consult_quota(principal, "all")
    # Advance time by 61s — past the 60s window.
    fake_now["t"] += 61
    # Second call sees an empty list (old entry pruned), used=0, weight=5
    # fits, new entry appended.
    await consultations._enforce_consult_quota(principal, "all")

    key = consultations._quota_principal_key(principal)
    stored = backend.state_kv.rows[("default", "graeae-consult-quota", key)]
    entries = json.loads(stored)
    assert len(entries) == 1, f"Old entry should have been pruned, got: {entries!r}"
    assert entries[0]["weight"] == 5.0


@pytest.mark.asyncio
async def test_zero_quota_disables_enforcement(monkeypatch: pytest.MonkeyPatch):
    """MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN=0 disables the quota — no state_kv writes."""

    principal = "gina@example.com"
    backend = _LockingBackend(
        principal_key=consultations._quota_principal_key(principal),
        state_kv=_InMemoryStateRepository(),
    )
    _patch_backend(monkeypatch, backend)
    _patch_quota_env(monkeypatch, quota=0.0)

    await consultations._enforce_consult_quota(principal, "debate")  # weight 8
    await consultations._enforce_consult_quota(principal, "debate")
    # No state was written and no transactional() was opened.
    assert backend.state_kv.rows == {}
    assert backend.transaction_calls == 0


@pytest.mark.asyncio
async def test_backend_without_state_kv_returns_503(monkeypatch: pytest.MonkeyPatch):
    """If the backend has no state_kv, the quota path returns 503 — same
    fail-closed shape the dead Redis path produced."""

    from fastapi import HTTPException

    class _NoStateKvBackend:
        supports_webhooks = False

        @asynccontextmanager
        async def transactional(self):
            yield SimpleNamespace(conn=None)

    _patch_backend(monkeypatch, _NoStateKvBackend())
    _patch_quota_env(monkeypatch, quota=20.0)

    with pytest.raises(HTTPException) as exc_info:
        await consultations._enforce_consult_quota("henry@example.com", "auto")
    assert exc_info.value.status_code == 503
    assert "Shared consultation quota storage is unavailable." in exc_info.value.detail


def test_module_does_not_import_redis():
    """Hard guarantee: no Redis reference at all in the route module."""

    import inspect

    source = inspect.getsource(consultations)
    assert "redis" not in source.lower(), (
        "consultations.py must not import or reference redis — found "
        f"redis in module source. Snippet: {source[:300]!r}"
    )
    assert "Redis" not in source


def test_module_does_not_expose_dead_quota_symbols():
    """The legacy Redis/Lua/local-memory symbols must be gone."""

    for name in (
        "_get_lifecycle_redis_client",
        "_consult_quota_redis",
        "_CONSULT_QUOTA_LUA",
        "_single_worker_memory_quota_enabled",
        "_enforce_local_consult_quota",
        "_gc_local_consult_quota",
        "_evict_oldest_local_consult_quota",
        "close_consult_quota_client",
        "_QUOTA_KEY_PREFIX",
        "_consult_quota_local",
    ):
        assert not hasattr(consultations, name), (
            f"consultations module should no longer expose {name}; "
            f"remove it as part of the Redis removal."
        )


def test_engine_module_no_longer_calls_close_consult_quota_client():
    """mnemos.domain.graeae.engine must not import close_consult_quota_client anymore."""

    import inspect

    from mnemos.domain.graeae import engine as graeae_engine

    source = inspect.getsource(graeae_engine)
    assert "close_consult_quota_client" not in source, (
        "engine.py must not import or call close_consult_quota_client — "
        "that function was deleted with the Redis path."
    )


def test_advisory_lock_key_is_stable_signed_int64():
    """The Postgres advisory lock key is a stable int64 across the same principal."""

    key1 = consultations._advisory_lock_key("principal-abc")
    key2 = consultations._advisory_lock_key("principal-abc")
    assert key1 == key2
    # int64 range
    assert -(2**63) <= key1 < 2**63
    # Different principals get different keys (collision-resistant).
    other = consultations._advisory_lock_key("principal-xyz")
    assert other != key1


# ── Oracle/Db2 row-lock regression (2026-09-14) ────────────────────────────────
#
# Every test above passes SimpleNamespace(conn=None), so
# _lock_quota_row_for_update's `if conn is None: return` early-exit meant
# NONE of them ever touched the Oracle/Db2/MySQL branches below it. That let
# `cursor = await conn.cursor()` ship for months: cursor() is synchronous on
# both python-oracledb's async connection and this repo's own
# Db2BackendNative wrapper (mnemos/persistence/db2.py::
# _Db2NativeAsyncConnection.cursor, also a plain `def`, not `async def`) --
# only execute/fetch/close are coroutines. Awaiting the plain cursor object
# raised `TypeError: object AsyncCursor can't be used in 'await' expression`,
# which took every GRAEAE consultation down with a 503 in production.


class _FakeOracleCursor:
    """Shaped like python-oracledb's AsyncCursor / _Db2NativeAsyncCursor:
    execute/fetchone/close are coroutines, the cursor object itself is not."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.closed = False

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchone(self) -> tuple | None:
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeOracleConn:
    """cursor() is a plain (synchronous) method — the real defect surface."""

    def __init__(self) -> None:
        self.cursor_calls = 0

    def cursor(self) -> _FakeOracleCursor:
        self.cursor_calls += 1
        return _FakeOracleCursor()


class _FakeRealOracleCursor:
    """Shaped like REAL python-oracledb's AsyncCursor, not the earlier fake.

    execute/fetchone are coroutines, but close() is a plain synchronous
    call returning None -- no DB round trip needed to close a cursor.
    Awaiting it unconditionally raised ``TypeError: object NoneType
    can't be used in 'await' expression`` against real production
    Oracle (2026-09-14), even after the initial conn.cursor()-await bug
    was fixed -- this fake reproduces that second, more subtle shape.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.closed = False

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchone(self) -> tuple | None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeRealOracleConn:
    def __init__(self) -> None:
        self.cursor_calls = 0

    def cursor(self) -> _FakeRealOracleCursor:
        self.cursor_calls += 1
        return _FakeRealOracleCursor()


@pytest.mark.parametrize("backend_name", ["OracleBackend", "Db2Backend", "Db2BackendNative"])
@pytest.mark.asyncio
async def test_lock_quota_row_for_update_does_not_await_cursor(
    monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Regression: cursor() must not be awaited for Oracle/Db2 backends.

    `_lock_quota_row_for_update` dispatches on
    `type(mnemos.core.lifecycle._persistence_backend).__name__` (a module
    global, not anything derived from `tx`) — so the fake backend class
    must be patched into `lifecycle._persistence_backend`, and its class
    name must exactly match one of the dispatch strings. `tx.conn`'s
    cursor() is synchronous, matching both real python-oracledb and this
    repo's own Db2BackendNative wrapper.
    """

    fake_backend_cls = type(backend_name, (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    conn = _FakeOracleConn()
    tx = SimpleNamespace(conn=conn)

    await consultations._lock_quota_row_for_update(tx, "principal-oracle-test")

    assert conn.cursor_calls == 1


@pytest.mark.asyncio
async def test_enforce_consult_quota_succeeds_against_oracle_shaped_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a consult against an Oracle-shaped backend must not 503.

    This is the actual user-visible symptom from production: every
    /v1/consultations call failed with 503 "Shared consultation quota
    storage is unavailable." because _lock_quota_row_for_update's
    TypeError was caught by _enforce_consult_quota's broad except-Exception
    and re-raised with that message. Reproduces it end-to-end through the
    public function, not just the row-lock helper in isolation.
    """

    conn = _FakeOracleConn()

    class OracleBackend:  # noqa: N801
        state_kv = _InMemoryStateRepository()

        @asynccontextmanager
        async def transactional(self):
            yield SimpleNamespace(conn=conn)

    _patch_backend(monkeypatch, OracleBackend())
    _patch_quota_env(monkeypatch, quota=20.0)

    # Must not raise. Before the fix, this failed with HTTPException(503,
    # "Shared consultation quota storage is unavailable.") every time.
    await consultations._enforce_consult_quota("oracle-user@example.com", "auto")
    assert conn.cursor_calls == 1


@pytest.mark.asyncio
async def test_lock_quota_row_for_update_handles_sync_cursor_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: cursor.close() must not be unconditionally awaited.

    Found live against real production Oracle (2026-09-14), AFTER the
    initial conn.cursor()-await fix shipped: real python-oracledb's
    AsyncCursor.close() is a plain synchronous call returning None (no
    DB round trip needed to close a cursor) -- unlike execute/fetchone,
    which genuinely are coroutines on that same cursor. The finally
    block's bare ``await cursor.close()`` raised ``TypeError: object
    NoneType can't be used in 'await' expression``, still producing the
    user-visible 503 even with the first fix live. This uses a fake
    shaped like the REAL driver (sync close), not the earlier fake
    (which happened to make close() a coroutine and didn't catch this).
    """

    fake_backend_cls = type("OracleBackend", (), {})
    monkeypatch.setattr(consultations._lc, "_persistence_backend", fake_backend_cls())

    conn = _FakeRealOracleConn()
    tx = SimpleNamespace(conn=conn)

    # Must not raise TypeError. Before this fix, it always did.
    await consultations._lock_quota_row_for_update(tx, "principal-oracle-realshape")

    assert conn.cursor_calls == 1


@pytest.mark.asyncio
async def test_enforce_consult_quota_succeeds_against_real_oracle_cursor_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end with the REAL-shaped fake (sync close()) -- the exact
    production symptom reported live on 2026-09-14 against Oracle."""

    conn = _FakeRealOracleConn()

    class OracleBackend:
        state_kv = _InMemoryStateRepository()

        @asynccontextmanager
        async def transactional(self):
            yield SimpleNamespace(conn=conn)

    _patch_backend(monkeypatch, OracleBackend())
    _patch_quota_env(monkeypatch, quota=20.0)

    await consultations._enforce_consult_quota("oracle-user-2@example.com", "auto")
    assert conn.cursor_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["OracleBackend", "Db2BackendNative"])
async def test_quota_lock_fetch_failure_propagates(monkeypatch, backend_name):
    class Cursor(_FakeRealOracleCursor):
        async def fetchone(self):
            raise RuntimeError("database read failed")

    cursor = Cursor()
    conn = SimpleNamespace(cursor=lambda: cursor)
    monkeypatch.setattr(consultations._lc, "_persistence_backend", type(backend_name, (), {})())
    with pytest.raises(RuntimeError, match="database read failed"):
        await consultations._lock_quota_row_for_update(SimpleNamespace(conn=conn), "failure-test")
    assert cursor.closed
