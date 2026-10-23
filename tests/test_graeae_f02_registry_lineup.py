"""F-02: the registry-weighted lineup must work on every backend that can
read model_registry, not just the Postgres pool.

Before the fix, ``GraeaeEngine._ranked_lineup`` was gated on
``mnemos.core.lifecycle._pool`` — a Postgres-only handle. On every other
persistence backend (SQLite, MySQL, MariaDB, Oracle, Db2) the registry
ranking was skipped entirely and the lineup silently fell back to the
static config.toml weights. The fix mirrors the pantheon
``catalog._registry_rows()`` shape: the backend-neutral
``consultations_audit.fetch_available_models`` path runs first, the
``_lc._pool`` (Postgres profile) remains an optional enrichment, and the
operator-visible "registry unavailable" warning fires once per process,
only when the registry genuinely cannot be read.

Each behaviour test below failed before the fix in the way it describes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from mnemos.core import lifecycle
from mnemos.domain.graeae import engine as engine_module
from mnemos.domain.graeae.engine import GraeaeEngine


# ── fakes ─────────────────────────────────────────────────────────────────────


class _FakeTx:
    pass


class _FakeAuditRepository:
    """Stand-in for ConsultationAuditRepository: returns the rows it was
    built with; raises ``exc`` when set. Records call count."""

    def __init__(self, rows: list | None = None, exc: Exception | None = None):
        self.rows = list(rows) if rows is not None else []
        self.exc = exc
        self.calls = 0

    async def fetch_available_models(self, tx) -> list:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return list(self.rows)


class _FakeBackend:
    """Stand-in for a persistence backend that advertises the
    consultations capability (Postgres, SQLite, MySQL, MariaDB, Oracle,
    Db2 all do)."""

    def __init__(self, rows: list | None = None, exc: Exception | None = None):
        self.consultations_audit = _FakeAuditRepository(rows, exc)

    def transactional(self):
        @asynccontextmanager
        async def _tx():
            yield _FakeTx()

        return _tx()


class _NoAuditBackend:
    """Backend object that advertises no consultations_audit capability."""


class _DistinctNameBackend(_FakeBackend):
    """A backend with a distinct type name so the cache-key source
    identity changes after a failover."""


class _PoolConn:
    def __init__(self, rows: list, calls: list, exc: Exception | None = None):
        self._rows = rows
        self._calls = calls
        self._exc = exc

    async def fetch(self, query, params=None):
        self._calls.append((query, params))
        if self._exc is not None:
            raise self._exc
        return list(self._rows)


class _FakePool:
    """asyncpg-shaped pool stand-in. Records every fetch for assertions."""

    def __init__(self, rows: list | None = None, exc: Exception | None = None):
        self.calls: list = []
        self.conn = _PoolConn(list(rows or []), self.calls, exc)

    def acquire(self):
        @asynccontextmanager
        async def _acquire():
            yield self.conn

        return _acquire()


class _FakeRankedLineupCache:
    """Inert ranked-lineup cache: always a miss, so every call re-reads
    the registry (this is what makes warn-once observable)."""

    def get(self, key):
        return None

    def set(self, key, value):
        return None


def _make_engine() -> GraeaeEngine:
    """GraeaeEngine bypassing __init__ (no http/settings/reliability
    stack needed for _ranked_lineup). Static weights: claude 0.60 >
    openai 0.50, so the unweighted fallback order is [claude, openai]."""
    engine = object.__new__(GraeaeEngine)
    engine.providers = {
        "openai": {"model": "gpt-5.5", "weight": 0.50},
        "claude": {"model": "claude-opus-4-8", "weight": 0.60},
    }
    engine._ranked_lineup_cache = _FakeRankedLineupCache()
    return engine


def _reset_warn(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_registry_ranking_warned", False)
    monkeypatch.setattr(lifecycle, "_pool", None, raising=False)
    monkeypatch.setattr(lifecycle, "_persistence_backend", None, raising=False)


_REGISTRY_UNAVAILABLE = "model_registry ranking unavailable"


# ── 1. non-Postgres backend: registry weights ARE applied ─────────────────────


async def test_non_postgres_backend_applies_registry_weights(monkeypatch, caplog):
    """A non-Postgres backend whose fetch_available_models carries
    graeae_weight must drive the lineup ordering — the registry weight
    (0.95) beats the static config weight (0.60).

    Fails before the fix: with no _lc._pool the registry branch was
    skipped entirely, the lineup fell back to the static order
    [claude, openai], and the 'registry unavailable' warning (added
    earlier for the silent degradation) was not even meant to cover a
    backend that CAN read the registry.
    """
    _reset_warn(monkeypatch)
    monkeypatch.setattr(lifecycle, "_pool", None)
    backend = _FakeBackend(
        [
            {"provider": "anthropic", "model_id": "claude-opus-4-8"},
            {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
        ]
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", backend)
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["openai", "claude"]
    assert lineup["openai"] == "gpt-5.5"
    assert lineup["claude"] == "claude-opus-4-8"
    assert backend.consultations_audit.calls == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


async def test_non_postgres_backend_without_weight_column_still_serves_lineup(
    monkeypatch, caplog
):
    """Backends whose neutral projection lacks graeae_weight (Postgres /
    MySQL / SQLite / Db2 guarantee only provider/model_id/display_name)
    still honour the registry: candidates missing a weight fall back to
    their static config weight, the lineup is still registry-sourced,
    and no warning is logged."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(lifecycle, "_pool", None)
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(
            [
                {"provider": "anthropic", "model_id": "claude-opus-4-8"},
                {"provider": "openai", "model_id": "gpt-5.5"},
            ]
        ),
    )
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["claude", "openai"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


# ── 2. Postgres: legacy pool query orders the lineup, byte-identical ─────────


async def test_postgres_backend_uses_pool_projection_for_ordering(monkeypatch):
    """With BOTH a Postgres backend and a pool, the legacy pool query
    (provider = ANY($1::text[]), ORDER BY graeae_weight DESC NULLS LAST)
    is what orders the lineup — identical to today's Postgres behavior.
    The neutral backend rows (no graeae_weight) must NOT override the
    pool projection's ordering."""
    _reset_warn(monkeypatch)
    pool = _FakePool(
        rows=[
            {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
            {"provider": "anthropic", "model_id": "claude-opus-4-8", "graeae_weight": 0.40},
        ]
    )
    monkeypatch.setattr(lifecycle, "_pool", pool)
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(
            # Deliberately graeae_weight-less: if the neutral rows won
            # the ordering, the pool query would never have run.
            [
                {"provider": "anthropic", "model_id": "claude-opus-4-8"},
                {"provider": "openai", "model_id": "gpt-5.5"},
            ]
        ),
    )
    engine = _make_engine()
    lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["openai", "claude"]
    assert len(pool.calls) == 1
    query, params = pool.calls[0]
    assert "provider = ANY($1::text[])" in query
    assert "ORDER BY graeae_weight DESC NULLS LAST" in query
    assert list(params) == ["openai", "anthropic"]
    assert lineup["openai"] == "gpt-5.5"
    assert lineup["claude"] == "claude-opus-4-8"


async def test_postgres_pool_failure_falls_back_to_backend_rows(monkeypatch, caplog):
    """Pool raises -> the neutral backend rows still rank the lineup
    (static weights, stable order) and nothing is logged at warning
    level — the registry WAS reachable via the backend."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(lifecycle, "_pool", _FakePool(exc=RuntimeError("pool down")))
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(
            [
                {"provider": "anthropic", "model_id": "claude-opus-4-8"},
                {"provider": "openai", "model_id": "gpt-5.5"},
            ]
        ),
    )
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["claude", "openai"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


async def test_pool_only_profile_preserves_legacy_behaviour(monkeypatch):
    """No persistence backend, pool set (the pre-F-02 Postgres shape):
    the legacy query runs and orders the lineup — byte-identical to the
    behaviour before the fix."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "_pool",
        _FakePool(
            rows=[
                {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
                {"provider": "anthropic", "model_id": "claude-opus-4-8", "graeae_weight": 0.40},
            ]
        ),
    )
    engine = _make_engine()
    lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["openai", "claude"]
    assert len(lifecycle._pool.calls) == 1
    query, params = lifecycle._pool.calls[0]
    assert "provider = ANY($1::text[])" in query
    assert "ORDER BY graeae_weight DESC NULLS LAST" in query


# ── 3. genuinely unavailable: warn ONCE, fall back to unweighted order ───────


async def test_unavailable_registry_warns_once_and_falls_back(monkeypatch, caplog):
    """No backend AND no pool: the lineup falls back to the static
    config weights and the operator-visible warning is logged exactly
    ONCE per process, no matter how many lineups are built (cache
    misses forced)."""
    _reset_warn(monkeypatch)
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup1 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup2 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup3 = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup1) == ["claude", "openai"]
    assert lineup1 == lineup2 == lineup3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _REGISTRY_UNAVAILABLE in warnings[0].getMessage()
    assert engine_module._registry_ranking_warned is True


async def test_backend_raises_warns_once(monkeypatch, caplog):
    """The backend exists but its fetch_available_models raises: same
    contract — one warning per process, unweighted fallback order."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(exc=RuntimeError("model_registry unreachable")),
    )
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup1 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup2 = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup1) == ["claude", "openai"]
    assert lineup1 == lineup2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _REGISTRY_UNAVAILABLE in warnings[0].getMessage()


async def test_backend_without_audit_capability_warns_once(monkeypatch, caplog):
    """A backend object that advertises no consultations_audit capability
    is a genuine unavailability: one warning, unweighted fallback order.
    Fails before the fix: the capability probe did not exist at all —
    ranking was gated only on the Postgres pool."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(lifecycle, "_persistence_backend", _NoAuditBackend())
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup1 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup2 = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup1) == ["claude", "openai"]
    assert lineup1 == lineup2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _REGISTRY_UNAVAILABLE in warnings[0].getMessage()


async def test_pool_only_profile_failure_warns_once(monkeypatch, caplog):
    """Pool-only (no backend) profile where the pool fetch fails: the
    legacy degradation path still warns once and falls back."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(lifecycle, "_pool", _FakePool(exc=RuntimeError("down")))
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup1 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup2 = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup1) == ["claude", "openai"]
    assert lineup1 == lineup2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _REGISTRY_UNAVAILABLE in warnings[0].getMessage()


# ── 4. per-call log spam guard ───────────────────────────────────────────────


async def test_no_per_call_log_spam_when_registry_is_fine(monkeypatch, caplog):
    """When the registry IS readable via the backend, repeated lineup
    builds (cache misses forced) log nothing at warning level — the
    F-02 requirement that the operator signal appears only for genuine
    unavailability."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(
            [
                {"provider": "anthropic", "model_id": "claude-opus-4-8"},
                {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
            ]
        ),
    )
    engine = _make_engine()
    with caplog.at_level(logging.DEBUG, logger="mnemos.domain.graeae.engine"):
        for _ in range(5):
            engine._ranked_lineup_cache = _FakeRankedLineupCache()
            lineup = await engine._ranked_lineup("reasoning", 2)
            assert list(lineup) == ["openai", "claude"]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


# ── selection override still wins over registry model ────────────────────────


async def test_selection_model_override_preserved(monkeypatch):
    """An explicit selection model override still wins over the registry
    model_id on the backend-neutral path."""
    _reset_warn(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "_persistence_backend",
        _FakeBackend(
            [
                {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
                {"provider": "anthropic", "model_id": "claude-opus-4-8"},
            ]
        ),
    )
    engine = _make_engine()
    lineup = await engine._ranked_lineup(
        "reasoning", 2, selection={"openai": "gpt-5.5-mini", "claude": None}
    )
    assert list(lineup) == ["openai", "claude"]
    assert lineup["openai"] == "gpt-5.5-mini"
    assert lineup["claude"] == "claude-opus-4-8"


# ── empty registry is a valid cached state, not a re-fetch trigger ──────────


async def test_empty_registry_is_cached_not_refetched(monkeypatch):
    """A backend that returns an empty registry list (no available rows)
    is a valid cached state: the second call for the same key is a cache
    hit, not a re-fetch. Fails before the fix: ``cache.get(...) or []``
    treated the cached empty list as a miss and re-queried every time."""
    _reset_warn(monkeypatch)
    backend = _FakeBackend(rows=[])
    monkeypatch.setattr(lifecycle, "_persistence_backend", backend)

    class _CountingCache:
        def __init__(self):
            self.fetches_via_backend = 0
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value):
            self._store[key] = value

    counting = _CountingCache()
    engine = object.__new__(GraeaeEngine)
    engine.providers = {
        "openai": {"model": "gpt-5.5", "weight": 0.50},
        "claude": {"model": "claude-opus-4-8", "weight": 0.60},
    }
    engine._ranked_lineup_cache = counting

    await engine._ranked_lineup("reasoning", 2)
    first_calls = backend.consultations_audit.calls
    await engine._ranked_lineup("reasoning", 2)
    second_calls = backend.consultations_audit.calls

    assert first_calls == 1
    assert second_calls == 1, "empty cached result was re-fetched"
    # The cache key now carries a backend/pool source identity, so assert
    # the stored value by its (stable) candidate component rather than a
    # hardcoded 2-tuple.
    assert len(counting._store) == 1
    (key,) = counting._store.keys()
    candidates_component = key[-1]
    assert set(candidates_component) == {"openai", "claude"}
    assert counting._store[key] == []


# ── backend failure must fall through to the pool (not cache empty) ────────


async def test_backend_failure_falls_through_to_pool(monkeypatch, caplog):
    """CRITICAL: a backend that cannot serve the registry (returns None /
    raises) must NOT be cached as an empty registry and must NOT short-
    circuit the Postgres pool enrichment. The pool then serves the lineup
    and no warning is logged (the registry WAS reachable via the pool).

    Fails before the fix: the backend-failure branch set rows=[] and
    warned immediately, so `rows is not None` skipped the pool path and
    an empty registry was cached."""
    _reset_warn(monkeypatch)
    pool = _FakePool(
        rows=[
            {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
            {"provider": "anthropic", "model_id": "claude-opus-4-8", "graeae_weight": 0.40},
        ]
    )
    # Backend raises -> _registry_rows_via_backend returns None.
    monkeypatch.setattr(
        lifecycle, "_persistence_backend", _FakeBackend(exc=RuntimeError("backend down"))
    )
    monkeypatch.setattr(lifecycle, "_pool", pool)
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup = await engine._ranked_lineup("reasoning", 2)
    # The pool projection drove the lineup: openai (0.95) beats claude.
    assert list(lineup) == ["openai", "claude"]
    assert lineup["openai"] == "gpt-5.5"
    # The pool was actually consulted.
    assert len(pool.calls) == 1
    # No warning: the registry was reachable via the pool.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


# ── warning fires only when BOTH backend and pool fail ──────────────────────


async def test_warning_only_when_both_paths_fail(monkeypatch, caplog):
    """HIGH: the operator-visible warning must fire only when both the
    backend-neutral path and the pool enrichment are exhausted (both
    returned None). A backend failure with a healthy pool must NOT warn.

    Fails before the fix: the warning fired at the backend-failure branch
    before the pool was even attempted."""
    _reset_warn(monkeypatch)
    # Backend raises (None) AND pool raises (None): both exhausted.
    monkeypatch.setattr(
        lifecycle, "_persistence_backend", _FakeBackend(exc=RuntimeError("backend down"))
    )
    monkeypatch.setattr(lifecycle, "_pool", _FakePool(exc=RuntimeError("pool down")))
    engine = _make_engine()
    with caplog.at_level(logging.WARNING, logger="mnemos.domain.graeae.engine"):
        lineup1 = await engine._ranked_lineup("reasoning", 2)
        engine._ranked_lineup_cache = _FakeRankedLineupCache()  # force re-read
        lineup2 = await engine._ranked_lineup("reasoning", 2)
    # Unweighted static fallback order.
    assert list(lineup1) == ["claude", "openai"]
    assert lineup1 == lineup2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _REGISTRY_UNAVAILABLE in warnings[0].getMessage()


# ── cache key incorporates backend/pool identity (failover safety) ──────────


async def test_cache_key_includes_source_identity(monkeypatch):
    """MEDIUM: the cache key must incorporate the backend/pool identity so
    a runtime failover (backend swapped) does not serve a lineup cached
    from the old source. After swapping the backend, a fresh fetch occurs
    (different key) rather than a stale hit."""
    _reset_warn(monkeypatch)

    class _CountingCache:
        def __init__(self):
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value):
            self._store[key] = value

    engine = object.__new__(GraeaeEngine)
    engine.providers = {
        "openai": {"model": "gpt-5.5", "weight": 0.50},
        "claude": {"model": "claude-opus-4-8", "weight": 0.60},
    }
    counting = _CountingCache()
    engine._ranked_lineup_cache = counting

    backend_a = _FakeBackend(
        rows=[
            {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.95},
            {"provider": "anthropic", "model_id": "claude-opus-4-8"},
        ]
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", backend_a)
    await engine._ranked_lineup("reasoning", 2)
    calls_after_a = backend_a.consultations_audit.calls
    keys_after_a = set(counting._store.keys())
    assert calls_after_a == 1
    assert len(keys_after_a) == 1

    # Failover: a differently-named backend type. The old cache entry must
    # NOT be reused, so a fresh fetch happens under a new key.
    backend_b = _DistinctNameBackend(
        rows=[
            {"provider": "openai", "model_id": "gpt-5.5", "graeae_weight": 0.10},
            {"provider": "anthropic", "model_id": "claude-opus-4-8", "graeae_weight": 0.90},
        ]
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", backend_b)
    await engine._ranked_lineup("reasoning", 2)
    calls_after_b = backend_b.consultations_audit.calls
    assert calls_after_b == 1, "stale lineup from the old backend was served"
    assert len(counting._store.keys()) == 2, "failover did not create a new cache entry"
    # The two keys differ only in their source-identity component.
    old_key = next(iter(keys_after_a))
    new_keys = set(counting._store.keys()) - keys_after_a
    (new_key,) = new_keys
    assert old_key[-1] == new_key[-1], "candidate component must be unchanged"
    assert old_key[0] != new_key[0], "source identity component must differ after failover"
    # The new backend's weights (claude 0.90) now win.
    lineup = await engine._ranked_lineup("reasoning", 2)
    assert list(lineup) == ["claude", "openai"]
