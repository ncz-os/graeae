from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from mnemos.api.routes import consultations
from mnemos.domain.graeae._cache import RedisResponseCache, make_response_cache
from mnemos.domain.graeae._quality import (
    RedisQualityTracker,
    _REDIS_QUALITY_EWMA_LUA,
    make_quality_tracker,
)


def _settings(storage_uri: str = "redis://redis:6379/1"):
    return SimpleNamespace(
        server=SimpleNamespace(profile="server", workers=1),
        rate_limit=SimpleNamespace(storage_uri=storage_uri),
        resilience=SimpleNamespace(
            fallback_warning=False,
            circuit_breaker_redis_prefix="test:cb:",
            rate_limiter_redis_prefix="test:rl:",
            concurrency_redis_prefix="test:conc:",
        ),
    )


@pytest.mark.asyncio
async def test_redis_response_cache_get_and_set_use_stable_key_shape():
    redis = SimpleNamespace(get=AsyncMock(), set=AsyncMock(), mget=AsyncMock())
    payload = {"openai": {"status": "success", "response_text": "ok"}}
    redis.get.return_value = json.dumps(payload)
    cache = RedisResponseCache(redis, ttl_seconds=123)

    result = await cache.get("Hello", "reasoning", "gpt-5")
    await cache.set("Hello", "reasoning", payload, "gpt-5")

    expected_digest = hashlib.sha256("hello:gpt-5:reasoning".encode()).hexdigest()
    expected_key = f"mnemos:graeae:cache:{expected_digest}"
    assert result == payload
    redis.get.assert_awaited_once_with(expected_key)
    redis.set.assert_awaited_once_with(
        expected_key,
        json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
        ex=123,
    )


@pytest.mark.asyncio
async def test_redis_response_cache_batch_lookup_uses_mget():
    redis = SimpleNamespace(
        get=AsyncMock(),
        set=AsyncMock(),
        mget=AsyncMock(return_value=[json.dumps({"a": 1}), None]),
    )
    cache = RedisResponseCache(redis)

    found = await cache.get_many(
        [
            ("p1", "reasoning", "m1"),
            ("p2", "reasoning", "m2"),
        ]
    )

    assert found == {("p1", "reasoning", "m1"): {"a": 1}}
    redis.mget.assert_awaited_once()
    assert all(
        key.startswith("mnemos:graeae:cache:") for key in redis.mget.await_args.args[0]
    )


@pytest.mark.asyncio
async def test_redis_quality_tracker_updates_provider_zset_with_lua():
    redis = SimpleNamespace(
        eval=AsyncMock(return_value=[0.8, 120.0]), zscore=AsyncMock(return_value=0.8)
    )
    tracker = RedisQualityTracker(redis, {"openai": 0.9})

    recorded_weight = await tracker.record_success("openai", 120)
    weight = await tracker.dynamic_weight("openai")

    args = redis.eval.await_args.args
    assert args[0] == _REDIS_QUALITY_EWMA_LUA
    assert args[1] == 1
    assert args[2] == "mnemos:graeae:quality:openai"
    assert args[3:] == (1.0, 120, 0.2, 0.9)
    assert "ZADD" in args[0]
    redis.zscore.assert_awaited_once_with(
        "mnemos:graeae:quality:openai", "success_ewma"
    )
    assert recorded_weight == 0.81
    assert weight == 0.81


def test_graeae_state_factories_select_redis_when_configured():
    redis = SimpleNamespace()
    settings = _settings()

    assert isinstance(
        make_response_cache(settings, redis_client=redis), RedisResponseCache
    )
    assert isinstance(
        make_quality_tracker(settings, {"openai": 0.9}, redis_client=redis),
        RedisQualityTracker,
    )


@pytest.mark.asyncio
async def test_consult_quota_uses_atomic_shared_redis_script(monkeypatch):
    redis = SimpleNamespace(eval=AsyncMock(return_value=[1, b"5", 0]))
    monkeypatch.setattr(
        consultations, "_get_lifecycle_redis_client", lambda: redis
    )
    monkeypatch.setenv("MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN", "20")

    await consultations._enforce_consult_quota("principal-1", "all")

    args = redis.eval.await_args.args
    assert args[0] == consultations._CONSULT_QUOTA_LUA
    assert args[1] == 2
    assert args[2].startswith("mnemos:graeae:consult-quota:")
    assert args[3] == f"{args[2]}:sequence"
    assert args[4:] == (5.0, 20.0, 60_000)

    redis.eval.return_value = [0, b"20", 12_500]
    with pytest.raises(HTTPException) as exc_info:
        await consultations._enforce_consult_quota("principal-1", "all")

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["Retry-After"] == "13"


@pytest.mark.parametrize("profile", ["edge", "dev", "personal"])
@pytest.mark.asyncio
async def test_single_worker_memory_profiles_use_atomic_local_quota(monkeypatch, profile):
    settings = _settings("memory://")
    settings.server.profile = profile
    monkeypatch.setattr(consultations, "get_settings", lambda: settings)
    monkeypatch.setattr(consultations, "_get_lifecycle_redis_client", lambda: None)
    monkeypatch.setenv("MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN", "5")
    consultations._consult_quota_local.clear()

    await consultations._enforce_consult_quota("local-principal", "all")
    with pytest.raises(HTTPException) as exc_info:
        await consultations._enforce_consult_quota("local-principal", "single")

    assert exc_info.value.status_code == 429


@pytest.mark.parametrize(
    ("profile", "workers"),
    [("server", 1), ("server", 4), ("edge", 2), ("dev", 2)],
)
@pytest.mark.asyncio
async def test_memory_quota_fails_closed_outside_single_worker_local_profiles(
    monkeypatch, profile, workers
):
    settings = _settings("memory://")
    settings.server = SimpleNamespace(profile=profile, workers=workers)
    monkeypatch.setattr(consultations, "get_settings", lambda: settings)
    monkeypatch.setattr(consultations, "_get_lifecycle_redis_client", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await consultations._enforce_consult_quota("principal", "single")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_route_owned_quota_client_is_closed_and_cleared():
    client = SimpleNamespace(aclose=AsyncMock())
    consultations._consult_quota_redis = client

    await consultations.close_consult_quota_client()

    client.aclose.assert_awaited_once()
    assert consultations._consult_quota_redis is None


@pytest.mark.asyncio
async def test_redis_state_closes_only_owned_clients():
    owned_cache_client = SimpleNamespace(aclose=AsyncMock())
    injected_cache_client = SimpleNamespace(aclose=AsyncMock())
    owned_quality_client = SimpleNamespace(aclose=AsyncMock())
    injected_quality_client = SimpleNamespace(aclose=AsyncMock())

    await RedisResponseCache(owned_cache_client, owns_client=True).close()
    await RedisResponseCache(injected_cache_client).close()
    await RedisQualityTracker(
        owned_quality_client, {"openai": 1.0}, owns_client=True
    ).close()
    await RedisQualityTracker(injected_quality_client, {"openai": 1.0}).close()

    owned_cache_client.aclose.assert_awaited_once()
    owned_quality_client.aclose.assert_awaited_once()
    injected_cache_client.aclose.assert_not_awaited()
    injected_quality_client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_close_releases_all_resources_and_clears_singleton(monkeypatch):
    from mnemos.domain.graeae import engine as engine_module

    instance = engine_module.GraeaeEngine.__new__(engine_module.GraeaeEngine)
    instance._closed = False
    instance._client = None
    resources = [SimpleNamespace(close=AsyncMock()) for _ in range(5)]
    (
        instance._cache,
        instance._quality,
        instance._circuit_breakers,
        instance._rate_limiters,
        instance._concurrency,
    ) = resources
    engine_module._graeae_engine = instance
    close_quota = AsyncMock()
    monkeypatch.setattr(consultations, "close_consult_quota_client", close_quota)

    await instance.close()

    for resource in resources:
        resource.close.assert_awaited_once()
    close_quota.assert_awaited_once()
    assert engine_module._graeae_engine is None
