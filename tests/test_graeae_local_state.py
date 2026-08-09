from __future__ import annotations

from types import SimpleNamespace

from mnemos.domain.graeae._cache import ResponseCache, make_response_cache
from mnemos.domain.graeae._quality import QualityTracker, make_quality_tracker


def _settings():
    return SimpleNamespace(
        # This URI configures SlowAPI's HTTP limiter only. It must not opt
        # GRAEAE's response cache or quality tracker back into Redis.
        rate_limit=SimpleNamespace(storage_uri="redis://redis:6379/1"),
        resilience=SimpleNamespace(fallback_warning=False),
    )


def test_graeae_state_factories_ignore_slowapi_redis_storage():
    settings = _settings()

    assert isinstance(make_response_cache(settings), ResponseCache)
    assert isinstance(
        make_quality_tracker(settings, {"openai": 0.9}),
        QualityTracker,
    )
