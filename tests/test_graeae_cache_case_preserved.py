"""F10 regression — GRAEAE response cache key is case-preserving.

The previous ``ResponseCache._key`` (and ``RedisResponseCache._key``)
normalized the prompt with ``.strip().lower()`` before hashing, which
made distinct requests whose only difference was letter-case share a
cached entry. The classical collision is ``"Return A"`` vs ``"Return a"``
— semantically different requests that should each get their own
cached response, not the first cached one's answer handed back to
both callers.

The fix hashes the EXACT framed content (case-preserved, only
leading/trailing whitespace stripped) and includes ``task_type`` and
``model`` with a structural separator so the key is not ambiguous
between a request of "task=foo, prompt=bar" and "task=foob, prompt=ar".

The fix must NOT introduce a general cache refactor; this test
specifically covers only the case-sensitivity contract.
"""

from __future__ import annotations

import pytest

from mnemos.domain.graeae import _cache


@pytest.fixture
def cache() -> _cache.ResponseCache:
    return _cache.ResponseCache(ttl_seconds=3600, max_entries=8, max_bytes=1 << 20)


def test_distinct_case_prompts_get_distinct_keys(cache: _cache.ResponseCache) -> None:
    """``"Return A"`` vs ``"Return a"`` must produce different cache keys."""
    upper = cache._key("Return A", "task_type_x", "model_x")
    lower = cache._key("Return a", "task_type_x", "model_x")
    assert upper != lower, (
        f"cache key must be case-sensitive — got identical hashes for "
        f"distinct-letter-case prompts: {upper!r} == {lower!r}"
    )


def test_same_case_prompt_gets_same_key(cache: _cache.ResponseCache) -> None:
    """The exact same text must hash to the same key (cache hits work)."""
    a1 = cache._key("hello world", "t", "")
    a2 = cache._key("hello world", "t", "")
    assert a1 == a2, "stable hashing broken: same input, different key"


def test_leading_and_trailing_whitespace_normalized_but_case_preserved(
    cache: _cache.ResponseCache,
) -> None:
    """Edge whitespace around an otherwise identical prompt must collide
    (cache hit for "   foo   " should hit "foo"); but mixed case
    continues to differ."""
    base = cache._key("Return A", "t", "")
    padded = cache._key("  Return A  ", "t", "")
    assert base == padded, (
        "leading/trailing whitespace must normalize for cache hits"
    )
    different_case = cache._key("Return a", "t", "")
    assert base != different_case, (
        "case differences must still NOT collide — edge-whitespace "
        "stripping must not silently lower() the prompt again"
    )


def test_uppercase_and_lowercase_dont_share_entry(cache: _cache.ResponseCache) -> None:
    """End-to-end cache probe: a value cached under ``"Return A"`` must
    NOT be returned for a ``get("Return a", ...)`` call."""
    cache.set("Return A", "consult", {"answer": "value-A"}, "")
    miss = cache.get("Return a", "consult", "")
    assert miss is None, (
        "case-different lookup returned a cached entry — the prior "
        "lowercased-key bug would have returned the 'Return A' cache value"
    )
    # And the original case still hits.
    hit = cache.get("Return A", "consult", "")
    assert hit == {"answer": "value-A"}


def test_model_field_partitions_cache(cache: _cache.ResponseCache) -> None:
    """Same prompt, different model → different key.

    Different models can answer a prompt differently (temperature drift,
    system prompt differences, alignment tuning) so the cache key must
    isolate them to avoid returning one model's answer to a request
    targeted at another. The previous omission of response-affecting
    config in the prior key schema was the F10-caller's flagged
    concern; this test pins that model participates in the key."""
    base = cache._key("hello", "t", "model-a")
    other = cache._key("hello", "t", "model-b")
    assert base != other, (
        "model must participate in the cache key — same prompt, different "
        "model must NOT share a cache entry"
    )


def test_task_type_field_partitions_cache(cache: _cache.ResponseCache) -> None:
    """Same prompt, different task_type → different key."""
    base = cache._key("hello", "task-type-a", "")
    other = cache._key("hello", "task-type-b", "")
    assert base != other, (
        "task_type must participate in the cache key"
    )


def test_redis_cache_matches_in_process_case_policy() -> None:
    """The Redis-backed cache must hash the same logical inputs to the
    same digest as the in-process cache. Otherwise a deployment that
    hits Redis for some calls and the in-process fallback for others
    would miss one cache's entries while hitting the other's."""

    class _StubRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

    redis = _StubRedis()
    in_process = _cache.ResponseCache()
    redis_cache = _cache.RedisResponseCache(redis, ttl_seconds=3600)

    # Same logical inputs must produce the same key (after the
    # key_prefix difference for redis).
    in_proc_key = in_process._key("hello world", "consult", "model-x")
    redis_digest = redis_cache._key("hello world", "consult", "model-x")
    assert redis_digest.endswith(in_proc_key), (
        "Redis cache must hash the same logical inputs to the same digest "
        f"(in-process={in_proc_key!r}, redis={redis_digest!r})"
    )


@pytest.mark.asyncio
async def test_redis_cache_distinct_case_distinct_keys() -> None:
    """Redis-cache variant: ``"Return A"`` and ``"Return a"`` must also
    produce different cache keys (the in-process fix must propagate to
    the Redis variant)."""

    class _StubRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

    redis = _StubRedis()
    redis_cache = _cache.RedisResponseCache(redis, ttl_seconds=3600)

    a = redis_cache._key("Return A", "consult", "")
    b = redis_cache._key("Return a", "consult", "")
    assert a != b, (
        f"Redis cache key must be case-sensitive — got identical hashes for "
        f"distinct-letter-case prompts: {a!r} == {b!r}"
    )
