from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mnemos.domain.graeae import provider_sync


pytestmark = pytest.mark.asyncio


async def test_openai_fetch_failure_is_reported(monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("401 Unauthorized"))
    )
    client = SimpleNamespace(get=AsyncMock(return_value=response))

    class _ClientContext:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(provider_sync.httpx, "AsyncClient", lambda **_kwargs: _ClientContext())
    monkeypatch.setattr(provider_sync, "_load_key", lambda _provider: "expired")

    summary = await provider_sync.sync_provider(None, "openai")

    assert summary["models_found"] == 0
    assert "401 Unauthorized" in summary["error"]


async def test_provider_sync_uses_canonical_key_resolver(monkeypatch):
    monkeypatch.setattr(provider_sync, "get_key", lambda provider: f"key-for-{provider}")

    assert provider_sync._load_key("openai") == "key-for-openai"
    assert provider_sync._load_key("together_ai") == "key-for-together_ai"


async def test_missing_live_provider_key_is_reported_as_error(monkeypatch):
    monkeypatch.setattr(provider_sync, "get_key", lambda _provider: None)

    summary = await provider_sync.sync_provider(None, "openai")

    assert summary["models_found"] == 0
    assert "requires a configured API key" in summary["error"]


async def test_gemini_partial_page_failure_does_not_mutate_registry(monkeypatch):
    first = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "models": [
                {
                    "name": "models/gemini-page-one",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "page-two",
        },
    )
    second = SimpleNamespace(
        raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("timeout"))
    )
    client = SimpleNamespace(get=AsyncMock(side_effect=[first, second]))

    class _ClientContext:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return False

    repo = SimpleNamespace(
        upsert_model=AsyncMock(),
        mark_models_unavailable=AsyncMock(),
        write_model_sync_log=AsyncMock(),
    )

    class _Transaction:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    backend = SimpleNamespace(
        consultations_audit=repo,
        transactional=lambda: _Transaction(),
    )
    monkeypatch.setattr(provider_sync.httpx, "AsyncClient", lambda **_kwargs: _ClientContext())
    monkeypatch.setattr(provider_sync, "_load_key", lambda _provider: "key")

    summary = await provider_sync.sync_provider(backend, "gemini")

    assert "timeout" in summary["error"]
    repo.upsert_model.assert_not_awaited()
    repo.mark_models_unavailable.assert_not_awaited()
