"""Shared fixtures extracted with the GRAEAE route tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


class _Transaction:
    async def __aenter__(self):
        return MagicMock(name="transaction")

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Consultations:
    def __init__(self):
        self.create_consultation_with_audit = AsyncMock(
            return_value="test-consultation-id"
        )


class _PersistenceBackend:
    """Small route-level backend; core's full database fake is not extracted."""

    supports_webhooks = False

    def __init__(self):
        self.consultations = _Consultations()

    def transactional(self):
        return _Transaction()


@pytest.fixture
def auth_headers():
    """Return the same headers as the original monorepo fixture."""
    return {"Authorization": "Bearer test-token-for-testing"}


@pytest.fixture
def mock_graeae_engine():
    """Return the route test double used before repository extraction."""
    engine = MagicMock()
    engine.consult = AsyncMock(
        return_value={
            "all_responses": {
                "openai": {
                    "response_text": "Paris is the capital of France.",
                    "final_score": 0.95,
                    "latency_ms": 1200,
                    "status": "success",
                    "model_id": "gpt-4o-mini",
                }
            },
            "consensus_response": "Paris is the capital of France.",
            "consensus_score": 0.95,
            "winning_muse": "openai",
            "cost": 0.02,
            "latency_ms": 1200,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "memory_ids": ["mem_seed_001"],
        }
    )
    engine.provider_status = MagicMock(return_value={"providers": {}, "cache": {}})
    engine.providers = {"openai": {"model": "gpt-4o-mini"}}
    return engine


@pytest_asyncio.fixture
async def client(monkeypatch, mock_graeae_engine):
    """Create an in-process client without importing core's test-only fakes."""
    import mnemos.api.routes.consultations as consultation_routes
    import mnemos.core.lifecycle as lifecycle
    import mnemos.domain.graeae.engine as graeae_engine
    from mnemos.api.dependencies import configure_auth
    from mnemos.api.main import app

    configure_auth({"enabled": False})
    monkeypatch.setenv("MNEMOS_GRAEAE_AUDIT_HMAC_KEY", "graeae-route-test-key")
    monkeypatch.setattr(lifecycle, "_pool", None)
    monkeypatch.setattr(lifecycle, "_persistence_backend", _PersistenceBackend())
    monkeypatch.setattr(lifecycle, "_cache", None)
    monkeypatch.setattr(lifecycle, "_worker_status", {"distillation_worker": "idle"})
    monkeypatch.setattr(
        consultation_routes,
        "require_consultations_backend",
        lambda: lifecycle._persistence_backend,
    )
    monkeypatch.setattr(graeae_engine, "get_graeae_engine", lambda: mock_graeae_engine)
    app.state.pool = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
