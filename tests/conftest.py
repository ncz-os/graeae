"""Shared fixtures for the extracted GRAEAE package tests."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


class _InMemoryStateRepository:
    """Tiny in-memory ``StateRepository`` for tests.

    Mirrors the canonical ``StateRepository`` surface used by the GRAEAE
    quota path (``get``/``set`` inside an open transaction). The
    in-process dict is fine for sequential tests; the test that
    exercises concurrent same-principal calls uses its own concurrent
    helper backend.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], str] = {}

    async def get(self, tx, key, *, owner_id="default", namespace="default"):
        value = self.rows.get((owner_id, namespace, key))
        if value is None:
            return None
        return {"key": key, "value": value, "version": 1}

    async def set(self, tx, key, value, *, owner_id="default", namespace="default", expires_at=None):
        _ = expires_at
        self.rows[(owner_id, namespace, key)] = value
        return {"key": key, "value": value, "version": 2}


class _ConsultationsRepository:
    async def create_consultation_with_audit(self, _tx, **_values):
        return uuid4()


class _ConsultationsBackend:
    supports_webhooks = False

    def __init__(self) -> None:
        self.consultations = _ConsultationsRepository()
        self.state_kv = _InMemoryStateRepository()

    @asynccontextmanager
    async def transactional(self):
        yield SimpleNamespace(conn=None)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token-for-testing"}


@pytest.fixture
def mock_graeae_engine():
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
            "timestamp": "2026-08-10T08:00:00",
            "memory_ids": [],
        }
    )
    engine.providers = {
        "openai": {"model": "gpt-4o-mini"},
        "anthropic": {"model": "claude-3-5-sonnet"},
    }
    return engine


@pytest_asyncio.fixture
async def client(monkeypatch, mock_graeae_engine):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api.dependencies import configure_auth
    from mnemos.api.main import app
    from mnemos.api.routes import consultations
    from mnemos.domain.graeae import engine as graeae_engine

    configure_auth({"enabled": False})
    monkeypatch.setenv("MNEMOS_GRAEAE_AUDIT_HMAC_KEY", "test-audit-key")
    monkeypatch.setattr(lifecycle, "_persistence_backend", _ConsultationsBackend())
    monkeypatch.setattr(
        consultations, "_schedule_outbox_deliveries", lambda _delivery_ids: None
    )
    monkeypatch.setattr(graeae_engine, "get_graeae_engine", lambda: mock_graeae_engine)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


__all__ = ["_InMemoryStateRepository", "_ConsultationsBackend"]
