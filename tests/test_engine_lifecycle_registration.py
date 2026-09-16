"""Singleton ownership registers cleanup without upward core imports."""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
async def test_unused_engine_shutdown_does_not_construct(monkeypatch):
    from mnemos.domain.graeae import engine

    factory = Mock(side_effect=AssertionError("shutdown constructed an engine"))
    monkeypatch.setattr(engine, "_graeae_engine", None)
    monkeypatch.setattr(engine, "GraeaeEngine", factory)
    await engine.close_graeae_engine_if_loaded()
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_singleton_registers_cleanup_without_api_bootstrap(monkeypatch):
    from mnemos.core import lifecycle
    from mnemos.domain.graeae import engine

    instance = Mock(close=AsyncMock())
    monkeypatch.setattr(engine, "_graeae_engine", None)
    monkeypatch.setattr(engine, "GraeaeEngine", Mock(return_value=instance))
    monkeypatch.setattr(lifecycle, "_lifespan_cleanup_hooks", {})
    assert engine.get_graeae_engine() is instance
    assert engine.get_graeae_engine() is instance
    assert list(lifecycle._lifespan_cleanup_hooks) == ["graeae engine"]
    await lifecycle._lifespan_cleanup_hooks["graeae engine"]()
    instance.close.assert_awaited_once()
