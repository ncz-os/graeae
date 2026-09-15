from __future__ import annotations

import pytest
from mnemos.domain.graeae.engine import GraeaeEngine
from mnemos.domain.graeae.provider_worker import ProviderQueryResponse

pytestmark = pytest.mark.asyncio


async def test_provider_worker_params_are_bounded_before_dispatch(monkeypatch):
    monkeypatch.setenv("MNEMOS_GRAEAE_MAX_OUTPUT_TOKENS", "128")
    monkeypatch.setenv("MNEMOS_GRAEAE_MAX_CANDIDATES", "2")

    engine = GraeaeEngine.__new__(GraeaeEngine)
    captured = {}

    async def worker(request):
        captured["params"] = request.params
        return ProviderQueryResponse(
            response_text="ok",
            latency_ms=1,
            status="success",
            cost=None,
            model_id_used="model-a",
            raw_provider_payload={"status": "success"},
        )

    engine.provider_worker = worker
    engine.providers = {"openai": {"model": "model-a"}}

    await engine._call_provider_worker(
        "openai",
        "prompt",
        "reasoning",
        180,
        generation_params={
            "max_tokens": 1_000_000,
            "temperature": -7,
            "top_p": 99,
        },
        request_params={
            "n": 99,
            "presence_penalty": 7,
            "frequency_penalty": -7,
            "parallel_tool_calls": True,
            "stop": ["x" * 600, "y", "z", "a", "b"],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    assert captured["params"]["generation_params"] == {
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert captured["params"]["request_params"]["n"] == 2
    assert captured["params"]["request_params"]["presence_penalty"] == 2.0
    assert captured["params"]["request_params"]["frequency_penalty"] == -2.0
    assert captured["params"]["request_params"]["parallel_tool_calls"] is True
    assert len(captured["params"]["request_params"]["stop"]) == 4
    assert len(captured["params"]["request_params"]["stop"][0]) == 512
    assert captured["params"]["request_params"]["tools"] == [
        {"type": "function", "function": {"name": "lookup"}}
    ]
