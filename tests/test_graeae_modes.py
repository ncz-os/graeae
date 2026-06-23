from __future__ import annotations

from typing import Callable

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from mnemos.domain.models import ConsultationRequest, SUPPORTED_CONSULTATION_MODES
from mnemos.domain.graeae.provider_worker import ProviderQueryResponse


class _AllowAll:
    def __init__(self):
        self.successes: list[str] = []
        self.failures: list[str] = []

    def is_allowed(self, name: str) -> bool:
        return True

    def record_success(self, name: str) -> None:
        self.successes.append(name)

    def record_failure(self, name: str) -> None:
        self.failures.append(name)

    def status(self) -> dict:
        return {}


class _FakeConcurrency:
    def __init__(self):
        self.acquired: list[str] = []
        self.released: list[str] = []

    def acquire(self, name: str) -> bool:
        self.acquired.append(name)
        return True

    def release(self, name: str) -> None:
        self.released.append(name)

    def status(self) -> dict:
        return {}


class _FakeQuality:
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def record_success(self, name: str, latency: int) -> None:
        return None

    def record_failure(self, name: str) -> None:
        return None

    def dynamic_weight(self, name: str) -> float:
        return self.weights[name]

    def status(self) -> dict:
        return {}


def _provider_config(
    model: str,
    weight: float,
    url: str = "https://example.invalid/v1/chat/completions",
    scope: str | None = None,
) -> dict:
    cfg = {
        "url": url,
        "model": model,
        "weight": weight,
        "api": "openai",
        "key_name": model,
    }
    if scope is not None:
        cfg["scope"] = scope
    return cfg


def _build_engine(
    monkeypatch,
    responses: dict[str, str] | Callable[[str, str], str] | None = None,
    providers: dict[str, dict] | None = None,
):
    import mnemos.core.lifecycle as lc
    import mnemos.domain.graeae.engine as graeae_engine
    from mnemos.domain.graeae.engine import GraeaeEngine

    monkeypatch.setattr(lc, "_pool", None)
    monkeypatch.setattr(graeae_engine, "get_elo_weights", lambda force_refresh=False: None)
    engine = GraeaeEngine()
    engine.providers = providers if providers is not None else {
        "alpha": _provider_config("alpha-model", 0.95),
        "beta": _provider_config("beta-model", 0.85),
        "gamma": _provider_config("gamma-model", 0.75),
        "delta": _provider_config("delta-model", 0.65),
    }
    engine._circuit_breakers = _AllowAll()
    engine._rate_limiters = _AllowAll()
    engine._concurrency = _FakeConcurrency()
    engine._quality = _FakeQuality({
        name: cfg["weight"] for name, cfg in engine.providers.items()
    })
    calls: list[dict] = []

    async def _fake_provider_worker(request):
        provider_name = request.provider
        prompt = request.params["prompt"]
        calls.append({
            "provider": provider_name,
            "prompt": prompt,
            "model_override": request.model,
        })
        if callable(responses):
            text = responses(provider_name, prompt)
        elif responses is not None:
            text = responses.get(provider_name, f"{provider_name} response")
        else:
            text = f"{provider_name} response to {prompt}"
        payload = {
            "status": "success",
            "response_text": text,
            "latency_ms": 10,
            "model_id": request.model or engine.providers[provider_name]["model"],
            "final_score": engine.providers[provider_name]["weight"],
        }
        return ProviderQueryResponse(
            response_text=payload["response_text"],
            latency_ms=payload["latency_ms"],
            status=payload["status"],
            cost=payload.get("cost"),
            model_id_used=payload["model_id"],
            raw_provider_payload=payload,
        )

    engine.provider_worker = _fake_provider_worker
    return engine, calls


def test_consultation_request_defaults_mode_to_auto():
    request = ConsultationRequest(prompt="short", task_type="architecture_design")
    assert request.mode == "auto"


@pytest.mark.asyncio
async def test_auto_short_architecture_design_returns_body(
    client: AsyncClient,
    auth_headers: dict,
    mock_graeae_engine,
):
    mock_graeae_engine.consult.reset_mock()
    resp = await client.post(
        "/v1/consultations",
        json={
            "prompt": "short",
            "task_type": "architecture_design",
            "mode": "auto",
        },
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    assert resp.content
    data = resp.json()
    assert data["mode"] == "auto"
    assert data["all_responses"]
    assert mock_graeae_engine.consult.await_args.kwargs["mode"] == "auto"


@pytest.mark.asyncio
async def test_unspecified_mode_defaults_to_auto(
    client: AsyncClient,
    auth_headers: dict,
    mock_graeae_engine,
):
    mock_graeae_engine.consult.reset_mock()
    resp = await client.post(
        "/v1/consultations",
        json={"prompt": "short", "task_type": "architecture_design"},
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["mode"] == "auto"
    assert mock_graeae_engine.consult.await_args.kwargs["mode"] == "auto"


@pytest.mark.asyncio
async def test_unknown_mode_rejected_by_pydantic(
    client: AsyncClient,
    auth_headers: dict,
):
    resp = await client.post(
        "/v1/consultations",
        json={
            "prompt": "short",
            "task_type": "architecture_design",
            "mode": "consensus",
        },
        headers=auth_headers,
    )

    assert resp.status_code == 422
    assert "consensus" in resp.text


def test_unknown_mode_rejected_in_request_model():
    with pytest.raises(ValidationError):
        ConsultationRequest(
            prompt="short",
            task_type="architecture_design",
            mode="consensus",
        )


@pytest.mark.asyncio
async def test_empty_engine_result_returns_502(
    client: AsyncClient,
    auth_headers: dict,
    mock_graeae_engine,
):
    mock_graeae_engine.consult.return_value = {}
    resp = await client.post(
        "/v1/consultations",
        json={"prompt": "short", "task_type": "architecture_design"},
        headers=auth_headers,
    )

    assert resp.status_code == 502
    assert "empty result" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_modes_endpoint_lists_all_seven_modes(
    client: AsyncClient,
    auth_headers: dict,
):
    resp = await client.get("/v1/consultations/modes", headers=auth_headers)

    assert resp.status_code == 200, resp.text
    names = {mode["name"] for mode in resp.json()["modes"]}
    assert names == set(SUPPORTED_CONSULTATION_MODES)
    assert resp.json()["validation"]["unknown_mode_status"] == 422


@pytest.mark.parametrize("mode", ["auto", "local", "external", "all"])
@pytest.mark.asyncio
async def test_existing_routing_strategy_modes_still_work(
    client: AsyncClient,
    auth_headers: dict,
    mock_graeae_engine,
    mode: str,
):
    mock_graeae_engine.consult.reset_mock()
    resp = await client.post(
        "/v1/consultations",
        json={"prompt": "route it", "task_type": "reasoning", "mode": mode},
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["mode"] == mode
    assert mock_graeae_engine.consult.await_args.kwargs["mode"] == mode


@pytest.mark.asyncio
async def test_single_mode_calls_exactly_one_muse(monkeypatch):
    engine, calls = _build_engine(monkeypatch)

    result = await engine.consult("fast check", "reasoning", mode="single")

    assert [call["provider"] for call in calls] == ["alpha"]
    assert list(result["all_responses"]) == ["alpha"]
    assert result["winning_muse"] == "alpha"


@pytest.mark.asyncio
async def test_debate_mode_runs_three_muses_for_two_rounds(monkeypatch):
    engine, calls = _build_engine(monkeypatch)

    result = await engine.consult("design the cache", "architecture_design", mode="debate")

    assert len(calls) == 6
    assert [call["provider"] for call in calls[:3]] == ["alpha", "beta", "gamma"]
    assert [call["provider"] for call in calls[3:]] == ["alpha", "beta", "gamma"]
    assert all(call["prompt"] == "design the cache" for call in calls[:3])
    assert all("Round 1 responses from the other muses" in call["prompt"] for call in calls[3:])
    assert set(result["round_1"]) == {"alpha", "beta", "gamma"}
    assert set(result["round_2"]) == {"alpha", "beta", "gamma"}
    assert set(result["all_responses"]) == {"alpha", "beta", "gamma"}


@pytest.mark.asyncio
async def test_majority_mode_reports_quorum_reached(monkeypatch):
    engine, calls = _build_engine(
        monkeypatch,
        responses={
            "alpha": "approve the design because it has a stable cache boundary",
            "beta": "approve the design because it has a stable cache boundary",
            "gamma": "approve the design because it has a stable cache boundary",
        },
    )

    result = await engine.consult("approve?", "architecture_design", mode="majority")

    assert [call["provider"] for call in calls] == ["alpha", "beta", "gamma"]
    assert result["quorum_reached"] is True
    assert result["consensus_score"] >= result["quorum_threshold"]


@pytest.mark.asyncio
async def test_majority_mode_reports_quorum_not_reached(monkeypatch):
    engine, _calls = _build_engine(
        monkeypatch,
        responses={
            "alpha": "approve caching with a write-through database layer",
            "beta": "reject the change and replace the API gateway first",
            "gamma": "defer the decision until hardware telemetry arrives",
        },
    )

    result = await engine.consult("approve?", "architecture_design", mode="majority")

    assert result["quorum_reached"] is False
    assert result["consensus_score"] < result["quorum_threshold"]


# ── Scope filter: mode=local|external actually narrows the muse lineup ────────
# Regression for the defect where `mode` was only a quorum label, so an
# `external` consult still queried local-GPU muses (and vice versa).

def _scoped_providers() -> dict[str, dict]:
    return {
        # public cloud FQDNs -> external by host inference
        "cloud_openai": _provider_config(
            "gpt", 0.90, url="https://api.openai.com/v1/chat/completions"),
        "cloud_xai": _provider_config(
            "grok", 0.86, url="https://api.x.ai/v1/chat/completions"),
        # LAN GPU servers (RFC-1918) -> local by host inference
        "lan_cerberus": _provider_config(
            "gemma", 0.70, url="http://192.168.207.96:8080/v1/chat/completions"),
        "lan_hydra": _provider_config(
            "qwen", 0.72, url="http://192.168.207.78:8080/v1/chat/completions"),
        # OAuth shim on loopback, but proxies a CLOUD model -> explicit scope wins
        "shim_openai": _provider_config(
            "gpt-oauth", 0.88,
            url="http://127.0.0.1:5079/openai/v1/chat/completions",
            scope="external"),
    }


def test_provider_scope_classification():
    from mnemos.domain.graeae.engine import _provider_scope

    p = _scoped_providers()
    assert _provider_scope(p["cloud_openai"]) == "external"
    assert _provider_scope(p["cloud_xai"]) == "external"
    assert _provider_scope(p["lan_cerberus"]) == "local"
    assert _provider_scope(p["lan_hydra"]) == "local"
    # explicit scope overrides the loopback-host inference
    assert _provider_scope(p["shim_openai"]) == "external"
    # bare hostname -> local; localhost -> local; unparseable -> external
    assert _provider_scope({"url": "http://cerberus:8080/v1"}) == "local"
    assert _provider_scope({"url": "http://localhost:5079/x"}) == "local"
    assert _provider_scope({"url": ""}) == "external"


@pytest.mark.asyncio
async def test_external_mode_queries_only_external_muses(monkeypatch):
    engine, calls = _build_engine(monkeypatch, providers=_scoped_providers())

    result = await engine.consult("frontier only", "reasoning", mode="external")

    queried = {call["provider"] for call in calls}
    assert queried == {"cloud_openai", "cloud_xai", "shim_openai"}
    assert set(result["all_responses"]) == {"cloud_openai", "cloud_xai", "shim_openai"}
    assert "lan_cerberus" not in result["all_responses"]
    assert "lan_hydra" not in result["all_responses"]


@pytest.mark.asyncio
async def test_local_mode_queries_only_local_muses(monkeypatch):
    engine, calls = _build_engine(monkeypatch, providers=_scoped_providers())

    result = await engine.consult("on-prem only", "reasoning", mode="local")

    queried = {call["provider"] for call in calls}
    assert queried == {"lan_cerberus", "lan_hydra"}
    assert set(result["all_responses"]) == {"lan_cerberus", "lan_hydra"}


@pytest.mark.asyncio
async def test_all_mode_queries_every_muse_regardless_of_scope(monkeypatch):
    engine, _calls = _build_engine(monkeypatch, providers=_scoped_providers())

    result = await engine.consult("everyone", "reasoning", mode="all")

    # `all` keeps the full lineup; cancelled muses still appear as keys.
    assert set(result["all_responses"]) == set(_scoped_providers())


@pytest.mark.asyncio
async def test_external_mode_with_no_external_muses_errors(monkeypatch):
    local_only = {
        "lan_cerberus": _provider_config(
            "gemma", 0.70, url="http://192.168.207.96:8080/v1/chat/completions"),
        "lan_hydra": _provider_config(
            "qwen", 0.72, url="http://192.168.207.78:8080/v1/chat/completions"),
    }
    engine, calls = _build_engine(monkeypatch, providers=local_only)

    result = await engine.consult("frontier only", "reasoning", mode="external")

    assert calls == []
    assert result["all_responses"] == {}
    assert "no external providers" in result["error"]
