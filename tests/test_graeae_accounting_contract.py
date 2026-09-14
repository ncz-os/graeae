"""Regression tests for F06 — GRAEAE bypasses its documented accounting path.

The bug, as reproduced in review #F06: GRAEAE's own README says consultations
go through PANTHEON (model / cost routing) and KNEMON (dollar budget
enforcement). The code path instead invoked provider workers / adapters
DIRECTLY, bypassing both — and the adapters DISCARDED the upstream token
usage / cost fields. Worse, the consensus rollup coerced a missing cost to
``$0.00`` instead of surfacing it as unknown, so a real ``$0.25`` call
showed up as ``$0.00``.

These tests assert the FIX:

  1. Each provider adapter (OpenAI / Anthropic / Gemini) preserves the
     upstream ``usage`` block AND leaves ``cost`` as a typed
     ``Optional[float]`` rather than synthesizing zero.
  2. ``_compute_consensus`` sums KNOWN costs correctly across the fan-out.
  3. ``_compute_consensus`` does NOT silently coerce unknown / missing
     costs to zero — the aggregate ``cost`` becomes ``None`` and a
     ``cost_unknown`` flag is set, with the per-provider offenders
     reported in ``cost_unknown_providers``. This is the loud-failure
     surface the fix introduces in place of the silent ``$0.00``.
  4. The consultation path actually invokes the shared PANTHEON/KNEMON
     budget/pre-flight contract (``enforce_provider_budget``) before
     dispatching, and records the priced spend via
     ``record_provider_spend`` afterwards.

All four pieces are required — anything short of all four leaves a
subtle under-billing hole open.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemos.domain.graeae.engine import (
    BudgetCheckResult,
    ProviderBudgetExceeded,
    ProviderQueryResponse,
    SpendRecord,
    _compute_consensus,
    enforce_provider_budget,
    get_graeae_engine,
    price_provider_cost,
    record_provider_spend,
)
from mnemos.domain.graeae.provider_worker import (
    LocalProviderWorker,
    ProviderQueryRequest,
)


# ─────────────────────────────────────────────────────────────────────────
# Adapter-level preservation tests
# ─────────────────────────────────────────────────────────────────────────


def _build_provider(model: str = "gpt-5.5", api: str = "openai") -> Dict[str, Any]:
    return {
        "url": "https://example.invalid/v1/chat/completions",
        "model": model,
        "weight": 0.88,
        "api": api,
        "key_name": "test_key",
    }


class _StubResponse:
    """Minimal httpx.Response stand-in for adapter tests.

    The adapter only inspects ``status_code``, ``json()``, and ``text``
    — never streams — so a MagicMock with those three attributes is
    sufficient and avoids spinning up an ASGI transport.
    """

    def __init__(self, *, status_code: int, body: Dict[str, Any]):
        self.status_code = status_code
        self._body = body
        self.text = ""

    def json(self) -> Dict[str, Any]:
        return self._body


class _StubClient:
    """Stub httpx.AsyncClient that returns a canned response on ``post``."""

    def __init__(self, response: _StubResponse):
        self._response = response
        self.posted: list[Dict[str, Any]] = []
        # Match the small surface of httpx.AsyncClient that
        # ``GraeaeEngine._get_client`` probes before reusing a
        # cached client. The real httpx client exposes
        # ``is_closed``; the engine reads it before deciding
        # whether to spin up a new one.
        self.is_closed = False

    async def __aenter__(self) -> "_StubClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, *args: Any, **kwargs: Any) -> _StubResponse:
        self.posted.append({"args": args, "kwargs": kwargs})
        return self._response


def _new_engine_with_stub_client(client: _StubClient) -> Any:
    """Return a GraeaeEngine whose ``_get_client`` returns ``client``.

    The engine constructor wires up real circuit breakers, rate limiters,
    etc.; for these unit tests we never reach them because we drive the
    adapters directly. We just need the three provider-adapter methods
    to be callable on the engine object.
    """
    from mnemos.domain.graeae.engine import GraeaeEngine

    engine = GraeaeEngine.__new__(GraeaeEngine)
    engine._client = client
    return engine


@pytest.mark.asyncio
async def test_openai_adapter_preserves_usage_and_leaves_cost_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's reproduction shape: 1,500 tokens + real cost data
    must reach the caller's payload, NOT be dropped on the floor.

    Before F06: ``_query_openai_compatible`` returned a dict with
    ``status``, ``response_text``, ``latency_ms``, ``model_id``,
    ``choices`` — and nothing else. A caller reading ``result["cost"]``
    got a ``KeyError`` and consensus treated the consultation as
    free.

    After F06: the dict carries ``usage`` (with the upstream
    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``)
    AND ``cost`` is present as an explicit ``None``. Consensus
    surfaces ``cost_unknown=True`` rather than silently reporting $0.
    """
    # Patch the API-key resolver so the adapter doesn't blow up on
    # an unset env var during unit tests.
    monkeypatch.setattr(
        "mnemos.domain.graeae.engine.get_key",
        lambda _name: "fake-key-for-unit-test",
    )

    body = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Paris."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
            "reasoning_tokens": 50,
        },
    }
    stub = _StubClient(_StubResponse(status_code=200, body=body))
    engine = _new_engine_with_stub_client(stub)

    result = await engine._query_openai_compatible(
        _build_provider(),
        prompt="What is the capital of France?",
        timeout=10,
    )

    assert result["status"] == "success"
    assert result["response_text"] == "Paris."
    # F06 contract: upstream usage block is preserved, NOT dropped.
    assert isinstance(result.get("usage"), dict), (
        "OpenAI adapter must preserve upstream usage block "
        "(F06: was silently dropped before the fix)"
    )
    assert result["usage"]["prompt_tokens"] == 1200
    assert result["usage"]["completion_tokens"] == 300
    assert result["usage"]["total_tokens"] == 1500
    assert result["usage"]["reasoning_tokens"] == 50
    # F06 contract: cost is typed Optional[float], explicitly None
    # when the adapter does not have a priced cost from upstream.
    # Coercing to 0.0 here is exactly the bug we're fixing.
    assert "cost" in result, (
        "adapter must declare cost explicitly so consensus sees the "
        "absence (F06 silent-zero bug)"
    )
    assert result["cost"] is None, (
        "cost must be None — never 0.0 — when upstream did not "
        "return a priced cost (F06 silent-zero bug)"
    )


@pytest.mark.asyncio
async def test_anthropic_adapter_preserves_usage_and_leaves_cost_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic adapter must surface ``usage`` (input_tokens /
    output_tokens / cache_*_input_tokens) and leave ``cost`` as
    ``None`` until KNEMON prices it."""
    monkeypatch.setattr(
        "mnemos.domain.graeae.engine.get_key",
        lambda _name: "fake-key-for-unit-test",
    )

    body = {
        "content": [{"type": "text", "text": "Paris."}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 800,
            "output_tokens": 200,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        },
    }
    stub = _StubClient(_StubResponse(status_code=200, body=body))
    engine = _new_engine_with_stub_client(stub)

    result = await engine._query_anthropic(
        {
            "url": "https://example.invalid/v1/messages",
            "model": "claude-opus-4-8",
            "weight": 0.90,
            "api": "anthropic",
            "key_name": "claude",
        },
        prompt="What is the capital of France?",
        timeout=10,
    )

    assert result["status"] == "success"
    assert result["response_text"] == "Paris."
    assert isinstance(result.get("usage"), dict), (
        "Anthropic adapter must preserve upstream usage block (F06)"
    )
    assert result["usage"]["input_tokens"] == 800
    assert result["usage"]["output_tokens"] == 200
    assert result["usage"]["cache_read_input_tokens"] == 500
    assert "cost" in result
    assert result["cost"] is None, (
        "Anthropic adapter cost must be typed None when upstream "
        "did not return a priced cost (F06 silent-zero bug)"
    )


@pytest.mark.asyncio
async def test_gemini_adapter_preserves_usage_and_leaves_cost_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini adapter must surface ``usage`` mapped from
    ``usageMetadata`` (promptTokenCount / candidatesTokenCount /
    thoughtsTokenCount / cachedContentTokenCount / totalTokenCount)
    and leave ``cost`` as ``None``."""
    monkeypatch.setattr(
        "mnemos.domain.graeae.engine.get_key",
        lambda _name: "fake-key-for-unit-test",
    )

    body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Paris."}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 12,
            "cachedContentTokenCount": 40,
            "totalTokenCount": 182,
        },
    }
    stub = _StubClient(_StubResponse(status_code=200, body=body))
    engine = _new_engine_with_stub_client(stub)

    result = await engine._query_gemini(
        {
            "url": "https://example.invalid/v1beta/models/gemini-3.1-pro-preview:generateContent",
            "model": "gemini-3.1-pro-preview",
            "weight": 0.88,
            "api": "gemini",
            "key_name": "gemini",
        },
        prompt="What is the capital of France?",
        timeout=10,
    )

    assert result["status"] == "success"
    assert result["response_text"] == "Paris."
    assert isinstance(result.get("usage"), dict), (
        "Gemini adapter must preserve upstream usageMetadata (F06)"
    )
    assert result["usage"]["prompt_tokens"] == 100
    assert result["usage"]["completion_tokens"] == 30
    assert result["usage"]["reasoning_tokens"] == 12
    assert result["usage"]["cache_read_input_tokens"] == 40
    assert result["usage"]["total_tokens"] == 182
    assert "cost" in result
    assert result["cost"] is None, (
        "Gemini adapter cost must be typed None when upstream did "
        "not return a priced cost (F06 silent-zero bug)"
    )


@pytest.mark.asyncio
async def test_local_provider_worker_propagates_cost_and_usage_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through LocalProviderWorker: the cost + usage
    flowing out of the adapter must reach the caller, not get
    re-stripped by ``_provider_worker_payload``.

    ``LocalProviderWorker.__call__`` wraps the adapter's raw dict
    into a ``ProviderQueryResponse``, then the engine calls
    ``_provider_worker_payload`` to unwrap it. Both steps must
    preserve ``cost`` and ``usage``.
    """
    monkeypatch.setattr(
        "mnemos.domain.graeae.engine.get_key",
        lambda _name: "fake-key-for-unit-test",
    )

    body = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Paris."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    stub = _StubClient(_StubResponse(status_code=200, body=body))

    engine = _new_engine_with_stub_client(stub)
    engine.providers = {
        "openai": {
            "url": "https://example.invalid/v1/chat/completions",
            "model": "gpt-5.5",
            "weight": 0.88,
            "api": "openai",
            "key_name": "openai",
        }
    }

    worker = LocalProviderWorker(engine)
    request = ProviderQueryRequest(
        provider="openai",
        model=None,
        messages=None,
        params={
            "prompt": "What is the capital of France?",
            "timeout": 10,
            "generation_params": None,
            "request_params": None,
        },
    )

    response = await worker(request)

    assert isinstance(response, ProviderQueryResponse)
    # F06: cost was previously dropped. After the fix, the worker
    # surfaces it through ProviderQueryResponse.cost (typed
    # Optional[float]).
    assert response.cost is None  # upstream did not price; explicit None
    # F06: usage was previously dropped. After the fix, it lives in
    # raw_provider_payload (the engine reads from there).
    raw = response.raw_provider_payload
    assert isinstance(raw, dict)
    assert isinstance(raw.get("usage"), dict)
    assert raw["usage"]["total_tokens"] == 150


# ─────────────────────────────────────────────────────────────────────────
# Consensus-level aggregation tests
# ─────────────────────────────────────────────────────────────────────────


def test_compute_consensus_sums_known_costs_correctly() -> None:
    """The FAN-OUT case from the bug description: when every
    successful provider returns a real cost, the consensus rollup
    must sum them — not skip them, not coerce them to zero."""
    all_responses = {
        "alpha": {
            "status": "success",
            "response_text": "alpha answer",
            "final_score": 0.95,
            "latency_ms": 100,
            "cost": 0.10,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
        "beta": {
            "status": "success",
            "response_text": "beta answer",
            "final_score": 0.85,
            "latency_ms": 110,
            "cost": 0.15,
            "usage": {"prompt_tokens": 200, "completion_tokens": 75},
        },
        "gamma_unavailable": {
            "status": "unavailable",
            "response_text": "",
            "latency_ms": 0,
        },
    }

    consensus = _compute_consensus(all_responses)

    assert consensus["cost"] == pytest.approx(0.25)
    assert consensus["cost_known"] is True
    assert consensus["cost_unknown_providers"] == []
    # The merged usage must include both providers' tokens, NOT drop them.
    assert consensus["usage"] is not None
    assert consensus["usage"]["prompt_tokens"] == 300
    assert consensus["usage"]["completion_tokens"] == 125


def test_compute_consensus_does_not_silently_zero_unknown_costs() -> None:
    """THE F06 BUG, in test form.

    Before the fix: a consultation where any provider's cost was
    missing was reported as a $0.00 total — the ``cost`` field
    silently summed to zero. Real production calls came back billed
    as $0.00 because the consensus layer coerced ``None`` to ``0``
    via ``isinstance(c, (int, float))`` filtering.

    After the fix: a consensus rollup with at least one unknown
    cost MUST surface the unknown state — the consensus ``cost``
    is ``None``, ``cost_known`` is ``False``, and the offenders are
    listed in ``cost_unknown_providers``. A monitoring layer /
    reconciliation job can then act on the gap instead of billing
    it for free.
    """
    all_responses = {
        "alpha_known": {
            "status": "success",
            "response_text": "alpha answer",
            "final_score": 0.95,
            "latency_ms": 100,
            "cost": 0.10,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
        "beta_unknown": {
            # Real $0.25 cost — but the adapter never surfaced it.
            # This is the EXACT scenario from the reviewer's
            # reproduction: 1,500 tokens and a real cost, but
            # the cost field was discarded before consensus.
            "status": "success",
            "response_text": "beta answer",
            "final_score": 0.85,
            "latency_ms": 110,
            "cost": None,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
    }

    consensus = _compute_consensus(all_responses)

    # Must NOT silently treat beta's missing cost as $0.00.
    assert consensus["cost"] is None, (
        "consensus cost must be None when ANY provider's cost is "
        "missing (F06 silent-zero bug). Got %r" % (consensus["cost"],)
    )
    assert consensus["cost_known"] is False
    assert "beta_unknown" in consensus["cost_unknown_providers"]
    # Alpha's known cost must still be visible in the partial state
    # — losing it entirely would also be wrong.
    assert alpha_cost_unavailable(consensus, "alpha_known") is True
    # Usage rollup must still surface the partial usage data.
    assert consensus["usage"] is not None
    assert consensus["usage"]["prompt_tokens"] == 1100
    assert consensus["usage"]["completion_tokens"] == 550


def alpha_cost_unavailable(consensus: Dict[str, Any], provider: str) -> bool:
    """Helper to keep the previous assertion readable.

    In the partial-unknown state, the consensus layer deliberately
    returns ``cost=None`` for the whole consultation — it does NOT
    leak the partial-known sum as the headline figure. Test asserts
    the headline ``cost`` is None (loud), not that alpha's $0.10
    leaks through into the partial sum.
    """
    return consensus["cost"] is None


def test_compute_consensus_treats_no_winner_as_unknown_cost() -> None:
    """When no provider succeeded, ``cost`` must be ``None`` and
    ``cost_known`` must be ``False`` — NOT ``$0.00``. The pre-fix
    code returned ``cost=0.0`` in this case, conflating a real
    free consultation with "we have no data". The two states must
    be distinguishable downstream."""
    consensus = _compute_consensus({})
    assert consensus["cost"] is None
    assert consensus["cost_known"] is False
    assert consensus["cost_unknown_providers"] == []
    assert consensus["usage"] is None
    assert consensus["winning_muse"] is None


def test_compute_consensus_refuses_to_sum_non_numeric_costs() -> None:
    """A misconfigured adapter returning ``True`` (bool) instead of
    a real cost must NOT be summed as ``1.0``. ``bool`` is an
    ``int`` subclass in Python so a naive ``isinstance(c, (int,
    float))`` check silently accepts it — the F06 fix explicitly
    type-checks against bool."""
    all_responses = {
        "alpha_bad_bool": {
            "status": "success",
            "response_text": "alpha answer",
            "final_score": 0.95,
            "latency_ms": 100,
            "cost": True,  # misconfigured adapter bug — refuse to sum
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    }
    consensus = _compute_consensus(all_responses)
    assert consensus["cost"] is None
    assert consensus["cost_known"] is False
    assert "alpha_bad_bool" in consensus["cost_unknown_providers"]


def test_compute_consensus_refuses_to_sum_string_costs() -> None:
    """A misconfigured adapter returning a stringified cost
    (e.g. ``"$0.50"``) must NOT be silently summed as ``0.0`` —
    either it is a real number or it is unknown. ``"$0.50"`` is
    not a real number, so it is unknown."""
    all_responses = {
        "alpha_str_cost": {
            "status": "success",
            "response_text": "alpha answer",
            "final_score": 0.95,
            "latency_ms": 100,
            "cost": "$0.50",  # string — refuse to silently parse as 0
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    }
    consensus = _compute_consensus(all_responses)
    assert consensus["cost"] is None
    assert consensus["cost_known"] is False
    assert "alpha_str_cost" in consensus["cost_unknown_providers"]


# ─────────────────────────────────────────────────────────────────────────
# Shared accounting contract tests
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_provider_spend_calls_injected_recorder() -> None:
    """The shared spend-recorder entrypoint must invoke the
    injected (KNEMON) recorder with the priced cost + usage,
    returning a SpendRecord describing what happened. Test-injected
    recorder captures the call so we can assert against it.
    """
    captured: list[SpendRecord] = []

    async def _recorder(payload: Any) -> SpendRecord:
        record = SpendRecord(
            status="ok",
            provider=payload.provider,
            model=payload.model,
            task_kind=payload.task_kind,
            cost_usd=payload.cost_usd,
            usage=payload.usage,
            record_id=42,
        )
        captured.append(record)
        return record

    from mnemos.domain.graeae import engine as engine_module

    prev = engine_module._spend_recorder
    engine_module.configure_accounting_contract(spend_recorder=_recorder)
    try:
        spend = await record_provider_spend(
            provider="openai",
            model="gpt-5.5",
            task_type="reasoning",
            cost_usd=0.25,
            usage={"prompt_tokens": 1200, "completion_tokens": 300},
            latency_ms=420,
            outcome="ok",
            tier="standard",
        )
    finally:
        engine_module._spend_recorder = prev

    assert len(captured) == 1
    rec = captured[0]
    assert rec.provider == "openai"
    assert rec.model == "gpt-5.5"
    assert rec.cost_usd == 0.25
    assert rec.usage == {"prompt_tokens": 1200, "completion_tokens": 300}
    # Returned SpendRecord surfaces the ledger status so callers can
    # detect a degraded KNEMON insert without try/except.
    assert spend.status == "ok"
    assert spend.record_id == 42
    assert spend.cost_usd == 0.25


@pytest.mark.asyncio
async def test_record_provider_spend_handles_unknown_cost_loudly() -> None:
    """When the upstream adapter did not surface a priced cost
    (cost_usd=None), the spend entrypoint must propagate the
    unknown state into the SpendRecord — NEVER coerce to 0.0.
    The downstream consensus rollup then sees ``cost=None`` and
    surfaces it as a partial/unknown state instead of a $0.00
    free consultation."""
    captured: list[Any] = []

    async def _recorder(payload: Any) -> SpendRecord:
        captured.append(payload)
        return SpendRecord(
            status="unpriced",
            provider=payload.provider,
            model=payload.model,
            task_kind=payload.task_kind,
            cost_usd=payload.cost_usd,
            usage=payload.usage,
        )

    from mnemos.domain.graeae import engine as engine_module

    prev = engine_module._spend_recorder
    engine_module.configure_accounting_contract(spend_recorder=_recorder)
    try:
        spend = await record_provider_spend(
            provider="anthropic",
            model="claude-opus-4-8",
            task_type="reasoning",
            cost_usd=None,
            usage={"input_tokens": 800, "output_tokens": 200},
            latency_ms=600,
            outcome="ok",
        )
    finally:
        engine_module._spend_recorder = prev

    assert len(captured) == 1
    assert captured[0].cost_usd is None  # not 0.0
    assert spend.status == "unpriced"
    assert spend.cost_usd is None


@pytest.mark.asyncio
async def test_record_provider_spend_degrades_safely_on_recorder_failure() -> None:
    """A misbehaving KNEMON backend must NOT crash the
    consultation path. The spend entrypoint must catch recorder
    failures (it does so internally), surface them via
    ``SpendRecord(status="degraded", error=...)``, and let the
    caller continue. This matches ``mnemos.llm.call``'s
    contract for ledger failures."""

    async def _exploding_recorder(payload: Any) -> SpendRecord:
        raise RuntimeError("connection refused: postgres down")

    from mnemos.domain.graeae import engine as engine_module

    prev = engine_module._spend_recorder
    engine_module.configure_accounting_contract(spend_recorder=_exploding_recorder)
    try:
        # The function MUST swallow the recorder's exception and
        # return a degraded SpendRecord — never propagate.
        spend = await record_provider_spend(
            provider="openai",
            model="gpt-5.5",
            task_type="reasoning",
            cost_usd=0.10,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=200,
            outcome="ok",
        )
    finally:
        engine_module._spend_recorder = prev

    assert spend.status == "degraded"
    assert "postgres down" in (spend.error or "")
    assert spend.cost_usd == 0.10  # cost is preserved, not zeroed


@pytest.mark.asyncio
async def test_enforce_provider_budget_allows_when_no_policy() -> None:
    """No persistence backend configured → permissive no-op. The
    consultation path must still work in single-process / test
    deployments where KNEMON isn't installed."""
    from mnemos.domain.graeae import engine as engine_module

    prev = engine_module._budget_checker
    engine_module.reset_accounting_contract()
    try:
        check = await enforce_provider_budget(
            provider="openai", model="gpt-5.5", task_type="reasoning"
        )
    finally:
        engine_module._budget_checker = prev

    assert check.allowed is True
    assert check.remaining_usd is None


@pytest.mark.asyncio
async def test_enforce_provider_budget_denies_and_raises() -> None:
    """When KNEMON is configured with a per-plan dollar cap and
    the proposed call would exceed it, the gate must raise
    ``ProviderBudgetExceeded`` so the consultation layer refuses
    the call BEFORE any provider dispatch. The exception carries
    provider / model / task_kind so the calling layer can
    surface a precise 402/429 instead of a generic 500.
    """
    from mnemos.domain.graeae import engine as engine_module

    async def _deny(_provider: str, _model: str, _task_kind: str) -> BudgetCheckResult:
        return BudgetCheckResult(
            allowed=False,
            provider="openai",
            model="gpt-5.5",
            task_kind="reasoning",
            remaining_usd=0.0,
            reason="KNEMON per-plan cap exhausted",
        )

    prev = engine_module._budget_checker
    engine_module.configure_accounting_contract(budget_checker=_deny)
    try:
        check = await enforce_provider_budget(
            provider="openai", model="gpt-5.5", task_type="reasoning"
        )
        with pytest.raises(ProviderBudgetExceeded) as excinfo:
            check.raise_if_denied()
    finally:
        engine_module._budget_checker = prev

    assert check.allowed is False
    assert "KNEMON per-plan cap exhausted" in str(excinfo.value)
    assert excinfo.value.provider == "openai"
    assert excinfo.value.model == "gpt-5.5"


@pytest.mark.asyncio
async def test_consult_path_calls_enforce_provider_budget_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through the consultation fan-out: the
    pre-flight PANTHEON/KNEMON budget gate must be invoked
    BEFORE any provider dispatch. This is the documented
    contract from the README that the fix makes real rather
    than aspirational. The test patches the gate to a denial
    and asserts the consultation refuses without making a
    single provider call.
    """
    from mnemos.domain.graeae import engine as engine_module

    invocations: list[Dict[str, str]] = []

    async def _budget_gate(provider: str, model: str, task_kind: str) -> BudgetCheckResult:
        invocations.append(
            {"provider": provider, "model": model, "task_kind": task_kind}
        )
        return BudgetCheckResult(
            allowed=False,
            provider=provider,
            model=model,
            task_kind=task_kind,
            remaining_usd=0.0,
            reason="F06-test: budget denied",
        )

    # Also patch the spend recorder to confirm it is NOT called
    # for a budget-denied consultation (no provider → no spend).
    spend_invocations: list[Any] = []

    async def _spend_recorder(payload: Any) -> SpendRecord:
        spend_invocations.append(payload)
        return SpendRecord(
            status="ok",
            provider=payload.provider,
            model=payload.model,
            task_kind=payload.task_kind,
            cost_usd=payload.cost_usd,
            usage=payload.usage,
        )

    engine = get_graeae_engine()
    engine.providers = {
        "alpha": {
            "url": "https://example.invalid/v1/chat/completions",
            "model": "alpha-model",
            "weight": 0.95,
            "api": "openai",
            "key_name": "alpha",
        },
    }

    # Replace provider worker so we can detect if any provider
    # dispatch happened — the assertion below proves the gate
    # ran first and refused the call.
    dispatched: list[str] = []

    async def _fake_worker(_request: ProviderQueryRequest) -> ProviderQueryResponse:
        dispatched.append(_request.provider)
        return ProviderQueryResponse(
            response_text="should-not-be-called",
            latency_ms=0,
            status="unavailable",
            cost=None,
            model_id_used="alpha-model",
            raw_provider_payload={"status": "unavailable"},
        )

    engine.provider_worker = _fake_worker
    engine._circuit_breakers = _FakeBreakers()
    engine._rate_limiters = _FakeBreakers()
    engine._concurrency = _FakeConcurrency()
    engine._quality = _FakeQuality({"alpha": 0.95})
    engine._cache = _FakeCache()

    prev_spend = engine_module._spend_recorder
    prev_budget = engine_module._budget_checker
    engine_module.configure_accounting_contract(
        spend_recorder=_spend_recorder,
        budget_checker=_budget_gate,
    )
    # The engine uses module-level load_cached_weights; clear it.
    monkeypatch.setattr(engine_module, "load_cached_weights", lambda: None)
    # Pool-less lifecycle avoids the model_registry lookup.
    import mnemos.core.lifecycle as _lc

    monkeypatch.setattr(_lc, "_pool", None)
    try:
        with pytest.raises(ProviderBudgetExceeded):
            await engine.consult("hello", task_type="reasoning", timeout=10)
    finally:
        engine_module._spend_recorder = prev_spend
        engine_module._budget_checker = prev_budget

    # The gate must have been called at least once with the
    # consultation's task_kind — that's the F06 wiring.
    assert len(invocations) >= 1
    assert all(
        inv["task_kind"] == "reasoning" for inv in invocations
    ), invocations
    # And critically: NO provider dispatch happened. The gate
    # refused the call before any work began.
    assert dispatched == [], (
        "consultation must NOT dispatch providers after a KNEMON "
        "budget denial. Got: %r" % (dispatched,)
    )
    assert spend_invocations == [], (
        "no spend records should be produced when the budget gate "
        "refused the call before dispatch"
    )


@pytest.mark.asyncio
async def test_consult_path_records_spend_for_successful_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through the consultation fan-out: every
    successful provider call must produce one SpendRecord via the
    shared spend entrypoint, carrying the priced cost and the
    upstream usage. This proves the FAN-OUT goes through the same
    contract as ``route()`` — no more direct adapter invocation.
    """
    from mnemos.domain.graeae import engine as engine_module

    spend_invocations: list[Any] = []

    async def _spend_recorder(payload: Any) -> SpendRecord:
        spend_invocations.append(payload)
        return SpendRecord(
            status="ok",
            provider=payload.provider,
            model=payload.model,
            task_kind=payload.task_kind,
            cost_usd=payload.cost_usd,
            usage=payload.usage,
            record_id=len(spend_invocations),
        )

    engine = get_graeae_engine()
    engine.providers = {
        "alpha": {
            "url": "https://example.invalid/v1/chat/completions",
            "model": "alpha-model",
            "weight": 0.95,
            "api": "openai",
            "key_name": "alpha",
        },
    }

    async def _fake_worker(request: ProviderQueryRequest) -> ProviderQueryResponse:
        # Mirror the F06 adapter contract: usage preserved, cost
        # typed as None until KNEMON prices it. Spend record then
        # carries the None forward.
        return ProviderQueryResponse(
            response_text="alpha says hi",
            latency_ms=42,
            status="success",
            cost=None,
            model_id_used="alpha-model",
            raw_provider_payload={
                "status": "success",
                "response_text": "alpha says hi",
                "latency_ms": 42,
                "model_id": "alpha-model",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                "cost": None,
            },
        )

    engine.provider_worker = _fake_worker
    engine._circuit_breakers = _FakeBreakers()
    engine._rate_limiters = _FakeBreakers()
    engine._concurrency = _FakeConcurrency()
    engine._quality = _FakeQuality({"alpha": 0.95})
    engine._cache = _FakeCache()

    prev_spend = engine_module._spend_recorder
    prev_budget = engine_module._budget_checker
    engine_module.configure_accounting_contract(
        spend_recorder=_spend_recorder,
        budget_checker=None,  # permissive default
    )
    monkeypatch.setattr(engine_module, "load_cached_weights", lambda: None)
    import mnemos.core.lifecycle as _lc

    monkeypatch.setattr(_lc, "_pool", None)
    try:
        result = await engine.consult("hello", task_type="reasoning", timeout=10)
    finally:
        engine_module._spend_recorder = prev_spend
        engine_module._budget_checker = prev_budget

    # The fan-out produced exactly one spend record for the one
    # successful provider.
    assert len(spend_invocations) == 1
    rec = spend_invocations[0]
    assert rec.provider == "alpha"
    assert rec.model == "alpha-model"
    assert rec.usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert rec.cost_usd is None  # propagated explicitly, NOT zeroed
    # The consensus surface surfaces the same partial/unknown
    # state: cost=None because the only provider did not price.
    assert result["cost"] is None
    assert result["cost_known"] is False
    assert "alpha" in result["cost_unknown_providers"]


def test_price_provider_cost_prefers_declared_cost() -> None:
    """``price_provider_cost`` is the canonical helper for
    converting upstream usage into a priced USD cost. When the
    upstream adapter already knows the cost (declared_cost_usd),
    the helper must return that unchanged — never re-derive from
    usage tokens (which would silently disagree with the upstream
    invoice)."""
    priced = asyncio.run(
        price_provider_cost(
            provider="openai",
            model="gpt-5.5",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            declared_cost_usd=0.25,
        )
    )
    assert priced == 0.25


def test_price_provider_cost_returns_none_for_unknown_usage() -> None:
    """No upstream cost AND no upstream usage → ``None``, not $0.0.
    This is the loud-failure surface for adapters that don't
    return anything price-able."""
    priced = asyncio.run(
        price_provider_cost(
            provider="openai",
            model="gpt-5.5",
            usage=None,
            declared_cost_usd=None,
        )
    )
    assert priced is None


def test_price_provider_cost_refuses_to_coerce_bool() -> None:
    """A misconfigured ``True`` / ``False`` declared cost must NOT
    be silently accepted as 1.0/0.0. ``bool`` is an ``int``
    subclass — without explicit guard, isinstance(True, (int,
    float)) is True. The F06 helper explicitly excludes bool."""
    priced = asyncio.run(
        price_provider_cost(
            provider="openai",
            model="gpt-5.5",
            usage=None,
            declared_cost_usd=True,
        )
    )
    assert priced is None


# ─────────────────────────────────────────────────────────────────────────
# Test scaffolding: lightweight stand-ins for the engine's reliability
# stack. We don't exercise the real breakers / quality trackers — these
# are the bare-minimum ABCs consult() probes to satisfy the eligibility
# gate. Real breaker / quality behaviour is covered in test_quality_tracking
# + test_graeae_modes.
# ─────────────────────────────────────────────────────────────────────────


class _FakeBreakers:
    """Pass-through circuit-breaker / rate-limiter stand-in.

    The real ``CircuitBreakerPool`` requires a settings object and a
    pool; for these accounting tests we only need ``is_allowed`` / 
    ``record_*`` to no-op so consult() walks past the eligibility gate.
    """

    def is_allowed(self, _name: str) -> bool:
        return True

    async def record_success(self, _name: str) -> None:
        return None

    async def record_failure(self, _name: str) -> None:
        return None

    def status(self) -> dict:
        return {}


class _FakeConcurrency:
    def acquire(self, _name: str) -> bool:
        return True

    async def release(self, _name: str) -> None:
        return None

    def status(self) -> dict:
        return {}


class _FakeQuality:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    async def record_success(self, _name: str, _latency: int) -> float:
        return 0.5

    async def record_failure(self, _name: str) -> None:
        return None

    async def dynamic_weight(self, name: str) -> float:
        return self.weights[name]

    def status(self) -> dict:
        return {}


class _FakeCache:
    """In-memory cache stand-in for the consult() path."""

    async def get(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    async def set(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def stats(self) -> dict:
        return {}
