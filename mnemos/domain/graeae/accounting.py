"""Resolve prices and durable accounting for each provider execution."""

import json

from mnemos.domain.knemon.attempts import Attempt, finite_cost
from mnemos.domain.knemon.budget import canonical_budget_key


async def provider_rates(provider, config, model):
    rates = dict(config)
    if str(config.get("model")) != model:
        for key in (
            "input_cost_per_mtok",
            "output_cost_per_mtok",
            "price_in",
            "price_out",
            "cost_per_mtok",
        ):
            rates.pop(key, None)
    # Optional catalog enrichment is local; no pricing feed/provider request.
    try:
        from mnemos.domain.pantheon.catalog import catalog_snapshot
    except ImportError:
        pass
    else:
        models = await catalog_snapshot()
        canonical, _ = canonical_budget_key(provider, model)
        for item in models:
            key, _ = canonical_budget_key(item.get("provider"), None)
            if (
                key == canonical
                and str(item.get("model_id") or item.get("id")) == model
            ):
                rates = {**item, **rates}
                break
    return rates


async def begin_attempt(
    provider, config, model, prompt, generation, request, messages, task_kind, timeout
):
    rates = await provider_rates(provider, config, model)

    def rate(*names):
        for key in names:
            value = finite_cost(rates.get(key))
            if value is not None:
                return value
        return None

    price_in = rate("input_cost_per_mtok", "price_in", "cost_per_mtok")
    price_out = rate("output_cost_per_mtok", "price_out", "cost_per_mtok")
    input_tokens = len(
        json.dumps(
            {"prompt": prompt, "messages": messages, "request": request}, default=str
        ).encode("utf-8")
    )
    if canonical_budget_key(provider, model)[0] == "anthropic":
        input_tokens *= 2  # cache creation can cost twice the base input rate
    output_tokens = int((generation or {}).get("max_tokens", 4096)) * int(
        (request or {}).get("n", 1)
    )
    estimate = (
        None
        if price_in is None or price_out is None
        else (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    )
    attempt = Attempt(
        provider=provider,
        model=model,
        subsystem="graeae",
        estimate=estimate,
        input_price=price_in,
        output_price=price_out,
        task_kind=str(task_kind or "chat"),
    )
    return await attempt.begin(timeout=timeout)


async def price_provider_cost(*, provider, model, usage, declared_cost_usd=None):
    from mnemos.domain.knemon.attempts import normalized_usage

    declared = finite_cost(declared_cost_usd)
    if declared is not None:
        return declared
    if not usage:
        return None
    rates = await provider_rates(provider, {}, model)
    price_in = finite_cost(rates.get("input_cost_per_mtok", rates.get("price_in")))
    price_out = finite_cost(rates.get("output_cost_per_mtok", rates.get("price_out")))
    if price_in is None or price_out is None:
        return None
    tokens = normalized_usage(usage)
    return (
        tokens["tokens_in"] * price_in + tokens["tokens_out"] * price_out
    ) / 1_000_000
