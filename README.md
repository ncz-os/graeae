# mnemos-graeae

GRAEAE is the Mnemos **multi-muse consult engine** — the reasoning bus that fans a
query out across many providers/models, ranks the responses, and returns a
consulted answer. It is a separately installable `mnemos.*` namespace
distribution (PEP 420) that overlays onto `mnemos-core`.

## What's inside

- **Consult engine** (`mnemos.domain.graeae.engine`): the multi-provider consult
  loop, fan-out, and arbitration.
- **Provider registry & sync** (`mnemos.domain.graeae.model_registry`,
  `…provider_sync`, `…sync_provider_models`, `…api_keys`): the catalog of
  providers/models, key handling, and periodic model discovery.
- **Ranking** (`mnemos.domain.graeae._quality`, `…elo_sync`, `…_cache`): response
  quality scoring and ELO-style ranking across providers.
- **Provider worker** (`mnemos.domain.graeae.provider_worker`): async execution of
  per-provider calls.
- **API** (`mnemos.api.routes.consultations`, `…routes.providers`): the
  `/consult` surface and provider/health introspection.

## Install

```bash
pip install mnemos-core mnemos-graeae
```

GRAEAE is bundled in the Mnemos umbrella image (`ghcr.io/ncz-os/mnemos`) and the
`server`/`full` bundles, and is separable for minimal core installs. When the
distribution is present, `mnemos-core` mounts the GRAEAE routes automatically;
when absent, core boots without them.

## Relationship to the rest of MNEMOS

GRAEAE depends on `mnemos-core` (one direction only). It pairs with
`mnemos-pantheon` (the unified LLM facade GRAEAE calls through) and
`mnemos-knemon` (which budgets GRAEAE consultations by cost tier). The
fleet-wide agent coordination layer that *uses* GRAEAE is a separate service,
**STIPHOS** (`mnemos-stiphos`).

> **Note on history:** the canonical source for this distribution is
> `gitlab.com/ncz-os/graeae` (mirrored to `github.com/ncz-os/graeae`). The
> earlier monolithic `graeae_api.py` application has been refactored into this
> packaged distribution plus the running GRAEAE service.
