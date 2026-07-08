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

## Operator Guardrails

Gateway calls are bounded before GRAEAE dispatches to upstream LLM providers:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MNEMOS_GRAEAE_MAX_OUTPUT_TOKENS` | `8192` | Maximum caller-requested output tokens forwarded to a provider. |
| `MNEMOS_GRAEAE_MAX_CANDIDATES` | `1` | Maximum caller-requested candidate count (`n`) forwarded to providers that support it. |

The engine also clips temperature/top-p, penalties, and stop sequences to
provider-safe ranges before dispatch. Operators can raise the token/candidate
limits explicitly for trusted deployments.

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


## Build infrastructure & partners

Continuous integration and package distribution for this project are generously
supported by our open-source infrastructure partners:

- **[GitLab](https://gitlab.com/)** — canonical source hosting and CI pipelines
  (format / lint / test gates), via the
  [GitLab for Open Source](https://about.gitlab.com/solutions/open-source/) program.
- **[Buildkite](https://buildkite.com/)** — CI/CD orchestration with hosted macOS
  and Linux agents, and our APT package registry host
  (`packages.buildkite.com/ncz-os/ncz`), via the
  [Buildkite Open Source](https://buildkite.com/pricing) program.

Thank you to both for backing open-source software.
