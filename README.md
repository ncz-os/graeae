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

GRAEAE is bundled in the Mnemos umbrella image (`ghcr.io/ncz-os/mnemos-enterprise`) and the
`server`/`full` bundles, and is separable for minimal core installs. When the
distribution is present, `mnemos-core` mounts the GRAEAE routes automatically;
when absent, core boots without them.

## Development

GRAEAE and core contribute separate portions of the same PEP 420 `mnemos`
namespace. For a split-repository checkout, install both distributions editably
instead of setting `PYTHONPATH`:

```bash
git clone https://gitlab.com/ncz-os/mnemos.git ../mnemos-core
python -m pip install -e ../mnemos-core -e ".[test]"
pytest
```

The GitLab test job uses this same setup from a clean checkout.

## Operator Guardrails

Gateway calls are bounded before GRAEAE dispatches to upstream LLM providers:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MNEMOS_GRAEAE_MAX_OUTPUT_TOKENS` | `8192` | Maximum caller-requested output tokens forwarded to a provider. |
| `MNEMOS_GRAEAE_MAX_CANDIDATES` | `1` | Maximum caller-requested candidate count (`n`) forwarded to providers that support it. |
| `MNEMOS_GRAEAE_AUDIT_HMAC_KEY` | no default; required | Secret used to authenticate the consultation audit chain. Generate a high-entropy value and provision it before enabling GRAEAE. Requests fail before provider dispatch when it is absent. |
| `MNEMOS_GRAEAE_CONSULT_QUOTA_PER_MIN` | `20` | Cost-weighted consultation units allowed per principal each minute. Enforced through the active persistence backend's `state_kv` ABC (no Redis dependency); row-locked for safe multi-instance coordination. |

The engine also clips temperature/top-p, penalties, and stop sequences to
provider-safe ranges before dispatch. Operators can raise the token/candidate
limits explicitly for trusted deployments.

## Relationship to the rest of MNEMOS

GRAEAE depends on `mnemos-core` and `mnemos-knemon`. It calls provider adapters
directly and uses KNEMON for per-attempt monetary admission and settlement.
`mnemos-pantheon` is an optional facade that can call GRAEAE; GRAEAE does not
route every provider call through PANTHEON. The
fleet-wide agent coordination layer that *uses* GRAEAE is a separate service,
**STIPHOS** (`mnemos-stiphos`).

> **Note on history:** the canonical source for this distribution is
> `gitlab.com/ncz-os/graeae` (mirrored to `github.com/ncz-os/graeae`). The
> earlier monolithic `graeae_api.py` application has been refactored into this
> packaged distribution plus the running GRAEAE service.


## Backend coverage

GRAEAE's consultation write path, its audit chain and its cost-weighted quota
all run through core's persistence ABC and work on every backend core supports.

**One path does not: `_ranked_lineup`'s `model_registry` ranking is
PostgreSQL-only.** It reads `mnemos.core.lifecycle._pool`, which core populates
only for the PostgreSQL profile and sets to `None` everywhere else. On SQLite,
MySQL, MariaDB, Oracle and Db2 the registry ranking is therefore skipped
entirely and the lineup falls back to the static `config.toml` weights with no
registry `model_id` override — even though those backends advertise the
`consultations` capability and GRAEAE's own backend-neutral provider sync is
actively populating them.

This is **not fixed**, and is not presented as fixed. It is now *diagnosable*
rather than silent: the degradation logs one warning per process naming the
active backend, where previously nothing was emitted at all (the surrounding
`logger.debug` only fired on an exception, never on the skipped branch, so a
frozen muse lineup was completely invisible). Reproducing the PostgreSQL
semantics needs a `graeae_weight` value per candidate provider, and no method on
`ConsultationsRepository` or `ConsultationAuditRepository` returns one —
`graeae_weight` appears in core's `persistence/base.py` only as a write-side
input to `update_arena_score`. A real fix needs a new abstract method in core
plus six backend implementations, which is a core-owner decision.

Separately, two raw-asyncpg, PostgreSQL-only audit writers
(`_write_audit_entry_on_conn`, `_write_memory_refs_on_conn`) were removed. They
had no callers anywhere — the live path has been
`backend.consultations.create_consultation_with_audit` — and were left as a
hazard for the next edit to reach for by mistake.

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


## Budget and agreement boundaries

Python 3.13 or newer is required, matching the core distribution. Development
installs need the real `../mnemos-core` and `../knemon` checkouts; fake resolver
packages and import-time runtime substitutions are not supported.

Every consultation participant and direct/streaming provider attempt reserves
its own estimated cost before execution. The configured provider/model rate and
bounded output allowance determine the reservation. A model override must have
its own rates. When caps are configured, missing prices or persistence deny the
attempt with an explicit error. Uncapped development can operate without a
ledger; that mode does not provide durable spend enforcement. Budget scope is
`graeae`; a PANTHEON request delegated here is accounted here once.

Usage is normalized across OpenAI, Anthropic, and Gemini adapters. Stream
completion, cancellation, and ambiguous transport failures finalize accounting;
missing usage retains a conservative estimate. Unknown consultation cost stays
null. KNEMON keeps estimated recovery costs after crashes and corrects them
when an authoritative late settlement arrives.

`consensus_score` is a lexical agreement heuristic, not a probability of
correctness or a semantic contradiction detector. Quorum groups require every
pair in the group to meet the threshold; similarity chains cannot create
agreement between their endpoints. Matching templates with different numeric
values and certain short negation pairs are rejected. These guards do not detect
all contradictions. `agreement_metric=lexical_complete_link` and
`contradiction_pairs` expose the method and observed guards.


### Live quota concurrency validation

`tests/test_quota_live.py` checks first-use admission from eight concurrent
requests against an empty principal with a one-unit quota. It requires a
dedicated test schema/database, `MNEMOS_TEST_ALLOW_RESET=1`, and either
`MNEMOS_TEST_ORACLE_DSN`, `MNEMOS_TEST_DB2_DSN`, `MNEMOS_TEST_MYSQL_DSN`, or
`MNEMOS_TEST_MARIADB_DSN`. Oracle/Db2 cleanup deletes only the random quota key;
MySQL/MariaDB fixtures create and drop a randomly named database. Oracle needs the `oracledb` driver and Db2 needs `ibm_db`.
The runtime creates the quota row before locking: Oracle does not gap-lock
missing rows. Native Db2 uses positional binds and retains the update lock.
MySQL/MariaDB first-use deadlocks are avoided by an insert-or-no-op before the
locking read. Duplicate-key races alone are tolerated; database fetch failures
deny admission.
