"""Back-compat shim — GRAEAE provider key resolution moved to mnemos-core.

The provider credential/config seam now lives in the neutral, shared
``mnemos.domain.providers`` module in ``mnemos-core`` so every add-on
(GRAEAE, PANTHEON, ...) resolves keys the same way without a peer add-on
owning the seam. This module re-exports that public API unchanged so
existing imports of ``mnemos.domain.graeae.api_keys`` keep working.

Prefer importing from ``mnemos.domain.providers`` directly in new code.
"""
from __future__ import annotations

from mnemos.domain.providers import (  # noqa: F401
    _LLM_PROVIDERS,
    _PROVIDER_ALIASES,
    _PROVIDER_ENV_VARS,
    get_key,
    get_provider_config,
    load_provider_registry,
)

__all__ = [
    "get_key",
    "get_provider_config",
    "load_provider_registry",
    "_PROVIDER_ENV_VARS",
    "_PROVIDER_ALIASES",
    "_LLM_PROVIDERS",
]
