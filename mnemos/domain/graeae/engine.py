#!/usr/bin/env python3
"""
GRAEAE Multi-Provider Consensus Engine

Queries multiple AI providers in parallel and returns all responses.

Provider registry
-----------------
Providers are declared in config.toml under [graeae.providers.<name>] — no
code changes needed to add or modify a provider. Keys are resolved from
~/.config/mnemos/api_keys.json (or $MNEMOS_KEYS_PATH). Built-in defaults
are used as a fallback when config.toml has no [graeae.providers] section.

API adapter styles (the "api" field in config.toml):
  "openai"    — OpenAI-compatible chat completions (Perplexity, Groq, xAI, OpenAI)
  "anthropic" — Anthropic Messages API
  "gemini"    — Google Gemini generateContent API

GPT-5 series is detected by model name ("gpt-5") and automatically uses
max_completion_tokens instead of max_tokens — no separate api_type needed.

Reliability stack (innermost to outermost):
  _concurrency     — asyncio.Semaphore per provider; sheds load when saturated
                     (Triton instance-slot model — skip, don't queue)
  _circuit_breaker — trips after 5 failures; recovers via HALF_OPEN probe
  _rate_limiter    — sliding-window RPM guard
  _quality         — rolling success-rate multiplier on base weight
  _cache           — in-memory LRU keyed on normalised prompt hash (1h TTL)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, Optional

import httpx
from mnemos.core.config import get_settings
from mnemos.core.provider_registry import GRAEAE_REGISTRY_MAP
from mnemos.core.resilience import (
    call_maybe_async,
    make_circuit_breaker_pool,
    make_concurrency_limiter,
    make_rate_limiter_pool,
)
from mnemos.domain.graeae._cache import make_response_cache
from mnemos.domain.graeae._quality import make_quality_tracker
from mnemos.domain.graeae.accounting import price_provider_cost as price_provider_cost
from mnemos.domain.graeae.elo_sync import load_cached_weights
from mnemos.domain.graeae.provider_worker import (
    LocalProviderWorker,
    ProviderQueryRequest,
    ProviderQueryResponse,
    ProviderWorker,
)
from mnemos.domain.providers import _PROVIDER_ENV_VARS, get_key

logger = logging.getLogger(__name__)


def _env_var_hint(key_name: str) -> str:
    """Return the env-var name an operator would export to bypass
    the Provider Registry File for a given key_name. Used in error
    messages so the hint is actionable."""
    return _PROVIDER_ENV_VARS.get(key_name, f"<{key_name.upper()}_API_KEY>")


_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_DEFAULT_MAX_CANDIDATES = 1
_DEFAULT_RANKED_LINEUP_TTL_SECONDS = 60
_DEFAULT_DEBATE_PEER_CONTEXT_CHARS = 64 * 1024
_RANKED_LINEUP_CACHE_MAX_ENTRIES = 128
_MAX_STOP_SEQUENCES = 4
_MAX_STOP_SEQUENCE_CHARS = 512


# #192: removed `ProviderResponse` (dataclass) — declared but
# never instantiated or referenced anywhere. The live provider-
# response shape is `ProviderQueryResponse` (used by
# `_provider_worker_payload`).


class ProviderStreamError(RuntimeError):
    """Structured provider error decoded from an SSE data frame."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "provider_error",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}


GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
}


def _normalize_gemini_finish_reason(raw: str | None) -> str:
    if raw is None:
        return "stop"
    raw_value = str(raw)
    canonical = {"stop", "length", "content_filter", "tool_calls"}
    if raw_value.lower() in canonical:
        return raw_value.lower()
    return GEMINI_FINISH_REASON_MAP.get(raw_value.upper(), "stop")


# Built-in provider defaults — used when config.toml has no [graeae.providers] section.
# Operators override these (or add new providers) via config.toml exclusively.
#
# Model IDs refreshed 2026-04-23 to current frontier (v3.1.2 Defect 3).
# Operators who want earlier generations override via config.toml.
# Defaults assume each provider's "flagship available to API" tier.
_BUILTIN_PROVIDERS: dict[str, dict] = {
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        # iter 56 registry-align: MiniMax-M2.7 FP4 is the current Together
        # default per llm-usage-policy-2026-05-22.md ($0.40/$1.20, 4M ctx,
        # weight=0.78 in model_registry).
        "model": "MiniMaxAI/MiniMax-M2.7",
        "weight": 0.78,
        "api": "openai",
        "key_name": "together_ai",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "weight": 0.80,
        "api": "openai",
        "key_name": "groq",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        # iter 56 registry-align: gpt-5.5 is arena rank 8 (score 1481) per
        # 2026-05-28 snapshot; supersedes 5.2-chat-latest.
        "model": "gpt-5.5",
        "weight": 0.88,
        "api": "openai",
        "key_name": "openai",
    },
    "claude": {
        "url": "https://api.anthropic.com/v1/messages",
        # iter 56 operator override: keep opus-4-6 for quality. Early-return
        # short-circuit (consult() gather block) handles the latency by
        # cancelling whichever muse is slowest once quorum reached, so opus
        # being slow on arch-design prompts is no longer a blocker.
        "model": "claude-opus-4-8",
        "weight": 0.90,
        "api": "anthropic",
        "key_name": "claude",
    },
    "perplexity": {
        "url": "https://api.perplexity.ai/chat/completions",
        "model": "sonar-pro",
        "weight": 0.88,
        "api": "openai",
        "key_name": "perplexity",
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        # iter 56 registry-align: grok-4.20-0309-reasoning is arena rank 21
        # (score 1453) per 2026-05-28 snapshot; supersedes 4-1-fast.
        "model": "grok-4.20-0309-reasoning",
        "weight": 0.86,
        "api": "openai",
        "key_name": "xai",
    },
    "nvidia": {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        # iter 56 registry-align: NGC Kimi K2.6 is arena rank 19 (score
        # 1457) + 80.2% SWE-bench Verified, free Tier-A per
        # llm-usage-policy-2026-05-22.md.
        "model": "moonshotai/kimi-k2.6",
        "weight": 0.80,
        "api": "openai",
        "key_name": "nvidia",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent",
        "model": "gemini-3.1-pro-preview",
        "weight": 0.88,
        "api": "gemini",
        "key_name": "gemini",
    },
}


def _builtin_fallback() -> dict[str, dict]:
    """Built-in provider defaults, each tagged ``_source="builtin"``.

    The tag marks these as baked-in fallbacks rather than operator-supplied
    config. Consumers such as the PANTHEON gateway use it to refuse a builtin
    vendor URL as an idempotent *connection* endpoint (a mass-distribution image
    must resolve its endpoint from runtime config, not a baked default); GRAEAE's
    own consensus path still uses the URLs to call muses. Providers loaded from
    config.toml are intentionally left untagged — absence of the tag means
    operator-configured.
    """
    return {
        name: {**dict(cfg), "_source": "builtin"}
        for name, cfg in _BUILTIN_PROVIDERS.items()
    }


def _load_providers() -> dict[str, dict]:
    """Load provider registry from config.toml [graeae.providers].

    Falls back to _BUILTIN_PROVIDERS if the section is absent.
    Providers with enabled=false are excluded.
    The TOML 'api' field is kept as-is; dispatch in _call_provider_worker() reads it.
    """
    try:
        from mnemos.core.config import GRAEAE_CONFIG

        registry = GRAEAE_CONFIG.get("providers", {})
    except ImportError:
        registry = {}

    if not registry:
        logger.debug("[GRAEAE] no providers in config.toml — using built-in defaults")
        return _builtin_fallback()

    providers: dict[str, dict] = {}
    for name, cfg in registry.items():
        if not cfg.get("enabled", True):
            logger.info(
                f"[GRAEAE] provider '{name}' disabled in config.toml — skipping"
            )
            continue
        required = {"url", "model", "weight", "api", "key_name"}
        missing = required - cfg.keys()
        if missing:
            logger.warning(
                f"[GRAEAE] provider '{name}' missing fields {missing} — skipping"
            )
            continue
        providers[name] = dict(cfg)

    if not providers:
        logger.warning(
            "[GRAEAE] config.toml [graeae.providers] is empty — using built-in defaults"
        )
        return _builtin_fallback()

    logger.info(
        f"[GRAEAE] loaded {len(providers)} providers from config.toml: {list(providers)}"
    )
    return providers


def _provider_scope(cfg: dict) -> str:
    """Classify a provider as ``"local"`` or ``"external"``.

    This is what makes ``consult(mode="local"|"external")`` an actual
    provider-*scope* filter rather than a mere quorum label. Resolution
    order:

      1. An explicit ``scope`` (or legacy ``location``) field on the
         provider config — ``"local"`` / ``"external"`` — always wins.
         This is REQUIRED for providers whose URL host does not reflect
         their true scope: e.g. an OpenAI/Anthropic muse reached through
         a loopback OAuth shim (``http://127.0.0.1:5079/...``) is an
         *external* model and must declare ``scope = "external"``, since
         host inference would otherwise call it local.
      2. Otherwise infer from the URL host: loopback / RFC-1918-private /
         link-local IPs, ``localhost``, ``*.local|*.lan|*.internal``, and
         bare (dotless) hostnames are ``"local"`` (LAN GPU servers);
         public FQDNs are ``"external"``.

    Unknown / unparseable hosts default to ``"external"`` — fail-open to
    the cloud panel rather than silently routing an external-only
    consult at an unverified endpoint.
    """
    explicit = str(cfg.get("scope") or cfg.get("location") or "").strip().lower()
    if explicit in {"local", "external"}:
        return explicit

    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(str(cfg.get("url", ""))).hostname or "").lower()
    if not host:
        return "external"
    if host == "localhost" or host.endswith((".local", ".lan", ".internal")):
        return "local"
    try:
        ip = ipaddress.ip_address(host)
        return (
            "local"
            if (ip.is_private or ip.is_loopback or ip.is_link_local)
            else "external"
        )
    except ValueError:
        # Bare hostname (no dots) => LAN box; dotted FQDN => public cloud.
        return "local" if "." not in host else "external"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image_url":
                image_url = block.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if url:
                    parts.append(f"[image_url: {url}]")
        return "\n".join(p for p in parts if p)
    return str(content)


def _chat_message_content_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return str(content) if content is not None else ""


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _provider_stream_error(data: dict[str, Any]) -> ProviderStreamError:
    raw_error = data.get("error")
    if isinstance(raw_error, dict):
        message = str(
            raw_error.get("message")
            or raw_error.get("detail")
            or "Provider stream error"
        )
        error_type = str(
            raw_error.get("type") or raw_error.get("code") or "provider_error"
        )
        raw_status = raw_error.get("status_code") or raw_error.get("status")
        try:
            status_code = int(raw_status) if raw_status is not None else 400
        except (TypeError, ValueError):
            status_code = 400
        if not 400 <= status_code <= 599:
            status_code = 400
        return ProviderStreamError(
            message,
            error_type=error_type,
            status_code=status_code,
            details=raw_error,
        )
    message = str(raw_error) if raw_error is not None else "Provider stream error"
    return ProviderStreamError(message, details={"error": raw_error})


def _openai_messages(messages: Optional[list[dict]], prompt: str) -> list[dict]:
    if messages:
        normalized: list[dict] = []
        for msg in messages:
            item = {k: v for k, v in msg.items() if v is not None}
            if "content" not in item and "tool_calls" not in item:
                item["content"] = ""
            normalized.append(item)
        return normalized
    return [{"role": "user", "content": prompt}]


def _anthropic_content_parts(content: Any) -> list[dict]:
    if isinstance(content, list):
        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "image_url":
                image_url = block.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if url:
                    parts.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
        return parts or [{"type": "text", "text": ""}]
    return [{"type": "text", "text": _content_text(content)}]


def _anthropic_tool_input(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def _anthropic_tool_use_blocks(tool_calls: Any) -> list[dict]:
    blocks: list[dict] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = function.get("name")
        call_id = call.get("id")
        if not name or not call_id:
            continue
        blocks.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": _anthropic_tool_input(function.get("arguments")),
            }
        )
    return blocks


def _append_anthropic_message(
    messages: list[dict], role: str, content: list[dict]
) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(content)
        return
    messages.append({"role": role, "content": content})


def _anthropic_messages(
    messages: Optional[list[dict]], prompt: str
) -> tuple[Optional[str], list[dict]]:
    if not messages:
        return None, [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    system_parts: list[str] = []
    converted: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            text = _content_text(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "assistant":
            content = (
                []
                if msg.get("content") is None and msg.get("tool_calls")
                else _anthropic_content_parts(msg.get("content"))
            )
            content.extend(_anthropic_tool_use_blocks(msg.get("tool_calls")))
            _append_anthropic_message(converted, "assistant", content)
            continue
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": _content_text(msg.get("content")),
                }
            ]
            _append_anthropic_message(converted, "user", content)
            continue
        _append_anthropic_message(
            converted, "user", _anthropic_content_parts(msg.get("content"))
        )
    if not converted:
        converted.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
    return ("\n\n".join(system_parts) if system_parts else None), converted


def _anthropic_finish_reason(stop_reason: Any) -> str:
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "end_turn":
        return "stop"
    return stop_reason or "stop"


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            converted.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object"}),
                }
            )
        else:
            converted.append(tool)
    return converted


def _anthropic_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice in ("any", "required"):
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        raise ValueError(f"unsupported Anthropic tool_choice {tool_choice!r}")
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        if tool_choice.get("type") == "function" and fn.get("name"):
            return {"type": "tool", "name": fn["name"]}
    raise ValueError(f"unsupported Anthropic tool_choice {tool_choice!r}")


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return min(max(parsed, minimum), maximum)


def _max_output_tokens() -> int:
    return _positive_int_env(
        "MNEMOS_GRAEAE_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS
    )


def _max_candidates() -> int:
    return _positive_int_env("MNEMOS_GRAEAE_MAX_CANDIDATES", _DEFAULT_MAX_CANDIDATES)


def _ranked_lineup_ttl_seconds() -> int:
    return _positive_int_env(
        "MNEMOS_GRAEAE_RANKED_LINEUP_TTL_SECONDS",
        _DEFAULT_RANKED_LINEUP_TTL_SECONDS,
    )


def _debate_peer_context_chars() -> int:
    return _positive_int_env(
        "MNEMOS_GRAEAE_DEBATE_PEER_CONTEXT_CHARS",
        _DEFAULT_DEBATE_PEER_CONTEXT_CHARS,
    )


class _RankedLineupCache:
    """Thread-safe TTL/LRU cache for model_registry ranking lookups."""

    def __init__(
        self, ttl_seconds: int, max_entries: int = _RANKED_LINEUP_CACHE_MAX_ENTRIES
    ):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: OrderedDict[
            tuple[str, ...], tuple[float, list[tuple[str, str | None, float]]]
        ] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[str, ...]) -> list[tuple[str, str | None, float]] | None:
        with self._lock:
            if key not in self._store:
                return None
            stored_at, value = self._store[key]
            if time.monotonic() - stored_at > self.ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return list(value)

    def set(
        self, key: tuple[str, ...], value: list[tuple[str, str | None, float]]
    ) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), list(value))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)


def _bounded_stop_sequences(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return value[:_MAX_STOP_SEQUENCE_CHARS]
    if isinstance(value, list):
        items = [
            str(item)[:_MAX_STOP_SEQUENCE_CHARS] for item in value[:_MAX_STOP_SEQUENCES]
        ]
        return items
    return None


def _bounded_generation_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    bounded: Dict[str, Any] = {}
    if "max_tokens" in params:
        bounded["max_tokens"] = _bounded_int(
            params["max_tokens"],
            default=4096,
            minimum=1,
            maximum=_max_output_tokens(),
        )
    if "temperature" in params:
        temperature = _bounded_float(params["temperature"], minimum=0.0, maximum=2.0)
        if temperature is not None:
            bounded["temperature"] = temperature
    if "top_p" in params:
        top_p = _bounded_float(params["top_p"], minimum=0.0, maximum=1.0)
        if top_p is not None:
            bounded["top_p"] = top_p
    return bounded


def _bounded_request_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    bounded: Dict[str, Any] = {}
    for field in ("tools", "tool_choice", "response_format", "user"):
        if field in params:
            bounded[field] = params[field]
    if "parallel_tool_calls" in params and isinstance(
        params["parallel_tool_calls"], bool
    ):
        bounded["parallel_tool_calls"] = params["parallel_tool_calls"]
    if "stop" in params:
        stop = _bounded_stop_sequences(params["stop"])
        if stop is not None:
            bounded["stop"] = stop
    if "n" in params:
        bounded["n"] = _bounded_int(
            params["n"],
            default=1,
            minimum=1,
            maximum=_max_candidates(),
        )
    for field in ("presence_penalty", "frequency_penalty"):
        if field in params:
            penalty = _bounded_float(params[field], minimum=-2.0, maximum=2.0)
            if penalty is not None:
                bounded[field] = penalty
    return bounded


def _gemini_part(content: Any) -> list[dict]:
    if isinstance(content, list):
        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append({"text": block.get("text", "")})
            elif block.get("type") == "image_url":
                image_url = block.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if url:
                    parts.append({"fileData": {"fileUri": url}})
        return parts or [{"text": ""}]
    return [{"text": _content_text(content)}]


def _gemini_contents(
    messages: Optional[list[dict]], prompt: str
) -> tuple[Optional[dict], list[dict]]:
    if not messages:
        return None, [{"role": "user", "parts": [{"text": prompt}]}]

    system_text: list[str] = []
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            text = _content_text(msg.get("content"))
            if text:
                system_text.append(text)
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Gemini does not support role={role}; supported: system, user, assistant"
            )
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            {"role": gemini_role, "parts": _gemini_part(msg.get("content"))}
        )
    if not contents:
        contents.append({"role": "user", "parts": [{"text": prompt}]})
    system_instruction = None
    if system_text:
        system_instruction = {"parts": [{"text": "\n\n".join(system_text)}]}
    return system_instruction, contents


class GraeaeEngine:
    """Multi-provider consensus reasoning engine."""

    def __init__(self):
        self.providers = _load_providers()
        self._client: Optional[httpx.AsyncClient] = None

        # Seed base weights from Arena.ai Elo leaderboard if available.
        # Uses on-disk cache — falls back to config.toml weights silently.
        elo = load_cached_weights()
        if elo:
            for name, w in elo.items():
                if name in self.providers:
                    self.providers[name]["weight"] = w
            logger.info(
                f"[GRAEAE] Elo weights applied for: {[p for p in elo if p in self.providers]}"
            )

        # Reliability stack — instantiated here; _concurrency lazily initialised
        # on first consult() call because asyncio.Semaphore needs a running loop.
        self._settings = get_settings()
        self._circuit_breakers = make_circuit_breaker_pool(
            self._settings,
            failure_threshold=5,
            cooldown_seconds=300,
        )
        self._rate_limiters = make_rate_limiter_pool(self._settings)
        provider_weights = {p: cfg["weight"] for p, cfg in self.providers.items()}
        self._quality = make_quality_tracker(self._settings, provider_weights)
        self._cache = make_response_cache(
            self._settings, ttl_seconds=3600, max_entries=500
        )
        self._ranked_lineup_cache = _RankedLineupCache(_ranked_lineup_ttl_seconds())
        self._concurrency: Optional[Any] = None
        self.provider_worker: ProviderWorker = LocalProviderWorker(self)

    def _get_concurrency(self) -> Any:
        """Lazy-init concurrency pool (requires running event loop)."""
        if self._concurrency is None:
            self._concurrency = make_concurrency_limiter(self._settings)
        return self._concurrency

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Pool sized for fan-out consults (mode=all queries ~8 muses
            # concurrently); default httpx keepalive=20 would cap reuse and
            # force re-handshakes under fan-out.
            self._client = httpx.AsyncClient(
                timeout=200,
                limits=httpx.Limits(
                    max_keepalive_connections=50,
                    max_connections=200,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        resources = [
            self._cache,
            self._quality,
            self._circuit_breakers,
            self._rate_limiters,
            self._concurrency,
        ]
        for resource in resources:
            close = getattr(resource, "close", None)
            if close is not None:
                await call_maybe_async(close)
        global _graeae_engine
        if _graeae_engine is self:
            _graeae_engine = None

    async def reload_from_registry(self, pool) -> Dict[str, str]:
        """Refresh self.providers[*]['model'] from model_registry.

        Selection policy (per provider):
          1. ELO override: Arena-ranked top within the family is the default.
          2. Newer-version override: any same-family model with a strictly
             higher version tuple than the Arena top wins (catches fresh
             releases like claude-opus-4-7 over Arena-leading 4-6, or
             gpt-5.5 the day it ships ahead of Arena-ranked 5.4).
          3. Live probe + n-1 fallback: each candidate, in rank order, gets
             a tiny generate call; the first that returns 200 wins. If the
             top pick fails (model retired, regional rollout, key tier),
             we fall through to the next-best — usually the previous
             generation.

        Two-phase to avoid stalling startup or holding a pool connection
        for ~minutes on slow upstreams:
          Phase 1: pull every provider's candidate list in one DB conn,
                   then RELEASE the connection.
          Phase 2: probe candidates in parallel ACROSS providers (sequential
                   within a provider so n-1 fallback ordering is preserved),
                   no DB held.

        url/api/key_name/weight are preserved from the existing provider
        config — only the model field changes. gemini's URL embeds the
        model name and is rebuilt on rotation.
        """
        # Phase 1 — DB-bound, release the connection before any HTTP probe.
        plans: Dict[str, list[str]] = {}
        api_keys: Dict[str, str] = {}
        async with pool.acquire() as conn:
            for name, mapping in _REGISTRY_MAP.items():
                if name not in self.providers:
                    continue
                cfg = self.providers[name]
                api_key = get_key(cfg["key_name"])
                if not api_key:
                    logger.info(
                        "[GRAEAE] reload: %s → no api key, keeping %s",
                        name,
                        cfg.get("model"),
                    )
                    continue
                candidates = await _ranked_candidates(
                    conn,
                    mapping["registry_provider"],
                    mapping["prefer"],
                )
                if not candidates:
                    logger.info(
                        "[GRAEAE] reload: %s → registry empty, keeping %s",
                        name,
                        cfg.get("model"),
                    )
                    continue
                plans[name] = candidates[:_PROBE_MAX_CANDIDATES]
                api_keys[name] = api_key

        if not plans:
            logger.info("[GRAEAE] manifest reload: no eligible providers")
            return {}

        # Phase 2 — probe in parallel across providers (no DB held).
        client = await self._get_client()

        async def _probe_one_provider(name: str, cands: list[str], api_key: str):
            cfg = self.providers[name]
            tried: list[str] = []
            for cand in cands:
                if await _probe_model(client, cfg, cand, api_key):
                    return name, cand, tried
                tried.append(cand)
            return name, None, tried

        results = await asyncio.gather(
            *(_probe_one_provider(n, plans[n], api_keys[n]) for n in plans),
            return_exceptions=True,
        )

        changes: Dict[str, str] = {}
        for r in results:
            if isinstance(r, BaseException):
                logger.warning(f"[GRAEAE] probe task crashed: {type(r).__name__}: {r}")
                continue
            name, chosen, tried = r
            cfg = self.providers[name]
            if chosen is None:
                logger.warning(
                    "[GRAEAE] %s: all %d candidates failed probe (%s) — keeping %s",
                    name,
                    len(tried),
                    tried,
                    cfg.get("model"),
                )
                continue
            if tried:
                logger.info(
                    "[GRAEAE] %s: probe failures %s → falling back to %s (n-%d)",
                    name,
                    tried,
                    chosen,
                    len(tried),
                )
            old = cfg.get("model")
            if old == chosen:
                continue
            cfg["model"] = chosen
            if name == "gemini":
                cfg["url"] = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{chosen}:generateContent"
                )
            changes[name] = f"{old} → {chosen}"
        if changes:
            logger.info(f"[GRAEAE] manifest refreshed from registry: {changes}")
        else:
            logger.info("[GRAEAE] manifest reload: no changes (already current)")
        return changes

    async def consult(
        self,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        selection: Optional[Dict[str, Optional[str]]] = None,
        mode: str = "auto",
    ) -> Dict:
        """Query eligible providers in parallel and return all responses.

        `selection` (v3.2 Custom Query mode) is an optional
        `{provider_name: model_override_or_None}` dict. When set, only
        those providers are considered for the fan-out; every other
        registered provider is omitted (not marked unavailable). A
        `model_override` value, if not None, overrides
        `self.providers[name]["model"]` for that one call.

        When `selection` is None, behavior is unchanged — every
        registered provider is considered (current auto-lineup).
        """
        task_type = task_type or "reasoning"
        mode = mode or "auto"
        if mode == "single":
            return await self.route_single(
                prompt,
                task_type,
                timeout=timeout,
                selection=selection,
            )
        if mode == "debate":
            return await self.route_debate(
                prompt,
                task_type,
                timeout=timeout,
                selection=selection,
            )
        if mode == "majority":
            return await self.route_majority(
                prompt,
                task_type,
                timeout=timeout,
                selection=selection,
            )
        if mode not in {"auto", "local", "external", "all"}:
            raise ValueError(f"unsupported consultation mode {mode!r}")

        # ── Selection-aware iteration list ───────────────────────────────────
        # If Custom Query set a lineup, respect it verbatim; unknown
        # provider names should have been rejected by the caller before
        # reaching the engine, but we guard defensively.
        if selection is not None:
            candidate_providers = [p for p in selection if p in self.providers]
        else:
            candidate_providers = list(self.providers)
            # ── Scope filter ─────────────────────────────────────────────
            # `local` / `external` are provider-SCOPE filters, not just
            # quorum labels: restrict the lineup to muses whose endpoint
            # matches the requested scope. `auto` / `all` keep every
            # registered muse. (A Custom Query `selection` is honored
            # verbatim and is not scope-filtered.)
            if mode in {"local", "external"}:
                candidate_providers = [
                    p
                    for p in candidate_providers
                    if _provider_scope(self.providers[p]) == mode
                ]
                logger.info(
                    "[GRAEAE] mode=%s scope filter -> %d/%d providers: %s",
                    mode,
                    len(candidate_providers),
                    len(self.providers),
                    candidate_providers,
                )
                if not candidate_providers:
                    logger.error(
                        "[GRAEAE] mode=%s selected zero providers "
                        "(no muse classified as %s)",
                        mode,
                        mode,
                    )
                    return {
                        "all_responses": {},
                        "error": f"no {mode} providers configured",
                    }

        # Cache identity is derived only after the scope filter. It binds the
        # normalized mode and every effective provider/model pair, preventing
        # local requests from consuming external responses (and stale lineups
        # from surviving a provider-model configuration change).
        effective_lineup = {
            name: (
                selection.get(name)
                if selection is not None and selection.get(name) is not None
                else self.providers[name].get("model")
            )
            for name in candidate_providers
        }
        cache_model = f"mode={mode}{_selection_cache_tag(effective_lineup)}"
        cached = await call_maybe_async(self._cache.get, prompt, task_type, cache_model)
        if cached is not None:
            logger.info(
                f"[GRAEAE] cache hit (task_type={task_type}, model={cache_model})"
            )
            return {
                "all_responses": cached,
                "cache_hit": True,
                **_compute_consensus(cached),
            }

        concurrency = self._get_concurrency()

        # ── F06: PANTHEON/KNEMON budget pre-flight ──────────────────────────
        # The README documents that consultations go through PANTHEON
        # for routing and KNEMON for budget enforcement. The check
        # itself is delegated to ``enforce_provider_budget`` which is
        # a no-op when no budget subsystem is configured. When KNEMON
        # is installed and configured, this raises
        # ``ProviderBudgetExceeded`` and refuses the call BEFORE any
        # provider dispatch — the documentation is now backed by code.
        # We only gate on the highest-weighted provider's task_kind to
        # keep the per-call overhead bounded; KNEMON's per-plan window
        # is the authoritative gate.
        try:
            anchor_provider = candidate_providers[0] if candidate_providers else None
            anchor_model = (
                self.providers[anchor_provider]["model"] if anchor_provider else ""
            )
            check = await enforce_provider_budget(
                provider=anchor_provider or "unknown",
                model=anchor_model,
                task_type=str(task_type or "consultation"),
            )
            if not check.allowed:
                logger.warning(
                    "[GRAEAE] consult refused by KNEMON budget gate: "
                    "provider=%s task=%s remaining_usd=%s reason=%s",
                    check.provider,
                    check.task_kind,
                    check.remaining_usd,
                    check.reason,
                )
                check.raise_if_denied()
            self._last_budget_check = check
        except ProviderBudgetExceeded:
            raise
        except Exception as exc:
            # Don't let a misconfigured budget checker take down the
            # whole consultation path — log + carry on, matching the
            # existing circuit-breaker tolerance.
            logger.error("[GRAEAE] budget gate raised unexpectedly: %s", exc)

        # ── Eligibility gate ─────────────────────────────────────────────────
        # A provider is skipped (not queued) if it is:
        #   • circuit-open (repeated recent failures)
        #   • rate-limited (RPM window exhausted)
        #   • saturated (all concurrency slots occupied)
        async def _check_provider_eligibility(name: str) -> tuple[str, bool]:
            if not await call_maybe_async(self._circuit_breakers.is_allowed, name):
                return name, False
            if not await call_maybe_async(self._rate_limiters.is_allowed, name):
                return name, False
            if not await call_maybe_async(concurrency.acquire, name):
                return name, False
            acquired.append(name)
            return name, True

        acquired: list[str] = []
        eligibility_tasks = [
            asyncio.create_task(_check_provider_eligibility(name))
            for name in candidate_providers
        ]
        try:
            eligibility = await asyncio.gather(*eligibility_tasks)
        except BaseException:
            for task in eligibility_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*eligibility_tasks, return_exceptions=True)
            await asyncio.gather(
                *(call_maybe_async(concurrency.release, name) for name in acquired)
            )
            raise
        active = [name for name, is_active in eligibility if is_active]
        skipped = [name for name, is_active in eligibility if not is_active]

        if skipped:
            logger.info(f"[GRAEAE] skipped providers: {skipped}")

        if not active:
            logger.error("[GRAEAE] all providers unavailable")
            return {
                "all_responses": {
                    name: _unavailable(self.providers[name]["model"])
                    for name in candidate_providers
                },
                "error": "all providers unavailable",
            }

        # ── Fan-out ──────────────────────────────────────────────────────────
        # When a selection supplied a per-provider model override, pass it
        # through as a per-task argument instead of mutating self.providers.
        # In-place mutation would race with the background reload_from_registry
        # task: a concurrent reload that lands between save-and-restore would
        # be silently undone by the finally:'s restore step. Snapshot-by-arg
        # is also re-entrant for concurrent overlapping consult() calls.
        overrides: Dict[str, Optional[str]] = {}
        if selection is not None:
            for name, override in selection.items():
                if override and name in self.providers:
                    overrides[name] = override
        tasks = [
            asyncio.create_task(
                self._call_provider_worker(
                    name,
                    prompt,
                    task_type,
                    timeout,
                    model_override=overrides.get(name),
                )
            )
            for name in active
        ]

        # iter 56 — early-return on quorum reached.
        # Old behaviour gathered ALL tasks before returning, so the slowest
        # muse (often claude-opus on long-thinking prompts) blocked
        # consensus even when 2/3 had already succeeded. Now: race the
        # tasks via asyncio.wait(FIRST_COMPLETED); once `early_return_after`
        # successful tasks are in, cancel the rest and proceed.
        #
        # Trigger only when mode is one of the quorum-style modes (auto,
        # all, external, local). `single`/`debate`/`majority` route to
        # specialised methods that wrap this call and want full results
        # — for those, early_return_after stays None and the loop drains
        # all tasks.
        early_return_after: Optional[int] = None
        early_return_grace_seconds: float = 30.0  # iter56 widen — see below
        if mode in {"auto", "all", "external", "local"} and len(tasks) >= 3:
            # 2/3 = 0.66 quorum threshold (matches route_majority default).
            # Was max(2,...) — bumped to max(3,...) so 3-of-5 lineup must
            # complete before any cancellation. Pairs with the wider
            # route_majority lineup (limit=5) so slow high-weight muses
            # like claude-opus + gpt-5.5 get to land.
            early_return_after = max(3, (len(tasks) * 2) // 3)

        results: list = [None] * len(tasks)
        try:
            if early_return_after is None:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(gathered):
                    results[i] = r
            else:
                import time as _time

                successes = 0
                remaining = set(tasks)
                task_to_idx = {t: i for i, t in enumerate(tasks)}
                quorum_met_at: Optional[float] = None
                while remaining:
                    # iter56 grace: shrink wait timeout once quorum met
                    # so we exit promptly after the grace window expires.
                    if quorum_met_at is not None:
                        grace_left = max(
                            0.1,
                            early_return_grace_seconds
                            - (_time.monotonic() - quorum_met_at),
                        )
                    else:
                        grace_left = None
                    done, remaining = await asyncio.wait(
                        remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=grace_left,
                    )
                    if not done and grace_left is not None:
                        # grace expired — cancel remaining + exit loop
                        break
                    for t in done:
                        idx = task_to_idx[t]
                        try:
                            r = t.result()
                        except BaseException as exc:
                            results[idx] = exc
                            continue
                        results[idx] = r
                        if isinstance(r, dict) and r.get("status") == "success":
                            successes += 1
                    if successes >= early_return_after and quorum_met_at is None:
                        quorum_met_at = _time.monotonic()
                        # do NOT break — keep waiting until grace expires
                        # or all tasks finish, whichever comes first.
                # Cancel any still-pending tasks. They may have partial work
                # which we drop on the floor — the cost of fastness over
                # completeness.
                for t in remaining:
                    if not t.done():
                        t.cancel()
                if remaining:
                    cancelled = await asyncio.gather(
                        *remaining,
                        return_exceptions=True,
                    )
                    for t, r in zip(remaining, cancelled):
                        idx = task_to_idx[t]
                        if results[idx] is None:
                            results[idx] = (
                                r
                                if not isinstance(r, asyncio.CancelledError)
                                else {
                                    "status": "cancelled",
                                    "response_text": "",
                                    "error": (
                                        "cancelled: quorum reached before this "
                                        "muse responded"
                                    ),
                                    "latency_ms": 0,
                                    "final_score": 0.0,
                                    "model_id": self.providers.get(active[idx], {}).get(
                                        "model", ""
                                    ),
                                }
                            )
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:

            async def _release_provider_slot(name: str) -> None:
                await call_maybe_async(concurrency.release, name)

            await asyncio.gather(*(_release_provider_slot(name) for name in active))

        all_responses: Dict = {}

        async def _record_provider_result(
            name: str,
            result: Any,
        ) -> tuple[str, Dict]:
            if isinstance(result, Exception):
                await call_maybe_async(self._circuit_breakers.record_failure, name)
                try:
                    await call_maybe_async(self._quality.record_failure, name)
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] quality failure record failed for %s: %s", name, exc
                    )
                err_msg = f"{type(result).__name__}: {str(result)[:400]}"
                logger.warning(f"[GRAEAE] muse {name} failed: {err_msg}")
                return name, {
                    "status": "error",
                    "response_text": "",
                    "error": err_msg,
                    "latency_ms": 0,
                    "model_id": self.providers[name]["model"],
                    "final_score": 0.0,
                }
            if not isinstance(result, dict):
                return name, {
                    "status": "error",
                    "response_text": "",
                    "error": f"unexpected provider result: {type(result).__name__}",
                    "latency_ms": 0,
                    "model_id": self.providers[name]["model"],
                    "final_score": 0.0,
                }
            if result.get("status") == "cancelled":
                result["final_score"] = 0.0
                return name, result
            if result.get("status") != "success":
                await call_maybe_async(self._circuit_breakers.record_failure, name)
                try:
                    await call_maybe_async(self._quality.record_failure, name)
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] quality failure record failed for %s: %s", name, exc
                    )
                result["final_score"] = 0.0
                return name, result
            await call_maybe_async(self._circuit_breakers.record_success, name)
            try:
                recorded_weight = await call_maybe_async(
                    self._quality.record_success,
                    name,
                    result.get("latency_ms", 0),
                )
            except Exception as exc:
                logger.error(
                    "[GRAEAE] quality success record failed for %s: %s", name, exc
                )
                recorded_weight = self.providers[name].get("weight", 0.0)
            if recorded_weight is None:
                try:
                    recorded_weight = await call_maybe_async(
                        self._quality.dynamic_weight,
                        name,
                    )
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] quality weight read failed for %s: %s", name, exc
                    )
                    recorded_weight = self.providers[name].get("weight", 0.0)
            result["final_score"] = recorded_weight
            # ── F06: KNEMON spend ledger ─────────────────────────────────
            # Fire-and-forget record of the priced cost + upstream usage
            # into the shared usage-ledger contract. The handler is
            # crash-safe by contract (returns SpendRecord with status)
            # so this never raises out into the consensus rollup. The
            # outcome field feeds KNEMON's own quality multiplier.
            try:
                if result.get("_accounted") and _spend_recorder is None:
                    return name, result
                spend = await record_provider_spend(
                    provider=name,
                    model=result.get("model_id", self.providers[name]["model"]),
                    task_type=str(task_type or "consultation"),
                    cost_usd=result.get("cost"),
                    usage=result.get("usage"),
                    latency_ms=int(result.get("latency_ms", 0) or 0),
                    outcome="ok",
                    tier=str(
                        self.providers[name].get("tier", "standard") or "standard"
                    ),
                )
                result["spend_record"] = spend.to_dict()
            except Exception as exc:  # last-resort safety net
                logger.error("[GRAEAE] spend record crashed for %s: %s", name, exc)
            return name, result

        recorded_responses = await asyncio.gather(
            *(
                _record_provider_result(name, result)
                for name, result in zip(active, results)
            )
        )
        all_responses.update(recorded_responses)

        for name in skipped:
            all_responses[name] = _unavailable(self.providers[name]["model"])

        # ── Cache successful result ──────────────────────────────────────────
        if any(r["status"] == "success" for r in all_responses.values()):
            try:
                await call_maybe_async(
                    self._cache.set, prompt, task_type, all_responses, cache_model
                )
            except Exception as exc:
                logger.error("[GRAEAE] response cache write failed: %s", exc)

        # ── Compute consensus fields (v3.2) ──────────────────────────────────
        # ConsultationResponse has exposed consensus_response,
        # consensus_score, winning_muse, cost, latency_ms since v3.0
        # but the engine only emitted all_responses; consultation_id
        # callers saw all five as None. Compute them here from
        # all_responses so the contract is honored instead of
        # aspirational.
        consensus = _compute_consensus(all_responses)
        return {"all_responses": all_responses, **consensus}

    async def route_single(
        self,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        selection: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict:
        """Run exactly one highest-weighted muse for the task."""
        lineup = await self._ranked_lineup(
            task_type=task_type,
            limit=1,
            selection=selection,
        )
        if not lineup:
            return {
                "all_responses": {},
                "error": "no providers available",
                **_compute_consensus({}),
            }
        return await self.consult(
            prompt,
            task_type,
            timeout=timeout,
            selection=lineup,
            mode="all",
        )

    async def route_debate(
        self,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        rounds: int = 2,
        selection: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict:
        """Run a two-round cross-muse debate and return refined responses."""
        lineup = await self._ranked_lineup(
            task_type=task_type,
            limit=3,
            selection=selection,
        )
        if not lineup:
            empty = _compute_consensus({})
            return {
                "all_responses": {},
                "round_1": {},
                "round_2": {},
                "error": "no providers available",
                **empty,
            }

        round_1 = await self.consult(
            prompt,
            task_type,
            timeout=timeout,
            selection=lineup,
            mode="all",
        )
        round_1_responses = round_1.get("all_responses", {})
        if rounds < 2:
            return {
                "all_responses": round_1_responses,
                "round_1": round_1_responses,
                "round_2": {},
                **_compute_consensus(round_1_responses),
            }

        async def _refine(
            provider_name: str, model_override: Optional[str]
        ) -> tuple[str, Dict]:
            refine_prompt = _debate_refinement_prompt(
                prompt=prompt,
                current_muse=provider_name,
                round_1_responses=round_1_responses,
            )
            refined = await self.consult(
                refine_prompt,
                task_type,
                timeout=timeout,
                selection={provider_name: model_override},
                mode="all",
            )
            response = refined.get("all_responses", {}).get(
                provider_name
            ) or _unavailable(self.providers[provider_name]["model"])
            return provider_name, response

        round_2_responses = dict(
            await asyncio.gather(
                *(
                    _refine(provider_name, model_override)
                    for provider_name, model_override in lineup.items()
                )
            )
        )

        consensus = _compute_consensus(round_2_responses)
        return {
            "all_responses": round_2_responses,
            "round_1": round_1_responses,
            "round_2": round_2_responses,
            **consensus,
            "cost": float(round_1.get("cost", 0.0) or 0.0)
            + float(consensus.get("cost", 0.0) or 0.0),
            "latency_ms": int(round_1.get("latency_ms", 0) or 0)
            + int(consensus.get("latency_ms", 0) or 0),
        }

    async def route_majority(
        self,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        quorum: float = 0.66,
        selection: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict:
        """Run up to five muses and report quorum agreement.

        iter56 widen: was limit=3 which cut off Grok-4.20 (0.86) +
        Together MiniMax-M2.7 (0.78) + NGC Kimi (0.50) before they
        could participate. limit=5 includes the next two highest-
        weight muses for wider editorial coverage. Caller can still
        force a smaller lineup via selection=.
        """
        lineup = await self._ranked_lineup(
            task_type=task_type,
            limit=5,
            selection=selection,
        )
        result = await self.consult(
            prompt,
            task_type,
            timeout=timeout,
            selection=lineup or selection,
            mode="all",
        )
        all_responses = result.get("all_responses", {})
        quorum_result = _compute_quorum(all_responses, quorum)
        consensus = _compute_consensus(all_responses)
        consensus["consensus_score"] = quorum_result["consensus_score"]
        if quorum_result["quorum_reached"] and quorum_result["quorum_muses"]:
            winning_muse = max(
                quorum_result["quorum_muses"],
                key=lambda name: all_responses[name].get("final_score", 0.0),
            )
            consensus["winning_muse"] = winning_muse
            consensus["consensus_response"] = all_responses[winning_muse].get(
                "response_text", ""
            )

        return {
            **result,
            **consensus,
            "quorum_reached": quorum_result["quorum_reached"],
            "quorum_threshold": quorum,
            "agreement_metric": "lexical_complete_link",
            "similarity_pairs": quorum_result["similarity_pairs"],
            "contradiction_pairs": quorum_result["contradiction_pairs"],
        }

    async def _ranked_lineup(
        self,
        task_type: str,
        limit: int,
        selection: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Optional[str]]:
        """Pick the highest-weighted providers, preferring model_registry weights."""
        del task_type  # model_registry stores graeae_weight globally today.
        candidates = [
            name
            for name in (
                selection.keys() if selection is not None else self.providers.keys()
            )
            if name in self.providers
        ]
        if not candidates:
            return {}

        ranked: list[tuple[str, Optional[str], float]] = []
        cache_key = tuple(candidates)
        ranked_lineup_cache = getattr(self, "_ranked_lineup_cache", None)
        if ranked_lineup_cache is None:
            ranked_lineup_cache = _RankedLineupCache(_ranked_lineup_ttl_seconds())
            self._ranked_lineup_cache = ranked_lineup_cache

        try:
            import mnemos.core.lifecycle as _lc

            if _lc._pool:
                registry_ranked = ranked_lineup_cache.get(cache_key)
                if registry_ranked is None:
                    registry_names = [
                        _REGISTRY_MAP.get(name, {}).get("registry_provider", name)
                        for name in candidates
                    ]
                    registry_to_graeae = {
                        cfg["registry_provider"]: name
                        for name, cfg in _REGISTRY_MAP.items()
                    }
                    async with _lc._pool.acquire() as conn:
                        rows = await conn.fetch(
                            """
                            SELECT provider, model_id, graeae_weight
                            FROM model_registry
                            WHERE provider = ANY($1::text[])
                              AND available = true
                              AND deprecated = false
                            ORDER BY graeae_weight DESC NULLS LAST
                            """,
                            registry_names,
                        )
                    seen: set[str] = set()
                    registry_ranked = []
                    for row in rows:
                        registry_provider = _row_get(row, "provider")
                        name = registry_to_graeae.get(
                            registry_provider, registry_provider
                        )
                        if name not in candidates or name in seen:
                            continue
                        registry_ranked.append(
                            (
                                name,
                                _row_get(row, "model_id"),
                                float(
                                    _row_get(
                                        row,
                                        "graeae_weight",
                                        self.providers[name].get("weight", 0.0),
                                    )
                                    or 0.0
                                ),
                            )
                        )
                        seen.add(name)
                    ranked_lineup_cache.set(cache_key, registry_ranked)
                for name, registry_model, weight in registry_ranked:
                    model_override = selection.get(name) if selection else None
                    ranked.append((name, model_override or registry_model, weight))
        except Exception as exc:
            logger.debug("[GRAEAE] model_registry ranking unavailable: %s", exc)

        ranked_names = {name for name, _, _ in ranked}
        for name in candidates:
            if name in ranked_names:
                continue
            ranked.append(
                (
                    name,
                    selection.get(name) if selection else None,
                    float(self.providers[name].get("weight", 0.0) or 0.0),
                )
            )
        ranked.sort(key=lambda item: item[2], reverse=True)
        return {name: model_override for name, model_override, _ in ranked[:limit]}

    async def route(
        self,
        provider: str,
        model: str,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> Dict:
        """Single-provider pass-through — consensus skipped, eligibility
        gates applied.

        Used by MNEMOS gateway (`/v1/chat/completions`) for explicit
        model selection. Before v3.2 this path deliberately skipped
        the reliability stack "caller responsible for load management";
        operators pointed out that the gateway was effectively the
        weakest surface of the service because openai_compat did not
        actually implement any load management. v3.2 closes that gap:
        the circuit breaker, rate limiter, and concurrency guard are
        applied here exactly as they are in consult(), so one
        misbehaving provider can't take down the gateway while
        consultations keep working.

        Args:
            provider: Provider name (must exist in self.providers)
            model: Override model name (optional; uses provider config if None)
            prompt: Query text
            task_type: Task type for logging/tracking
            timeout: Request timeout in seconds
            generation_params: OpenAI generation controls (temperature,
                max_tokens, top_p) to map per provider
            request_params: OpenAI-compatible request fields already
                validated by the gateway (tools, response_format, stop, etc.)
            messages: Full chat messages to preserve system/history and
                multimodal content for adapters that support it

        Returns:
            Dict with status, response_text, latency_ms, model_id, error
        """
        if provider not in self.providers:
            logger.warning(
                f"[GRAEAE] unknown provider '{provider}' — returning unavailable"
            )
            return _unavailable(
                model or provider,
                error=f"provider '{provider}' not registered in this deployment",
            )

        provider_config = dict(self.providers[provider])
        if model:
            provider_config["model"] = model

        # Key-missing is a common failure and silently produces a 401/403
        # upstream with no visible reason. Pre-check the key and emit a
        # targeted error so operators don't have to tail debug logs.
        api_key = get_key(provider_config["key_name"])
        if not api_key:
            logger.error(
                "[GRAEAE] route(%s) failed: missing api_key (key_name=%s) — "
                "set the %s environment variable or add the key to the "
                "Provider Registry File (MNEMOS_KEYS_PATH / "
                "~/.config/mnemos/api_keys.json)",
                provider,
                provider_config["key_name"],
                _env_var_hint(provider_config["key_name"]),
            )
            return _unavailable(
                provider_config["model"],
                error=(
                    f"missing api_key for provider '{provider}' "
                    f"(key_name={provider_config['key_name']})"
                ),
            )

        # v3.2 reliability gate: circuit-breaker → rate-limiter →
        # concurrency. Mirrors the consult() eligibility loop so
        # gateway traffic is first-class not second-class.
        if not await call_maybe_async(self._circuit_breakers.is_allowed, provider):
            logger.info("[GRAEAE] route(%s) refused: circuit open", provider)
            return _unavailable(
                provider_config["model"],
                error=f"provider '{provider}' circuit open",
            )
        if not await call_maybe_async(self._rate_limiters.is_allowed, provider):
            logger.info("[GRAEAE] route(%s) refused: rate limited", provider)
            return _unavailable(
                provider_config["model"],
                error=f"provider '{provider}' rate-limited",
            )

        # ── F06: PANTHEON/KNEMON budget pre-flight (gateway path) ────────
        # The README documents that every GRAEAE provider call goes
        # through PANTHEON's routing + KNEMON's dollar budget. Before
        # this fix the gateway path invoked the adapter directly,
        # bypassing both — the SAME underbilling hole as the
        # consultation fan-out, just on the gateway surface. Mirror
        # the consult() gate here so a single-provider gateway call
        # is also refused when KNEMON's per-plan cap is exhausted.
        # The check is a no-op when no budget subsystem is installed.
        # Run BEFORE concurrency.acquire so a denied call does not
        # hold a slot — same order as the consult() flow.
        try:
            check = await enforce_provider_budget(
                provider=provider,
                model=model or provider_config["model"],
                task_type=str(task_type or "chat"),
            )
            if not check.allowed:
                logger.warning(
                    "[GRAEAE] route(%s) refused by KNEMON budget gate: "
                    "remaining_usd=%s reason=%s",
                    provider,
                    check.remaining_usd,
                    check.reason,
                )
                check.raise_if_denied()
        except ProviderBudgetExceeded:
            raise
        except Exception as exc:
            # Don't let a misconfigured budget checker take down the
            # gateway path — log + carry on, matching the
            # consult() tolerance.
            logger.error(
                "[GRAEAE] route(%s) budget gate raised unexpectedly: %s", provider, exc
            )

        concurrency = self._get_concurrency()
        if not await call_maybe_async(concurrency.acquire, provider):
            logger.info("[GRAEAE] route(%s) refused: concurrency saturated", provider)
            return _unavailable(
                provider_config["model"],
                error=f"provider '{provider}' concurrency saturated",
            )

        try:
            try:
                # Pass `model` through as model_override so the gateway's
                # per-call selection (e.g. /v1/chat/completions with
                # model="claude-opus-4-7") actually reaches dispatch
                # instead of being silently overwritten by whatever
                # self.providers[provider]["model"] currently holds.
                query_kwargs: Dict[str, Any] = {"model_override": model}
                if generation_params:
                    query_kwargs["generation_params"] = generation_params
                if request_params:
                    query_kwargs["request_params"] = request_params
                if messages is not None:
                    query_kwargs["messages"] = messages
                result = await self._call_provider_worker(
                    provider, prompt, task_type, timeout, **query_kwargs
                )
            except Exception as e:
                # Record the failure against the breaker so repeated
                # gateway-path failures actually trip it, and quality
                # tracker so the weight reflects reality.
                await call_maybe_async(self._circuit_breakers.record_failure, provider)
                try:
                    await call_maybe_async(self._quality.record_failure, provider)
                except Exception as quality_exc:
                    logger.error(
                        "[GRAEAE] quality failure record failed for route(%s): %s",
                        provider,
                        quality_exc,
                    )
                logger.error(f"[GRAEAE] route({provider}) failed: {e}", exc_info=True)
                return _unavailable(
                    provider_config["model"],
                    error=f"{type(e).__name__}: {e}",
                )
            # Success path — credit the breaker + quality tracker so
            # the gateway's successes count toward reopening a
            # half-open circuit, not just consultations' successes.
            await call_maybe_async(self._circuit_breakers.record_success, provider)
            try:
                await call_maybe_async(
                    self._quality.record_success, provider, result.get("latency_ms", 0)
                )
            except Exception as exc:
                logger.error(
                    "[GRAEAE] quality record failed after successful route(%s): %s",
                    provider,
                    exc,
                )
            # ── F06: KNEMON spend ledger (gateway path) ─────────────────
            # The gateway path is the OTHER surface that bypasses the
            # KNEMON ledger in the F06 review. Record here too so a
            # single-provider gateway call (e.g. /v1/chat/completions)
            # produces the same priced spend record as a consultation
            # fan-out. Failure here must NOT raise out into the gateway
            # response — degrade to log + continue.
            if not result.get("_accounted") or _spend_recorder is not None:
                try:
                    spend = await record_provider_spend(
                        provider=provider,
                        model=result.get("model_id", provider_config["model"]),
                        task_type=str(task_type or "chat"),
                        cost_usd=result.get("cost"),
                        usage=result.get("usage"),
                        latency_ms=int(result.get("latency_ms", 0) or 0),
                        outcome=(
                            "ok"
                            if result.get("status") == "success"
                            else str(result.get("status") or "err")
                        ),
                        tier=str(provider_config.get("tier", "standard") or "standard"),
                    )
                    result["spend_record"] = spend.to_dict()
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] spend record crashed in route(%s): %s",
                        provider,
                        exc,
                    )
            logger.debug(
                f"[GRAEAE] route({provider}, {model or 'default'}) → {result['status']}"
            )
            return result
        finally:
            await call_maybe_async(concurrency.release, provider)

    async def route_stream(
        self,
        provider: str,
        model: str,
        prompt: str,
        task_type: str = "reasoning",
        timeout: int = 180,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Single-provider streaming route.

        OpenAI-style providers use native SSE streaming. Other adapters fall
        back to one non-streaming provider call emitted as a single delta, so
        callers still get a faithful OpenAI SSE envelope without silently
        ignoring stream=True.
        """
        if provider not in self.providers:
            raise RuntimeError(
                f"provider '{provider}' not registered in this deployment"
            )

        provider_config = dict(self.providers[provider])
        if model:
            provider_config["model"] = model

        api_key = get_key(provider_config["key_name"])
        if not api_key:
            raise RuntimeError(
                f"missing api_key for provider '{provider}' "
                f"(key_name={provider_config['key_name']})"
            )

        if not await call_maybe_async(self._circuit_breakers.is_allowed, provider):
            raise RuntimeError(f"provider '{provider}' circuit open")
        if not await call_maybe_async(self._rate_limiters.is_allowed, provider):
            raise RuntimeError(f"provider '{provider}' rate-limited")

        # ── F06: PANTHEON/KNEMON budget pre-flight (streaming gateway) ───
        # Streaming gateway traffic is the third provider-invocation
        # surface in GRAEAE — same documented contract as consult()
        # and route(): refuse the call BEFORE any chunk is generated
        # when KNEMON's per-plan dollar cap is exhausted. No-op when
        # no budget subsystem is installed (single-process / test
        # deployments keep working). Run BEFORE concurrency.acquire
        # so a denied call does not hold a slot.
        try:
            check = await enforce_provider_budget(
                provider=provider,
                model=model or provider_config["model"],
                task_type=str(task_type or "stream"),
            )
            if not check.allowed:
                logger.warning(
                    "[GRAEAE] route_stream(%s) refused by KNEMON budget gate: "
                    "remaining_usd=%s reason=%s",
                    provider,
                    check.remaining_usd,
                    check.reason,
                )
                check.raise_if_denied()
        except ProviderBudgetExceeded:
            raise
        except Exception as exc:
            logger.error(
                "[GRAEAE] route_stream(%s) budget gate raised unexpectedly: %s",
                provider,
                exc,
            )

        concurrency = self._get_concurrency()
        if not await call_maybe_async(concurrency.acquire, provider):
            raise RuntimeError(f"provider '{provider}' concurrency saturated")

        try:
            try:
                provider_cfg = dict(self.providers[provider])
                if model:
                    provider_cfg["model"] = model
                    if provider_cfg.get("api") == "gemini":
                        provider_cfg["url"] = (
                            f"https://generativelanguage.googleapis.com/v1beta/"
                            f"models/{model}:generateContent"
                        )

                if provider_cfg.get("api") == "openai":
                    async with aclosing(
                        self._stream_openai_compatible(
                            provider_cfg,
                            prompt,
                            timeout,
                            generation_params=generation_params,
                            request_params=request_params,
                            messages=messages,
                        )
                    ) as upstream:
                        async for chunk in upstream:
                            yield chunk
                else:
                    result = await self._call_provider_worker(
                        provider,
                        prompt,
                        task_type,
                        timeout,
                        model_override=model,
                        generation_params=generation_params,
                        request_params=request_params,
                        messages=messages,
                    )
                    for choice in result.get("choices") or []:
                        msg = choice.get("message") or {}
                        index = choice.get("index", 0)
                        if msg.get("role"):
                            yield {"index": index, "role": msg["role"]}
                        content = msg.get("content")
                        if content:
                            yield {"index": index, "content": content}
                        if msg.get("tool_calls"):
                            yield {"index": index, "tool_calls": msg["tool_calls"]}
                        finish_reason = (
                            choice.get("finish_reason")
                            or result.get("finish_reason")
                            or "stop"
                        )
                        if provider_cfg.get("api") == "gemini":
                            finish_reason = _normalize_gemini_finish_reason(
                                finish_reason
                            )
                        yield {
                            "index": index,
                            "finish_reason": finish_reason,
                        }
                    if not result.get("choices"):
                        yield {"index": 0, "content": result.get("response_text", "")}
                        finish_reason = result.get("finish_reason") or "stop"
                        if provider_cfg.get("api") == "gemini":
                            finish_reason = _normalize_gemini_finish_reason(
                                finish_reason
                            )
                        yield {"index": 0, "finish_reason": finish_reason}
            except Exception:
                await call_maybe_async(self._circuit_breakers.record_failure, provider)
                try:
                    await call_maybe_async(self._quality.record_failure, provider)
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] quality failure record failed for stream(%s): %s",
                        provider,
                        exc,
                    )
                raise
            else:
                await call_maybe_async(self._circuit_breakers.record_success, provider)
                try:
                    await call_maybe_async(self._quality.record_success, provider, 0)
                except Exception as exc:
                    logger.error(
                        "[GRAEAE] quality record failed after successful stream(%s): %s",
                        provider,
                        exc,
                    )
                # F06: streaming gateway path also feeds the KNEMON
                # spend ledger. Streamers don't have a final usage
                # block at this layer (we'd need to chunk the SSE and
                # accumulate), so we record what we have: model id
                # + latency + outcome. The KNEMON-priced cost is
                # deliberately left None rather than coerced to 0.0
                # — the stream path will fill it from the next
                # client-prompt batch via the upstream ledger when
                # the operator enables chunk-level accounting.
                if _spend_recorder is not None:
                    try:
                        spend = await record_provider_spend(
                            provider=provider,
                            model=model or provider_config["model"],
                            task_type=str(task_type or "stream"),
                            cost_usd=None,
                            usage=None,
                            latency_ms=0,
                            outcome="ok",
                            tier=str(
                                provider_config.get("tier", "standard") or "standard"
                            ),
                        )
                        logger.debug(
                            "[GRAEAE] stream spend recorded for %s status=%s",
                            provider,
                            spend.status,
                        )
                    except Exception as exc:
                        logger.error(
                            "[GRAEAE] stream spend record crashed for %s: %s",
                            provider,
                            exc,
                        )
        finally:
            await call_maybe_async(concurrency.release, provider)

    async def _call_provider_worker(
        self,
        provider_name: str,
        prompt: str,
        task_type: str,
        timeout: int,
        model_override: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> Dict:
        generation_params = _bounded_generation_params(generation_params)
        request_params = _bounded_request_params(request_params)
        from mnemos.domain.graeae.accounting import begin_attempt

        config = self.providers[provider_name]
        attempt = await begin_attempt(
            provider_name,
            config,
            model_override or config["model"],
            prompt,
            generation_params,
            request_params,
            messages,
            task_type,
            timeout,
        )
        outcome = "err"
        try:
            response = await self.provider_worker(
                ProviderQueryRequest(
                    provider=provider_name,
                    model=model_override,
                    messages=messages,
                    params={
                        "prompt": prompt,
                        "task_type": task_type,
                        "timeout": timeout,
                        "generation_params": generation_params,
                        "request_params": request_params,
                    },
                )
            )
            result = self._provider_worker_payload(response)
            attempt.observe(result)
            outcome = "ok" if result.get("status") == "success" else "err"
            result["_accounted"] = True
            return result
        finally:
            await attempt.finish(outcome)

    # #189: removed `_query_provider` — thin pass-through wrapper
    # over `_call_provider_worker` with no callers. The 3 real
    # call sites (lines 695, 1046, 1132) invoke
    # `_call_provider_worker` directly. Stale doc refs at
    # `_load_providers` and `_probe_model` updated to point at
    # `_call_provider_worker` instead.

    def _provider_worker_payload(self, response: ProviderQueryResponse) -> Dict:
        return response.raw_provider_payload

    def _openai_payload(
        self,
        provider: Dict,
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        generation_params = generation_params or {}
        request_params = request_params or {}

        is_gpt5 = provider["model"].startswith("gpt-5")
        tokens_key = "max_completion_tokens" if is_gpt5 else "max_tokens"
        payload: Dict[str, Any] = {
            "model": provider["model"],
            "messages": _openai_messages(messages, prompt),
            tokens_key: generation_params.get("max_tokens", 4096),
        }
        if "temperature" in generation_params:
            payload["temperature"] = generation_params["temperature"]
        elif not is_gpt5:
            payload["temperature"] = 0.7
        if "top_p" in generation_params:
            payload["top_p"] = generation_params["top_p"]
        for field in (
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "user",
        ):
            if field in request_params:
                payload[field] = request_params[field]
        if stream:
            payload["stream"] = True
        return payload

    async def _query_openai_compatible(
        self,
        provider: Dict,
        prompt: str,
        timeout: int,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> Dict:
        """Query OpenAI-compatible APIs (Perplexity, Groq, xAI, OpenAI)."""
        api_key = get_key(provider["key_name"])
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._openai_payload(
            provider,
            prompt,
            generation_params=generation_params,
            request_params=request_params,
            messages=messages,
        )
        client = await self._get_client()
        resp = await client.post(
            provider["url"], json=payload, headers=headers, timeout=timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices in response: {data}")
        normalized_choices = []
        for i, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            message = message if isinstance(message, dict) else {}
            normalized_choices.append(
                {
                    "index": choice.get("index", i),
                    "message": message,
                    "finish_reason": choice.get("finish_reason") or "stop",
                }
            )
        if not normalized_choices:
            raise RuntimeError(f"No valid choices in response: {data}")
        first_message = normalized_choices[0].get("message")
        # ── F06 ────────────────────────────────────────────────────────────
        # Preserve upstream usage + cost end-to-end. OpenAI's
        # `/v1/chat/completions` returns a top-level ``usage`` block
        # (prompt_tokens / completion_tokens / total_tokens, plus
        # reasoning_tokens on o-series) and a numeric total cost is
        # NOT in the wire payload — it has to come from the
        # downstream KNEMON pricing layer. Capture usage here so the
        # consensus layer never has to treat a real response as
        # "no cost, no tokens" by default. ``cost`` stays ``None``
        # until KNEMON's pricing step populates it; that propagates
        # as an explicit ``None`` to the consensus rollup rather than
        # a silent zero.
        usage_block = data.get("usage")
        usage_dict: Optional[Dict[str, Any]]
        if isinstance(usage_block, dict):
            usage_dict = dict(usage_block)
        else:
            usage_dict = None
        return {
            "status": "success",
            "response_text": _chat_message_content_text(first_message),
            "latency_ms": 0,
            "model_id": provider["model"],
            "choices": normalized_choices,
            "usage": usage_dict,
            # Cost is intentionally left as None here; the KNEMON-costed
            # spend path populates it before consensus rollup. Setting
            # a default of 0.0 here would re-introduce the F06 silent-
            # zero bug downstream.
            "cost": None,
        }

    async def _stream_openai_compatible(
        self,
        provider: Dict,
        prompt: str,
        timeout: int,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        api_key = get_key(provider["key_name"])
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._openai_payload(
            provider,
            prompt,
            generation_params=generation_params,
            request_params=request_params,
            messages=messages,
            stream=True,
        )
        from mnemos.domain.graeae.accounting import begin_attempt

        provider_name = next(
            (
                name
                for name, cfg in self.providers.items()
                if cfg.get("key_name") == provider.get("key_name")
                and cfg.get("url") == provider.get("url")
            ),
            str(provider.get("name") or provider.get("api") or "openai"),
        )
        attempt = await begin_attempt(
            provider_name,
            provider,
            provider["model"],
            prompt,
            generation_params,
            request_params,
            messages,
            "stream",
            timeout,
        )
        outcome = "err"
        try:
            client = await self._get_client()
            async with client.stream(
                "POST", provider["url"], json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"HTTP {resp.status_code}: {body[:200]!r}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug(
                            "[GRAEAE] skipped non-JSON stream frame: %s", raw[:120]
                        )
                        continue
                    if isinstance(data, dict) and "error" in data:
                        raise _provider_stream_error(data)
                    if not isinstance(data, dict):
                        logger.debug(
                            "[GRAEAE] skipped non-object stream frame: %s", raw[:120]
                        )
                        continue
                    attempt.observe(data)
                    if data.get("usage"):
                        yield {"usage": data["usage"]}
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        chunk = {"index": choice.get("index", 0)}
                        if delta.get("role"):
                            chunk["role"] = delta["role"]
                        if delta.get("content") is not None:
                            chunk["content"] = delta["content"]
                        if delta.get("tool_calls"):
                            chunk["tool_calls"] = delta["tool_calls"]
                        if choice.get("finish_reason") is not None:
                            chunk["finish_reason"] = choice["finish_reason"]
                        if len(chunk) > 1:
                            yield chunk
            outcome = "ok"
        finally:
            await attempt.finish(outcome)

    async def _query_anthropic(
        self,
        provider: Dict,
        prompt: str,
        timeout: int,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> Dict:
        """Query Anthropic Claude API."""
        generation_params = generation_params or {}
        request_params = request_params or {}
        api_key = get_key(provider["key_name"])
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system, anthropic_messages = _anthropic_messages(messages, prompt)
        payload = {
            "model": provider["model"],
            "max_tokens": generation_params.get("max_tokens", 4096),
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if "temperature" in generation_params:
            payload["temperature"] = generation_params["temperature"]
        if "top_p" in generation_params:
            payload["top_p"] = generation_params["top_p"]
        if "stop" in request_params:
            payload["stop_sequences"] = _as_list(request_params["stop"])
        if "tools" in request_params:
            payload["tools"] = _anthropic_tools(request_params["tools"])
        if "tool_choice" in request_params:
            payload["tool_choice"] = _anthropic_tool_choice(
                request_params["tool_choice"]
            )
        client = await self._get_client()
        resp = await client.post(
            provider["url"], json=payload, headers=headers, timeout=timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
        tool_calls = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(
                                block.get("input", {}), separators=(",", ":")
                            ),
                        },
                    }
                )
        message: Dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        # ── F06 ────────────────────────────────────────────────────────────
        # Anthropic's Messages API returns a top-level ``usage`` block
        # (input_tokens / output_tokens / cache_creation_input_tokens /
        # cache_read_input_tokens). Capture it verbatim so the cost
        # pipeline can price cache reads separately and the consensus
        # layer never has to treat Anthropic responses as zero-usage.
        # See `_query_openai_compatible` for the cost-stays-None rationale.
        anthropic_usage = data.get("usage")
        usage_dict: Optional[Dict[str, Any]]
        if isinstance(anthropic_usage, dict):
            usage_dict = {
                k: anthropic_usage.get(k)
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "tokens_in",
                    "tokens_out",
                )
                if k in anthropic_usage
            }
        else:
            usage_dict = None
        return {
            "status": "success",
            "response_text": text,
            "latency_ms": 0,
            "model_id": provider["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": _anthropic_finish_reason(data.get("stop_reason")),
                }
            ],
            "usage": usage_dict,
            "cost": None,
        }

    async def _query_gemini(
        self,
        provider: Dict,
        prompt: str,
        timeout: int,
        generation_params: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        messages: Optional[list[dict]] = None,
    ) -> Dict:
        """Query Google Gemini API."""
        generation_params = generation_params or {}
        request_params = request_params or {}
        api_key = get_key(provider["key_name"])
        headers = {"x-goog-api-key": api_key}
        system_instruction, contents = _gemini_contents(messages, prompt)
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": generation_params.get("max_tokens", 4096),
            "temperature": generation_params.get("temperature", 0.7),
        }
        if "top_p" in generation_params:
            generation_config["topP"] = generation_params["top_p"]
        if "stop" in request_params:
            generation_config["stopSequences"] = _as_list(request_params["stop"])
        if request_params.get("n") not in (None, 1):
            generation_config["candidateCount"] = request_params["n"]
        if "presence_penalty" in request_params:
            generation_config["presencePenalty"] = request_params["presence_penalty"]
        if "frequency_penalty" in request_params:
            generation_config["frequencyPenalty"] = request_params["frequency_penalty"]
        response_format = request_params.get("response_format")
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ):
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        client = await self._get_client()
        resp = await client.post(
            provider["url"], headers=headers, json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in response: {data}")
        choices = []
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            content = content if isinstance(content, dict) else {}
            parts = content.get("parts")
            parts = parts if isinstance(parts, list) else []
            text_parts = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                value = part.get("text")
                if value is not None:
                    text_parts.append(str(value))
            text = "".join(text_parts)
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text if text else None},
                    "finish_reason": _normalize_gemini_finish_reason(
                        candidate.get("finishReason")
                    ),
                }
            )
        text = _chat_message_content_text(
            choices[0].get("message") if choices else None
        )
        # ── F06 ────────────────────────────────────────────────────────────
        # Gemini's `generateContent` returns a separate ``usageMetadata``
        # block (promptTokenCount / candidatesTokenCount /
        # thoughtsTokenCount / cachedContentTokenCount / totalTokenCount).
        # Map it into the same canonical ``usage`` shape as OpenAI /
        # Anthropic so downstream code has one schema. ``cost`` stays
        # None — KNEMON prices it after the call returns. See the
        # `_query_openai_compatible` comment for the silent-zero rationale.
        gemini_usage = data.get("usageMetadata")
        usage_dict: Optional[Dict[str, Any]]
        if isinstance(gemini_usage, dict):
            tokens_in = gemini_usage.get("promptTokenCount") or gemini_usage.get(
                "tokens_in"
            )
            tokens_out = gemini_usage.get("candidatesTokenCount") or gemini_usage.get(
                "tokens_out"
            )
            reasoning = gemini_usage.get("thoughtsTokenCount") or gemini_usage.get(
                "tokens_reasoning"
            )
            cached = gemini_usage.get("cachedContentTokenCount") or gemini_usage.get(
                "cache_read_input_tokens"
            )
            total = gemini_usage.get("totalTokenCount") or gemini_usage.get(
                "total_tokens"
            )
            usage_dict = {
                k: v
                for k, v in (
                    ("prompt_tokens", tokens_in),
                    ("completion_tokens", tokens_out),
                    ("reasoning_tokens", reasoning),
                    ("cache_read_input_tokens", cached),
                    ("total_tokens", total),
                )
                if v is not None
            }
            if not usage_dict:
                usage_dict = None
        else:
            usage_dict = None
        return {
            "status": "success",
            "response_text": text,
            "latency_ms": 0,
            "model_id": provider["model"],
            "choices": choices,
            "usage": usage_dict,
            "cost": None,
        }

    def provider_status(self) -> Dict:
        """Circuit breaker, concurrency, rate limiter, quality, and cache stats."""
        status = {
            "circuit_breakers": self._circuit_breakers.status(),
            "rate_limiters": self._rate_limiters.status(),
            "quality": self._quality.status(),
            "cache": self._cache.stats(),
        }
        quality_backend_status = getattr(self._quality, "backend_status", None)
        if quality_backend_status is not None:
            status["quality_backend"] = quality_backend_status()
        if self._concurrency:
            status["concurrency"] = self._concurrency.status()
        return status


def _compute_consensus(all_responses: Dict[str, Dict]) -> Dict:
    """Roll up per-provider responses into consensus fields.

    Emits:
      consensus_response — text of the highest-scoring successful
                           provider (winning muse). Empty string if
                           no provider succeeded.
      consensus_score    — the winner's final_score, or 0.0.

                           v4.4.0 (F11) caveat for callers that
                           surface this value: when the consultation
                           was run via ``route_majority``,
                           ``consult`` may overwrite this field with
                           the textual-agreement score from
                           :func:`_compute_quorum`. That overwrite is
                           a LEXICAL/SURFACE-LEVEL measure, not a
                           semantic one — high textual overlap does
                           not imply high semantic agreement (see
                           :func:`_detect_negation_contradiction`).
                           Logs and operator dashboards MUST label
                           this field "textual agreement" or
                           "lexical similarity", never "consensus" or
                           "correctness".
      winning_muse       — the provider name of the winner, or None.
      cost               — sum of per-provider KNOWN cost fields. When
                           ANY successful provider's cost is absent
                           (``None``) or non-numeric, this is returned
                           as ``None`` rather than silently coerced
                           to zero — that was the F06 silent-
                           underbilling bug: a real $0.25 call
                           reported as $0.00 because the adapter
                           dropped cost data before consensus.
                           Callers MUST treat a ``None`` cost as a
                           partial / unknown-cost state, not as a
                           real zero. ``cost_known`` and
                           ``cost_unknown_providers`` accompany
                           it so the surrounding record layer can
                           surface the gap explicitly.
      latency_ms         — max latency across providers (parallel
                           fan-out: wall-clock to all_responses is
                           dominated by the slowest successful call).
      usage              — merged usage counters across the winners
                           of the consensus (input/output/reasoning/
                           cache_read tokens). ``None`` when the
                           providers that landed supplied no usage
                           data; sum is partial in that case.

    Contract: returns ALL keys even when there's no winner so callers
    never have to check for "field present" vs "field set". A
    no-winner consultation has consensus_response="", consensus_score=
    0.0, winning_muse=None, cost=None, latency_ms=0,
    cost_known=False, cost_unknown_providers=[], usage=None.
    """
    successes = [
        (name, resp)
        for name, resp in all_responses.items()
        if resp.get("status") == "success"
    ]
    if successes:
        winner_name, winner_resp = max(
            successes, key=lambda kv: kv[1].get("final_score", 0.0)
        )
    else:
        winner_name, winner_resp = None, None

    # ── F06 cost rollup ──────────────────────────────────────────────────
    # Iterate per-provider cost explicitly so the absence of cost is
    # NEVER silently coerced to 0. Use `is None` for the sentinel,
    # never `or 0`/`or 0.0` — that was the exact pattern that masked
    # the original underbilling. Keep the known partial-sum separate
    # so the caller can decide whether to log, surface a warning,
    # or mark the consultation as "needs cost reconciliation".
    known_costs: list[float] = []
    unknown_cost_providers: list[str] = []
    for name, resp in all_responses.items():
        if resp.get("status") != "success":
            continue
        c = resp.get("cost")
        if c is None:
            # Provider returned successfully but did not surface a
            # numeric cost — must remain unknown, not zero.
            unknown_cost_providers.append(name)
            continue
        if isinstance(c, bool):
            # bool is a subclass of int — refuse silently adding
            # True/False as if it were a real cost. A misconfigured
            # adapter returning ``True`` would otherwise sum as 1.0
            # per provider, which is exactly the kind of silent-
            # silent we want to surface instead.
            unknown_cost_providers.append(name)
            continue
        if isinstance(c, (int, float)):
            known_costs.append(float(c))
            continue
        # Anything that isn't numeric and isn't None is unknown.
        unknown_cost_providers.append(name)

    if unknown_cost_providers:
        total_cost: Optional[float] = None
        cost_known = False
    elif not all_responses:
        # No providers at all (caller passed an empty dict) — there
        # is no spend to report, so ``cost`` is unknown, not zero.
        # A real ``cost=0.0`` only happens when every successful
        # provider actually returned a literal 0.0 (e.g. a free
        # tier or a $0 promo). ``cost=0.0`` must NEVER be the
        # default for "we don't know" or "nothing happened".
        total_cost = None
        cost_known = False
    else:
        total_cost = sum(known_costs)
        cost_known = True

    latencies = [int(resp.get("latency_ms", 0) or 0) for resp in all_responses.values()]
    max_latency = max(latencies) if latencies else 0

    # ── F06 usage rollup ─────────────────────────────────────────────────
    # Sum the per-provider `usage` block over successful responses.
    # Per-field handling: a token category (tokens_in, tokens_out,
    # reasoning, cache_read) is reported as the sum across providers
    # that have it; if NO provider reports a given field, it is
    # omitted from the rolled-up usage dict rather than reported as
    # zero.
    usage_totals: Dict[str, int] = {}
    for resp in all_responses.values():
        if resp.get("status") != "success":
            continue
        u = resp.get("usage")
        if not isinstance(u, dict):
            continue
        for key, raw in u.items():
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + int(raw)

    return {
        "consensus_response": winner_resp.get("response_text", "")
        if winner_resp
        else "",
        "consensus_score": float(winner_resp.get("final_score", 0.0))
        if winner_resp
        else 0.0,
        "winning_muse": winner_name,
        # F06: `cost` is None when ANY provider's cost is unknown.
        # `cost_known` is implicitly True iff all known.
        "cost": total_cost,
        "cost_known": cost_known,
        "cost_unknown_providers": list(unknown_cost_providers),
        "usage": usage_totals or None,
        "latency_ms": max_latency,
    }


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _debate_refinement_prompt(
    prompt: str,
    current_muse: str,
    round_1_responses: Dict[str, Dict],
) -> str:
    others: list[str] = []
    for name, response in round_1_responses.items():
        if name == current_muse:
            continue
        text = str(response.get("response_text") or "").strip()
        if text:
            others.append(f"{name}: {text}")
    context = (
        "\n\n".join(others) if others else "No other round-1 responses were available."
    )
    context = context[: _debate_peer_context_chars()]
    return (
        "Original consultation prompt:\n"
        f"{prompt}\n\n"
        "Round 1 responses from the other muses:\n"
        f"{context}\n\n"
        "Refine your answer. Address useful objections, keep what still holds, "
        "and be explicit where you disagree."
    )


def _text_similarity(left: str, right: str) -> float:
    """Char-level SequenceMatcher ratio on whitespace-normalized lowercased text.

    F11 (2026-09-14): this score measures TEXTUAL AGREEMENT, not semantic
    consensus. Two texts can score high while meaning the opposite — the
    classic example is ``"the answer is safe"`` vs ``"the answer is not
    safe"``, which SequenceMatcher rates ~0.90 because the characters
    overlap heavily but which any human would call a contradiction. See
    :func:`_detect_negation_contradiction` for the explicit modifier
    callers should compose on top of this score before reporting a
    "consensus reached" verdict. Callers that surface this value to
    operators / logs / dashboards MUST describe it as "textual
    agreement", not "consensus" or "correctness agreement".
    """
    left = " ".join(left.lower().split())
    right = " ".join(right.lower().split())
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


# Sentence-level content words the cheap negation-contradiction check
# considers "potentially-shared". Common English stopwords and one-letter
# tokens are excluded so function words don't drive a false contradiction
# flag (e.g. "the cat is not here" vs "the cat is here" would still be
# caught on "here", which IS a real content word, but "it is not X"
# vs "it is X" would not be picked up on "it" / "is").
_NEGATION_TOKENS = {
    "not",
    "n't",
    "never",
    "no",
    "without",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "wouldnt",
    "couldnt",
    "shouldnt",
    "cant",
    "cannot",
}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "into",
    "from",
    "about",
    "above",
    "below",
    "between",
    "after",
    "before",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "also",
}


def _detect_negation_contradiction(left: str, right: str) -> bool:
    """Cheap heuristic: do these two texts NEGATE each other?

    Returns True when:
      * both texts are short-ish (≤ ~40 words; longer "essays" can share
        negated phrases for emphasis and we don't want false positives),
      * both texts contain at least one shared non-stopword content
        word (so "safe" vs "not safe" qualifies on "safe", but two
        unrelated negations like "not hot" / "not cold" do NOT —
        they share no content word and the heuristic returns False),
      * the SET of negation cues present in the two texts is not equal.
        "not safe" vs "safe" → {"not"} vs {} → True.
        "not safe" vs "very not safe" → {"not"} vs {"not", "very"} → False.

    This is deliberately narrow: it catches the exact ``"safe"`` vs
    ``"not safe"`` class of false-positive textual-agreement flips
    the F11 review reproduced, without pretending to be a full
    contradiction detector. Negation-detection in natural language is
    an open research problem — this is a guard-rail, not a theorem.
    """
    if not left or not right:
        return False
    left_norm = " ".join(left.lower().split())
    right_norm = " ".join(right.lower().split())
    left_tokens = left_norm.split()
    right_tokens = right_norm.split()
    if not left_tokens or not right_tokens:
        return False
    # Bound to sentence-pair-scale phrasing; longer "essays" warrant
    # a real NLP contradiction pipeline, not this heuristic.
    if len(left_tokens) > 40 or len(right_tokens) > 40:
        return False

    # Split contractions the tokenizer left intact ("isn't" → "isn't" is
    # one token; we already enumerate the spelled-out forms in
    # _NEGATION_TOKENS but also strip the trailing apostrophe so
    # ``"isn't"`` matches the entry ``"isnt"`` below).
    def _neg_cues(tokens: list[str]) -> set[str]:
        cues: set[str] = set()
        for tok in tokens:
            stripped = tok.strip(".,;:!?\"'()[]{}")
            if not stripped:
                continue
            lowered = stripped.lower()
            cue = lowered.rstrip("'")
            if cue in _NEGATION_TOKENS:
                cues.add(cue)
                continue
            # "n't" is part of contractions; check suffix only on the
            # *raw* token so we don't accidentally match "isn't" twice.
            if lowered.endswith("n't") and "n't" in _NEGATION_TOKENS:
                cues.add("n't")
        return cues

    left_cues = _neg_cues(left_tokens)
    right_cues = _neg_cues(right_tokens)

    def _content_words(tokens: list[str]) -> set[str]:
        words: set[str] = set()
        for tok in tokens:
            w = tok.strip(".,;:!?\"'()[]{}").lower()
            if len(w) < 3 or w in _STOPWORDS or w in _NEGATION_TOKENS:
                continue
            words.add(w)
        return words

    left_content = _content_words(left_tokens)
    right_content = _content_words(right_tokens)
    shared = left_content & right_content
    if not shared:
        return False
    # Cue sets differ → one side asserts, the other denies, on shared
    # content. Same cue set (both negated, or both un-negated) is fine.
    return bool(left_cues ^ right_cues)


def _detect_numeric_contradiction(left: str, right: str) -> bool:
    """Reject matching statement templates with different numeric literals."""
    pattern = r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?:%)?"
    left_numbers = re.findall(pattern, left)
    right_numbers = re.findall(pattern, right)
    return bool(
        left_numbers
        and right_numbers
        and left_numbers != right_numbers
        and re.sub(pattern, "#", " ".join(left.lower().split()))
        == re.sub(pattern, "#", " ".join(right.lower().split()))
    )


def _textual_agreement(left: str, right: str) -> float:
    """Composite score: textual similarity reduced when the texts contradict.

    Returns ``_text_similarity(left, right)`` unchanged unless
    :func:`_detect_negation_contradiction` flags the pair, in which case
    the score is floored to a small value so it cannot count toward a
    quorum / high-agreement verdict. ``0.05`` was chosen so that even
    a 2/3 majority with all-muses-positive-but-one-muse-negating
    produces per-pair scores that never reach the default ``0.66``
    quorum threshold — i.e. negation-contradicted pairs cannot
    silently inflate quorum_reached.
    """
    base = _text_similarity(left, right)
    if _detect_negation_contradiction(left, right) or _detect_numeric_contradiction(
        left, right
    ):
        return min(base, 0.05)
    return base


def _compute_quorum(all_responses: Dict[str, Dict], quorum: float) -> Dict[str, Any]:
    """Compute textual-agreement quorum across successful provider responses.

    F11 (2026-09-14) — IMPORTANT — this function does NOT measure
    semantic consensus. ``_text_similarity`` is a character-level
    SequenceMatcher ratio on whitespace-normalized lower-cased text;
    two texts can score high while meaning the opposite. The composite
    :func:`_textual_agreement` reduces the per-pair score to a floor of
    ``0.05`` when :func:`_detect_negation_contradiction` flags the pair,
    so negation-contradicted pairs cannot inflate ``quorum_reached``.

    Field semantics:
      * ``quorum_reached``     — True iff the largest textual-agreement
                                 component has at least ``quorum`` muses.
      * ``quorum_muses``       — the names of the muses in that component
                                 (empty list when ``quorum_reached`` is False).
      * ``consensus_score``    — surfaced as ``consensus_score`` on the
                                 wire (v3.0 public contract — DO NOT
                                 rename without a schema bump); it is
                                 the textual-agreement score of the
                                 quorum component. CALLERS MUST treat
                                 this as "textual agreement", NOT as
                                 "semantic consensus" or "correctness
                                 agreement". A high ``consensus_score``
                                 with all-muses-actually-disagreeing
                                 answers is the kind of bug a real
                                 NLP-driven consensus layer would catch;
                                 this lexical scorer cannot.
      * ``similarity_pairs``   — the per-pair textual similarity scores.
      * ``contradiction_pairs``— NEW in v4.4.0 (F11): a map keyed
                                 ``"left:right"`` containing those
                                 pairs whose negation-contradiction
                                 heuristic fired (value True). Lets
                                 operators audit whether the
                                 textual-agreement floor actually
                                 mattered for a given response.

    Availability-weighted consensus ("how many providers responded")
    is a SEPARATE concept from this textual-agreement score; this
    function is not a calibrated correctness vote.
    """
    successes = {
        name: str(resp.get("response_text") or "").strip()
        for name, resp in all_responses.items()
        if resp.get("status") == "success"
        and str(resp.get("response_text") or "").strip()
    }
    if len(successes) < 2:
        return {
            "quorum_reached": False,
            "quorum_muses": [],
            "consensus_score": 0.0,
            "similarity_pairs": {},
            "contradiction_pairs": {},
        }

    similarity_pairs: dict[str, float] = {}
    contradiction_pairs: dict[str, bool] = {}
    best_score = 0.0
    for left, right in combinations(successes, 2):
        score = _textual_agreement(successes[left], successes[right])
        similarity_pairs[f"{left}:{right}"] = round(score, 4)
        best_score = max(best_score, score)
        # Surface the negation-contradiction flag for operator audit
        # regardless of whether the pair joined a quorum component.
        if _detect_negation_contradiction(
            successes[left], successes[right]
        ) or _detect_numeric_contradiction(successes[left], successes[right]):
            contradiction_pairs[f"{left}:{right}"] = True
            # Also flag in the reverse-key direction so a caller
            # looking up the pair by either ordering finds it.
            contradiction_pairs[f"{right}:{left}"] = True

    # Complete-link groups: every member must agree with every other.
    # Transitive A~B~C does not establish A~C. Greedy grouping is bounded
    # and conservative, rather than an exponential maximum-clique search.
    components: list[list[str]] = []
    for name in sorted(successes):
        for group in components:
            if all(
                similarity_pairs.get(
                    f"{name}:{other}", similarity_pairs.get(f"{other}:{name}", 0.0)
                )
                >= quorum
                for other in group
            ):
                group.append(name)
                break
        else:
            components.append([name])
    quorum_size = max(2, math.ceil(quorum * len(successes)))
    quorum_muses = max(components, key=len)
    quorum_reached = len(quorum_muses) >= quorum_size

    if quorum_reached and len(quorum_muses) > 1:
        component_scores = [
            similarity_pairs.get(
                f"{left}:{right}", similarity_pairs.get(f"{right}:{left}", 0.0)
            )
            for left, right in combinations(quorum_muses, 2)
        ]
        consensus_score = min(component_scores) if component_scores else best_score
    else:
        consensus_score = min(best_score, max(0.0, quorum - 0.01))

    return {
        "agreement_metric": "lexical_complete_link",
        "quorum_reached": quorum_reached,
        "quorum_muses": quorum_muses if quorum_reached else [],
        "consensus_score": round(float(consensus_score), 4),
        "similarity_pairs": similarity_pairs,
        "contradiction_pairs": contradiction_pairs,
    }


def _selection_cache_tag(selection: Optional[Dict[str, Optional[str]]]) -> str:
    """Deterministic string suffix for the response cache key when a
    Custom Query selection is active. Different lineups must not share
    a cache entry — two callers asking the same prompt under different
    lineups expect different result sets.
    """
    if not selection:
        return ""
    parts = sorted(f"{name}={override or ''}" for name, override in selection.items())
    return "|" + ",".join(parts)


def _unavailable(model_id: str, error: str = "") -> Dict:
    """Uniform shape for provider failures.

    `error` (v3.1.2 diagnostic) is a short human-readable cause — e.g.
    "missing api_key", "HTTP 401 Unauthorized", "timeout after 30s" —
    surfaced by callers in logs and 503 responses so operators can
    diagnose without running the stack under DEBUG logging.
    """
    return {
        "status": "unavailable",
        "response_text": "",
        "latency_ms": 0,
        "model_id": model_id,
        "final_score": 0.0,
        "error": error,
    }


# ── Registry-backed manifest refresh ──────────────────────────────────────────
# Arena-ranked models always win over pattern matches when available.
_REGISTRY_MAP = GRAEAE_REGISTRY_MAP


_VERSION_RE = re.compile(r"(\d+)(?:[.\-_](\d+))?(?:[.\-_](\d+))?(?:[.\-_](\d+))?")
_VERSION_PAD = 4


def _extract_version(model_id: str) -> tuple:
    """Best-effort version tuple extraction for cherry-pick ordering.

    Captures the first dotted/hyphen-separated digit sequence, strips
    date-snapshot components (anything ≥ 1900 — years and YYYYMMDD
    stamps), and pads to a fixed length so prefix-shorter tuples don't
    outrank longer ones in ascending sort. Examples:

      gpt-5.4              → (5, 4, 0, 0)
      gpt-5                → (5, 0, 0, 0)
      gpt-5-2025-08-07     → (5, 0, 0, 0)   # 2025 = date, truncated
      claude-opus-4-7      → (4, 7, 0, 0)
      claude-haiku-4-5-20251001 → (4, 5, 0, 0)   # date stamp dropped
      gemini-3.1-pro-preview    → (3, 1, 0, 0)
      grok-4-0709          → (4, 709, 0, 0)

    After negation, gpt-5.4 sorts before gpt-5 in ascending order, so
    .sort() picks the highest version first.
    """
    m = _VERSION_RE.search(model_id)
    raw = [int(x) for x in m.groups() if x is not None] if m else []
    cleaned: list[int] = []
    for p in raw:
        # Anything ≥ 100 in a version slot is almost always a date code,
        # release stamp, or parameter count, not a major/minor version
        # — provider naming conventions don't ship versions like 5.235.
        # Stripping at 100 prevents 'grok-4-0709' (4, 709) from outranking
        # 'grok-4-1-fast' (4, 1) and 'gpt-5-2025-08-07' from outranking
        # 'gpt-5.4'.
        if p >= 100:
            break
        cleaned.append(p)
    cleaned.extend([0] * max(0, _VERSION_PAD - len(cleaned)))
    return tuple(cleaned[:_VERSION_PAD])


# How many candidates to probe before giving up on a provider. Each probe is
# one billable 1-token call, so we cap it. n-1 fallback covers "top pick is
# retired"; n-2 covers a double-rotation window. The cap is deliberately
# above _RISERS_MAX + arena_top + n-1 + n-2 so the Arena-ranked floor always
# gets a probe even when several post-Arena variants exist (e.g. four
# gpt-5.5 variants released the same day before Arena reviewed any).
_PROBE_MAX_CANDIDATES = 6
_RISERS_MAX = 3


def _is_reasoning_variant(model_id: str) -> bool:
    """True if ``model_id`` is a 'reasoning' SKU.

    Reasoning variants emit provider-specific reasoning blocks alongside
    the user-facing answer — e.g. xAI Grok-reasoning appends
    ``\\confidence{N}``; OpenAI o1 emits hidden chain-of-thought tokens
    that bleed through some adapters. The non-reasoning sibling
    (``...-non-reasoning`` or unsuffixed) returns clean text, which is
    what downstream consumers (consensus scoring, OpenAI-compat gateway,
    auto-context injection) expect. Prefer non-reasoning when both are
    available.
    """
    lower = model_id.lower()
    return lower.endswith("-reasoning") and not lower.endswith("-non-reasoning")


async def _ranked_candidates(
    conn, registry_provider: str, prefer: list[str]
) -> list[str]:
    """Return model_ids ordered by selection priority (best first).

    Two-stage policy:
      Stage A (post-Arena risers): same-family models with a STRICTLY higher
        version tuple than the Arena-ranked top — covers fresh releases that
        Arena hasn't ranked yet.
      Stage B (Arena top + tier): the Arena-ranked top, then everything else
        in the same family tier by Arena score / version / last_synced.
      Stage C (other families): models matching later prefer-list entries
        (n-1 family) by the same internal ordering. Used as fallback when
        the primary family is unhealthy.

    Within each stage, ties break by version tuple DESC, arena_score DESC,
    last_synced DESC, **non-reasoning preferred over reasoning variants**,
    len(model_id) ASC (prefer canonical short names).

    The non-reasoning tiebreak was added in v4.1.2 after xAI split
    grok-4-1-fast into reasoning + non-reasoning SKUs with identical
    family/version/weight; the prior ``len()``-only tiebreak accidentally
    promoted ``grok-4.20-0309-reasoning`` (27 chars) over
    ``grok-4.20-0309-non-reasoning`` (31 chars), so every consultation
    came back tagged ``\\confidence{N}`` instead of clean text.
    """
    rows = await conn.fetch(
        """
        SELECT model_id, arena_score, last_synced
        FROM model_registry
        WHERE provider = $1 AND available AND NOT deprecated
        """,
        registry_provider,
    )
    if not rows:
        return []

    annotated: list[dict] = []
    for r in rows:
        mid = r["model_id"]
        family_rank: Optional[int] = None
        for i, pat in enumerate(prefer):
            if pat.lower() in mid.lower():
                family_rank = i
                break
        # Drop entries that match no family pattern AND have no Arena rank
        # (e.g. groq's whisper variants in the llama family slot).
        if family_rank is None and not r["arena_score"]:
            continue
        annotated.append(
            {
                "mid": mid,
                "family_rank": family_rank if family_rank is not None else len(prefer),
                "version": _extract_version(mid),
                "arena": float(r["arena_score"] or 0),
                "synced": r["last_synced"].timestamp() if r["last_synced"] else 0,
            }
        )
    if not annotated:
        return []

    def _internal_key(a: dict) -> tuple:
        return (
            a["family_rank"],
            tuple(-x for x in a["version"]),
            -a["arena"],
            -a["synced"],
            _is_reasoning_variant(a["mid"]),  # False (0) before True (1)
            len(a["mid"]),
        )

    # Identify the Arena top within the strongest family tier (lowest
    # family_rank that has any Arena entries).
    arena_top: Optional[dict] = None
    for tier in sorted({a["family_rank"] for a in annotated}):
        in_tier = [a for a in annotated if a["family_rank"] == tier and a["arena"] > 0]
        if in_tier:
            arena_top = max(in_tier, key=lambda a: a["arena"])
            break

    risers: list[dict] = []
    if arena_top is not None:
        risers = [
            a
            for a in annotated
            if a["family_rank"] == arena_top["family_rank"]
            and a["version"] > arena_top["version"]
            and a["mid"] != arena_top["mid"]
        ]
        risers.sort(key=_internal_key)
        # Cap risers so Arena-ranked floor isn't crowded out of the probe
        # budget on providers that ship many same-day variants of a new
        # version (e.g. four gpt-5.5 release variants).
        risers = risers[:_RISERS_MAX]

    riser_ids = {a["mid"] for a in risers}
    others = sorted(
        (
            a
            for a in annotated
            if (arena_top is None or a["mid"] != arena_top["mid"])
            and a["mid"] not in riser_ids
        ),
        key=_internal_key,
    )

    ranked: list[str] = []
    seen: set[str] = set()
    for entry in risers + ([arena_top] if arena_top else []) + others:
        if entry is None:
            continue
        if entry["mid"] in seen:
            continue
        ranked.append(entry["mid"])
        seen.add(entry["mid"])
    return ranked


async def _probe_model(
    client: httpx.AsyncClient,
    provider_cfg: dict,
    model_id: str,
    api_key: str,
    timeout: int = 15,
) -> bool:
    """Probe model_id with a tiny generate call; True iff HTTP 200.

    Probe bodies match the shape of _call_provider_worker so a passing probe
    means the model can actually be dispatched against — not just that
    it appears in /v1/models. Token budgets are tuned per family:

      gpt-5.x reasoning models silently consume internal reasoning tokens
      from max_completion_tokens, so max_completion_tokens=1 always 400s
      with "max_tokens too low". Bumped to 128 to cover the reasoning
      phase + a 1-token output. Cost is pennies per startup × 8 providers.
    """
    api = provider_cfg["api"]
    try:
        if api == "openai":
            url = provider_cfg["url"]
            is_gpt5 = model_id.startswith("gpt-5")
            if is_gpt5:
                body = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_completion_tokens": 128,
                }
            else:
                body = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                    "temperature": 0,
                }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        elif api == "anthropic":
            url = provider_cfg["url"]
            body = {
                "model": model_id,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "hi"}],
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        elif api == "gemini":
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{model_id}:generateContent"
            )
            body = {
                "contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 4, "temperature": 0},
            }
            headers = {"x-goog-api-key": api_key}
        else:
            return False
        resp = await client.post(url, json=body, headers=headers, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        logger.debug(f"[GRAEAE] probe error for {model_id}: {type(e).__name__}: {e}")
        return False


# ── F06: Shared execution / accounting contract ────────────────────────────
#
# Background. The README documents that consultations flow through
# PANTHEON (model / cost routing) and KNEMON (dollar-budget enforcement).
# Before F06, GRAEAE invoked provider adapters directly and never
# touched the per-call spend ledger that KNEMON records for every
# other LLM-facing subsystem in MNEMOS — so a real $0.25 consultation
# surfaced to callers as $0.00. The fix introduces ONE shared
# execution/accounting contract here and re-points GRAEAE's per-call
# code at it:
#
#   record_provider_spend(...)  — fire-and-forget append to KNEMON's
#                                 usage_ledger, carrying upstream
#                                 token usage AND the priced cost
#                                 the adapter collected. Both
#                                 branches return a small
#                                 ``SpendRecord`` so callers can
#                                 inspect status without try/except.
#
#   enforce_provider_budget(... ) — pre-dispatch budget gate. When
#                                   the deployment has a configured
#                                   persistence backend with KNEMON
#                                   budget enforcement, this raises
#                                   ``ProviderBudgetExceeded`` and
#                                   refuses the call. When no
#                                   backend / no KNEMON subsystem is
#                                   available, this is a no-op
#                                   (with an info-level log) so
#                                   single-process / test
#                                   deployments keep working.
#
# These helpers are intentionally tolerant of missing mnemos-core
# extensions (mnemos-knemon / mnemos-pantheon are separate
# distributions). The import probe is one-time per process so the
# hot path stays cheap.

# One-time import probe — caches whether the KNEMON usage_ledger
# machinery is importable + whether the persistence backend
# implements it. Set on first call to ``record_provider_spend`` /
# ``enforce_provider_budget``.
_F06_LEGERD_PROBE: Optional[bool] = None


class ProviderBudgetExceeded(RuntimeError):
    """Raised by ``enforce_provider_budget`` when a KNEMON-configured
    dollar cap would be violated by the proposed call.

    The exception carries ``provider``, ``model``, and ``task_kind``
    attributes so the consultation layer can surface a precise
    402/429 response instead of a generic 500. It is intentionally
    a ``RuntimeError`` subclass so legacy callers that catch the
    bare exception still work, but its semantic role is "refuse
    the call, do not silently down-budget"."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        task_kind: str,
        remaining_usd: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.task_kind = task_kind
        self.remaining_usd = remaining_usd


@dataclass
class SpendRecord:
    """Result of a single ``record_provider_spend`` call.

    ``status`` is one of:

      * ``"ok"`` — KNEMON usage_ledger insert succeeded; ``record_id``
        populated.
      * ``"degraded"`` — a priced cost and usage were captured but
        the ledger insert failed. ``error`` holds the repr of the
        failing exception. Same contract as ``mnemos.llm.call``.
      * ``"unpriced"`` — the adapter did not surface a numeric cost
        AND no KNEMON pricing layer was available to compute one.
        We intentionally DID NOT coerce the unknown cost to zero;
        the consensus rollup will surface ``cost=None``.
      * ``"skipped"`` — no persistence backend is configured. We
        logged at debug level and moved on so single-process
        deployments keep working.
    """

    status: str
    provider: str
    model: str
    task_kind: str
    cost_usd: Optional[float]
    usage: Optional[Dict[str, int]]
    record_id: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spend_status": self.status,
            "spend_provider": self.provider,
            "spend_model": self.model,
            "spend_task_kind": self.task_kind,
            "spend_cost_usd": self.cost_usd,
            "spend_record_id": self.record_id,
            "spend_error": self.error,
        }


@dataclass
class BudgetCheckResult:
    """Result of ``enforce_provider_budget``.

    ``allowed`` is True when the call may proceed; False when it
    must be refused. ``remaining_usd`` is the configured dollar
    headroom at the time of the check; ``None`` when no budget is
    configured (every call is allowed).
    """

    allowed: bool
    provider: str
    model: str
    task_kind: str
    remaining_usd: Optional[float] = None
    reason: Optional[str] = None

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise ProviderBudgetExceeded(
                self.reason or f"provider {self.provider} budget exceeded",
                provider=self.provider,
                model=self.model,
                task_kind=self.task_kind,
                remaining_usd=self.remaining_usd,
            )


# Test/operator injection points. Tests monkeypatch these to verify
# GRAEAE's consultation path actually exercises the accounting
# contract — without standing up a real Postgres usage_ledger table.
_spend_recorder = None  # Callable[[SpendRecord], None]
_budget_checker = None  # Callable[[str, str, str], BudgetCheckResult]


def configure_accounting_contract(
    *,
    spend_recorder=None,
    budget_checker=None,
) -> None:
    """Inject custom KNEMON-spend / PANTHEON-budget handlers (tests).

    Both arguments default to ``None`` which means "use the
    auto-discovery path against mnemos-core". Tests should set
    BOTH before exercising the consultation path so the test
    asserts against the injected hooks rather than racing the
    persistence backend's setup state.
    """
    global _spend_recorder, _budget_checker
    _spend_recorder = spend_recorder
    _budget_checker = budget_checker


def reset_accounting_contract() -> None:
    """Clear any test-injected accounting handlers (test cleanup)."""
    global _spend_recorder, _budget_checker
    _spend_recorder = None
    _budget_checker = None


def _resolve_spend_recorder():
    """Return the active spend recorder (test-injected or KNEMON).

    Returns a callable that takes a ``UsageLedgerRecord``-like
    payload and returns a ``SpendRecord`` describing what happened.
    Falls back to a no-op recorder when no backend is configured —
    GRAEAE's reliability stack must work in single-process tests
    that don't have a Postgres usage_ledger.
    """
    if _spend_recorder is not None:
        return _spend_recorder

    try:
        import mnemos.core.lifecycle as _lc  # noqa: WPS433 — local import
    except Exception:
        return _noop_spend_recorder

    backend = getattr(_lc, "_persistence_backend", None)
    if backend is None or not hasattr(backend, "record_usage_ledger"):
        return _noop_spend_recorder

    async def _knemon_spend(record) -> SpendRecord:
        # KNEMON's UsageLedgerRecord wants pre-priced
        # ``est_cost_usd``; if the upstream adapter didn't return
        # one we pass ``None`` rather than coercing to $0.00.
        try:
            from mnemos.persistence.base import UsageLedgerRecord as _Rec
        except Exception:
            return SpendRecord(
                status="unpriced",
                provider=record.provider,
                model=record.model,
                task_kind=record.task_kind,
                cost_usd=record.cost_usd,
                usage=record.usage,
                error="mnemos.persistence.base unavailable",
            )
        try:
            from mnemos.domain.knemon.attempts import normalized_usage
            from mnemos.domain.knemon.budget import canonical_budget_key

            usage = normalized_usage(record.usage)
            canonical_provider, _ = canonical_budget_key(record.provider, record.model)
            ledger_record = _Rec(
                provider=canonical_provider,
                model=record.model,
                task_kind=record.task_kind,
                **usage,
                latency_ms=int(record.latency_ms or 0),
                outcome=record.outcome,
                caller_subsystem="graeae",
                tier=record.tier or "standard",
                est_cost_usd=record.cost_usd,
            )
        except Exception as exc:  # malformed record shape
            return SpendRecord(
                status="degraded",
                provider=record.provider,
                model=record.model,
                task_kind=record.task_kind,
                cost_usd=record.cost_usd,
                usage=record.usage,
                error=repr(exc),
            )
        try:
            backend_obj = _lc._persistence_backend
            assert backend_obj is not None
            async with backend_obj.transactional() as tx:
                ledger_result = await backend_obj.record_usage_ledger(tx, ledger_record)
            return SpendRecord(
                status="ok",
                provider=record.provider,
                model=record.model,
                task_kind=record.task_kind,
                cost_usd=record.cost_usd,
                usage=record.usage,
                record_id=int(getattr(ledger_result, "id", 0)) or None,
            )
        except Exception as exc:
            return SpendRecord(
                status="degraded",
                provider=record.provider,
                model=record.model,
                task_kind=record.task_kind,
                cost_usd=record.cost_usd,
                usage=record.usage,
                error=repr(exc),
            )

    return _knemon_spend


async def _noop_spend_recorder(record) -> SpendRecord:
    logger.debug(
        "[GRAEAE] no persistence backend configured; spend for %s/%s "
        "stays in-process (cost_usd=%s).",
        record.provider,
        record.model,
        record.cost_usd,
    )
    return SpendRecord(
        status="skipped",
        provider=record.provider,
        model=record.model,
        task_kind=record.task_kind,
        cost_usd=record.cost_usd,
        usage=record.usage,
    )


def _resolve_budget_checker():
    """Return the active budget checker (test-injected or PANTHEON).

    Falls back to a permissive no-op when no persistence backend
    exposes a budget policy. Production deployments wire this
    through mnemos-knemon's per-plan window checks; we accept
    any object with an async ``check_budget(provider, model,
    task_kind)`` returning a ``BudgetCheckResult``.
    """
    if _budget_checker is not None:
        return _budget_checker

    async def check(provider, model, task_kind):
        from mnemos.core import lifecycle
        from mnemos.domain.knemon.budget import evaluate_usage_budget

        decision = await evaluate_usage_budget(
            lifecycle._persistence_backend, provider=provider, caller_subsystem="graeae"
        )
        return BudgetCheckResult(
            allowed=decision.allowed,
            provider=provider,
            model=model,
            task_kind=task_kind,
            remaining_usd=(
                decision.remaining_usd
                if math.isfinite(decision.remaining_usd)
                else None
            ),
            reason=decision.reason,
        )

    return check


async def _permissive_budget_checker(
    provider: str, model: str, task_kind: str
) -> BudgetCheckResult:
    return BudgetCheckResult(
        allowed=True,
        provider=provider,
        model=model,
        task_kind=task_kind,
        remaining_usd=None,
        reason=None,
    )


@dataclass
class _SpendInput:
    """Internal payload handed to the spend recorder.

    Kept private because callers should construct via
    ``record_provider_spend`` which validates inputs.
    """

    provider: str
    model: str
    task_kind: str
    cost_usd: Optional[float]
    usage: Optional[Dict[str, int]]
    latency_ms: int
    outcome: str
    tier: str = "standard"


async def record_provider_spend(
    *,
    provider: str,
    model: str,
    task_type: str,
    cost_usd: Optional[float],
    usage: Optional[Dict[str, Any]],
    latency_ms: int = 0,
    outcome: str = "ok",
    tier: str = "standard",
) -> SpendRecord:
    """Append one provider call to KNEMON's spend ledger.

    Single shared entrypoint used by ``route()``, the consultation
    fan-out, the embedding path (when it lives in GRAEAE), and the
    fallback worker. Returning a ``SpendRecord`` rather than
    raising keeps the consultation hot path crash-safe — a failed
    ledger insert degrades to ``status="degraded"`` with the
    failure captured, not a 500 to the caller. Operators reading
    ``spend_status`` on the consultation record can see when their
    ledger has fallen behind without grepping logs.

    Args:
      provider: provider name (e.g. ``"openai"``).
      model: model id actually used (post override).
      task_type: routing category (``"reasoning"``,
        ``"embedding"``, ``"consultation"``). Routed through KNEMON's
        task_kind column; preserves traceability across the
        upstream ``llm-usage-policy`` table.
      cost_usd: priced dollar cost (Optional[None] = unpriced). The
        adapter layer is responsible for the upstream ``usage``
        block; ``record_provider_spend`` does NOT re-derive a
        cost from token counts — that would silently re-introduce
        the F06 bug if the upstream cost was a real number and
        the token math disagreed.
      usage: upstream token usage dict (None when the adapter did
        not surface one). Same fidelity rules as cost.
      latency_ms: wall-clock latency for the call.
      outcome: ``"ok"`` / ``"err"`` / ``"timeout"`` — matches the
        ``mnemos.llm`` outcome vocabulary.
      tier: KNEMON plan tier (``"standard"`` by default).

    Returns:
      ``SpendRecord`` describing what happened.
    """
    payload = _SpendInput(
        provider=str(provider),
        model=str(model),
        task_kind=str(task_type or "consultation"),
        cost_usd=(
            float(cost_usd)
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool)
            else None
        ),
        usage=dict(usage) if isinstance(usage, dict) else None,
        latency_ms=max(0, int(latency_ms or 0)),
        outcome=str(outcome or "ok"),
        tier=str(tier or "standard"),
    )
    recorder = _resolve_spend_recorder()
    try:
        result = await recorder(payload)
    except Exception as exc:
        # Crash-safe by contract: a misbehaving KNEMON backend
        # MUST NOT propagate into the consultation hot path. We
        # convert the failure into a SpendRecord carrying
        # ``status="degraded"`` + the error repr, matching the
        # ``mnemos.llm.call`` ledger-failure contract. The priced
        # cost (if any) is preserved so callers can still surface
        # the spend on the consultation record for reconciliation.
        logger.error(
            "[GRAEAE] spend recorder crashed for %s/%s: %s",
            payload.provider,
            payload.model,
            exc,
        )
        result = SpendRecord(
            status="degraded",
            provider=payload.provider,
            model=payload.model,
            task_kind=payload.task_kind,
            cost_usd=payload.cost_usd,
            usage=payload.usage,
            error=repr(exc),
        )
    return result


async def enforce_provider_budget(
    *,
    provider: str,
    model: str,
    task_type: str,
) -> BudgetCheckResult:
    """Pre-dispatch budget gate. Raises when the call is denied.

    The README states consultations go through PANTHEON's triage
    + KNEMON's dollar budget enforcement; this function is the
    place that contract actually shows up in code. When no
    PANTHEON/KNEMON plugin is installed the check is a no-op so
    single-process deployments keep working — the contract is
    ``always check, possibly permissive``, never ``silently
    skip``.
    """
    checker = _resolve_budget_checker()
    return await checker(provider, model, task_type or "consultation")


# ── Module-level singleton ─────────────────────────────────────────────────────

_graeae_engine: Optional[GraeaeEngine] = None


def get_graeae_engine() -> GraeaeEngine:
    global _graeae_engine
    if _graeae_engine is None:
        _graeae_engine = GraeaeEngine()
    return _graeae_engine
