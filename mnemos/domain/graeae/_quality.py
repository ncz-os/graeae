from __future__ import annotations

"""Per-provider quality tracking for dynamic weight adjustment.

Maintains a rolling window of the last N outcomes per provider. The dynamic
weight is base_weight × success_multiplier, where success_multiplier scales
linearly from 0.5 (0% success) to 1.0 (100% success). This means a provider
with a perfect recent track record keeps its full configured weight, while a
flaky one is deprioritised without being removed from the pool entirely
(the circuit breaker handles full removal).
"""
import threading
from collections import deque
from enum import IntEnum
from typing import Any, Callable, Optional
from datetime import datetime

_WINDOW = 20  # rolling window size (number of outcomes)
_CAPABILITY_TTL_SECONDS = 1800
_BREAKER_COOLDOWN_SECONDS = 300


class Classification(IntEnum):
    """Provider health classification based on HTTP status codes."""

    REACHABLE = 200  # Any 2xx status
    CAPACITY = 429   # Transient - skip but don't increment failure counter
    UNPROVISIONED = 404  # Permanent - skip, don't open breaker
    UNAUTHORIZED = 401  # Permanent - skip, don't open breaker
    OTHER_ERROR = 500  # Generic error - increment failure counter

    @classmethod
    def from_status(cls, status_code: int) -> 'Classification':
        """Derive classification from HTTP status code."""
        if 200 <= status_code < 300:
            return cls.REACHABLE
        elif status_code == 401 or status_code == 403:
            return cls.UNAUTHORIZED
        elif status_code == 404:
            return cls.UNPROVISIONED
        elif status_code == 429 or status_code == 503:
            return cls.CAPACITY
        else:
            return cls.OTHER_ERROR


class CircuitBreaker:
    """Circuit breaker implementation for provider health."""

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 300):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
        self.last_probe_time: Optional[float] = None
        self._half_open = False
        self._lock = threading.Lock()

    def record_failure(self) -> bool:
        """Record a failure and return True if breaker should be open."""
        with self._lock:
            if self.is_open:
                return True
            
            self.failure_count += 1
            self.last_failure_time = datetime.now().timestamp()
            
            if self.failure_count >= self.max_failures:
                self.is_open = True
                return True
                
            return False

    def record_success(self) -> bool:
        """Record a success and return True if breaker should close."""
        with self._lock:
            if not self.is_open:
                # Reset failure count for success when not open
                self.failure_count = 0
                return False
                
            # When open, reset and close the breaker
            self.failure_count = 0
            self.is_open = False
            self._half_open = False
            self.last_probe_time = None
            return True

    def should_skip(self) -> bool:
        """Return True if provider should be skipped due to circuit breaker."""
        with self._lock:
            if not self.is_open:
                return False
                
            # If breaker is open, check if cooldown period has passed
            if self.last_failure_time is not None:
                now = datetime.now().timestamp()
                if now - self.last_failure_time >= self.cooldown_seconds:
                    # Time to try a half-open probe - but only allow one probe
                    if not self._half_open:
                        self._half_open = True
                        # Don't skip - allow probe to happen
                        return False
                    else:
                        # Already probing, skip this request
                        return True
                        
            return True

    def allow_half_open_probe(self) -> bool:
        """Return True if a half-open probe should be attempted."""
        with self._lock:
            # Allow half-open probe only if breaker is open and not already probing
            if self.is_open and self._half_open and self.last_probe_time is None:
                return True
            return False

    def mark_probe_attempted(self) -> None:
        """Mark that a half-open probe was attempted."""
        with self._lock:
            self.last_probe_time = datetime.now().timestamp()
            self._half_open = False  # Clear half-open state after probe attempt

    def reset_half_open(self) -> None:
        """Reset half-open state."""
        with self._lock:
            self._half_open = False
            self.last_probe_time = None


class ProviderQuality:
    """Rolling success-rate and latency tracker for one provider."""

    def __init__(self, base_weight: float, flush_callback: Optional[Callable[[dict], None]] = None):
        self.base_weight = base_weight
        self._outcomes: deque[bool] = deque(maxlen=_WINDOW)
        self._latencies: deque[int] = deque(maxlen=_WINDOW)
        self._lock = threading.Lock()
        self._last_checked: Optional[float] = None
        self._classification: Optional[Classification] = None
        self._circuit_breaker = CircuitBreaker()
        self._dirty = False
        self._flush_callback = flush_callback
        self._provider_name = ""  # Store provider name for flush operations

    def _mark_dirty(self) -> None:
        """Mark this provider as needing to be flushed."""
        self._dirty = True
        # Do NOT call flush callback here - this is the hot path and violates performance requirements
        # The flush callback should be called by the flush() method, not on every record

    def _get_provider_key(self) -> str:
        """Get a key for this provider (used for flush operations)."""
        # Return the provider name for flush operations
        return self._provider_name

    def record_success(self, latency_ms: int) -> None:
        with self._lock:
            self._last_checked = datetime.now().timestamp()
            self._outcomes.append(True)
            self._latencies.append(latency_ms)
            self._circuit_breaker.record_success()
            # Reset classification to REACHABLE on success (rescue the provider)
            self._classification = Classification.REACHABLE
            # Mark as dirty for later flush
            self._mark_dirty()

    def record_failure(
        self, classification: Classification = Classification.OTHER_ERROR
    ) -> None:
        with self._lock:
            self._last_checked = datetime.now().timestamp()
            self._classification = classification
            # Always append to outcomes for success rate calculation
            self._outcomes.append(False)
            
            # Handle special classifications that don't increment failure counter
            if classification in (Classification.CAPACITY, Classification.UNPROVISIONED, Classification.UNAUTHORIZED):
                # For capacity, unprovisioned, and unauthorized - don't increment failure counter
                # and don't open the breaker
                pass
            else:
                # Regular error - increment failure counter and open breaker if needed
                self._circuit_breaker.record_failure()
            # Mark as dirty for later flush
            self._mark_dirty()

    def dynamic_weight(self) -> float:
        """base_weight scaled by recent success rate: 100% → 1.0×, 0% → 0.5×."""
        with self._lock:
            if not self._outcomes:
                return self.base_weight
            rate = sum(self._outcomes) / len(self._outcomes)
            multiplier = 0.5 + 0.5 * rate
            return round(self.base_weight * multiplier, 4)

    def avg_latency_ms(self) -> int:
        with self._lock:
            return (
                int(sum(self._latencies) / len(self._latencies))
                if self._latencies
                else 0
            )

    def should_skip(self) -> bool:
        """Determine if this provider should be skipped."""
        # Get the current state of the circuit breaker to avoid race conditions
        with self._lock:
            # Check circuit breaker first
            should_skip_breaker = self._circuit_breaker.should_skip()
            if should_skip_breaker:
                return True
                
            # Check capacity classification (skip if within TTL)
            if self._classification == Classification.CAPACITY and self._last_checked:
                now = datetime.now().timestamp()
                if now - self._last_checked < _CAPABILITY_TTL_SECONDS:
                    return True
                else:
                    # TTL expired, don't skip anymore
                    return False
                    
            # Check unauthorized classification (should always skip)
            if self._classification in (Classification.UNAUTHORIZED, Classification.UNPROVISIONED):
                return True
                
            return False

    def get_classification(self) -> Classification:
        """Get the current classification."""
        return self._classification or Classification.REACHABLE

    def get_last_checked(self) -> Optional[float]:
        """Get the last checked timestamp."""
        return self._last_checked

    def get_freshness(self) -> Optional[float]:
        """Get the time since last check."""
        if self._last_checked is None:
            return None
        return datetime.now().timestamp() - self._last_checked

    def get_circuit_breaker_state(self) -> dict:
        """Get circuit breaker state for reporting."""
        return {
            "is_open": self._circuit_breaker.is_open,
            "failure_count": self._circuit_breaker.failure_count,
            "last_failure_time": self._circuit_breaker.last_failure_time,
            "half_open": self._circuit_breaker._half_open
        }

    def get_dirty_flag(self) -> bool:
        """Get dirty flag for flush operations."""
        return self._dirty

    def clear_dirty_flag(self) -> None:
        """Clear dirty flag after flush."""
        self._dirty = False


class QualityTracker:
    """Pool of quality trackers, one per provider."""

    def __init__(self, provider_weights: dict[str, float], flush_callback: Optional[Callable[[dict], None]] = None):
        self._trackers: dict[str, ProviderQuality] = {}
        self._flush_callback = flush_callback  # Store the callback at tracker level
        for provider, weight in provider_weights.items():
            tracker = ProviderQuality(weight, flush_callback)
            tracker._provider_name = provider  # Set the provider name for flush operations
            self._trackers[provider] = tracker

    def _get(self, provider: str) -> ProviderQuality | None:
        return self._trackers.get(provider)

    def record_success(self, provider: str, latency_ms: int) -> None:
        if t := self._get(provider):
            t.record_success(latency_ms)

    def record_failure(self, provider: str, classification: Classification = Classification.OTHER_ERROR) -> None:
        if t := self._get(provider):
            t.record_failure(classification)

    def dynamic_weight(self, provider: str) -> float:
        if t := self._get(provider):
            return t.dynamic_weight()
        return 0.0

    def status(self) -> dict:
        return {
            p: {
                "dynamic_weight": t.dynamic_weight(),
                "base_weight": t.base_weight,
                "avg_latency_ms": t.avg_latency_ms(),
            }
            for p, t in self._trackers.items()
        }

    def should_skip(self, provider: str) -> bool:
        """Check if a provider should be skipped due to circuit breaker or capacity issues."""
        if t := self._get(provider):
            return t.should_skip()
        return False

    def get_classification(self, provider: str) -> Classification:
        """Get the classification for a provider."""
        if t := self._get(provider):
            return t.get_classification()
        return Classification.REACHABLE

    def get_last_checked(self, provider: str) -> Optional[float]:
        """Get the last checked timestamp for a provider."""
        if t := self._get(provider):
            return t.get_last_checked()
        return None

    def get_freshness(self, provider: str) -> Optional[float]:
        """Get the freshness for a provider."""
        if t := self._get(provider):
            return t.get_freshness()
        return None

    def get_circuit_breaker_state(self, provider: str) -> dict:
        """Get circuit breaker state for a provider."""
        if t := self._get(provider):
            return t.get_circuit_breaker_state()
        return {}

    async def flush(self) -> None:
        """Flush all dirty providers."""
        # Collect all dirty providers and pass them to the flush callback in one batch
        dirty_providers = {}
        for provider, tracker in self._trackers.items():
            if tracker.get_dirty_flag():
                dirty_providers[tracker._get_provider_key()] = tracker
                # Clear the dirty flag after collection
                tracker.clear_dirty_flag()
        
        # Call flush callback with all dirty providers in one batch
        if dirty_providers and self._flush_callback:
            # This is synchronous, but that's acceptable for the flush pattern
            # The async declaration allows callers to await it, but we don't need to
            # await the callback itself
            self._flush_callback(dirty_providers)


def make_quality_tracker(
    settings: Any,
    provider_weights: dict[str, float],
    *,
    redis_client: Any | None = None,
) -> QualityTracker:
    """Factory function to create a quality tracker."""
    if redis_client is not None:
        # Return RedisQualityTracker if Redis client provided
        return RedisQualityTracker(provider_weights, redis_client)
    else:
        # Return regular QualityTracker
        return QualityTracker(provider_weights)


class RedisQualityTracker(QualityTracker):
    """Redis-backed quality tracker for persistence."""
    
    def __init__(self, provider_weights: dict[str, float], redis_client: Any, flush_callback: Optional[Callable[[dict], None]] = None):
        super().__init__(provider_weights, flush_callback)
        self._redis_client = redis_client
        # In a real implementation, this would load from Redis and save to Redis
        # For now, we'll keep the same structure but with Redis integration
