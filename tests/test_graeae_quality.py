"""Tests for provider quality tracking functionality."""

import unittest
from unittest.mock import MagicMock
from mnemos.domain.graeae._quality import (
    Classification,
    ProviderQuality,
    QualityTracker,
    CircuitBreaker
)


class TestClassification(unittest.TestCase):
    def test_from_status_reachable(self):
        """Test that 2xx statuses map to REACHABLE."""
        self.assertEqual(Classification.from_status(200), Classification.REACHABLE)
        self.assertEqual(Classification.from_status(201), Classification.REACHABLE)
        self.assertEqual(Classification.from_status(299), Classification.REACHABLE)

    def test_from_status_capacity(self):
        """Test that 429 and 503 map to CAPACITY."""
        self.assertEqual(Classification.from_status(429), Classification.CAPACITY)
        self.assertEqual(Classification.from_status(503), Classification.CAPACITY)

    def test_from_status_unprovisioned(self):
        """Test that 404 maps to UNPROVISIONED."""
        self.assertEqual(Classification.from_status(404), Classification.UNPROVISIONED)

    def test_from_status_unauthorized(self):
        """Test that 401 and 403 map to UNAUTHORIZED."""
        self.assertEqual(Classification.from_status(401), Classification.UNAUTHORIZED)
        self.assertEqual(Classification.from_status(403), Classification.UNAUTHORIZED)

    def test_from_status_other_error(self):
        """Test that other statuses map to OTHER_ERROR."""
        self.assertEqual(Classification.from_status(500), Classification.OTHER_ERROR)
        self.assertEqual(Classification.from_status(501), Classification.OTHER_ERROR)
        self.assertEqual(Classification.from_status(400), Classification.OTHER_ERROR)


class TestCircuitBreaker(unittest.TestCase):
    def test_record_failure_opens_breaker(self):
        """Test that 3 consecutive failures open the breaker."""
        cb = CircuitBreaker(max_failures=3)
        # Record 3 failures
        self.assertFalse(cb.record_failure())  # Should not open yet
        self.assertFalse(cb.record_failure())  # Should not open yet
        self.assertTrue(cb.record_failure())   # Should open breaker
        
        # Verify breaker is open
        self.assertTrue(cb.is_open)
        self.assertEqual(cb.failure_count, 3)

    def test_record_success_closes_breaker(self):
        """Test that success closes the breaker."""
        cb = CircuitBreaker(max_failures=3)
        # Record 3 failures to open breaker
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_open)
        
        # Record a success to close breaker
        cb.record_success()
        self.assertFalse(cb.is_open)
        self.assertEqual(cb.failure_count, 0)

    def test_should_skip_when_closed(self):
        """Test that should_skip returns False when breaker is closed."""
        cb = CircuitBreaker(max_failures=3)
        self.assertFalse(cb.should_skip())

    def test_should_skip_when_open(self):
        """Test that should_skip returns True when breaker is open."""
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()  # Trigger breaker open
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.should_skip())


class TestProviderQuality(unittest.TestCase):
    def test_record_success_updates_classification(self):
        """Test that success resets classification to REACHABLE."""
        pq = ProviderQuality(1.0)
        pq.record_failure(Classification.CAPACITY)
        self.assertEqual(pq.get_classification(), Classification.CAPACITY)
        
        pq.record_success(100)
        self.assertEqual(pq.get_classification(), Classification.REACHABLE)

    def test_record_failure_sets_classification(self):
        """Test that failure properly sets classification."""
        pq = ProviderQuality(1.0)
        pq.record_failure(Classification.UNAUTHORIZED)
        self.assertEqual(pq.get_classification(), Classification.UNAUTHORIZED)
        
        pq.record_failure(Classification.OTHER_ERROR)
        self.assertEqual(pq.get_classification(), Classification.OTHER_ERROR)


class TestQualityTracker(unittest.TestCase):
    def test_should_skip_with_capacity(self):
        """Test that capacity classification skips within TTL."""
        qt = QualityTracker({"provider1": 1.0})
        # Record a capacity failure
        qt.record_failure("provider1", Classification.CAPACITY)
        
        # Should skip within TTL
        self.assertTrue(qt.should_skip("provider1"))

    def test_get_classification(self):
        """Test getting classification."""
        qt = QualityTracker({"provider1": 1.0})
        self.assertEqual(qt.get_classification("provider1"), Classification.REACHABLE)
        
        # Record a failure
        qt.record_failure("provider1", Classification.CAPACITY)
        self.assertEqual(qt.get_classification("provider1"), Classification.CAPACITY)


if __name__ == '__main__':
    unittest.main()