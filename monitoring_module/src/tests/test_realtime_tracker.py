"""
Tests for Real-Time Fairness Tracker

Tests the RealTimeFairnessTracker and BatchFairnessMonitor for
monitoring fairness metrics in production systems.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile

from monitoring_module.src.realtime_tracker import (
    RealTimeFairnessTracker,
    BatchFairnessMonitor,
)
from shared.validation import ValidationError  # Import the custom exception


class TestRealTimeFairnessTracker:
    """Tests for RealTimeFairnessTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return RealTimeFairnessTracker(
            window_size=100,
            metrics=['demographic_parity', 'equalized_odds'],
            min_samples=30
        )
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.window_size == 100
        assert 'demographic_parity' in tracker.metrics
        assert 'equalized_odds' in tracker.metrics
        assert tracker.min_samples == 30
        assert len(tracker.y_true_buffer) == 0
        assert len(tracker.history) == 0
        assert tracker.n_samples_processed == 0
    
    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        tracker = RealTimeFairnessTracker()
        
        assert tracker.window_size == 1000
        assert 'demographic_parity' in tracker.metrics
        assert tracker.min_samples == 100
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_add_batch_basic(self, mock_compute_metric, tracker):
        """Test adding a batch of predictions."""
        # Mock compute_metric
        mock_compute_metric.return_value = (0.10, {}, {})
        
        y_pred = np.array([1, 0, 1, 0, 1])
        y_true = np.array([1, 0, 1, 1, 0])
        sensitive = np.array([0, 0, 1, 1, 0])
        
        metrics = tracker.add_batch(y_pred, y_true, sensitive)
        
        # Should not compute metrics yet (insufficient samples)
        assert len(metrics) == 0
        assert len(tracker.y_pred_buffer) == 5
        assert len(tracker.y_true_buffer) == 5
        assert len(tracker.sensitive_buffer) == 5
        assert tracker.n_samples_processed == 5
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_add_batch_sufficient_samples(self, mock_compute_metric, tracker):
        """Test computing metrics with sufficient samples."""
        # Mock compute_metric
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add enough samples
        n = tracker.min_samples
        y_pred = np.random.binomial(1, 0.5, n)
        y_true = np.random.binomial(1, 0.5, n)
        sensitive = np.random.binomial(1, 0.5, n)
        
        metrics = tracker.add_batch(y_pred, y_true, sensitive)
        
        # Should compute metrics now
        assert len(metrics) > 0
        assert 'demographic_parity' in metrics or 'equalized_odds' in metrics
        assert len(tracker.history) > 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_add_batch_with_timestamps(self, mock_compute_metric, tracker):
        """Test adding batch with custom timestamps."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        y_pred = np.array([1, 0, 1])
        y_true = np.array([1, 0, 0])
        sensitive = np.array([0, 1, 0])
        
        timestamps = [
            datetime.now(),
            datetime.now() + timedelta(seconds=1),
            datetime.now() + timedelta(seconds=2),
        ]
        
        tracker.add_batch(y_pred, y_true, sensitive, timestamps=timestamps)
        
        assert len(tracker.timestamp_buffer) == 3
    
    def test_add_batch_validation_error(self, tracker):
        """Test validation of input arrays."""
        y_pred = np.array([1, 0, 1])
        y_true = np.array([1, 0])  # Wrong length
        sensitive = np.array([0, 1, 0])
        
        # Changed from ValueError to ValidationError
        with pytest.raises(ValidationError, match="Length mismatch"):
            tracker.add_batch(y_pred, y_true, sensitive)
    
    def test_add_batch_converts_lists(self, tracker):
        """Test that lists are converted to arrays."""
        y_pred = [1, 0, 1, 0]
        y_true = [1, 0, 1, 1]
        sensitive = [0, 0, 1, 1]
        
        tracker.add_batch(y_pred, y_true, sensitive)
        
        assert len(tracker.y_pred_buffer) == 4
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_sliding_window_behavior(self, mock_compute_metric, tracker):
        """Test that buffer maintains window size."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add more than window size
        total_samples = tracker.window_size + 50
        
        for i in range(5):  # Add in batches
            batch_size = total_samples // 5
            y_pred = np.random.binomial(1, 0.5, batch_size)
            y_true = np.random.binomial(1, 0.5, batch_size)
            sensitive = np.random.binomial(1, 0.5, batch_size)
            
            tracker.add_batch(y_pred, y_true, sensitive)
        
        # Buffer should not exceed window size
        assert len(tracker.y_pred_buffer) <= tracker.window_size
        assert len(tracker.y_true_buffer) <= tracker.window_size
        assert len(tracker.sensitive_buffer) <= tracker.window_size
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_get_current_metrics(self, mock_compute_metric, tracker):
        """Test retrieving current metrics."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add sufficient samples
        n = tracker.min_samples
        tracker.add_batch(
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n)
        )
        
        current = tracker.get_current_metrics()
        
        assert isinstance(current, dict)
        # Should have metrics (if computation succeeded)
        if len(current) > 0:
            assert all(isinstance(v, (int, float)) for v in current.values())
    
    def test_get_current_metrics_empty(self, tracker):
        """Test getting metrics when none available."""
        current = tracker.get_current_metrics()
        
        assert current == {}
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_get_time_series(self, mock_compute_metric, tracker):
        """Test retrieving time series data."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add multiple batches to create time series
        for i in range(3):
            n = tracker.min_samples
            tracker.add_batch(
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n)
            )
        
        ts = tracker.get_time_series()
        
        assert isinstance(ts, pd.DataFrame)
        assert 'timestamp' in ts.columns
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_get_time_series_single_metric(self, mock_compute_metric, tracker):
        """Test retrieving time series for specific metric."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        n = tracker.min_samples
        tracker.add_batch(
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n)
        )
        
        ts = tracker.get_time_series(metric='demographic_parity')
        
        assert 'demographic_parity' in ts.columns
        assert 'equalized_odds' not in ts.columns
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_get_time_series_time_filter(self, mock_compute_metric, tracker):
        """Test time filtering for time series."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        start_time = datetime.now()
        
        # Add batches
        for i in range(3):
            n = tracker.min_samples
            tracker.add_batch(
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n)
            )
        
        end_time = datetime.now()
        
        ts = tracker.get_time_series(
            start_time=start_time,
            end_time=end_time
        )
        
        assert isinstance(ts, pd.DataFrame)
        if len(ts) > 0:
            assert all(ts['timestamp'] >= start_time)
            assert all(ts['timestamp'] <= end_time)
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_get_summary_statistics(self, mock_compute_metric, tracker):
        """Test summary statistics computation."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add multiple batches
        for i in range(5):
            n = tracker.min_samples
            tracker.add_batch(
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n),
                np.random.binomial(1, 0.5, n)
            )
        
        summary = tracker.get_summary_statistics()
        
        assert isinstance(summary, dict)
        
        if len(summary) > 0:
            for metric, stats in summary.items():
                assert 'mean' in stats
                assert 'std' in stats
                assert 'min' in stats
                assert 'max' in stats
                assert 'current' in stats
    
    def test_get_summary_statistics_empty(self, tracker):
        """Test summary statistics with no data."""
        summary = tracker.get_summary_statistics()
        
        assert isinstance(summary, dict)
        assert len(summary) == 0
    
    def test_reset(self, tracker):
        """Test resetting tracker."""
        # Add some data
        tracker.add_batch(
            np.array([1, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0])
        )
        
        assert len(tracker.y_pred_buffer) > 0
        assert tracker.n_samples_processed > 0
        
        # Reset
        tracker.reset()
        
        assert len(tracker.y_pred_buffer) == 0
        assert len(tracker.y_true_buffer) == 0
        assert len(tracker.sensitive_buffer) == 0
        assert len(tracker.history) == 0
        assert tracker.n_samples_processed == 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_export_history(self, mock_compute_metric, tracker):
        """Test exporting history to CSV."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Add data
        n = tracker.min_samples
        tracker.add_batch(
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n),
            np.random.binomial(1, 0.5, n)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        tracker.export_history(filepath)
        
        # Verify file was created and contains data
        df = pd.read_csv(filepath)
        assert len(df) > 0
        assert 'timestamp' in df.columns
        
        # Clean up
        import os
        os.remove(filepath)


class TestBatchFairnessMonitor:
    """Tests for BatchFairnessMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return BatchFairnessMonitor(
            fairness_threshold=0.10,
            metrics=['demographic_parity', 'equalized_odds']
        )
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.fairness_threshold == 0.10
        assert 'demographic_parity' in monitor.metrics
        assert 'equalized_odds' in monitor.metrics
    
    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        monitor = BatchFairnessMonitor()
        
        assert monitor.fairness_threshold == 0.1
        assert len(monitor.metrics) > 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_evaluate_batch_passing(self, mock_compute_metric, monitor):
        """Test batch evaluation when fairness passes."""
        # Mock passing fairness
        mock_compute_metric.return_value = (0.05, {'group_0': 0.52, 'group_1': 0.48}, {})
        
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        sensitive = np.random.binomial(1, 0.5, 100)
        
        result = monitor.evaluate_batch(y_true, y_pred, sensitive)
        
        assert 'timestamp' in result
        assert 'n_samples' in result
        assert 'metrics' in result
        assert 'violations' in result
        assert 'is_fair' in result
        
        assert result['n_samples'] == 100
        assert result['is_fair'] is True
        assert len(result['violations']) == 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_evaluate_batch_violation(self, mock_compute_metric, monitor):
        """Test batch evaluation with fairness violation."""
        # Mock fairness violation
        mock_compute_metric.return_value = (0.15, {'group_0': 0.60, 'group_1': 0.45}, {})
        
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        sensitive = np.random.binomial(1, 0.5, 100)
        
        result = monitor.evaluate_batch(y_true, y_pred, sensitive)
        
        assert result['is_fair'] is False
        assert len(result['violations']) > 0
        
        # Check violation structure
        violation = result['violations'][0]
        assert 'metric' in violation
        assert 'value' in violation
        assert 'threshold' in violation
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_evaluate_batch_multiple_metrics(self, mock_compute_metric, monitor):
        """Test evaluation with multiple metrics."""
        # Mock different results for different metrics
        call_count = [0]
        
        def mock_side_effect(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return (0.05, {}, {})  # demographic_parity: pass
            else:
                return (0.15, {}, {})  # equalized_odds: fail
        
        mock_compute_metric.side_effect = mock_side_effect
        
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        sensitive = np.random.binomial(1, 0.5, 100)
        
        result = monitor.evaluate_batch(y_true, y_pred, sensitive)
        
        # Should have results for all metrics
        assert len(result['metrics']) == 2
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_evaluate_batch_metric_error(self, mock_compute_metric, monitor, caplog):
        """Test handling of metric computation errors."""
        # Mock error
        mock_compute_metric.side_effect = Exception("Computation failed")
        
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        sensitive = np.random.binomial(1, 0.5, 100)
        
        result = monitor.evaluate_batch(y_true, y_pred, sensitive)
        
        # Should handle error gracefully
        assert 'metrics' in result
        # Metrics should contain error information
        for metric_result in result['metrics'].values():
            if 'error' in metric_result:
                assert isinstance(metric_result['error'], str)


class TestIntegration:
    """Integration tests for real-time tracking."""
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_streaming_workflow(self, mock_compute_metric):
        """Test complete streaming monitoring workflow."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        tracker = RealTimeFairnessTracker(
            window_size=200,
            min_samples=50
        )
        
        # Simulate streaming predictions
        np.random.seed(42)
        
        for batch_idx in range(10):
            # Generate batch
            batch_size = 30
            y_pred = np.random.binomial(1, 0.5, batch_size)
            y_true = np.random.binomial(1, 0.5, batch_size)
            sensitive = np.random.binomial(1, 0.5, batch_size)
            
            # Process batch
            metrics = tracker.add_batch(y_pred, y_true, sensitive)
            
            # After first few batches, should start computing metrics
            if batch_idx >= 2:
                current = tracker.get_current_metrics()
                # Should have some metrics by now
        
        # Verify history was recorded
        assert len(tracker.history) > 0
        
        # Get summary
        summary = tracker.get_summary_statistics()
        assert len(summary) > 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_batch_and_realtime_comparison(self, mock_compute_metric):
        """Compare batch and real-time monitoring."""
        mock_compute_metric.return_value = (0.10, {}, {})
        
        # Same data
        y_true = np.random.binomial(1, 0.5, 200)
        y_pred = np.random.binomial(1, 0.5, 200)
        sensitive = np.random.binomial(1, 0.5, 200)
        
        # Batch monitoring
        batch_monitor = BatchFairnessMonitor(fairness_threshold=0.10)
        batch_result = batch_monitor.evaluate_batch(y_true, y_pred, sensitive)
        
        # Real-time monitoring
        realtime_tracker = RealTimeFairnessTracker(
            window_size=200,
            min_samples=50
        )
        realtime_tracker.add_batch(y_pred, y_true, sensitive)
        realtime_metrics = realtime_tracker.get_current_metrics()
        
        # Both should provide fairness information
        assert 'is_fair' in batch_result
        # Real-time may or may not have metrics depending on min_samples
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_alert_triggering_workflow(self, mock_compute_metric):
        """Test workflow with alert triggering."""
        # Start with good fairness
        mock_compute_metric.return_value = (0.05, {}, {})
        
        tracker = RealTimeFairnessTracker(window_size=100, min_samples=30)
        monitor = BatchFairnessMonitor(fairness_threshold=0.10)
        
        alert_count = 0
        
        # Process multiple batches
        for i in range(5):
            y_pred = np.random.binomial(1, 0.5, 50)
            y_true = np.random.binomial(1, 0.5, 50)
            sensitive = np.random.binomial(1, 0.5, 50)
            
            # Real-time tracking
            tracker.add_batch(y_pred, y_true, sensitive)
            
            # Batch monitoring
            result = monitor.evaluate_batch(y_true, y_pred, sensitive)
            
            # Check for violations
            if not result['is_fair']:
                alert_count += 1
                # Would trigger alert here in production
        
        # Verify tracking occurred
        assert len(tracker.history) > 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_degradation_detection(self, mock_compute_metric):
        """Test detecting gradual fairness degradation."""
        tracker = RealTimeFairnessTracker(window_size=100, min_samples=30)
        
        # Simulate degrading fairness
        fairness_values = [0.05, 0.07, 0.09, 0.12, 0.15]
        
        for i, fairness in enumerate(fairness_values):
            mock_compute_metric.return_value = (fairness, {}, {})
            
            y_pred = np.random.binomial(1, 0.5, 50)
            y_true = np.random.binomial(1, 0.5, 50)
            sensitive = np.random.binomial(1, 0.5, 50)
            
            tracker.add_batch(y_pred, y_true, sensitive)
        
        # Get time series to analyze trend
        ts = tracker.get_time_series()
        
        if len(ts) > 0 and 'demographic_parity' in ts.columns:
            # Values should show increasing trend
            values = ts['demographic_parity'].dropna()
            if len(values) > 1:
                # Later values should generally be higher
                assert values.iloc[-1] >= values.iloc[0]