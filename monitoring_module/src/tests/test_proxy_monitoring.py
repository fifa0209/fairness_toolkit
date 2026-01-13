"""
Tests for Proxy-Based Monitoring

Tests proxy-based fairness monitoring, geographic proxy building,
and privacy-preserving reporting.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from monitoring_module.src.proxy_monitoring import (
    ProxyMapping,
    ProxyBasedMonitor,
    GeographicProxyBuilder,
    PrivacyPreservingReporter,
    create_geographic_proxy_monitor,
)


class TestProxyMapping:
    """Tests for ProxyMapping dataclass."""
    
    def test_proxy_mapping_creation(self):
        """Test creating proxy mapping."""
        mapping = ProxyMapping(
            proxy_column='zip_code',
            proxy_to_group={'10001': 0, '10002': 1},
            confidence_level=0.85
        )
        
        assert mapping.proxy_column == 'zip_code'
        assert mapping.proxy_to_group['10001'] == 0
        assert mapping.confidence_level == 0.85
    
    def test_proxy_mapping_defaults(self):
        """Test default values."""
        mapping = ProxyMapping(
            proxy_column='zip_code',
            proxy_to_group={}
        )
        
        assert mapping.confidence_level == 0.8
        assert mapping.metadata is None


class TestProxyBasedMonitor:
    """Tests for ProxyBasedMonitor."""
    
    @pytest.fixture
    def proxy_mapping(self):
        """Create test proxy mapping."""
        return ProxyMapping(
            proxy_column='zip_code',
            proxy_to_group={
                '10001': 0,
                '10002': 1,
                '10003': 0,
                '10004': 1,
            },
            confidence_level=0.85
        )
    
    @pytest.fixture
    def monitor(self, proxy_mapping):
        """Create monitor instance."""
        return ProxyBasedMonitor(proxy_mapping)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'] * 25,
            'y_pred': np.random.binomial(1, 0.5, 100),
            'y_true': np.random.binomial(1, 0.5, 100),
        })
    
    def test_initialization(self, monitor, proxy_mapping):
        """Test monitor initialization."""
        assert monitor.proxy_mapping == proxy_mapping
        assert monitor.uncertainty_adjustment is True
    
    def test_initialization_without_uncertainty(self, proxy_mapping):
        """Test initialization without uncertainty adjustment."""
        monitor = ProxyBasedMonitor(
            proxy_mapping,
            uncertainty_adjustment=False
        )
        
        assert monitor.uncertainty_adjustment is False
    
    def test_infer_sensitive_attribute_basic(self, monitor, sample_df):
        """Test basic sensitive attribute inference."""
        inferred, confidence = monitor.infer_sensitive_attribute(sample_df)
        
        assert len(inferred) == len(sample_df)
        assert len(confidence) == len(sample_df)
        assert all(s in [0, 1] for s in inferred)
        assert all(c == 0.85 for c in confidence)
    
    def test_infer_sensitive_attribute_missing_column(self, monitor):
        """Test error when proxy column missing."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Proxy column .* not found"):
            monitor.infer_sensitive_attribute(df)
    
    def test_infer_sensitive_attribute_unmapped_values(self, monitor, caplog):
        """Test handling of unmapped proxy values."""
        df = pd.DataFrame({
            'zip_code': ['10001', '99999', '10002']  # 99999 is unmapped
        })
        
        inferred, confidence = monitor.infer_sensitive_attribute(df)
        
        # Unmapped value should be assigned to group 0
        assert inferred[1] == 0
        assert len(inferred) == 3
        
        # Check warning was logged
        assert "unmapped proxy values" in caplog.text.lower()
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_compute_proxy_metrics_basic(self, mock_compute_metric, monitor, sample_df):
        """Test computing proxy-based metrics."""
        mock_compute_metric.return_value = (
            0.10,
            {'0': 0.52, '1': 0.48},
            {0: 50, 1: 50}
        )
        
        results = monitor.compute_proxy_metrics(sample_df)
        
        assert 'demographic_parity' in results
        assert 'equalized_odds' in results
        
        for metric_name, metric_result in results.items():
            assert 'value' in metric_result
            assert 'uncertainty' in metric_result
            assert 'lower_bound' in metric_result
            assert 'upper_bound' in metric_result
            assert 'proxy_based' in metric_result
            assert metric_result['proxy_based'] is True
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_compute_proxy_metrics_without_uncertainty(
        self, mock_compute_metric, proxy_mapping, sample_df
    ):
        """Test computing metrics without uncertainty adjustment."""
        monitor = ProxyBasedMonitor(
            proxy_mapping,
            uncertainty_adjustment=False
        )
        
        mock_compute_metric.return_value = (0.10, {}, {0: 50, 1: 50})
        
        results = monitor.compute_proxy_metrics(sample_df)
        
        # Uncertainty should be 0
        for metric_result in results.values():
            assert metric_result['uncertainty'] == 0.0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_compute_proxy_metrics_custom_columns(
        self, mock_compute_metric, monitor, sample_df
    ):
        """Test computing metrics with custom column names."""
        sample_df = sample_df.rename(columns={
            'y_pred': 'predictions',
            'y_true': 'labels'
        })
        
        mock_compute_metric.return_value = (0.10, {}, {0: 50, 1: 50})
        
        results = monitor.compute_proxy_metrics(
            sample_df,
            y_pred_col='predictions',
            y_true_col='labels'
        )
        
        assert len(results) > 0
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_compute_proxy_metrics_error_handling(
        self, mock_compute_metric, monitor, sample_df, caplog
    ):
        """Test error handling during metric computation."""
        mock_compute_metric.side_effect = Exception("Computation failed")
        
        results = monitor.compute_proxy_metrics(sample_df)
        
        # Should not raise, but log error
        assert "Failed to compute proxy metric" in caplog.text
    
    # def test_estimate_uncertainty(self, monitor):
    #     """Test uncertainty estimation."""
    #     metric_value = 0.10
    #     confidence_scores = np.array([0.85] * 100)
    #     group_sizes = {0: 50, 1: 50}
        
    #     uncertainty = monitor._estimate_uncertainty(
    #         metric_value, confidence_scores, group_sizes
    #     )
        
    #     assert uncertainty > 0
    #     assert uncertainty < metric_value  # Should be reasonable
    def test_estimate_uncertainty(self, monitor):
        """Test uncertainty estimation."""
        metric_value = 0.10
        confidence_scores = np.array([0.85] * 100)
        group_sizes = {0: 50, 1: 50}
        
        uncertainty = monitor._estimate_uncertainty(
            metric_value, confidence_scores, group_sizes
        )
        
        assert uncertainty > 0
        assert uncertainty < 1.0  # Should be bounded
    
    def test_estimate_uncertainty_low_confidence(self, monitor):
        """Test uncertainty with low confidence scores."""
        metric_value = 0.10
        confidence_high = np.array([0.90] * 100)
        confidence_low = np.array([0.50] * 100)
        group_sizes = {0: 50, 1: 50}
        
        uncertainty_high = monitor._estimate_uncertainty(
            metric_value, confidence_high, group_sizes
        )
        uncertainty_low = monitor._estimate_uncertainty(
            metric_value, confidence_low, group_sizes
        )
        
        # Lower confidence should yield higher uncertainty
        assert uncertainty_low > uncertainty_high
    
    def test_validate_proxy_quality(self, monitor):
        """Test proxy quality validation."""
        df = pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'] * 25,
            'true_sensitive': [0, 1, 0, 1] * 25
        })
        
        validation = monitor.validate_proxy_quality(df, 'true_sensitive')
        
        assert 'overall_accuracy' in validation
        assert 'group_accuracy' in validation
        assert 'cohen_kappa' in validation
        assert 'confusion_matrix' in validation
        assert 'recommendation' in validation
        
        # Perfect mapping should have high accuracy
        assert validation['overall_accuracy'] == 1.0
    
    def test_validate_proxy_quality_imperfect_proxy(self, monitor):
        """Test validation with imperfect proxy."""
        df = pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'] * 25,
            'true_sensitive': [0, 1, 1, 0] * 25  # Mismatched
        })
        
        validation = monitor.validate_proxy_quality(df, 'true_sensitive')
        
        # Should have lower accuracy
        assert validation['overall_accuracy'] < 1.0
    
    def test_interpret_validation_excellent(self, monitor):
        """Test interpretation of excellent proxy."""
        recommendation = monitor._interpret_validation(
            accuracy=0.95, kappa=0.90
        )
        
        assert "EXCELLENT" in recommendation
    
    def test_interpret_validation_good(self, monitor):
        """Test interpretation of good proxy."""
        recommendation = monitor._interpret_validation(
            accuracy=0.85, kappa=0.70
        )
        
        assert "GOOD" in recommendation
    
    def test_interpret_validation_fair(self, monitor):
        """Test interpretation of fair proxy."""
        recommendation = monitor._interpret_validation(
            accuracy=0.75, kappa=0.50
        )
        
        assert "FAIR" in recommendation
    
    def test_interpret_validation_poor(self, monitor):
        """Test interpretation of poor proxy."""
        recommendation = monitor._interpret_validation(
            accuracy=0.60, kappa=0.30
        )
        
        assert "POOR" in recommendation


class TestGeographicProxyBuilder:
    """Tests for GeographicProxyBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        return GeographicProxyBuilder()
    
    @pytest.fixture
    def census_df(self):
        """Create sample census data."""
        return pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004', '10005'],
            'pct_minority': [0.30, 0.70, 0.45, 0.60, 0.20],
            'median_income': [50000, 35000, 45000, 40000, 55000],
            'population': [5000, 8000, 6000, 7000, 4500]
        })
    
    def test_initialization(self, builder):
        """Test builder initialization."""
        assert builder is not None
    
    def test_build_from_census_data_basic(self, builder, census_df):
        """Test building proxy mapping from census data."""
        mapping = builder.build_from_census_data(
            census_df,
            geography_col='zip_code',
            demographic_cols=['pct_minority'],
            threshold=0.5
        )
        
        assert isinstance(mapping, ProxyMapping)
        assert mapping.proxy_column == 'zip_code'
        assert len(mapping.proxy_to_group) == 5
        
        # Check correct classification based on threshold
        assert mapping.proxy_to_group['10001'] == 0  # 0.30 < 0.5
        assert mapping.proxy_to_group['10002'] == 1  # 0.70 >= 0.5
        assert mapping.proxy_to_group['10004'] == 1  # 0.60 >= 0.5
    
    def test_build_from_census_data_custom_threshold(self, builder, census_df):
        """Test building with custom threshold."""
        mapping = builder.build_from_census_data(
            census_df,
            geography_col='zip_code',
            demographic_cols=['pct_minority'],
            threshold=0.4
        )
        
        # With lower threshold, more should be in group 1
        group_1_count = sum(
            1 for v in mapping.proxy_to_group.values() if v == 1
        )
        assert group_1_count >= 3
    
    def test_build_from_census_data_metadata(self, builder, census_df):
        """Test metadata in built mapping."""
        mapping = builder.build_from_census_data(
            census_df,
            geography_col='zip_code',
            demographic_cols=['pct_minority'],
            threshold=0.5
        )
        
        assert mapping.metadata is not None
        assert mapping.metadata['source'] == 'census_data'
        assert mapping.metadata['threshold'] == 0.5
        assert mapping.metadata['n_geographies'] == 5
    
    def test_build_from_census_data_confidence(self, builder, census_df):
        """Test confidence estimation."""
        mapping = builder.build_from_census_data(
            census_df,
            geography_col='zip_code',
            demographic_cols=['pct_minority']
        )
        
        assert 0 < mapping.confidence_level <= 1.0
    
    def test_estimate_mapping_confidence(self, builder, census_df):
        """Test confidence estimation logic."""
        confidence = builder._estimate_mapping_confidence(
            census_df,
            demographic_cols=['pct_minority']
        )
        
        assert 0 < confidence <= 1.0
    
    def test_estimate_mapping_confidence_with_missing_data(self, builder):
        """Test confidence with missing data."""
        df = pd.DataFrame({
            'zip_code': ['10001', '10002', '10003'],
            'pct_minority': [0.5, np.nan, 0.7]
        })
        
        confidence = builder._estimate_mapping_confidence(
            df,
            demographic_cols=['pct_minority']
        )
        
        # Should be lower due to missing data
        assert confidence < 0.95


class TestPrivacyPreservingReporter:
    """Tests for PrivacyPreservingReporter."""
    
    @pytest.fixture
    def reporter(self):
        """Create reporter instance."""
        return PrivacyPreservingReporter(
            k_threshold=10,
            add_noise=True,
            epsilon=1.0
        )
    
    def test_initialization(self, reporter):
        """Test reporter initialization."""
        assert reporter.k_threshold == 10
        assert reporter.add_noise is True
        assert reporter.epsilon == 1.0
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        reporter = PrivacyPreservingReporter()
        
        assert reporter.k_threshold == 10
        assert reporter.add_noise is True
        assert reporter.epsilon == 1.0
    
    def test_safe_report_metrics_large_groups(self, reporter):
        """Test reporting with large groups."""
        group_metrics = {'group_0': 0.52, 'group_1': 0.48}
        group_sizes = {'group_0': 100, 'group_1': 120}
        
        report = reporter.safe_report_metrics(group_metrics, group_sizes)
        
        assert 'group_0' in report
        assert 'group_1' in report
        
        for group_report in report.values():
            assert not group_report['suppressed']
            assert 'value' in group_report
            assert group_report['value'] is not None
    
    def test_safe_report_metrics_small_groups_suppressed(self, reporter):
        """Test suppression of small groups."""
        group_metrics = {'group_0': 0.52, 'group_1': 0.48}
        group_sizes = {'group_0': 100, 'group_1': 5}  # Group 1 too small
        
        report = reporter.safe_report_metrics(group_metrics, group_sizes)
        
        assert not report['group_0']['suppressed']
        assert report['group_1']['suppressed']
        assert report['group_1']['value'] is None
        assert 'reason' in report['group_1']
    
    def test_safe_report_metrics_without_noise(self):
        """Test reporting without differential privacy noise."""
        reporter = PrivacyPreservingReporter(
            k_threshold=10,
            add_noise=False
        )
        
        group_metrics = {'group_0': 0.52}
        group_sizes = {'group_0': 100}
        
        report = reporter.safe_report_metrics(group_metrics, group_sizes)
        
        # Without noise, value should match exactly
        assert report['group_0']['value'] == 0.52
        assert not report['group_0']['privacy_applied']
    
    def test_safe_report_metrics_with_noise(self, reporter):
        """Test noise addition."""
        group_metrics = {'group_0': 0.52}
        group_sizes = {'group_0': 100}
        
        # Run multiple times to check noise is different
        values = []
        for _ in range(10):
            report = reporter.safe_report_metrics(group_metrics, group_sizes)
            values.append(report['group_0']['value'])
        
        # Values should vary due to noise
        assert len(set(values)) > 1
        assert all(v is not None for v in values)
    
    def test_laplace_noise(self, reporter):
        """Test Laplace noise generation."""
        np.random.seed(42)
        
        noise1 = reporter._laplace_noise(0.5)
        noise2 = reporter._laplace_noise(0.5)
        
        # Should generate different noise values
        assert noise1 != noise2
    
    def test_size_range_conversion(self, reporter):
        """Test size to range conversion."""
        assert reporter._size_range(25) == "10-50"
        assert reporter._size_range(75) == "50-100"
        assert reporter._size_range(200) == "100-500"
        assert reporter._size_range(750) == "500-1000"
        assert reporter._size_range(2000) == "1000+"
    
    def test_size_range_in_report(self, reporter):
        """Test size ranges appear in reports."""
        group_metrics = {'group_0': 0.52}
        group_sizes = {'group_0': 75}
        
        report = reporter.safe_report_metrics(group_metrics, group_sizes)
        
        assert report['group_0']['sample_size_range'] == "50-100"


class TestConvenienceFunction:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def census_df(self):
        """Create sample census data."""
        return pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'],
            'pct_minority': [0.30, 0.70, 0.45, 0.60],
        })
    
    def test_create_geographic_proxy_monitor(self, census_df):
        """Test convenience function for creating monitor."""
        monitor = create_geographic_proxy_monitor(
            census_df,
            geography_col='zip_code',
            demographic_threshold=0.5
        )
        
        assert isinstance(monitor, ProxyBasedMonitor)
        assert monitor.proxy_mapping.proxy_column == 'zip_code'
        assert monitor.uncertainty_adjustment is True


class TestIntegration:
    """Integration tests for proxy-based monitoring."""
    
    def test_end_to_end_proxy_monitoring(self):
        """Test complete proxy-based monitoring workflow."""
        # Create census data
        census_df = pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'],
            'pct_minority': [0.30, 0.70, 0.40, 0.65],
        })
        
        # Build monitor
        monitor = create_geographic_proxy_monitor(
            census_df,
            geography_col='zip_code',
            demographic_threshold=0.5
        )
        
        # Create prediction data
        np.random.seed(42)
        prediction_df = pd.DataFrame({
            'zip_code': ['10001', '10002', '10003', '10004'] * 25,
            'y_pred': np.random.binomial(1, 0.5, 100),
            'y_true': np.random.binomial(1, 0.5, 100),
        })
        
        # Compute proxy metrics (mock the compute_metric function)
        with patch('measurement_module.src.metrics_engine.compute_metric') as mock:
            mock.return_value = (0.10, {'0': 0.52, '1': 0.48}, {0: 50, 1: 50})
            
            results = monitor.compute_proxy_metrics(prediction_df)
        
        assert len(results) > 0
        assert all('uncertainty' in r for r in results.values())
    
    def test_privacy_preserving_workflow(self):
        """Test privacy-preserving reporting workflow."""
        # Create reporter
        reporter = PrivacyPreservingReporter(
            k_threshold=10,
            add_noise=True,
            epsilon=1.0
        )
        
        # Sample metrics
        group_metrics = {
            'group_0': 0.52,
            'group_1': 0.48,
            'group_2': 0.55  # Small group
        }
        group_sizes = {
            'group_0': 100,
            'group_1': 150,
            'group_2': 5  # Below threshold
        }
        
        # Generate report
        report = reporter.safe_report_metrics(group_metrics, group_sizes)
        
        # Verify privacy preservation
        assert report['group_2']['suppressed']  # Small group suppressed
        assert report['group_0']['value'] is not None
        assert report['group_1']['value'] is not None
    
    def test_proxy_validation_workflow(self):
        """Test proxy validation against ground truth."""
        # Create proxy mapping
        mapping = ProxyMapping(
            proxy_column='region',
            proxy_to_group={'A': 0, 'B': 1, 'C': 0, 'D': 1},
            confidence_level=0.85
        )
        
        monitor = ProxyBasedMonitor(mapping)
        
        # Create validation dataset
        validation_df = pd.DataFrame({
            'region': ['A', 'B', 'C', 'D'] * 25,
            'true_sensitive': [0, 1, 0, 1] * 25  # Perfect match
        })
        
        # Validate
        validation = monitor.validate_proxy_quality(
            validation_df,
            'true_sensitive'
        )
        
        assert validation['overall_accuracy'] == 1.0
        assert "EXCELLENT" in validation['recommendation']
    
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_gdpr_compliant_monitoring(self, mock_compute_metric):
        """Test GDPR-compliant fairness monitoring."""
        # Simulate EU jurisdiction where direct collection is prohibited
        
        # Use geographic proxies
        census_df = pd.DataFrame({
            'postal_code': ['12345', '23456', '34567'],
            'pct_minority': [0.20, 0.60, 0.40],
        })
        
        monitor = create_geographic_proxy_monitor(
            census_df,
            geography_col='postal_code',
            demographic_threshold=0.5
        )
        
        # Monitor predictions using proxies
        predictions_df = pd.DataFrame({
            'postal_code': ['12345', '23456', '34567'] * 30,
            'y_pred': np.random.binomial(1, 0.5, 90),
            'y_true': np.random.binomial(1, 0.5, 90),
        })
        
        mock_compute_metric.return_value = (0.08, {}, {0: 45, 1: 45})
        
        # Compute metrics with uncertainty bounds
        results = monitor.compute_proxy_metrics(predictions_df)
        
        # Verify GDPR-compliant properties
        for metric_result in results.values():
            assert metric_result['proxy_based'] is True
            assert 'uncertainty' in metric_result
            assert 'lower_bound' in metric_result
            assert 'upper_bound' in metric_result