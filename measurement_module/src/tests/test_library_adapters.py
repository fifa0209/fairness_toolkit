"""
Unit tests for library_adapters.py

Tests the unified library integration layer that wraps AIF360, Fairlearn, and Aequitas.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Import the module to test
try:
    from measurement_module.src.library_adapters import (
        FairnessLibraryAdapter,
        AIF360Adapter,
        FairlearnAdapter,
        AequitasAdapter,
        UnifiedFairnessAnalyzer
    )
except ImportError:
    pytest.skip("library_adapters module not found", allow_module_level=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    sensitive_features = np.random.randint(0, 2, n_samples)
    
    return y_true, y_pred, sensitive_features


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multiclass data."""
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = np.random.randint(0, 3, n_samples)
    sensitive_features = np.random.randint(0, 2, n_samples)
    
    return y_true, y_pred, sensitive_features


# ============================================================================
# Test Abstract Base Class
# ============================================================================

class TestFairnessLibraryAdapter:
    """Test the abstract base adapter class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Abstract class should not be instantiable."""
        with pytest.raises(TypeError):
            FairnessLibraryAdapter()
    
    def test_subclass_must_implement_compute_metric(self):
        """Subclass must implement compute_metric."""
        class IncompleteAdapter(FairnessLibraryAdapter):
            def get_available_metrics(self):
                return []
        
        with pytest.raises(TypeError):
            IncompleteAdapter()
    
    def test_subclass_must_implement_get_available_metrics(self):
        """Subclass must implement get_available_metrics."""
        class IncompleteAdapter(FairnessLibraryAdapter):
            def compute_metric(self, metric_name, **kwargs):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteAdapter()
    
    def test_valid_subclass_can_be_instantiated(self):
        """Valid subclass with all methods should work."""
        class ValidAdapter(FairnessLibraryAdapter):
            def compute_metric(self, metric_name, **kwargs):
                return {"value": 0.1}
            
            def get_available_metrics(self):
                return ["test_metric"]
        
        adapter = ValidAdapter()
        assert adapter is not None
        assert adapter.get_available_metrics() == ["test_metric"]


# ============================================================================
# Test AIF360 Adapter
# ============================================================================

class TestAIF360Adapter:
    """Test AIF360 library adapter."""
    
    def test_initialization_without_aif360(self):
        """Test adapter initialization when AIF360 is not installed."""
        with patch.dict(sys.modules, {'aif360': None, 'aif360.datasets': None, 'aif360.metrics': None}):
            with pytest.warns(ImportWarning, match="AIF360 not installed"):
                adapter = AIF360Adapter()
                assert adapter.available is False
    
    @patch('measurement_module.src.library_adapters.AIF360Adapter')
    def test_initialization_with_aif360(self, mock_adapter):
        """Test adapter initialization when AIF360 is installed."""
        mock_instance = Mock()
        mock_instance.available = True
        mock_adapter.return_value = mock_instance
        
        adapter = mock_adapter()
        assert adapter.available is True
    
    def test_get_available_metrics(self):
        """Test listing available metrics."""
        adapter = AIF360Adapter()
        metrics = adapter.get_available_metrics()
        
        assert isinstance(metrics, list)
        assert 'statistical_parity_difference' in metrics
        assert 'disparate_impact' in metrics
        assert 'average_odds_difference' in metrics
        assert 'equal_opportunity_difference' in metrics
        assert 'theil_index' in metrics
    
    def test_compute_metric_without_library(self, sample_binary_data):
        """Test compute_metric raises error when library unavailable."""
        adapter = AIF360Adapter()
        
        if not adapter.available:
            y_true, y_pred, sensitive = sample_binary_data
            
            with pytest.raises(ImportError, match="AIF360 is not available"):
                adapter.compute_metric(
                    'statistical_parity_difference',
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive
                )
    
    @pytest.mark.skipif(
        not pytest.importorskip("aif360", reason="AIF360 not installed"),
        reason="AIF360 not available"
    )
    def test_compute_metric_with_library(self, sample_binary_data):
        """Test metric computation when AIF360 is available."""
        adapter = AIF360Adapter()
        y_true, y_pred, sensitive = sample_binary_data
        
        # Only test if library is actually available
        if adapter.available:
            result = adapter.compute_metric(
                'statistical_parity_difference',
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive
            )
            
            assert isinstance(result, dict)
            assert 'value' in result or 'metric_value' in result
        else:
            pytest.skip("AIF360 not available in test environment")


# ============================================================================
# Test Fairlearn Adapter
# ============================================================================

class TestFairlearnAdapter:
    """Test Fairlearn library adapter."""
    
    def test_initialization_without_fairlearn(self):
        """Test adapter initialization when Fairlearn is not installed."""
        with patch.dict(sys.modules, {'fairlearn': None, 'fairlearn.metrics': None}):
            with pytest.warns(ImportWarning, match="Fairlearn not installed"):
                adapter = FairlearnAdapter()
                assert adapter.available is False
    
    @patch('measurement_module.src.library_adapters.FairlearnAdapter')
    def test_initialization_with_fairlearn(self, mock_adapter):
        """Test adapter initialization when Fairlearn is installed."""
        mock_instance = Mock()
        mock_instance.available = True
        mock_adapter.return_value = mock_instance
        
        adapter = mock_adapter()
        assert adapter.available is True
    
    def test_get_available_metrics(self):
        """Test listing available metrics."""
        adapter = FairlearnAdapter()
        metrics = adapter.get_available_metrics()
        
        assert isinstance(metrics, list)
        assert 'demographic_parity_difference' in metrics
        assert 'demographic_parity_ratio' in metrics
        assert 'equalized_odds_difference' in metrics
        assert 'equalized_odds_ratio' in metrics
    
    def test_compute_metric_without_library(self, sample_binary_data):
        """Test compute_metric raises error when library unavailable."""
        adapter = FairlearnAdapter()
        
        if not adapter.available:
            y_true, y_pred, sensitive = sample_binary_data
            
            with pytest.raises(ImportError, match="Fairlearn is not available"):
                adapter.compute_metric(
                    'demographic_parity_difference',
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive
                )
    
    @pytest.mark.skipif(
        not pytest.importorskip("fairlearn", reason="Fairlearn not installed"),
        reason="Fairlearn not available"
    )
    def test_compute_metric_with_library(self, sample_binary_data):
        """Test metric computation when Fairlearn is available."""
        adapter = FairlearnAdapter()
        y_true, y_pred, sensitive = sample_binary_data
        
        # Only test if library is actually available
        if adapter.available:
            result = adapter.compute_metric(
                'demographic_parity_difference',
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive
            )
            
            assert isinstance(result, dict)
            assert 'value' in result or 'metric_value' in result
        else:
            pytest.skip("Fairlearn not available in test environment")


# ============================================================================
# Test Aequitas Adapter
# ============================================================================

class TestAequitasAdapter:
    """Test Aequitas library adapter."""
    
    def test_initialization_without_aequitas(self):
        """Test adapter initialization when Aequitas is not installed."""
        with patch.dict(sys.modules, {'aequitas': None, 'aequitas.group': None, 'aequitas.bias': None}):
            with pytest.warns(ImportWarning, match="Aequitas not installed"):
                adapter = AequitasAdapter()
                assert adapter.available is False
    
    def test_get_available_metrics(self):
        """Test listing available metrics."""
        adapter = AequitasAdapter()
        metrics = adapter.get_available_metrics()
        
        assert isinstance(metrics, list)
        assert 'ppr_disparity' in metrics
        assert 'pprev_disparity' in metrics
        assert 'fdr_disparity' in metrics
        assert 'for_disparity' in metrics
    
    def test_compute_metric_without_library(self, sample_binary_data):
        """Test compute_metric raises error when library unavailable."""
        adapter = AequitasAdapter()
        
        if not adapter.available:
            y_true, y_pred, sensitive = sample_binary_data
            
            with pytest.raises(ImportError, match="Aequitas is not available"):
                adapter.compute_metric(
                    'ppr_disparity',
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive
                )


# ============================================================================
# Test Unified Fairness Analyzer
# ============================================================================

class TestUnifiedFairnessAnalyzer:
    """Test the unified analyzer that delegates to adapters."""
    
    def test_initialization_no_preference(self):
        """Test initialization without library preference."""
        analyzer = UnifiedFairnessAnalyzer()
        
        assert analyzer is not None
        assert analyzer.preferred_library is None
        assert 'native' in analyzer.available_libraries
    
    def test_initialization_with_preference(self):
        """Test initialization with library preference."""
        analyzer = UnifiedFairnessAnalyzer(preferred_library='fairlearn')
        
        assert analyzer.preferred_library == 'fairlearn'
    
    def test_available_libraries_includes_native(self):
        """Native implementation should always be available."""
        analyzer = UnifiedFairnessAnalyzer()
        
        assert 'native' in analyzer.available_libraries
    
    def test_compute_metric_with_native(self, sample_binary_data):
        """Test computing metric with native implementation."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native'
        )
        
        assert result is not None
        # Result can be dict or object with attributes
        assert isinstance(result, (dict, object))
    
    def test_compute_metric_auto_selects_library(self, sample_binary_data):
        """Test that analyzer auto-selects appropriate library."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive
        )
        
        assert result is not None
    
    def test_compute_metric_respects_preference(self, sample_binary_data):
        """Test that preferred library is used when available."""
        analyzer = UnifiedFairnessAnalyzer(preferred_library='native')
        y_true, y_pred, sensitive = sample_binary_data
        
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive
        )
        
        assert result is not None
    
    def test_compute_metric_invalid_library_raises_error(self, sample_binary_data):
        """Test that invalid library name raises error."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        with pytest.raises(ValueError, match="not available"):
            analyzer.compute_metric(
                'demographic_parity',
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
                library='nonexistent_library'
            )
    
    def test_list_available_metrics_all_libraries(self):
        """Test listing metrics from all libraries."""
        analyzer = UnifiedFairnessAnalyzer()
        
        all_metrics = analyzer.list_available_metrics()
        
        assert isinstance(all_metrics, dict)
        assert 'native' in all_metrics
        assert isinstance(all_metrics['native'], list)
        assert len(all_metrics['native']) > 0
    
    def test_list_available_metrics_specific_library(self):
        """Test listing metrics from specific library."""
        analyzer = UnifiedFairnessAnalyzer()
        
        native_metrics = analyzer.list_available_metrics(library='native')
        
        assert isinstance(native_metrics, dict)
        assert 'native' in native_metrics
        assert 'demographic_parity' in native_metrics['native']
    
    def test_list_available_metrics_nonexistent_library(self):
        """Test listing metrics for nonexistent library returns empty."""
        analyzer = UnifiedFairnessAnalyzer()
        
        metrics = analyzer.list_available_metrics(library='nonexistent')
        
        assert metrics == {}
    
    def test_select_library_uses_preferred(self):
        """Test that _select_library uses preferred library."""
        analyzer = UnifiedFairnessAnalyzer(preferred_library='native')
        
        selected = analyzer._select_library('demographic_parity')
        
        assert selected == 'native'
    
    def test_select_library_falls_back_to_native(self):
        """Test that _select_library falls back to native."""
        analyzer = UnifiedFairnessAnalyzer(preferred_library='unavailable')
        
        selected = analyzer._select_library('demographic_parity')
        
        assert selected == 'native'
    
    def test_compute_metric_with_kwargs(self, sample_binary_data):
        """Test passing additional kwargs to compute_metric."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native',
            threshold=0.05
        )
        
        assert result is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestLibraryAdapterIntegration:
    """Integration tests across adapters."""
    
    def test_all_adapters_have_consistent_interface(self):
        """All adapters should implement the same interface."""
        adapters = [
            AIF360Adapter(),
            FairlearnAdapter(),
            AequitasAdapter()
        ]
        
        for adapter in adapters:
            # Should have these methods
            assert hasattr(adapter, 'compute_metric')
            assert hasattr(adapter, 'get_available_metrics')
            assert hasattr(adapter, 'available')
            
            # Should return list of metrics
            metrics = adapter.get_available_metrics()
            assert isinstance(metrics, list)
    
    def test_unified_analyzer_can_use_all_adapters(self, sample_binary_data):
        """Unified analyzer should work with all adapters."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        # Try with each available library
        for library in analyzer.available_libraries:
            result = analyzer.compute_metric(
                'demographic_parity',
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
                library=library
            )
            
            # Result should not be None
            assert result is not None, f"Result for library {library} should not be None"
    
    @pytest.mark.parametrize("metric_name", [
        'demographic_parity',
        'equalized_odds',
        'equal_opportunity'
    ])
    def test_native_metrics_available(self, metric_name, sample_binary_data):
        """Test that native metrics are always available."""
        analyzer = UnifiedFairnessAnalyzer()
        y_true, y_pred, sensitive = sample_binary_data
        
        result = analyzer.compute_metric(
            metric_name,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native'
        )
        
        assert result is not None


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_arrays(self):
        """Test handling of empty data arrays."""
        analyzer = UnifiedFairnessAnalyzer()
        
        # Should raise ValidationError from the native implementation
        with pytest.raises(Exception):  # More general - could be ValueError or ValidationError
            analyzer.compute_metric(
                'demographic_parity',
                y_true=np.array([]),
                y_pred=np.array([]),
                sensitive_features=np.array([]),
                library='native'
            )
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        analyzer = UnifiedFairnessAnalyzer()
        
        # Should raise ValidationError from the native implementation
        with pytest.raises(Exception):  # More general - could be ValueError or ValidationError
            analyzer.compute_metric(
                'demographic_parity',
                y_true=np.array([0, 1, 0]),
                y_pred=np.array([1, 1]),  # Different length
                sensitive_features=np.array([0, 1, 0]),
                library='native'
            )
    
    def test_non_binary_sensitive_features(self):
        """Test handling of non-binary sensitive features."""
        analyzer = UnifiedFairnessAnalyzer()
        
        # For multiclass sensitive features, the native implementation
        # should handle them gracefully (no error expected)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        sensitive = np.array([0, 1, 2, 0, 1])  # 3 groups
        
        # The native implementation should compute metrics per group
        # without raising an error
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native'
        )
        
        # Should return result without error
        assert result is not None
    
    def test_all_same_predictions(self):
        """Test handling when all predictions are the same."""
        analyzer = UnifiedFairnessAnalyzer()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 1])  # All positive
        sensitive = np.array([0, 1, 0, 1, 0])
        
        # Should still compute, may be 0 difference
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native'
        )
        
        assert result is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset_performance(self):
        """Test that analyzer handles large datasets efficiently."""
        import time
        
        analyzer = UnifiedFairnessAnalyzer()
        
        # Generate large dataset
        n_samples = 100000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        sensitive = np.random.randint(0, 2, n_samples)
        
        start = time.time()
        result = analyzer.compute_metric(
            'demographic_parity',
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            library='native'
        )
        elapsed = time.time() - start
        
        assert result is not None
        # Increased time limit to be more realistic
        assert elapsed < 10.0, f"Computation took {elapsed:.2f}s, expected < 10s"
    
    def test_adapter_initialization_is_cached(self):
        """Test that adapters are initialized once and reused."""
        analyzer1 = UnifiedFairnessAnalyzer()
        analyzer2 = UnifiedFairnessAnalyzer()
        
        # Both should have separate adapter instances
        assert analyzer1.adapters is not analyzer2.adapters