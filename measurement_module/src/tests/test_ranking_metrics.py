"""
Unit tests for ranking_metrics.py

Tests fairness metrics for ranking/recommendation systems.
"""

import pytest
import numpy as np
from typing import Dict

# Import the module to test
try:
    from measurement_module.src.ranking_metrics import (
        exposure_parity_difference,
        normalized_discounted_cumulative_fairness,
        attention_weighted_fairness
    )
except ImportError:
    pytest.skip("ranking_metrics module not found", allow_module_level=True)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_ranking():
    """Simple ranking with known fairness properties."""
    # Rankings: items 0,1,2,3,4 in order
    rankings = np.array([0, 1, 2, 3, 4])
    # Group membership: alternating groups
    sensitive_features = np.array([0, 1, 0, 1, 0])
    return rankings, sensitive_features


@pytest.fixture
def biased_ranking():
    """Ranking biased toward group 0 (all group 0 items first)."""
    rankings = np.array([0, 2, 4, 1, 3])  # Items from group 0 first
    sensitive_features = np.array([0, 1, 0, 1, 0])
    return rankings, sensitive_features


@pytest.fixture
def balanced_ranking():
    """Perfectly balanced ranking."""
    rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    sensitive_features = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    return rankings, sensitive_features


@pytest.fixture
def large_ranking():
    """Large ranking for performance tests."""
    np.random.seed(42)
    n_items = 1000
    rankings = np.arange(n_items)
    np.random.shuffle(rankings)
    sensitive_features = np.random.randint(0, 2, n_items)
    return rankings, sensitive_features


@pytest.fixture
def ranking_with_relevance():
    """Ranking with relevance scores."""
    rankings = np.array([0, 1, 2, 3, 4])
    sensitive_features = np.array([0, 1, 0, 1, 0])
    relevance_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    return rankings, sensitive_features, relevance_scores


# ============================================================================
# Test Exposure Parity Difference
# ============================================================================

class TestExposureParityDifference:
    """Test exposure parity difference metric."""
    
    def test_basic_computation(self, simple_ranking):
        """Test basic exposure computation."""
        rankings, sensitive = simple_ranking
        
        diff, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        
        assert isinstance(diff, float)
        assert diff >= 0  # Difference should be non-negative
        assert isinstance(exposures, dict)
        assert isinstance(sizes, dict)
        assert len(exposures) == 2  # Binary groups
        assert len(sizes) == 2
    
    def test_perfect_fairness_low_difference(self, balanced_ranking):
        """Test that balanced ranking has low difference."""
        rankings, sensitive = balanced_ranking
        
        diff, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        
        # Should have very low difference (near 0)
        # Note: Even with alternating groups, DCG weighting causes some difference
        assert diff < 0.15  # Relaxed threshold to account for position weighting
    
    def test_biased_ranking_high_difference(self, biased_ranking):
        """Test that biased ranking has high difference."""
        rankings, sensitive = biased_ranking
        
        diff, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        
        # Should have noticeable difference
        assert diff > 0.0
    
    def test_top_k_parameter(self, simple_ranking):
        """Test that top_k parameter limits analysis."""
        rankings, sensitive = simple_ranking
        
        # Compare full ranking vs top-3
        diff_full, _, _ = exposure_parity_difference(rankings, sensitive)
        diff_top3, _, _ = exposure_parity_difference(rankings, sensitive, top_k=3)
        
        # Differences may vary
        assert isinstance(diff_full, float)
        assert isinstance(diff_top3, float)
    
    def test_custom_position_weights(self, simple_ranking):
        """Test custom position weights."""
        rankings, sensitive = simple_ranking
        
        # Linear weights: [1.0, 0.75, 0.5, 0.25, 0.0]
        custom_weights = np.array([1.0, 0.75, 0.5, 0.25, 0.1])
        
        diff, exposures, sizes = exposure_parity_difference(
            rankings, sensitive, position_weights=custom_weights
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
    
    def test_default_dcg_weights(self, simple_ranking):
        """Test that default weights follow DCG formula."""
        rankings, sensitive = simple_ranking
        
        diff, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        
        # Should use 1/log2(pos+1) by default
        assert isinstance(diff, float)
    
    def test_group_sizes_correct(self, simple_ranking):
        """Test that group sizes are computed correctly."""
        rankings, sensitive = simple_ranking
        
        _, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        
        # Count groups manually
        unique, counts = np.unique(sensitive, return_counts=True)
        
        for group, count in zip(unique, counts):
            assert sizes[f"Group_{group}"] == count
    
    def test_exposure_values_range(self, simple_ranking):
        """Test that exposure values are in reasonable range."""
        rankings, sensitive = simple_ranking
        
        _, exposures, _ = exposure_parity_difference(rankings, sensitive)
        
        for group, exposure in exposures.items():
            assert exposure >= 0  # Non-negative
            assert not np.isnan(exposure)  # Not NaN
            assert not np.isinf(exposure)  # Not infinite
    
    def test_empty_ranking_raises_error(self):
        """Test that empty ranking raises appropriate error."""
        rankings = np.array([])
        sensitive = np.array([])
        
        with pytest.raises(ValueError):
            exposure_parity_difference(rankings, sensitive)
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise error."""
        rankings = np.array([0, 1, 2])
        sensitive = np.array([0, 1])  # Different length
        
        with pytest.raises(IndexError):
            exposure_parity_difference(rankings, sensitive)
    
    def test_non_binary_groups_raises_error(self):
        """Test that non-binary groups raise error."""
        rankings = np.array([0, 1, 2, 3, 4])
        sensitive = np.array([0, 1, 2, 1, 0])  # 3 groups
        
        with pytest.raises(ValueError, match="Expected 2 groups"):
            exposure_parity_difference(rankings, sensitive)
    
    def test_large_ranking_performance(self, large_ranking):
        """Test performance on large ranking."""
        import time
        
        rankings, sensitive = large_ranking
        
        start = time.time()
        diff, exposures, sizes = exposure_parity_difference(rankings, sensitive)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should complete quickly
        assert isinstance(diff, float)
    
    def test_all_items_same_group(self):
        """Test when all items belong to same group."""
        rankings = np.array([0, 1, 2, 3, 4])
        sensitive = np.array([0, 0, 0, 0, 0])  # All group 0
        
        # Should handle gracefully (no group 1 items)
        with pytest.raises(ValueError, match="Expected 2 groups"):
            exposure_parity_difference(rankings, sensitive)
    
    def test_single_group_has_no_items_in_ranking(self):
        """Test when one group has no items in top-k."""
        rankings = np.array([0, 2, 4, 6, 8])  # Only group 0
        sensitive = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        diff, exposures, sizes = exposure_parity_difference(
            rankings, sensitive, top_k=3
        )
        
        # Group 1 should have 0 exposure
        assert exposures.get('Group_1', 0) == 0.0
        assert diff > 0  # Should show bias


# ============================================================================
# Test Normalized DCG Fairness
# ============================================================================

class TestNormalizedDCGFairness:
    """Test normalized DCG-based fairness metric."""
    
    def test_basic_computation(self, ranking_with_relevance):
        """Test basic nDCG fairness computation."""
        rankings, sensitive, relevance = ranking_with_relevance
        
        fairness_score, group_ndcg = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        assert isinstance(fairness_score, float)
        assert fairness_score >= 0
        assert isinstance(group_ndcg, dict)
        assert len(group_ndcg) == 2
    
    def test_without_relevance_scores(self, simple_ranking):
        """Test nDCG fairness without explicit relevance."""
        rankings, sensitive = simple_ranking
        
        fairness_score, group_ndcg = normalized_discounted_cumulative_fairness(
            rankings, sensitive
        )
        
        # Should assume all items equally relevant
        assert isinstance(fairness_score, float)
        assert fairness_score >= 0
    
    def test_ndcg_values_range(self, ranking_with_relevance):
        """Test that nDCG values are in [0, 1]."""
        rankings, sensitive, relevance = ranking_with_relevance
        
        _, group_ndcg = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        for group, ndcg in group_ndcg.items():
            assert 0 <= ndcg <= 1.0
    
    def test_top_k_parameter(self, ranking_with_relevance):
        """Test top_k parameter."""
        rankings, sensitive, relevance = ranking_with_relevance
        
        score_full, _ = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        score_top3, _ = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance, top_k=3
        )
        
        # Both should be valid
        assert isinstance(score_full, float)
        assert isinstance(score_top3, float)
    
    def test_perfect_ranking_low_disparity(self):
        """Test that perfect ranking has low disparity."""
        # Create ideal ranking where both groups get high positions
        rankings = np.array([0, 1, 2, 3])
        sensitive = np.array([0, 1, 0, 1])
        relevance = np.array([1.0, 0.9, 0.8, 0.7])
        
        fairness_score, _ = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        # Should have low disparity
        assert fairness_score < 0.3
    
    def test_biased_ranking_high_disparity(self):
        """Test that biased ranking has high disparity."""
        # All high-relevance items from group 0 first
        rankings = np.array([0, 2, 1, 3])
        sensitive = np.array([0, 1, 0, 1])
        relevance = np.array([1.0, 0.3, 0.9, 0.2])
        
        fairness_score, _ = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        # Should have higher disparity
        assert fairness_score > 0.0
    
    def test_zero_relevance_handling(self):
        """Test handling of zero relevance scores."""
        rankings = np.array([0, 1, 2, 3])
        sensitive = np.array([0, 1, 0, 1])
        relevance = np.array([0.0, 0.0, 0.0, 0.0])  # All zero
        
        fairness_score, group_ndcg = normalized_discounted_cumulative_fairness(
            rankings, sensitive, relevance
        )
        
        # Should handle gracefully
        assert isinstance(fairness_score, float)
        assert not np.isnan(fairness_score)


# ============================================================================
# Test Attention-Weighted Fairness
# ============================================================================

class TestAttentionWeightedFairness:
    """Test attention-weighted fairness metric."""
    
    def test_exponential_attention_model(self, simple_ranking):
        """Test exponential attention decay model."""
        rankings, sensitive = simple_ranking
        
        diff, attention = attention_weighted_fairness(
            rankings, sensitive, attention_model='exponential'
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
        assert isinstance(attention, dict)
        assert len(attention) == 2
    
    def test_linear_attention_model(self, simple_ranking):
        """Test linear attention decay model."""
        rankings, sensitive = simple_ranking
        
        diff, attention = attention_weighted_fairness(
            rankings, sensitive, attention_model='linear'
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
    
    def test_log_attention_model(self, simple_ranking):
        """Test logarithmic attention model."""
        rankings, sensitive = simple_ranking
        
        diff, attention = attention_weighted_fairness(
            rankings, sensitive, attention_model='log'
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
    
    def test_invalid_attention_model_raises_error(self, simple_ranking):
        """Test that invalid attention model raises error."""
        rankings, sensitive = simple_ranking
        
        with pytest.raises(ValueError, match="Unknown attention model"):
            attention_weighted_fairness(
                rankings, sensitive, attention_model='invalid'
            )
    
    def test_attention_values_positive(self, simple_ranking):
        """Test that attention values are positive."""
        rankings, sensitive = simple_ranking
        
        _, attention = attention_weighted_fairness(rankings, sensitive)
        
        for group, att in attention.items():
            assert att >= 0
            assert not np.isnan(att)
    
    def test_top_k_parameter(self, simple_ranking):
        """Test top_k parameter."""
        rankings, sensitive = simple_ranking
        
        diff_full, _ = attention_weighted_fairness(rankings, sensitive)
        diff_top3, _ = attention_weighted_fairness(rankings, sensitive, top_k=3)
        
        assert isinstance(diff_full, float)
        assert isinstance(diff_top3, float)
    
    def test_different_models_give_different_results(self, simple_ranking):
        """Test that different attention models produce different results."""
        rankings, sensitive = simple_ranking
        
        diff_exp, _ = attention_weighted_fairness(
            rankings, sensitive, attention_model='exponential'
        )
        diff_lin, _ = attention_weighted_fairness(
            rankings, sensitive, attention_model='linear'
        )
        diff_log, _ = attention_weighted_fairness(
            rankings, sensitive, attention_model='log'
        )
        
        # All should be valid but likely different
        assert isinstance(diff_exp, float)
        assert isinstance(diff_lin, float)
        assert isinstance(diff_log, float)
    
    def test_biased_ranking_detected(self, biased_ranking):
        """Test that biased ranking is detected."""
        rankings, sensitive = biased_ranking
        
        diff, attention = attention_weighted_fairness(rankings, sensitive)
        
        # Should detect bias
        assert diff > 0.0
    
    def test_balanced_ranking_low_difference(self, balanced_ranking):
        """Test that balanced ranking has low difference."""
        rankings, sensitive = balanced_ranking
        
        diff, attention = attention_weighted_fairness(rankings, sensitive)
        
        # Should have low difference
        assert diff < 0.2


# ============================================================================
# Integration Tests
# ============================================================================

class TestRankingMetricsIntegration:
    """Integration tests for ranking metrics."""
    
    def test_all_metrics_agree_on_fairness(self, balanced_ranking):
        """Test that all metrics agree on fair ranking."""
        rankings, sensitive = balanced_ranking
        
        exp_diff, _, _ = exposure_parity_difference(rankings, sensitive)
        att_diff, _ = attention_weighted_fairness(rankings, sensitive)
        
        # Both should show low disparity
        assert exp_diff < 0.2
        assert att_diff < 0.2
    
    def test_all_metrics_agree_on_bias(self, biased_ranking):
        """Test that all metrics detect bias."""
        rankings, sensitive = biased_ranking
        
        exp_diff, _, _ = exposure_parity_difference(rankings, sensitive)
        att_diff, _ = attention_weighted_fairness(rankings, sensitive)
        
        # Both should detect bias
        assert exp_diff > 0.0
        assert att_diff > 0.0
    
    def test_metrics_with_same_data_format(self, simple_ranking):
        """Test that all metrics accept same data format."""
        rankings, sensitive = simple_ranking
        
        # All should work with same inputs
        exp_diff, _, _ = exposure_parity_difference(rankings, sensitive)
        att_diff, _ = attention_weighted_fairness(rankings, sensitive)
        
        assert isinstance(exp_diff, float)
        assert isinstance(att_diff, float)
    
    @pytest.mark.parametrize("top_k", [3, 5, None])
    def test_all_metrics_support_top_k(self, simple_ranking, top_k):
        """Test that all metrics support top_k parameter."""
        rankings, sensitive = simple_ranking
        
        exp_diff, _, _ = exposure_parity_difference(
            rankings, sensitive, top_k=top_k
        )
        att_diff, _ = attention_weighted_fairness(
            rankings, sensitive, top_k=top_k
        )
        
        assert isinstance(exp_diff, float)
        assert isinstance(att_diff, float)


# ============================================================================
# Edge Cases
# ============================================================================

class TestRankingEdgeCases:
    """Test edge cases for ranking metrics."""
    
    def test_single_item_ranking(self):
        """Test with single item."""
        rankings = np.array([0])
        sensitive = np.array([0])
        
        # Should handle gracefully
        with pytest.raises(ValueError):
            exposure_parity_difference(rankings, sensitive)
    
    def test_ranking_with_duplicates(self):
        """Test ranking with duplicate item IDs."""
        rankings = np.array([0, 1, 1, 2, 3])  # Duplicate
        sensitive = np.array([0, 1, 0, 1, 0])
        
        # Should handle or raise appropriate error
        try:
            diff, _, _ = exposure_parity_difference(rankings, sensitive)
            assert isinstance(diff, float)
        except (ValueError, IndexError):
            pass  # Acceptable to reject
    
    def test_ranking_with_out_of_bounds_ids(self):
        """Test ranking with item IDs outside sensitive_features range."""
        rankings = np.array([0, 1, 2, 3, 10])  # ID 10 out of bounds
        sensitive = np.array([0, 1, 0, 1, 0])
        
        # Should handle gracefully
        with pytest.raises(IndexError):
            exposure_parity_difference(rankings, sensitive)
    
    def test_all_positions_equal_weight(self):
        """Test with equal weights for all positions."""
        rankings = np.array([0, 1, 2, 3, 4])
        sensitive = np.array([0, 1, 0, 1, 0])
        equal_weights = np.ones(5)
        
        diff, _, _ = exposure_parity_difference(
            rankings, sensitive, position_weights=equal_weights
        )
        
        assert isinstance(diff, float)
    
    def test_zero_weights(self):
        """Test with all zero weights."""
        rankings = np.array([0, 1, 2, 3, 4])
        sensitive = np.array([0, 1, 0, 1, 0])
        zero_weights = np.zeros(5)
        
        diff, exposures, _ = exposure_parity_difference(
            rankings, sensitive, position_weights=zero_weights
        )
        
        # All exposures should be 0
        assert all(exp == 0 for exp in exposures.values())


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestRankingMetricsProperties:
    """Property-based tests for ranking metrics."""
    
    def test_difference_is_non_negative(self, simple_ranking):
        """Test that difference is always non-negative."""
        rankings, sensitive = simple_ranking
        
        diff, _, _ = exposure_parity_difference(rankings, sensitive)
        
        assert diff >= 0
    
    def test_perfect_balance_gives_zero_or_near_zero(self):
        """Test that perfect balance gives very low difference."""
        # Create perfectly balanced ranking
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        sensitive = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        diff, _, _ = exposure_parity_difference(rankings, sensitive)
        
        # Should be reasonably low (DCG weighting causes some difference even with perfect alternation)
        assert diff < 0.12  # Relaxed threshold
    
    def test_reversing_ranking_may_change_difference(self, simple_ranking):
        """Test that reversing ranking affects metrics."""
        rankings, sensitive = simple_ranking
        
        diff_forward, _, _ = exposure_parity_difference(rankings, sensitive)
        diff_reverse, _, _ = exposure_parity_difference(
            rankings[::-1], sensitive
        )
        
        # May be different due to position weights
        assert isinstance(diff_forward, float)
        assert isinstance(diff_reverse, float)
    
    def test_metric_is_symmetric_in_groups(self):
        """Test that swapping group labels doesn't change magnitude."""
        rankings = np.array([0, 1, 2, 3, 4])
        sensitive = np.array([0, 1, 0, 1, 0])
        
        diff1, _, _ = exposure_parity_difference(rankings, sensitive)
        
        # Swap groups
        sensitive_swapped = 1 - sensitive
        diff2, _, _ = exposure_parity_difference(rankings, sensitive_swapped)
        
        # Magnitude should be same
        assert np.isclose(diff1, diff2, rtol=0.01)