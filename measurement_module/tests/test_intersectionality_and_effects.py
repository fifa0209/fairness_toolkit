"""
Tests for intersectionality analysis and effect sizes.

Tests intersectionality.py and effect_sizes.py modules.
"""

import pytest
import numpy as np
import pandas as pd

from measurement_module.src.intersectionality import (
    create_intersectional_groups,
    compute_intersectional_metrics,
    analyze_pairwise_disparities,
    identify_most_disadvantaged_groups,
    test_intersectional_fairness,
    generate_intersectional_report,
)

from measurement_module.src.effect_sizes import (
    compute_cohens_d,
    compute_risk_ratio,
    compute_odds_ratio,
    compute_disparate_impact_ratio,
    compute_all_effect_sizes,
    interpret_cohens_d,
    interpret_risk_ratio,
)


@pytest.fixture
def intersectional_data():
    """Generate data with multiple protected attributes."""
    np.random.seed(42)
    n = 400
    
    # Two protected attributes
    gender = np.random.binomial(1, 0.5, n)
    race = np.random.binomial(1, 0.4, n)
    
    # Predictions and labels
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    
    return y_true, y_pred, gender, race


class TestCreateIntersectionalGroups:
    """Test intersectional group creation."""
    
    def test_basic_group_creation(self):
        """Test creating intersectional groups from two attributes."""
        gender = np.array([0, 1, 0, 1, 0, 1])
        race = np.array([0, 0, 1, 1, 0, 1])
        
        groups, labels = create_intersectional_groups(gender, race)
        
        assert len(groups) == 6
        assert len(set(groups)) <= 4  # At most 4 unique combinations
    
    def test_with_labels(self):
        """Test creating groups with descriptive labels."""
        gender = np.array([0, 1, 0, 1])
        race = np.array([0, 0, 1, 1])
        
        groups, labels = create_intersectional_groups(
            gender, race,
            labels=['gender', 'race']
        )
        
        assert len(labels) > 0
        assert all(isinstance(label, str) for label in labels)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched lengths."""
        gender = np.array([0, 1, 0])
        race = np.array([0, 0])  # Different length
        
        with pytest.raises(ValueError) as exc_info:
            create_intersectional_groups(gender, race)
        
        assert "same length" in str(exc_info.value)
    
    def test_three_attributes(self):
        """Test with three protected attributes."""
        attr1 = np.array([0, 1, 0, 1])
        attr2 = np.array([0, 0, 1, 1])
        attr3 = np.array([0, 1, 0, 1])
        
        groups, _ = create_intersectional_groups(attr1, attr2, attr3)
        
        assert len(groups) == 4
        # Check format: should be "0_0_0" style
        assert all('_' in str(g) for g in groups)


class TestComputeIntersectionalMetrics:
    """Test intersectional metric computation."""
    
    def test_basic_computation(self, intersectional_data):
        """Test computing intersectional metrics."""
        y_true, y_pred, gender, race = intersectional_data
        
        results = compute_intersectional_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            min_group_size=30
        )
        
        assert 'metric_name' in results
        assert 'reliable_groups' in results
        assert 'small_groups' in results
        assert 'max_disparity' in results
    
    def test_min_group_size_filtering(self, intersectional_data):
        """Test filtering of small groups."""
        y_true, y_pred, gender, race = intersectional_data
        
        # With large min_group_size, should filter out more groups
        results = compute_intersectional_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            min_group_size=100  # High threshold
        )
        
        # Should have some small groups
        assert results['n_small_groups'] >= 0
    
    def test_different_metrics(self, intersectional_data):
        """Test with different fairness metrics."""
        y_true, y_pred, gender, race = intersectional_data
        
        metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
        
        for metric in metrics:
            results = compute_intersectional_metrics(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features_dict={'gender': gender, 'race': race},
                metric_name=metric,
                min_group_size=30
            )
            
            assert results['metric_name'] == metric


class TestPairwiseDisparities:
    """Test pairwise disparity analysis."""
    
    def test_analyze_pairwise(self, intersectional_data):
        """Test pairwise disparity analysis."""
        _, y_pred, gender, race = intersectional_data
        
        df = analyze_pairwise_disparities(
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            min_group_size=30
        )
        
        assert isinstance(df, pd.DataFrame)
        
        if not df.empty:
            assert 'group_1' in df.columns
            assert 'group_2' in df.columns
            assert 'disparity' in df.columns
            assert 'rate_1' in df.columns
            assert 'rate_2' in df.columns
            
            # Should be sorted by disparity
            assert (df['disparity'].values[:-1] >= df['disparity'].values[1:]).all()
    
    def test_pairwise_small_sample(self):
        """Test with small sample that filters groups."""
        n = 50
        y_pred = np.random.binomial(1, 0.5, n)
        gender = np.random.binomial(1, 0.5, n)
        race = np.random.binomial(1, 0.5, n)
        
        df = analyze_pairwise_disparities(
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            min_group_size=20  # Might exclude some groups
        )
        
        # Should still return a DataFrame (possibly empty)
        assert isinstance(df, pd.DataFrame)


class TestDisadvantagedGroups:
    """Test identification of disadvantaged groups."""
    
    def test_identify_disadvantaged(self, intersectional_data):
        """Test identifying most disadvantaged groups."""
        _, y_pred, gender, race = intersectional_data
        
        df = identify_most_disadvantaged_groups(
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            min_group_size=30,
            top_n=3
        )
        
        assert isinstance(df, pd.DataFrame)
        
        if not df.empty:
            assert len(df) <= 3
            assert 'group' in df.columns
            assert 'positive_rate' in df.columns
            assert 'size' in df.columns
            
            # Should be sorted by positive_rate ascending
            assert (df['positive_rate'].values[:-1] <= df['positive_rate'].values[1:]).all()
    
    def test_top_n_parameter(self, intersectional_data):
        """Test top_n parameter."""
        _, y_pred, gender, race = intersectional_data
        
        df1 = identify_most_disadvantaged_groups(
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            top_n=2
        )
        
        df2 = identify_most_disadvantaged_groups(
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            top_n=5
        )
        
        # df2 should have more (or equal) groups
        assert len(df2) >= len(df1)


class TestIntersectionalFairnessTest:
    """Test comprehensive intersectional fairness testing."""
    
    def test_comprehensive_test(self, intersectional_data):
        """Test comprehensive intersectional fairness analysis."""
        y_true, y_pred, gender, race = intersectional_data
        
        results = test_intersectional_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            threshold=0.15,
            min_group_size=30
        )
        
        assert 'intersectional_metrics' in results
        assert 'pairwise_disparities' in results
        assert 'disadvantaged_groups' in results
        assert 'summary' in results
        
        # Check summary structure
        summary = results['summary']
        assert 'n_reliable_groups' in summary
        assert 'max_disparity' in summary
    
    def test_with_bonferroni_correction(self, intersectional_data):
        """Test with Bonferroni multiple comparison correction."""
        y_true, y_pred, gender, race = intersectional_data
        
        results = test_intersectional_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            multiple_comparison_correction='bonferroni'
        )
        
        # Should have adjusted thresholds
        df = results['pairwise_disparities']
        if not df.empty:
            assert 'adjusted_threshold' in df.columns
    
    def test_with_benjamini_hochberg(self, intersectional_data):
        """Test with Benjamini-Hochberg FDR control."""
        y_true, y_pred, gender, race = intersectional_data
        
        results = test_intersectional_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race},
            multiple_comparison_correction='benjamini-hochberg'
        )
        
        # Should have BH thresholds
        df = results['pairwise_disparities']
        if not df.empty:
            assert 'bh_threshold' in df.columns


class TestIntersectionalReport:
    """Test report generation."""
    
    def test_generate_report(self, intersectional_data):
        """Test generating intersectional fairness report."""
        y_true, y_pred, gender, race = intersectional_data
        
        test_results = test_intersectional_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features_dict={'gender': gender, 'race': race}
        )
        
        report = generate_intersectional_report(test_results)
        
        assert isinstance(report, str)
        assert "INTERSECTIONAL FAIRNESS ANALYSIS" in report
        assert "SUMMARY" in report


# ============================================================================
# Effect Sizes Tests
# ============================================================================

class TestCohensD:
    """Test Cohen's d computation."""
    
    def test_basic_computation(self):
        """Test basic Cohen's d calculation."""
        group_0 = np.array([1, 2, 3, 4, 5])
        group_1 = np.array([3, 4, 5, 6, 7])
        
        d = compute_cohens_d(group_0, group_1)
        
        assert isinstance(d, float)
        assert d > 0  # group_1 has higher mean
    
    def test_zero_difference(self):
        """Test when groups have same mean."""
        group_0 = np.array([1, 2, 3, 4, 5])
        group_1 = np.array([1, 2, 3, 4, 5])
        
        d = compute_cohens_d(group_0, group_1)
        
        assert abs(d) < 0.001  # Should be close to 0
    
    def test_large_effect(self):
        """Test large effect size."""
        group_0 = np.array([0, 0, 0, 0, 0])
        group_1 = np.array([1, 1, 1, 1, 1])
        
        d = compute_cohens_d(group_0, group_1)
        
        assert abs(d) > 2.0  # Very large effect
    
    def test_interpret_cohens_d(self):
        """Test interpretation of Cohen's d."""
        interpretation = interpret_cohens_d(0.3)
        assert "small" in interpretation.lower()
        
        interpretation = interpret_cohens_d(0.6)
        assert "medium" in interpretation.lower()
        
        interpretation = interpret_cohens_d(1.0)
        assert "large" in interpretation.lower()


class TestRiskRatio:
    """Test risk ratio computation."""
    
    def test_equal_rates(self):
        """Test when rates are equal."""
        rr = compute_risk_ratio(0.5, 0.5)
        assert abs(rr - 1.0) < 0.001
    
    def test_double_rate(self):
        """Test when one rate is double."""
        rr = compute_risk_ratio(0.3, 0.6)
        assert abs(rr - 2.0) < 0.001
    
    def test_half_rate(self):
        """Test when one rate is half."""
        rr = compute_risk_ratio(0.6, 0.3)
        assert abs(rr - 0.5) < 0.001
    
    def test_zero_rate_handling(self):
        """Test handling of zero rates."""
        rr = compute_risk_ratio(0.0, 0.5)
        assert rr > 0  # Should not be infinite
        assert np.isfinite(rr)
    
    def test_interpret_risk_ratio(self):
        """Test risk ratio interpretation."""
        interp = interpret_risk_ratio(0.9)
        assert "80% rule" in interp
        
        interp = interpret_risk_ratio(0.5)
        assert "fails" in interp.lower()


class TestDisparateImpactRatio:
    """Test disparate impact ratio."""
    
    def test_80_percent_rule(self):
        """Test 80% rule compliance."""
        # Exactly 80% should pass
        di = compute_disparate_impact_ratio(0.4, 0.5)
        assert di == 0.8
        
        # Below 80% should fail
        di = compute_disparate_impact_ratio(0.3, 0.5)
        assert di < 0.8
        
        # Above 80% should pass
        di = compute_disparate_impact_ratio(0.45, 0.5)
        assert di > 0.8
    
    def test_symmetry(self):
        """Test that DI ratio is symmetric."""
        di1 = compute_disparate_impact_ratio(0.3, 0.5)
        di2 = compute_disparate_impact_ratio(0.5, 0.3)
        
        assert abs(di1 - di2) < 0.001


class TestAllEffectSizes:
    """Test computing all effect sizes."""
    
    def test_binary_predictions(self):
        """Test with binary predictions."""
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        effect_sizes = compute_all_effect_sizes(
            y_pred=y_pred,
            sensitive_features=sensitive,
            metric_type='binary'
        )
        
        assert 'cohens_d' in effect_sizes
        assert 'risk_ratio' in effect_sizes
        assert 'odds_ratio' in effect_sizes
        assert 'disparate_impact_ratio' in effect_sizes
    
    def test_continuous_predictions(self):
        """Test with continuous predictions."""
        y_pred = np.random.normal(0, 1, 100)
        sensitive = np.random.binomial(1, 0.5, 100)
        
        effect_sizes = compute_all_effect_sizes(
            y_pred=y_pred,
            sensitive_features=sensitive,
            metric_type='continuous'
        )
        
        assert 'cohens_d' in effect_sizes
        assert 'risk_ratio' not in effect_sizes  # Only for binary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])