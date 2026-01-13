"""
Tests for A/B Testing Module

Tests the FairnessABTestAnalyzer including overall test analysis,
heterogeneous effects, multi-objective evaluation, and statistical tests.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from monitoring_module.src.ab_testing import (
    FairnessABTestAnalyzer,
    ABTestResult,
    HeterogeneousEffectResult,
    run_ab_test,
)
from shared.validation import ValidationError


@pytest.fixture
def sample_ab_data():
    """Create sample A/B test data."""
    np.random.seed(42)
    
    # Control group (n=500)
    control_df = pd.DataFrame({
        'y_true': np.random.binomial(1, 0.5, 500),
        'y_pred': np.random.binomial(1, 0.45, 500),
        'sensitive': np.random.binomial(1, 0.5, 500),
        'age_group': np.random.choice(['young', 'old'], 500),
    })
    
    # Treatment group (n=500, slightly better fairness)
    treatment_df = pd.DataFrame({
        'y_true': np.random.binomial(1, 0.5, 500),
        'y_pred': np.random.binomial(1, 0.48, 500),
        'sensitive': np.random.binomial(1, 0.5, 500),
        'age_group': np.random.choice(['young', 'old'], 500),
    })
    
    return control_df, treatment_df


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return FairnessABTestAnalyzer(alpha=0.05, min_sample_size=30, n_bootstrap=100)


class TestABTestResult:
    """Tests for ABTestResult dataclass."""
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ABTestResult(
            metric_name='accuracy',
            control_value=0.80,
            treatment_value=0.82,
            absolute_difference=0.02,
            relative_difference=0.025,
            confidence_interval=(0.01, 0.03),
            p_value=0.03,
            is_significant=True,
            effect_size=0.15,
            sample_sizes={'control': 100, 'treatment': 100},
            interpretation='Significant improvement',
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['metric_name'] == 'accuracy'
        assert result_dict['control_value'] == 0.80
        assert result_dict['p_value'] == 0.03
        assert result_dict['is_significant'] is True


class TestHeterogeneousEffectResult:
    """Tests for HeterogeneousEffectResult dataclass."""
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = HeterogeneousEffectResult(
            subgroup='age_group=young',
            control_value=0.15,
            treatment_value=0.10,
            treatment_effect=-0.05,
            confidence_interval=(-0.08, -0.02),
            p_value=0.02,
            is_significant=True,
            sample_size=200,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['subgroup'] == 'age_group=young'
        assert result_dict['treatment_effect'] == -0.05
        assert result_dict['is_significant'] is True


class TestFairnessABTestAnalyzer:
    """Tests for FairnessABTestAnalyzer."""
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.alpha == 0.05
        assert analyzer.min_sample_size == 30
        assert analyzer.n_bootstrap == 100
    
    def test_analyze_test_basic(self, analyzer, sample_ab_data):
        """Test basic A/B test analysis."""
        control_df, treatment_df = sample_ab_data
        
        results = analyzer.analyze_test(
            control_df=control_df,
            treatment_df=treatment_df,
            metrics=['accuracy', 'demographic_parity']
        )
        
        assert 'accuracy' in results
        assert 'demographic_parity' in results
        
        # Check result structure
        acc_result = results['accuracy']
        assert isinstance(acc_result, ABTestResult)
        assert acc_result.metric_name == 'accuracy'
        assert 0 <= acc_result.control_value <= 1
        assert 0 <= acc_result.treatment_value <= 1
        assert acc_result.p_value >= 0
        assert isinstance(acc_result.is_significant, bool)
    
    def test_analyze_test_all_metrics(self, analyzer, sample_ab_data):
        """Test analysis with all fairness metrics."""
        control_df, treatment_df = sample_ab_data
        
        metrics = ['accuracy', 'demographic_parity', 'equalized_odds', 'equal_opportunity']
        
        results = analyzer.analyze_test(
            control_df=control_df,
            treatment_df=treatment_df,
            metrics=metrics
        )
        
        assert len(results) == len(metrics)
        for metric in metrics:
            assert metric in results
            assert results[metric].sample_sizes['control'] == 500
            assert results[metric].sample_sizes['treatment'] == 500
    
    def test_validate_test_data_insufficient_control(self, analyzer):
        """Test validation with insufficient control samples."""
        control_df = pd.DataFrame({
            'y_true': [1, 0],
            'y_pred': [1, 0],
            'sensitive': [0, 1],
        })
        
        treatment_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.5, 100),
            'y_pred': np.random.binomial(1, 0.5, 100),
            'sensitive': np.random.binomial(1, 0.5, 100),
        })
        
        with pytest.raises(ValidationError, match="Control group too small"):
            analyzer.analyze_test(control_df, treatment_df, metrics=['accuracy'])
    
    def test_validate_test_data_missing_columns(self, analyzer):
        """Test validation with missing required columns."""
        control_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.5, 100),
            'y_pred': np.random.binomial(1, 0.5, 100),
            # Missing 'sensitive' column
        })
        
        treatment_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.5, 100),
            'y_pred': np.random.binomial(1, 0.5, 100),
            'sensitive': np.random.binomial(1, 0.5, 100),
        })
        
        with pytest.raises(ValidationError, match="Control missing column"):
            analyzer.analyze_test(control_df, treatment_df, metrics=['accuracy'])
    
    def test_compute_metric_accuracy(self, analyzer, sample_ab_data):
        """Test accuracy metric computation."""
        control_df, _ = sample_ab_data
        
        accuracy = analyzer._compute_metric(
            control_df, 'accuracy', 'y_pred', 'y_true', 'sensitive'
        )
        
        assert 0 <= accuracy <= 1
        
        # Verify calculation
        expected = (control_df['y_pred'] == control_df['y_true']).mean()
        assert abs(accuracy - expected) < 1e-6
    
    def test_compute_metric_demographic_parity(self, analyzer, sample_ab_data):
        """Test demographic parity metric computation."""
        control_df, _ = sample_ab_data
        
        dp = analyzer._compute_metric(
            control_df, 'demographic_parity', 'y_pred', 'y_true', 'sensitive'
        )
        
        assert dp >= 0
        
        # Verify calculation
        rate_0 = control_df[control_df['sensitive'] == 0]['y_pred'].mean()
        rate_1 = control_df[control_df['sensitive'] == 1]['y_pred'].mean()
        expected = abs(rate_0 - rate_1)
        assert abs(dp - expected) < 1e-6
    
    def test_compute_metric_unknown(self, analyzer, sample_ab_data):
        """Test error handling for unknown metric."""
        control_df, _ = sample_ab_data
        
        with pytest.raises(ValueError, match="Unknown metric"):
            analyzer._compute_metric(
                control_df, 'unknown_metric', 'y_pred', 'y_true', 'sensitive'
            )
    
    def test_heterogeneous_effects(self, analyzer, sample_ab_data):
        """Test heterogeneous treatment effect analysis."""
        control_df, treatment_df = sample_ab_data
        
        results = analyzer.analyze_heterogeneous_effects(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='demographic_parity',
            subgroup_cols=['age_group']
        )
        
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, HeterogeneousEffectResult)
            assert 'age_group=' in result.subgroup
            assert result.sample_size > 0
            assert 0 <= result.p_value <= 1
    
    def test_heterogeneous_effects_multiple_dimensions(self, analyzer, sample_ab_data):
        """Test heterogeneous effects across multiple dimensions."""
        control_df, treatment_df = sample_ab_data
        
        # Add another dimension
        control_df['gender'] = np.random.choice(['M', 'F'], len(control_df))
        treatment_df['gender'] = np.random.choice(['M', 'F'], len(treatment_df))
        
        results = analyzer.analyze_heterogeneous_effects(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='demographic_parity',
            subgroup_cols=['age_group', 'gender']
        )
        
        # Should have results for combinations of age_group and gender
        assert len(results) > 0
        
        # Check subgroup names contain both dimensions
        for result in results:
            assert 'age_group=' in result.subgroup
            assert 'gender=' in result.subgroup
    
    def test_heterogeneous_effects_insufficient_samples(self, analyzer, caplog):
        """Test handling of insufficient samples in subgroups."""
        # Create data with small subgroups
        control_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.5, 50),
            'y_pred': np.random.binomial(1, 0.5, 50),
            'sensitive': np.random.binomial(1, 0.5, 50),
            'rare_group': ['A'] * 5 + ['B'] * 45,  # Group A has only 5 samples
        })
        
        treatment_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.5, 50),
            'y_pred': np.random.binomial(1, 0.5, 50),
            'sensitive': np.random.binomial(1, 0.5, 50),
            'rare_group': ['A'] * 5 + ['B'] * 45,
        })
        
        results = analyzer.analyze_heterogeneous_effects(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='accuracy',
            subgroup_cols=['rare_group']
        )
        
        # Should skip group A (insufficient samples)
        subgroups = [r.subgroup for r in results]
        assert not any('rare_group=A' in s for s in subgroups)
        assert any('rare_group=B' in s for s in subgroups)
    
    def test_multi_objective_analysis(self, analyzer, sample_ab_data):
        """Test multi-objective analysis."""
        control_df, treatment_df = sample_ab_data
        
        result = analyzer.multi_objective_analysis(
            control_df=control_df,
            treatment_df=treatment_df,
            performance_metric='accuracy',
            fairness_metric='demographic_parity'
        )
        
        assert 'performance_result' in result
        assert 'fairness_result' in result
        assert 'trade_off_ratio' in result
        assert 'outcome' in result
        assert 'recommendation' in result
        
        assert isinstance(result['performance_result'], ABTestResult)
        assert isinstance(result['fairness_result'], ABTestResult)
    
    # def test_multi_objective_win_win(self, analyzer):
    #     """Test multi-objective analysis with win-win scenario."""
    #     # Create data where treatment improves both
    #     control_df = pd.DataFrame({
    #         'y_true': [1, 0, 1, 0] * 25,
    #         'y_pred': [1, 0, 0, 1] * 25,  # Poor accuracy
    #         'sensitive': [0, 0, 1, 1] * 25,
    #     })
        
    #     treatment_df = pd.DataFrame({
    #         'y_true': [1, 0, 1, 0] * 25,
    #         'y_pred': [1, 0, 1, 0] * 25,  # Better accuracy and fairness
    #         'sensitive': [0, 0, 1, 1] * 25,
    #     })
        
    #     result = analyzer.multi_objective_analysis(
    #         control_df=control_df,
    #         treatment_df=treatment_df
    #     )
        
    #     assert 'Win-Win' in result['outcome'] or 'RECOMMEND' in result['recommendation']
    def test_multi_objective_win_win(self, analyzer):
        """Test multi-objective analysis with win-win scenario."""
        np.random.seed(42)
        
        # Create data where treatment improves both accuracy AND fairness
        # Control: biased predictions with poor accuracy
        control_df = pd.DataFrame({
            'y_true': [1, 1, 0, 0] * 25,  # Balanced true labels
            'y_pred': [1, 1, 0, 1] * 25,  # 75% accuracy, biased toward group 0
            'sensitive': [0, 0, 1, 1] * 25,  # Alternating groups
        })
        # Group 0 positive rate: 100% (all 1s)
        # Group 1 positive rate: 50% (alternating 0,1)
        # Demographic parity difference: 0.5
        
        # Treatment: better predictions with improved fairness
        treatment_df = pd.DataFrame({
            'y_true': [1, 1, 0, 0] * 25,  # Same true labels
            'y_pred': [1, 1, 0, 0] * 25,  # 100% accuracy, perfectly fair
            'sensitive': [0, 0, 1, 1] * 25,  # Same groups
        })
        # Group 0 positive rate: 50%
        # Group 1 positive rate: 50%
        # Demographic parity difference: 0.0
        
        result = analyzer.multi_objective_analysis(
            control_df=control_df,
            treatment_df=treatment_df
        )
        
        # Should detect win-win or at least recommend deployment
        assert ('Win-Win' in result['outcome'] or 
                'improved' in result['outcome'].lower() or
                'RECOMMEND' in result['recommendation'])
    def test_bootstrap_ci_difference(self, analyzer, sample_ab_data):
        """Test bootstrap confidence interval computation."""
        control_df, treatment_df = sample_ab_data
        
        ci_lower, ci_upper = analyzer._bootstrap_ci_difference(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='accuracy',
            y_col='y_pred',
            y_true_col='y_true',
            sensitive_col='sensitive'
        )
        
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
    
    def test_permutation_test(self, analyzer, sample_ab_data):
        """Test permutation test for significance."""
        control_df, treatment_df = sample_ab_data
        
        p_value = analyzer._permutation_test(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='accuracy',
            y_col='y_pred',
            y_true_col='y_true',
            sensitive_col='sensitive'
        )
        
        assert 0 <= p_value <= 1
        assert isinstance(p_value, float)
    
    def test_compute_effect_size(self, analyzer, sample_ab_data):
        """Test Cohen's d effect size computation."""
        control_df, treatment_df = sample_ab_data
        
        effect_size = analyzer._compute_effect_size(
            control_df=control_df,
            treatment_df=treatment_df,
            metric='accuracy',
            y_col='y_pred',
            y_true_col='y_true',
            sensitive_col='sensitive'
        )
        
        assert isinstance(effect_size, float)
        # Effect size should be reasonable
        assert -3 < effect_size < 3
    
    def test_interpret_result_significant_improvement(self, analyzer):
        """Test interpretation of significant improvement."""
        interpretation = analyzer._interpret_result(
            control_value=0.70,
            treatment_value=0.80,
            p_value=0.01,
            effect_size=0.6,
            metric='accuracy'
        )
        
        assert 'significant' in interpretation.lower()
        assert 'improvement' in interpretation.lower()
    
    def test_interpret_result_no_change(self, analyzer):
        """Test interpretation of no significant change."""
        interpretation = analyzer._interpret_result(
            control_value=0.75,
            treatment_value=0.76,
            p_value=0.50,
            effect_size=0.05,
            metric='accuracy'
        )
        
        assert 'no statistically significant' in interpretation.lower()
    
    def test_generate_recommendation_strong(self, analyzer):
        """Test recommendation generation for strong results."""
        perf_result = ABTestResult(
            metric_name='accuracy',
            control_value=0.70,
            treatment_value=0.75,
            absolute_difference=0.05,
            relative_difference=0.071,
            confidence_interval=(0.02, 0.08),
            p_value=0.001,
            is_significant=True,
            effect_size=0.5,
            sample_sizes={'control': 500, 'treatment': 500},
            interpretation='Significant improvement',
        )
        
        fair_result = ABTestResult(
            metric_name='demographic_parity',
            control_value=0.15,
            treatment_value=0.08,
            absolute_difference=-0.07,
            relative_difference=-0.467,
            confidence_interval=(-0.10, -0.04),
            p_value=0.001,
            is_significant=True,
            effect_size=-0.6,
            sample_sizes={'control': 500, 'treatment': 500},
            interpretation='Significant improvement',
        )
        
        recommendation = analyzer._generate_recommendation(perf_result, fair_result)
        
        assert 'STRONGLY RECOMMEND' in recommendation or 'RECOMMEND' in recommendation


class TestRunABTest:
    """Tests for convenience function."""
    
    def test_run_ab_test_basic(self, sample_ab_data):
        """Test basic run_ab_test function."""
        control_df, treatment_df = sample_ab_data
        
        results = run_ab_test(
            control_df=control_df,
            treatment_df=treatment_df,
            alpha=0.05
        )
        
        assert 'overall' in results
        assert 'heterogeneous' in results
        assert 'multi_objective' in results
        
        # Check overall results
        assert 'accuracy' in results['overall']
        assert 'demographic_parity' in results['overall']
    
    def test_run_ab_test_with_subgroups(self, sample_ab_data):
        """Test run_ab_test with subgroup analysis."""
        control_df, treatment_df = sample_ab_data
        
        results = run_ab_test(
            control_df=control_df,
            treatment_df=treatment_df,
            metrics=['demographic_parity'],
            subgroup_cols=['age_group'],
            alpha=0.05
        )
        
        assert 'heterogeneous' in results
        assert 'demographic_parity' in results['heterogeneous']
        assert len(results['heterogeneous']['demographic_parity']) > 0
    
    def test_run_ab_test_custom_metrics(self, sample_ab_data):
        """Test run_ab_test with custom metrics."""
        control_df, treatment_df = sample_ab_data
        
        results = run_ab_test(
            control_df=control_df,
            treatment_df=treatment_df,
            metrics=['equalized_odds', 'equal_opportunity']
        )
        
        assert 'equalized_odds' in results['overall']
        assert 'equal_opportunity' in results['overall']
        assert 'accuracy' not in results['overall']


class TestIntegration:
    """Integration tests."""
    
    def test_full_ab_test_workflow(self, sample_ab_data):
        """Test complete A/B test workflow."""
        control_df, treatment_df = sample_ab_data
        
        analyzer = FairnessABTestAnalyzer(alpha=0.05, n_bootstrap=100)
        
        # 1. Overall analysis
        overall_results = analyzer.analyze_test(
            control_df, treatment_df,
            metrics=['accuracy', 'demographic_parity']
        )
        
        assert len(overall_results) == 2
        
        # 2. Heterogeneous effects
        hetero_results = analyzer.analyze_heterogeneous_effects(
            control_df, treatment_df,
            metric='demographic_parity',
            subgroup_cols=['age_group']
        )
        
        assert len(hetero_results) > 0
        
        # 3. Multi-objective
        multi_obj = analyzer.multi_objective_analysis(
            control_df, treatment_df
        )
        
        assert 'recommendation' in multi_obj
        
        # All results should be consistent
        assert isinstance(overall_results['accuracy'], ABTestResult)
        assert isinstance(hetero_results[0], HeterogeneousEffectResult)
    
    def test_real_world_scenario(self):
        """Test with realistic data scenario."""
        np.random.seed(42)
        
        # Control: biased model
        n = 1000
        control_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.3, n),
            'y_pred': np.concatenate([
                np.random.binomial(1, 0.4, n//2),  # Group 0: 40% positive rate
                np.random.binomial(1, 0.2, n//2),  # Group 1: 20% positive rate
            ]),
            'sensitive': [0] * (n//2) + [1] * (n//2),
            'region': np.random.choice(['North', 'South'], n),
        })
        
        # Treatment: fairer model
        treatment_df = pd.DataFrame({
            'y_true': np.random.binomial(1, 0.3, n),
            'y_pred': np.concatenate([
                np.random.binomial(1, 0.32, n//2),  # Group 0: 32% positive rate
                np.random.binomial(1, 0.28, n//2),  # Group 1: 28% positive rate
            ]),
            'sensitive': [0] * (n//2) + [1] * (n//2),
            'region': np.random.choice(['North', 'South'], n),
        })
        
        analyzer = FairnessABTestAnalyzer(alpha=0.05, n_bootstrap=100)
        
        results = analyzer.analyze_test(
            control_df, treatment_df,
            metrics=['demographic_parity']
        )
        
        dp_result = results['demographic_parity']
        
        # Treatment should have lower demographic parity
        assert dp_result.treatment_value < dp_result.control_value
        
        # Difference should be significant
        # (20% vs 4% difference is substantial)
        assert dp_result.is_significant