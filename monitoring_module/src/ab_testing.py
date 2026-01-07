"""
A/B Testing for Fairness Interventions

Provides rigorous statistical analysis of fairness interventions through
A/B testing, including power analysis, heterogeneous treatment effects,
and multi-objective evaluation.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from itertools import product

from shared.logging import get_logger
from shared.validation import validate_predictions, ValidationError

logger = get_logger(__name__)


@dataclass
class ABTestResult:
    """Results from an A/B test analysis."""
    
    metric_name: str
    control_value: float
    treatment_value: float
    absolute_difference: float
    relative_difference: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    effect_size: float
    sample_sizes: Dict[str, int]
    interpretation: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'control_value': self.control_value,
            'treatment_value': self.treatment_value,
            'absolute_difference': self.absolute_difference,
            'relative_difference': self.relative_difference,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'effect_size': self.effect_size,
            'sample_sizes': self.sample_sizes,
            'interpretation': self.interpretation,
        }


@dataclass
class HeterogeneousEffectResult:
    """Results from heterogeneous treatment effect analysis."""
    
    subgroup: str
    control_value: float
    treatment_value: float
    treatment_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    sample_size: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'subgroup': self.subgroup,
            'control_value': self.control_value,
            'treatment_value': self.treatment_value,
            'treatment_effect': self.treatment_effect,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'sample_size': self.sample_size,
        }


class FairnessABTestAnalyzer:
    """
    Analyze A/B tests for fairness interventions.
    
    Provides comprehensive statistical analysis including:
    - Overall treatment effects
    - Heterogeneous effects across subgroups
    - Multi-objective evaluation (fairness + performance)
    - Statistical power analysis
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        min_sample_size: int = 30,
        n_bootstrap: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level
            min_sample_size: Minimum samples per group
            n_bootstrap: Bootstrap samples for CI
            random_state: Random seed
        """
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        logger.info(
            f"Initialized FairnessABTestAnalyzer "
            f"(alpha={alpha}, min_n={min_sample_size})"
        )
    
    def analyze_test(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metrics: List[str],
        y_col: str = 'y_pred',
        y_true_col: str = 'y_true',
        sensitive_col: str = 'sensitive',
    ) -> Dict[str, ABTestResult]:
        """
        Analyze A/B test for multiple metrics.
        
        Args:
            control_df: Control group data
            treatment_df: Treatment group data
            metrics: List of metrics to evaluate
            y_col: Prediction column name
            y_true_col: True label column name
            sensitive_col: Sensitive attribute column name
        
        Returns:
            Dictionary mapping metric names to ABTestResult objects
        """
        logger.info(f"Analyzing A/B test for {len(metrics)} metrics")
        
        # Validate inputs
        self._validate_test_data(control_df, treatment_df)
        
        results = {}
        
        for metric in metrics:
            logger.info(f"Computing metric: {metric}")
            
            # Compute metric values
            control_value = self._compute_metric(
                control_df, metric, y_col, y_true_col, sensitive_col
            )
            treatment_value = self._compute_metric(
                treatment_df, metric, y_col, y_true_col, sensitive_col
            )
            
            # Statistical test
            test_result = self._statistical_test(
                control_df, treatment_df, metric,
                y_col, y_true_col, sensitive_col
            )
            
            # Effect size
            effect_size = self._compute_effect_size(
                control_df, treatment_df, metric,
                y_col, y_true_col, sensitive_col
            )
            
            # Interpretation
            interpretation = self._interpret_result(
                control_value, treatment_value,
                test_result['p_value'], effect_size, metric
            )
            
            result = ABTestResult(
                metric_name=metric,
                control_value=control_value,
                treatment_value=treatment_value,
                absolute_difference=treatment_value - control_value,
                relative_difference=(
                    (treatment_value - control_value) / control_value
                    if control_value != 0 else 0
                ),
                confidence_interval=test_result['ci'],
                p_value=test_result['p_value'],
                is_significant=test_result['p_value'] < self.alpha,
                effect_size=effect_size,
                sample_sizes={
                    'control': len(control_df),
                    'treatment': len(treatment_df)
                },
                interpretation=interpretation,
            )
            
            results[metric] = result
            
            logger.info(
                f"{metric}: control={control_value:.4f}, "
                f"treatment={treatment_value:.4f}, "
                f"p={test_result['p_value']:.4f}"
            )
        
        return results
    
    def analyze_heterogeneous_effects(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        subgroup_cols: List[str],
        y_col: str = 'y_pred',
        y_true_col: str = 'y_true',
        sensitive_col: str = 'sensitive',
    ) -> List[HeterogeneousEffectResult]:
        """
        Analyze heterogeneous treatment effects across subgroups.
        
        This identifies which demographic groups benefit most/least
        from the fairness intervention.
        
        Args:
            control_df: Control group data
            treatment_df: Treatment group data
            metric: Metric to evaluate
            subgroup_cols: Columns defining subgroups
            y_col: Prediction column name
            y_true_col: True label column name
            sensitive_col: Sensitive attribute column name
        
        Returns:
            List of HeterogeneousEffectResult objects
        """
        logger.info(
            f"Analyzing heterogeneous effects for {metric} "
            f"across {len(subgroup_cols)} dimensions"
        )
        
        results = []
        
        # Get unique values for each subgroup column
        subgroup_values = {
            col: sorted(
                set(control_df[col].unique()) | set(treatment_df[col].unique())
            )
            for col in subgroup_cols
        }
        
        # Analyze each subgroup combination
        for subgroup_combo in product(*subgroup_values.values()):
            # Create subgroup identifier
            subgroup_dict = dict(zip(subgroup_cols, subgroup_combo))
            subgroup_name = ', '.join(
                f"{k}={v}" for k, v in subgroup_dict.items()
            )
            
            # Filter data for this subgroup
            control_mask = pd.Series([True] * len(control_df))
            treatment_mask = pd.Series([True] * len(treatment_df))
            
            for col, val in subgroup_dict.items():
                control_mask &= (control_df[col] == val)
                treatment_mask &= (treatment_df[col] == val)
            
            control_sub = control_df[control_mask]
            treatment_sub = treatment_df[treatment_mask]
            
            # Skip if insufficient data
            if (len(control_sub) < self.min_sample_size or 
                len(treatment_sub) < self.min_sample_size):
                logger.warning(
                    f"Skipping {subgroup_name}: insufficient samples "
                    f"(control={len(control_sub)}, treatment={len(treatment_sub)})"
                )
                continue
            
            # Compute metric for subgroup
            control_value = self._compute_metric(
                control_sub, metric, y_col, y_true_col, sensitive_col
            )
            treatment_value = self._compute_metric(
                treatment_sub, metric, y_col, y_true_col, sensitive_col
            )
            
            # Bootstrap CI for treatment effect
            treatment_effect = treatment_value - control_value
            ci_lower, ci_upper = self._bootstrap_ci_difference(
                control_sub, treatment_sub, metric,
                y_col, y_true_col, sensitive_col
            )
            
            # Permutation test
            p_value = self._permutation_test_subgroup(
                control_sub, treatment_sub, metric,
                y_col, y_true_col, sensitive_col
            )
            
            result = HeterogeneousEffectResult(
                subgroup=subgroup_name,
                control_value=control_value,
                treatment_value=treatment_value,
                treatment_effect=treatment_effect,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                is_significant=p_value < self.alpha,
                sample_size=len(control_sub) + len(treatment_sub),
            )
            
            results.append(result)
            
            logger.info(
                f"{subgroup_name}: effect={treatment_effect:.4f}, "
                f"p={p_value:.4f}"
            )
        
        return results
    
    def multi_objective_analysis(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        performance_metric: str = 'accuracy',
        fairness_metric: str = 'demographic_parity',
        y_col: str = 'y_pred',
        y_true_col: str = 'y_true',
        sensitive_col: str = 'sensitive',
    ) -> Dict[str, any]:
        """
        Analyze trade-offs between performance and fairness.
        
        Args:
            control_df: Control group data
            treatment_df: Treatment group data
            performance_metric: Performance metric name
            fairness_metric: Fairness metric name
            y_col: Prediction column name
            y_true_col: True label column name
            sensitive_col: Sensitive attribute column name
        
        Returns:
            Dictionary with multi-objective analysis results
        """
        logger.info("Performing multi-objective analysis")
        
        # Analyze both metrics
        results = self.analyze_test(
            control_df, treatment_df,
            metrics=[performance_metric, fairness_metric],
            y_col=y_col, y_true_col=y_true_col, sensitive_col=sensitive_col
        )
        
        perf_result = results[performance_metric]
        fair_result = results[fairness_metric]
        
        # Compute trade-off ratio
        perf_change = perf_result.absolute_difference
        fair_change = fair_result.absolute_difference
        
        # For fairness, lower is better (reduction in disparity)
        # For performance, higher is usually better
        if performance_metric == 'accuracy':
            trade_off_ratio = -fair_change / perf_change if perf_change != 0 else 0
        else:
            # Generic case
            trade_off_ratio = fair_change / perf_change if perf_change != 0 else 0
        
        # Interpretation
        if perf_change > 0 and fair_change < 0:
            outcome = "Win-Win: Better performance AND better fairness"
        elif perf_change < 0 and fair_change < 0:
            outcome = f"Trade-off: {abs(perf_change):.1%} performance loss for {abs(fair_change):.3f} fairness gain"
        elif perf_change > 0 and fair_change > 0:
            outcome = "Lose-Lose: Worse performance AND worse fairness"
        else:
            outcome = "Mixed results"
        
        return {
            'performance_result': perf_result,
            'fairness_result': fair_result,
            'trade_off_ratio': trade_off_ratio,
            'outcome': outcome,
            'recommendation': self._generate_recommendation(
                perf_result, fair_result
            ),
        }
    
    def _validate_test_data(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame
    ):
        """Validate A/B test data."""
        if len(control_df) < self.min_sample_size:
            raise ValidationError(
                f"Control group too small: {len(control_df)} < {self.min_sample_size}"
            )
        
        if len(treatment_df) < self.min_sample_size:
            raise ValidationError(
                f"Treatment group too small: {len(treatment_df)} < {self.min_sample_size}"
            )
        
        # Check for required columns
        required_cols = ['y_pred', 'y_true', 'sensitive']
        for col in required_cols:
            if col not in control_df.columns:
                raise ValidationError(f"Control missing column: {col}")
            if col not in treatment_df.columns:
                raise ValidationError(f"Treatment missing column: {col}")
    
    def _compute_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> float:
        """Compute a single metric value."""
        y_pred = df[y_col].values
        y_true = df[y_true_col].values
        sensitive = df[sensitive_col].values
        
        if metric == 'accuracy':
            return (y_pred == y_true).mean()
        
        elif metric == 'demographic_parity':
            rate_0 = y_pred[sensitive == 0].mean()
            rate_1 = y_pred[sensitive == 1].mean()
            return abs(rate_0 - rate_1)
        
        elif metric == 'equalized_odds':
            # TPR difference
            tpr_0 = y_pred[(y_true == 1) & (sensitive == 0)].mean()
            tpr_1 = y_pred[(y_true == 1) & (sensitive == 1)].mean()
            tpr_diff = abs(tpr_0 - tpr_1)
            
            # FPR difference
            fpr_0 = y_pred[(y_true == 0) & (sensitive == 0)].mean()
            fpr_1 = y_pred[(y_true == 0) & (sensitive == 1)].mean()
            fpr_diff = abs(fpr_0 - fpr_1)
            
            return max(tpr_diff, fpr_diff)
        
        elif metric == 'equal_opportunity':
            # TPR difference only
            tpr_0 = y_pred[(y_true == 1) & (sensitive == 0)].mean()
            tpr_1 = y_pred[(y_true == 1) & (sensitive == 1)].mean()
            return abs(tpr_0 - tpr_1)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _statistical_test(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> Dict[str, any]:
        """Perform statistical test on metric difference."""
        # Bootstrap confidence interval for difference
        ci_lower, ci_upper = self._bootstrap_ci_difference(
            control_df, treatment_df, metric,
            y_col, y_true_col, sensitive_col
        )
        
        # Permutation test for p-value
        p_value = self._permutation_test(
            control_df, treatment_df, metric,
            y_col, y_true_col, sensitive_col
        )
        
        return {
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
        }
    
    def _bootstrap_ci_difference(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> Tuple[float, float]:
        """Bootstrap CI for metric difference."""
        differences = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            control_boot = control_df.sample(
                n=len(control_df), replace=True, random_state=None
            )
            treatment_boot = treatment_df.sample(
                n=len(treatment_df), replace=True, random_state=None
            )
            
            # Compute difference
            control_val = self._compute_metric(
                control_boot, metric, y_col, y_true_col, sensitive_col
            )
            treatment_val = self._compute_metric(
                treatment_boot, metric, y_col, y_true_col, sensitive_col
            )
            
            differences.append(treatment_val - control_val)
        
        # Percentile CI
        ci_lower = np.percentile(differences, 100 * self.alpha / 2)
        ci_upper = np.percentile(differences, 100 * (1 - self.alpha / 2))
        
        return ci_lower, ci_upper
    
    def _permutation_test(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> float:
        """Permutation test for metric difference."""
        # Observed difference
        control_val = self._compute_metric(
            control_df, metric, y_col, y_true_col, sensitive_col
        )
        treatment_val = self._compute_metric(
            treatment_df, metric, y_col, y_true_col, sensitive_col
        )
        observed_diff = abs(treatment_val - control_val)
        
        # Combine data
        combined = pd.concat([control_df, treatment_df])
        
        # Permutation distribution
        null_diffs = []
        n_control = len(control_df)
        
        for _ in range(1000):  # 1000 permutations
            # Shuffle group assignment
            shuffled = combined.sample(frac=1, random_state=None)
            perm_control = shuffled.iloc[:n_control]
            perm_treatment = shuffled.iloc[n_control:]
            
            # Compute difference
            perm_control_val = self._compute_metric(
                perm_control, metric, y_col, y_true_col, sensitive_col
            )
            perm_treatment_val = self._compute_metric(
                perm_treatment, metric, y_col, y_true_col, sensitive_col
            )
            
            null_diffs.append(abs(perm_treatment_val - perm_control_val))
        
        # P-value
        p_value = np.mean(np.array(null_diffs) >= observed_diff)
        
        return p_value
    
    def _permutation_test_subgroup(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> float:
        """Permutation test for subgroup (same as main test)."""
        return self._permutation_test(
            control_df, treatment_df, metric,
            y_col, y_true_col, sensitive_col
        )
    
    def _compute_effect_size(
        self,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> float:
        """Compute Cohen's d effect size."""
        # Get metric distributions
        control_vals = self._get_metric_distribution(
            control_df, metric, y_col, y_true_col, sensitive_col
        )
        treatment_vals = self._get_metric_distribution(
            treatment_df, metric, y_col, y_true_col, sensitive_col
        )
        
        # Cohen's d
        mean_diff = np.mean(treatment_vals) - np.mean(control_vals)
        pooled_std = np.sqrt(
            (np.var(control_vals) + np.var(treatment_vals)) / 2
        )
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def _get_metric_distribution(
        self,
        df: pd.DataFrame,
        metric: str,
        y_col: str,
        y_true_col: str,
        sensitive_col: str
    ) -> np.ndarray:
        """Get distribution for effect size computation."""
        # For accuracy, use individual correct/incorrect
        if metric == 'accuracy':
            return (df[y_col].values == df[y_true_col].values).astype(float)
        
        # For fairness metrics, use predictions
        return df[y_col].values
    
    def _interpret_result(
        self,
        control_value: float,
        treatment_value: float,
        p_value: float,
        effect_size: float,
        metric: str
    ) -> str:
        """Generate interpretation of A/B test result."""
        is_significant = p_value < self.alpha
        
        # Direction
        if metric == 'accuracy':
            improved = treatment_value > control_value
            change_desc = "improved" if improved else "decreased"
        else:  # Fairness metrics (lower is better)
            improved = treatment_value < control_value
            change_desc = "improved" if improved else "worsened"
        
        # Magnitude
        abs_diff = abs(treatment_value - control_value)
        
        if abs(effect_size) >= 0.8:
            magnitude = "large"
        elif abs(effect_size) >= 0.5:
            magnitude = "medium"
        elif abs(effect_size) >= 0.2:
            magnitude = "small"
        else:
            magnitude = "negligible"
        
        # Combine
        if is_significant:
            if improved:
                return f"Statistically significant improvement ({magnitude} effect)"
            else:
                return f"Statistically significant degradation ({magnitude} effect)"
        else:
            return f"No statistically significant change (p={p_value:.3f})"
    
    def _generate_recommendation(
        self,
        perf_result: ABTestResult,
        fair_result: ABTestResult
    ) -> str:
        """Generate deployment recommendation."""
        perf_improved = perf_result.treatment_value > perf_result.control_value
        fair_improved = fair_result.treatment_value < fair_result.control_value
        
        perf_sig = perf_result.is_significant
        fair_sig = fair_result.is_significant
        
        if fair_sig and fair_improved:
            if perf_sig and perf_improved:
                return "STRONGLY RECOMMEND: Improves both fairness and performance"
            elif not perf_sig:
                return "RECOMMEND: Significantly improves fairness with no significant performance impact"
            else:
                # Performance decreased
                perf_loss = abs(perf_result.relative_difference)
                fair_gain = abs(fair_result.relative_difference)
                if perf_loss < 0.02:  # Less than 2% loss
                    return f"RECOMMEND: Minor performance loss ({perf_loss:.1%}) for significant fairness gain"
                else:
                    return f"EVALUATE: {perf_loss:.1%} performance loss for fairness gain - stakeholder decision needed"
        
        elif fair_sig and not fair_improved:
            return "DO NOT RECOMMEND: Significantly worsens fairness"
        
        else:
            # Not significant fairness change
            if perf_sig and perf_improved:
                return "NEUTRAL: Performance improved but no fairness impact"
            else:
                return "DO NOT RECOMMEND: No significant benefits observed"


# Convenience function
def run_ab_test(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    subgroup_cols: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Convenience function to run complete A/B test analysis.
    
    Args:
        control_df: Control group data
        treatment_df: Treatment group data
        metrics: List of metrics to evaluate
        subgroup_cols: Columns for heterogeneous effects
        alpha: Significance level
    
    Returns:
        Dictionary with all analysis results
    """
    if metrics is None:
        metrics = ['accuracy', 'demographic_parity']
    
    analyzer = FairnessABTestAnalyzer(alpha=alpha)
    
    # Overall test
    overall_results = analyzer.analyze_test(
        control_df, treatment_df, metrics
    )
    
    # Heterogeneous effects (if subgroups specified)
    hetero_results = {}
    if subgroup_cols:
        for metric in metrics:
            hetero_results[metric] = analyzer.analyze_heterogeneous_effects(
                control_df, treatment_df, metric, subgroup_cols
            )
    
    # Multi-objective analysis
    multi_obj = analyzer.multi_objective_analysis(
        control_df, treatment_df
    )
    
    return {
        'overall': overall_results,
        'heterogeneous': hetero_results,
        'multi_objective': multi_obj,
    }