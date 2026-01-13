"""
Statistical Power Analysis for Fairness Tests

Provides tools to compute required sample sizes, power calculations,
and minimum detectable effects for fairness experiments.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PowerAnalysisResult:
    """Results from power analysis."""
    
    metric_name: str
    analysis_type: str  # 'sample_size', 'power', or 'effect_size'
    
    # Input parameters
    alpha: float
    power: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size_per_group: Optional[int] = None
    
    # Computed result
    result_value: float = None
    
    # Additional info
    assumptions: Dict[str, any] = None
    interpretation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'analysis_type': self.analysis_type,
            'alpha': self.alpha,
            'power': self.power,
            'effect_size': self.effect_size,
            'sample_size_per_group': self.sample_size_per_group,
            'result_value': self.result_value,
            'assumptions': self.assumptions,
            'interpretation': self.interpretation,
        }


class FairnessPowerAnalyzer:
    """
    Statistical power analysis for fairness tests.
    
    Helps determine:
    - Required sample size for desired power
    - Statistical power for given sample size
    - Minimum detectable effect size
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize power analyzer.
        
        Args:
            alpha: Significance level (Type I error rate)
        """
        self.alpha = alpha
        logger.info(f"Initialized FairnessPowerAnalyzer (alpha={alpha})")
    
    def sample_size_for_proportion_test(
        self,
        baseline_rate: float,
        target_rate: float,
        power: float = 0.80,
        ratio: float = 1.0
    ) -> PowerAnalysisResult:
        """
        Calculate required sample size for two-proportion test.
        
        This is relevant for demographic parity tests where we're
        comparing positive prediction rates between groups.
        
        Args:
            baseline_rate: Expected rate in control/reference group
            target_rate: Expected rate in treatment/comparison group
            power: Desired statistical power (1 - Type II error)
            ratio: Sample size ratio (n_treatment / n_control)
        
        Returns:
            PowerAnalysisResult with required sample sizes
        """
        logger.info(
            f"Computing sample size for proportion test: "
            f"p1={baseline_rate}, p2={target_rate}, power={power}"
        )
        
        # Effect size (difference in proportions)
        effect_size = abs(target_rate - baseline_rate)
        
        # Pooled proportion
        p_bar = (baseline_rate + target_rate) / 2
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        # Formula: n = [(z_α + z_β)² × (p1(1-p1) + p2(1-p2)/r)] / (p1 - p2)²
        numerator = (z_alpha + z_beta) ** 2
        numerator *= (
            baseline_rate * (1 - baseline_rate) +
            (target_rate * (1 - target_rate)) / ratio
        )
        denominator = effect_size ** 2
        
        n_control = int(np.ceil(numerator / denominator))
        n_treatment = int(np.ceil(n_control * ratio))
        
        result = PowerAnalysisResult(
            metric_name='demographic_parity',
            analysis_type='sample_size',
            alpha=self.alpha,
            power=power,
            effect_size=effect_size,
            sample_size_per_group=n_control,
            result_value=n_control,
            assumptions={
                'baseline_rate': baseline_rate,
                'target_rate': target_rate,
                'group_ratio': ratio,
                'n_control': n_control,
                'n_treatment': n_treatment,
                'total_n': n_control + n_treatment,
            },
            interpretation=self._interpret_sample_size(
                n_control, n_treatment, effect_size, power
            ),
        )
        
        logger.info(
            f"Required sample size: n_control={n_control}, "
            f"n_treatment={n_treatment}"
        )
        
        return result
    
    def power_for_proportion_test(
        self,
        baseline_rate: float,
        target_rate: float,
        n_control: int,
        n_treatment: Optional[int] = None
    ) -> PowerAnalysisResult:
        """
        Calculate statistical power for two-proportion test.
        
        Uses the standard formula for power of a two-proportion z-test.
        
        Args:
            baseline_rate: Rate in control group
            target_rate: Rate in treatment group
            n_control: Sample size in control group
            n_treatment: Sample size in treatment group (defaults to n_control)
        
        Returns:
            PowerAnalysisResult with computed power
        """
        if n_treatment is None:
            n_treatment = n_control
        
        logger.info(
            f"Computing power: p1={baseline_rate}, p2={target_rate}, "
            f"n1={n_control}, n2={n_treatment}"
        )
        
        # Effect size
        effect_size = abs(target_rate - baseline_rate)
        
        # Pooled proportion for null hypothesis
        p_pooled = (baseline_rate + target_rate) / 2
        
        # Standard error under null
        se_null = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        
        # Standard error under alternative
        se_alt = np.sqrt(
            baseline_rate * (1 - baseline_rate) / n_control +
            target_rate * (1 - target_rate) / n_treatment
        )
        
        # Z critical value for two-tailed test at alpha
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        
        # Calculate power
        # For a two-sided test, power = P(reject H0 | H1 true)
        # This is P(|Z| > z_alpha | delta != 0)
        
        # Under H1, test statistic ~ N(delta/se_alt, 1)
        # We reject if |observed_diff| > z_alpha * se_null
        # Converting to z-score under H1: reject if |Z| > z_alpha * se_null / se_alt
        
        # Non-centrality parameter
        ncp = effect_size / se_alt
        
        # Critical value in z-score units under alternative
        critical_z = z_alpha * se_null / se_alt
        
        # Power for two-sided test
        # P(Z > critical_z | ncp) + P(Z < -critical_z | ncp) where Z ~ N(ncp, 1)
        power = 1 - stats.norm.cdf(critical_z - ncp) + stats.norm.cdf(-critical_z - ncp)
        
        result = PowerAnalysisResult(
            metric_name='demographic_parity',
            analysis_type='power',
            alpha=self.alpha,
            power=power,
            effect_size=effect_size,
            sample_size_per_group=n_control,
            result_value=power,
            assumptions={
                'baseline_rate': baseline_rate,
                'target_rate': target_rate,
                'n_control': n_control,
                'n_treatment': n_treatment,
            },
            interpretation=self._interpret_power(power, effect_size),
        )
        
        logger.info(f"Computed power: {power:.3f}")
        
        return result
    
    def minimum_detectable_effect(
        self,
        baseline_rate: float,
        n_control: int,
        n_treatment: Optional[int] = None,
        power: float = 0.80
    ) -> PowerAnalysisResult:
        """
        Calculate minimum detectable effect size.
        
        Args:
            baseline_rate: Rate in control group
            n_control: Sample size in control group
            n_treatment: Sample size in treatment group
            power: Desired power
        
        Returns:
            PowerAnalysisResult with minimum detectable effect
        """
        if n_treatment is None:
            n_treatment = n_control
        
        logger.info(
            f"Computing MDE: p1={baseline_rate}, "
            f"n1={n_control}, n2={n_treatment}, power={power}"
        )
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled variance (conservative estimate)
        pooled_var = baseline_rate * (1 - baseline_rate)
        
        # MDE calculation
        mde = (z_alpha + z_beta) * np.sqrt(
            pooled_var * (1/n_control + 1/n_treatment)
        )
        
        result = PowerAnalysisResult(
            metric_name='demographic_parity',
            analysis_type='effect_size',
            alpha=self.alpha,
            power=power,
            effect_size=mde,
            sample_size_per_group=n_control,
            result_value=mde,
            assumptions={
                'baseline_rate': baseline_rate,
                'n_control': n_control,
                'n_treatment': n_treatment,
            },
            interpretation=self._interpret_mde(mde, baseline_rate),
        )
        
        logger.info(f"Minimum detectable effect: {mde:.4f}")
        
        return result
    
    def subgroup_power_analysis(
        self,
        overall_n: int,
        subgroup_proportions: Dict[str, float],
        baseline_rate: float,
        target_rate: float,
        power_threshold: float = 0.80
    ) -> Dict[str, PowerAnalysisResult]:
        """
        Analyze power for detecting effects in subgroups.
        
        Critical for ensuring we can detect heterogeneous treatment
        effects across demographic intersections.
        
        Args:
            overall_n: Total sample size
            subgroup_proportions: Dict of subgroup -> proportion
            baseline_rate: Expected baseline rate
            target_rate: Expected target rate
            power_threshold: Minimum acceptable power
        
        Returns:
            Dictionary mapping subgroup names to PowerAnalysisResult
        """
        logger.info(
            f"Analyzing subgroup power for {len(subgroup_proportions)} groups"
        )
        
        results = {}
        
        for subgroup, proportion in subgroup_proportions.items():
            # Subgroup sample size
            n_subgroup = int(overall_n * proportion)
            
            # Assume equal split between control and treatment
            n_control_sub = n_subgroup // 2
            n_treatment_sub = n_subgroup // 2
            
            # Compute power for this subgroup
            power_result = self.power_for_proportion_test(
                baseline_rate=baseline_rate,
                target_rate=target_rate,
                n_control=n_control_sub,
                n_treatment=n_treatment_sub
            )
            
            # Add subgroup identifier
            power_result.metric_name = f"demographic_parity_{subgroup}"
            
            # Check if underpowered
            if power_result.power < power_threshold:
                logger.warning(
                    f"Subgroup '{subgroup}' underpowered: "
                    f"power={power_result.power:.3f} < {power_threshold}"
                )
                power_result.interpretation += (
                    f" WARNING: Underpowered for subgroup analysis "
                    f"(below {power_threshold:.0%} threshold)."
                )
            
            results[subgroup] = power_result
        
        return results
    
    def sample_size_for_equivalence_test(
        self,
        margin: float,
        baseline_rate: float,
        power: float = 0.80
    ) -> PowerAnalysisResult:
        """
        Calculate sample size for equivalence test.
        
        Used when goal is to show two groups are equivalent
        (within some margin) rather than different.
        
        Args:
            margin: Equivalence margin (e.g., 0.05 for 5%)
            baseline_rate: Expected rate
            power: Desired power
        
        Returns:
            PowerAnalysisResult with required sample size
        """
        logger.info(
            f"Computing sample size for equivalence test: "
            f"margin={margin}, p={baseline_rate}, power={power}"
        )
        
        # For equivalence, we use TOST (two one-sided tests)
        # More stringent than regular test
        z_alpha = stats.norm.ppf(1 - self.alpha)  # One-sided
        z_beta = stats.norm.ppf(power)
        
        # Sample size (conservative)
        variance = baseline_rate * (1 - baseline_rate)
        n = ((z_alpha + z_beta) ** 2 * 2 * variance) / (margin ** 2)
        n = int(np.ceil(n))
        
        result = PowerAnalysisResult(
            metric_name='equivalence_test',
            analysis_type='sample_size',
            alpha=self.alpha,
            power=power,
            effect_size=margin,
            sample_size_per_group=n,
            result_value=n,
            assumptions={
                'equivalence_margin': margin,
                'baseline_rate': baseline_rate,
                'test_type': 'TOST',
            },
            interpretation=(
                f"To demonstrate equivalence within {margin:.2%} margin "
                f"with {power:.0%} power, need n={n} per group. "
                f"Total sample size: {2*n}"
            ),
        )
        
        logger.info(f"Required sample size for equivalence: {n} per group")
        
        return result
    
    def _interpret_sample_size(
        self,
        n_control: int,
        n_treatment: int,
        effect_size: float,
        power: float
    ) -> str:
        """Generate interpretation for sample size result."""
        total_n = n_control + n_treatment
        
        interpretation = (
            f"To detect a difference of {effect_size:.2%} with {power:.0%} power "
            f"at α={self.alpha}, need {n_control} samples in control and "
            f"{n_treatment} in treatment (total: {total_n}). "
        )
        
        if effect_size < 0.05:
            interpretation += (
                "Note: Very small effect size may not be practically significant."
            )
        
        if total_n > 10000:
            interpretation += (
                "Warning: Large sample size required. Consider accepting larger "
                "effect size or lower power."
            )
        
        return interpretation
    
    def _interpret_power(self, power: float, effect_size: float) -> str:
        """Generate interpretation for power result."""
        if power >= 0.80:
            assessment = "adequate"
        elif power >= 0.70:
            assessment = "marginal"
        else:
            assessment = "insufficient"
        
        interpretation = (
            f"Statistical power is {power:.1%}, which is {assessment} "
            f"for detecting an effect size of {effect_size:.2%}. "
        )
        
        if power < 0.80:
            interpretation += (
                f"Recommend increasing sample size to achieve 80% power. "
                f"Current design has {(1-power):.1%} risk of missing true effect."
            )
        
        return interpretation
    
    def _interpret_mde(self, mde: float, baseline_rate: float) -> str:
        """Generate interpretation for MDE result."""
        relative_mde = mde / baseline_rate if baseline_rate > 0 else 0
        
        interpretation = (
            f"With current sample size, can reliably detect effects "
            f"of {mde:.2%} or larger (absolute) or {relative_mde:.1%} (relative). "
        )
        
        if mde > 0.10:
            interpretation += (
                "Warning: Only large effects detectable. "
                "Consider increasing sample size for more sensitivity."
            )
        elif mde < 0.02:
            interpretation += (
                "Good sensitivity: can detect small effects."
            )
        
        return interpretation


class ExperimentDesigner:
    """
    Helper for designing fairness experiments with proper power.
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """
        Initialize designer.
        
        Args:
            alpha: Significance level
            power: Target power
        """
        self.alpha = alpha
        self.power = power
        self.analyzer = FairnessPowerAnalyzer(alpha=alpha)
        
        logger.info(
            f"Initialized ExperimentDesigner (alpha={alpha}, power={power})"
        )
    
    def design_ab_test(
        self,
        baseline_rate: float,
        minimum_effect: float,
        subgroups: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Design A/B test with proper power for all analyses.
        
        Args:
            baseline_rate: Expected baseline positive rate
            minimum_effect: Minimum effect to detect
            subgroups: Subgroup proportions (if subgroup analysis needed)
        
        Returns:
            Dictionary with design recommendations
        """
        logger.info("Designing A/B test")
        
        target_rate = baseline_rate + minimum_effect
        
        # Overall sample size
        overall_result = self.analyzer.sample_size_for_proportion_test(
            baseline_rate=baseline_rate,
            target_rate=target_rate,
            power=self.power
        )
        
        design = {
            'overall': overall_result,
            'recommendations': [],
        }
        
        # Add buffer for attrition
        attrition_rate = 0.10
        buffered_n = int(overall_result.result_value / (1 - attrition_rate))
        design['recommendations'].append(
            f"Account for ~{attrition_rate:.0%} attrition: "
            f"recruit {buffered_n} per group"
        )
        
        # Subgroup analysis
        if subgroups:
            total_n = overall_result.assumptions['total_n']
            subgroup_results = self.analyzer.subgroup_power_analysis(
                overall_n=total_n,
                subgroup_proportions=subgroups,
                baseline_rate=baseline_rate,
                target_rate=target_rate,
                power_threshold=self.power
            )
            
            design['subgroups'] = subgroup_results
            
            # Check if any subgroups underpowered
            underpowered = [
                name for name, result in subgroup_results.items()
                if result.power < self.power
            ]
            
            if underpowered:
                design['recommendations'].append(
                    f"WARNING: Subgroups {underpowered} are underpowered. "
                    f"Consider increasing overall sample size or accepting "
                    f"lower power for subgroup analyses."
                )
        
        # Duration estimate
        if 'daily_volume' in design:
            daily_volume = design['daily_volume']
            total_needed = overall_result.assumptions['total_n']
            days_needed = int(np.ceil(total_needed / daily_volume))
            design['estimated_duration_days'] = days_needed
            design['recommendations'].append(
                f"At {daily_volume} samples/day, expect {days_needed} day duration"
            )
        
        return design


# Convenience function
def quick_sample_size_calculator(
    current_disparity: float,
    target_disparity: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Quick calculator for required sample size.
    
    Args:
        current_disparity: Current demographic parity difference
        target_disparity: Target demographic parity after intervention
        alpha: Significance level
        power: Desired power
    
    Returns:
        Required sample size per group
    """
    # Assume 50% baseline rate (conservative)
    baseline_rate = 0.5
    
    analyzer = FairnessPowerAnalyzer(alpha=alpha)
    result = analyzer.sample_size_for_proportion_test(
        baseline_rate=baseline_rate,
        target_rate=baseline_rate + (target_disparity - current_disparity),
        power=power
    )
    
    return result.sample_size_per_group