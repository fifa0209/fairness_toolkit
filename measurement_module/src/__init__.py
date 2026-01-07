"""Measurement Module Source - Core Implementation"""

# Use relative imports (dot notation) since we're inside the package
from .fairness_analyzer_simple import FairnessAnalyzer
from .metrics_engine import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)
from .statistical_validation import (
    bootstrap_confidence_interval,
    compute_effect_size_cohens_d,
)

__all__ = [
    'FairnessAnalyzer',
    'demographic_parity_difference',
    'equalized_odds_difference',
    'equal_opportunity_difference',
    'bootstrap_confidence_interval',
    'compute_effect_size_cohens_d',
]