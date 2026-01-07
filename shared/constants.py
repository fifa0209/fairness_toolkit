"""
Constants for fairness pipeline.
Central location for configuration values, metric definitions, and thresholds.
"""

# Supported fairness metrics (48-hour scope: binary classification only)
FAIRNESS_METRICS = {
    "demographic_parity": {
        "name": "Demographic Parity Difference",
        "description": "Difference in positive prediction rates between groups",
        "formula": "P(Ŷ=1|A=a) - P(Ŷ=1|A=b)",
        "fair_range": (-0.1, 0.1),
        "library": "fairlearn",
    },
    "equalized_odds": {
        "name": "Equalized Odds Difference",
        "description": "Max difference in TPR and FPR between groups",
        "formula": "max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)",
        "fair_range": (-0.1, 0.1),
        "library": "fairlearn",
    },
    "equal_opportunity": {
        "name": "Equal Opportunity Difference",
        "description": "Difference in true positive rates between groups",
        "formula": "TPR_a - TPR_b",
        "fair_range": (-0.1, 0.1),
        "library": "fairlearn",
    },
}

# Protected attribute types
PROTECTED_ATTRIBUTES = {
    "race": ["White", "Black", "Asian", "Hispanic", "Other"],
    "gender": ["Male", "Female", "Non-binary"],
    "age": ["<25", "25-40", "40-60", ">60"],
    "binary": ["Group_0", "Group_1"],  # Generic binary for demo
}

# Statistical validation parameters
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 1000
MIN_GROUP_SIZE = 30  # Minimum samples for reliable statistics

# Effect size thresholds (Cohen's d interpretation)
EFFECT_SIZE_THRESHOLDS = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": 1.2,
}

# Fairness thresholds (industry-informed defaults)
FAIRNESS_THRESHOLDS = {
    "strict": 0.05,    # Research/high-stakes
    "moderate": 0.10,  # Standard recommendation
    "lenient": 0.20,   # Initial pilots
}

# Supported model types
SUPPORTED_MODELS = {
    "sklearn": ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"],
    "xgboost": ["XGBClassifier"],
    "pytorch": ["NeuralNetwork"],
}

# Bias detection thresholds
BIAS_DETECTION_THRESHOLDS = {
    "representation_bias": {
        "severe": 0.3,   # >30% difference from benchmark
        "moderate": 0.2,
        "mild": 0.1,
    },
    "proxy_correlation": {
        "high": 0.7,     # Correlation with protected attribute
        "medium": 0.5,
        "low": 0.3,
    },
}

# Reweighting methods
REWEIGHTING_METHODS = {
    "inverse_propensity": "Inverse propensity weighting based on group size",
    "uniform": "Equal weight across all groups",
}

# Fairness constraint types (for training)
CONSTRAINT_TYPES = {
    "demographic_parity": "fairlearn.reductions.DemographicParity",
    "equalized_odds": "fairlearn.reductions.EqualizedOdds",
    "equal_opportunity": "fairlearn.reductions.TruePositiveRateParity",
}

# Monitoring configuration
MONITORING_DEFAULTS = {
    "window_size": 1000,
    "drift_test": "ks_2samp",  # Kolmogorov-Smirnov
    "drift_alpha": 0.05,
    "alert_cooldown_minutes": 60,
}

# Severity levels for alerts
SEVERITY_LEVELS = ["LOW", "HIGH", "CRITICAL"]

# MLflow tracking defaults
MLFLOW_DEFAULTS = {
    "experiment_name": "fairness_pipeline",
    "artifact_location": "./mlruns",
    "log_frequency": "per_epoch",
}

# Visualization defaults
VIZ_DEFAULTS = {
    "color_scheme": {
        "fair": "#2ecc71",
        "warning": "#f39c12", 
        "unfair": "#e74c3c",
        "neutral": "#95a5a6",
    },
    "figure_size": (10, 6),
    "dpi": 100,
}

# Task types
TASK_TYPES = ["binary_classification", "regression"]

# File paths (relative to project root)
DEFAULT_PATHS = {
    "config": "config.yml",
    "data": "data/",
    "models": "models/",
    "logs": "logs/",
    "reports": "reports/",
}