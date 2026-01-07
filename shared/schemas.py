"""
Data schemas for fairness pipeline.
Defines dataclasses for structured data exchange between modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


@dataclass
class FairnessMetricResult:
    """Result from fairness metric computation with statistical validation."""
    
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    group_metrics: Dict[str, float]
    group_sizes: Dict[str, int]
    interpretation: str
    is_fair: bool
    threshold: float
    effect_size: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "group_metrics": self.group_metrics,
            "group_sizes": self.group_sizes,
            "interpretation": self.interpretation,
            "is_fair": self.is_fair,
            "threshold": self.threshold,
            "effect_size": self.effect_size,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BiasDetectionResult:
    """Result from bias detection analysis."""
    
    bias_type: str  # 'representation', 'measurement', 'proxy'
    detected: bool
    severity: str  # 'low', 'medium', 'high'
    affected_groups: List[str]
    evidence: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias_type": self.bias_type,
            "detected": self.detected,
            "severity": self.severity,
            "affected_groups": self.affected_groups,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PipelineConfig:
    """Configuration for fairness pipeline execution."""
    
    # Data configuration
    target_column: str
    protected_attribute: str
    favorable_label: int = 1
    
    # Metric configuration
    fairness_metrics: List[str] = field(default_factory=lambda: ["demographic_parity", "equalized_odds"])
    fairness_threshold: float = 0.1
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Mitigation configuration
    apply_reweighting: bool = True
    reweighting_method: str = "inverse_propensity"
    
    # Training configuration
    use_fairness_constraints: bool = False
    constraint_type: Optional[str] = None  # 'demographic_parity', 'equalized_odds'
    
    # Monitoring configuration
    enable_monitoring: bool = True
    monitoring_window_size: int = 1000
    drift_test_alpha: float = 0.05
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.target_column:
            errors.append("target_column is required")
        if not self.protected_attribute:
            errors.append("protected_attribute is required")
        if self.fairness_threshold <= 0 or self.fairness_threshold >= 1:
            errors.append("fairness_threshold must be between 0 and 1")
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            errors.append("confidence_level must be between 0 and 1")
        if self.bootstrap_samples < 100:
            errors.append("bootstrap_samples should be at least 100")
        
        valid_metrics = ["demographic_parity", "equalized_odds", "equal_opportunity"]
        invalid = [m for m in self.fairness_metrics if m not in valid_metrics]
        if invalid:
            errors.append(f"Invalid metrics: {invalid}. Valid: {valid_metrics}")
        
        if self.use_fairness_constraints and not self.constraint_type:
            errors.append("constraint_type required when use_fairness_constraints=True")
        
        return errors


@dataclass
class DatasetMetadata:
    """Metadata about a dataset for fairness analysis."""
    
    name: str
    n_samples: int
    n_features: int
    task_type: str  # 'binary_classification', 'regression'
    protected_attribute: str
    protected_groups: List[str]
    group_distribution: Dict[str, int]
    class_balance: Optional[Dict[str, int]] = None
    
    @property
    def min_group_size(self) -> int:
        return min(self.group_distribution.values())
    
    @property
    def imbalance_ratio(self) -> float:
        sizes = list(self.group_distribution.values())
        return max(sizes) / min(sizes) if sizes else 1.0


@dataclass
class ModelMetadata:
    """Metadata about a model for fairness tracking."""
    
    model_name: str
    model_type: str  # 'sklearn', 'pytorch', 'xgboost'
    fairness_intervention: Optional[str] = None  # 'reweighting', 'reductions', 'regularization'
    training_timestamp: datetime = field(default_factory=datetime.now)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "fairness_intervention": self.fairness_intervention,
            "training_timestamp": self.training_timestamp.isoformat(),
            "hyperparameters": self.hyperparameters,
        }


@dataclass
class MonitoringAlert:
    """Alert from fairness monitoring system."""
    
    alert_type: str  # 'drift', 'threshold_violation', 'data_quality'
    severity: str  # 'CRITICAL', 'HIGH', 'LOW'
    metric_name: str
    current_value: float
    reference_value: Optional[float]
    affected_groups: List[str]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "reference_value": self.reference_value,
            "affected_groups": self.affected_groups,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }