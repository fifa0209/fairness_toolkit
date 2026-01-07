"""
Shared utilities for fairness pipeline.

Provides common schemas, constants, logging, and validation
used across measurement, mitigation, training, and monitoring modules.
"""

from shared.schemas import (
    FairnessMetricResult,
    BiasDetectionResult,
    PipelineConfig,
    DatasetMetadata,
    ModelMetadata,
    MonitoringAlert
)

from shared.constants import (
    FAIRNESS_METRICS,
    PROTECTED_ATTRIBUTES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_SAMPLES,
    MIN_GROUP_SIZE,
    FAIRNESS_THRESHOLDS,
    BIAS_DETECTION_THRESHOLDS,
    CONSTRAINT_TYPES,
    MONITORING_DEFAULTS
)

from shared.logging import (
    get_logger,
    log_metric,
    log_pipeline_stage,
    log_fairness_result,
    log_bias_detection,
    log_monitoring_alert,
    PipelineLogger
)

from shared.validation import (
    validate_dataframe,
    validate_protected_attribute,
    validate_predictions,
    validate_config,
    validate_sample_weights,
    validate_group_metrics,
    validate_confidence_interval,
    safe_divide,
    ValidationError
)

__version__ = "0.1.0"
__author__ = "FairML Consulting"

__all__ = [
    # Schemas
    "FairnessMetricResult",
    "BiasDetectionResult",
    "PipelineConfig",
    "DatasetMetadata",
    "ModelMetadata",
    "MonitoringAlert",
    # Constants
    "FAIRNESS_METRICS",
    "PROTECTED_ATTRIBUTES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "MIN_GROUP_SIZE",
    "FAIRNESS_THRESHOLDS",
    "BIAS_DETECTION_THRESHOLDS",
    "CONSTRAINT_TYPES",
    "MONITORING_DEFAULTS",
    # Logging
    "get_logger",
    "log_metric",
    "log_pipeline_stage",
    "log_fairness_result",
    "log_bias_detection",
    "log_monitoring_alert",
    "PipelineLogger",
    # Validation
    "validate_dataframe",
    "validate_protected_attribute",
    "validate_predictions",
    "validate_config",
    "validate_sample_weights",
    "validate_group_metrics",
    "validate_confidence_interval",
    "safe_divide",
    "ValidationError",
]