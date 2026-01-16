"""
Logging utilities for fairness pipeline.
Provides structured logging with context for debugging and auditing.
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path


# Configure root logger format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_file: Optional file path for logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration."""
    return setup_logger(name)
# Setup basic config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a metric with structured context.
    
    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        context: Additional context (e.g., group, threshold)
    """
    context_str = ""
    if context:
        context_items = [f"{k}={v}" for k, v in context.items()]
        context_str = f" [{', '.join(context_items)}]"
    
    logger.info(f"METRIC: {metric_name}={value:.4f}{context_str}")


def log_pipeline_stage(
    logger: logging.Logger,
    stage: str,
    status: str = "started",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log pipeline execution stage.
    
    Args:
        logger: Logger instance
        stage: Stage name (e.g., 'measurement', 'mitigation')
        status: Status ('started', 'completed', 'failed')
        details: Additional details
    """
    detail_str = ""
    if details:
        detail_items = [f"{k}={v}" for k, v in details.items()]
        detail_str = f" - {', '.join(detail_items)}"
    
    level = logging.INFO if status != "failed" else logging.ERROR
    logger.log(level, f"STAGE [{status.upper()}]: {stage}{detail_str}")


def log_fairness_result(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    is_fair: bool,
    threshold: float,
    group_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log fairness metric result with interpretation.
    
    Args:
        logger: Logger instance
        metric_name: Fairness metric name
        value: Metric value
        is_fair: Whether metric passes fairness threshold
        threshold: Fairness threshold used
        group_metrics: Per-group metric values
    """
    status = "PASS" if is_fair else "FAIL"
    logger.info(
        f"FAIRNESS [{status}]: {metric_name}={value:.4f} "
        f"(threshold={threshold:.4f})"
    )
    
    if group_metrics:
        for group, gval in group_metrics.items():
            logger.info(f"  └─ {group}: {gval:.4f}")


def log_bias_detection(
    logger: logging.Logger,
    bias_type: str,
    detected: bool,
    severity: str,
    affected_groups: list,
) -> None:
    """
    Log bias detection result.
    
    Args:
        logger: Logger instance
        bias_type: Type of bias detected
        detected: Whether bias was detected
        severity: Severity level
        affected_groups: List of affected groups
    """
    if detected:
        groups_str = ", ".join(affected_groups)
        logger.warning(
            f"BIAS DETECTED [{severity}]: {bias_type} "
            f"affects {groups_str}"
        )
    else:
        logger.info(f"BIAS CHECK: {bias_type} - no significant bias detected")


def log_monitoring_alert(
    logger: logging.Logger,
    alert_type: str,
    severity: str,
    metric_name: str,
    message: str,
) -> None:
    """
    Log monitoring alert.
    
    Args:
        logger: Logger instance
        alert_type: Type of alert
        severity: Alert severity
        metric_name: Metric that triggered alert
        message: Alert message
    """
    level_map = {
        "LOW": logging.INFO,
        "HIGH": logging.WARNING,
        "CRITICAL": logging.ERROR,
    }
    level = level_map.get(severity, logging.WARNING)
    
    logger.log(
        level,
        f"ALERT [{severity}] {alert_type}: {metric_name} - {message}"
    )


def log_config_validation(
    logger: logging.Logger,
    config_name: str,
    errors: list,
) -> None:
    """
    Log configuration validation results.
    
    Args:
        logger: Logger instance
        config_name: Name of configuration
        errors: List of validation errors
    """
    if errors:
        logger.error(f"CONFIG VALIDATION FAILED: {config_name}")
        for err in errors:
            logger.error(f"  └─ {err}")
    else:
        logger.info(f"CONFIG VALIDATION PASSED: {config_name}")


class PipelineLogger:
    """
    Context manager for pipeline stage logging.
    Automatically logs start/end and captures exceptions.
    """
    
    def __init__(self, logger: logging.Logger, stage: str):
        self.logger = logger
        self.stage = stage
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_pipeline_stage(self.logger, self.stage, "started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            log_pipeline_stage(
                self.logger,
                self.stage,
                "completed",
                {"duration_seconds": f"{duration:.2f}"}
            )
        else:
            log_pipeline_stage(
                self.logger,
                self.stage,
                "failed",
                {"error": str(exc_val), "duration_seconds": f"{duration:.2f}"}
            )
        
        # Don't suppress exceptions
        return False