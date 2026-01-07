"""
Monitoring Module - Production fairness tracking and alerting.

Provides:
- RealTimeFairnessTracker: Track metrics with sliding windows
- FairnessDriftDetector: Detect statistical changes in fairness
- ThresholdAlertSystem: Simple threshold-based alerts
- FairnessMonitoringDashboard: Interactive visualizations

Quick Start:
    from monitoring_module import RealTimeFairnessTracker
    
    tracker = RealTimeFairnessTracker(window_size=1000)
    
    # Process predictions
    tracker.add_batch(y_pred, y_true, sensitive_features)
    
    # Get current metrics
    metrics = tracker.get_current_metrics()
    if metrics['demographic_parity'] > 0.1:
        alert("Fairness violation!")
"""

from monitoring_module.src.realtime_tracker import (
    RealTimeFairnessTracker,
    BatchFairnessMonitor,
)

from monitoring_module.src.drift_detection import (
    FairnessDriftDetector,
    ThresholdAlertSystem,
)

# Dashboard imports (optional - requires plotly)
try:
    from monitoring_module.src.dashboard import (
        FairnessMonitoringDashboard,
        generate_monitoring_report,
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    FairnessMonitoringDashboard = None
    generate_monitoring_report = None

__all__ = [
    # Real-time tracking
    'RealTimeFairnessTracker',
    'BatchFairnessMonitor',
    # Drift detection
    'FairnessDriftDetector',
    'ThresholdAlertSystem',
    # Dashboard (if available)
    'FairnessMonitoringDashboard',
    'generate_monitoring_report',
    'DASHBOARD_AVAILABLE',
]