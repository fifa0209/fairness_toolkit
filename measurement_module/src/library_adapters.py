# measurement_module/src/library_adapters.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import warnings

class FairnessLibraryAdapter(ABC):
    """Abstract base class for fairness library adapters."""
    
    @abstractmethod
    def compute_metric(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """Compute metric using the underlying library."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> list:
        """Return list of metrics supported by this adapter."""
        pass

class AIF360Adapter(FairnessLibraryAdapter):
    """Adapter for IBM AIF360 library."""
    
    def __init__(self):
        try:
            from aif360.datasets import BinaryLabelDataset
            from aif360.metrics import BinaryLabelDatasetMetric
            self.available = True
        except ImportError:
            self.available = False
            warnings.warn("AIF360 not installed. Install with: pip install aif360", ImportWarning)
    
    def compute_metric(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """
        Compute metric using AIF360.
        
        Supported metrics:
        - statistical_parity_difference
        - disparate_impact
        - average_odds_difference
        """
        if not self.available:
            raise ImportError("AIF360 is not available")
        
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
        
        # Convert inputs to AIF360 format
        # Implementation here...
        return {"value": 0.0}  # Placeholder
    
    def get_available_metrics(self) -> list:
        return [
            'statistical_parity_difference',
            'disparate_impact',
            'average_odds_difference',
            'equal_opportunity_difference',
            'theil_index'
        ]

class FairlearnAdapter(FairnessLibraryAdapter):
    """Adapter for Microsoft Fairlearn library."""
    
    def __init__(self):
        try:
            from fairlearn.metrics import (
                demographic_parity_difference,
                equalized_odds_difference
            )
            self.available = True
        except ImportError:
            self.available = False
            warnings.warn("Fairlearn not installed. Install with: pip install fairlearn", ImportWarning)
    
    def compute_metric(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """Compute metric using Fairlearn."""
        if not self.available:
            raise ImportError("Fairlearn is not available")
        
        from fairlearn.metrics import MetricFrame
        # Implementation here...
        return {"value": 0.0}  # Placeholder
    
    def get_available_metrics(self) -> list:
        return [
            'demographic_parity_difference',
            'demographic_parity_ratio',
            'equalized_odds_difference',
            'equalized_odds_ratio'
        ]

class AequitasAdapter(FairnessLibraryAdapter):
    """Adapter for Aequitas library."""
    
    def __init__(self):
        try:
            from aequitas.group import Group
            from aequitas.bias import Bias
            self.available = True
        except ImportError:
            self.available = False
            warnings.warn("Aequitas not installed. Install with: pip install aequitas", ImportWarning)
    
    def compute_metric(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """Compute metric using Aequitas."""
        if not self.available:
            raise ImportError("Aequitas is not available")
        
        from aequitas.group import Group
        from aequitas.bias import Bias
        # Implementation here...
        return {"value": 0.0}  # Placeholder
    
    def get_available_metrics(self) -> list:
        return [
            'ppr_disparity',  # Predicted Positive Rate
            'pprev_disparity',  # Predicted Prevalence
            'fdr_disparity',  # False Discovery Rate
            'for_disparity'  # False Omission Rate
        ]

class UnifiedFairnessAnalyzer:
    """
    Unified interface that delegates to multiple fairness libraries.
    
    This is the main entry point that users interact with.
    """
    
    def __init__(self, preferred_library: Optional[str] = None):
        """
        Initialize with optional library preference.
        
        Args:
            preferred_library: 'aif360', 'fairlearn', 'aequitas', or None for auto
        """
        self.adapters = {
            'aif360': AIF360Adapter(),
            'fairlearn': FairlearnAdapter(),
            'aequitas': AequitasAdapter(),
        }
        
        self.preferred_library = preferred_library
        
        # Check which libraries are available
        self.available_libraries = ['native']  # Native is always available
        for name, adapter in self.adapters.items():
            if adapter.available:
                self.available_libraries.append(name)
    
    def compute_metric(
        self,
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        library: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute fairness metric using specified or preferred library.
        
        Args:
            metric_name: Metric to compute
            y_true: True labels
            y_pred: Predictions
            sensitive_features: Protected attributes
            library: Which library to use (None = auto-select)
            **kwargs: Additional arguments
        
        Returns:
            Standardized metric result dictionary
        """
        # Auto-select library if not specified
        if library is None:
            library = self._select_library(metric_name)
        
        if library not in self.available_libraries:
            raise ValueError(
                f"Library '{library}' not available. "
                f"Available: {self.available_libraries}"
            )
        
        # Use native implementation if requested or library unavailable
        if library == 'native':
            from .fairness_analyzer_simple import FairnessAnalyzer
            analyzer = FairnessAnalyzer()
            return analyzer.compute_metric(
                y_true, y_pred, sensitive_features,
                metric=metric_name, **kwargs
            )
        
        # Delegate to library adapter
        adapter = self.adapters[library]
        return adapter.compute_metric(
            metric_name,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            **kwargs
        )
    
    def _select_library(self, metric_name: str) -> str:
        """Auto-select best library for a metric."""
        if self.preferred_library and self.preferred_library in self.available_libraries:
            return self.preferred_library
        
        # Default to native implementation
        return 'native'
    
    def list_available_metrics(self, library: Optional[str] = None) -> Dict[str, list]:
        """List all available metrics across libraries."""
        if library:
            if library == 'native':
                return {
                    'native': [
                        'demographic_parity',
                        'equalized_odds',
                        'equal_opportunity'
                    ]
                }
            elif library in self.adapters and self.adapters[library].available:
                return {library: self.adapters[library].get_available_metrics()}
            return {}
        
        # Return all metrics from all libraries
        all_metrics = {}
        
        # Add native metrics first
        all_metrics['native'] = [
            'demographic_parity',
            'equalized_odds',
            'equal_opportunity'
        ]
        
        # Add metrics from available adapters
        for name, adapter in self.adapters.items():
            if adapter.available:
                all_metrics[name] = adapter.get_available_metrics()
        
        return all_metrics