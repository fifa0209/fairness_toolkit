"""
Pipeline Builder - Config-driven sklearn pipeline construction.

Dynamically builds sklearn pipelines from YAML configuration.
"""

from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline

from pipeline_module.src.transformers import InstanceReweighting, GroupBalancer
from shared.logging import get_logger

logger = get_logger(__name__)


class FairnessPipelineBuilder:
    """
    Build sklearn pipelines from configuration.
    
    Example:
        >>> config = {
        ...     'bias_mitigation': {
        ...         'method': 'reweighting',
        ...         'params': {'alpha': 0.8}
        ...     }
        ... }
        >>> builder = FairnessPipelineBuilder(config)
        >>> pipeline = builder.build()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.steps: List[tuple] = []
    
    def add_bias_mitigation(self) -> 'FairnessPipelineBuilder':
        """Add bias mitigation transformer based on config."""
        mitigation_config = self.config.get('bias_mitigation', {})
        method = mitigation_config.get('method', 'reweighting')
        params = mitigation_config.get('params', {})
        
        if method == 'reweighting':
            transformer = InstanceReweighting(**params)
            self.steps.append(('reweighting', transformer))
            logger.info(f"Added InstanceReweighting: {params}")
        
        elif method == 'resampling':
            transformer = GroupBalancer(**params)
            self.steps.append(('resampling', transformer))
            logger.info(f"Added GroupBalancer: {params}")
        
        else:
            logger.warning(f"Unknown mitigation method: {method}")
        
        return self
    
    def add_preprocessing(self) -> 'FairnessPipelineBuilder':
        """Add preprocessing steps from config."""
        preprocessing = self.config.get('preprocessing', [])
        
        for step_config in preprocessing:
            step_name = step_config.get('name')
            step_type = step_config.get('type')
            params = step_config.get('params', {})
            
            if step_type == 'StandardScaler':
                from sklearn.preprocessing import StandardScaler
                self.steps.append((step_name, StandardScaler(**params)))
            
            elif step_type == 'MinMaxScaler':
                from sklearn.preprocessing import MinMaxScaler
                self.steps.append((step_name, MinMaxScaler(**params)))
            
            else:
                logger.warning(f"Unknown preprocessing type: {step_type}")
        
        return self
    
    def add_model(self, model: Any) -> 'FairnessPipelineBuilder':
        """Add final model to pipeline."""
        self.steps.append(('model', model))
        logger.info(f"Added model: {type(model).__name__}")
        return self
    
    def build(self) -> Pipeline:
        """Build and return sklearn Pipeline."""
        if not self.steps:
            raise ValueError("No steps added to pipeline")
        
        pipeline = Pipeline(self.steps)
        logger.info(f"Built pipeline with {len(self.steps)} steps")
        
        return pipeline
    
    def get_step_names(self) -> List[str]:
        """Get names of all pipeline steps."""
        return [name for name, _ in self.steps]


def build_fairness_pipeline(
    config: Dict[str, Any],
    model: Any,
) -> Pipeline:
    """
    Convenience function to build pipeline from config.
    
    Args:
        config: Pipeline configuration
        model: Final model to add
        
    Returns:
        Configured sklearn Pipeline
    """
    builder = FairnessPipelineBuilder(config)
    
    if config.get('bias_mitigation'):
        builder.add_bias_mitigation()
    
    if config.get('preprocessing'):
        builder.add_preprocessing()
    
    builder.add_model(model)
    
    return builder.build()