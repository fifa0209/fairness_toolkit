"""
Config Loader - Load and validate pipeline configuration from YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

from shared.logging import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    Load and validate pipeline configuration.
    
    Example:
        >>> loader = ConfigLoader('config.yml')
        >>> config = loader.load()
        >>> errors = loader.validate()
    """
    
    def __init__(self, config_path: str):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {self.config_path}")
        return self.config
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required sections
        required_sections = ['bias_detection', 'bias_mitigation']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate bias detection config
        if 'bias_detection' in self.config:
            bd_config = self.config['bias_detection']
            
            if 'protected_attribute' not in bd_config:
                errors.append("bias_detection.protected_attribute is required")
            
            valid_checks = ['representation', 'proxy', 'statistical_disparity']
            checks = bd_config.get('checks', [])
            invalid_checks = [c for c in checks if c not in valid_checks]
            if invalid_checks:
                errors.append(f"Invalid bias checks: {invalid_checks}")
        
        # Validate bias mitigation config
        if 'bias_mitigation' in self.config:
            bm_config = self.config['bias_mitigation']
            
            valid_methods = ['reweighting', 'resampling', 'none']
            method = bm_config.get('method')
            if method and method not in valid_methods:
                errors.append(f"Invalid mitigation method: {method}")
        
        if errors:
            logger.error(f"Config validation failed: {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        else:
            logger.info("Config validation passed")
        
        return errors
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def save(self, output_path: str) -> None:
        """Save configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved config to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load and validate config.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    loader = ConfigLoader(config_path)
    config = loader.load()
    
    errors = loader.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")
    
    return config