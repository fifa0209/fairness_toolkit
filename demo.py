from pathlib import Path
import yaml

# Load configuration
config_path = Path('config.yml')

if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Loaded configuration from config.yml")
else:
    # Use default configuration
    config = {
        'fairness_threshold': 0.1,
        'bootstrap_samples': 1000,
        'fairness_metrics': ['demographic_parity', 'equalized_odds', 'equal_opportunity'],
        'bias_detection': {
            'protected_attribute': 'gender',
            'representation_threshold': 0.2,
            'proxy_threshold': 0.5
        },
        'bias_mitigation': {
            'method': 'reweighting',
            'params': {}
        },
        'training': {
            'use_fairness_constraints': True,
            'constraint_type': 'demographic_parity',
            'eps': 0.05
        },
        'monitoring': {
            'window_size': 1000,
            'drift_threshold': 0.05
        }
    }
    print("⚠️  Using default configuration (config.yml not found)")

print("\n📋 Configuration:")
print(f"  Fairness threshold: {config['fairness_threshold']}")
print(f"  Metrics: {config['fairness_metrics']}")
print(f"  Mitigation method: {config['bias_mitigation']['method']}")
print(f"  Use fairness constraints: {config['training']['use_fairness_constraints']}")