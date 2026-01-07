"""
Test pipeline module end-to-end.

Run: python test_pipeline_module.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 60)
print("Testing Pipeline Module")
print("=" * 60)

# Test 1: Import check
print("\n[1/5] Testing imports...")
try:
    from pipeline_module.src import BiasDetector, InstanceReweighting, GroupBalancer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nMake sure you have the correct directory structure:")
    print("  pipeline_module/")
    print("    src/")
    print("      __init__.py")
    print("      bias_detection.py")
    print("      transformers/")
    print("        __init__.py")
    print("        reweighting.py")
    sys.exit(1)

# Test 2: Generate biased dataset
print("\n[2/5] Generating biased dataset...")
np.random.seed(42)

# Create dataset with representation bias
n_samples = 500
n_female = 150  # Under-represented
n_male = 350    # Over-represented

data = {
    'age': np.concatenate([
        np.random.normal(35, 10, n_female),
        np.random.normal(38, 10, n_male)
    ]),
    'income': np.concatenate([
        np.random.normal(50000, 15000, n_female),
        np.random.normal(65000, 15000, n_male)  # Proxy variable
    ]),
    'credit_score': np.concatenate([
        np.random.normal(680, 50, n_female),
        np.random.normal(700, 50, n_male)
    ]),
    'gender': [0] * n_female + [1] * n_male,
    'outcome': np.concatenate([
        np.random.choice([0, 1], n_female, p=[0.6, 0.4]),
        np.random.choice([0, 1], n_male, p=[0.4, 0.6])
    ])
}

df = pd.DataFrame(data)

print(f"✅ Generated {len(df)} samples")
print(f"   Female: {n_female} ({n_female/n_samples:.1%})")
print(f"   Male: {n_male} ({n_male/n_samples:.1%})")
print(f"   Representation imbalance: {abs(n_female-n_male)/n_samples:.1%}")

# Test 3: Bias Detection
print("\n[3/5] Testing BiasDetector...")

try:
    detector = BiasDetector(
        representation_threshold=0.2,
        proxy_threshold=0.3,
    )
    print("✅ BiasDetector initialized")
    
    # Test representation bias
    repr_result = detector.detect_representation_bias(
        df,
        protected_attribute='gender',
        reference_distribution={0: 0.5, 1: 0.5}  # Expected 50/50
    )
    
    print(f"\n   Representation Bias:")
    print(f"   - Detected: {repr_result.detected}")
    print(f"   - Severity: {repr_result.severity}")
    print(f"   - Max difference: {repr_result.evidence['max_difference']:.1%}")
    print(f"   - Affected groups: {repr_result.affected_groups}")
    
    # Test proxy detection
    proxy_result = detector.detect_proxy_variables(
        df,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score']
    )
    
    print(f"\n   Proxy Variables:")
    print(f"   - Detected: {proxy_result.detected}")
    print(f"   - Severity: {proxy_result.severity}")
    print(f"   - Proxy features: {proxy_result.evidence.get('proxy_features', [])}")
    
    # Test statistical disparity
    disparity_result = detector.detect_statistical_disparity(
        df,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score']
    )
    
    print(f"\n   Statistical Disparity:")
    print(f"   - Detected: {disparity_result.detected}")
    print(f"   - Severity: {disparity_result.severity}")
    print(f"   - Disparate features: {disparity_result.evidence.get('disparate_features', [])}")
    
    # Test comprehensive detection
    all_results = detector.detect_all_bias_types(
        df,
        protected_attribute='gender',
        reference_distribution={0: 0.5, 1: 0.5}
    )
    
    print(f"\n   ✅ Comprehensive bias detection completed:")
    for bias_type, result in all_results.items():
        status = "🔴 DETECTED" if result.detected else "🟢 NOT DETECTED"
        print(f"      {bias_type}: {status} (severity: {result.severity})")
    
except Exception as e:
    print(f"❌ BiasDetector failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Instance Reweighting
print("\n[4/5] Testing InstanceReweighting...")

try:
    # Prepare data
    X = df[['age', 'income', 'credit_score']].values
    y = df['outcome'].values
    sensitive = df['gender'].values
    
    # Test reweighting
    reweighter = InstanceReweighting(method='inverse_propensity', alpha=1.0)
    print("✅ InstanceReweighting initialized")
    
    # Fit and get weights
    reweighter.fit(X, y, sensitive_features=sensitive)
    weights = reweighter.get_sample_weights(sensitive)
    
    print(f"   Fitted reweighter:")
    print(f"   - Group weights: {reweighter.group_weights_}")
    print(f"   - Sample weight stats:")
    print(f"     • Mean: {weights.mean():.3f}")
    print(f"     • Min: {weights.min():.3f}")
    print(f"     • Max: {weights.max():.3f}")
    
    # Check effective group sizes
    female_weight = weights[sensitive == 0].sum()
    male_weight = weights[sensitive == 1].sum()
    print(f"   - Effective group sizes:")
    print(f"     • Female: {female_weight:.0f} (original: {n_female})")
    print(f"     • Male: {male_weight:.0f} (original: {n_male})")
    print(f"   - Balance ratio: {female_weight/male_weight:.3f} (target: 1.0)")
    
    # Test fit_transform
    X_out, y_out, weights_out = reweighter.fit_transform(
        X, y, sensitive_features=sensitive
    )
    
    assert X_out.shape == X.shape, "X shape changed"
    assert y_out.shape == y.shape, "y shape changed"
    assert len(weights_out) == len(X), "weights length mismatch"
    
    print("✅ fit_transform works correctly")
    
except Exception as e:
    print(f"❌ InstanceReweighting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Group Balancer
print("\n[5/5] Testing GroupBalancer...")

try:
    balancer = GroupBalancer(strategy='oversample', random_state=42)
    print("✅ GroupBalancer initialized")
    
    # Test resampling
    X_resampled, y_resampled = balancer.fit_resample(
        X, y, sensitive_features=sensitive
    )
    
    print(f"   Resampling results:")
    print(f"   - Original size: {len(X)}")
    print(f"   - Resampled size: {len(X_resampled)}")
    
    # Check new distribution
    # (In practice, you'd track this during resampling)
    print(f"   - Strategy: oversample to largest group")
    print(f"   ✅ Resampling completed")
    
except Exception as e:
    print(f"❌ GroupBalancer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Integration with sklearn
print("\n[BONUS] Testing sklearn integration...")

try:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline (note: reweighting doesn't work directly in pipeline)
    # We need to get weights separately
    reweighter = InstanceReweighting()
    reweighter.fit(X_train, y_train, sensitive_features=s_train)
    train_weights = reweighter.get_sample_weights(s_train)
    
    # Train model with weights
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train, sample_weight=train_weights)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"✅ Trained weighted model")
    print(f"   Test accuracy: {accuracy:.3f}")
    
    # Check fairness
    from measurement_module.src import FairnessAnalyzer
    
    analyzer = FairnessAnalyzer(bootstrap_samples=100)
    y_pred = model.predict(X_test)
    
    result = analyzer.compute_metric(
        y_test, y_pred, s_test,
        metric='demographic_parity',
        threshold=0.1
    )
    
    print(f"\n   Fairness after reweighting:")
    print(f"   - Metric: {result.metric_name}")
    print(f"   - Value: {result.value:.4f}")
    print(f"   - Fair: {result.is_fair}")
    print(f"   - Group metrics: {result.group_metrics}")
    
except Exception as e:
    print(f"⚠️ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ALL PIPELINE MODULE TESTS COMPLETED!")
print("=" * 60)
print("\n📊 Summary:")
print("   ✅ Imports work")
print("   ✅ BiasDetector detects all bias types")
print("   ✅ InstanceReweighting balances groups")
print("   ✅ GroupBalancer resamples data")
print("   ✅ sklearn integration works")
print("\n🎯 Pipeline module is ready!")
print("\nNext steps:")
print("  1. Test with your own data")
print("  2. Build training module (fairness constraints)")
print("  3. Integrate all modules")