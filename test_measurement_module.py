"""
Simplified test for measurement module.
Tests only what actually exists in your codebase.

Run: python test_measurement_module.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 60)
print("Testing Measurement Module (Simple)")
print("=" * 60)

# Test 1: Import check
print("\n[1/4] Testing imports...")
try:
    # Prefer explicit import to avoid package resolution issues when running
    # this script directly from the repository root.
    from measurement_module.src.fairness_analyzer_simple import FairnessAnalyzer
    print("✅ FairnessAnalyzer imported (explicit) successfully")
except ImportError:
    try:
        from measurement_module.src import FairnessAnalyzer
        print("✅ FairnessAnalyzer imported from package successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nCurrent issue: Cannot import FairnessAnalyzer")
        print("\nVerify:")
        print("  1. File exists: measurement_module/src/fairness_analyzer_simple.py")
        print("  2. measurement_module/src/__init__.py exports FairnessAnalyzer")
        sys.exit(1)

# Test 2: Generate test data
print("\n[2/4] Generating test data...")
np.random.seed(42)

n_samples = 500
n_group_0 = 250
n_group_1 = 250

# Create biased data (group 1 approved more often)
sensitive_features = np.array([0] * n_group_0 + [1] * n_group_1)
y_true = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])

# Add bias: group 1 gets higher approval rate
y_pred = y_true.copy()
for i in range(n_samples):
    if sensitive_features[i] == 1:  # Group 1
        if np.random.random() < 0.15:
            y_pred[i] = 1
    else:  # Group 0
        if np.random.random() < 0.15 and y_pred[i] == 1:
            y_pred[i] = 0

approval_rate_0 = y_pred[sensitive_features == 0].mean()
approval_rate_1 = y_pred[sensitive_features == 1].mean()

print(f"✅ Generated {n_samples} samples")
print(f"   Group 0 size: {n_group_0}, approval rate: {approval_rate_0:.2%}")
print(f"   Group 1 size: {n_group_1}, approval rate: {approval_rate_1:.2%}")
print(f"   Disparity: {abs(approval_rate_1 - approval_rate_0):.2%}")

# Test 3: Initialize FairnessAnalyzer
print("\n[3/4] Testing FairnessAnalyzer initialization...")
try:
    analyzer = FairnessAnalyzer()
    print("✅ FairnessAnalyzer initialized successfully")
    print(f"   Analyzer type: {type(analyzer)}")
    print(f"   Available methods: {[m for m in dir(analyzer) if not m.startswith('_')]}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try to compute metrics (adapt based on your API)
print("\n[4/4] Testing metric computation...")
try:
    # Try different possible method names
    method_attempts = [
        ('compute_metric', lambda: analyzer.compute_metric(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            metric='demographic_parity'
        )),
        ('compute_all_metrics', lambda: analyzer.compute_all_metrics(
            y_true, y_pred, sensitive_features
        )),
        ('analyze', lambda: analyzer.analyze(
            y_true, y_pred, sensitive_features
        )),
        ('evaluate', lambda: analyzer.evaluate(
            y_true, y_pred, sensitive_features
        )),
    ]
    
    success = False
    for method_name, method_call in method_attempts:
        if hasattr(analyzer, method_name):
            print(f"\n   Trying method: {method_name}()")
            try:
                result = method_call()
                print(f"✅ {method_name}() worked!")
                print(f"   Result type: {type(result)}")
                
                # Try to display result
                if isinstance(result, dict):
                    print(f"   Result keys: {list(result.keys())}")
                    for key, value in list(result.items())[:3]:  # Show first 3 items
                        print(f"     {key}: {value}")
                elif hasattr(result, '__dict__'):
                    print(f"   Result attributes: {list(result.__dict__.keys())}")
                else:
                    print(f"   Result: {result}")
                
                success = True
                break
            except TypeError as e:
                print(f"   ⚠️  Method exists but signature different: {e}")
                print(f"   Try checking the method signature in fairness_analyzer_simple.py")
            except Exception as e:
                print(f"   ❌ Method failed: {e}")
                import traceback
                traceback.print_exc()
    
    if not success:
        print("\n❌ Could not find working method to compute metrics")
        print("\nPlease check your FairnessAnalyzer class and share:")
        print("  1. The method names available")
        print("  2. The method signatures (parameters)")
        print("\nYou can find this in: measurement_module/src/fairness_analyzer_simple.py")
        
except Exception as e:
    print(f"❌ Metric computation test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ BASIC TESTS COMPLETED!")
print("=" * 60)
print("\nWhat works:")
print("  ✅ Imports")
print("  ✅ Data generation")
print("  ✅ FairnessAnalyzer initialization")
print("\nNext: Share your fairness_analyzer_simple.py code so we can")
print("      create a complete test that matches your actual API")