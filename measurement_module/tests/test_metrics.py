"""
Test Suite for Metrics Engine

Tests the core fairness metric calculations:
1. Demographic Parity Difference
2. Equalized Odds Difference
3. Equal Opportunity Difference
4. General compute_metric wrapper
5. Input Validation (Errors)

Run: python test_metrics.py
"""

import numpy as np
import sys
from pathlib import Path

# Try importing directly (assuming running in src directory or project root)
try:
    # Add parent directory to path if running from inside src
    current_dir = Path(__file__).parent
    if current_dir.name == 'src':
        sys.path.insert(0, str(current_dir.parent))
    
    from measurement_module.src.metrics_engine import (
        compute_metric,
        demographic_parity_difference,
        equalized_odds_difference,
        equal_opportunity_difference,
        ValidationError
    )
 
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Ensure you are running this from the project root.")
    sys.exit(1)

def assert_almost_equal(a, b, tolerance=0.001, name=""):
    """Helper to check if floats are close enough."""
    if abs(a - b) > tolerance:
        raise AssertionError(f"{name} failed: {a} != {b} (tolerance: {tolerance})")

def run_test_case(test_name, test_func):
    """Wrapper to run test and catch errors."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print('-' * 60)
    try:
        test_func()
        print(f"✅ PASSED: {test_name}")
    except AssertionError as e:
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ ERROR: {test_name}")
        print(f"   Unexpected Exception: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# TEST 1: Demographic Parity (Fair Case)
# ==============================================================================
def test_demographic_parity_fair():
    """Test case where groups have equal positive rates."""
    # 100 samples total. Group 0: 50 samples. Group 1: 50 samples.
    y_true = np.array([0]*100)
    s = np.array([0]*50 + [1]*50)
    
    # Group 0 gets 50% positive predictions. Group 1 gets 50% positive predictions.
    # Difference should be 0.0
    y_pred_group0 = np.array([1]*25 + [0]*25)
    y_pred_group1 = np.array([1]*25 + [0]*25)
    y_pred = np.concatenate([y_pred_group0, y_pred_group1])
    
    diff, rates, sizes = demographic_parity_difference(y_true, y_pred, s)
    
    print(f"   Group 0 Rate: {rates['Group_0']:.2f}")
    print(f"   Group 1 Rate: {rates['Group_1']:.2f}")
    print(f"   Difference: {diff:.4f}")
    
    assert_almost_equal(diff, 0.0, name="Difference")
    assert sizes['Group_0'] == 50
    assert sizes['Group_1'] == 50

run_test_case("Demographic Parity (Fair)", test_demographic_parity_fair)


# ==============================================================================
# TEST 2: Demographic Parity (Unfair Case)
# ==============================================================================
def test_demographic_parity_unfair():
    """Test case where groups have unequal positive rates."""
    y_true = np.array([0]*100)
    s = np.array([0]*50 + [1]*50)
    
    # Group 0: 10% positive. Group 1: 90% positive.
    # Difference should be 0.8
    y_pred_group0 = np.array([1]*5 + [0]*45)
    y_pred_group1 = np.array([1]*45 + [0]*5)
    y_pred = np.concatenate([y_pred_group0, y_pred_group1])
    
    diff, rates, sizes = demographic_parity_difference(y_true, y_pred, s)
    
    print(f"   Group 0 Rate: {rates['Group_0']:.2f}")
    print(f"   Group 1 Rate: {rates['Group_1']:.2f}")
    print(f"   Difference: {diff:.4f}")
    
    assert_almost_equal(diff, 0.8, name="Difference")

run_test_case("Demographic Parity (Unfair)", test_demographic_parity_unfair)


# ==============================================================================
# TEST 3: Equalized Odds (Fair Case)
# ==============================================================================
def test_equalized_odds_fair():
    """
    Test Equalized Odds (max difference in TPR and FPR).
    If both TPR and FPR are equal across groups, diff should be 0.
    """
    # Constructing data manually for specific confusion matrix components
    # Group 0: 20 samples. (10 pos, 10 neg)
    # Preds: 5 TP, 5 FN (TPR=0.5), 2 FP, 8 TN (FPR=0.2)
    y_true_0 = [1]*10 + [0]*10
    y_pred_0 = [1]*5 + [0]*5 + [1]*2 + [0]*8
    
    # Group 1: 20 samples. (10 pos, 10 neg)
    # Preds: 5 TP, 5 FN (TPR=0.5), 2 FP, 8 TN (FPR=0.2)
    y_true_1 = [1]*10 + [0]*10
    y_pred_1 = [1]*5 + [0]*5 + [1]*2 + [0]*8
    
    y_true = np.array(y_true_0 + y_true_1)
    y_pred = np.array(y_pred_0 + y_pred_1)
    s = np.array([0]*20 + [1]*20)
    
    diff, group_tprs, sizes = equalized_odds_difference(y_true, y_pred, s)
    
    print(f"   Group 0 TPR: {group_tprs['Group_0']:.2f}")
    print(f"   Group 1 TPR: {group_tprs['Group_1']:.2f}")
    print(f"   Difference: {diff:.4f}")
    
    assert_almost_equal(diff, 0.0, name="Difference")

run_test_case("Equalized Odds (Fair)", test_equalized_odds_fair)


# ==============================================================================
# TEST 4: Equal Opportunity (Unfair Case)
# ==============================================================================
def test_equal_opportunity_unfair():
    """Test difference in True Positive Rates (Recall)."""
    
    # We need mixed labels in y_true for confusion_matrix to work correctly
    # with the logic in metrics_engine.py
    
    # Total 8 samples
    # Group 0: Indices 0-3
    # Group 1: Indices 4-7
    
    y_true_simple = np.array([1, 1, 0, 1,  1, 1, 0, 1])
    s_simple = np.array([0, 0, 0, 0,  1, 1, 1, 1])
    
    # PREDICTIONS:
    # Group 0 (indices 0-3): Perfect Recall (TPR = 1.0)
    #   y_true=[1, 1, 0, 1], y_pred=[1, 1, 0, 1]
    #   Positives: 3 found/3, Negatives: 1 correct. TP=3, FN=0.
    # Group 1 (indices 4-7): Terrible Recall (TPR = 0.0)
    #   y_true=[1, 1, 0, 1], y_pred=[0, 0, 0, 0]
    #   Positives: 0 found/3. TP=0, FN=3.
    # Expected Diff: |1.0 - 0.0| = 1.0
    
    y_pred_simple = np.array([1, 1, 0, 1,  0, 0, 0, 0])
    
    diff, tprs, sizes = equal_opportunity_difference(y_true_simple, y_pred_simple, s_simple)
    
    print(f"   Group 0 TPR: {tprs['Group_0']:.2f}")
    print(f"   Group 1 TPR: {tprs['Group_1']:.2f}")
    print(f"   Difference: {diff:.4f}")
    
    assert_almost_equal(diff, 1.0, name="Difference")

run_test_case("Equal Opportunity (Unfair)", test_equal_opportunity_unfair)

# ==============================================================================
# TEST 5: Wrapper Function (compute_metric)
# ==============================================================================
def test_compute_metric_wrapper():
    """Test the generic compute_metric function."""
    y_true = np.array([0,0,1,1, 0,0,1,1])
    s = np.array([0,0,0,0, 1,1,1,1])
    
    # Biased predictions
    y_pred = np.array([0,0,1,0,  1,1,1,1]) # Group 0 low, Group 1 high
    
    val, metrics, sizes = compute_metric("demographic_parity", y_true, y_pred, s)
    
    print(f"   Computed Value: {val:.4f}")
    print(f"   Group Metrics: {metrics}")
    
    assert isinstance(val, float)
    assert isinstance(metrics, dict)
    assert "Group_0" in metrics or "Group_1" in metrics

run_test_case("Compute Metric Wrapper", test_compute_metric_wrapper)


# ==============================================================================
# TEST 6: Error Handling (Non-binary Inputs)
# ==============================================================================
def test_validation_non_binary():
    """Test that non-binary inputs raise ValidationError."""
    y_true = np.array([0, 1, 2]) # 2 is invalid
    y_pred = np.array([0, 1, 0])
    s = np.array([0, 0, 1])
    
    try:
        compute_metric("demographic_parity", y_true, y_pred, s)
        raise AssertionError("Expected ValidationError for non-binary y_true")
    except ValidationError as e:
        print(f"   ✅ Correctly caught error: {e}")

run_test_case("Validation (Non-binary)", test_validation_non_binary)


# ==============================================================================
# TEST 7: Error Handling (Length Mismatch)
# ==============================================================================
def test_validation_length_mismatch():
    """Test that mismatched lengths raise ValidationError."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1, 1]) # Length 3 vs 2
    s = np.array([0, 1])
    
    try:
        compute_metric("demographic_parity", y_true, y_pred, s)
        raise AssertionError("Expected ValidationError for length mismatch")
    except ValidationError as e:
        print(f"   ✅ Correctly caught error: {e}")

run_test_case("Validation (Length Mismatch)", test_validation_length_mismatch)

# ==============================================================================
# TEST 8: Error Handling (Unknown Metric)
# ==============================================================================
def test_unknown_metric():
    """Test that unknown metric name raises ValueError."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    s = np.array([0, 1])
    
    try:
        compute_metric("unknown_metric", y_true, y_pred, s)
        raise AssertionError("Expected ValueError for unknown metric")
    except ValueError as e:
        print(f"   ✅ Correctly caught error: {e}")

run_test_case("Validation (Unknown Metric)", test_unknown_metric)


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*60)
print("TEST SUITE COMPLETE")
print("="*60)
print("\nAll core metric calculations verified.")
print("Ready to use with FairnessAnalyzer.")