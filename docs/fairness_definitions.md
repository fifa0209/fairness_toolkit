# Fairness Definitions

**A comprehensive guide to fairness metrics in the Fairness Pipeline Development Toolkit**

---

## Overview

This document explains the fairness metrics implemented in the toolkit, their mathematical definitions, when to use each metric, and their practical implications.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Classification Fairness Metrics](#classification-fairness-metrics)
3. [Regression Fairness Metrics](#regression-fairness-metrics)
4. [Metric Selection Guide](#metric-selection-guide)
5. [Trade-offs and Limitations](#trade-offs-and-limitations)
6. [Real-World Examples](#real-world-examples)

---

## Core Concepts

### Protected Attributes

**Definition**: Characteristics that identify demographic groups protected by law or policy (e.g., race, gender, age, disability status).

**Notation**: We use `A` to denote a protected attribute, where:
- `A = 0`: Reference group (e.g., male, majority race)
- `A = 1`: Protected group (e.g., female, minority race)

### Predictions and Outcomes

- **Y**: True outcome (actual label)
- **Ŷ**: Predicted outcome (model prediction)
- **Ŝ**: Predicted score (probability or continuous value)

### Group Fairness vs. Individual Fairness

**Group Fairness**: Ensures statistical parity across demographic groups.
- Focus: Aggregate statistics
- Example: Equal approval rates across gender groups

**Individual Fairness**: Ensures similar individuals receive similar treatment.
- Focus: Individual-level similarity
- Example: Two applicants with identical qualifications should receive similar scores

**This toolkit focuses on group fairness metrics**, as they are:
1. More tractable to measure
2. Easier to operationalize
3. Aligned with legal frameworks (disparate impact doctrine)

---

## Classification Fairness Metrics

### 1. Demographic Parity (Statistical Parity)

#### Definition

Groups should receive positive predictions at equal rates, regardless of protected attribute membership.

#### Mathematical Formulation

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

Or equivalently, the **demographic parity difference**:

```
DP_diff = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|
```

#### Interpretation

- **DP_diff = 0**: Perfect parity (equal selection rates)
- **DP_diff > 0**: Disparity exists (one group selected more often)
- **Threshold**: Typically DP_diff ≤ 0.10 (10%) is considered acceptable

#### When to Use

âœ… **Use demographic parity when:**
- The goal is equal opportunity regardless of qualifications
- Historical discrimination has created unequal starting points
- You want to ensure representation (e.g., hiring, admissions)
- True outcomes are unavailable or unreliable

❌ **Don't use demographic parity when:**
- Groups have legitimately different base rates (e.g., disease prevalence)
- Accuracy is critical (DP can reduce overall model performance)
- Legal requirements mandate equal treatment based on merit

#### Example: Loan Approval

```
Group 0 (Male):   Approval rate = 60%
Group 1 (Female): Approval rate = 45%

DP_diff = |0.60 - 0.45| = 0.15 (15% disparity)
Status: UNFAIR (exceeds 10% threshold)
```

#### Code Example

```python
from measurement_module import FairnessAnalyzer

analyzer = FairnessAnalyzer()
result = analyzer.compute_metric(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=gender,
    metric='demographic_parity',
    threshold=0.10
)

print(f"DP Difference: {result.value:.3f}")
print(f"Fair: {result.is_fair}")
```

---

### 2. Equalized Odds

#### Definition

Both true positive rates (TPR) and false positive rates (FPR) should be equal across groups.

#### Mathematical Formulation

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)  [TPR equality]
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)  [FPR equality]
```

The **equalized odds difference** is:

```
EO_diff = max(|TPR₀ - TPR₁|, |FPR₀ - FPR₁|)
```

#### Interpretation

- **EO_diff = 0**: Perfect equalized odds
- **EO_diff > 0**: Disparity in error rates across groups
- **Threshold**: Typically EO_diff ≤ 0.10 is acceptable

#### When to Use

âœ… **Use equalized odds when:**
- You have reliable ground truth labels
- Both types of errors (false positives and false negatives) matter equally
- You want to ensure equal quality of service across groups
- Accuracy is important but fairness is also critical

❌ **Don't use equalized odds when:**
- Ground truth is unavailable or unreliable
- One error type is much more costly than the other
- Base rates differ substantially across groups

#### Example: Credit Default Prediction

```
           Group 0 (Male)  Group 1 (Female)
TPR        0.75            0.65
FPR        0.20            0.30

TPR diff = |0.75 - 0.65| = 0.10
FPR diff = |0.20 - 0.30| = 0.10

EO_diff = max(0.10, 0.10) = 0.10
Status: BORDERLINE (at threshold)
```

#### Code Example

```python
result = analyzer.compute_metric(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=gender,
    metric='equalized_odds',
    threshold=0.10
)

print(f"TPR Group 0: {result.group_metrics['TPR_0']:.3f}")
print(f"TPR Group 1: {result.group_metrics['TPR_1']:.3f}")
print(f"FPR Group 0: {result.group_metrics['FPR_0']:.3f}")
print(f"FPR Group 1: {result.group_metrics['FPR_1']:.3f}")
```

---

### 3. Equal Opportunity

#### Definition

True positive rates (TPR) should be equal across groups. This is a **relaxed version of equalized odds** that only considers positive outcomes.

#### Mathematical Formulation

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
```

The **equal opportunity difference** is:

```
EOpp_diff = |TPR₀ - TPR₁|
```

#### Interpretation

- **EOpp_diff = 0**: Perfect equal opportunity
- **EOpp_diff > 0**: Disparity in benefit allocation
- **Threshold**: Typically EOpp_diff ≤ 0.10

#### When to Use

âœ… **Use equal opportunity when:**
- False negatives are more costly than false positives
- The focus is on equal access to positive outcomes
- You want to ensure qualified individuals have equal chances
- Ground truth for positive cases is reliable

❌ **Don't use equal opportunity when:**
- False positives are equally or more costly
- You need comprehensive error rate parity

#### Example: Medical Diagnosis

```
Disease detection (Y=1 is sick, Ŷ=1 is diagnosed)

           Group 0  Group 1
TPR        0.90     0.75

EOpp_diff = |0.90 - 0.75| = 0.15
Status: UNFAIR (Group 1 has 15% lower detection rate)
```

This means Group 1 patients with the disease are less likely to be correctly diagnosed—a serious fairness violation.

---

## Regression Fairness Metrics

### Mean Absolute Error (MAE) Parity

#### Definition

The average prediction error should be equal across groups.

#### Mathematical Formulation

```
MAE_diff = |MAE₀ - MAE₁|

where MAE_g = E[|Y - Ŷ| | A = g]
```

#### When to Use

âœ… **Use MAE parity for regression when:**
- Predictions are continuous (e.g., salary, loan amount)
- You want equal prediction quality across groups
- Symmetric errors (over/under prediction) are equally bad

#### Example: Salary Prediction

```
           Group 0  Group 1
MAE        $5,000   $8,500

MAE_diff = |$5,000 - $8,500| = $3,500
Status: Prediction quality is worse for Group 1
```

---

## Metric Selection Guide

### Decision Tree

```
START: What is your ML task?
│
├─ CLASSIFICATION
│  │
│  ├─ Do you have reliable ground truth?
│  │  │
│  │  ├─ NO → Use Demographic Parity
│  │  │
│  │  └─ YES → What matters most?
│  │     │
│  │     ├─ Equal selection rates → Demographic Parity
│  │     │
│  │     ├─ Equal error rates → Equalized Odds
│  │     │
│  │     └─ Equal benefit access → Equal Opportunity
│  │
│  └─ Are base rates equal across groups?
│     │
│     ├─ YES → Any metric works
│     │
│     └─ NO → Avoid Demographic Parity
│
└─ REGRESSION → Use MAE Parity (or RMSE Parity)
```

### Comparison Table

| Metric | Requires Labels? | Handles Different Base Rates? | Focus | Use Case |
|--------|------------------|-------------------------------|-------|----------|
| **Demographic Parity** | No | No | Selection rates | Hiring, admissions |
| **Equalized Odds** | Yes | Yes | Error rates | Credit scoring, fraud |
| **Equal Opportunity** | Yes | Yes | Benefit access | Medical diagnosis |
| **MAE Parity** | Yes | Yes | Prediction quality | Salary, pricing |

---

## Trade-offs and Limitations

### The Impossibility Theorem

**Key Result**: It is mathematically impossible to satisfy demographic parity, equalized odds, and perfect accuracy simultaneously when base rates differ across groups.

**Implication**: You must choose which fairness criterion to prioritize based on your domain and values.

### Accuracy vs. Fairness Trade-off

Enforcing fairness constraints typically reduces overall accuracy:

```
Baseline Model:  Accuracy = 85%, DP_diff = 0.20
Fair Model:      Accuracy = 82%, DP_diff = 0.08

Trade-off: -3% accuracy for +12% fairness improvement
```

**Question to ask**: Is this trade-off acceptable for your application?

### Group Size Considerations

All metrics become unreliable with small group sizes:

- **Minimum group size**: n ≥ 30 (statistical validity)
- **Recommended size**: n ≥ 100 (stable estimates)
- **Small groups**: Use confidence intervals, interpret cautiously

### Intersectionality

Metrics computed for single protected attributes may hide disparities in intersectional groups:

```
Gender alone:   DP_diff = 0.08 (Fair)
Race alone:     DP_diff = 0.09 (Fair)

Black women:    DP_diff = 0.18 (Unfair!)
```

**Current toolkit limitation**: Binary protected attributes only. Intersectional analysis is documented but not fully implemented.

---

## Real-World Examples

### Example 1: Hiring Algorithm

**Context**: A company uses ML to screen job applications.

**Metric Choice**: Demographic Parity
- **Why**: Equal opportunity hiring is the goal
- **Threshold**: 10% (aligned with four-fifths rule)

**Before mitigation**:
```
Male candidates:   Selection rate = 65%
Female candidates: Selection rate = 48%
DP_diff = 0.17 (UNFAIR)
```

**After mitigation** (reweighting + constraints):
```
Male candidates:   Selection rate = 58%
Female candidates: Selection rate = 52%
DP_diff = 0.06 (FAIR)
```

---

### Example 2: Loan Default Prediction

**Context**: Bank predicts which loans will default.

**Metric Choice**: Equalized Odds
- **Why**: Both false positives (denied good applicants) and false negatives (approved bad applicants) are costly
- **Ground truth**: Default data available

**Before mitigation**:
```
           White    Black
TPR        0.78     0.65
FPR        0.15     0.25
EO_diff = max(0.13, 0.10) = 0.13 (UNFAIR)
```

**After mitigation**:
```
           White    Black
TPR        0.75     0.72
FPR        0.18     0.20
EO_diff = max(0.03, 0.02) = 0.03 (FAIR)
```

---

### Example 3: Medical Risk Scoring

**Context**: Predict which patients need preventive intervention.

**Metric Choice**: Equal Opportunity
- **Why**: Missing high-risk patients (false negatives) is much worse than over-screening (false positives)
- **Focus**: Equal sensitivity across racial groups

**Before mitigation**:
```
           White    Hispanic
TPR        0.88     0.73
EOpp_diff = 0.15 (UNFAIR)
```

**After calibration**:
```
           White    Hispanic
TPR        0.85     0.83
EOpp_diff = 0.02 (FAIR)
```

---

## References

### Academic Sources

1. **Hardt, M., Price, E., & Srebro, N. (2016)**. "Equality of Opportunity in Supervised Learning." *NeurIPS*.
   - Defines equalized odds and equal opportunity

2. **Chouldechova, A. (2017)**. "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." *Big Data*.
   - Discusses impossibility results

3. **Corbett-Davies, S., & Goel, S. (2018)**. "The Measure and Mismeasure of Fairness: A Critical Review of Fair Machine Learning." *arXiv*.
   - Comprehensive survey of fairness definitions

### Legal Framework

- **Four-Fifths Rule** (U.S. Equal Employment Opportunity Commission): Selection rate for any group should be at least 80% of the rate for the highest-selected group
  - Translates to: DP_diff ≤ 0.20

### Implementation Resources

- **Fairlearn**: https://fairlearn.org/
- **IBM AIF360**: https://aif360.mybluemix.net/
- **Aequitas**: http://aequitas.dssg.io/

---

## FAQ

**Q: Can I achieve perfect fairness without sacrificing accuracy?**

A: Generally no, especially when base rates differ across groups. There's typically a fairness-accuracy trade-off that must be explicitly managed.

**Q: Which metric should I use?**

A: It depends on your domain, stakeholders, and what type of fairness matters most. Use the decision tree in the Metric Selection Guide.

**Q: What if my groups have different base rates?**

A: Avoid demographic parity. Use equalized odds or equal opportunity, which account for different prevalence rates.

**Q: How do I set the fairness threshold?**

A: Common thresholds:
- 0.10 (10%): Standard in practice
- 0.20 (20%): Legal four-fifths rule
- Domain-specific: Consult stakeholders and legal counsel

**Q: What about intersectional fairness?**

A: Current toolkit limitation. Analyze single attributes first, then manually examine key intersections (e.g., race × gender) using the same metrics.

---

## Glossary

- **Base Rate**: Proportion of positive outcomes in a group
- **Disparate Impact**: Disproportionate adverse effect on a protected group
- **False Negative Rate (FNR)**: P(Ŷ = 0 | Y = 1)
- **False Positive Rate (FPR)**: P(Ŷ = 1 | Y = 0)
- **Group Fairness**: Fairness defined by aggregate statistics across groups
- **Protected Attribute**: Characteristic defining legally protected groups
- **True Negative Rate (TNR)**: P(Ŷ = 0 | Y = 0) = 1 - FPR
- **True Positive Rate (TPR)**: P(Ŷ = 1 | Y = 1) = Sensitivity = Recall

---

*Last updated: January 2026*