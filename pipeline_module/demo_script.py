"""
Pipeline Module Demo Script

This script demonstrates the complete pipeline module functionality.
Convert to Jupyter notebook: jupytext --to notebook demo_script.py

Or run directly: python demo_script.py
"""

# %% [markdown]
# # Fairness Pipeline Module Demo
# 
# This notebook demonstrates:
# 1. Bias detection (representation, proxy, statistical)
# 2. Bias mitigation (reweighting)
# 3. Report generation
# 4. Integration with sklearn

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pipeline_module.src import BiasDetector, InstanceReweighting
from pipeline_module.src.bias_report import BiasReportGenerator
from pipeline_module.src.config_loader import load_config

print("✅ Imports successful")

# %% [markdown]
# ## 1. Generate Sample Dataset
# 
# We'll create a loan approval dataset with intentional bias:
# - Female applicants are under-represented (30% vs 50% expected)
# - Income is correlated with gender (proxy variable)
# - Different approval rates by gender

# %%
np.random.seed(42)

# Create biased dataset
n_female = 150
n_male = 350
n_total = n_female + n_male

data = {
    'age': np.concatenate([
        np.random.normal(35, 10, n_female),
        np.random.normal(38, 10, n_male)
    ]),
    'income': np.concatenate([
        np.random.normal(50000, 15000, n_female),
        np.random.normal(65000, 15000, n_male)  # Proxy!
    ]),
    'credit_score': np.concatenate([
        np.random.normal(680, 50, n_female),
        np.random.normal(700, 50, n_male)
    ]),
    'employment_years': np.concatenate([
        np.random.randint(1, 20, n_female),
        np.random.randint(1, 20, n_male)
    ]),
    'gender': [0] * n_female + [1] * n_male,  # 0=Female, 1=Male
}

df = pd.DataFrame(data)

# Add outcome (loan approval) with bias
df['loan_approved'] = 0
for idx in range(len(df)):
    score = (
        (df.loc[idx, 'income'] - 40000) / 50000 * 0.4 +
        (df.loc[idx, 'credit_score'] - 600) / 200 * 0.4 +
        df.loc[idx, 'gender'] * 0.15  # Bias toward males
    )
    score += np.random.normal(0, 0.1)
    df.loc[idx, 'loan_approved'] = 1 if score > 0.5 else 0

print(f"Dataset created: {len(df)} samples")
print(f"Female: {n_female} ({n_female/n_total:.1%})")
print(f"Male: {n_male} ({n_male/n_total:.1%})")
print(f"\nApproval rates:")
for gender in [0, 1]:
    rate = df[df['gender'] == gender]['loan_approved'].mean()
    name = "Female" if gender == 0 else "Male"
    print(f"  {name}: {rate:.1%}")

# %% [markdown]
# ## 2. Bias Detection
# 
# Run comprehensive bias detection to identify issues

# %%
print("\n" + "="*60)
print("BIAS DETECTION")
print("="*60)

detector = BiasDetector(
    representation_threshold=0.2,
    proxy_threshold=0.3,
    statistical_alpha=0.05
)

# Run all bias checks
results = detector.detect_all_bias_types(
    df,
    protected_attribute='gender',
    reference_distribution={0: 0.5, 1: 0.5},
    feature_columns=['age', 'income', 'credit_score', 'employment_years']
)

# Display results
for bias_type, result in results.items():
    print(f"\n{bias_type.upper()}")
    print(f"  Detected: {result.detected}")
    print(f"  Severity: {result.severity}")
    if result.detected:
        print(f"  Recommendations:")
        for rec in result.recommendations[:3]:
            print(f"    - {rec}")

# %% [markdown]
# ## 3. Generate Bias Report

# %%
reporter = BiasReportGenerator()

for name, result in results.items():
    reporter.add_result(name, result)

# Print summary
reporter.print_summary()

# Save reports (optional - uncomment to save)
# reporter.save_json('reports/bias_report.json')
# reporter.save_markdown('reports/bias_report.md')

# %% [markdown]
# ## 4. Prepare Data for Training

# %%
feature_cols = ['age', 'income', 'credit_score', 'employment_years']
X = df[feature_cols].values
y = df['loan_approved'].values
sensitive = df['gender'].values

# Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# %% [markdown]
# ## 5. Baseline Model (No Mitigation)

# %%
print("\n" + "="*60)
print("BASELINE MODEL (No Mitigation)")
print("="*60)

model_baseline = LogisticRegression(random_state=42, max_iter=1000)
model_baseline.fit(X_train, y_train)

y_pred_baseline = model_baseline.predict(X_test)
accuracy_baseline = (y_pred_baseline == y_test).mean()

print(f"Accuracy: {accuracy_baseline:.3f}")

# Calculate fairness metrics
baseline_rates = {}
for gender in [0, 1]:
    mask = s_test == gender
    rate = y_pred_baseline[mask].mean()
    baseline_rates[gender] = rate
    name = "Female" if gender == 0 else "Male"
    print(f"{name} positive rate: {rate:.3f}")

baseline_dp = abs(baseline_rates[0] - baseline_rates[1])
print(f"\nDemographic Parity Difference: {baseline_dp:.3f}")
print(f"Fair (threshold=0.1): {baseline_dp <= 0.1}")

# %% [markdown]
# ## 6. Model with Reweighting Mitigation

# %%
print("\n" + "="*60)
print("MODEL WITH REWEIGHTING")
print("="*60)

# Apply reweighting
reweighter = InstanceReweighting(method='inverse_propensity', alpha=1.0)
reweighter.fit(X_train, y_train, sensitive_features=s_train)
train_weights = reweighter.get_sample_weights(s_train)

print("Reweighting applied:")
print(f"  Group weights: {reweighter.group_weights_}")

# Check effective balance
female_weight = train_weights[s_train == 0].sum()
male_weight = train_weights[s_train == 1].sum()
print(f"  Effective group sizes:")
print(f"    Female: {female_weight:.0f}")
print(f"    Male: {male_weight:.0f}")
print(f"    Balance ratio: {female_weight/male_weight:.3f}")

# Train with weights
model_fair = LogisticRegression(random_state=42, max_iter=1000)
model_fair.fit(X_train, y_train, sample_weight=train_weights)

y_pred_fair = model_fair.predict(X_test)
accuracy_fair = (y_pred_fair == y_test).mean()

print(f"\nAccuracy: {accuracy_fair:.3f}")

# Calculate fairness metrics
fair_rates = {}
for gender in [0, 1]:
    mask = s_test == gender
    rate = y_pred_fair[mask].mean()
    fair_rates[gender] = rate
    name = "Female" if gender == 0 else "Male"
    print(f"{name} positive rate: {rate:.3f}")

fair_dp = abs(fair_rates[0] - fair_rates[1])
print(f"\nDemographic Parity Difference: {fair_dp:.3f}")
print(f"Fair (threshold=0.1): {fair_dp <= 0.1}")

# %% [markdown]
# ## 7. Comparison

# %%
print("\n" + "="*60)
print("COMPARISON: BASELINE VS REWEIGHTED")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Baseline', 'Reweighted'],
    'Accuracy': [accuracy_baseline, accuracy_fair],
    'DP Difference': [baseline_dp, fair_dp],
    'Fair (DP<0.1)': [baseline_dp <= 0.1, fair_dp <= 0.1]
})

print(comparison.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy comparison
axes[0].bar(['Baseline', 'Reweighted'], [accuracy_baseline, accuracy_fair])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_ylim(0, 1)

# Fairness comparison
axes[1].bar(['Baseline', 'Reweighted'], [baseline_dp, fair_dp])
axes[1].axhline(y=0.1, color='r', linestyle='--', label='Fairness Threshold')
axes[1].set_ylabel('Demographic Parity Difference')
axes[1].set_title('Fairness (Lower is Better)')
axes[1].legend()

plt.tight_layout()
# plt.savefig('reports/comparison.png')  # Uncomment to save
plt.show()

print("\n✅ Demo completed!")

# %% [markdown]
# ## Key Takeaways
# 
# 1. **Bias Detection**: Successfully identified representation bias, proxy variables, and statistical disparity
# 2. **Mitigation**: Instance reweighting balanced effective group sizes
# 3. **Trade-off**: Small accuracy change, significant fairness improvement
# 4. **Integration**: Works seamlessly with sklearn models
# 
# ## Next Steps
# 
# - Try different mitigation methods (resampling vs reweighting)
# - Experiment with alpha parameter (smoothing)
# - Integrate with measurement module for detailed fairness metrics
# - Add to CI/CD pipeline for automated fairness checks