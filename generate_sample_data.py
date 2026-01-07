"""
Generate sample loan approval dataset with realistic bias patterns.

This creates a synthetic dataset that demonstrates common fairness issues:
- Gender bias in loan approvals
- Correlation between protected attributes and financial features
- Realistic financial data distributions

Usage:
    python generate_sample_data.py
    python generate_sample_data.py --output data/custom_loans.csv --samples 500
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_loan_dataset(n_samples=200, seed=42, bias_strength=0.3):
    """
    Generate synthetic loan approval dataset with built-in bias.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        bias_strength: How much bias to introduce (0=none, 1=extreme)
    
    Returns:
        DataFrame with loan application data
    """
    np.random.seed(seed)
    
    print(f"🏗️  Generating {n_samples} loan applications...")
    
    # Protected attributes (binary for 48-hour scope)
    gender = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])  # 0=Female, 1=Male
    race = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])    # 0=Group_A, 1=Group_B
    
    # Age (correlated with income/experience)
    age = np.random.normal(35, 10, n_samples).clip(22, 65).astype(int)
    
    # Income (higher for older, with some gender gap to create bias)
    base_income = 30000 + (age - 22) * 2000
    gender_bonus = gender * 5000 * bias_strength  # Males tend to earn more (bias)
    noise = np.random.normal(0, 8000, n_samples)
    income = (base_income + gender_bonus + noise).clip(35000, 120000).astype(int)
    
    # Credit score (correlated with income and age)
    base_credit = 580 + (income / 1000) * 1.5 + (age - 22) * 2
    noise = np.random.normal(0, 20, n_samples)
    credit_score = (base_credit + noise).clip(600, 850).astype(int)
    
    # Employment years (correlated with age)
    employment_years = ((age - 22) * 0.6 + np.random.normal(0, 2, n_samples)).clip(1, 20).astype(int)
    
    # Debt to income ratio (lower is better)
    base_dti = 0.45 - (credit_score - 600) / 1000
    noise = np.random.normal(0, 0.05, n_samples)
    debt_to_income = (base_dti + noise).clip(0.15, 0.50).round(2)
    
    # Loan approval (this is where bias is introduced)
    # Base approval on legitimate factors
    approval_score = (
        (credit_score - 600) / 250 * 0.4 +          # Credit score (40%)
        (income - 35000) / 85000 * 0.3 +            # Income (30%)
        (1 - debt_to_income / 0.50) * 0.2 +         # DTI ratio (20%)
        (employment_years / 20) * 0.1               # Experience (10%)
    )
    
    # Add bias: males get a boost, females get a penalty
    bias_adjustment = (gender - 0.5) * bias_strength * 0.3
    approval_score += bias_adjustment
    
    # Add some randomness
    approval_score += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary approval (threshold at 0.5)
    loan_approved = (approval_score > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'gender': gender,
        'race': race,
        'loan_approved': loan_approved,
    })
    
    return df


def print_dataset_summary(df):
    """Print summary statistics about the generated dataset."""
    
    print("\n" + "=" * 60)
    print("📊 Dataset Summary")
    print("=" * 60)
    
    print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Columns:")
    for col in df.columns:
        print(f"   • {col}: {df[col].dtype}")
    
    print("\n👥 Protected Attributes:")
    print(f"   Gender distribution:")
    gender_counts = df['gender'].value_counts()
    print(f"     - Female (0): {gender_counts.get(0, 0)} ({gender_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"     - Male (1): {gender_counts.get(1, 0)} ({gender_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print(f"\n   Race distribution:")
    race_counts = df['race'].value_counts()
    print(f"     - Group_A (0): {race_counts.get(0, 0)} ({race_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"     - Group_B (1): {race_counts.get(1, 0)} ({race_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print("\n🎯 Target Variable:")
    approval_counts = df['loan_approved'].value_counts()
    print(f"   Loan approval distribution:")
    print(f"     - Denied (0): {approval_counts.get(0, 0)} ({approval_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"     - Approved (1): {approval_counts.get(1, 0)} ({approval_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print("\n⚖️ Fairness Analysis (Demographic Parity):")
    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        mask = df['gender'] == gender_val
        approval_rate = df[mask]['loan_approved'].mean()
        print(f"   {gender_name} approval rate: {approval_rate:.1%}")
    
    female_rate = df[df['gender'] == 0]['loan_approved'].mean()
    male_rate = df[df['gender'] == 1]['loan_approved'].mean()
    dp_diff = abs(male_rate - female_rate)
    
    print(f"\n   Demographic Parity Difference: {dp_diff:.3f}")
    if dp_diff > 0.1:
        print(f"   ⚠️  BIAS DETECTED: Difference exceeds 0.1 threshold")
    else:
        print(f"   ✅ Within acceptable threshold (< 0.1)")
    
    print("\n📈 Financial Features:")
    print(f"   Age: {df['age'].min()}-{df['age'].max()} (mean: {df['age'].mean():.1f})")
    print(f"   Income: ${df['income'].min():,}-${df['income'].max():,} (mean: ${df['income'].mean():,.0f})")
    print(f"   Credit Score: {df['credit_score'].min()}-{df['credit_score'].max()} (mean: {df['credit_score'].mean():.0f})")
    print(f"   Employment Years: {df['employment_years'].min()}-{df['employment_years'].max()} (mean: {df['employment_years'].mean():.1f})")
    print(f"   Debt-to-Income: {df['debt_to_income'].min():.2f}-{df['debt_to_income'].max():.2f} (mean: {df['debt_to_income'].mean():.2f})")


def main():
    parser = argparse.ArgumentParser(description="Generate sample loan dataset")
    parser.add_argument(
        '--output',
        type=str,
        default='data/sample_loan_data.csv',
        help=r"D:\Research\Turing\TuringProject\fairness_toolkit\data\data.csv"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=200,
        help="Number of samples to generate"
    )
    parser.add_argument(
        '--bias',
        type=float,
        default=0.3,
        help="Bias strength (0=none, 1=extreme)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    df = generate_loan_dataset(
        n_samples=args.samples,
        seed=args.seed,
        bias_strength=args.bias
    )
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    # Print summary
    print_dataset_summary(df)
    
    print("\n" + "=" * 60)
    print("🚀 Ready to test!")
    print("=" * 60)
    print("\nRun the test with:")
    print(f"  python test_with_real_dataset.py \\")
    print(f"    --data {output_path} \\")
    print(f"    --protected-attr gender \\")
    print(f"    --target loan_approved")


if __name__ == "__main__":
    main()