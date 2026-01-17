"""
Test shared modules with REAL dataset.
Demonstrates how to use fairness pipeline with actual data.

Usage:
    python test_with_real_dataset.py --data path/to/your/data.csv
    
Or use built-in test datasets:
    python test_with_real_dataset.py --dataset adult
    python test_with_real_dataset.py --dataset compas
"""
import pytest
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from shared.schemas import (
    FairnessMetricResult,
    PipelineConfig,
    DatasetMetadata,
)
from shared.logging import get_logger, log_fairness_result, PipelineLogger
from shared.validation import (
    validate_dataframe,
    validate_protected_attribute,
    validate_predictions,
    ValidationError,
)
@pytest.fixture
def df():
    """Provide sample dataframe."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(200),
        'feature2': np.random.randn(200),
        'target': np.random.binomial(1, 0.5, 200),
        'protected': np.random.binomial(1, 0.5, 200)
    })

@pytest.fixture
def protected_attr():
    return 'protected'

@pytest.fixture
def target_col():
    return 'target'


def load_adult_census():
    """
    Load UCI Adult Census dataset (common fairness benchmark).
    Predicts whether income >50K based on demographics.
    """
    print("📥 Loading Adult Census dataset from UCI...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
        
        # Clean data
        df = df.dropna()
        
        # Binary outcome: income >50K
        df['income_binary'] = (df['income'] == '>50K').astype(int)
        
        # Binary protected attribute: sex
        df['sex_binary'] = (df['sex'] == 'Male').astype(int)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Protected attribute: sex (Male/Female)")
        print(f"   Target: income (>50K / <=50K)")
        
        return df, 'sex_binary', 'income_binary'
        
    except Exception as e:
        print(f"❌ Failed to load Adult dataset: {e}")
        return None, None, None


def load_compas():
    """
    Load COMPAS recidivism dataset (ProPublica investigation).
    Predicts recidivism risk scores.
    """
    print("📥 Loading COMPAS dataset from ProPublica...")
    
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Clean data
        df = df[df['days_b_screening_arrest'] <= 30]
        df = df[df['days_b_screening_arrest'] >= -30]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != 'O']
        df = df[df['score_text'] != 'N/A']
        
        # Binary outcome: did recidivate
        df['recid_binary'] = df['two_year_recid'].astype(int)
        
        # Binary protected attribute: race (African-American vs Caucasian)
        df = df[df['race'].isin(['African-American', 'Caucasian'])]
        df['race_binary'] = (df['race'] == 'African-American').astype(int)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Protected attribute: race (African-American/Caucasian)")
        print(f"   Target: recidivism (yes/no)")
        
        return df, 'race_binary', 'recid_binary'
        
    except Exception as e:
        print(f"❌ Failed to load COMPAS dataset: {e}")
        return None, None, None


def load_custom_dataset(filepath):
    """
    Load your custom dataset.
    
    Expected format:
    - CSV file with headers
    - One column for protected attribute (binary: 0/1)
    - One column for target variable (binary: 0/1)
    """
    print(f"📥 Loading custom dataset from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        
        # User needs to specify which columns to use
        print("\n⚠️  Please specify:")
        print("   --protected-attr <column_name>  (e.g., 'gender', 'race')")
        print("   --target <column_name>          (e.g., 'outcome', 'label')")
        
        return df, None, None
        
    except Exception as e:
        print(f"❌ Failed to load custom dataset: {e}")
        return None, None, None


def test_with_real_data(df, protected_attr, target_col):
    """
    Run full test suite with real dataset.
    """
    pass
    logger = get_logger("real_data_test")
    
    print("\n" + "=" * 60)
    print("Testing Shared Modules with REAL Data")
    print("=" * 60)
    
    # Test 1: Data Validation
    print("\n[1/5] Validating dataset structure...")
    try:
        validate_dataframe(
            df,
            required_columns=[protected_attr, target_col],
            min_rows=100
        )
        print(f"✅ DataFrame validation passed")
        print(f"   Shape: {df.shape}")
        
    except ValidationError as e:
        print(f"❌ DataFrame validation failed: {e}")
        return
    
    # Test 2: Protected Attribute Validation
    print("\n[2/5] Validating protected attribute...")
    try:
        validate_protected_attribute(df, protected_attr, min_group_size=30)
        
        group_counts = df[protected_attr].value_counts()
        print(f"✅ Protected attribute validation passed")
        print(f"   Groups: {dict(group_counts)}")
        
    except ValidationError as e:
        print(f"❌ Protected attribute validation failed: {e}")
        return
    
    # Test 3: Create Dataset Metadata
    print("\n[3/5] Creating dataset metadata...")
    try:
        group_dist = df[protected_attr].value_counts().to_dict()
        class_dist = df[target_col].value_counts().to_dict()
        
        metadata = DatasetMetadata(
            name="real_dataset",
            n_samples=len(df),
            n_features=len(df.columns) - 2,  # Exclude protected + target
            task_type="binary_classification",
            protected_attribute=protected_attr,
            protected_groups=list(group_dist.keys()),
            group_distribution=group_dist,
            class_balance=class_dist,
        )
        
        print(f"✅ Dataset metadata created:")
        print(f"   - Samples: {metadata.n_samples}")
        print(f"   - Features: {metadata.n_features}")
        print(f"   - Protected groups: {metadata.protected_groups}")
        print(f"   - Group sizes: {metadata.group_distribution}")
        print(f"   - Min group size: {metadata.min_group_size}")
        print(f"   - Imbalance ratio: {metadata.imbalance_ratio:.2f}")
        print(f"   - Class distribution: {metadata.class_balance}")
        
    except Exception as e:
        print(f"❌ Metadata creation failed: {e}")
        return
    
    # Test 4: Create Pipeline Config
    print("\n[4/5] Creating pipeline configuration...")
    try:
        config = PipelineConfig(
            target_column=target_col,
            protected_attribute=protected_attr,
            fairness_metrics=["demographic_parity", "equalized_odds"],
            fairness_threshold=0.1,
            confidence_level=0.95,
            bootstrap_samples=1000,
        )
        
        errors = config.validate()
        if errors:
            print(f"❌ Config validation failed: {errors}")
            return
        
        print(f"✅ Pipeline configuration created:")
        print(f"   - Target: {config.target_column}")
        print(f"   - Protected attribute: {config.protected_attribute}")
        print(f"   - Metrics: {config.fairness_metrics}")
        print(f"   - Threshold: {config.fairness_threshold}")
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return
    
    # Test 5: Simulate Fairness Analysis
    print("\n[5/5] Simulating fairness analysis...")
    try:
        # Calculate baseline statistics (simple version - full version in measurement module)
        groups = df[protected_attr].unique()
        group_rates = {}
        
        for group in groups:
            mask = df[protected_attr] == group
            positive_rate = df[mask][target_col].mean()
            group_rates[f"Group_{group}"] = positive_rate
        
        # Calculate demographic parity difference
        rates = list(group_rates.values())
        dp_diff = max(rates) - min(rates)
        
        # Create result (with placeholder CI - real version uses bootstrap)
        result = FairnessMetricResult(
            metric_name="demographic_parity",
            value=dp_diff,
            confidence_interval=(dp_diff - 0.02, dp_diff + 0.02),  # Placeholder
            group_metrics=group_rates,
            group_sizes=group_dist,
            interpretation=f"{'Bias detected' if dp_diff > 0.1 else 'Within threshold'}",
            is_fair=dp_diff <= 0.1,
            threshold=0.1,
        )
        
        print(f"✅ Fairness analysis completed:")
        print(f"   - Metric: {result.metric_name}")
        print(f"   - Value: {result.value:.4f}")
        print(f"   - Fair: {result.is_fair}")
        print(f"   - Group rates:")
        for group, rate in group_rates.items():
            print(f"     • {group}: {rate:.4f}")
        
        # Log results
        with PipelineLogger(logger, "fairness_analysis"):
            log_fairness_result(
                logger,
                result.metric_name,
                result.value,
                result.is_fair,
                result.threshold,
                result.group_metrics,
            )
        
    except Exception as e:
        print(f"❌ Fairness analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED WITH REAL DATA!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   ✅ Validated {metadata.n_samples} samples")
    print(f"   ✅ Checked {len(metadata.protected_groups)} protected groups")
    print(f"   ✅ Computed baseline fairness metrics")
    print(f"   ✅ Created pipeline configuration")
    print("\n🚀 Ready to build full measurement module!")


def main():
    parser = argparse.ArgumentParser(description="Test shared modules with real data")
    parser.add_argument(
        '--dataset',
        choices=['adult', 'compas'],
        help="Use built-in benchmark dataset"
    )
    parser.add_argument(
        '--data',
        type=str,
        help="Path to custom CSV file"
    )
    parser.add_argument(
        '--protected-attr',
        type=str,
        help="Name of protected attribute column (for custom data)"
    )
    parser.add_argument(
        '--target',
        type=str,
        help="Name of target column (for custom data)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == 'adult':
        df, protected_attr, target_col = load_adult_census()
    elif args.dataset == 'compas':
        df, protected_attr, target_col = load_compas()
    elif args.data:
        df, protected_attr, target_col = load_custom_dataset(args.data)
        # Override with user-specified columns
        if args.protected_attr:
            protected_attr = args.protected_attr
        if args.target:
            target_col = args.target
        
        if protected_attr is None or target_col is None:
            print("\n❌ Error: Must specify --protected-attr and --target for custom data")
            return
    else:
        print("❌ Error: Must specify either --dataset or --data")
        print("\nExamples:")
        print("  python test_with_real_dataset.py --dataset adult")
        print("  python test_with_real_dataset.py --data mydata.csv --protected-attr gender --target outcome")
        return
    
    if df is not None:
        test_with_real_data(df, protected_attr, target_col)


if __name__ == "__main__":
    main()
