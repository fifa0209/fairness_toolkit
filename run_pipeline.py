"""
Fairness Pipeline Orchestrator

Executes the complete fairness pipeline:
1. Measure baseline fairness
2. Detect bias in data
3. Apply mitigation
4. Train fair model
5. Validate fairness
6. Setup monitoring

Usage:
    python run_pipeline.py --config config.yml
    python run_pipeline.py --config config.yml --data data/sample_loan_data.csv
"""

import sys
import platform

# ============================================================================
# Windows Encoding Fix
# ============================================================================
if platform.system() == 'Windows':
    try:
        if sys.version_info >= (3, 7):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Simple ASCII-safe symbols for Windows compatibility
CHECK = "[OK]   " if platform.system() == 'Windows' else "✓"
CROSS = "[FAIL] " if platform.system() == 'Windows' else "✗"
WARN = "[WARN] " if platform.system() == 'Windows' else "⚠"
INFO = "[INFO] " if platform.system() == 'Windows' else "ℹ"
CLOCK = "[TIME] " if platform.system() == 'Windows' else "⏱"
ROCKET = "[GO]   " if platform.system() == 'Windows' else "🚀"
TARGET = "[TEST] " if platform.system() == 'Windows' else "🎯"
CHART = "[DATA] " if platform.system() == 'Windows' else "📊"
BOOK = "[DOC]  " if platform.system() == 'Windows' else "📖"
WRENCH = "[FIX]  " if platform.system() == 'Windows' else "🔧"
FIRE = "[HOT]  " if platform.system() == 'Windows' else "🔥"
STAR = "[STAR] " if platform.system() == 'Windows' else "⭐"


import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# # Module 1: Measurement
# from measurement_module.src import FairnessAnalyzer
# from measurement_module.src.mlops_integration import log_fairness_metrics_to_mlflow, log_fairness_report

# # Module 2: Pipeline (Bias Detection & Mitigation)
# from pipeline_module.src.bias_detection import BiasDetector
# from pipeline_module.src.transformers.reweighting import InstanceReweighting
# from pipeline_module.src.transformers.resampling import GroupBalancer

# # Module 3: Training (Fair Model Training)
# from training_module.src.sklearn_wrappers import ReductionsWrapper
# from training_module.src.calibration import GroupFairnessCalibrator

# # Module 4: Monitoring (Production Tracking)
# from monitoring_module.src.realtime_tracker import RealTimeFairnessTracker
# from monitoring_module.src.drift_detection import FairnessDriftDetector
# Import all modules with error handling
try:
    from shared.logging import get_logger, PipelineLogger
    from shared.schemas import PipelineConfig
except ImportError as e:
    print(f"{CROSS} Error importing shared modules: {e}")
    print(f"{INFO} Make sure shared/ directory exists with logging.py and schemas.py")
    sys.exit(1)

try:
    from measurement_module.src import FairnessAnalyzer
except ImportError as e:
    print(f"{CROSS} Error importing measurement_module: {e}")
    print(f"{INFO} Make sure measurement_module.py exists")
    sys.exit(1)

try:
    from pipeline_module.src.bias_detection import BiasDetector
    from pipeline_module.src.transformers.reweighting import InstanceReweighting
    from pipeline_module.src.transformers.resampling import GroupBalancer
except ImportError as e:
    print(f"{CROSS} Error importing pipeline_module: {e}")
    print(f"{INFO} Make sure pipeline module exists with BiasDetector and InstanceReweighting")
    sys.exit(1)

try:
    from training_module.src.sklearn_wrappers import ReductionsWrapper
    from training_module.src.calibration import GroupFairnessCalibrator
except ImportError as e:
    print(f"{CROSS} Error importing training_module: {e}")
    print(f"{INFO} Make sure training module exists with ReductionsWrapper")
    sys.exit(1)

try:
    from monitoring_module.src.realtime_tracker import RealTimeFairnessTracker
    from monitoring_module.src.drift_detection import FairnessDriftDetector
except ImportError as e:
    print(f"{CROSS} Error importing monitoring_module: {e}")
    print(f"{INFO} Make sure monitoring module exists")
    sys.exit(1)

logger = get_logger(__name__)

# Try MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class FairnessPipelineOrchestrator:
    """
    Orchestrate complete fairness pipeline execution.
    
    Coordinates all 4 modules to execute end-to-end workflow:
    1. Measurement Module - Baseline fairness metrics
    2. Pipeline Module - Bias detection and mitigation  
    3. Training Module - Fair model training
    4. Monitoring Module - Production monitoring setup
    """
    
    def __init__(self, config_path: str):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE and self.config.get('mlflow', {}).get('enabled', False):
            mlflow.set_experiment(
                self.config['mlflow'].get('experiment_name', 'fairness_pipeline')
            )
            self.mlflow_run = mlflow.start_run()
            logger.info(f"{ROCKET} MLflow tracking enabled")
            
            # Log config parameters
            mlflow.log_params({
                'fairness_threshold': self.config.get('fairness_threshold', 0.1),
                'mitigation_method': self.config.get('bias_mitigation', {}).get('method', 'reweighting'),
                'use_constraints': self.config.get('training', {}).get('use_fairness_constraints', False),
            })
        else:
            self.mlflow_run = None
            if not MLFLOW_AVAILABLE:
                logger.warning(f"{WARN} MLflow not installed - experiment tracking disabled")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"{CROSS} Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['data', 'bias_detection', 'fairness_metrics']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"{CROSS} Missing required config field: {field}")
        
        logger.info(f"{CHECK} Loaded config from {self.config_path}")
        return config
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from CSV.
        
        Args:
            data_path: Optional override path (takes precedence over config)
            
        Returns:
            Loaded DataFrame
        """
        if data_path:
            df = pd.read_csv(data_path)
            logger.info(f"{CHART} Loaded data from CLI argument: {data_path}")
        else:
            # Use config path
            data_path = self.config.get('data', {}).get('path')
            if not data_path:
                raise ValueError(f"{CROSS} No data path specified in config or CLI")
            df = pd.read_csv(data_path)
            logger.info(f"{CHART} Loaded data from config: {data_path}")
        
        logger.info(f"{INFO} Dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Log data statistics to MLflow
        if self.mlflow_run:
            mlflow.log_params({
                'n_samples': len(df),
                'n_features': len(df.columns),
            })
        
        return df
    
    def run_measurement(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        stage: str = "baseline"
    ) -> Dict[str, Any]:
        """
        Measure fairness metrics using FairnessAnalyzer.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute values
            stage: "baseline" or "final" for logging
            
        Returns:
            Dictionary of measurement results (MetricResult objects)
        """
        with PipelineLogger(logger, f"measurement_{stage}"):
            # Initialize FairnessAnalyzer (no parameters needed)
            analyzer = FairnessAnalyzer()
            
            metrics = self.config.get('fairness_metrics', ['demographic_parity'])
            results = {}
            
            logger.info(f"{TARGET} Computing {len(metrics)} fairness metric(s)...")
            
            for metric in metrics:
                # Call compute_metric with correct signature
                result = analyzer.compute_metric(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features,
                    metric=metric
                )
                
                results[metric] = result
                
                # Log to MLflow
                if self.mlflow_run:
                    mlflow.log_metric(f"{stage}_{metric}", result.value)
                    mlflow.log_metric(f"{stage}_{metric}_fair", int(result.is_fair))
                    if hasattr(result, 'p_value') and result.p_value is not None:
                        mlflow.log_metric(f"{stage}_{metric}_pvalue", result.p_value)
                
                # Pretty print
                status = f"{CHECK} FAIR" if result.is_fair else f"{CROSS} UNFAIR"
                logger.info(
                    f"  {metric}: {result.value:.4f} {status} "
                    f"(threshold: {result.threshold})"
                )
                
                # Log confidence interval if available
                if hasattr(result, 'confidence_interval') and result.confidence_interval:
                    ci = result.confidence_interval
                    logger.info(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            
            return results
    
    def run_bias_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect various types of bias in raw data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of bias detection results
        """
        with PipelineLogger(logger, "bias_detection"):
            detector = BiasDetector(
                representation_threshold=self.config.get('bias_detection', {}).get(
                    'representation_threshold', 0.2
                ),
                proxy_threshold=self.config.get('bias_detection', {}).get(
                    'proxy_threshold', 0.5
                ),
            )
            
            protected_attr = self.config['bias_detection']['protected_attribute']
            
            logger.info(f"{WRENCH} Scanning data for bias patterns...")
            results = detector.detect_all_bias_types(
                df,
                protected_attribute=protected_attr,
                reference_distribution=self.config['bias_detection'].get('reference_distribution'),
            )
            
            # # Log and summarize results
            # detected_count = sum(1 for r in results.values() if r.detected)
            # logger.info(f"{INFO} Bias detection complete: {detected_count}/{len(results)} types detected")
            
            # for bias_type, result in results.items():
            #     status = f"{FIRE} DETECTED" if result.detected else f"{CHECK} CLEAR"
            #     logger.info(
            #         f"  {bias_type}: {status} "
            #         f"(severity: {result.severity:.2f})"
            #     )
                
            #     if self.mlflow_run:
            #         mlflow.log_metric(f"bias_{bias_type}_detected", int(result.detected))
            #         mlflow.log_metric(f"bias_{bias_type}_severity", result.severity)
            
            # return results
            
                        # Log and summarize results
            detected_count = sum(1 for r in results.values() if r.detected)
            logger.info(f"{INFO} Bias detection complete: {detected_count}/{len(results)} types detected")
            
            # ==============================================================================
            # UPDATED LOOP: Handles severity as both float and string
            # ==============================================================================
            for bias_type, result in results.items():
                status = f"{FIRE} DETECTED" if result.detected else f"{CHECK} CLEAR"
                
                # FIX: Try to format as float, otherwise just print the string
                try:
                    severity_display = f"{float(result.severity):.2f}"
                except (ValueError, TypeError):
                    severity_display = str(result.severity)

                logger.info(
                    f"  {bias_type}: {status} "
                    f"(severity: {severity_display})"
                )
                
                if self.mlflow_run:
                    # FIX: MLflow metrics must be numbers. 
                    # If severity is a string (e.g., "High"), log it as a Parameter instead.
                    try:
                        severity_val = float(result.severity)
                        mlflow.log_metric(f"bias_{bias_type}_severity", severity_val)
                    except (ValueError, TypeError):
                        # Fallback: Log as a parameter (string) if it can't be a number
                        mlflow.log_param(f"bias_{bias_type}_severity", str(result.severity))
            # ==============================================================================
            
            return results
    
    def run_mitigation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> tuple:
        """
        Apply bias mitigation technique.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: Protected attribute values
            
        Returns:
            Tuple of (X_transformed, y_transformed, sample_weights)
            If no mitigation applied, weights will be None
        """
        with PipelineLogger(logger, "mitigation"):
            method = self.config.get('bias_mitigation', {}).get('method', 'reweighting')
            
            logger.info(f"{WRENCH} Applying mitigation method: {method}")
            
            if method == 'reweighting':
                reweighter = InstanceReweighting(
                    **self.config.get('bias_mitigation', {}).get('params', {})
                )
                X_out, y_out, weights = reweighter.fit_transform(
                    X, y, sensitive_features=sensitive_features
                )
                
                # Log statistics
                logger.info(
                    f"{CHECK} Reweighting applied: "
                    f"weights in [{weights.min():.2f}, {weights.max():.2f}], "
                    f"mean={weights.mean():.2f}"
                )
                
                if self.mlflow_run:
                    mlflow.log_metric('weight_min', weights.min())
                    mlflow.log_metric('weight_max', weights.max())
                    mlflow.log_metric('weight_mean', weights.mean())
                
                return X_out, y_out, weights
            
            elif method == 'none':
                logger.info(f"{INFO} No mitigation applied (method='none')")
                return X, y, None
            
            else:
                logger.error(f"{CROSS} Unknown mitigation method: {method}")
                raise ValueError(f"Unknown mitigation method: {method}")
    
    def run_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ):
        """
        Train fair model using specified strategy.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_train: Protected attributes for training
            sample_weights: Optional sample weights from mitigation
            
        Returns:
            Trained model
        """
        with PipelineLogger(logger, "training"):
            from sklearn.linear_model import LogisticRegression
            
            use_constraints = self.config.get('training', {}).get('use_fairness_constraints', False)
            
            if use_constraints:
                # Train with fairness constraints using ReductionsWrapper
                constraint_type = self.config.get('training', {}).get('constraint_type', 'demographic_parity')
                eps = self.config.get('training', {}).get('eps', 0.05)
                
                logger.info(
                    f"{ROCKET} Training with fairness constraints: "
                    f"{constraint_type} (eps={eps})"
                )
                
                model = ReductionsWrapper(
                    base_estimator=LogisticRegression(random_state=42, max_iter=1000),
                    constraint=constraint_type,
                    eps=eps,
                )
                model.fit(X_train, y_train, sensitive_features=sensitive_train)
                
                if self.mlflow_run:
                    mlflow.log_param('training_method', 'reductions')
                    mlflow.log_param('constraint_type', constraint_type)
                    mlflow.log_param('constraint_eps', eps)
            
            else:
                # Train with sample weights
                logger.info(f"{ROCKET} Training with sample weights")
                
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train, sample_weight=sample_weights)
                
                if self.mlflow_run:
                    mlflow.log_param('training_method', 'weighted')
            
            logger.info(f"{CHECK} Model training complete")
            return model
    
    def run_validation(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Validate final model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            sensitive_test: Protected attributes for test
            
        Returns:
            Dictionary with accuracy and fairness results
        """
        with PipelineLogger(logger, "validation"):
            # Generate predictions
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            logger.info(f"{STAR} Test accuracy: {accuracy:.4f}")
            
            if self.mlflow_run:
                mlflow.log_metric("test_accuracy", accuracy)
            
            # Measure fairness on final model
            fairness_results = self.run_measurement(
                y_test, y_pred, sensitive_test, stage="final"
            )
            
            results = {
                'accuracy': accuracy,
                'fairness': fairness_results
            }
            
            return results
    
    def setup_monitoring(self, model=None) -> RealTimeFairnessTracker:
        """
        Setup production monitoring tracker.
        
        Args:
            model: Optional trained model for monitoring
            
        Returns:
            Configured RealTimeFairnessTracker instance
        """
        with PipelineLogger(logger, "monitoring_setup"):
            tracker = RealTimeFairnessTracker(
                window_size=self.config.get('monitoring', {}).get('window_size', 1000),
                metrics=self.config.get('fairness_metrics', ['demographic_parity']),
            )
            
            logger.info(
                f"{CHECK} Monitoring tracker initialized "
                f"(window_size={self.config.get('monitoring', {}).get('window_size', 1000)})"
            )
            
            # Demonstrate monitoring capability
            if model is not None:
                logger.info(f"{INFO} Tracker ready for production deployment")
                logger.info(f"{INFO} Use tracker.update(predictions, labels, sensitive_features) to track in production")
            
            return tracker
    
    def run(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete fairness pipeline.
        
        Args:
            data_path: Optional data path override
            
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("=" * 80)
        logger.info(f"{ROCKET} STARTING FAIRNESS PIPELINE")
        logger.info("=" * 80)
        
        try:
            # ==================================================================
            # DATA PREPARATION
            # ==================================================================
            df = self.load_data(data_path)
            
            # Extract columns from config
            protected_attr = self.config['bias_detection']['protected_attribute']
            target_col = self.config['data']['target_column']
            feature_cols = self.config['data'].get('feature_columns')
            
            if not feature_cols:
                # Use all columns except protected and target
                feature_cols = [c for c in df.columns if c not in [protected_attr, target_col]]
                logger.info(f"{INFO} Auto-detected {len(feature_cols)} feature columns")
            
            X = df[feature_cols].values
            y = df[target_col].values
            sensitive = df[protected_attr].values
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, sensitive, 
                test_size=self.config.get('data', {}).get('test_size', 0.3),
                random_state=42, 
                stratify=y
            )
            
            logger.info(f"{INFO} Train: {len(X_train)} samples | Test: {len(X_test)} samples")
            
            # ==================================================================
            # STEP 1: BASELINE MEASUREMENT
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{TARGET} STEP 1: BASELINE MEASUREMENT")
            logger.info("=" * 80)
            
            # Train baseline unfair model for comparison
            from sklearn.linear_model import LogisticRegression
            baseline_model = LogisticRegression(random_state=42, max_iter=1000)
            baseline_model.fit(X_train, y_train)
            y_pred_baseline = baseline_model.predict(X_test)
            
            baseline_acc = (y_pred_baseline == y_test).mean()
            logger.info(f"{INFO} Baseline model accuracy: {baseline_acc:.4f}")
            
            if self.mlflow_run:
                mlflow.log_metric("baseline_accuracy", baseline_acc)
            
            # Measure baseline fairness
            baseline_results = self.run_measurement(y_test, y_pred_baseline, s_test, stage="baseline")
            self.results['baseline'] = {
                'accuracy': baseline_acc,
                'fairness': baseline_results
            }
            
            # ==================================================================
            # STEP 2: BIAS DETECTION IN DATA
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{WRENCH} STEP 2: BIAS DETECTION")
            logger.info("=" * 80)
            
            bias_results = self.run_bias_detection(df)
            self.results['bias_detection'] = bias_results
            
            # ==================================================================
            # STEP 3: TRANSFORM DATA AND TRAIN FAIR MODEL
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{ROCKET} STEP 3: MITIGATION & TRAINING")
            logger.info("=" * 80)
            
            # Apply mitigation
            X_train_mit, y_train_mit, weights = self.run_mitigation(X_train, y_train, s_train)
            
            # Train fair model
            fair_model = self.run_training(X_train_mit, y_train_mit, s_train, weights)
            
            # ==================================================================
            # STEP 4: FINAL VALIDATION
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{STAR} STEP 4: FINAL VALIDATION")
            logger.info("=" * 80)
            
            validation_results = self.run_validation(fair_model, X_test, y_test, s_test)
            self.results['validation'] = validation_results
            
            # ==================================================================
            # STEP 5: SETUP MONITORING
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{CHART} STEP 5: MONITORING SETUP")
            logger.info("=" * 80)
            
            tracker = self.setup_monitoring(model=fair_model)
            self.results['tracker'] = tracker
            
            # ==================================================================
            # SAVE ARTIFACTS
            # ==================================================================
            if self.mlflow_run:
                logger.info(f"\n{BOOK} Logging artifacts to MLflow...")
                mlflow.sklearn.log_model(fair_model, "fair_model")
                mlflow.sklearn.log_model(baseline_model, "baseline_model")
                mlflow.log_artifact(str(self.config_path), "config")
                logger.info(f"{CHECK} Artifacts logged successfully")
            
            # ==================================================================
            # PIPELINE COMPLETE
            # ==================================================================
            logger.info("\n" + "=" * 80)
            logger.info(f"{CHECK} PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            self._print_summary()
            
            return self.results
        
        except Exception as e:
            logger.error(f"\n{CROSS} Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            if self.mlflow_run:
                mlflow.end_run()
                logger.info(f"{INFO} MLflow run ended")
    
    def _print_summary(self):
        """Print comprehensive pipeline execution summary."""
        print("\n" + "=" * 80)
        print(f"{CHART} PIPELINE SUMMARY REPORT")
        print("=" * 80)
        
        # Baseline vs Final Comparison
        print(f"\n{TARGET} BASELINE METRICS (Unfair Model):")
        print(f"  Accuracy: {self.results['baseline']['accuracy']:.4f}")
        for metric, result in self.results['baseline']['fairness'].items():
            status = f"{CHECK} FAIR" if result.is_fair else f"{CROSS} UNFAIR"
            print(f"  {metric}: {result.value:.4f} {status}")
        
        # Bias detection
        print(f"\n{WRENCH} BIAS DETECTION RESULTS:")
        for bias_type, result in self.results['bias_detection'].items():
            if result.detected:
                print(f"  {FIRE} {bias_type}: DETECTED (severity: {result.severity:.2f})")
            else:
                print(f"  {CHECK} {bias_type}: CLEAR")
        
        # Final results
        print(f"\n{STAR} FINAL METRICS (Fair Model):")
        print(f"  Accuracy: {self.results['validation']['accuracy']:.4f}")
        for metric, result in self.results['validation']['fairness'].items():
            status = f"{CHECK} FAIR" if result.is_fair else f"{CROSS} UNFAIR"
            print(f"  {metric}: {result.value:.4f} {status}")
        
        # Improvement analysis
        print(f"\n{ROCKET} FAIRNESS IMPROVEMENT:")
        baseline_fair = self.results['baseline']['fairness']
        final_fair = self.results['validation']['fairness']
        
        for metric in baseline_fair.keys():
            baseline_val = baseline_fair[metric].value
            final_val = final_fair[metric].value
            improvement = baseline_val - final_val
            pct_change = (improvement / baseline_val * 100) if baseline_val != 0 else 0
            
            if improvement > 0:
                print(f"  {metric}: {CHECK} Improved by {improvement:.4f} ({pct_change:.1f}%)")
            elif improvement < 0:
                print(f"  {metric}: {WARN} Worsened by {abs(improvement):.4f} ({abs(pct_change):.1f}%)")
            else:
                print(f"  {metric}: No change")
        
        # Accuracy trade-off
        acc_diff = self.results['validation']['accuracy'] - self.results['baseline']['accuracy']
        print(f"\n{INFO} ACCURACY TRADE-OFF:")
        if acc_diff >= 0:
            print(f"  {CHECK} Accuracy maintained/improved: +{acc_diff:.4f}")
        else:
            print(f"  {WARN} Accuracy reduced: {acc_diff:.4f}")
        
        print("\n" + "=" * 80)
        
        if self.mlflow_run:
            print(f"\n{BOOK} View detailed results in MLflow UI:")
            print(f"  Run: mlflow ui")
            print(f"  Then open: http://localhost:5000")


def main():
    """Main entry point for pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end fairness pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config config.yml
  python run_pipeline.py --config config.yml --data data/loan_data.csv
  
For more information, see README.md
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration YAML file (default: config.yml)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data CSV file (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"{CROSS} Configuration file not found: {args.config}")
        print(f"{INFO} Create a config.yml file or specify path with --config")
        sys.exit(1)
    
    # Run pipeline
    try:
        orchestrator = FairnessPipelineOrchestrator(args.config)
        results = orchestrator.run(data_path=args.data)
        
        print(f"\n{CHECK} Pipeline execution complete!")
        
        if MLFLOW_AVAILABLE:
            print(f"{CHART} View results: mlflow ui")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n{CROSS} Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()