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
CHECK = "[OK]   " if platform.system() == 'Windows' else f"{{"CHECK"}}"
CROSS = "[FAIL] " if platform.system() == 'Windows' else f"{{"CROSS"}}"
WARN = "[WARN] " if platform.system() == 'Windows' else f"{{"WARN"}}"
INFO = "[INFO] " if platform.system() == 'Windows' else f"{{"INFO"}}"
CLOCK = "[TIME] " if platform.system() == 'Windows' else f"{{"CLOCK"}}"
ROCKET = "[GO]   " if platform.system() == 'Windows' else f"{{"ROCKET"}}"
TARGET = "[TEST] " if platform.system() == 'Windows' else f"{{"TARGET"}}"
CHART = "[DATA] " if platform.system() == 'Windows' else f"{{"CHART"}}"
BOOK = "[DOC]  " if platform.system() == 'Windows' else f"{{"BOOK"}}"
WRENCH = "[FIX]  " if platform.system() == 'Windows' else f"{{"WRENCH"}}"
FIRE = "[HOT]  " if platform.system() == 'Windows' else f"{{"FIRE"}}"
STAR = "[STAR] " if platform.system() == 'Windows' else f"{{"STAR"}}"


import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import all modules
from shared.logging import get_logger, PipelineLogger
from shared.schemas import PipelineConfig

from measurement_module import FairnessAnalyzer
from pipeline_module import BiasDetector, InstanceReweighting
from training_module import ReductionsWrapper
from monitoring_module import RealTimeFairnessTracker, generate_monitoring_report

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
    
    Coordinates all 4 modules to execute end-to-end workflow.
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
            logger.info("MLflow tracking enabled")
        else:
            self.mlflow_run = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {self.config_path}")
        return config
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load dataset."""
        if data_path:
            df = pd.read_csv(data_path)
        else:
            # Use config path
            data_path = self.config.get('data', {}).get('path')
            if not data_path:
                raise ValueError("No data path specified")
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded data: {len(df)} samples, {len(df.columns)} columns")
        return df
    
    def run_measurement(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Step 1: Measure baseline fairness.
        
        Returns:
            Dictionary of measurement results
        """
        with PipelineLogger(logger, "measurement"):
            analyzer = FairnessAnalyzer(
                confidence_level=self.config.get('fairness_threshold', 0.95),
                bootstrap_samples=self.config.get('bootstrap_samples', 1000),
            )
            
            metrics = self.config.get('fairness_metrics', ['demographic_parity'])
            results = {}
            
            for metric in metrics:
                result = analyzer.compute_metric(
                    y_true,
                    y_pred,
                    sensitive_features,
                    metric=metric,
                    threshold=self.config.get('fairness_threshold', 0.1),
                )
                
                results[metric] = result
                
                # Log to MLflow
                if self.mlflow_run:
                    mlflow.log_metric(f"baseline_{metric}", result.value)
                    mlflow.log_metric(f"baseline_{metric}_fair", int(result.is_fair))
                
                logger.info(
                    f"Baseline {metric}: {result.value:.4f} "
                    f"(fair: {result.is_fair})"
                )
            
            return results
    
    def run_bias_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 2: Detect bias in data.
        
        Returns:
            Dictionary of bias detection results
        """
        with PipelineLogger(logger, "bias_detection"):
            detector = BiasDetector(
                representation_threshold=0.2,
                proxy_threshold=0.5,
            )
            
            protected_attr = self.config['bias_detection']['protected_attribute']
            
            results = detector.detect_all_bias_types(
                df,
                protected_attribute=protected_attr,
                reference_distribution=self.config['bias_detection'].get('reference_distribution'),
            )
            
            # Log results
            for bias_type, result in results.items():
                logger.info(
                    f"Bias detection - {bias_type}: "
                    f"{'DETECTED' if result.detected else 'NOT DETECTED'} "
                    f"(severity: {result.severity})"
                )
                
                if self.mlflow_run:
                    mlflow.log_metric(f"bias_{bias_type}_detected", int(result.detected))
            
            return results
    
    def run_mitigation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> tuple:
        """
        Step 3: Apply bias mitigation.
        
        Returns:
            Tuple of (X, y, weights) or (X_resampled, y_resampled, None)
        """
        with PipelineLogger(logger, "mitigation"):
            method = self.config.get('bias_mitigation', {}).get('method', 'reweighting')
            
            if method == 'reweighting':
                reweighter = InstanceReweighting(
                    **self.config['bias_mitigation'].get('params', {})
                )
                X_out, y_out, weights = reweighter.fit_transform(
                    X, y, sensitive_features=sensitive_features
                )
                logger.info(f"Reweighting applied: weights range [{weights.min():.2f}, {weights.max():.2f}]")
                return X_out, y_out, weights
            
            elif method == 'none':
                logger.info("No mitigation applied")
                return X, y, None
            
            else:
                raise ValueError(f"Unknown mitigation method: {method}")
    
    def run_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        sample_weights: np.ndarray = None,
    ):
        """
        Step 4: Train fair model.
        
        Returns:
            Trained model
        """
        with PipelineLogger(logger, "training"):
            from sklearn.linear_model import LogisticRegression
            
            use_constraints = self.config.get('training', {}).get('use_fairness_constraints', False)
            
            if use_constraints:
                # Train with fairness constraints
                model = ReductionsWrapper(
                    base_estimator=LogisticRegression(random_state=42, max_iter=1000),
                    constraint=self.config['training'].get('constraint_type', 'demographic_parity'),
                    eps=self.config['training'].get('eps', 0.05),
                )
                model.fit(X_train, y_train, sensitive_features=sensitive_train)
                logger.info("Trained with fairness constraints")
            
            else:
                # Train with sample weights
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train, sample_weight=sample_weights)
                logger.info("Trained with sample weights")
            
            return model
    
    def run_validation(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Step 5: Validate final fairness.
        
        Returns:
            Dictionary of validation results
        """
        with PipelineLogger(logger, "validation"):
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            if self.mlflow_run:
                mlflow.log_metric("test_accuracy", accuracy)
            
            # Fairness metrics
            analyzer = FairnessAnalyzer()
            metrics = self.config.get('fairness_metrics', ['demographic_parity'])
            
            results = {'accuracy': accuracy, 'fairness': {}}
            
            for metric in metrics:
                result = analyzer.compute_metric(
                    y_test,
                    y_pred,
                    sensitive_test,
                    metric=metric,
                    threshold=self.config.get('fairness_threshold', 0.1),
                )
                
                results['fairness'][metric] = result
                
                logger.info(
                    f"Final {metric}: {result.value:.4f} "
                    f"(fair: {result.is_fair})"
                )
                
                if self.mlflow_run:
                    mlflow.log_metric(f"final_{metric}", result.value)
                    mlflow.log_metric(f"final_{metric}_fair", int(result.is_fair))
            
            return results
    
    def setup_monitoring(self) -> RealTimeFairnessTracker:
        """
        Step 6: Setup production monitoring.
        
        Returns:
            Configured tracker
        """
        with PipelineLogger(logger, "monitoring_setup"):
            tracker = RealTimeFairnessTracker(
                window_size=self.config.get('monitoring', {}).get('window_size', 1000),
                metrics=self.config.get('fairness_metrics', ['demographic_parity']),
            )
            
            logger.info("Monitoring tracker initialized")
            return tracker
    
    def run(self, data_path: str = None) -> Dict[str, Any]:
        """
        Execute complete pipeline.
        
        Args:
            data_path: Optional data path (overrides config)
            
        Returns:
            Dictionary of all results
        """
        logger.info("=" * 60)
        logger.info("STARTING FAIRNESS PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Extract columns
            protected_attr = self.config['bias_detection']['protected_attribute']
            target_col = self.config['data']['target_column']
            feature_cols = self.config['data'].get('feature_columns')
            
            if not feature_cols:
                feature_cols = [c for c in df.columns if c not in [protected_attr, target_col]]
            
            X = df[feature_cols].values
            y = df[target_col].values
            sensitive = df[protected_attr].values
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, sensitive, test_size=0.3, random_state=42, stratify=y
            )
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Train baseline for comparison
            from sklearn.linear_model import LogisticRegression
            baseline = LogisticRegression(random_state=42, max_iter=1000)
            baseline.fit(X_train, y_train)
            y_pred_baseline = baseline.predict(X_test)
            
            # Step 1: Measure baseline
            baseline_results = self.run_measurement(y_test, y_pred_baseline, s_test)
            self.results['baseline'] = baseline_results
            
            # Step 2: Detect bias
            bias_results = self.run_bias_detection(df)
            self.results['bias_detection'] = bias_results
            
            # Step 3: Mitigate
            X_train_mit, y_train_mit, weights = self.run_mitigation(X_train, y_train, s_train)
            
            # Step 4: Train
            model = self.run_training(X_train_mit, y_train_mit, s_train, weights)
            
            # Step 5: Validate
            validation_results = self.run_validation(model, X_test, y_test, s_test)
            self.results['validation'] = validation_results
            
            # Step 6: Setup monitoring
            tracker = self.setup_monitoring()
            self.results['tracker'] = tracker
            
            # Save model and config
            if self.mlflow_run:
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifact(str(self.config_path))
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            self._print_summary()
            
            return self.results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            if self.mlflow_run:
                mlflow.end_run()
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        
        # Baseline
        print("\n📊 Baseline Metrics:")
        for metric, result in self.results['baseline'].items():
            status = f"{{"CHECK"}} FAIR" if result.is_fair else f"{{"CROSS"}} UNFAIR"
            print(f"  {metric}: {result.value:.4f} {status}")
        
        # Bias detection
        print("\n🔍 Bias Detection:")
        for bias_type, result in self.results['bias_detection'].items():
            status = "🔴 DETECTED" if result.detected else "🟢 CLEAR"
            print(f"  {bias_type}: {status} (severity: {result.severity})")
        
        # Final validation
        print("\n✨ Final Results:")
        print(f"  Accuracy: {self.results['validation']['accuracy']:.4f}")
        for metric, result in self.results['validation']['fairness'].items():
            status = f"{{"CHECK"}} FAIR" if result.is_fair else f"{{"CROSS"}} UNFAIR"
            print(f"  {metric}: {result.value:.4f} {status}")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run fairness pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = FairnessPipelineOrchestrator(args.config)
    results = orchestrator.run(data_path=args.data)
    
    print("\n✅ Pipeline execution complete!")
    print(f"{{"CHART"}} Check MLflow UI for detailed results: mlflow ui")


if __name__ == "__main__":
    main()