# bias_detection.py
import numpy as np
import pandas as pd
from scipy import stats

from shared.schemas import BiasDetectionResult
from shared.validation import validate_dataframe, validate_protected_attribute
from shared.logging import get_logger, log_bias_detection
from shared.constants import BIAS_DETECTION_THRESHOLDS

logger = get_logger(__name__)

class BiasDetector:
    def __init__(self, representation_threshold=0.2, proxy_threshold=0.5, statistical_alpha=0.05):
        self.representation_threshold = representation_threshold
        self.proxy_threshold = proxy_threshold
        self.statistical_alpha = statistical_alpha
        logger.info(f"BiasDetector initialized: thresholds={representation_threshold}, {proxy_threshold}")

    def detect_representation_bias(self, df, protected_attribute, reference_distribution=None):
        logger.info(f"Detecting representation bias for '{protected_attribute}'...")
        validate_dataframe(df, required_columns=[protected_attribute])
        validate_protected_attribute(df, protected_attribute, min_group_size=10)
        
        actual_counts = df[protected_attribute].value_counts()
        actual_distribution = (actual_counts / len(df)).to_dict()
        
        if reference_distribution is None:
            n_groups = len(actual_counts)
            reference_distribution = {k: 1.0/n_groups for k in actual_counts.index}
        
        differences = {}
        max_diff = 0.0
        affected_groups = []
        
        for group in actual_distribution.keys():
            actual = actual_distribution[group]
            # Handle type mismatch for keys (e.g. int 0 vs string '0')
            ref_val = reference_distribution.get(group)
            if ref_val is None:
                 ref_val = reference_distribution.get(str(group), 1.0 / len(actual_distribution))
            
            diff = abs(actual - ref_val)
            differences[group] = diff
            
            if diff > max_diff:
                max_diff = diff
            
            if diff > self.representation_threshold * 0.5:
                affected_groups.append(str(group))
        
        detected = max_diff > self.representation_threshold
        
        if max_diff > BIAS_DETECTION_THRESHOLDS['representation_bias']['severe']:
            severity = 'high'
        elif max_diff > BIAS_DETECTION_THRESHOLDS['representation_bias']['moderate']:
            severity = 'medium'
        else:
            severity = 'low'
        
        recommendations = []
        if detected:
            recommendations.append(f"Consider resampling or reweighting.")
            recommendations.append(f"Max difference: {max_diff:.1%}")

        result = BiasDetectionResult(
            bias_type='representation', detected=detected, severity=severity,
            affected_groups=affected_groups, evidence={}, recommendations=recommendations
        )
        log_bias_detection(logger, 'representation', detected, severity, affected_groups)
        return result

    def detect_proxy_variables(self, df, protected_attribute, feature_columns=None):
        logger.info(f"Detecting proxy variables for '{protected_attribute}'...")
        validate_dataframe(df, required_columns=[protected_attribute])
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if protected_attribute in feature_columns:
                feature_columns.remove(protected_attribute)
        
        if len(feature_columns) == 0:
            return BiasDetectionResult(bias_type='proxy', detected=False, severity='low', affected_groups=[], evidence={}, recommendations=["No numeric features"])

        protected_encoded = pd.Categorical(df[protected_attribute]).codes
        proxy_features = []
        
        for col in feature_columns:
            try:
                mask = df[col].notna() & (protected_encoded >= 0)
                if mask.sum() < 10: continue
                
                corr, p_value = stats.spearmanr(df.loc[mask, col], protected_encoded[mask])
                if abs(corr) > self.proxy_threshold and p_value < self.statistical_alpha:
                    proxy_features.append(col)
            except Exception:
                continue
        
        detected = len(proxy_features) > 0
        severity = 'low'
        if detected:
            severity = 'medium' # Simplified for brevity
        
        return BiasDetectionResult(
            bias_type='proxy', detected=detected, severity=severity, affected_groups=proxy_features,
            evidence={}, recommendations=["Proxy variables found"] if detected else ["No proxies"]
        )

    def detect_statistical_disparity(self, df, protected_attribute, feature_columns=None):
        logger.info(f"Detecting statistical disparity for '{protected_attribute}'...")
        validate_dataframe(df, required_columns=[protected_attribute])
        validate_protected_attribute(df, protected_attribute, min_group_size=10)
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if protected_attribute in feature_columns:
                feature_columns.remove(protected_attribute)
        
        disparate_features = []
        groups = df[protected_attribute].unique()
        
        for col in feature_columns:
            try:
                data_0 = df[df[protected_attribute] == groups[0]][col].dropna()
                data_1 = df[df[protected_attribute] == groups[1]][col].dropna()
                
                if len(data_0) < 10 or len(data_1) < 10: continue

                stat, p_val = stats.mannwhitneyu(data_0, data_1, alternative='two-sided')
                
                if p_val < self.statistical_alpha:
                    disparate_features.append(col)
            except Exception:
                continue
        
        detected = len(disparate_features) > 0
        return BiasDetectionResult(
            bias_type='measurement', detected=detected, severity='medium' if detected else 'low',
            affected_groups=disparate_features, evidence={}, recommendations=["Disparity found"] if detected else ["No disparity"]
        )

    def detect_all_bias_types(self, df, protected_attribute, reference_distribution=None, feature_columns=None):
        logger.info("Running comprehensive bias detection...")
        results = {}
        try:
            results['representation'] = self.detect_representation_bias(df, protected_attribute, reference_distribution)
            results['proxy'] = self.detect_proxy_variables(df, protected_attribute, feature_columns)
            results['statistical_disparity'] = self.detect_statistical_disparity(df, protected_attribute, feature_columns)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        return results