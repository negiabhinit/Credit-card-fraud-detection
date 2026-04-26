"""
Monitoring & Evaluation Module for Credit Card Fraud Detection
================================================================
Handles model monitoring, performance tracking, and drift detection.
"""

import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor for tracking model performance over time.
    
    Features:
    - Performance metrics tracking
    - Data drift detection
    - Performance alerts
    - Historical metrics storage
    """
    
    def __init__(
        self,
        alert_threshold: float = 0.05,
        drift_threshold: float = 0.05,
        history_path: str = "reports/metrics/history.json"
    ):
        """
        Initialize the model monitor.
        
        Parameters
        ----------
        alert_threshold : float
            Threshold for performance drop alerts (e.g., 0.05 = 5%)
        drift_threshold : float
            Threshold for data drift alerts
        history_path : str
            Path to store historical metrics
        """
        self.alert_threshold = alert_threshold
        self.drift_threshold = drift_threshold
        self.history_path = history_path
        
        self.baseline_metrics = None
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load historical metrics."""
        try:
            history_file = Path(self.history_path)
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
        return []
    
    def _save_history(self) -> None:
        """Save historical metrics."""
        try:
            history_file = Path(self.history_path)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def set_baseline(self, metrics: Dict) -> None:
        """
        Set baseline metrics for comparison.
        
        Parameters
        ----------
        metrics : Dict
            Baseline performance metrics
        """
        self.baseline_metrics = metrics
        logger.info("Baseline metrics set")
        logger.info(f"Baseline F1: {metrics.get('f1', 0):.4f}")
        logger.info(f"Baseline ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    
    def check_performance(
        self,
        current_metrics: Dict,
        log_alerts: bool = True
    ) -> Dict[str, any]:
        """
        Check current performance against baseline.
        
        Parameters
        ----------
        current_metrics : Dict
            Current model metrics
        log_alerts : bool
            Whether to log alerts
            
        Returns
        -------
        Dict
            Alert information
        """
        if self.baseline_metrics is None:
            logger.warning("No baseline set. Use set_baseline() first.")
            return {"status": "no_baseline"}
        
        alerts = []
        warnings = []
        
        # Check each metric
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        for metric in metric_names:
            baseline_value = self.baseline_metrics.get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            if baseline_value > 0:
                drop = baseline_value - current_value
                drop_pct = drop / baseline_value
                
                if drop_pct > self.alert_threshold:
                    alert = {
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "drop": drop,
                        "drop_pct": drop_pct,
                        "severity": "critical" if drop_pct > 0.1 else "warning"
                    }
                    alerts.append(alert)
                    
                    if log_alerts:
                        logger.warning(
                            f"{metric.upper()} dropped by {drop_pct*100:.1f}% "
                            f"(baseline: {baseline_value:.4f}, current: {current_value:.4f})"
                        )
                elif drop_pct > self.alert_threshold / 2:
                    warning = {
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "drop_pct": drop_pct
                    }
                    warnings.append(warning)
        
        # Record to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "alerts": alerts
        })
        self._save_history()
        
        return {
            "status": "alert" if alerts else "ok",
            "alerts": alerts,
            "warnings": warnings,
            "metrics": current_metrics
        }
    
    def detect_drift(
        self,
        X_baseline: np.ndarray,
        X_current: np.ndarray
    ) -> Dict[str, float]:
        """
        Detect data drift using statistical tests.
        
        Parameters
        ----------
        X_baseline : np.ndarray
            Baseline data
        X_current : np.ndarray
            Current data
            
        Returns
        -------
        Dict
            Drift metrics
        """
        results = {}
        
        # Population Stability Index (PSI)
        results['psi'] = self._calculate_psi(X_baseline, X_current)
        
        # Basic distribution statistics
        for i in range(min(10, X_baseline.shape[1])):  # Check first 10 features
            baseline_mean = np.mean(X_baseline[:, i])
            current_mean = np.mean(X_current[:, i])
            
            if baseline_mean != 0:
                mean_shift = abs(current_mean - baseline_mean) / abs(baseline_mean)
                results[f'feature_{i}_mean_shift'] = mean_shift
        
        # Check if drift detected
        drift_detected = results['psi'] > self.drift_threshold
        
        if drift_detected:
            logger.warning(f"Data drift detected! PSI: {results['psi']:.4f}")
        else:
            logger.info(f"No significant drift. PSI: {results['psi']:.4f}")
        
        results['drift_detected'] = drift_detected
        
        return results
    
    def _calculate_psi(
        self,
        X_baseline: np.ndarray,
        X_current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Minor change
        PSI > 0.2: Major change
        """
        # Flatten arrays
        baseline = X_baseline.flatten()
        current = X_current.flatten()
        
        # Create bins using baseline distribution
        percentiles = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        percentiles[0] = -np.inf
        percentiles[-1] = np.inf
        
        # Calculate expected and actual percentages
        expected = []
        actual = []
        
        for i in range(bins):
            expected_pct = np.mean((baseline >= percentiles[i]) & (baseline < percentiles[i + 1]))
            actual_pct = np.mean((current >= percentiles[i]) & (current < percentiles[i + 1]))
            
            expected.append(max(expected_pct, 1e-10))
            actual.append(max(actual_pct, 1e-10))
        
        # Calculate PSI
        psi = np.sum((actual - expected) * np.log(actual / expected))
        
        return psi
    
    def get_performance_summary(self) -> Dict:
        """
        Get summary of historical performance.
        
        Returns
        -------
        Dict
            Performance summary
        """
        if not self.history:
            return {"status": "no_data"}
        
        # Get latest metrics
        latest = self.history[-1]['metrics']
        
        # Calculate trends
        if len(self.history) > 1:
            previous = self.history[-2]['metrics']
            
            trends = {}
            for metric in ['f1', 'roc_auc', 'pr_auc']:
                if metric in latest and metric in previous:
                    change = latest[metric] - previous[metric]
                    trends[metric] = change
            
            return {
                "latest": latest,
                "previous": previous,
                "trends": trends,
                "alerts_count": len(self.history[-1].get('alerts', []))
            }
        
        return {
            "latest": latest,
            "alerts_count": 0
        }


class DataQualityChecker:
    """
    Check data quality and generate reports.
    """
    
    def __init__(self):
        self.quality_checks = []
        
    def check_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.05
    ) -> Dict:
        """
        Check for missing values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to check
        max_missing_pct : float
            Maximum acceptable percentage of missing values
            
        Returns
        -------
        Dict
            Missing value check results
        """
        results = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)
            results[col] = {
                "missing_count": int(missing_count),
                "missing_pct": float(missing_pct),
                "passed": missing_pct <= max_missing_pct
            }
        
        return results
    
    def check_duplicates(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Check for duplicate records.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to check
            
        Returns
        -------
        Dict
            Duplicate check results
        """
        n_duplicates = df.duplicated().sum()
        
        return {
            "duplicate_count": int(n_duplicates),
            "duplicate_pct": float(n_duplicates / len(df)),
            "passed": n_duplicates == 0
        }
    
    def check_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> Dict:
        """
        Check for outliers using IQR method.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to check
        columns : List[str]
            Columns to check. If None, check all numeric columns
        threshold : float
            IQR multiplier for outlier detection
            
        Returns
        -------
        Dict
            Outlier check results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            results[col] = {
                "outlier_count": int(outliers),
                "outlier_pct": float(outliers / len(df)),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        
        return results
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict:
        """
        Run all data quality checks.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to check
            
        Returns
        -------
        Dict
            All check results
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_values": self.check_missing_values(df),
            "duplicates": self.check_duplicates(df),
            "outliers": self.check_outliers(df)
        }


def generate_evaluation_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "reports/evaluation_report.json"
) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Parameters
    ----------
    model : FraudDetectionModel
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    output_path : str
        Path to save report
        
    Returns
    -------
    Dict
        Evaluation report
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_scores = model.predict_scores(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "pr_auc": float(average_precision_score(y_test, y_scores))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }
    
    # Business metrics
    business_metrics = {
        "fraud_catch_rate": confusion["true_positives"] / (confusion["true_positives"] + confusion["false_negatives"]) if (confusion["true_positives"] + confusion["false_negatives"]) > 0 else 0,
        "false_alarm_rate": confusion["false_positives"] / (confusion["false_positives"] + confusion["true_negatives"]) if (confusion["false_positives"] + confusion["true_negatives"]) > 0 else 0,
    }
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "business_metrics": business_metrics
    }
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")
    
    return report


# Entry point for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test monitoring
    monitor = ModelMonitor()
    
    # Set baseline
    baseline = {
        'accuracy': 0.99,
        'precision': 0.80,
        'recall': 0.75,
        'f1': 0.77,
        'roc_auc': 0.95,
        'pr_auc': 0.80
    }
    monitor.set_baseline(baseline)
    
    # Check current performance
    current = {
        'accuracy': 0.98,
        'precision': 0.75,
        'recall': 0.70,
        'f1': 0.72,
        'roc_auc': 0.93,
        'pr_auc': 0.75
    }
    
    result = monitor.check_performance(current)
    print(f"\nStatus: {result['status']}")
    print(f"Alerts: {len(result.get('alerts', []))}")