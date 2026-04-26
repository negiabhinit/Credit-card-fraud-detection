"""
Tests for Credit Card Fraud Detection System
================================
Unit and integration tests for fraud detection models and pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fraud_detector import (
    FraudDetectionModel,
    RandomForestFraudDetector,
    XGBoostFraudDetector,
    create_fraud_detector
)
from src.features.preprocessing import FraudDetectionPreprocessor
from src.data.loader import load_data_for_prediction
from src.monitoring.monitor import ModelMonitor, DataQualityChecker


# Fixtures
@pytest.fixture
def sample_data():
    """Generate synthetic credit card transaction data for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 30
    
    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'V{i}' for i in range(1, n_features + 1)]
    )
    X['Time'] = np.random.uniform(0, 172800, n_samples)
    X['Amount'] = np.random.uniform(1, 1000, n_samples)
    
    # Generate labels (highly imbalanced - ~1% fraud)
    y = np.random.binomial(1, 0.01, n_samples)
    
    return X, y


@pytest.fixture
def balanced_data():
    """Generate balanced sample data."""
    np.random.seed(42)
    
    n_samples = 500
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'V{i}' for i in range(1, n_features + 1)]
    )
    X['Time'] = np.random.uniform(0, 172800, n_samples)
    X['Amount'] = np.random.uniform(1, 1000, n_samples)
    
    # 50% fraud for balanced testing
    y = np.array([0] * 250 + [1] * 250)
    
    return X, y


# Test FraudDetectionModel base class
class TestFraudDetectionModel:
    """Tests for base fraud detection model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = FraudDetectionModel("test_model")
        assert model.model_name == "test_model"
        assert not model.is_fitted
    
    def test_predict_without_fit_raises(self, sample_data):
        """Test predict raises error when model not fitted."""
        X, y = sample_data
        model = create_fraud_detector("random_forest")
        
        with pytest.raises(ValueError):
            model.predict(X)


# Test Random Forest Model
class TestRandomForestFraudDetector:
    """Tests for Random Forest fraud detector."""
    
    def test_random_forest_creation(self):
        """Test Random Forest model can be created."""
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        assert model.n_estimators == 10
        assert model.max_depth == 5
    
    def test_random_forest_fit_predict(self, sample_data):
        """Test Random Forest can fit and predict."""
        X, y = sample_data
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.model is not None
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Test probability scores
        scores = model.predict_scores(X)
        assert all(0 <= s <= 1 for s in scores)
    
    def test_random_forest_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
    
    def test_threshold_optimization(self, sample_data):
        """Test threshold optimization."""
        X, y = sample_data
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        optimal_threshold = model.find_optimal_threshold(X, y, metric='f1')
        
        assert 0 <= optimal_threshold <= 1


# Test Preprocessor
class TestFraudDetectionPreprocessor:
    """Tests for fraud detection preprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = FraudDetectionPreprocessor()
        assert preprocessor.scale_amount
        assert preprocessor.scale_time
        assert preprocessor.add_temporal_features
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test preprocessor fit and transform."""
        X, y = sample_data
        preprocessor = FraudDetectionPreprocessor(
            scale_amount=True,
            scale_time=True,
            add_temporal_features=True
        )
        
        X_transformed = preprocessor.fit_transform(X)
        
        # Check temporal features added
        assert 'HourOfDay' in X_transformed.columns
        assert 'IsNight' in X_transformed.columns
        assert 'IsWeekend' in X_transformed.columns
        assert 'LogAmount' in X_transformed.columns
    
    def test_preprocessor_robust_scaling(self, sample_data):
        """Test robust scaling."""
        X, y = sample_data
        preprocessor = FraudDetectionPreprocessor(
            scale_type='robust'
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Check scaling applied
        assert X_transformed is not None


# Test Monitoring
class TestModelMonitor:
    """Tests for model monitor."""
    
    def test_monitor_initialization(self):
        """Test monitor can be initialized."""
        monitor = ModelMonitor()
        assert monitor.alert_threshold == 0.05
        assert monitor.drift_threshold == 0.05
    
    def test_set_baseline(self):
        """Test setting baseline metrics."""
        monitor = ModelMonitor()
        baseline = {
            'accuracy': 0.99,
            'precision': 0.80,
            'recall': 0.75,
            'f1': 0.77,
            'roc_auc': 0.95,
            'pr_auc': 0.80
        }
        
        monitor.set_baseline(baseline)
        
        assert monitor.baseline_metrics == baseline
    
    def test_check_performance(self):
        """Test performance checking."""
        monitor = ModelMonitor()
        baseline = {
            'accuracy': 0.99,
            'precision': 0.80,
            'recall': 0.75,
            'f1': 0.77,
            'roc_auc': 0.95,
            'pr_auc': 0.80
        }
        monitor.set_baseline(baseline)
        
        # Current metrics with slight drop
        current = {
            'accuracy': 0.98,
            'precision': 0.75,
            'recall': 0.70,
            'f1': 0.72,
            'roc_auc': 0.93,
            'pr_auc': 0.75
        }
        
        result = monitor.check_performance(current)
        
        assert result['status'] in ['ok', 'alert']
        assert 'metrics' in result


# Test Data Quality Checker
class TestDataQualityChecker:
    """Tests for data quality checker."""
    
    def test_checker_initialization(self):
        """Test data quality checker can be initialized."""
        checker = DataQualityChecker()
        assert checker is not None
    
    def test_check_missing_values(self):
        """Test missing value checking."""
        checker = DataQualityChecker()
        
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, 2, 3, 4]
        })
        
        results = checker.check_missing_values(df)
        
        assert 'A' in results
        assert results['A']['missing_count'] == 1
        assert results['B']['missing_count'] == 0
    
    def test_check_duplicates(self):
        """Test duplicate checking."""
        checker = DataQualityChecker()
        
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [1, 2, 2, 3]
        })
        
        results = checker.check_duplicates(df)
        
        assert results['duplicate_count'] == 1
    
    def test_check_outliers(self):
        """Test outlier checking."""
        checker = DataQualityChecker()
        
        df = pd.DataFrame({
            'A': list(range(100)),
            'B': list(range(100))
        })
        
        results = checker.check_outliers(df)
        
        assert 'A' in results
        assert 'outlier_count' in results['A']


# Integration Tests
class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_training_pipeline(self, sample_data):
        """Test complete training pipeline."""
        X, y = sample_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Preprocess
        preprocessor = FraudDetectionPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_processed, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test_processed, y_test)
        
        assert metrics['accuracy'] >= 0
        assert 'f1' in metrics
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model can be saved and loaded."""
        X, y = sample_data
        
        model = RandomForestFraudDetector(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Save
        model_path = tmp_path / "model.joblib"
        model.save(str(model_path))
        
        # Load
        new_model = RandomForestFraudDetector()
        new_model.load(str(model_path))
        
        assert new_model.is_fitted
        
        # Compare predictions
        orig_pred = model.predict(X)
        new_pred = new_model.predict(X)
        
        assert np.array_equal(orig_pred, new_pred)


# Test fixtures
@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for model."""
    return tmp_path / "test_model.joblib"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])