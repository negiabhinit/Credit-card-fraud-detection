"""
Fraud Detection Models Module
============================
Contains model implementations for credit card fraud detection.
Includes Random Forest, XGBoost, and ensemble methods with handling for imbalanced data.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)
import joblib

logger = logging.getLogger(__name__)

# Try importing XGBoost
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"XGBoost not available: {e}. Using Random Forest only.")


class FraudDetectionModel:
    """
    Base class for fraud detection models.
    Provides common functionality for training, evaluation, and prediction.
    """
    
    def __init__(self, model_name: str = "fraud_detector"):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None
        self.classes_ = None
        self.metrics_ = {}
        
    def fit(self, X, y) -> 'FraudDetectionModel':
        """Train the model."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba
            pred = self.model.predict(X)
            return np.column_stack([1 - pred, pred])
    
    def predict_scores(self, X) -> np.ndarray:
        """Get fraud scores (probability of fraud)."""
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def evaluate(
        self, X, y_true,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : array-like
            Features
        y_true : array-like
            True labels
        threshold : float
            Decision threshold
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = (self.predict_scores(X) >= threshold).astype(int)
        y_scores = self.predict_scores(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'pr_auc': average_precision_score(y_true, y_scores),
            'average_precision': average_precision_score(y_true, y_scores)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # Business metrics
        if metrics['tp'] + metrics['fn'] > 0:
            metrics['fraud_catch_rate'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        if metrics['tp'] + metrics['fp'] > 0:
            metrics['precision_rate'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
        
        self.metrics_ = metrics
        return metrics
    
    def find_optimal_threshold(
        self, X, y_true,
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal decision threshold.
        
        Parameters
        ----------
        X : array-like
            Features
        y_true : array-like
            True labels
        metric : str
            Metric to optimize ('f1', 'recall', 'precision', 'fraud_catch_rate')
            
        Returns
        -------
        float
            Optimal threshold
        """
        y_scores = self.predict_scores(X)
        thresholds = np.linspace(0.01, 0.99, 99)
        
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'fraud_catch_rate':
                if y_pred.sum() > 0:
                    score = recall_score(y_true, y_pred, zero_division=0)
                else:
                    score = 0
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        logger.info(f"Optimal threshold ({metric}): {best_threshold:.3f} (score: {best_score:.4f})")
        return best_threshold
    
    def cross_validate(
        self, X, y,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        dict
            Cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
        
        return {
            f'{scoring}_mean': scores.mean(),
            f'{scoring}_std': scores.std(),
            'scores': scores.tolist()
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names_,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None


class RandomForestFraudDetector(FraudDetectionModel):
    """
    Random Forest based fraud detector.
    Handles imbalanced data using class_weight parameter.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X, y) -> 'RandomForestFraudDetector':
        """Train the Random Forest model."""
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            self.classes_ = self.model.classes_.tolist()
        
        logger.info("Random Forest training complete")
        return self


class XGBoostFraudDetector(FraudDetectionModel):
    """
    XGBoost based fraud detector.
    Handles imbalanced data using scale_pos_weight.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 1,
        random_state: int = 42,
        use_gpu: bool = False,
        **kwargs
    ):
        super().__init__("XGBoost")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.use_gpu = use_gpu
        
    def fit(self, X, y) -> 'XGBoostFraudDetector':
        """Train the XGBoost model."""
        logger.info("Training XGBoost model...")
        
        # Calculate scale_pos_weight for imbalanced data
        if self.scale_pos_weight == 1:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            if n_pos > 0:
                self.scale_pos_weight = n_neg / n_pos
                logger.info(f"Auto-calculated scale_pos_weight: {self.scale_pos_weight:.2f}")
        
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            self.classes_ = self.model.classes_.tolist() if hasattr(self.model, 'classes_') else [0, 1]
        
        logger.info("XGBoost training complete")
        return self


class EnsembleFraudDetector(FraudDetectionModel):
    """
    Ensemble fraud detector combining multiple models.
    Uses soft voting for better probability estimates.
    """
    
    def __init__(
        self,
        models: Optional[list] = None,
        voting: str = "soft",
        weights: Optional[list] = None
    ):
        super().__init__("Ensemble")
        
        self.models = models or []
        self.voting = voting
        self.weights = weights
        
        # Create base estimators if models provided
        if models:
            self._create_ensemble()
    
    def _create_ensemble(self):
        """Create the voting ensemble."""
        estimators = [(f'model_{i}', model.model) for i, model in enumerate(self.models)]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )
    
    def fit(self, X, y) -> 'EnsembleFraudDetector':
        """Train the ensemble."""
        logger.info("Training Ensemble model...")
        
        # Fit all base models
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            model.fit(X, y)
        
        # Create ensemble with fitted models
        estimators = [(f'model_{i}', model.model) for i, model in enumerate(self.models)]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )
        
        # Fit ensemble (this doesn't re-fit base models)
        self.model.fit(X, y)
        self.is_fitted = True
        
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            self.classes_ = [0, 1]
        
        logger.info("Ensemble training complete")
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities from ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Average predictions from all models
        probas = np.array([m.predict_proba(X) for m in self.models])
        
        if self.weights:
            weights = np.array(self.weights) / sum(self.weights)
            weighted_proba = np.average(probas, axis=0, weights=weights)
            return weighted_proba
        else:
            return np.mean(probas, axis=0)


def create_fraud_detector(
    model_type: str = "random_forest",
    **kwargs
) -> FraudDetectionModel:
    """
    Factory function to create fraud detection models.
    
    Parameters
    ----------
    model_type : str
        Type of model: "random_forest", "xgboost", "ensemble"
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    FraudDetectionModel
        Configured fraud detector
    """
    if model_type == "random_forest":
        return RandomForestFraudDetector(**kwargs)
    elif model_type == "xgboost":
        return XGBoostFraudDetector(**kwargs)
    elif model_type == "ensemble":
        return EnsembleFraudDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    X_train, y_train,
    X_test, y_test,
    model_type: str = "ensemble",
    find_threshold: bool = True,
    **kwargs
) -> Tuple[FraudDetectionModel, Dict[str, float]]:
    """
    Train and evaluate a fraud detection model.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    model_type : str
        Type of model to use
    find_threshold : bool
        Whether to find optimal threshold
    **kwargs : Model parameters
        
    Returns
    -------
    Tuple[model, metrics]
        Trained model and evaluation metrics
    """
    # Create model
    if model_type == "ensemble":
        # Create base models for ensemble
        rf = RandomForestFraudDetector(**kwargs.get('random_forest', {}))
        if XGBOOST_AVAILABLE:
            xgb = XGBoostFraudDetector(**kwargs.get('xgboost', {}))
            model = EnsembleFraudDetector(models=[rf, xgb], weights=kwargs.get('weights', [1, 2]))
        else:
            model = rf
    else:
        model = create_fraud_detector(model_type, **kwargs)
    
    # Train
    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Find optimal threshold
    if find_threshold:
        optimal_threshold = model.find_optimal_threshold(X_test, y_test, metric='f1')
        metrics['optimal_threshold'] = optimal_threshold
        
        # Re-evaluate with optimal threshold
        metrics_optimal = model.evaluate(X_test, y_test, threshold=optimal_threshold)
        metrics.update({f'{k}_optimal': v for k, v in metrics_optimal.items()})
    
    return model, metrics