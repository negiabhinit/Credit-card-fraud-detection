"""
LSTM-based Fraud Detection Model
==================================
Deep learning approach using LSTM for fraud detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib

logger = logging.getLogger(__name__)

# Try importing TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("TensorFlow available for LSTM")
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}")


class LSTMDetector:
    """
    LSTM-based fraud detector.
    Processes transactions as sequences for temporal pattern detection.
    """
    
    def __init__(
        self,
        sequence_length: int = 1,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 256,
        class_weight: Optional[Dict] = None,
        random_state: int = 42
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight or {0: 1, 1: 1}
        self.random_state = random_state
        
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None
        
    def _build_model(self, n_features: int) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            # LSTM layer - input shape: (sequence_length, n_features)
            LSTM(self.lstm_units, input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(self.dense_units, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(self.dense_units // 2, activation='relu'),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for LSTM input."""
        # Reshape to (samples, sequence_length, features)
        if X.ndim == 2:
            n_samples, n_features = X.shape
            # If sequence_length > 1, we need to create sequences
            # For now, use sequence_length=1 for tabular-like input
            X_seq = X.reshape(n_samples, self.sequence_length, n_features)
        return X_seq
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'LSTMDetector':
        """Train the LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        logger.info("Training LSTM model...")
        
        # Store feature names
        self.feature_names_ = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Prepare data
        X_array = X.values if hasattr(X, 'values') else X
        X_seq = self._prepare_sequences(X_array)
        
        # Calculate class weight if not provided
        if self.class_weight[1] == 1:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            if n_pos > 0:
                self.class_weight = {0: 1, 1: n_neg / n_pos}
                logger.info(f"Auto class weight: {self.class_weight}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Build and train model
        self.model = self._build_model(X_seq.shape[-1])
        
        self.model.fit(
            X_seq, y,
            validation_split=validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        logger.info("LSTM training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_seq = self._prepare_sequences(X)
        return (self.model.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_seq = self._prepare_sequences(X)
        proba = self.model.predict(X_seq, verbose=0)
        return np.hstack([1 - proba, proba])
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get fraud scores (probability of fraud)."""
        proba = self.predict_proba(X)
        return proba[:, 1]
    
    def evaluate(
        self, 
        X, 
        y_true,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = (self.predict_scores(X) >= threshold).astype(int)
        y_scores = self.predict_scores(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'pr_auc': average_precision_score(y_true, y_scores)
        }
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        if metrics['tp'] + metrics['fn'] > 0:
            metrics['fraud_catch_rate'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model."""
        if self.model:
            self.model.save(path)
            logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        self.model = tf.keras.models.load_model(path)
        self.is_fitted = True
        logger.info(f"LSTM model loaded from {path}")


class HybridEnsembleDetector:
    """
    Hybrid ensemble combining Random Forest + LSTM.
    Provides best of both worlds.
    """
    
    def __init__(
        self,
        rf_model=None,
        lstm_model=None,
        weights: Tuple[float, float] = (0.5, 0.5)
    ):
        self.rf_model = rf_model
        self.lstm_model = lstm_model
        self.weights = weights
        self.is_fitted = False
        
    def fit(self, X, y, **kwargs) -> 'HybridEnsembleDetector':
        """Train both models."""
        if self.rf_model:
            logger.info("Training Random Forest...")
            self.rf_model.fit(X, y)
        
        if self.lstm_model and TF_AVAILABLE:
            logger.info("Training LSTM...")
            self.lstm_model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """Get weighted ensemble predictions."""
        probas = []
        
        if self.rf_model and self.rf_model.is_fitted:
            probas.append(self.rf_model.predict_proba(X) * self.weights[0])
        
        if self.lstm_model and self.lstm_model.is_fitted:
            probas.append(self.lstm_model.predict_proba(X) * self.weights[1])
        
        if probas:
            return np.sum(probas, axis=0)
        else:
            raise ValueError("No models fitted")
    
    def predict_scores(self, X) -> np.ndarray:
        """Get fraud scores."""
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def evaluate(self, X, y_true, threshold: float = 0.5) -> Dict:
        """Evaluate ensemble."""
        y_pred = (self.predict_scores(X) >= threshold).astype(int)
        y_scores = self.predict_scores(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'pr_auc': average_precision_score(y_true, y_scores)
        }


def train_lstm_fraud_detector(
    X_train, y_train,
    X_test, y_test,
    lstm_units: int = 64,
    epochs: int = 10,
    batch_size: int = 256
) -> Tuple[LSTMDetector, Dict]:
    """Train and evaluate LSTM detector."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed")
    
    model = LSTMDetector(
        lstm_units=lstm_units,
        epochs=epochs,
        batch_size=batch_size
    )
    
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics


if __name__ == "__main__":
    # Quick test
    from src.data.loader import load_credit_card_data
    from src.features.preprocessing import fraudDetectionPreprocessor
    
    logging.basicConfig(level=logging.INFO)
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_credit_card_data(
        'data/raw/creditcard.csv', test_size=0.3
    )
    
    print("Preprocessing...")
    prep = fraudDetectionPreprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p = prep.transform(X_test)
    
    print("\nTraining LSTM...")
    lstm_model, metrics = train_lstm_fraud_detector(
        X_train_p, y_train, X_test_p, y_test,
        lstm_units=32, epochs=5, batch_size=512
    )
    
    print("\n=== LSTM Results ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")