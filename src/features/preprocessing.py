"""
Preprocessing Module for Credit Card Fraud Detection
=============================================
Handles data preprocessing, scaling, and feature engineering.
"""

import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# Feature column names (V1-V28 are PCA transformed features)
PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
NON_PCA_FEATURES = ['Time', 'Amount']
ALL_FEATURES = PCA_FEATURES + NON_PCA_FEATURES


class fraudDetectionPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for Credit Card Fraud Detection data.
    
    Handles:
    - Amount scaling (using RobustScaler for outlier robustness)
    - Time feature engineering (extract hour, day patterns)
    - Feature selection
    """
    
    def __init__(
        self,
        scale_amount: bool = True,
        scale_time: bool = True,
        add_temporal_features: bool = True,
        scale_type: str = "robust"  # "robust" or "standard"
    ):
        self.scale_amount = scale_amount
        self.scale_time = scale_time
        self.add_temporal_features = add_temporal_features
        self.scale_type = scale_type
        
        self.amount_scaler: Optional[StandardScaler] = None
        self.time_scaler: Optional[StandardScaler] = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the preprocessor."""
        logger.info("Fitting fraud detection preprocessor...")
        
        X_copy = X.copy()
        
        # Fit scalers if needed
        if self.scale_amount and 'Amount' in X_copy.columns:
            if self.scale_type == "robust":
                self.amount_scaler = RobustScaler()
            else:
                self.amount_scaler = StandardScaler()
            self.amount_scaler.fit(X_copy[['Amount']])
            
        if self.scale_time and 'Time' in X_copy.columns:
            if self.scale_type == "robust":
                self.time_scaler = RobustScaler()
            else:
                self.time_scaler = StandardScaler()
            self.time_scaler.fit(X_copy[['Time']])
        
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        X_transformed = X.copy()
        
        # Scale Amount
        if self.scale_amount and self.amount_scaler is not None:
            if 'Amount' in X_transformed.columns:
                X_transformed['Amount'] = self.amount_scaler.transform(
                    X_transformed[['Amount']]
                )
                
        # Scale Time
        if self.scale_time and self.time_scaler is not None:
            if 'Time' in X_transformed.columns:
                X_transformed['Time'] = self.time_scaler.transform(
                    X_transformed[['Time']]
                )
                
        # Add temporal features (hour of day, day of week)
        if self.add_temporal_features and 'Time' in X.columns:
            # Time is seconds from first transaction
            # 24*60*60 = 86400 seconds in a day
            X_transformed['HourOfDay'] = (X['Time'] / 3600) % 24  # Hour (0-23)
            X_transformed['IsNight'] = ((X['Time'] / 3600) % 24 < 6).astype(int)  # Late night
            X_transformed['IsWeekend'] = (
                (X['Time'] / 86400) % 7 >= 5  # Saturday and Sunday
            ).astype(int)
        
        # Log-Transform Amount for better distribution
        if 'Amount' in X_transformed.columns:
            X_transformed['LogAmount'] = np.log1p(np.abs(X_transformed['Amount']))
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from the dataset."""
    
    def __init__(self, features: Optional[List[str]] = None, drop_features: Optional[List[str]] = None):
        self.features = features
        self.drop_features = drop_features
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.features is not None:
            self.selected_features_ = self.features
        elif self.drop_features is not None:
            self.selected_features_ = [c for c in X.columns if c not in self.drop_features]
        else:
            self.selected_features_ = X.columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline(
    scale_amount: bool = True,
    scale_time: bool = True,
    add_temporal_features: bool = True,
    selected_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Create a sklearn preprocessing pipeline.
    
    Parameters
    ----------
    scale_amount : bool
        Whether to scale the Amount feature
    scale_time : bool
        Whether to scale the Time feature
    add_temporal_features : bool
        Whether to add temporal features (hour, day, etc.)
    selected_features : List[str], optional
        Features to select. If None, select all.
        
    Returns
    -------
    Pipeline
        Sklearn preprocessing pipeline
    """
    steps = []
    
    # Feature selection
    if selected_features:
        steps.append(('selector', FeatureSelector(features=selected_features)))
    
    # Preprocessing
    steps.append((
        'preprocessor',
        fraudDetectionPreprocessor(
            scale_amount=scale_amount,
            scale_time=scale_time,
            add_temporal_features=add_temporal_features
        )
    ))
    
    return Pipeline(steps)


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_amount: bool = True,
    scale_time: bool = True,
    add_temporal_features: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess training and test data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    scale_amount : bool
        Whether to scale Amount
    scale_time : bool
        Whether to scale Time
    add_temporal_features : bool
        Whether to add temporal features
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed training and test data
    """
    preprocessor = fraudDetectionPreprocessor(
        scale_amount=scale_amount,
        scale_time=scale_time,
        add_temporal_features=add_temporal_features
    )
    
    # Fit on training data only
    preprocessor.fit(X_train)
    
    # Transform both train and test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Preprocessed data shape - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed


def get_feature_importanceExplanation(X: pd.DataFrame) -> dict:
    """
    Get feature names and their descriptions.
    
    Returns
    -------
    dict
        Dictionary mapping feature names to descriptions
    """
    descriptions = {
        **{f'V{i}': f'PCA transformed feature {i}' for i in range(1, 29)},
        'Time': 'Seconds elapsed since first transaction',
        'Amount': 'Transaction amount in original currency',
        'HourOfDay': 'Hour of the day (0-23)',
        'IsNight': 'Whether transaction occurred between midnight and 6 AM',
        'IsWeekend': 'Whether transaction occurred on weekend',
        'LogAmount': 'Log-transformed transaction amount'
    }
    return {f: descriptions.get(f, 'Unknown feature') for f in X.columns}