"""
Data Loader Module for Credit Card Fraud Detection
===================================================
Handles data loading from Kaggle or local CSV files.
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_credit_card_data(
    data_path: str,
    test_size: float = 0.3,
    random_state: int = 42,
    download_from_kaggle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split the credit card fraud dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file or directory for Kaggle download
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    download_from_kaggle : bool
        If True and data not found locally, attempt to download from Kaggle
        
    Returns
    -------
    Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
    """
    # Check if data file exists locally
    data_file = Path(data_path)
    
    if data_file.exists():
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    elif download_from_kaggle:
        logger.info("Data not found locally. Attempting to download from Kaggle...")
        df = _download_from_kaggle(data_path)
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Log dataset statistics
    _log_dataset_stats(df)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data - stratify to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def _download_from_kaggle(data_path: str) -> pd.DataFrame:
    """
    Download credit card fraud dataset from Kaggle.
    Requires kaggle API to be configured.
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        logger.info("Downloading Credit Card Fraud dataset from Kaggle...")
        api.dataset_download_file(
            'mlg-ulb/creditcardfraud',
            'creditcard.csv',
            path=Path(data_path).parent
        )
        
        # Extract if needed and load
        data_file = Path(data_path)
        if data_file.suffix == '.csv':
            return pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
            
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        raise
    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        raise


def _log_dataset_stats(df: pd.DataFrame) -> None:
    """Log basic dataset statistics."""
    logger.info("=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info(f"Total transactions: {len(df):,}")
    logger.info(f"Features: {df.shape[1] - 1}")
    logger.info(f"Fraudulent transactions: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    logger.info(f"Legitimate transactions: {(df['Class'] == 0).sum():,}")
    logger.info(f"Class imbalance ratio: 1:{int((df['Class'] == 0).sum() / df['Class'].sum())}")
    logger.info("=" * 50)


def load_data_for_prediction(
    data_path: str,
    has_labels: bool = False
) -> pd.DataFrame:
    """
    Load data for batch prediction.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing transaction data
    has_labels : bool
        If True, expects 'Class' column in the data
        
    Returns
    -------
    pd.DataFrame
        Loaded transaction data
    """
    logger.info(f"Loading transaction data from {data_path}")
    df = pd.read_csv(data_path)
    
    if has_labels and 'Class' in df.columns:
        logger.info(f"Loaded {len(df)} transactions with labels")
    else:
        logger.info(f"Loaded {len(df)} transactions for prediction")
    
    return df


# Pipeline components for sklearn Pipelines
class DataLoader:
    """SKLearn-compatible data loader."""
    
    def __init__(self, data_path: str, test_size: float = 0.3, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Fit the loader - no-op for this stateless loader."""
        return self
    
    def transform(self, X, y=None):
        """Transform loads and splits the data."""
        return load_credit_card_data(
            self.data_path,
            self.test_size,
            self.random_state
        )
    
    def fit_transform(self, X, y=None):
        """Fit and transform."""
        return self.fit(X, y).transform(X, y)