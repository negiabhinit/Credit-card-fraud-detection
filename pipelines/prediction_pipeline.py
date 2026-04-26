"""
Batch Prediction Pipeline for Credit Card Fraud Detection
=====================================================
Handles batch prediction for fraud detection on historical transaction data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import yaml
import json
import pandas as pd
import numpy as np
import joblib

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data_for_prediction
from src.features.preprocessing import fraudDetectionPreprocessor

logger = logging.getLogger(__name__)


class BatchPredictionPipeline:
    """
    Batch prediction pipeline for fraud detection.
    
    Processes large datasets in chunks for memory efficiency
    and generates fraud predictions with confidence scores.
    """
    
    def __init__(
        self,
        model_path: str = "models/fraud_detector_model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        config_path: str = "config/config.yaml",
        threshold: float = 0.5
    ):
        """
        Initialize the batch prediction pipeline.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file
        preprocessor_path : str
            Path to the preprocessor file
        config_path : str
            Path to configuration file
        threshold : float
            Decision threshold for fraud classification
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config = self._load_config(config_path)
        self.threshold = threshold
        
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def load(self) -> 'BatchPredictionPipeline':
        """
        Load the trained model and preprocessor.
        
        Returns
        -------
        self
        """
        logger.info("Loading model and preprocessor...")
        
        # Load model
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load preprocessor
        try:
            preprocessor_path = Path(self.preprocessor_path)
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
            else:
                logger.warning(f"Preprocessor not found: {self.preprocessor_path}, using raw data")
                self.preprocessor = None
        except Exception as e:
            logger.warning(f"Failed to load preprocessor: {e}")
            self.preprocessor = None
        
        self.is_loaded = True
        return self
    
    def predict(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        chunk_size: int = 10000,
        save_scores: bool = True
    ) -> pd.DataFrame:
        """
        Run batch prediction on transaction data.
        
        Parameters
        ----------
        data_path : str
            Path to input CSV file
        output_path : str, optional
            Path to save results. If None, returns DataFrame
        chunk_size : int
            Number of rows to process at a time
        save_scores : bool
            Whether to save prediction scores
            
        Returns
        -------
        pd.DataFrame
            Prediction results with fraud probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load() first.")
        
        logger.info(f"Loading transaction data from {data_path}")
        
        # Load data
        df = load_data_for_prediction(data_path, has_labels=False)
        total_transactions = len(df)
        
        logger.info(f"Processing {total_transactions:,} transactions in chunks of {chunk_size:,}...")
        
        # Process in chunks
        all_results = []
        
        for i in range(0, total_transactions, chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            
            # Preprocess
            if self.preprocessor:
                chunk_processed = self.preprocessor.transform(chunk)
            else:
                chunk_processed = chunk
            
            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                scores = self.model.predict_proba(chunk_processed)[:, 1]
            else:
                scores = self.model.predict(chunk_processed)
            
            # Create results
            results = pd.DataFrame({
                'fraud_probability': scores,
                'is_fraud': (scores >= self.threshold).astype(int)
            }, index=chunk.index)
            
            # Add original data if requested
            if save_scores:
                results = pd.concat([chunk, results], axis=1)
            
            all_results.append(results)
            
            # Progress logging
            processed = min(i + chunk_size, total_transactions)
            progress = processed / total_transactions * 100
            logger.info(f"Progress: {processed:,}/{total_transactions:,} ({progress:.1f}%)")
        
        # Combine all results
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Generate summary
        n_fraud = results_df['is_fraud'].sum()
        fraud_rate = results_df['is_fraud'].mean()
        
        logger.info("=" * 50)
        logger.info("Batch Prediction Summary")
        logger.info("=" * 50)
        logger.info(f"Total transactions: {total_transactions:,}")
        logger.info(f"Predicted frauds: {n_fraud:,} ({fraud_rate*100:.2f}%)")
        logger.info(f"Threshold used: {self.threshold:.3f}")
        logger.info("=" * 50)
        
        # Save results
        if output_path:
            self._save_results(results_df, output_path)
        
        return results_df
    
    def predict_with_labels(
        self,
        data_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run prediction with evaluation metrics (when labels are available).
        
        Parameters
        ----------
        data_path : str
            Path to input CSV file (must include 'Class' column)
        output_path : str, optional
            Path to save results
            
        Returns
        -------
        pd.DataFrame
            Prediction results with evaluation metrics
        """
        logger.info("Loading transaction data with labels...")
        df = load_data_for_prediction(data_path, has_labels=True)
        
        if 'Class' not in df.columns:
            raise ValueError("'Class' column not found in data")
        
        # Separate features and labels
        X = df.drop('Class', axis=1)
        y_true = df['Class']
        
        # Preprocess
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Get predictions
        scores = self.model.predict_proba(X_processed)[:, 1]
        y_pred = (scores >= self.threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores),
            'pr_auc': average_precision_score(y_true, scores)
        }
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        logger.info("\n--- Evaluation Metrics ---")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
        logger.info("-----------------------\n")
        
        # Create results DataFrame
        results_df = pd.concat([
            X,
            pd.Series(y_true, name='true_class'),
            pd.Series(scores, name='fraud_probability'),
            pd.Series(y_pred, name='predicted_class')
        ], axis=1)
        
        if output_path:
            self._save_results(results_df, output_path)
        
        # Save metrics
        metrics_path = Path(output_path).parent / 'batch_metrics.json' if output_path else None
        if metrics_path:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        """Save prediction results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if output_file.suffix == '.csv':
            results_df.to_csv(output_path, index=False)
        elif output_file.suffix == '.parquet':
            results_df.to_parquet(output_path, index=False)
        elif output_file.suffix == '.json':
            results_df.to_json(output_path, orient='records', lines=True)
        else:
            # Default to CSV
            results_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update decision threshold.
        
        Parameters
        ----------
        threshold : float
            New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Decision threshold updated to {threshold:.3f}")
    
    def optimize_threshold(
        self,
        X,
        y_true,
        metric: str = 'f1'
    ) -> float:
        """
        Optimize threshold using validation data.
        
        Parameters
        ----------
        X : array-like
            Features
        y_true : array-like
            True labels
        metric : str
            Metric to optimize
            
        Returns
        -------
        float
            Optimal threshold
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Preprocess
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        # Get scores
        scores = self.model.predict_proba(X_processed)[:, 1]
        
        # Find optimal threshold
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_score = 0
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        logger.info(f"Optimal threshold ({metric}): {best_threshold:.3f}")
        self.threshold = best_threshold
        
        return best_threshold


def run_batch_prediction(
    data_path: str,
    model_path: str = "models/fraud_detector_model.joblib",
    output_path: str = "reports/predictions.csv",
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Convenience function to run batch prediction.
    
    Parameters
    ----------
    data_path : str
        Path to input transaction data
    model_path : str
        Path to trained model
    output_path : str
        Path to save predictions
    threshold : float
        Decision threshold
        
    Returns
    -------
    pd.DataFrame
        Prediction results
    """
    pipeline = BatchPredictionPipeline(
        model_path=model_path,
        threshold=threshold
    )
    
    pipeline.load()
    results = pipeline.predict(data_path, output_path)
    
    return results


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run batch prediction
    results = run_batch_prediction(
        data_path="data/raw/creditcard.csv",
        model_path="models/fraud_detector_model.joblib",
        output_path="reports/predictions.csv"
    )
    
    print(f"\n✅ Batch prediction complete!")
    print(f"Results shape: {results.shape}")