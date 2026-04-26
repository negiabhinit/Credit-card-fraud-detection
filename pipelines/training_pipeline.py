"""
Training Pipeline for Credit Card Fraud Detection
==============================================
Orchestrates the full training workflow from data loading to model training and evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import yaml
import json
import pandas as pd
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_credit_card_data, load_data_for_prediction
from src.features.preprocessing import fraudDetectionPreprocessor, preprocess_data
from src.models.fraud_detector import (
    create_fraud_detector,
    train_and_evaluate,
    FraudDetectionModel,
    RandomForestFraudDetector,
    XGBoostFraudDetector
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline for fraud detection.
    
    Handles:
    - Data loading and splitting
    - Preprocessing
    - Model training with hyperparameter tuning
    - Evaluation and threshold optimization
    - Model persistence
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the training pipeline.
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessor = None
        self.metrics = {}
        self.feature_importance = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'raw_data_path': 'data/raw/creditcard.csv',
                'test_size': 0.3,
                'random_state': 42
            },
            'model': {
                'type': 'ensemble',
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'training': {
                'find_optimal_threshold': True
            }
        }
    
    def run(
        self,
        data_path: Optional[str] = None,
        model_type: Optional[str] = None,
        save_model: bool = True,
        save_reports: bool = True
    ) -> Tuple[FraudDetectionModel, Dict[str, Any]]:
        """
        Run the complete training pipeline.
        
        Parameters
        ----------
        data_path : str, optional
            Path to data file. If None, uses config path
        model_type : str, optional
            Type of model to train. If None, uses config
        save_model : bool
            Whether to save the trained model
        save_reports : bool
            Whether to save evaluation reports
            
        Returns
        -------
        Tuple[model, metrics]
            Trained model and evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading data...")
        data_path = data_path or self.config['data']['raw_data_path']
        X_train, X_test, y_train, y_test = self._load_data(data_path)
        
        # Step 2: Preprocess data
        logger.info("\n[Step 2/5] Preprocessing data...")
        X_train_processed, X_test_processed = self._preprocess_data(X_train, X_test)
        
        # Step 3: Train model
        logger.info("\n[Step 3/5] Training model...")
        model_type = model_type or self.config['model']['type']
        self.model = self._train_model(
            X_train_processed, y_train,
            X_test_processed, y_test,
            model_type
        )
        
        # Step 4: Evaluate model
        logger.info("\n[Step 4/5] Evaluating model...")
        self.metrics = self._evaluate_model(
            X_test_processed, y_test
        )
        
        # Step 5: Save artifacts
        logger.info("\n[Step 5/5] Saving artifacts...")
        if save_model:
            self._save_model()
        if save_reports:
            self._save_reports()
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Complete!")
        logger.info("=" * 60)
        
        return self.model, self.metrics
    
    def _load_data(self, data_path: str) -> Tuple:
        """Load and split data."""
        test_size = self.config['data'].get('test_size', 0.3)
        random_state = self.config['data'].get('random_state', 42)
        
        return load_credit_card_data(
            data_path=data_path,
            test_size=test_size,
            random_state=random_state
        )
    
    def _preprocess_data(self, X_train, X_test) -> Tuple:
        """Preprocess training and test data."""
        self.preprocessor = fraudDetectionPreprocessor(
            scale_amount=True,
            scale_time=True,
            add_temporal_features=True
        )
        
        # Fit on training data
        self.preprocessor.fit(X_train)
        
        # Transform both datasets
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Preprocessed data shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed
    
    def _train_model(
        self,
        X_train, y_train,
        X_test, y_test,
        model_type: str
    ) -> FraudDetectionModel:
        """Train the fraud detection model."""
        model_config = self.config['model']
        
        # Configure model parameters
        kwargs = {
            'random_forest': model_config.get('random_forest', {}),
            'xgboost': model_config.get('xgboost', {})
        }
        
        if model_type == 'ensemble':
            kwargs['weights'] = model_config.get('ensemble', {}).get('weights', [1, 2])
        
        # Train model
        model, metrics = train_and_evaluate(
            X_train, y_train,
            X_test, y_test,
            model_type=model_type,
            find_threshold=self.config['training'].get('find_optimal_threshold', True),
            **kwargs
        )
        
        self.metrics = metrics
        
        # Get feature importance
        self.feature_importance = model.get_feature_importance()
        
        return model
    
    def _evaluate_model(self, X_test, y_test) -> Dict:
        """Evaluate model performance."""
        # Use optimal threshold if available
        threshold = self.metrics.get('optimal_threshold', 0.5)
        metrics = self.model.evaluate(X_test, y_test, threshold=threshold)
        
        # Log key metrics
        logger.info("\n--- Model Performance ---")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"Fraud Catch Rate: {metrics.get('fraud_catch_rate', 0):.4f}")
        logger.info(f"Optimal Threshold: {threshold:.3f}")
        logger.info("------------------------\n")
        
        return metrics
    
    def _save_model(self) -> None:
        """Save trained model."""
        model_path = Path(self.config.get('paths', {}).get('models', 'models'))
        model_path.mkdir(parents=True, exist_ok=True)
        
        model_file = model_path / 'fraud_detector_model.joblib'
        self.model.save(str(model_file))
        
        # Save preprocessor
        if self.preprocessor:
            preprocessor_file = model_path / 'preprocessor.joblib'
            import joblib
            joblib.dump(self.preprocessor, preprocessor_file)
            logger.info(f"Preprocessor saved to {preprocessor_file}")
    
    def _save_reports(self) -> None:
        """Save evaluation reports."""
        reports_path = Path(self.config.get('paths', {}).get('reports', 'reports'))
        reports_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = reports_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_csv_file = reports_path / 'metrics.csv'
        metrics_df.to_csv(metrics_csv_file, index=False)
        logger.info(f"Metrics saved to {metrics_csv_file}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_file = reports_path / 'feature_importance.csv'
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Feature importance saved to {importance_file}")


def run_training_pipeline(
    data_path: str = "data/raw/creditcard.csv",
    config_path: str = "config/config.yaml",
    model_type: str = "ensemble"
) -> Tuple[FraudDetectionModel, Dict[str, Any]]:
    """
    Convenience function to run the training pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to the credit card fraud dataset
    config_path : str
        Path to configuration file
    model_type : str
        Type of model to train
        
    Returns
    -------
    Tuple[model, metrics]
        Trained model and evaluation metrics
    """
    pipeline = TrainingPipeline(config_path=config_path)
    return pipeline.run(data_path=data_path, model_type=model_type)


# Entry point
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    model, metrics = run_training_pipeline(
        data_path="data/raw/creditcard.csv",
        model_type="ensemble"
    )
    
    print("\n✅ Training complete!")
    print(f"Model: {model.model_name}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")