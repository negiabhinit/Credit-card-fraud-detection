# Credit Card Fraud Detection ML System

A comprehensive machine learning system for detecting credit card fraud using batch prediction on historical transaction data.

## 📋 Overview

This project implements:
- **Data Pipeline**: Load and preprocess credit card transaction data from Kaggle dataset
- **ML Models**: Random Forest, XGBoost, and Ensemble methods with imbalance handling
- **Training Pipeline**: End-to-end training with cross-validation and hyperparameter tuning
- **Batch Prediction**: Process large transaction datasets efficiently
- **Monitoring**: Performance tracking, drift detection, and alerts

## 🏗️ Project Structure

```
fraud-detection-ml-project/
├── config/
│   └── config.yaml           # Project configuration
├── data/
│   ├── raw/               # Raw data storage
│   └── processed/         # Processed data
├── models/
│   ├── checkpoints/      # Model checkpoints
│   └── fraud_detector_model.joblib
├── pipelines/
│   ├── training_pipeline.py    # Training orchestration
│   └── prediction_pipeline.py # Batch prediction
├── reports/
│   ├── figures/          # Visualizations
│   └── metrics/         # Performance metrics
├── src/
│   ├── data/
│   │   └── loader.py       # Data loading
│   ├── features/
│   │   └── preprocessing.py  # Feature engineering
│   ├── models/
│   │   └── fraud_detector.py  # ML models
│   └── monitoring/
│       └── monitor.py        # Monitoring & alerts
├── tests/
│   └── test_fraud_detection.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd fraud-detection-ml-project
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Credit Card Fraud dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Save to: `data/raw/creditcard.csv`

### 3. Train Model

```python
from pipelines.training_pipeline import run_training_pipeline

model, metrics = run_training_pipeline(
    data_path="data/raw/creditcard.csv",
    model_type="ensemble"  # or "random_forest", "xgboost"
)

print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### 4. Run Batch Prediction

```python
from pipelines.prediction_pipeline import run_batch_prediction

results = run_batch_prediction(
    data_path="data/raw/creditcard.csv",
    output_path="reports/predictions.csv",
    threshold=0.5
)
```

## 📊 Key Features

### Imbalanced Data Handling
- Uses `class_weight='balanced'` for Random Forest
- Auto-calculates `scale_pos_weight` for XGBoost
- Ensemble with weighted voting

### Threshold Optimization
- Automatically finds optimal threshold for F1 score
- Supports optimization for precision/recall/business metrics

### Model Evaluation
- Standard metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Business metrics: Fraud Catch Rate, False Alarm Rate

## 📈 Performance

Expected results on the Kaggle Credit Card Fraud dataset:
- ROC-AUC: > 0.95
- PR-AUC: > 0.80
- F1 Score: > 0.80
- Recall: > 0.80

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  type: "ensemble"  # random_forest, xgboost, ensemble
  
training:
  n_folds: 5
  tuning:
    enabled: true
    n_trials: 20
    
evaluation:
  business_metrics:
    fraud_catch_rate: 0.8
    false_alarm_rate: 0.1
```

## ��� Testing

Run tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_fraud_detection.py::TestRandomForestFraudDetector -v
```

## 📖 Documentation

### Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Time | float | Seconds since first transaction |
| V1-V28 | float | PCA-transformed features |
| Amount | float | Transaction amount |
| Class | int | 0=legitimate, 1=fraud |

### API Usage

```python
# Load and preprocess data
from src.data.loader import load_credit_card_data
from src.features.preprocessing import FraudDetectionPreprocessor

X_train, X_test, y_train, y_test = load_credit_card_data("data/raw/creditcard.csv")
preprocessor = FraudDetectionPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)

# Train model
from src.models.fraud_detector import RandomForestFraudDetector
model = RandomForestFraudDetector(n_estimators=100)
model.fit(X_train_processed, y_train)

# Predict
scores = model.predict_scores(X_test)
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

## 📝 License

MIT License

## 🤝 Credits

Dataset: Machine Learning Group - ULB
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud