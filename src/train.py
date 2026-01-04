
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data import load_dataframe
from src.preprocess import build_preprocess_pipeline

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'


def train(model_name: str = 'rf', mlflow_enabled: bool = False):
    df = load_dataframe()
    X = df.drop(columns=['target'])
    y = df['target']

    preprocess = build_preprocess_pipeline()

    if model_name == 'lr':
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline([('prep', preprocess), ('clf', clf)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)

    # CV and metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    y_pred = pipe.predict(X_test)
    proba = None
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        proba = np.zeros_like(y_pred, dtype=float)

    metrics = {
        'cv_roc_auc_mean': float(cv_auc.mean()),
        'cv_roc_auc_std': float(cv_auc.std()),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, proba)) if proba is not None else None,
        'model': model_name
    }

    # Save artifacts
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Optional MLflow logging
    if mlflow_enabled:
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_experiment('heart-disease')
            with mlflow.start_run(run_name=f'{model_name}-run'):
                mlflow.log_params({'model': model_name})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, 'model')
        except Exception as e:
            print(f"MLflow not available or failed: {e}")

    print("Saved:", MODEL_PATH, METRICS_PATH)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rf', 'lr'], default='rf')
    parser.add_argument('--mlflow', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=False)
    args = parser.parse_args()
    train(model_name=args.model, mlflow_enabled=args.mlflow)
