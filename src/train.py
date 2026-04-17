"""
Training module for Credit Card Fraud Detection project.
Includes functions for training, hyperparameter tuning, and full pipeline execution.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from time import time
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocess import (
    drop_duplicates,
    handle_missing_values,
    prepare_features_target,
    train_test_split_data,
    scale_features,
    apply_smote,
)
from evaluate import evaluate_model, compare_models
from utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
)
from config import (
    DATA_PATH,
    MODELS_PATH,
    FIGURES_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    FEATURE_COLS,
    TARGET_COL,
)


def train_logistic_regression(X_train, y_train, hyperparams=None):
    """
    Train Logistic Regression model.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        hyperparams (dict): Hyperparameters
    
    Returns:
        fitted model
    """
    if hyperparams is None:
        hyperparams = {'max_iter': 1000, 'random_state': RANDOM_STATE, 'n_jobs': -1}
    
    print(f"\n{'='*60}")
    print(f"TRAINING: Logistic Regression")
    print(f"{'='*60}")
    
    model = LogisticRegression(**hyperparams)
    start = time()
    model.fit(X_train, y_train)
    elapsed = time() - start
    
    print(f"✓ Training completed in {elapsed:.2f}s")
    print(f"  Coefficients shape: {model.coef_.shape}")
    
    return model


def train_random_forest(X_train, y_train, hyperparams=None):
    """
    Train Random Forest model.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        hyperparams (dict): Hyperparameters
    
    Returns:
        fitted model
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100, 'max_depth': 15,
            'random_state': RANDOM_STATE, 'n_jobs': -1
        }
    
    print(f"\n{'='*60}")
    print(f"TRAINING: Random Forest")
    print(f"{'='*60}")
    
    model = RandomForestClassifier(**hyperparams)
    start = time()
    model.fit(X_train, y_train)
    elapsed = time() - start
    
    print(f"✓ Training completed in {elapsed:.2f}s")
    print(f"  Trees: {model.n_estimators}")
    
    return model


def train_xgboost(X_train, y_train, hyperparams=None):
    """
    Train XGBoost model.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        hyperparams (dict): Hyperparameters
    
    Returns:
        fitted model
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100, 'max_depth': 6,
            'learning_rate': 0.1, 'random_state': RANDOM_STATE,
            'eval_metric': 'logloss'
        }
    
    print(f"\n{'='*60}")
    print(f"TRAINING: XGBoost")
    print(f"{'='*60}")
    
    model = xgb.XGBClassifier(**hyperparams)
    start = time()
    model.fit(X_train, y_train, verbose=False)
    elapsed = time() - start
    
    print(f"✓ Training completed in {elapsed:.2f}s")
    print(f"  Estimators: {model.n_estimators}")
    
    return model


def tune_logistic_regression(X_train, y_train, cv=5):
    """
    Hyperparameter tuning for Logistic Regression using GridSearchCV.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Cross-validation folds
    
    Returns:
        GridSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: Logistic Regression")
    print(f"{'='*60}")
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    
    start = time()
    grid_search.fit(X_train, y_train)
    elapsed = time() - start
    
    print(f"\n✓ Grid search completed in {elapsed:.2f}s")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_random_forest(X_train, y_train, cv=3, n_iter=10, tune_sample_size=None):
    """
    Hyperparameter tuning for Random Forest using RandomizedSearchCV.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Cross-validation folds
        n_iter (int): Number of sampled parameter settings
        tune_sample_size (int, optional): If provided, tune on a random subset
            of this many rows for faster experimentation
    
    Returns:
        RandomizedSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: Random Forest")
    print(f"{'='*60}")
    
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    X_tune, y_tune = X_train, y_train
    if tune_sample_size is not None and len(X_train) > tune_sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(X_train), size=tune_sample_size, replace=False)
        X_tune = X_train.iloc[sample_idx] if hasattr(X_train, 'iloc') else X_train[sample_idx]
        y_tune = y_train.iloc[sample_idx] if hasattr(y_train, 'iloc') else y_train[sample_idx]
        print(f"  Using subset for tuning: {len(X_tune):,} / {len(X_train):,} rows")
    
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    start = time()
    random_search.fit(X_tune, y_tune)
    elapsed = time() - start
    
    print(f"\n✓ Randomized search completed in {elapsed:.2f}s")
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score (F1): {random_search.best_score_:.4f}")

    if tune_sample_size is not None and len(X_train) > tune_sample_size:
        print("  Re-training best Random Forest on full training data...")
        best_model_full = clone(random_search.best_estimator_)
        best_model_full.fit(X_train, y_train)
        random_search.best_estimator_ = best_model_full
        print("  ✓ Full-data retraining complete")
    
    return random_search


def tune_xgboost(X_train, y_train, cv=3, n_iter=10, tune_sample_size=None):
    """
    Hyperparameter tuning for XGBoost using RandomizedSearchCV.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Cross-validation folds
        n_iter (int): Number of sampled parameter settings
        tune_sample_size (int, optional): If provided, tune on a random subset
            of this many rows for faster experimentation
    
    Returns:
        RandomizedSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: XGBoost")
    print(f"{'='*60}")
    
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
        , 'colsample_bytree': [0.8, 1.0]
    }

    X_tune, y_tune = X_train, y_train
    if tune_sample_size is not None and len(X_train) > tune_sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(X_train), size=tune_sample_size, replace=False)
        X_tune = X_train.iloc[sample_idx] if hasattr(X_train, 'iloc') else X_train[sample_idx]
        y_tune = y_train.iloc[sample_idx] if hasattr(y_train, 'iloc') else y_train[sample_idx]
        print(f"  Using subset for tuning: {len(X_tune):,} / {len(X_train):,} rows")
    
    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist'
    )
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    start = time()
    random_search.fit(X_tune, y_tune)
    elapsed = time() - start
    
    print(f"\n✓ Randomized search completed in {elapsed:.2f}s")
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score (F1): {random_search.best_score_:.4f}")

    if tune_sample_size is not None and len(X_train) > tune_sample_size:
        print("  Re-training best XGBoost on full training data...")
        best_model_full = clone(random_search.best_estimator_)
        best_model_full.fit(X_train, y_train, verbose=False)
        random_search.best_estimator_ = best_model_full
        print("  ✓ Full-data retraining complete")
    
    return random_search


def save_model(model, filename):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filename (str): Filename to save to
    """
    filepath = f"{MODELS_PATH}/{filename}"
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filename):
    """
    Load trained model from disk.
    
    Args:
        filename (str): Filename to load from
    
    Returns:
        Loaded model
    """
    filepath = f"{MODELS_PATH}/{filename}"
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


def _select_best_model(results_dict):
    """
    Select best model emphasizing fraud-detection metrics.

    Priority: F1 and Recall, with PR-AUC as tie support.
    """
    def score(item):
        metrics = item[1]['metrics']
        return (
            0.5 * metrics.get('f1', 0.0)
            + 0.35 * metrics.get('recall', 0.0)
            + 0.15 * metrics.get('pr_auc', 0.0)
        )

    best = max(results_dict.items(), key=score)
    return best[0], best[1]['model']


def run_training_pipeline():
    """
    End-to-end training pipeline entry point.
    Run with: python src/train.py
    """
    print("\nLoading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place creditcard.csv in the data/ folder."
        )
    df = load_data()

    print("\nPreprocessing data...")
    df = drop_duplicates(df)
    df = handle_missing_values(df)

    if 'Time' in df.columns and 'Time' not in FEATURE_COLS:
        print("✓ Consistent feature policy: using V1-V28 + Amount (excluding Time)")

    X, y = prepare_features_target(df, FEATURE_COLS, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    scaler_path = f"{MODELS_PATH}/scaler.pkl"
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, save_path=scaler_path
    )
    joblib.dump(FEATURE_COLS, f"{MODELS_PATH}/feature_cols.pkl")
    print(f"✓ Saved feature columns to {MODELS_PATH}/feature_cols.pkl")

    baseline_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
    }
    joblib.dump(baseline_data, f"{MODELS_PATH}/baseline_data.pkl")

    print("\nHandling imbalance with SMOTE on training data...")
    X_train_smt, y_train_smt = apply_smote(X_train_scaled, y_train)
    smote_data = {
        'X_train': X_train_smt,
        'X_test': X_test_scaled,
        'y_train': y_train_smt,
        'y_test': y_test,
    }
    joblib.dump(smote_data, f"{MODELS_PATH}/smote_data.pkl")
    print("Preprocessing complete")

    results = {}

    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train_smt, y_train_smt)
    save_model(lr_model, 'logistic_regression.pkl')
    lr_eval = evaluate_model(lr_model, X_test_scaled, y_test, model_name='Logistic Regression')
    lr_eval['model'] = lr_model
    results['Logistic Regression'] = lr_eval

    print("\nTraining Random Forest...")
    rf_search = tune_random_forest(
        X_train_smt,
        y_train_smt,
        cv=3,
        n_iter=10,
        tune_sample_size=50000,
    )
    rf_model = rf_search.best_estimator_
    save_model(rf_model, 'random_forest.pkl')
    rf_eval = evaluate_model(rf_model, X_test_scaled, y_test, model_name='Random Forest')
    rf_eval['model'] = rf_model
    results['Random Forest'] = rf_eval

    print("\nTraining XGBoost...")
    xgb_search = tune_xgboost(
        X_train_smt,
        y_train_smt,
        cv=3,
        n_iter=10,
        tune_sample_size=50000,
    )
    xgb_model = xgb_search.best_estimator_
    save_model(xgb_model, 'xgboost.pkl')
    xgb_eval = evaluate_model(xgb_model, X_test_scaled, y_test, model_name='XGBoost')
    xgb_eval['model'] = xgb_model
    results['XGBoost'] = xgb_eval

    print("\nSaving figures...")
    for model_name, result in results.items():
        safe_name = model_name.lower().replace(' ', '_')
        plot_confusion_matrix(
            y_test,
            result['predictions'],
            model_name=model_name,
            save_path=f"{FIGURES_PATH}/confusion_matrix_{safe_name}.png",
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, result in results.items():
        plot_roc_curve(y_test, result['probabilities'], model_name=model_name, ax=ax)
    ax.set_title('ROC Curve Comparison')
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/roc_curve_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, result in results.items():
        plot_precision_recall_curve(y_test, result['probabilities'], model_name=model_name, ax=ax)
    ax.set_title('Precision-Recall Curve Comparison')
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/pr_curve_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    plot_feature_importance(
        rf_model,
        FEATURE_COLS,
        model_name='Random Forest',
        save_path=f"{FIGURES_PATH}/feature_importance_rf.png",
        top_n=20,
    )
    plot_feature_importance(
        xgb_model,
        FEATURE_COLS,
        model_name='XGBoost',
        save_path=f"{FIGURES_PATH}/feature_importance_xgboost.png",
        top_n=20,
    )

    comparison_df = compare_models(results)
    comparison_csv = f"{METRICS_PATH}/model_comparison.csv"
    comparison_json = f"{METRICS_PATH}/model_comparison.json"

    print("\nSaving metrics...")
    comparison_df.to_csv(comparison_csv)
    comparison_df.to_json(comparison_json, orient='index', indent=2)
    print(f"✓ Saved metrics CSV to {comparison_csv}")
    print(f"✓ Saved metrics JSON to {comparison_json}")

    best_model_name, best_model = _select_best_model(results)
    save_model(best_model, 'best_model.pkl')
    print(f"Best model: {best_model_name}")

    print("\nTraining pipeline complete")
    return {
        'results': results,
        'comparison': comparison_df,
        'best_model_name': best_model_name,
    }


if __name__ == "__main__":
    run_training_pipeline()
