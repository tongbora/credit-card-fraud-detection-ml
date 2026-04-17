"""
Training module for Credit Card Fraud Detection project.
Includes functions for training and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import joblib
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from config import MODELS_PATH, RANDOM_STATE


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


def tune_random_forest(X_train, y_train, cv=5):
    """
    Hyperparameter tuning for Random Forest using GridSearchCV.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Cross-validation folds
    
    Returns:
        GridSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: Random Forest")
    print(f"{'='*60}")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8]
    }
    
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    
    start = time()
    grid_search.fit(X_train, y_train)
    elapsed = time() - start
    
    print(f"\n✓ Grid search completed in {elapsed:.2f}s")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_xgboost(X_train, y_train, cv=5):
    """
    Hyperparameter tuning for XGBoost using GridSearchCV.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Cross-validation folds
    
    Returns:
        GridSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: XGBoost")
    print(f"{'='*60}")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    
    start = time()
    grid_search.fit(X_train, y_train)
    elapsed = time() - start
    
    print(f"\n✓ Grid search completed in {elapsed:.2f}s")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search


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
