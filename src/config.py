"""
Configuration file for Credit Card Fraud Detection project.
Centralized place for all hyperparameters and settings.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")
METRICS_PATH = os.path.join(OUTPUTS_PATH, "metrics")
REPORTS_PATH = os.path.join(OUTPUTS_PATH, "reports")

# Ensure output directories exist
for path in [MODELS_PATH, OUTPUTS_PATH, FIGURES_PATH, METRICS_PATH, REPORTS_PATH]:
    os.makedirs(path, exist_ok=True)

# Data settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_SPLIT = True

# Class labels
LEGITIMATE_CLASS = 0
FRAUDULENT_CLASS = 1

# SMOTE settings (for handling imbalance)
SMOTE_RANDOM_STATE = 42
SAMPLING_STRATEGY = 0.5  # Oversample minority to 50% of majority

# Model hyperparameters
LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

# GridSearch settings
GRID_SEARCH_CV = 5
N_JOBS = -1

# Evaluation thresholds
THRESHOLD_FRAUD = 0.5

# Feature configuration
FEATURE_COLS = ['V' + str(i) for i in range(1, 29)] + ['Amount']  # V1-V28 + Amount
TARGET_COL = 'Class'
