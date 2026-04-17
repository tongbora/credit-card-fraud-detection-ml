"""
Preprocessing module for Credit Card Fraud Detection project.
Handles feature engineering, scaling, train/test split, and class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from config import (
    TEST_SIZE, RANDOM_STATE, STRATIFY_SPLIT, 
    SMOTE_RANDOM_STATE, SAMPLING_STRATEGY, MODELS_PATH, FEATURE_COLS
)


def drop_duplicates(df):
    """
    Remove duplicate rows from dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Dataset with duplicates removed
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        print(f"✓ Dropped {removed} duplicate rows")
    else:
        print(f"✓ No duplicates found")
    
    return df_clean


def handle_missing_values(df):
    """
    Handle missing values (if any).
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    missing_count = df.isnull().sum().sum()
    
    if missing_count > 0:
        print(f"⚠ Found {missing_count} missing values")
        df = df.dropna()
        print(f"✓ Removed rows with missing values")
    else:
        print(f"✓ No missing values found")
    
    return df


def prepare_features_target(df, feature_cols, target_col='Class'):
    """
    Separate features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataset
        feature_cols (list): List of feature column names
        target_col (str): Name of target column
    
    Returns:
        tuple: (X, y) features and target
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"✓ Separated features ({X.shape[1]} cols) and target")
    
    return X, y


def scale_features(X_train, X_test, scaler=None, save_path=None):
    """
    Scale numerical features using StandardScaler.
    Fit on training data, apply to both train and test.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features
        X_test (pd.DataFrame or np.ndarray): Test features
        scaler (StandardScaler): Fitted scaler (if None, fit on train)
        save_path (str): Path to save scaler
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print(f"✓ Fitted scaler on training data")
    else:
        X_train_scaled = scaler.transform(X_train)
        print(f"✓ Applied existing scaler to training data")
    
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Scaled test features")
    
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"✓ Saved scaler to {save_path}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_test_split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train and test sets using stratification.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train/test split (stratified)")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Train fraud rate: {y_train.sum()/len(y_train)*100:.2f}%")
    print(f"  Test fraud rate: {y_test.sum()/len(y_test)*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, sampling_strategy=SAMPLING_STRATEGY, random_state=SMOTE_RANDOM_STATE):
    """
    Apply SMOTE to handle class imbalance on training data only.
    
    Args:
        X_train (np.ndarray or pd.DataFrame): Training features
        y_train (np.ndarray or pd.Series): Training target
        sampling_strategy (float): Ratio of minority to majority samples
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train_smote, y_train_smote)
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"✓ Applied SMOTE (sampling_strategy={sampling_strategy})")
    print(f"  After SMOTE - Training set: {X_train_smote.shape[0]} samples")
    print(f"  After SMOTE - Fraud rate: {y_train_smote.sum()/len(y_train_smote)*100:.2f}%")
    
    return X_train_smote, y_train_smote


def preprocess_pipeline(df, use_smote=True, scale=True, feature_cols=None, target_col='Class'):
    """
    Complete preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Raw dataset
        use_smote (bool): Whether to apply SMOTE
        scale (bool): Whether to scale features
        feature_cols (list): Feature columns to use
        target_col (str): Target column name
    
    Returns:
        dict: Dictionary with all preprocessed data and transformers
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Remove duplicates and handle missing values
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    
    # Step 2: Prepare features and target
    X, y = prepare_features_target(df, feature_cols, target_col)
    
    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Step 4: Scale features
    if scale:
        scaler_path = f"{MODELS_PATH}/scaler.pkl"
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, save_path=scaler_path
        )
    else:
        X_train_scaled, X_test_scaled, scaler = X_train, X_test, None
        print(f"✓ Skipped feature scaling")
    
    # Step 5: Handle class imbalance with SMOTE
    if use_smote:
        X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
        print(f"✓ Skipped SMOTE (using baseline imbalanced data)")
    
    print("="*60 + "\n")
    
    return {
        'X_train': X_train_final,
        'X_test': X_test_scaled,
        'y_train': y_train_final,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
