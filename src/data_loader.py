"""
Data loader module for Credit Card Fraud Detection project.
Handles loading and initial data exploration.
"""

import pandas as pd
import numpy as np
from src.config import DATA_PATH


def load_data():
    """
    Load credit card dataset from CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {DATA_PATH}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def get_dataset_info(df):
    """
    Print dataset information including shape, dtypes, missing values.
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        dict: Dictionary with dataset information
    """
    info = {
        'shape': df.shape,
        'rows': df.shape[0],
        'cols': df.shape[1],
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
    print(f"Data Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"Duplicate Rows: {info['duplicates']}")
    print("="*60 + "\n")
    
    return info


def get_class_distribution(df, target_col='Class'):
    """
    Get class distribution information.
    
    Args:
        df (pd.DataFrame): The dataset
        target_col (str): Name of target column
    
    Returns:
        dict: Class distribution statistics
    """
    class_dist = df[target_col].value_counts()
    class_pct = df[target_col].value_counts(normalize=True) * 100
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    print(f"Legitimate (0): {class_dist[0]:,} ({class_pct[0]:.2f}%)")
    print(f"Fraudulent  (1): {class_dist[1]:,} ({class_pct[1]:.2f}%)")
    print(f"Imbalance Ratio: {class_dist[0]/class_dist[1]:.2f}:1")
    print("="*60 + "\n")
    
    return {
        'legitimate': class_dist[0],
        'fraudulent': class_dist[1],
        'legitimate_pct': class_pct[0],
        'fraudulent_pct': class_pct[1],
        'imbalance_ratio': class_dist[0] / class_dist[1]
    }
