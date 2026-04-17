"""
Prediction module for Credit Card Fraud Detection project.
Includes functions for making predictions and preparing data for predictions.
"""

import numpy as np
import pandas as pd
import joblib
from config import MODELS_PATH


class FraudDetectionPredictor:
    """
    Class for making fraud detection predictions using trained models.
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize predictor with model and scaler.
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_name = model_path.split('/')[-1].replace('.pkl', '')
    
    def predict(self, features):
        """
        Make prediction for a single transaction.
        
        Args:
            features (dict or list): Transaction features
        
        Returns:
            dict: Prediction result
        """
        if isinstance(features, dict):
            features = np.array(list(features.values())).reshape(1, -1)
        else:
            features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        result = {
            'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'fraud_probability': probability,
            'model': self.model_name,
            'confidence': max(self.model.predict_proba(features_scaled)[0])
        }
        
        return result
    
    def batch_predict(self, dataframe):
        """
        Make predictions for multiple transactions.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with transaction features
        
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        features = dataframe.values
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        result_df = dataframe.copy()
        result_df['Prediction'] = predictions
        result_df['Fraud_Probability'] = probabilities
        result_df['Prediction_Label'] = result_df['Prediction'].apply(
            lambda x: 'Fraudulent' if x == 1 else 'Legitimate'
        )
        
        return result_df


def create_sample_transaction(feature_values=None):
    """
    Create a sample transaction for UI demonstration.
    
    Args:
        feature_values (dict): Custom feature values (optional)
    
    Returns:
        dict: Sample transaction data
    """
    if feature_values is None:
        # Typical legitimate transaction
        feature_values = {
            'V1': -1.358, 'V2': -0.043, 'V3': 2.136, 'V4': 1.465,
            'V5': -0.619, 'V6': -0.991, 'V7': -0.305, 'V8': 0.085,
            'V9': 0.159, 'V10': -0.046, 'V11': -0.073, 'V12': -0.268,
            'V13': -0.539, 'V14': -0.055, 'V15': 0.040, 'V16': 0.085,
            'V17': -0.255, 'V18': -0.171, 'V19': -0.046, 'V20': -0.351,
            'V21': -0.148, 'V22': -0.420, 'V23': 0.048, 'V24': 0.102,
            'V25': 0.191, 'V26': -0.328, 'V27': 0.047, 'V28': 0.005,
            'Amount': 149.62
        }
    
    return feature_values


def validate_transaction_features(features, expected_features):
    """
    Validate that transaction has all required features.
    
    Args:
        features (dict): Transaction features
        expected_features (list): Expected feature names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(features, dict):
        return False, "Features must be a dictionary"
    
    missing = set(expected_features) - set(features.keys())
    if missing:
        return False, f"Missing features: {missing}"
    
    try:
        for feat, val in features.items():
            if feat in expected_features:
                float(val)
    except (ValueError, TypeError):
        return False, "All feature values must be numeric"
    
    return True, ""
