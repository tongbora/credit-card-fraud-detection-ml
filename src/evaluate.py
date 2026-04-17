"""
Evaluation module for Credit Card Fraud Detection project.
Includes functions for model evaluation and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
from config import METRICS_PATH


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        y_test (array-like): Test target
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': pr_auc
    }
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")
    print(f"{'='*60}\n")
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'metrics': metrics
    }


def compare_models(results_dict):
    """
    Create comparison table of all models.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and results as values
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = {}
    for model_name, result in results_dict.items():
        comparison_data[model_name] = result['metrics']
    
    df = pd.DataFrame(comparison_data).T
    df = df.round(4)
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    print(df)
    print(f"{'='*60}\n")
    
    return df


def get_best_model(results_dict, metric='roc_auc'):
    """
    Identify best model based on specified metric.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and results as values
        metric (str): Metric to use for comparison
    
    Returns:
        str: Name of best model
    """
    best_model = max(results_dict.items(), 
                    key=lambda x: x[1]['metrics'].get(metric, 0))
    return best_model[0]


def save_evaluation_report(results_dict, comparison_df, best_model, save_path=None):
    """
    Save evaluation report to file.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and results as values
        comparison_df (pd.DataFrame): Comparison table
        best_model (str): Name of best model
        save_path (str): Path to save report
    
    Returns:
        str: Report text
    """
    report = f"""
{'='*80}
CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT
{'='*80}

BEST MODEL: {best_model}

MODEL COMPARISON METRICS:
{comparison_df.to_string()}

{'='*80}
DETAILED METRICS BY MODEL:
{'='*80}

"""
    
    for model_name, result in results_dict.items():
        report += f"\n{model_name}:\n"
        metrics = result['metrics']
        report += f"  Accuracy:    {metrics['accuracy']:.4f}\n"
        report += f"  Precision:   {metrics['precision']:.4f}\n"
        report += f"  Recall:      {metrics['recall']:.4f}\n"
        report += f"  F1-Score:    {metrics['f1']:.4f}\n"
        report += f"  ROC-AUC:     {metrics['roc_auc']:.4f}\n"
        report += f"  Specificity: {metrics['specificity']:.4f}\n"
    
    report += f"\n{'='*80}\n"
    report += """
KEY INSIGHTS FOR FRAUD DETECTION:
- Recall is crucial: We want to catch as many fraud cases as possible
- Precision matters: We don't want too many false alarms
- ROC-AUC: Overall discrimination ability between fraud and legitimate
- For imbalanced data: Pay more attention to recall, F1, and ROC-AUC than accuracy
{'='*80}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {save_path}")
    
    return report
