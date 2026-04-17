"""
Utility functions for Credit Card Fraud Detection project.
Includes metrics computation, visualization, and helper functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, accuracy_score
)
from config import FIGURES_PATH, METRICS_PATH


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_pred_proba (array-like): Predicted probabilities
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
    
    return metrics


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, 
                                target_names=['Legitimate', 'Fraudulent']))
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
        save_path (str): Path to save figure
    
    Returns:
        np.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.close()
    return cm


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None, ax=None):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        model_name (str): Name of the model
        save_path (str): Path to save figure
        ax (matplotlib.axes.Axes): Axes to plot on
    
    Returns:
        tuple: (fpr, tpr, roc_auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path and ax is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
        plt.close()
    
    return fpr, tpr, roc_auc


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", save_path=None, ax=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        model_name (str): Name of the model
        save_path (str): Path to save figure
        ax (matplotlib.axes.Axes): Axes to plot on
    
    Returns:
        tuple: (precision, recall, pr_auc)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path and ax is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PR curve to {save_path}")
        plt.close()
    
    return precision, recall, pr_auc


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot class distribution bar chart.
    
    Args:
        y (array-like): Target variable
        title (str): Plot title
        save_path (str): Path to save figure
    """
    class_counts = pd.Series(y).value_counts().sort_index()
    class_pct = pd.Series(y).value_counts(normalize=True).sort_index() * 100
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, 
                  color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add count and percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, class_pct.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved class distribution to {save_path}")
    
    plt.close()


def plot_feature_importance(model, feature_names, model_name="Model", save_path=None, top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        model_name (str): Name of the model
        save_path (str): Path to save figure
        top_n (int): Number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠ Model {model_name} does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(top_features)), top_importances, color='#3498db', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance to {save_path}")
    
    plt.close()


def create_metrics_comparison_table(results_dict, save_path=None):
    """
    Create and save a comparison table of model metrics.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics dicts as values
        save_path (str): Path to save as CSV
    
    Returns:
        pd.DataFrame: Comparison table
    """
    df_comparison = pd.DataFrame(results_dict).T
    df_comparison = df_comparison.round(4)
    
    if save_path:
        df_comparison.to_csv(save_path)
        print(f"✓ Saved metrics comparison to {save_path}")
    
    return df_comparison


def plot_metrics_comparison(results_dict, metrics_to_plot=None, save_path=None):
    """
    Plot comparison of metrics across models.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics dicts as values
        metrics_to_plot (list): Which metrics to plot
        save_path (str): Path to save figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    df = pd.DataFrame(results_dict).T
    df = df[df.columns.intersection(metrics_to_plot)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison plot to {save_path}")
    
    plt.close()
