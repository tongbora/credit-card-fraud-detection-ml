"""
Credit Card Fraud Detection - Gradio Web Interface
Interactive dashboard for model demonstration and predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import gradio as gr
import joblib
from PIL import Image

# Add src to path
sys.path.insert(0, './src')
from predict import FraudDetectionPredictor, create_sample_transaction, validate_transaction_features
from config import MODELS_PATH, FIGURES_PATH, FEATURE_COLS, METRICS_PATH

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_comparison_data():
    """Load model comparison metrics and visualizations."""
    try:
        comparison_df = pd.read_csv(f'{METRICS_PATH}/model_comparison.csv', index_col=0)
        return comparison_df
    except:
        return None

def load_visualization(filename):
    """Load and return visualization image."""
    path = f'{FIGURES_PATH}/{filename}'
    if os.path.exists(path):
        return Image.open(path)
    return None

def make_prediction(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount):
    """Make a single prediction.Expects 28 V features plus Amount."""
    try:
        # Create feature dictionary
        features = {
            'V1': float(v1), 'V2': float(v2), 'V3': float(v3), 'V4': float(v4),
            'V5': float(v5), 'V6': float(v6), 'V7': float(v7), 'V8': float(v8),
            'V9': float(v9), 'V10': float(v10), 'V11': float(v11), 'V12': float(v12),
            'V13': float(v13), 'V14': float(v14), 'V15': float(v15), 'V16': float(v16),
            'V17': float(v17), 'V18': float(v18), 'V19': float(v19), 'V20': float(v20),
            'V21': float(v21), 'V22': float(v22), 'V23': float(v23), 'V24': float(v24),
            'V25': float(v25), 'V26': float(v26), 'V27': float(v27), 'V28': float(v28),
            'Amount': float(amount)
        }
        
        # Load best model (XGBoost)
        predictor = FraudDetectionPredictor(f'{MODELS_PATH}/xgboost.pkl', f'{MODELS_PATH}/scaler.pkl')
        result = predictor.predict(features)
        
        # Format result
        prediction_text = f"""
        <div style="text-align: center; padding: 20px; background-color: {'#ffe74c' if result['prediction'] == 'Fraudulent' else '#91e8c4'}; border-radius: 10px;">
            <h2>Prediction: <b>{result['prediction']}</b></h2>
            <p>Fraud Probability: <b>{result['fraud_probability']:.2%}</b></p>
            <p>Confidence: <b>{result['confidence']:.2%}</b></p>
            <p><small>Model: {result['model']}</small></p>
        </div>
        """
        
        return prediction_text
    except Exception as e:
        return f"<div style='color: red; padding: 10px;'><b>Error:</b> {str(e)}</div>"

def process_batch_predictions(file):
    """Process batch predictions from CSV upload."""
    try:
        df = pd.read_csv(file.name)
        
        # Validate columns
        required_cols = FEATURE_COLS
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return None, f"❌ Missing columns: {missing_cols}"
        
        # Make predictions
        predictor = FraudDetectionPredictor(f'{MODELS_PATH}/xgboost.pkl', f'{MODELS_PATH}/scaler.pkl')
        result_df = predictor.batch_predict(df[required_cols])
        
        # Count predictions
        fraud_count = (result_df['Prediction'] == 1).sum()
        legitimate_count = (result_df['Prediction'] == 0).sum()
        
        summary = f"""
        ✓ Processed {len(result_df)} transactions
        - Fraudulent: {fraud_count} ({fraud_count/len(result_df)*100:.1f}%)
        - Legitimate: {legitimate_count} ({legitimate_count/len(result_df)*100:.1f}%)
        """
        
        return result_df, summary
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==============================================================================
# BUILD GRADIO INTERFACE
# ==============================================================================

with gr.Blocks(title="Credit Card Fraud Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔐 Credit Card Fraud Detection System
        ### Final Year Machine Learning Project
        
        An end-to-end machine learning system for detecting fraudulent credit card transactions using advanced algorithms and SMOTE-based resampling.
        """
    )
    
    # ==============================================================================
    # TAB 1: PROJECT OVERVIEW
    # ==============================================================================
    
    with gr.Tab("📊 Project Overview"):
        gr.Markdown("""
        ## About This Project
        
        **Objective:** Build a machine learning model to detect fraudulent credit card transactions in real-time.
        
        ### Dataset Overview
        - **Total Transactions:** 284,807
        - **Fraudulent Cases:** 492 (0.17%)
        - **Features:** 28 PCA-transformed features (V1-V28) + Amount + Time
        - **Class Distribution:** Highly imbalanced (99.83% legitimate)
        
        ### Problem Type
        **Binary Classification with Severe Class Imbalance**
        
        ### Why This Matters
        - Financial institutions need to identify fraud while minimizing false alarms
        - Missing frauds (False Negatives) → Direct financial loss
        - False alarms (False Positives) → Customer frustration
        - Traditional accuracy metrics are misleading on imbalanced data
        
        ### Solution Approach
        1. **Preprocessing:** Feature scaling + SMOTE resampling
        2. **Modeling:** Logistic Regression, Random Forest, XGBoost
        3. **Evaluation:** Recall, Precision, F1-Score, ROC-AUC, PR-AUC
        4. **Best Model:** XGBoost (selected for superior fraud detection capability)
        
        ### Why These Models?
        - **Logistic Regression:** Fast baseline with interpretability
        - **Random Forest:** Handles non-linearity with feature importance
        - **XGBoost:** State-of-the-art gradient boosting with imbalance handling
        """)
    
    # ==============================================================================
    # TAB 2: MODEL COMPARISON
    # ==============================================================================
    
    with gr.Tab("📈 Model Comparison"):
        gr.Markdown("## Model Performance Comparison")
        
        comparison_df = load_comparison_data()
        if comparison_df is not None:
            gr.DataFrame(comparison_df, label="Performance Metrics")
        else:
            gr.Markdown("⚠️ Metrics not yet generated. Run training notebooks first.")
        
        with gr.Row():
            gr.Markdown("### Visualizations:")
        
        with gr.Row():
            img_comparison = load_visualization('model_comparison_summary.png')
            if img_comparison:
                gr.Image(img_comparison, label="Model Comparison Summary")
        
        with gr.Row():
            img_class_dist = load_visualization('01_class_distribution.png')
            if img_class_dist:
                gr.Image(img_class_dist, label="Class Distribution")
        
        with gr.Row():
            img_amount = load_visualization('02_amount_distribution.png')
            if img_amount:
                gr.Image(img_amount, label="Transaction Amount Distribution")
        
        gr.Markdown("""
        ### Key Insights
        
        **Best Model: XGBoost**
        
        ✨ **Why XGBoost?**
        - Highest ROC-AUC score for overall discrimination
        - Excellent recall (catches most frauds)
        - Good precision (minimizes false alarms)
        - Strong F1-score balancing both metrics
        
        **For Fraud Detection, We Prioritize:**
        1. **Recall** - Don't miss frauds (catch as many as possible)
        2. **Precision** - Minimize false alarms (customer experience)
        3. **ROC-AUC** - Overall model quality
        """)
    
    # ==============================================================================
    # TAB 3: SINGLE PREDICTION
    # ==============================================================================
    
    with gr.Tab("🎯 Single Prediction"):
        gr.Markdown("""
        ## Predict for a Single Transaction
        
        Enter transaction features (V1-V28 are PCA-transformed, pre-scaled features, Amount is in dollars).
        
        **Sample values provided** - you can modify them to test different scenarios.
        """)
        
        # Create inputs for all 28 V features
        v_inputs = []
        with gr.Row():
            for i in range(1, 5):
                v_inputs.append(gr.Number(label=f"V{i}", value=-1.3 if i == 1 else -0.04))
        
        with gr.Row():
            for i in range(5, 9):
                v_inputs.append(gr.Number(label=f"V{i}", value=-0.6))
        
        with gr.Row():
            for i in range(9, 13):
                v_inputs.append(gr.Number(label=f"V{i}", value=0.05))
        
        with gr.Row():
            for i in range(13, 17):
                v_inputs.append(gr.Number(label=f"V{i}", value=-0.3))
        
        with gr.Row():
            for i in range(17, 21):
                v_inputs.append(gr.Number(label=f"V{i}", value=-0.1))
        
        with gr.Row():
            for i in range(21, 25):
                v_inputs.append(gr.Number(label=f"V{i}", value=-0.1))
        
        with gr.Row():
            for i in range(25, 29):
                v_inputs.append(gr.Number(label=f"V{i}", value=0.1))
        
        with gr.Row():
            amount_input = gr.Number(label="Amount ($)", value=149.62)
        
        with gr.Row():
            predict_btn = gr.Button("🔍 Predict", variant="primary")
            clear_btn = gr.Button("🔄 Reset")
        
        output_pred = gr.HTML(label="Prediction Result")
        
        # Connect predict button
        predict_btn.click(
            fn=make_prediction,
            inputs=v_inputs + [amount_input],
            outputs=output_pred
        )
    
    # ==============================================================================
    # TAB 4: BATCH PREDICTION
    # ==============================================================================
    
    with gr.Tab("📁 Batch Prediction"):
        gr.Markdown("""
        ## Batch Predictions from CSV
        
        Upload a CSV file with columns: V1, V2, ..., V28, Amount
        
        The model will predict on all rows and you can download the results.
        """)
        
        with gr.Row():
            csv_input = gr.File(label="Upload CSV File", file_types=['.csv'])
            process_batch_btn = gr.Button("⚙️ Process", variant="primary")
        
        batch_output_text = gr.Textbox(label="Summary", interactive=False)
        batch_output_df = gr.Dataframe(label="Predictions", interactive=False)
        
        with gr.Row():
            download_btn = gr.Button("⬇️ Download as CSV (Right-click table)")
        
        def batch_predict_wrapper(file):
            if file is None:
                return "", None
            df, summary = process_batch_predictions(file)
            return summary, df
        
        process_batch_btn.click(
            fn=batch_predict_wrapper,
            inputs=csv_input,
            outputs=[batch_output_text, batch_output_df]
        )
    
    # ==============================================================================
    # TAB 5: EDA & INSIGHTS
    # ==============================================================================
    
    with gr.Tab("🔬 EDA & Insights"):
        gr.Markdown("""
        ## Exploratory Data Analysis Findings
        
        ### Class Imbalance Problem
        """)
        
        img_class = load_visualization('01_class_distribution.png')
        if img_class:
            gr.Image(img_class, label="Class Distribution")
        
        gr.Markdown("""
        - Only **0.17%** of transactions are fraudulent
        - **99.83%** are legitimate
        - A naive model predicting "all legitimate" achieves high accuracy!
        - **Solution:** Don't use accuracy; use Recall, Precision, F1, ROC-AUC
        
        ### Transaction Amount Patterns
        """)
        
        img_amount = load_visualization('02_amount_distribution.png')
        if img_amount:
            gr.Image(img_amount, label="Amount Distribution")
        
        gr.Markdown("""
        - Fraudulent transactions vary widely in amount
        - No single amount threshold can identify fraud
        - Legitimate transactions also span large range
        
        ### Feature Importance
        """)
        
        img_rf_imp = load_visualization('feature_importance_rf.png')
        if img_rf_imp:
            gr.Image(img_rf_imp, label="Random Forest Feature Importance")
        
        img_xgb_imp = load_visualization('feature_importance_xgb.png')
        if img_xgb_imp:
            gr.Image(img_xgb_imp, label="XGBoost Feature Importance")
        
        gr.Markdown("""
        - V4, V12, V14 are among the most predictive features
        - Different features matter for fraud vs. legitimate
        - Complex interactions captured by tree-based models
        """)
    
    # ==============================================================================
    # TAB 6: PRESENTATION HELPER
    # ==============================================================================
    
    with gr.Tab("📝 Presentation Helper"):
        gr.Markdown("""
        ## Key Points for Your Presentation
        
        ### 1. Problem Statement
        - **Challenge:** Detect fraudulent credit card transactions in real-time
        - **Imbalance:** 0.17% fraud rate makes this challenging
        - **Objective:** Maximize fraud detection while minimizing false alarms
        
        ### 2. Preprocessing & Imbalance Handling
        - **Data Cleaning:** Removed duplicates, checked missing values
        - **Feature Scaling:** StandardScaler on all 29 features
        - **Train/Test Split:** Stratified to preserve imbalance in both sets
        - **SMOTE:** Applied only on training set to avoid data leakage
          - Oversampled minority class to 50% balance
          - Kept test set untouched for unbiased evaluation
        
        ### 3. Models Trained
        1. **Logistic Regression** (Baseline)
           - Fast, interpretable, provides baselines
           
        2. **Random Forest**
           - Handles non-linearity, shows feature importance
           
        3. **XGBoost** (Best Model)
           - State-of-the-art gradient boosting
           - Built-in imbalance handling
        
        ### 4. Hyperparameter Tuning
        - Used GridSearchCV with 5-fold cross-validation
        - Optimized for ROC-AUC (best overall metric for imbalanced data)
        - Practical parameter ranges to avoid overfitting
        
        ### 5. Evaluation Metrics
        - **Accuracy:** Misleading on imbalanced data
        - **Recall:** "Of all frauds, how many did we catch?" → Priority!
        - **Precision:** "Of our fraud predictions, how many are correct?"
        - **F1-Score:** Harmonic mean of Recall & Precision
        - **ROC-AUC:** Overall discrimination ability
        
        ### 6. Results Summary
        - **Best Model:** XGBoost
        - **Why?** Highest ROC-AUC, excellent recall, good precision
        - **Trade-off:** Minimal false positives while catching most fraud
        
        ### 7. Key Takeaways
        - ✅ Class imbalance requires special preprocessing (SMOTE)
        - ✅ Metrics matter more than accuracy for imbalanced problems
        - ✅ Ensemble methods (XGBoost) outperform linear models
        - ✅ Fraud detection: Recall > Precision > Accuracy
        
        ### 8. Likely Professor Questions & Answers
        
        **Q: Why standardize features?**
        A: Ensures fair feature contributions; necessary for distance-based and gradient-descent models.
        
        **Q: Why apply SMOTE only to training data?**
        A: Prevents data leakage; ensures test set reflects real-world imbalance.
        
        **Q: Why not use accuracy?**
        A: With 99.83% legitimate transactions, a model predicting "all legitimate" would achieve 99.83% accuracy but catch zero frauds!
        
        **Q: How did you tune hyperparameters?**
        A: GridSearchCV with ROC-AUC scoring and 5-fold cross-validation.
        
        **Q: Why XGBoost over Random Forest?**
        A: Higher ROC-AUC score and better precision-recall balance in cross-validation.
        """)

# ==============================================================================
# LAUNCH INTERFACE
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LAUNCHING CREDIT CARD FRAUD DETECTION INTERFACE")
    print("="*70)
    print("\nGradio app is running at: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
