# 🔐 Credit Card Fraud Detection
## Final Year Machine Learning Project

A comprehensive end-to-end machine learning system for detecting fraudulent credit card transactions using advanced algorithms, SMOTE-based resampling for class imbalance handling, and a professional Gradio web interface.

---

## 📋 Project Overview

### Objective
Build a machine learning model to accurately detect fraudulent credit card transactions while minimizing false alarms. This project addresses the challenge of severe class imbalance (0.17% fraud rate) and demonstrates best practices in machine learning engineering.

### Dataset
- **Source:** Credit Card Fraud Detection Dataset (Kaggle/ULB)
- **Transactions:** 284,807
- **Fraudulent Cases:** 492 (0.17%)
- **Features:** 28 PCA-transformed features (V1-V28) + Amount + Time
- **Target:** Class (0 = Legitimate, 1 = Fraudulent)

### Key Challenge
The dataset is **highly imbalanced** - only 0.17% of transactions are fraudulent. Traditional accuracy metrics are misleading because a naive model predicting "all legitimate" would achieve 99.83% accuracy without catching any fraud!

### Solution Approach
1. **Preprocessing:** Feature scaling + SMOTE resampling
2. **Modeling:** Logistic Regression, Random Forest, XGBoost
3. **Evaluation:** Recall, Precision, F1-Score, ROC-AUC, PR-AUC
4. **Deployment:** Interactive Gradio web interface

---

## 🏗️ Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv                 # Main dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration & constants
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocess.py                # Preprocessing pipeline
│   ├── train.py                     # Model training functions
│   ├── evaluate.py                  # Model evaluation
│   ├── predict.py                   # Prediction utilities
│   ├── smote_data.pkl               # SMOTE-processed data
│   └── feature_cols.pkl             # Feature column names
│
├── outputs/
│   ├── metrics/                     # Evaluation metrics tables
│   │   └── model_comparison.csv
│
├── app.py                           # Gradio web interface
├── models/
│   ├── logistic_regression.pkl      # Trained LR model
│   ├── random_forest.pkl            # Trained RF model
│   ├── xgboost.pkl                  # Trained XGBoost model
│   ├── scaler.pkl                   # Feature scaler
│   ├── baseline_data.pkl            # Baseline preprocessed data
│   ├── smote_data.pkl               # SMOTE-processed data
│   └── feature_cols.pkl             # Feature column names
│
├── ui_styles.css                    # UI styling overrides
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd credit-card-fraud-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place `creditcard.csv` in the `data/` directory:
```
credit-card-fraud-detection/data/creditcard.csv
```

### ⬇️ How to Download
 
**Option 1 – Kaggle Website**
1. Go to [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Sign in or create a free Kaggle account
3. Click **Download** to get `creditcard.csv`
4. Place the file in the `data/` folder of this project

### 3. Run Training Pipeline (No Notebooks Required)

Run one command:

```bash
python src/train.py
```

This command runs end-to-end:
- data loading and validation
- preprocessing and scaling
- baseline + SMOTE training flow
- Logistic Regression, Random Forest, and XGBoost training
- practical randomized tuning (student-laptop friendly)
- evaluation + metric saving
- model and metrics saving (including `models/best_model.pkl`)

### 4. Launch Web Interface

```bash
# After training completes and models are saved by src/train.py
python app.py

# Open browser and navigate to:
# http://localhost:7860
```

---

## 📊 Model Performance Summary

| Metric | Logistic Regression | Random Forest (BEST) | XGBoost |
|--------|--------------------|--------------|----|
| Accuracy | 0.9873 | 0.9994 | 0.9990 |
| Precision | 0.1051 | 0.8987 | 0.6696 |
| Recall | 0.8737 | 0.7474 | 0.7895 |
| F1-Score | 0.1876 | 0.8161 | 0.7246 |
| ROC-AUC | 0.9619 | 0.9649 | 0.9685 |
| Specificity | 0.9875 | 0.9999 | 0.9993 |

**Best Model: Random Forest**
- Highest F1-score (0.8161)
- Highest precision (0.8987)
- Strong accuracy and specificity

---

## 🎯 Technology Stack

### Core ML Libraries
- **scikit-learn** - Machine learning models and utilities
- **XGBoost** - Gradient boosting
- **imbalanced-learn** - SMOTE for handling class imbalance

### Data Processing
- **pandas** - Data manipulation
- **NumPy** - Numerical computations
- **matplotlib, seaborn** - Visualization

### Web Interface
- **Gradio** - Interactive web dashboard

### Development
- **joblib** - Model serialization

---

## 📈 Key Features

### 1. **Comprehensive EDA**
- Class distribution analysis
- Missing value checks
- Correlation analysis
- Amount distribution by class
- Statistical summaries

### 2. **Advanced Preprocessing**
- Stratified train/test split
- Feature scaling (StandardScaler)
- SMOTE resampling for class balance
- Data leakage prevention

### 3. **Multiple Models**
- Logistic Regression (baseline)
- Random Forest (tree-based)
- XGBoost (advanced boosting)

### 4. **Proper Evaluation**
- Classification reports
- ROC-AUC and PR-AUC
- Precision, recall, and F1-score

### 5. **Interactive Web Interface**
- Project overview
- Data insights
- Methodology summary
- Model comparison
- Conclusion highlights
- Batch prediction with CSV upload

---

## 🔍 How to Use the Web Interface

### Tab 1: Overview
- Project objective and context
- Dataset summary
- Why fraud detection matters
- Solution approach

### Tab 2: Data & Insights
- Class distribution visualization
- Key dataset takeaways

### Tab 3: Methodology
- Preprocessing steps
- Modeling approach
- Evaluation focus

### Tab 4: Model Comparison
- Performance metrics table
- Metric comparison charts

### Tab 5: Conclusion
- Best model summary
- Recommended next steps

### Tab 6: Batch Prediction
- Upload CSV with multiple transactions
- Download predictions
- See summary statistics

---

## 🎓 What This Project Demonstrates

### Machine Learning Best Practices
✅ Proper train/test split with stratification  
✅ Feature scaling to prevent bias  
✅ Handling class imbalance with SMOTE  
✅ Data leakage prevention  
✅ Practical hyperparameter tuning with RandomizedSearchCV  
✅ Appropriate metrics for imbalanced data  
✅ Model comparison and selection  

### Software Engineering
✅ Modular code structure  
✅ Configuration management  
✅ Error handling  
✅ Code documentation  
✅ Reproducible results  
✅ Professional web interface  

### Data Science Communication
✅ Clear visualizations  
✅ Comprehensive documentation  
✅ Interpretable models  

---

## 📝 Understanding the Code

### config.py (Configuration)
Centralized settings: paths, hyperparameters, random seeds

### data_loader.py
Functions to load and explore the dataset

### preprocess.py
Data cleaning, scaling, train/test split, SMOTE application

### train.py
Main entry point for one-command training pipeline (`python src/train.py`)

### evaluate.py
Model evaluation and comparison utilities

### utils.py
Visualization and metrics calculation functions

### predict.py
Prediction interface and batch processing

### app.py
Gradio web interface with all tabs and functionality

---

## ❓ Frequently Asked Questions

**Q: Why use SMOTE?**
A: SMOTE (Synthetic Minority Oversampling Technique) creates synthetic minority examples to balance the training data. This helps the model learn fraud patterns without biasing towards the majority class.

**Q: Why not apply SMOTE to test data?**
A: Test data must reflect real-world imbalance to get unbiased performance estimates. Applying SMOTE to test data would overestimate model performance.

**Q: Why XGBoost?**
A: XGBoost achieved the highest ROC-AUC score in cross-validation and provides excellent precision-recall trade-off, which is critical for fraud detection.

**Q: Why not just accuracy?**
A: With 99.83% legitimate transactions, a model predicting "all legitimate" gets 99.83% accuracy but catches 0% of frauds. Recall, Precision, and ROC-AUC are more meaningful metrics.

**Q: Can I improve the model further?**
A: Yes! Options include: ensemble methods, different resampling ratios, threshold tuning, additional feature engineering, or deep learning approaches.

---

## 🔧 Troubleshooting

### ImportError: No module named 'xgboost'
```bash
pip install xgboost
```

### Model files not found
Ensure you've run `python src/train.py` successfully.

### Gradio port already in use
```bash
python app.py --server_port 7861  # Use different port
```

---

## 📚 Future Improvements

- [ ] Try deep learning models (LSTM, neural networks)
- [ ] Implement real-time model monitoring
- [ ] Add threshold tuning for production deployment
- [ ] Implement automatic retraining pipeline
- [ ] Add explainability tools (SHAP, LIME)
- [ ] Deploy to cloud (Docker, AWS, GCP, Azure)
- [ ] Add more sophisticated feature engineering
- [ ] Implement A/B testing framework

---

## 👥 Team & Contributions

**Project:** Credit Card Fraud Detection - Final Year ML Project

**Implemented Algorithms:**
- Logistic Regression: Chea Chanrithyyuth
- Random Forest: Tong Bora
- XGBoost: Dorn Dana

---

## 📄 License

This project is for educational purposes. Use freely for learning and demonstrations.


---

**Last Updated:** April 2026  
**Version:** 1.0  
**Status:** Complete
