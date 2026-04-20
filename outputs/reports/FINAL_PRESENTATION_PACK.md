# Credit Card Fraud Detection — Final Presentation Pack

## Verified Sources Used
- [README.md](../../README.md)
- [PROJECT_SUMMARY.md](../../PROJECT_SUMMARY.md)
- [presentation_notes.md](../../presentation_notes.md)
- [EXECUTION_GUIDE.md](../../EXECUTION_GUIDE.md)
- [outputs/metrics/model_comparison.csv](../metrics/model_comparison.csv)
- Figures from [outputs/figures](../figures)

## Important Accuracy Note
Project docs include older benchmark text in places, but this final presentation follows the **actual current pipeline outputs** from [outputs/metrics/model_comparison.csv](../metrics/model_comparison.csv):
- Logistic Regression — Precision 0.1051, Recall 0.8737, F1 0.1876
- Random Forest — Precision 0.8987, Recall 0.7474, F1 0.8161
- XGBoost — Precision 0.6696, Recall 0.7895, F1 0.7246
- **Best model by F1-score: Random Forest**

---

## Best Slide Order (Final)
1. Title
2. Introduction: Problem & Objective
3. Dataset Overview
4. Data Insights: Imbalance & EDA
5. Methodology: Preprocessing + SMOTE
6. Models Trained
7. Comparative Analysis (Real Metrics)
8. Visual Analytics (Precision/Recall/F1 charts)
9. ROC & PR Curves
10. System Demo (Gradio Workflow)
11. Conclusion
12. Future Improvements
13. Q&A

---

## Slide-by-Slide Content + Speaker Notes

## Slide 1 — Title
**On slide**
- Credit Card Fraud Detection
- Machine Learning Final Project
- Student Name: ____________________
- Course / Semester: ____________________

**Visual**
- Clean white cover with blue accent and project icon

**Speaker notes (20–30s)**
- Introduce yourself and project context.
- State this is an end-to-end ML system from data to deployment.

---

## Slide 2 — Introduction: Problem & Objective
**On slide**
- Card fraud causes major financial losses.
- Detect fraud quickly with high reliability.
- Challenge: avoid too many false alerts while catching real fraud.
- Objective: build and demo a practical ML fraud detector.

**Speaker notes (45s)**
- Explain business impact of false negatives (missed fraud) and false positives (customer friction).
- Emphasize practical deployment mindset, not only notebook experimentation.

---

## Slide 3 — Dataset Overview
**On slide**
- Source: Credit Card Fraud Detection dataset (Kaggle/ULB)
- Total transactions: **284,807**
- Fraud cases: **492 (0.17%)**
- Features used: **V1–V28 + Amount**
- Target: Class (0 = Legitimate, 1 = Fraud)

**Speaker notes (45s)**
- Mention PCA-transformed variables are anonymized financial behavior signals.
- Highlight that class imbalance is the central modeling challenge.

---

## Slide 4 — Data Insights: EDA Findings
**On slide**
- Fraud class is extremely rare (0.17%).
- Transaction amount distribution is highly skewed.
- Multiple features contribute to fraud pattern (no single rule).

**Visuals (real figures)**
- [outputs/figures/01_class_distribution.png](../figures/01_class_distribution.png)
- [outputs/figures/02_amount_distribution.png](../figures/02_amount_distribution.png)
- [outputs/figures/03_top_correlations.png](../figures/03_top_correlations.png)

**Speaker notes (60s)**
- Explain why accuracy alone is misleading with this imbalance.
- Use the class distribution figure to motivate SMOTE.

---

## Slide 5 — Methodology: Preprocessing & SMOTE
**On slide**
- Data cleaning and preprocessing pipeline
  - duplicate/missing checks
  - feature scaling (StandardScaler)
  - stratified train-test split
- SMOTE applied on training data to address minority underrepresentation
- leakage-safe workflow (fit transforms on train; evaluate on untouched test)

**Speaker notes (60s)**
- Explain SMOTE simply: synthetic minority examples between neighbors.
- Emphasize why applying SMOTE only on train is critical.

---

## Slide 6 — Models Trained
**On slide**
- Logistic Regression (baseline)
- Random Forest (ensemble trees)
- XGBoost (gradient boosting)
- Tuning approach: practical randomized search for demo-ready runtime

**Speaker notes (45s)**
- Position LR as baseline, RF/XGBoost as stronger nonlinear models.
- Mention tradeoff between speed and performance.

---

## Slide 7 — Comparative Analysis (Real Metrics)
**On slide (table)**
- Logistic Regression: Precision 0.1051 | Recall 0.8737 | F1 0.1876
- Random Forest: Precision 0.8987 | Recall 0.7474 | F1 0.8161
- XGBoost: Precision 0.6696 | Recall 0.7895 | F1 0.7246

**Key highlight**
- **Best Model: Random Forest (highest F1-score = 0.8161)**

**Data source**
- [outputs/metrics/model_comparison.csv](../metrics/model_comparison.csv)

**Speaker notes (70s)**
- Explain that RF gives strongest precision-recall balance in this run.
- Discuss why LR recall is high but precision is too low for production alerts.

---

## Slide 8 — Visual Analytics (Presentation Charts)
**On slide**
- Precision comparison chart
- Recall comparison chart
- F1-score comparison chart

**Visuals**
- Generated from real metric CSV into: [outputs/reports/charts](./charts)

**Speaker notes (50s)**
- Walk through each chart quickly.
- Reinforce that F1 is the balancing metric used to select best model.

---

## Slide 9 — ROC/PR Evaluation
**On slide**
- ROC and PR curves compare discrimination and precision-recall behavior.
- Useful for imbalanced-class evaluation.

**Visuals (real figures)**
- [outputs/figures/roc_curve_comparison.png](../figures/roc_curve_comparison.png)
- [outputs/figures/pr_curve_comparison.png](../figures/pr_curve_comparison.png)

**Speaker notes (55s)**
- Explain ROC (TPR vs FPR) and PR (precision vs recall).
- Mention PR is often more informative for rare positive classes.

---

## Slide 10 — System Demo (Gradio App)
**On slide**
- Dashboard sections:
  - Model comparison
  - Single transaction prediction
  - Batch CSV prediction
- Outputs:
  - prediction label
  - fraud probability
  - downloadable batch results

**Visual**
- Use live demo (recommended)
- If screenshots are captured, place them here.

**Speaker notes (60–90s)**
- Show one normal and one suspicious single prediction.
- Show batch upload and summary counts.
- Keep demo short and stable.

---

## Slide 11 — Conclusion
**On slide**
- Built complete fraud detection pipeline + dashboard.
- Addressed severe class imbalance with SMOTE.
- Compared 3 models with imbalanced-data metrics.
- **Random Forest selected as best by F1-score (0.8161)**.

**Speaker notes (45s)**
- Summarize technical and practical contribution.
- Emphasize end-to-end reproducibility.

---

## Slide 12 — Future Improvements
**On slide**
- Threshold optimization by business cost
- Explainability (SHAP/LIME)
- Continuous retraining / drift monitoring
- Ensemble stacking and probability calibration
- Production API + monitoring dashboard

**Speaker notes (40s)**
- Show understanding of next-stage engineering beyond classroom scope.

---

## Slide 13 — Q&A
**On slide**
- Thank you
- Questions?

**Speaker notes**
- Prepare short, direct answers using section below.

---

## Likely Professor Questions + Short Answers

1. **Why is accuracy not enough?**
   - Fraud is only 0.17%; predicting all legitimate gives high accuracy but fails fraud detection.

2. **Why use SMOTE?**
   - To improve minority-class learning without discarding majority samples.

3. **Why Random Forest as best here?**
   - Highest F1-score (0.8161) with very high precision (0.8987), giving strongest balance.

4. **What’s the biggest challenge?**
   - Severe class imbalance and choosing proper metrics over misleading accuracy.

5. **How to improve recall further?**
   - Tune decision threshold, apply cost-sensitive learning, and optimize recall-focused objective.

6. **How would this go to production?**
   - Model serving API, logging, monitoring drift, scheduled retraining, and alert governance.

---

## What to Say for Each Section (Quick Script)
- **Intro:** real-world impact + objective.
- **Dataset:** imbalance challenge.
- **EDA:** evidence from class/amount/correlation plots.
- **Method:** preprocessing + SMOTE + leakage-safe split.
- **Models:** LR, RF, XGBoost rationale.
- **Results:** present table and explain RF selection by F1.
- **Demo:** show UI flow and practical outputs.
- **Conclusion:** technical outcome + deployment readiness + next steps.

---

## Delivery Notes
- Keep each slide to 3–5 bullets max.
- Let figures do the heavy communication.
- For live demo, run:
  - `python app.py`
- Use this as your canonical source to avoid old metric mismatches in earlier docs.
