"""Credit Card Fraud Detection - Professional Demo UI"""

import os
import sys
import socket
import tempfile
import pandas as pd
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.predict import FraudDetectionPredictor
from src.config import MODELS_PATH, FIGURES_PATH, FEATURE_COLS, METRICS_PATH


PREDICTOR = None
ACTIVE_MODEL_NAME = ""

SAMPLE_TRANSACTION = {
    "V1": -1.358,
    "V2": -0.043,
    "V3": 2.136,
    "Amount": 149.62,
}


def get_predictor():
    global PREDICTOR, ACTIVE_MODEL_NAME
    if PREDICTOR is not None:
        return PREDICTOR

    scaler_path = f"{MODELS_PATH}/scaler.pkl"
    candidate_models = [
        "best_model.pkl",
        "xgboost.pkl",
        "random_forest.pkl",
        "logistic_regression.pkl",
    ]
    for model_file in candidate_models:
        model_path = f"{MODELS_PATH}/{model_file}"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            PREDICTOR = FraudDetectionPredictor(model_path, scaler_path)
            ACTIVE_MODEL_NAME = model_file.replace(".pkl", "")
            return PREDICTOR

    return None


def model_not_loaded_message():
    return "Model not loaded. Please run `python src/train.py` first."


def load_comparison_data():
    metrics_path = f"{METRICS_PATH}/model_comparison.csv"
    metrics_json_path = f"{METRICS_PATH}/model_comparison.json"

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path, index_col=0)
    elif os.path.exists(metrics_json_path):
        df = pd.read_json(metrics_json_path, orient="index")
    else:
        return None, "Unknown"

    rename_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "specificity": "Specificity",
    }
    available = [c for c in rename_map.keys() if c in df.columns]
    if not available:
        return None, "Unknown"

    view_df = df[available].copy().round(4).rename(columns=rename_map)
    best_idx = view_df["F1"].idxmax() if "F1" in view_df.columns else view_df.iloc[:, 0].idxmax()
    return view_df.reset_index().rename(columns={"index": "Model"}), str(best_idx)


def load_roc_image_path():
    candidates = [
        f"{FIGURES_PATH}/model_comparison_summary.png",
        f"{FIGURES_PATH}/roc_curve_comparison.png",
        f"{FIGURES_PATH}/pr_curve_comparison.png",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def build_features(amount, v1, v2, v3):
    features = {f"V{i}": 0.0 for i in range(1, 29)}
    features["V1"] = float(v1)
    features["V2"] = float(v2)
    features["V3"] = float(v3)
    features["Amount"] = float(amount)
    return features


def use_sample_data():
    return [
        SAMPLE_TRANSACTION["Amount"],
        SAMPLE_TRANSACTION["V1"],
        SAMPLE_TRANSACTION["V2"],
        SAMPLE_TRANSACTION["V3"],
    ]


def reset_inputs():
    return [0.0, 0.0, 0.0, 0.0]


def make_prediction(amount, v1, v2, v3):
    try:
        predictor = get_predictor()
        if predictor is None:
            return """
            <div class='result-card result-error'>
                <div class='result-eyebrow'>Model unavailable</div>
                <div class='result-title'>Please train first</div>
                <div class='result-prob'>""" + model_not_loaded_message() + """</div>
            </div>
            """

        features = build_features(amount, v1, v2, v3)
        result = predictor.predict(features)

        is_fraud = result["prediction"] == "Fraudulent"
        label = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"
        icon = "⚠️" if is_fraud else "✅"
        card_class = "result-fraud" if is_fraud else "result-normal"
        model_label = ACTIVE_MODEL_NAME.replace("_", " ").title()
        probability = float(result["fraud_probability"])
        probability_pct = probability * 100
        bar_class = "bar-fraud" if is_fraud else "bar-normal"

        return f"""
        <div class="result-card {card_class}">
            <div class="result-eyebrow">{icon} Prediction Result</div>
            <div class="result-title">{label}</div>
            <div class="result-prob">Fraud probability: <strong>{probability_pct:.1f}%</strong></div>
            <div class="prob-bar"><span class="{bar_class}" style="width: {probability_pct:.1f}%"></span></div>
            <div class="result-model">Model used: {model_label}</div>
        </div>
        """
    except Exception as exc:
        return f"""
        <div class='result-card result-error'>
            <div class='result-eyebrow'>Prediction error</div>
            <div class='result-title'>Unable to predict</div>
            <div class='result-prob'>{str(exc)}</div>
        </div>
        """


def preview_uploaded_csv(file):
    if file is None:
        return "Upload a CSV file to preview it.", pd.DataFrame()

    try:
        df = pd.read_csv(file.name)
        return f"Previewing {os.path.basename(file.name)} — {len(df)} rows", df.head(10)
    except Exception as exc:
        return f"Error reading CSV: {str(exc)}", pd.DataFrame()


def process_batch_predictions(file):
    if file is None:
        return "Upload a CSV file", None, pd.DataFrame()

    try:
        predictor = get_predictor()
        if predictor is None:
            return model_not_loaded_message(), None, pd.DataFrame()

        df = pd.read_csv(file.name)

        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            return f"❌ Missing columns: {', '.join(missing)}", None, pd.DataFrame()

        result_df = predictor.batch_predict(df[FEATURE_COLS])
        fraud_count = int((result_df["Prediction"] == 1).sum())
        total = len(result_df)
        legitimate_count = total - fraud_count

        with tempfile.NamedTemporaryFile(delete=False, suffix="_predictions.csv") as tmp:
            result_df.to_csv(tmp.name, index=False)
            output_path = tmp.name

        summary = {
            "total": total,
            "fraud": fraud_count,
            "normal": legitimate_count,
            "fraud_rate": (fraud_count / total) if total else 0,
            "path": output_path,
        }
        summary_html = build_batch_summary_html(summary)
        preview_df = result_df[["Prediction_Label", "Fraud_Probability"]].head(20).copy()
        preview_df["Fraud_Probability"] = preview_df["Fraud_Probability"].map(lambda x: f"{x:.1%}")
        return summary_html, output_path, preview_df
    except Exception as exc:
        return f"<div class='summary-alert error'>Error: {str(exc)}</div>", None, pd.DataFrame()


def find_available_port(start_port=7860, max_tries=20):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {start_port}-{start_port + max_tries - 1}.")


MODEL_COLORS = {
    'Logistic Regression': '#2563eb',
    'Random Forest': '#10b981',
    'XGBoost': '#f59e0b',
}


def metric_value(row, key):
    return float(row.get(key, row.get(key.lower(), 0.0)))


def percent_text(value):
    return f'{value:.1%}'


def load_dataset_summary():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
    default_summary = {
        "total": 284807,
        "fraud": 492,
        "normal": 284807 - 492,
        "fraud_rate": 492 / 284807,
    }

    if not os.path.exists(dataset_path):
        return default_summary

    try:
        df = pd.read_csv(dataset_path, usecols=["Class"])
        total = int(len(df))
        fraud = int((df["Class"] == 1).sum())
        normal = total - fraud
        return {
            "total": total,
            "fraud": fraud,
            "normal": normal,
            "fraud_rate": (fraud / total) if total else 0,
        }
    except Exception:
        return default_summary


def create_class_distribution_chart(summary):
    labels = ["Normal", "Fraud"]
    values = [summary["normal"], summary["fraud"]]
    colors = ["#2563eb", "#ef4444"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="#dbe4f0", linewidth=1.2)
    ax.set_title("Class Distribution: Normal vs Fraud", fontsize=14, fontweight="bold", color="#0f172a", pad=12)
    ax.grid(axis="y", color="#e5e7eb", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11, colors="#334155")
    ax.tick_params(axis="y", labelsize=10, colors="#64748b")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")

    total = max(summary["total"], 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value/total:.2%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#334155",
        )

    plt.tight_layout(pad=1.2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_class_distribution.png")
    fig.savefig(tmp.name, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return tmp.name


def build_overview_html():
    return """
    <div class='center-hero'>
      <div class='hero-kicker'>Interactive Presentation Dashboard</div>
      <h1 class='hero-title'>Credit Card Fraud Detection</h1>
      <p class='hero-subtitle'>Detect fraudulent transactions using machine learning.</p>
      <div class='overview-grid'>
        <div class='overview-card'>
          <h4>Problem Statement</h4>
          <p>Financial fraud is rare but highly damaging. The challenge is to detect fraudulent transactions accurately without disrupting legitimate users.</p>
        </div>
        <div class='overview-card'>
          <h4>Why It Matters</h4>
          <p>Missed fraud causes direct financial loss, while false alerts degrade customer trust. A balanced model is critical for real-world operations.</p>
        </div>
      </div>
    </div>
    """


def build_data_insights_html(summary):
    return f"""
    <div class='section-eyebrow'>Dataset & Imbalance Insight</div>
    <div class='insight-title'>Credit Card Transaction Dataset</div>
    <div class='insight-grid'>
      <div class='insight-item'><span>Total transactions</span><strong>{summary['total']:,}</strong></div>
      <div class='insight-item'><span>Normal transactions</span><strong>{summary['normal']:,}</strong></div>
      <div class='insight-item'><span>Fraud transactions</span><strong>{summary['fraud']:,}</strong></div>
      <div class='insight-item'><span>Fraud ratio</span><strong>{percent_text(summary['fraud_rate'])}</strong></div>
    </div>
    <div class='imbalance-note'>
      Fraud is a rare-event class. This imbalance can make naive models appear accurate while still missing actual fraud.
      That is why we prioritize precision, recall, and F1-score rather than plain accuracy.
    </div>
    """


def build_methodology_html():
    return """
    <div class='section-eyebrow'>Pipeline</div>
    <div class='insight-title'>Methodology</div>
    <p class='method-intro'>
      The pipeline focuses on reliable evaluation for imbalanced fraud data: clean preprocessing,
      leakage-safe splits, and comparative model training.
    </p>

    <div class='method-shell'>
      <div class='method-grid'>
        <div class='method-card'>
          <h4>Preprocessing</h4>
          <ul>
            <li>Feature scaling for stable model training</li>
            <li>Train / test split for reliable evaluation</li>
            <li>SMOTE to address severe class imbalance</li>
          </ul>
        </div>
        <div class='method-card'>
          <h4>Modeling</h4>
          <ul>
            <li>Logistic Regression for baseline linear behavior</li>
            <li>Random Forest for robust nonlinear detection</li>
            <li>XGBoost for boosted tree performance</li>
          </ul>
        </div>
      </div>

      <div class='mini-model-grid'>
        <div class='mini-model-card'>
          <h5>Logistic Regression</h5>
          <p>Fast and interpretable baseline model to set comparison reference.</p>
        </div>
        <div class='mini-model-card'>
          <h5>Random Forest</h5>
          <p>Ensemble tree method with strong precision-recall balance on this dataset.</p>
        </div>
        <div class='mini-model-card'>
          <h5>XGBoost</h5>
          <p>Gradient boosting model that captures complex fraud behavior patterns.</p>
        </div>
      </div>
    </div>
    """


def build_best_model_banner(metrics_df, best_model):
    if metrics_df is None or metrics_df.empty:
        return "<div class='best-model-banner'>Best model will appear after training.</div>"

    row = metrics_df[metrics_df["Model"] == best_model]
    if row.empty:
        row = metrics_df.iloc[[0]]
        best_model = str(row.iloc[0]["Model"])

    best_f1 = metric_value(row.iloc[0], "F1")
    return f"""
    <div class='best-model-banner'>
      <strong>Best Model: {best_model}</strong>
      <span>Why: highest F1-score ({best_f1:.4f}) with strong precision-recall balance.</span>
    </div>
    """


def build_conclusion_html(metrics_df, best_model):
    if metrics_df is None or metrics_df.empty:
        return """
        <div class='insight-title'>Conclusion</div>
        <ul class='insight-list'>
          <li>Train models to generate final comparison outputs.</li>
          <li>This tab summarizes final recommendation and challenges.</li>
        </ul>
        """

    row = metrics_df[metrics_df["Model"] == best_model]
    if row.empty:
        row = metrics_df.iloc[[0]]
        best_model = str(row.iloc[0]["Model"])

    precision = metric_value(row.iloc[0], "Precision")
    recall = metric_value(row.iloc[0], "Recall")
    f1 = metric_value(row.iloc[0], "F1")

    return f"""
    <div class='section-eyebrow'>Final Takeaway</div>
    <div class='insight-title'>Conclusion</div>
    <ul class='insight-list'>
      <li><strong>Summary:</strong> End-to-end fraud detection pipeline built from preprocessing to live prediction demo.</li>
      <li><strong>Best model:</strong> {best_model} (Precision {percent_text(precision)} | Recall {percent_text(recall)} | F1 {percent_text(f1)}).</li>
      <li><strong>Key takeaway:</strong> Random Forest provides best balance for fraud detection in this project run.</li>
      <li><strong>Challenges:</strong> severe class imbalance and model training/evaluation time.</li>
    </ul>
    """


def build_model_cards_html(metrics_df, best_model):
    if metrics_df is None or metrics_df.empty:
        return "<div class='empty-state'>No metrics available. Run training first.</div>"

    cards = []
    for _, row in metrics_df.iterrows():
        model_name = str(row.get('Model', 'Unknown'))
        precision = metric_value(row, 'Precision')
        recall = metric_value(row, 'Recall')
        f1 = metric_value(row, 'F1')
        accent = MODEL_COLORS.get(model_name, '#6366f1')
        is_best = model_name == best_model

        cards.append(f"""
        <div class="model-card {'best-card' if is_best else ''}">
          <div class="model-card-top">
            <div class="model-name">{model_name}</div>
            {"<div class='best-badge'>Best Model</div>" if is_best else ""}
          </div>
          <div class="model-score-row">
            <span>Precision</span><strong>{percent_text(precision)}</strong>
          </div>
          <div class="mini-bar"><span style="width:{precision*100:.1f}%; background:{accent}"></span></div>
          <div class="model-score-row">
            <span>Recall</span><strong>{percent_text(recall)}</strong>
          </div>
          <div class="mini-bar"><span style="width:{recall*100:.1f}%; background:{accent}"></span></div>
          <div class="model-score-row">
            <span>F1-score</span><strong>{percent_text(f1)}</strong>
          </div>
          <div class="mini-bar"><span style="width:{f1*100:.1f}%; background:{accent}"></span></div>
        </div>
        """)

    return f"<div class='model-cards-grid'>{''.join(cards)}</div>"


def build_insight_card(metrics_df, best_model):
    if metrics_df is None or metrics_df.empty:
        return "<div class='empty-state'>Best model insight will appear after training.</div>"

    best_row = metrics_df[metrics_df['Model'] == best_model]
    if best_row.empty:
        best_row = metrics_df.iloc[[0]]
        best_model = str(best_row.iloc[0]['Model'])

    row = best_row.iloc[0]
    precision = metric_value(row, 'Precision')
    recall = metric_value(row, 'Recall')
    f1 = metric_value(row, 'F1')

    return f"""
    <div class='insight-card'>
      <div class='section-eyebrow'>Model Insight</div>
      <div class='insight-title'>Best Model: {best_model}</div>
      <div class='insight-grid'>
        <div class='insight-item'><span>Why chosen</span><strong>Highest F1-score</strong></div>
        <div class='insight-item'><span>Precision</span><strong>{percent_text(precision)}</strong></div>
        <div class='insight-item'><span>Recall</span><strong>{percent_text(recall)}</strong></div>
        <div class='insight-item'><span>F1-score</span><strong>{percent_text(f1)}</strong></div>
      </div>
      <ul class='insight-list'>
        <li>Strong precision keeps false alarms low.</li>
        <li>Balanced recall helps catch fraudulent transactions.</li>
        <li>Best overall trade-off for live demo presentation.</li>
      </ul>
    </div>
    """


def create_metric_chart(metrics_df, metric_name, title):
    if metrics_df is None or metrics_df.empty or metric_name not in metrics_df.columns:
        return None

    df = metrics_df[['Model', metric_name]].copy()
    colors = [MODEL_COLORS.get(model, '#6366f1') for model in df['Model']]
    best_idx = metrics_df[metric_name].idxmax() if metric_name in metrics_df.columns else None
    if best_idx is not None:
        best_model = str(metrics_df.loc[best_idx, 'Model'])
        colors = [
            '#14b8a6' if model == best_model else MODEL_COLORS.get(model, '#6366f1')
            for model in df['Model']
        ]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    bars = ax.bar(df['Model'], df[metric_name], color=colors, width=0.56, edgecolor='#dbe4f0', linewidth=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#0f172a', pad=14)
    ax.grid(axis='y', color='#e5e7eb', linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10, colors='#334155')
    ax.tick_params(axis='y', labelsize=9, colors='#64748b')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')

    for bar, value in zip(bars, df[metric_name]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            percent_text(float(value)),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#334155',
        )

    plt.tight_layout(pad=1.2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{metric_name.lower()}.png')
    fig.savefig(tmp.name, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return tmp.name


def build_summary_cards_html(total, fraud_count, normal_count):
    fraud_rate = fraud_count / total if total else 0
    normal_rate = normal_count / total if total else 0
    return f"""
    <div class='summary-grid'>
      <div class='summary-stat'><span>Total rows</span><strong>{total}</strong></div>
      <div class='summary-stat danger'><span>Predicted fraud</span><strong>{fraud_count} ({percent_text(fraud_rate)})</strong></div>
      <div class='summary-stat success'><span>Predicted normal</span><strong>{normal_count} ({percent_text(normal_rate)})</strong></div>
    </div>
    """


def build_batch_summary_html(summary):
    return f"""
    <div class='batch-summary-card'>
      <div class='section-eyebrow'>Batch Prediction Summary</div>
      <div class='summary-grid'>
        <div class='summary-stat'><span>Total rows</span><strong>{summary['total']}</strong></div>
        <div class='summary-stat danger'><span>Predicted fraud</span><strong>{summary['fraud']}</strong></div>
        <div class='summary-stat success'><span>Predicted normal</span><strong>{summary['normal']}</strong></div>
      </div>
      <div class='summary-footnote'>Fraud rate: {percent_text(summary['fraud_rate'])}</div>
    </div>
    """


def build_css():
    return """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
      --bg: #f5f5f5;
      --panel: #ffffff;
      --panel-soft: #ffffff;
      --text: #111827;
      --text-secondary: #374151;
      --muted: #6b7280;
      --line: #e5e7eb;
      --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      --shadow-soft: 0 1px 2px rgba(0, 0, 0, 0.05);
      --blue: #2563eb;
      --emerald: #10b981;
      --amber: #f59e0b;
      --red: #dc2626;
      --radius-xl: 12px;
      --radius-lg: 8px;
      --radius-md: 6px;
      --radius-sm: 4px;
      --font: 'Inter', sans-serif;
    }

    body,
    .gradio-container {
      --body-background-fill: #ffffff !important;
      --body-background-fill-dark: #ffffff !important;
      --background-fill-primary: #ffffff !important;
      --background-fill-secondary: #ffffff !important;
      --block-background-fill: #ffffff !important;
    }

    *, *::before, *::after { box-sizing: border-box; }
    html,
    body,
    .gradio-container,
    .gradio-container > .main,
    .gradio-container .main,
    .gradio-container .app,
    .gradio-container .contain,
    .gradio-container .wrap,
    #root {
      font-family: var(--font) !important;
      background: var(--bg) !important;
      color: var(--text) !important;
      min-height: 100vh;
    }

    html,
    body,
    #root,
    .gradio-container,
    .gradio-container > .main,
    .gradio-container .main,
    .gradio-container .app,
    .gradio-container .contain,
    .gradio-container .wrap,
    .gradio-container .tabs,
    .gradio-container .tabitem {
      background-color: #ffffff !important;
      background-image: none !important;
    }

    .gradio-container {
      max-width: 100% !important;
      width: 100% !important;
      margin: 0 !important;
      padding: 30px 40px !important;
    }

    footer { display: none !important; }

    .center-hero {
      padding: 40px 30px;
      text-align: center;
      margin: 0 auto 30px;
      max-width: 1200px;
      background: var(--panel-soft);
      border-radius: var(--radius-lg);
      border: 1px solid var(--line);
    }

    .hero-kicker {
      display: inline-block;
      background: #dbeafe;
      color: #1e40af;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }

    .hero-title {
      font-size: 48px;
      line-height: 1.1;
      margin: 0 0 8px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: var(--text);
    }

    .hero-subtitle {
      max-width: 600px;
      margin: 0 auto;
      font-size: 18px;
      line-height: 1.6;
      color: var(--text-secondary);
    }

    .overview-grid {
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .overview-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 20px;
      text-align: left;
      box-shadow: var(--shadow-soft);
    }

    .overview-card h4 {
      margin: 0 0 10px;
      font-size: 18px;
      font-weight: 700;
      color: var(--text);
    }

    .overview-card p {
      margin: 0;
      color: var(--text-secondary);
      line-height: 1.7;
      font-size: 15px;
    }

    .presentation-tab-wrap {
      max-width: 1200px;
      margin: 0 auto;
    }

    .gradio-container .tabs {
      max-width: 1200px;
      margin: 0 auto;
    }

    .gradio-container .tab-nav {
      background: linear-gradient(180deg, #eff6ff 0%, #e0ecff 100%) !important;
      border: 1px solid #bfdbfe !important;
      border-radius: 10px !important;
      padding: 4px !important;
      gap: 6px !important;
      box-shadow: var(--shadow-soft) !important;
      margin: 0 auto 20px !important;
      max-width: 1200px;
    }

    .gradio-container .tab-nav button,
    .gradio-container .tab-nav [role="tab"] {
      background: #ffffff !important;
      border-radius: 7px !important;
      border: 1px solid #cbd5e1 !important;
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      font-weight: 800 !important;
      font-size: 15px !important;
      line-height: 1.2 !important;
      min-height: 38px !important;
      padding: 8px 14px !important;
      opacity: 1 !important;
      text-shadow: none !important;
      filter: none !important;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }

    .gradio-container .tab-nav button *,
    .gradio-container .tab-nav [role="tab"] * {
      color: inherit !important;
      -webkit-text-fill-color: inherit !important;
      opacity: 1 !important;
      font-weight: inherit !important;
    }

    .gradio-container .tab-nav button:not(.selected):not([aria-selected="true"]),
    .gradio-container .tab-nav [role="tab"]:not([aria-selected="true"]) {
      background: #ffffff !important;
      border-color: #cbd5e1 !important;
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
    }

    .gradio-container .tab-nav button:not(.selected):not([aria-selected="true"]) *,
    .gradio-container .tab-nav [role="tab"]:not([aria-selected="true"]) * {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
    }

    .gradio-container .tab-nav button:hover,
    .gradio-container .tab-nav [role="tab"]:hover {
      background: #eaf2ff !important;
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      border-color: #93c5fd !important;
    }

    .gradio-container .tab-nav button.selected,
    .gradio-container .tab-nav button[aria-selected="true"],
    .gradio-container .tab-nav [role="tab"][aria-selected="true"] {
      background: #1e3a8a !important;
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      border-color: #1e3a8a !important;
      font-weight: 800 !important;
      box-shadow: 0 2px 8px rgba(30, 58, 138, 0.35) !important;
    }

    .gradio-container .tab-nav button.selected *,
    .gradio-container .tab-nav button[aria-selected="true"] *,
    .gradio-container .tab-nav [role="tab"][aria-selected="true"] * {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      opacity: 1 !important;
    }

    .gradio-container [role="tablist"] > button,
    .gradio-container [role="tablist"] > [role="tab"] {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      opacity: 1 !important;
      background: #ffffff !important;
      border: 1px solid #cbd5e1 !important;
      font-weight: 800 !important;
    }

    .gradio-container [role="tablist"] > button[aria-selected="true"],
    .gradio-container [role="tablist"] > [role="tab"][aria-selected="true"] {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      background: #1e3a8a !important;
      border-color: #1e3a8a !important;
    }

    .gradio-container [role="tablist"] > button *,
    .gradio-container [role="tablist"] > [role="tab"] * {
      color: inherit !important;
      -webkit-text-fill-color: inherit !important;
      opacity: 1 !important;
    }

    .section-card {
      background: #ffffff !important;
      border: 1px solid var(--line) !important;
      border-radius: var(--radius-lg) !important;
      padding: 30px !important;
      margin: 0 auto 20px !important;
      max-width: 1200px;
      box-shadow: var(--shadow-soft) !important;
    }

    .section-card h3 {
      font-size: 28px !important;
      font-weight: 800 !important;
      letter-spacing: -0.02em !important;
      color: var(--text) !important;
      margin-bottom: 20px !important;
      padding: 0 !important;
      border: none !important;
    }

    .section-eyebrow {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }

    .section-card p,
    .section-card li {
      color: var(--text-secondary) !important;
      font-size: 15px !important;
      line-height: 1.7 !important;
    }

    .method-intro {
      margin: 0 0 16px;
      color: var(--text-secondary) !important;
      font-size: 16px;
      font-weight: 500;
      line-height: 1.7;
      max-width: 900px;
    }

    .method-shell {
      background: #ffffff !important;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 16px;
    }

    .method-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
      margin-bottom: 12px;
    }

    .method-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 14px;
      box-shadow: var(--shadow-soft);
    }

    .method-card h4 {
      margin: 0 0 10px;
      color: var(--text) !important;
      font-size: 16px;
      font-weight: 700;
    }

    .method-card ul {
      margin: 0;
      padding-left: 18px;
      color: var(--text-secondary);
      line-height: 1.7;
    }

    .method-card ul li {
      color: var(--text-secondary);
      margin: 4px 0;
      font-size: 14px;
    }

    .mini-model-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .mini-model-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 12px;
      box-shadow: var(--shadow-soft);
    }

    .mini-model-card h5 {
      margin: 0 0 8px;
      color: var(--text) !important;
      font-size: 14px;
      font-weight: 700;
    }

    .mini-model-card p {
      margin: 0;
      color: var(--text-secondary) !important;
      font-size: 13px;
      line-height: 1.6;
    }

    .insight-title {
      font-size: 24px;
      font-weight: 800;
      color: var(--text);
      margin-bottom: 16px;
    }

    .insight-grid, .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 16px 0;
    }

    .insight-item, .summary-stat {
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 12px;
    }

    .insight-item span, .summary-stat span {
      display: block;
      font-size: 11px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }

    .insight-item strong, .summary-stat strong {
      font-size: 18px;
      color: var(--text);
      font-weight: 700;
    }

    .insight-list {
      margin: 0;
      padding-left: 18px;
      color: var(--text-secondary);
      line-height: 1.8;
    }

    .insight-list li { 
      margin: 8px 0; 
      font-size: 15px;
    }

    .imbalance-note {
      border: 1px solid var(--line);
      background: var(--panel-soft);
      border-radius: var(--radius-lg);
      padding: 14px;
      color: var(--text-secondary);
      line-height: 1.7;
      margin-top: 12px;
      font-size: 14px;
    }

    .model-cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }

    .model-card {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 16px;
      box-shadow: var(--shadow-soft);
    }

    .model-card::before {
      display: none;
    }

    .model-card:hover {
      transform: none;
      box-shadow: var(--shadow);
      border-color: var(--blue);
    }

    .best-card {
      border-color: var(--blue) !important;
      background: #dbeafe !important;
      box-shadow: var(--shadow) !important;
    }

    .best-card::before {
      display: none;
    }

    .model-card-top {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      margin-bottom: 12px;
    }

    .model-name {
      font-size: 16px;
      font-weight: 700;
      color: var(--text);
      line-height: 1.2;
    }

    .best-badge {
      white-space: nowrap;
      background: var(--emerald);
      color: white;
      border: 1px solid var(--emerald);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-radius: 4px;
      padding: 4px 8px;
    }

    .model-score-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      color: var(--text-secondary);
      margin-top: 8px;
    }

    .model-score-row strong {
      color: var(--text);
      font-size: 14px;
      font-weight: 700;
    }

    .mini-bar {
      height: 6px;
      width: 100%;
      background: var(--line);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 6px;
    }

    .mini-bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
    }

    .insight-card, .batch-summary-card, .summary-alert {
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 16px;
      box-shadow: var(--shadow-soft);
    }

    .best-model-banner {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      border: 1px solid var(--emerald);
      background: #f0fdf4;
      color: var(--emerald);
      border-radius: var(--radius-lg);
      padding: 12px 14px;
      margin-bottom: 16px;
    }

    .best-model-banner strong {
      font-size: 15px;
      color: var(--text);
    }

    .best-model-banner span {
      font-size: 14px;
      color: var(--text-secondary);
    }

    .analytics-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }

    .chart-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      overflow: hidden;
      box-shadow: var(--shadow-soft);
    }

    .chart-card .gradio-image, .chart-card img {
      border-radius: var(--radius-lg);
    }

    .chart-wide {
      margin-top: 16px;
    }

    .prediction-shell {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }

    .form-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 16px;
      box-shadow: var(--shadow-soft);
    }

    .gradio-container input[type="number"],
    .gradio-container input[type="text"],
    .gradio-container textarea,
    .gradio-container .wrap input {
      background: var(--panel) !important;
      color: var(--text) !important;
      border: 1px solid var(--line) !important;
      border-radius: 6px !important;
      min-height: 40px !important;
      font-family: var(--font) !important;
      box-shadow: none !important;
      font-size: 14px !important;
    }

    .gradio-container input:focus,
    .gradio-container textarea:focus {
      border-color: var(--blue) !important;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    .gradio-container label span,
    .gradio-container .label-wrap span {
      color: var(--muted) !important;
      font-size: 11px !important;
      font-weight: 700 !important;
      letter-spacing: 0.05em !important;
      text-transform: uppercase !important;
    }

    .gradio-container button {
      border-radius: 6px !important;
      min-height: 40px !important;
      font-weight: 700 !important;
      letter-spacing: 0.01em !important;
      transition: all 0.2s ease !important;
      font-family: var(--font) !important;
    }

    .gradio-container button:hover {
      transform: none !important;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }

    .gradio-container button.primary,
    .gradio-container .gr-button-primary {
      background: var(--blue) !important;
      color: white !important;
      border: none !important;
    }

    .gradio-container button.secondary,
    .gradio-container .gr-button-secondary {
      background: var(--panel) !important;
      color: var(--text) !important;
      border: 1px solid var(--line) !important;
    }

    .result-card {
      border-radius: var(--radius-lg);
      padding: 24px;
      border: 2px solid var(--line);
      box-shadow: var(--shadow-soft);
      background: var(--panel);
      text-align: center;
    }

    .result-eyebrow {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 8px;
      color: var(--muted);
    }

    .result-title {
      font-size: 42px;
      line-height: 1.05;
      margin: 0;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .result-prob {
      margin-top: 12px;
      font-size: 16px;
      color: var(--text-secondary);
    }

    .prob-bar {
      height: 12px;
      background: var(--line);
      border-radius: 6px;
      overflow: hidden;
      margin-top: 12px;
    }

    .prob-bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
    }

    .bar-fraud { background: var(--red); }
    .bar-normal { background: var(--emerald); }

    .result-fraud { 
      border-color: var(--red); 
      background: #fef2f2;
    }
    .result-fraud .result-title { color: var(--red); }

    .result-normal { 
      border-color: var(--emerald); 
      background: #f0fdf4;
    }
    .result-normal .result-title { color: var(--emerald); }

    .result-error { 
      border-color: var(--amber); 
      background: #fffbeb;
    }
    .result-error .result-title { color: var(--amber); }

    .summary-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .summary-stat.danger { 
      border-color: var(--red); 
      background: #fef2f2;
    }

    .summary-stat.success { 
      border-color: var(--emerald); 
      background: #f0fdf4;
    }

    .summary-footnote {
      color: var(--text-secondary);
      font-size: 13px;
      margin-top: 12px;
    }

    .empty-state {
      padding: 16px;
      border: 1px dashed var(--line);
      border-radius: var(--radius-lg);
      color: var(--muted);
      background: var(--panel-soft);
      text-align: center;
      font-size: 14px;
    }

    .gradio-container table,
    .gradio-container .gr-dataframe,
    .gradio-container [data-testid="dataframe"] {
      border: 1px solid var(--line) !important;
      border-radius: var(--radius-lg) !important;
      overflow: hidden !important;
      box-shadow: var(--shadow-soft) !important;
    }

    .gradio-container table th,
    .gradio-container .gr-dataframe th {
      background: var(--panel-soft) !important;
      color: var(--text) !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.05em !important;
      font-size: 11px !important;
      border-bottom: 1px solid var(--line) !important;
    }

    .gradio-container table td,
    .gradio-container .gr-dataframe td {
      color: var(--text-secondary) !important;
      font-size: 14px !important;
    }

    .gradio-container .file-preview,
    .gradio-container .upload-container,
    .gradio-container [data-testid="file-upload"] {
      background: var(--panel) !important;
      border: 1px dashed var(--blue) !important;
      border-radius: var(--radius-lg) !important;
    }

    .gradio-container .gr-group,
    .gradio-container .gr-box,
    .gradio-container .block,
    .gradio-container [data-testid="block"],
    .gradio-container [data-testid="block-group"] {
      background: #ffffff !important;
      border: none !important;
    }

    .gradio-container .gr-group.section-card,
    .gradio-container .gr-box.section-card,
    .gradio-container .block.section-card,
    .gradio-container .section-card {
      background: #ffffff !important;
      border: 1px solid var(--line) !important;
    }

    @media (max-width: 1100px) {
      .analytics-grid, .prediction-shell { grid-template-columns: 1fr; }
      .overview-grid, .method-grid, .mini-model-grid { grid-template-columns: 1fr; }
      .insight-grid, .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
      .section-card { padding: 16px !important; margin: 0 10px 14px !important; }
      .center-hero { padding: 24px 16px; }
      .hero-title { font-size: 32px; }
      .insight-grid, .summary-grid { grid-template-columns: 1fr; }
      .gradio-container { padding: 16px !important; }
    }
    """


CSS = build_css()
APP_THEME = gr.themes.Soft().set(
  body_background_fill='#f5f5f5',
  body_background_fill_dark='#f5f5f5',
  background_fill_primary='#f5f5f5',
  background_fill_secondary='#f5f5f5',
  block_background_fill='#ffffff',
)


# ─── Build Data ───────────────────────────────────────────────────────────────
metrics_df, best_model = load_comparison_data()
dataset_summary = load_dataset_summary()
class_distribution_chart = create_class_distribution_chart(dataset_summary)
cards_html = build_model_cards_html(metrics_df, best_model)
insight_html = build_insight_card(metrics_df, best_model)
best_banner_html = build_best_model_banner(metrics_df, best_model)
data_insights_html = build_data_insights_html(dataset_summary)
methodology_html = build_methodology_html()
conclusion_html = build_conclusion_html(metrics_df, best_model)
precision_chart = create_metric_chart(metrics_df, 'Precision', 'Precision Comparison')
recall_chart = create_metric_chart(metrics_df, 'Recall', 'Recall Comparison')
f1_chart = create_metric_chart(metrics_df, 'F1', 'F1-score Comparison')
roc_path = load_roc_image_path()


# ─── Build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title='Credit Card Fraud Detection') as demo:
  with gr.Tabs(elem_classes=['presentation-tab-wrap']):
    with gr.Tab('1. Overview'):
      with gr.Group(elem_classes=['section-card']):
        gr.HTML(build_overview_html())

    with gr.Tab('2. Data & Insights'):
      with gr.Group(elem_classes=['section-card']):
        gr.HTML(data_insights_html)
        gr.Image(value=class_distribution_chart, show_label=False, interactive=False)

    with gr.Tab('3. Methodology'):
      with gr.Group(elem_classes=['section-card']):
        gr.HTML(methodology_html)

    with gr.Tab('4. Model Comparison'):
      with gr.Group(elem_classes=['section-card']):
        gr.HTML(best_banner_html)
        gr.HTML(cards_html)
        gr.HTML(insight_html)

        with gr.Row(elem_classes=['analytics-grid']):
          with gr.Column(elem_classes=['chart-card']):
            if precision_chart:
              gr.Image(value=precision_chart, show_label=False, interactive=False)
          with gr.Column(elem_classes=['chart-card']):
            if recall_chart:
              gr.Image(value=recall_chart, show_label=False, interactive=False)
          with gr.Column(elem_classes=['chart-card']):
            if f1_chart:
              gr.Image(value=f1_chart, show_label=False, interactive=False)

        with gr.Row(elem_classes=['chart-wide']):
          if roc_path:
            gr.Image(value=roc_path, show_label=False, interactive=False)
          else:
            gr.Markdown('*ROC comparison figure will appear here after training.*')

        if metrics_df is not None and not metrics_df.empty:
          with gr.Accordion('Detailed Metrics Table', open=False):
            table_df = metrics_df.copy()
            for col in ['Precision', 'Recall', 'F1', 'Accuracy', 'ROC AUC', 'PR AUC', 'Specificity']:
              if col in table_df.columns:
                table_df[col] = table_df[col].map(lambda x: f'{x:.4f}')
            gr.Dataframe(value=table_df, interactive=False, wrap=True)

    with gr.Tab('5. Conclusion'):
      with gr.Group(elem_classes=['section-card']):
        gr.HTML(conclusion_html)

    with gr.Tab('6. Live Prediction (Demo)'):
      with gr.Group(elem_classes=['section-card']):
        gr.Markdown('### Single Prediction')
        gr.Markdown('Enter transaction features to test a single payment.')

        with gr.Row(elem_classes=['prediction-shell']):
          with gr.Column(elem_classes=['form-card']):
            with gr.Row():
              amount_input = gr.Number(label='Amount', value=0.0, precision=6)
              v1_input = gr.Number(label='V1', value=0.0, precision=6)
            with gr.Row():
              v2_input = gr.Number(label='V2', value=0.0, precision=6)
              v3_input = gr.Number(label='V3', value=0.0, precision=6)

            with gr.Row():
              sample_btn = gr.Button('Load Sample', variant='secondary')
              predict_btn = gr.Button('Run Prediction', variant='primary')
              reset_btn = gr.Button('Reset', variant='secondary')

          with gr.Column(elem_classes=['form-card']):
            output_pred = gr.HTML()

        sample_btn.click(fn=use_sample_data, outputs=[amount_input, v1_input, v2_input, v3_input])
        reset_btn.click(fn=reset_inputs, outputs=[amount_input, v1_input, v2_input, v3_input])
        predict_btn.click(
          fn=make_prediction,
          inputs=[amount_input, v1_input, v2_input, v3_input],
          outputs=output_pred,
          show_progress='full',
        )

      with gr.Group(elem_classes=['section-card']):
        gr.Markdown('### Batch Prediction')
        gr.Markdown('Upload a CSV file with V1–V28 and Amount columns.')

        with gr.Row(elem_classes=['prediction-shell']):
          with gr.Column(elem_classes=['form-card']):
            csv_input = gr.File(label='Upload CSV', file_types=['.csv'])
            preview_note = gr.Markdown('*Upload a CSV file to preview it.*')
            preview_table = gr.Dataframe(interactive=False)
            process_btn = gr.Button('Process File', variant='primary')

          with gr.Column(elem_classes=['form-card']):
            batch_summary = gr.HTML()
            batch_download = gr.File(label='Download Results')
            batch_result_preview = gr.Dataframe(label='Prediction Preview', interactive=False)

        csv_input.change(fn=preview_uploaded_csv, inputs=csv_input, outputs=[preview_note, preview_table])
        process_btn.click(
          fn=process_batch_predictions,
          inputs=csv_input,
          outputs=[batch_summary, batch_download, batch_result_preview],
          show_progress='full',
        )


if __name__ == '__main__':
    base_port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    selected_port = find_available_port(start_port=base_port, max_tries=20)
    demo.launch(
        share=False,
        server_name='127.0.0.1',
        server_port=selected_port,
      theme=APP_THEME,
      css=CSS,
    )
