"""Credit Card Fraud Detection - Professional Demo UI"""

import os
import sys
import socket
import tempfile
import pandas as pd
import gradio as gr

sys.path.insert(0, "./src")
from predict import FraudDetectionPredictor
from config import MODELS_PATH, FIGURES_PATH, FEATURE_COLS, METRICS_PATH


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

    key_cols = [c for c in ["precision", "recall", "f1"] if c in df.columns]
    if not key_cols:
        return None, "Unknown"

    view_df = df[key_cols].copy().round(4)
    best_idx = view_df["f1" if "f1" in view_df.columns else key_cols[0]].idxmax() if len(key_cols) > 0 else view_df.index[0]
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
            return "<div class='result-card result-error'>⚠️ " + model_not_loaded_message() + "</div>"

        features = build_features(amount, v1, v2, v3)
        result = predictor.predict(features)

        is_fraud = result["prediction"] == "Fraudulent"
        label = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
        card_class = "result-fraud" if is_fraud else "result-normal"
        model_label = ACTIVE_MODEL_NAME.replace("_", " ").title()

        return f"""
        <div class="result-card {card_class}">
            <div class="result-title">{label}</div>
            <div class="result-prob">Confidence: <b>{result['fraud_probability']:.1%}</b></div>
            <div class="result-model">Model: {model_label}</div>
        </div>
        """
    except Exception as exc:
        return f"<div class='result-card result-error'>❌ Error: {str(exc)}</div>"


def preview_uploaded_csv(file):
    if file is None:
        return "Upload a CSV file to preview it.", pd.DataFrame()

    try:
        df = pd.read_csv(file.name)
        return f"✅ Previewing: {os.path.basename(file.name)} ({len(df)} rows)", df.head(10)
    except Exception as exc:
        return f"❌ Error reading CSV: {str(exc)}", pd.DataFrame()


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

        summary = f"""
✅ **Batch processing complete**
- Total transactions: {total}
- Fraudulent: {fraud_count} ({fraud_count/total*100:.1f}%)
- Legitimate: {legitimate_count} ({legitimate_count/total*100:.1f}%)
        """.strip()
        return summary, output_path, result_df.head(20)
    except Exception as exc:
        return f"❌ Error: {str(exc)}", None, pd.DataFrame()


def find_available_port(start_port=7860, max_tries=20):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {start_port}-{start_port + max_tries - 1}.")


def build_model_cards_html(metrics_df, best_model):
    if metrics_df is None or metrics_df.empty:
        return "<div class='no-data'>📊 No metrics available. Run training first.</div>"

    cards = []
    for _, row in metrics_df.iterrows():
        model_name = str(row.get("Model", "Unknown"))
        is_best = model_name == best_model
        best_badge = "<div class='badge-best'>🏆 BEST</div>" if is_best else ""
        card_class = "model-card best-card" if is_best else "model-card"

        precision = float(row.get('Precision', 0))
        recall = float(row.get('Recall', 0))
        f1 = float(row.get('F1', 0))

        cards.append(f"""
        <div class="{card_class}">
          {best_badge}
          <h4 class="card-title">{model_name}</h4>
          <div class="metric">
            <span class="label">Precision</span>
            <span class="value">{precision:.4f}</span>
          </div>
          <div class="metric">
            <span class="label">Recall</span>
            <span class="value">{recall:.4f}</span>
          </div>
          <div class="metric">
            <span class="label">F1-score</span>
            <span class="value">{f1:.4f}</span>
          </div>
        </div>
        """)

    return f"<div class='model-cards-grid'>{''.join(cards)}</div>"


CSS = """
/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-12px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(12px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.8;
  }
}

/* ===== GLOBAL ===== */
body, .gradio-container {
  background: linear-gradient(to bottom, #f8fafc 0%, #ffffff 100%) !important;
  animation: fadeIn 0.6s ease-out;
}

.gradio-container {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* ===== HEADER ===== */
.header-wrap {
  text-align: center;
  padding: 40px 20px 20px 20px;
  margin-bottom: 8px;
  animation: fadeIn 0.8s ease-out 0.2s both;
}

.title {
  font-size: 52px;
  font-weight: 800;
  color: #0f172a;
  margin: 0 0 12px 0;
  letter-spacing: -1.2px;
  animation: slideInLeft 0.8s ease-out 0.3s both;
}

.subtitle {
  font-size: 20px;
  color: #64748b;
  font-weight: 500;
  margin: 0;
  animation: slideInRight 0.8s ease-out 0.4s both;
}

/* ===== SECTIONS ===== */
.section-card {
  background: #ffffff;
  border-radius: 0;
  padding: 40px 60px;
  margin: 0 0 2px 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  border: none;
  border-bottom: 1px solid #e2e8f0;
  animation: fadeIn 1s ease-out 0.5s both;
  width: 100%;
  box-sizing: border-box;
}

.section-card:last-of-type {
  border-bottom: none;
}

.section-card h3 {
  font-size: 28px;
  font-weight: 700;
  color: #0f172a;
  margin: 0 0 24px 0;
  animation: slideInLeft 0.6s ease-out 0.6s both;
}

/* ===== MODEL CARDS ===== */
.model-cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 28px;
  margin-bottom: 24px;
  overflow: hidden;
}

.model-card {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  border: 2px solid #e2e8f0;
  border-radius: 16px;
  padding: 32px 28px;
  box-shadow: 0 2px 16px rgba(0, 0, 0, 0.06);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  animation: fadeIn 0.8s ease-out both;
}

.model-card:nth-child(1) {
  animation-delay: 0.6s;
}

.model-card:nth-child(2) {
  animation-delay: 0.7s;
}

.model-card:nth-child(3) {
  animation-delay: 0.8s;
}

.model-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.16);
  border-color: #cbd5e1;
}

.best-card {
  border: 2px solid #10b981;
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #ffffff 100%);
  box-shadow: 0 4px 24px rgba(16, 185, 129, 0.2);
  animation: pulse 3s ease-in-out infinite, fadeIn 0.8s ease-out 0.6s both !important;
}

.best-card:hover {
  box-shadow: 0 16px 48px rgba(16, 185, 129, 0.35);
}

.badge-best {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #d1fae5;
  color: #047857;
  font-size: 12px;
  font-weight: 700;
  padding: 8px 14px;
  border-radius: 8px;
  animation: slideInRight 0.6s ease-out 0.8s both;
}

.card-title {
  font-size: 24px;
  font-weight: 700;
  color: #0f172a;
  margin: 0 0 20px 0;
  animation: slideInLeft 0.5s ease-out 0.9s both;
}

.metric {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 0;
  border-bottom: 1px solid #f1f5f9;
  animation: fadeIn 0.6s ease-out 1s both;
}

.metric:last-child {
  border-bottom: none;
}

.label {
  font-size: 15px;
  color: #64748b;
  font-weight: 600;
}

.value {
  font-size: 20px;
  color: #0f172a;
  font-weight: 700;
}

/* ===== RESULT CARD ===== */
.result-card {
  text-align: center;
  padding: 36px;
  border-radius: 20px;
  max-width: 800px;
  margin: 24px auto 0;
  border: 3px solid;
  background: #ffffff;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
  animation: fadeIn 0.8s ease-out 0.8s both;
}

.result-title {
  font-size: 44px;
  font-weight: 800;
  margin: 0 0 16px 0;
  line-height: 1.1;
  animation: slideInUp 0.6s ease-out 1s both;
}

.result-prob {
  font-size: 26px;
  font-weight: 600;
  margin: 16px 0 0 0;
  animation: fadeIn 0.6s ease-out 1.1s both;
}

.result-model {
  font-size: 15px;
  margin-top: 16px;
  opacity: 0.7;
  font-weight: 500;
  animation: fadeIn 0.6s ease-out 1.2s both;
}

.result-fraud {
  border-color: #ef4444;
  background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
  color: #991b1b;
}

.result-normal {
  border-color: #10b981;
  background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
  color: #065f46;
}

.result-error {
  border-color: #f59e0b;
  background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
  color: #92400e;
}

/* ===== DATA TABLE ===== */
.gradio-container table {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
  animation: fadeIn 0.8s ease-out 0.75s both;
}

.gradio-container table th {
  background: linear-gradient(135deg, #f1f5f9, #ffffff) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  border-bottom: 2px solid #cbd5e1 !important;
  animation: slideInUp 0.5s ease-out 0.8s both;
}

.gradio-container table td {
  color: #475569 !important;
  border-color: #e2e8f0 !important;
  animation: fadeIn 0.6s ease-out 0.85s both;
}

/* ===== INPUTS ===== */
.gradio-container input, .gradio-container textarea {
  border-radius: 10px !important;
  border: 1px solid #cbd5e1 !important;
  background: #ffffff !important;
  font-size: 15px !important;
  transition: all 0.3s ease !important;
  animation: fadeIn 0.8s ease-out 0.65s both;
}

.gradio-container input:focus, .gradio-container textarea:focus {
  border-color: #10b981 !important;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
  transform: scale(1.01);
}

.gradio-container input:hover, .gradio-container textarea:hover {
  border-color: #94a3b8 !important;
}

/* ===== BUTTONS ===== */
button {
  min-height: 46px !important;
  border-radius: 10px !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
  animation: fadeIn 0.8s ease-out 0.7s both;
}

button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
}

button:active {
  transform: translateY(0) scale(0.98) !important;
}

/* ===== FILE UPLOAD ===== */
.gradio-container .file-contents {
  border-radius: 12px !important;
  background: #f8fafc !important;
  border: 1px solid #e2e8f0 !important;
}

/* ===== UTILITY ===== */
.no-data {
  text-align: center;
  padding: 32px;
  color: #64748b;
  font-size: 18px;
  opacity: 0.8;
}
"""


# Build UI
metrics_df, best_model = load_comparison_data()
cards_html = build_model_cards_html(metrics_df, best_model)

with gr.Blocks(title="Credit Card Fraud Detection") as demo:
    # Header
    gr.Markdown(f"""
    <div class="header-wrap">
      <div class="title">💳 Credit Card Fraud Detection</div>
      <div class="subtitle">Machine Learning Final Project</div>
    </div>
    """)

    # Model Summary Section
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### 📊 Model Performance Summary")
        gr.Markdown(f"**Best Model:** {best_model}")
        gr.HTML(cards_html)
        if metrics_df is not None and not metrics_df.empty:
            gr.Dataframe(value=metrics_df, interactive=False, wrap=True)

    # ROC Curve Section
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### 📈 ROC Curve Comparison")
        roc_path = load_roc_image_path()
        if roc_path:
            gr.Image(value=roc_path)
        else:
            gr.Markdown("*Comparison chart unavailable. Run training first.*")

    # Single Prediction Section
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### 🔎 Test Single Transaction")
        gr.Markdown("*Enter transaction features to predict if it's fraudulent.*")

        with gr.Row():
            amount_input = gr.Number(label="Amount", value=0.0, precision=6)
            v1_input = gr.Number(label="V1", value=0.0, precision=6)
            v2_input = gr.Number(label="V2", value=0.0, precision=6)
            v3_input = gr.Number(label="V3", value=0.0, precision=6)

        with gr.Row():
            sample_btn = gr.Button("📋 Use Sample", variant="secondary")
            predict_btn = gr.Button("🔍 Predict", variant="primary")
            reset_btn = gr.Button("🔄 Reset", variant="secondary")

        output_pred = gr.HTML()

        sample_btn.click(fn=use_sample_data, outputs=[amount_input, v1_input, v2_input, v3_input])
        reset_btn.click(fn=reset_inputs, outputs=[amount_input, v1_input, v2_input, v3_input])
        predict_btn.click(
            fn=make_prediction,
            inputs=[amount_input, v1_input, v2_input, v3_input],
            outputs=output_pred,
            show_progress="full",
        )

    # Batch Prediction Section
    with gr.Group(elem_classes=["section-card"]):
        gr.Markdown("### 📦 Batch Prediction")
        gr.Markdown("*Upload a CSV file with V1-V28 and Amount columns.*")

        with gr.Row():
            csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            process_btn = gr.Button("▶️ Process", variant="primary", scale=0)

        preview_note = gr.Markdown("*Upload a CSV file to preview it.*")
        preview_table = gr.Dataframe(interactive=False)

        batch_summary = gr.Markdown()
        batch_result_preview = gr.Dataframe(label="Prediction Results", interactive=False)
        batch_download = gr.File(label="📥 Download Results")

        csv_input.change(fn=preview_uploaded_csv, inputs=csv_input, outputs=[preview_note, preview_table])
        process_btn.click(
            fn=process_batch_predictions,
            inputs=csv_input,
            outputs=[batch_summary, batch_download, batch_result_preview],
            show_progress="full",
        )


if __name__ == "__main__":
    base_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    selected_port = find_available_port(start_port=base_port, max_tries=20)
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=selected_port,
        theme=gr.themes.Soft(),
        css=CSS,
    )
