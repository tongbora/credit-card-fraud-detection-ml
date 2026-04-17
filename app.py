"""Credit Card Fraud Detection - Demo UI"""

import os
import sys
import tempfile
import pandas as pd
import gradio as gr

sys.path.insert(0, "./src")
from predict import FraudDetectionPredictor
from config import MODELS_PATH, FIGURES_PATH, FEATURE_COLS, METRICS_PATH


PREDICTOR = None
ACTIVE_MODEL_NAME = ""


def get_predictor():
    global PREDICTOR, ACTIVE_MODEL_NAME
    if PREDICTOR is not None:
        return PREDICTOR

    scaler_path = f"{MODELS_PATH}/scaler.pkl"
    candidate_models = [
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

    raise FileNotFoundError("No trained model/scaler found in models/. Run training first.")


def load_comparison_data():
    metrics_path = f"{METRICS_PATH}/model_comparison.csv"
    if not os.path.exists(metrics_path):
        return None, "Metrics file not found"

    df = pd.read_csv(metrics_path, index_col=0)
    key_cols = [c for c in ["precision", "recall", "f1"] if c in df.columns]
    if not key_cols:
        return None, "Required columns not found in metrics file"

    view_df = df[key_cols].copy().round(4)
    view_df.columns = ["Precision", "Recall", "F1-score"][: len(view_df.columns)]
    best_idx = view_df["F1-score"].idxmax() if "F1-score" in view_df.columns else view_df.index[0]
    return view_df.reset_index().rename(columns={"index": "Model"}), str(best_idx)


def load_roc_image_path():
    candidate = f"{FIGURES_PATH}/model_comparison_summary.png"
    return candidate if os.path.exists(candidate) else None


def build_features(values):
    return {f"V{i}": float(values[i - 1]) for i in range(1, 29)} | {"Amount": float(values[28])}


def use_sample_data():
    return [
        -1.358,
        -0.043,
        2.136,
        1.465,
        -0.619,
        -0.991,
        -0.305,
        0.085,
        0.159,
        -0.046,
        -0.073,
        -0.268,
        -0.539,
        -0.055,
        0.040,
        0.085,
        -0.255,
        -0.171,
        -0.046,
        -0.351,
        -0.148,
        -0.420,
        0.048,
        0.102,
        0.191,
        -0.328,
        0.047,
        0.005,
        149.62,
    ]


def reset_inputs():
    return [0.0] * 28 + [0.0]


def make_prediction(*values):
    try:
        predictor = get_predictor()
        features = build_features(values)
        result = predictor.predict(features)

        is_fraud = result["prediction"] == "Fraudulent"
        label = "Fraud" if is_fraud else "Not Fraud"
        bg = "#3a1115" if is_fraud else "#102f20"
        border = "#ff4d4f" if is_fraud else "#2ecc71"

        return f"""
        <div style="text-align:center;padding:24px;border-radius:14px;background:{bg};border:2px solid {border};">
            <div style="font-size:30px;font-weight:800;">{label}</div>
            <div style="font-size:18px;margin-top:8px;">Probability: <b>{result['fraud_probability']:.2%}</b></div>
            <div style="font-size:13px;opacity:0.85;margin-top:8px;">Model: {ACTIVE_MODEL_NAME}</div>
        </div>
        """
    except Exception as exc:
        return f"<div style='padding:14px;border-radius:10px;background:#3a1115;color:#ffd6d6;'>Error: {str(exc)}</div>"


def process_batch_predictions(file):
    if file is None:
        return "Upload a CSV file", None, None

    try:
        predictor = get_predictor()
        df = pd.read_csv(file.name)

        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            return f"Missing columns: {', '.join(missing)}", None, None

        result_df = predictor.batch_predict(df[FEATURE_COLS])
        fraud_count = int((result_df["Prediction"] == 1).sum())
        total = len(result_df)

        with tempfile.NamedTemporaryFile(delete=False, suffix="_predictions.csv") as tmp:
            result_df.to_csv(tmp.name, index=False)
            output_path = tmp.name

        summary = f"Processed: {total} | Fraud: {fraud_count} | Not Fraud: {total - fraud_count}"
        return summary, result_df, output_path
    except Exception as exc:
        return f"Error: {str(exc)}", None, None


CSS = """
body { background: #0f1117; }
.gradio-container { max-width: 1200px !important; }
.title-wrap { text-align:center; margin-bottom: 8px; }
.sub { opacity: 0.9; font-size: 16px; }
.small-note { opacity: 0.9; font-size: 14px; margin-bottom: 6px; }
button { min-height: 48px !important; font-size: 16px !important; }
"""


with gr.Blocks(title="Credit Card Fraud Detection System", theme=gr.themes.Base(), css=CSS) as demo:
    gr.Markdown(
        """
        <div class="title-wrap">
          <h1>Credit Card Fraud Detection System</h1>
          <div class="sub">ML-powered fraud detection demo</div>
          <div class="small-note">Credit Card Fraud Detection System<br>Detect fraudulent transactions using machine learning models.<br>Use the tabs to test predictions and compare models.</div>
        </div>
        """
    )

    with gr.Tab("Model Comparison"):
        metrics_df, best_model = load_comparison_data()
        gr.Markdown(f"### Best Model: {best_model}" if best_model != "Metrics file not found" else "### Best Model: N/A")
        if metrics_df is not None:
            gr.DataFrame(metrics_df, label="Precision / Recall / F1-score", interactive=False)
        else:
            gr.Markdown("Metrics unavailable")

        roc_path = load_roc_image_path()
        if roc_path:
            gr.Image(roc_path, label="ROC / Comparison", show_download_button=False)

    with gr.Tab("Single Prediction"):
        gr.Markdown("### Enter transaction values")

        v_inputs = []
        for row_start in [1, 5, 9, 13, 17, 21, 25]:
            with gr.Row():
                for i in range(row_start, min(row_start + 4, 29)):
                    v_inputs.append(gr.Number(label=f"V{i}", value=0.0, precision=6))

        amount_input = gr.Number(label="Amount", value=0.0, precision=6)

        with gr.Row():
            sample_btn = gr.Button("Use Sample Data", variant="secondary", scale=1)
            predict_btn = gr.Button("Predict", variant="primary", scale=1)
            reset_btn = gr.Button("Reset", variant="secondary", scale=1)

        output_pred = gr.HTML(label="Result")

        sample_btn.click(fn=use_sample_data, outputs=v_inputs + [amount_input])
        reset_btn.click(fn=reset_inputs, outputs=v_inputs + [amount_input])
        predict_btn.click(fn=make_prediction, inputs=v_inputs + [amount_input], outputs=output_pred)

    with gr.Tab("Batch Prediction"):
        gr.Markdown("Upload CSV with V1–V28 and Amount")

        with gr.Row():
            csv_input = gr.File(label="CSV File", file_types=[".csv"])
            process_btn = gr.Button("Process", variant="primary")

        batch_summary = gr.Textbox(label="Status", interactive=False)
        batch_table = gr.Dataframe(label="Results", interactive=False)
        batch_download = gr.File(label="Download Results")

        process_btn.click(
            fn=process_batch_predictions,
            inputs=csv_input,
            outputs=[batch_summary, batch_table, batch_download],
        )


if __name__ == "__main__":
    print("\nLaunching Credit Card Fraud Detection System")
    print("Open: http://127.0.0.1:7860\n")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
