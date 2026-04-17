"""Credit Card Fraud Detection - Demo UI"""

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
    return "Model not loaded. Please run training first."


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
    view_df.columns = ["Precision", "Recall", "F1"][: len(view_df.columns)]
    best_idx = view_df["F1"].idxmax() if "F1" in view_df.columns else view_df.index[0]
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
            return (
                "<div style='padding:18px;border-radius:12px;background:#2a2a2a;text-align:center;'>"
                f"{model_not_loaded_message()}"
                "</div>"
            )

        features = build_features(amount, v1, v2, v3)
        result = predictor.predict(features)

        is_fraud = result["prediction"] == "Fraudulent"
        label = "FRAUD DETECTED" if is_fraud else "LEGITIMATE TRANSACTION"
        bg = "#45181b" if is_fraud else "#143825"
        border = "#ff4d4f" if is_fraud else "#2ecc71"

        return f"""
        <div style="text-align:center;padding:28px;border-radius:16px;background:{bg};border:2px solid {border};max-width:720px;margin:0 auto;">
            <div style="font-size:34px;font-weight:900;letter-spacing:0.5px;">{label}</div>
            <div style="font-size:20px;margin-top:10px;">Probability: <b>{result['fraud_probability']:.2%}</b></div>
            <div style="font-size:13px;opacity:0.85;margin-top:8px;">Model: {ACTIVE_MODEL_NAME}</div>
        </div>
        """
    except Exception as exc:
        return f"<div style='padding:14px;border-radius:10px;background:#45181b;color:#ffd6d6;'>Error: {str(exc)}</div>"


def process_batch_predictions(file):
    if file is None:
        return "Upload a CSV file", None

    try:
        predictor = get_predictor()
        if predictor is None:
            return model_not_loaded_message(), None

        df = pd.read_csv(file.name)

        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            return f"Missing columns: {', '.join(missing)}", None

        result_df = predictor.batch_predict(df[FEATURE_COLS])
        fraud_count = int((result_df["Prediction"] == 1).sum())
        total = len(result_df)
        legitimate_count = total - fraud_count

        with tempfile.NamedTemporaryFile(delete=False, suffix="_predictions.csv") as tmp:
            result_df.to_csv(tmp.name, index=False)
            output_path = tmp.name

        summary = (
            f"Total rows: {total}\n"
            f"Fraud count: {fraud_count}\n"
            f"Legitimate count: {legitimate_count}"
        )
        return summary, output_path
    except Exception as exc:
        return f"Error: {str(exc)}", None


def find_available_port(start_port=7860, max_tries=20):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(
        f"Cannot find empty port in range: {start_port}-{start_port + max_tries - 1}."
    )


CSS = """
body { background: #0f1117; }
.gradio-container { max-width: 980px !important; }
.title-wrap { text-align:center; margin: 10px 0 14px 0; }
.sub { opacity: 0.9; font-size: 16px; margin-top: 4px; }
.small-note { opacity: 0.95; font-size: 14px; margin-top: 8px; margin-bottom: 8px; }
.block { padding-top: 8px !important; padding-bottom: 8px !important; }
button { min-height: 52px !important; font-size: 16px !important; border-radius: 10px !important; }
"""


with gr.Blocks(title="Credit Card Fraud Detection System") as demo:
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
        gr.Markdown(f"### Best Model: {best_model}")
        if metrics_df is not None:
            gr.DataFrame(metrics_df, label="Model | Precision | Recall | F1", interactive=False)
        else:
            gr.Markdown("Metrics unavailable")

        roc_path = load_roc_image_path()
        if roc_path:
            gr.Image(roc_path, label="ROC / Comparison")

    with gr.Tab("Predict"):
        gr.Markdown("### Predict Transaction")

        with gr.Row():
            amount_input = gr.Number(label="Amount", value=0.0, precision=6)
            v1_input = gr.Number(label="V1", value=0.0, precision=6)
            v2_input = gr.Number(label="V2", value=0.0, precision=6)
            v3_input = gr.Number(label="V3", value=0.0, precision=6)

        with gr.Row():
            sample_btn = gr.Button("Use Sample Transaction", variant="secondary", scale=1)
            predict_btn = gr.Button("Predict", variant="primary", scale=1)
            reset_btn = gr.Button("Reset", variant="secondary", scale=1)

        output_pred = gr.HTML(label="")

        sample_btn.click(fn=use_sample_data, outputs=[amount_input, v1_input, v2_input, v3_input])
        reset_btn.click(fn=reset_inputs, outputs=[amount_input, v1_input, v2_input, v3_input])
        predict_btn.click(fn=make_prediction, inputs=[amount_input, v1_input, v2_input, v3_input], outputs=output_pred)

    with gr.Tab("Batch Prediction"):
        gr.Markdown("Upload CSV with V1–V28 and Amount")

        with gr.Row():
            csv_input = gr.File(label="CSV File", file_types=[".csv"])
            process_btn = gr.Button("Process", variant="primary")

        batch_summary = gr.Textbox(label="Summary", interactive=False)
        batch_download = gr.File(label="Download Results")

        process_btn.click(
            fn=process_batch_predictions,
            inputs=csv_input,
            outputs=[batch_summary, batch_download],
        )


if __name__ == "__main__":
    base_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    selected_port = find_available_port(start_port=base_port, max_tries=20)
    print("\nLaunching Credit Card Fraud Detection System")
    print(f"Open: http://127.0.0.1:{selected_port}\n")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=selected_port,
        theme=gr.themes.Base(),
        css=CSS,
    )
