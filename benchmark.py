"""
benchmark.py -- Automated Benchmark Tracker for Thermal Respiratory Rate Pipeline.

Calculates and logs:
  1. Regression Metrics: MAE, RMSE, R2, MAPE
  2. Clinical Classification Metrics: Normal (12-20 bpm) vs Tachypnea (>20 bpm) / Bradypnea (<12 bpm)
  3. Tolerance-Band Classification (AAMI/ANSI standard): Hit within +/-1.5 bpm and +/-2.0 bpm
  4. F1-Score (Macro, Weighted, Binary), Precision, Recall
  5. Confusion Matrices (Saved to result/ as plots and logged in benchmark_history.json)
  6. Updates BENCHMARK_REPORT.md for documentation and project presentations.
"""

import json
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "result")
BENCHMARK_HISTORY_FILE = os.path.join(MODELS_DIR, "benchmark_history.json")
REPORT_MD_FILE = os.path.join(BASE_DIR, "BENCHMARK_REPORT.md")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def categorize_bpm(bpm):
    """Categorize continuous BPM into clinical respiratory bands."""
    if bpm is None:
        return "Unknown"
    if bpm < 12.0:
        return "Bradypnea"
    elif bpm <= 20.0:
        return "Normal"
    else:
        return "Tachypnea"


def evaluate_predictions(y_true, y_pred, step_id="step", model_name="Ensemble"):
    """Compute complete regression and classification metrics + confusion matrix."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 1. Regression Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0)

    # 2. Clinical Classification (Normal vs Tachypnea vs Bradypnea)
    labels_order = ["Normal", "Tachypnea"]
    # Check if Bradypnea present
    has_brady = any(y < 12.0 for y in y_true) or any(y < 12.0 for y in y_pred)
    if has_brady:
        labels_order = ["Bradypnea", "Normal", "Tachypnea"]

    y_true_cat = [categorize_bpm(y) for y in y_true]
    y_pred_cat = [categorize_bpm(y) for y in y_pred]

    cm_clinical = confusion_matrix(y_true_cat, y_pred_cat, labels=labels_order).tolist()
    acc_clinical = float(accuracy_score(y_true_cat, y_pred_cat))
    f1_macro = float(f1_score(y_true_cat, y_pred_cat, labels=labels_order, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_cat, y_pred_cat, labels=labels_order, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true_cat, y_pred_cat, labels=labels_order, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_cat, y_pred_cat, labels=labels_order, average="macro", zero_division=0))

    # 3. Clinical Tolerance Classification (ANSI/AAMI Standard: Within +/- 1.5 BPM & +/- 2.0 BPM)
    diff = np.abs(y_true - y_pred)
    
    # Tolerance +/- 1.5 BPM
    hit_1_5 = (diff <= 1.5).astype(int)
    # Target is all 1s (ideal is 100% within tolerance)
    acc_tol_1_5 = float(np.mean(hit_1_5))

    # Tolerance +/- 2.0 BPM
    hit_2_0 = (diff <= 2.0).astype(int)
    acc_tol_2_0 = float(np.mean(hit_2_0))

    # Binary clinical condition: Normal (0) vs Abnormal/Tachypnea (1)
    y_true_abnormal = np.array([1 if (y < 12.0 or y > 20.0) else 0 for y in y_true])
    y_pred_abnormal = np.array([1 if (y < 12.0 or y > 20.0) else 0 for y in y_pred])
    cm_binary = confusion_matrix(y_true_abnormal, y_pred_abnormal, labels=[0, 1]).tolist()
    f1_binary = float(f1_score(y_true_abnormal, y_pred_abnormal, average="binary", zero_division=0))

    # 4. Save Confusion Matrix Plot
    plot_filename = f"confusion_matrix_{step_id}_{model_name.replace(' ', '_').lower()}.png"
    plot_path = os.path.join(RESULT_DIR, plot_filename)
    _save_cm_plot(cm_clinical, labels_order, plot_path, f"{model_name} - {step_id}")

    return {
        "regression": {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
            "MAPE_percent": round(mape, 2),
        },
        "clinical_classification": {
            "labels": labels_order,
            "confusion_matrix": cm_clinical,
            "accuracy": round(acc_clinical, 3),
            "f1_macro": round(f1_macro, 3),
            "f1_weighted": round(f1_weighted, 3),
            "precision_macro": round(prec_macro, 3),
            "recall_macro": round(rec_macro, 3),
        },
        "binary_abnormal_classification": {
            "labels": ["Normal", "Abnormal"],
            "confusion_matrix": cm_binary,
            "f1_score": round(f1_binary, 3),
        },
        "tolerance_rates": {
            "within_1_5_bpm_percent": round(acc_tol_1_5 * 100.0, 1),
            "within_2_0_bpm_percent": round(acc_tol_2_0 * 100.0, 1),
        },
        "plot_path": os.path.relpath(plot_path, BASE_DIR).replace("\\", "/"),
    }


def _save_cm_plot(cm, labels, output_path, title):
    """Plot and save a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=f"Confusion Matrix: {title}",
        ylabel="Ground Truth",
        xlabel="Predicted"
    )

    thresh = np.max(cm) / 2.0 if np.max(cm) > 0 else 1.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i][j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def record_benchmark(step_id, step_name, description, models_eval, baselines_eval=None):
    """Save a new benchmark entry to benchmark_history.json and update BENCHMARK_REPORT.md."""
    history = load_benchmark_history()

    entry = {
        "step_id": step_id,
        "step_name": step_name,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "models": models_eval,
        "baselines": baselines_eval or {},
    }

    # Replace existing step or append
    history = [h for h in history if h["step_id"] != step_id]
    history.append(entry)

    with open(BENCHMARK_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    generate_markdown_report(history)
    print(f"[benchmark] Recorded {step_id}: {step_name}")
    return entry


def load_benchmark_history():
    """Load benchmark history list."""
    if os.path.exists(BENCHMARK_HISTORY_FILE):
        try:
            with open(BENCHMARK_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def generate_markdown_report(history):
    """Generate a comprehensive Markdown report summarizing all benchmark steps."""
    md = [
        "# Respiratory Rate Prediction & Improvement Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This document tracks the incremental progress and accuracy benchmarks across every optimization phase.",
        "",
        "---",
        "",
        "## 1. Incremental Progress Summary (Ensemble Model)",
        "",
        "| Step ID | Step Name | MAE (bpm) | RMSE | R² Score | F1 Macro | Within ±1.5 bpm | Within ±2.0 bpm |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in history:
        sid = item.get("step_id", "")
        sname = item.get("step_name", "")
        ens = (item.get("models") or {}).get("Ensemble") or (item.get("models") or {}).get("Random Forest") or {}
        reg = ens.get("regression", {})
        cls_ = ens.get("clinical_classification", {})
        tol = ens.get("tolerance_rates", {})

        mae = reg.get("MAE", "N/A")
        rmse = reg.get("RMSE", "N/A")
        r2 = reg.get("R2", "N/A")
        f1 = cls_.get("f1_macro", "N/A")
        tol15 = f"{tol.get('within_1_5_bpm_percent', 'N/A')}%"
        tol20 = f"{tol.get('within_2_0_bpm_percent', 'N/A')}%"

        md.append(f"| **{sid}** | {sname} | **{mae}** | {rmse} | {r2} | **{f1}** | {tol15} | {tol20} |")

    md.extend([
        "",
        "---",
        "",
        "## 2. Detailed Step-by-Step Benchmarks",
        "",
    ])

    for item in history:
        sid = item.get("step_id", "")
        sname = item.get("step_name", "")
        desc = item.get("description", "")
        tstamp = item.get("timestamp", "")

        md.extend([
            f"### Phase: {sname} (`{sid}`)",
            f"- **Timestamp**: {tstamp}",
            f"- **Description**: {desc}",
            "",
            "#### Model Performance Comparison Table:",
            "",
            "| Model | MAE | RMSE | R² | F1 Macro | F1 Weighted | Accuracy | Within ±1.5 bpm | Within ±2.0 bpm |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for m_name, m_data in (item.get("models") or {}).items():
            reg = m_data.get("regression", {})
            cls_ = m_data.get("clinical_classification", {})
            tol = m_data.get("tolerance_rates", {})

            md.append(
                f"| **{m_name}** | {reg.get('MAE')} | {reg.get('RMSE')} | {reg.get('R2')} | "
                f"**{cls_.get('f1_macro')}** | {cls_.get('f1_weighted')} | {cls_.get('accuracy')} | "
                f"{tol.get('within_1_5_bpm_percent')}% | {tol.get('within_2_0_bpm_percent')}% |"
            )

        md.extend([
            "",
            "#### Confusion Matrices:",
            "",
        ])

        for m_name, m_data in (item.get("models") or {}).items():
            cls_ = m_data.get("clinical_classification", {})
            labels = cls_.get("labels", [])
            cm = cls_.get("confusion_matrix", [])
            plot_path = m_data.get("plot_path", "")

            md.append(f"**{m_name} (Labels: {', '.join(labels)})**:")
            md.append("")
            # Format matrix table
            header = "| True \\ Pred | " + " | ".join(labels) + " |"
            sep = "| :--- | " + " | ".join([":---:"] * len(labels)) + " |"
            md.append(header)
            md.append(sep)
            for i, row in enumerate(cm):
                md.append(f"| **{labels[i]}** | " + " | ".join(str(val) for val in row) + " |")
            md.append("")
            if plot_path and os.path.exists(os.path.join(BASE_DIR, plot_path)):
                md.append(f"![{m_name} Confusion Matrix]({plot_path})")
                md.append("")

        md.extend(["---", ""])

    with open(REPORT_MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    print(f"Benchmark module initialized. History file: {BENCHMARK_HISTORY_FILE}")
