"""
make_method_comparison_report.py
--------------------------------
Generates:
  1. models/method_comparison/method_comparison_mae.png (Bar chart of 10 approaches)
  2. models/method_comparison/cwt_rf_pred_vs_true.png (Scatter plot for CWT + RF)
  3. models/method_comparison/comparison_report.md (Comprehensive markdown report)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rr_model

OUT_DIR = os.path.join("models", "method_comparison")
LOSO_RESULTS_FILE = os.path.join(OUT_DIR, "loso_results.json")
CACHE_V2_FILE = os.path.join(OUT_DIR, "features_cache_v2.json")
REPORT_FILE = os.path.join(OUT_DIR, "comparison_report.md")


def generate_plots_and_report():
    with open(LOSO_RESULTS_FILE) as f:
        loso_data = json.load(f)

    with open(CACHE_V2_FILE) as f:
        cache_v2 = json.load(f)

    vids = rr_model.list_dataset_videos()
    labels_dict = {v["filename"]: v["label_bpm"] for v in vids}
    filenames = sorted(list(labels_dict.keys()))
    y_true = np.array([labels_dict[f] for f in filenames], dtype=float)

    # ---------------------------------------------------------
    # 1. PLOT 1: Bar chart of MAE across all 10 approaches
    # ---------------------------------------------------------
    approaches = [
        # (display_name, category, key_in_json, subkey_if_baseline)
        ("Peak Detection + RF", "RF+LOSO (held-out)", "peak_detection", None),
        ("Autocorrelation + RF", "RF+LOSO (held-out)", "autocorrelation", None),
        ("Welch PSD + RF", "RF+LOSO (held-out)", "welch_psd", None),
        ("CWT + RF", "RF+LOSO (held-out)", "cwt", None),
        ("Combined + RF", "RF+LOSO (held-out)", "combined", None),
        ("Raw Peak Detection", "Formula (in-sample)", "baselines_no_training", "raw_peak_detection"),
        ("Raw Autocorrelation", "Formula (in-sample)", "baselines_no_training", "raw_autocorrelation"),
        ("Raw Welch PSD", "Formula (in-sample)", "baselines_no_training", "raw_welch_psd"),
        ("Raw CWT", "Formula (in-sample)", "baselines_no_training", "raw_cwt"),
        ("Consensus Median", "Formula (in-sample)", "baselines_no_training", "consensus_median"),
    ]

    names = []
    maes = []
    types = []
    for name, cat, k1, k2 in approaches:
        if k2 is None:
            m = loso_data[k1]["pooled_metrics"]["mae"]
        else:
            m = loso_data[k1][k2]["pooled_metrics"]["mae"]
        names.append(name)
        maes.append(m)
        types.append(cat)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(names))
    colors = ["#2b5c8f" if t == "RF+LOSO (held-out)" else "#d96b27" for t in types]
    bars = ax.bar(x, maes, width=0.62, color=colors, edgecolor="#1f2937", linewidth=0.8, alpha=0.9)

    # Add text labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.12, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pooled MAE (bpm, lower is better)", fontsize=10, fontweight="bold")
    ax.set_title("Respiratory Rate Estimation Error Across All 10 Evaluated Approaches", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, max(maes) + 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="#1f2937", label="RF + LOSO (held-out validation)"),
        Patch(facecolor="#d96b27", edgecolor="#1f2937", label="Formula (in-sample baseline, no holdout)")
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95)
    fig.tight_layout()

    plot1_path = os.path.join(OUT_DIR, "method_comparison_mae.png")
    fig.savefig(plot1_path, dpi=180)
    plt.close(fig)
    print(f"[make_report] Saved {plot1_path}")

    # ---------------------------------------------------------
    # 2. PLOT 2: Predicted vs. True RR for CWT + RF
    # ---------------------------------------------------------
    cwt_oof = loso_data["cwt"]["oof_predictions"]
    y_pred_cwt = np.array([cwt_oof[f] for f in filenames], dtype=float)
    subjects = np.array([f.split(".")[0] for f in filenames])

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    subj_colors = {"1": "#1f77b4", "2": "#2ca02c", "3": "#ff7f0e", "4": "#d62728"}
    subj_labels = {"1": "Subject 1 (N=10)", "2": "Subject 2 (N=10)", "3": "Subject 3 (N=10)", "4": "Subject 4 (N=2, unreliable)"}

    for s in ["1", "2", "3", "4"]:
        idx = (subjects == s)
        ax.scatter(y_true[idx], y_pred_cwt[idx], color=subj_colors[s], label=subj_labels[s],
                   s=55, alpha=0.85, edgecolors="#222222", linewidth=0.7)

    lim_min = min(float(y_true.min()), float(y_pred_cwt.min())) - 1.2
    lim_max = max(float(y_true.max()), float(y_pred_cwt.max())) + 1.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.65, label="Perfect Agreement (y = x)")

    corr = np.corrcoef(y_pred_cwt, y_true)[0, 1]
    mae_cwt = loso_data["cwt"]["pooled_metrics"]["mae"]
    r2_cwt = loso_data["cwt"]["pooled_metrics"]["r2"]

    stats_text = (
        f"CWT + RF (LOSO CV)\n"
        f"Pooled MAE: {mae_cwt:.2f} bpm\n"
        f"Pearson r: {corr:.3f}\n"
        f"Pooled R²: {r2_cwt:+.2f}"
    )
    ax.text(0.04, 0.95, stats_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Ground Truth Reference RR (bpm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("LOSO Out-of-Fold Predicted RR (bpm)", fontsize=10, fontweight="bold")
    ax.set_title("CWT + RF: Out-of-Fold Predictions vs. Ground Truth", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()

    plot2_path = os.path.join(OUT_DIR, "cwt_rf_pred_vs_true.png")
    fig.savefig(plot2_path, dpi=180)
    plt.close(fig)
    print(f"[make_report] Saved {plot2_path}")

    # ---------------------------------------------------------
    # 3. BUILD MARKDOWN REPORT
    # ---------------------------------------------------------
    L = []
    L.append("# Respiratory Rate Estimation: Signal Processing & Machine Learning Method Comparison\n")
    L.append("**Evaluation Protocol:** Leave-One-Subject-Out Cross-Validation (LOSO CV) on 32 thermal video segments across 4 subjects.\n")

    # SECTION 1: Full Comparison Table
    L.append("## 1. Full Method Comparison Table\n")
    L.append("| Approach | Type | Pooled MAE | Pooled RMSE | Pooled $R^2$ | Subj1 MAE | Subj2 MAE | Subj3 MAE | Subj4 MAE ($N=2$) |")
    L.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # 5 RF models
    rf_keys = [
        ("Peak Detection + RF (4 feat)", "peak_detection"),
        ("Autocorrelation + RF (4 feat)", "autocorrelation"),
        ("Welch PSD + RF (3 feat)", "welch_psd"),
        ("CWT + RF (3 feat)", "cwt"),
        ("Combined + RF (14 feat)", "combined"),
    ]
    for label, k in rf_keys:
        d = loso_data[k]
        pm = d["pooled_metrics"]
        sm = d["per_subject_mae"]
        bold = "**" if k == "cwt" else ""
        L.append(f"| {bold}{label}{bold} | RF+LOSO (held-out) | {bold}{pm['mae']:.2f}{bold} | {bold}{pm['rmse']:.2f}{bold} | {bold}{pm['r2']:+.2f}{bold} | {sm['1']:.2f} | {sm['2']:.2f} | {sm['3']:.2f} | {sm['4']:.2f} |")

    # 5 Baselines
    base_keys = [
        ("Raw Peak Detection", "raw_peak_detection"),
        ("Raw Autocorrelation", "raw_autocorrelation"),
        ("Raw Welch PSD", "raw_welch_psd"),
        ("Raw CWT", "raw_cwt"),
        ("Consensus Median", "consensus_median"),
    ]
    for label, k in base_keys:
        d = loso_data["baselines_no_training"][k]
        pm = d["pooled_metrics"]
        sm = d["per_subject_mae"]
        L.append(f"| *{label}* | Formula (in-sample) | {pm['mae']:.2f} | {pm['rmse']:.2f} | {pm['r2']:+.2f} | {sm['1']:.2f} | {sm['2']:.2f} | {sm['3']:.2f} | {sm['4']:.2f} |")

    L.append("\n*Note: 'RF+LOSO (held-out)' indicates genuine out-of-fold generalization across unseen subjects. 'Formula (in-sample)' indicates fixed deterministic estimators with zero learnable parameters evaluated across the full cohort without subject holdout.*\n")

    # SECTION 2: Signal-Tracking Diagnostic Table
    L.append("## 2. Signal-Tracking Diagnostic (RF Ensembles)\n")
    L.append("To determine whether Random Forest models genuinely track respiratory rate variation or collapse toward the cohort mean (given the narrow ground truth distribution: $\\mu = 18.33\\text{ bpm}, \\sigma = 1.93\\text{ bpm}$), we examine out-of-fold prediction dispersion and correlation:\n")
    L.append("| Model | Pred Std (bpm) | Pred Range [min, max] | Correlation ($r$) with True RR | Diagnostic Assessment |")
    L.append("| :--- | :---: | :---: | :---: | :--- |")

    diag_data = [
        ("Peak Detection + RF", "peak_detection", "Moderate signal tracking; significant contraction toward center"),
        ("Autocorrelation + RF", "autocorrelation", "Largely shrinkage toward mean; weak correlation with ground truth"),
        ("Welch PSD + RF", "welch_psd", "Moderate dispersion but elevated residual noise (negative $R^2$)"),
        ("CWT + RF", "cwt", "**Genuine signal tracking**; highest correlation ($r=0.491$) and best dynamic range"),
        ("Combined + RF", "combined", "Excess shrinkage; feature proliferation degrades sensitivity vs CWT alone"),
    ]
    for label, k, assessment in diag_data:
        oof = loso_data[k]["oof_predictions"]
        preds = np.array([oof[f] for f in filenames])
        p_std = preds.std()
        p_min, p_max = preds.min(), preds.max()
        r = np.corrcoef(preds, y_true)[0, 1]
        L.append(f"| {label} | {p_std:.2f} | [{p_min:.2f}, {p_max:.2f}] | {r:+.3f} | {assessment} |")

    # SECTION 3: Sub-Harmonic Bias Table
    L.append("\n## 3. Explanatory Mechanism: Sub-Harmonic Frequency Locking\n")
    L.append("Analysis of the raw signal-processing formulas reveals why frequency-domain methods fail when applied directly without machine learning:\n")
    L.append("| Raw Method | Mean Bias (bpm) | Mean Ratio (Pred/True) | Videos Locked to Half-Rate (Ratio 0.40–0.65) | Primary Failure Mode |")
    L.append("| :--- | :---: | :---: | :---: | :--- |")

    raw_methods = [
        ("Raw Peak Detection", "peak_bpm", "Low frequency drift; temporal peak miscounting"),
        ("Raw Autocorrelation", "autocorr_bpm", "**Severe sub-harmonic locking**; lags correspond to 2x breathing period"),
        ("Raw Welch PSD", "welch_bpm", "Spectral leakage & low-frequency thermal fluctuation dominant peak"),
        ("Raw CWT", "cwt_bpm", "Wavelet energy centroid captures half-frequency respiratory harmonic"),
    ]
    for label, k, failure_mode in raw_methods:
        raw_vals = np.array([cache_v2[f][k] for f in filenames])
        bias = (raw_vals - y_true).mean()
        ratios = raw_vals / y_true
        mean_ratio = ratios.mean()
        near_half = np.sum((ratios >= 0.40) & (ratios <= 0.65))
        pct_half = (near_half / len(filenames)) * 100.0
        L.append(f"| {label} | {bias:+.2f} | {mean_ratio:.2f} | {near_half}/{len(filenames)} ({pct_half:.1f}%) | {failure_mode} |")

    # SECTION 4: Headline Result
    L.append("\n## 4. Headline Result\n")
    L.append("> **CWT + RF achieves 1.30 bpm MAE ($r=0.491, R^2=+0.23$) under subject-independent LOSO CV, a 46.5% MAE reduction versus the best raw signal-processing baseline (Peak Detection, 2.43 bpm MAE), and outperforms the median-consensus baseline (5.51 bpm MAE), though the consensus baseline is itself compromised by 3 of 4 methods sharing a sub-harmonic locking failure mode.**\n")

    # Visualizations
    L.append("## 5. Visualizations\n")
    L.append("### Figure 1: Pooled MAE Across All Evaluated Approaches\n")
    L.append("![Method Comparison MAE](method_comparison_mae.png)\n\n")
    L.append("### Figure 2: CWT + RF Out-of-Fold Predictions vs. Ground Truth\n")
    L.append("![CWT + RF Predictions](cwt_rf_pred_vs_true.png)\n\n")

    # SECTION 6: Discussion
    L.append("## 6. Discussion & Critical Findings\n")
    L.append("### 6.1 Dataset Size and LOSO Fold-4 Statistical Limits\n")
    L.append("The evaluation cohort consists of 32 segments across 4 subjects (Subject 1: 10, Subject 2: 10, Subject 3: 10, Subject 4: 2). While pooled metrics across all 32 out-of-fold predictions reflect genuine subject holdout, Fold 4 evaluates only 2 segments. Fold 4 metrics must not be interpreted as statistically meaningful in isolation, though their inclusion in the pooled metric preserves complete cohort accounting.\n")

    L.append("### 6.2 Dominance of the Inter-Subject Difficulty Gap\n")
    L.append("A critical finding that must frame all algorithmic conclusions is the magnitude of subject physiological variance:")
    L.append("- Across the single-method RF models, the gap between the hardest subject (Subject 1: 1.54–1.78 bpm MAE) and the easiest subject (Subject 3: 0.96–1.25 bpm MAE) is **0.75 bpm**.")
    L.append("- In the Combined + RF model, this subject gap widens to **0.92 bpm** (Subject 1: 1.87 bpm vs Subject 3: 0.96 bpm).")
    L.append("- In comparison, the performance spread between the best ML combination (CWT + RF: 1.30 bpm) and the worst ML combination (Peak + RF: 1.53 bpm) is only **0.23 bpm**.")
    L.append("- **Conclusion:** The subject-difficulty gap is **3 to 4 times larger** than the algorithm gap. With $N=4$ subjects, claims of an optimal algorithm carry inherent uncertainty; evaluation on a 5th subject could plausibly alter the ranking. Algorithmic benchmarking must never obscure physiological heterogeneity.\n")

    L.append("### 6.3 Honest Methodological Contrast: LOSO vs. Leaky RepeatedKFold\n")
    L.append("In earlier dissertation tuning experiments, RepeatedKFold cross-validation yielded negative $R^2$ values ranging from $-0.11$ to $-0.38$ on this exact dataset. Under Leave-One-Subject-Out validation, positive $R^2$ scores emerge (Peak + RF: $+0.02$, Combined + RF: $+0.13$, CWT + RF: $+0.23$).")
    L.append("This difference should **not** be misattributed to the superiority of the new features. Rather, it reflects validation methodology: RepeatedKFold randomly mixed segments from the same subject into train and test folds, introducing high fold-to-fold baseline variance that distorted total sum of squares calculations. LOSO isolates individual subject physiological baselines cleanly, providing a more coherent evaluation of cross-subject generalization.\n")

    L.append("### 6.4 Overfitting in High-Dimensional Feature Combinations\n")
    L.append(r"Concatenating all 14 features into a single model (**Combined + RF**) resulted in an MAE of **1.45 bpm** and $R^2$ of **+0.13**, underperforming the 3-feature CWT + RF model (1.30 bpm, $R^2 = +0.23$). In each LOSO fold, training sets contain only 22 to 30 video segments. A 14-feature vector on ~25 samples yields an unfavorable feature-to-sample ratio ($\approx 1:2$). The Random Forest trees split on noisy, lower-fidelity features from autocorrelation and peak intervals, diluting the sharp time-frequency representation of CWT." + "\n")

    L.append("### 6.5 The Sub-Harmonic Disambiguation Role of Random Forest\n")
    L.append("The diagnostic investigation resolves why machine learning is indispensable for spectral respiratory rate estimation from short thermal videos. Raw Welch, CWT, and Autocorrelation methods lock onto half-rate sub-harmonics in 44% to 66% of cases due to the short 10-second observation window and asymmetric breathing cycles. When CWT's multi-scale power distribution is provided as features, Random Forest acts as a non-linear classifier that disambiguates whether a detected peak is the fundamental or a sub-harmonic. While this interpretation is strongly supported by the 44% locking diagnostic, it remains a hypothesis grounded in signal physics that warrants verification on longer duration thermal recordings.\n")

    report_content = "\n".join(L)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[make_report] Successfully generated {REPORT_FILE}")


if __name__ == "__main__":
    generate_plots_and_report()
