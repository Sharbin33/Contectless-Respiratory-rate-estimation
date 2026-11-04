"""make_tuning_report.py - builds models/tuning_report.md and 2 plots from tuning_results.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = "models"
NAMES = ["random_forest", "gradient_boosting", "xgboost", "ensemble"]
LABEL = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
         "xgboost": "XGBoost", "ensemble": "Ensemble"}

res = json.load(open(os.path.join(M, "tuning_results.json")))
oof = json.load(open(os.path.join(M, "tuning_oof.json")))
n = res["n_samples"]

# ---- plot 1: MAE before/after (A, B, C) ----
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(NAMES))
w = 0.26
for i, (key, lab) in enumerate([("A_old_default", "A: old features, default"),
                                ("B_new_default", "B: new features, default"),
                                ("C_new_tuned", "C: new features, tuned")]):
    vals = [res[nm][key]["mae"] for nm in NAMES]
    ax.bar(x + (i - 1) * w, vals, w, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[nm] for nm in NAMES])
ax.set_ylabel("MAE (bpm, lower is better)")
ax.set_title("Nested-CV MAE: before vs after")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(M, "tuning_mae_bar.png"), dpi=150)
plt.close(fig)

# ---- plot 2: predicted vs reference (Ensemble, B vs C) ----
y = np.array(oof["y"])
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(y, oof["B_ensemble"], label="B: default", alpha=0.7)
ax.scatter(y, oof["C_ensemble"], label="C: tuned", alpha=0.7)
lo, hi = float(y.min()) - 1, float(y.max()) + 1
ax.plot([lo, hi], [lo, hi], "k--", label="perfect")
ax.set_xlabel("Reference RR (bpm)")
ax.set_ylabel("Predicted RR (bpm)")
ax.set_title("Ensemble: predicted vs reference")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(M, "tuning_pred_vs_ref.png"), dpi=150)
plt.close(fig)

# ---- markdown report ----
L = []
L.append("# Hyperparameter Tuning Report\n")
L.append(f"Samples: {n}. All numbers are nested cross-validation (outer folds only).\n")
L.append("## Before vs after (B = new features default, C = new features tuned)\n")
L.append("| Model | MAE B | MAE C | MAE improv % | RMSE B | RMSE C | RMSE improv % | R2 B | R2 C | R2 change |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for nm in NAMES:
    b, c, g = res[nm]["B_new_default"], res[nm]["C_new_tuned"], res[nm]["tuning_gain_B_to_C"]
    L.append(f"| {LABEL[nm]} | {b['mae']:.2f} | {c['mae']:.2f} | {g['mae_improvement_pct']:.1f} | "
             f"{b['rmse']:.2f} | {c['rmse']:.2f} | {g['rmse_improvement_pct']:.1f} | "
             f"{b['r2']:.2f} | {c['r2']:.2f} | {g['r2_change']:+.2f} |")
L.append("\n## Where the gain came from\n")
L.append("| Model | A to B (new features) MAE % | B to C (tuning) MAE % | Wilcoxon p (B vs C) |")
L.append("|---|---|---|---|")
for nm in NAMES:
    p = res[nm]["wilcoxon_p_B_vs_C"]
    ptxt = "n/a" if p is None else f"{p:.3f}"
    L.append(f"| {LABEL[nm]} | {res[nm]['feature_gain_A_to_B']['mae_improvement_pct']:.1f} | "
             f"{res[nm]['tuning_gain_B_to_C']['mae_improvement_pct']:.1f} | {ptxt} |")
L.append("\n## Videos within tolerance of the reference RR\n")
L.append("| Model | within 2 bpm (B) | within 2 bpm (C) | within 3 bpm (B) | within 3 bpm (C) |")
L.append("|---|---|---|---|---|")
for nm in NAMES:
    b, c = res[nm]["B_new_default"], res[nm]["C_new_tuned"]
    L.append(f"| {LABEL[nm]} | {b['within_2_bpm_pct']:.0f}% | {c['within_2_bpm_pct']:.0f}% | "
             f"{b['within_3_bpm_pct']:.0f}% | {c['within_3_bpm_pct']:.0f}% |")
L.append("\n## Chosen hyperparameters\n")
for nm, params in res["best_params"].items():
    L.append(f"- **{LABEL[nm]}**: {params}")
L.append("\n## Top 3 features per tuned model\n")
new_feats = {"autocorr_bpm", "welch_bpm", "cwt_bpm"}
for nm, top in res["top3_feature_importances"].items():
    txt = ", ".join(f"{f} ({v:.2f})" for f, v in top)
    flag = " (includes a new signal-method feature)" if any(f in new_feats for f, _ in top) else ""
    L.append(f"- {LABEL[nm]}: {txt}{flag}")
L.append("\n## Honest conclusion\n")
ens = res["ensemble"]
p = ens["wilcoxon_p_B_vs_C"]
sig = (p is not None) and p < 0.05
L.append(f"Ensemble MAE changed by {ens['tuning_gain_B_to_C']['mae_improvement_pct']:.1f}% from tuning "
         f"and {ens['feature_gain_A_to_B']['mae_improvement_pct']:.1f}% from the new features.")
if sig:
    L.append("The tuning improvement is statistically significant (Wilcoxon p < 0.05) on this dataset.")
else:
    L.append(f"The tuning change is NOT statistically significant on {n} videos "
             "(Wilcoxon p >= 0.05 or unavailable). Treat it as a small or uncertain change and "
             "collect more data before claiming improvement.")
L.append("\n![MAE](tuning_mae_bar.png)\n\n![Pred vs ref](tuning_pred_vs_ref.png)\n")
open(os.path.join(M, "tuning_report.md"), "w").write("\n".join(L))
print("wrote models/tuning_report.md, tuning_mae_bar.png, tuning_pred_vs_ref.png")
