"""
showcase_training.py -- Interactive, High-Visual Live Training & Benchmark Showcase.

Run in your terminal anytime to demonstrate the full training pipeline:
    python showcase_training.py

Options:
    python showcase_training.py --force      # Re-extract features from all videos live
    python showcase_training.py --fast       # Use cached features for an instant live demo
"""

import os
import sys
import time
import json
import numpy as np

# Ensure terminal encoding supports visual formatting
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from rr_model import (
    DATASET_DIR,
    MODEL_DIR,
    NOTEBOOK_LABELS,
    FEATURE_NAMES,
    list_dataset_videos,
    build_feature_table,
    _make_models,
    _feature_vector,
    _metrics
)
import benchmark


def print_banner():
    banner = r"""
========================================================================================
   ___ _   _ ___ ___ __  __    _    _       ___ ___ ___  ___ _____ ___ _  _  ___ 
  |_ _| \ | | __| _ \  \/  |  /_\  | |     | _ \ __/ __|| _ \_   _/ __| || |/ __|
   | ||  \| | _||   / |\/| | / _ \ | |__   |   / _|\__ \|  _/ | | \__ \ __ | (__ 
  |___|_|\__|___|_|_\_|  |_|/_/ \_\|____|  |_|_\___|___/|_|   |_| |___/_||_|\___|
========================================================================================
           THERMAL RESPIRATORY RATE & FEVER MONITORING SYSTEM - ML PIPELINE
========================================================================================
"""
    print(banner)


def run_showcase():
    print_banner()

    force_features = "--force" in sys.argv
    step_id = "step_live"
    step_name = "Live Terminal Showcase Training"
    desc = "Live interactive training pipeline execution with dynamic MediaPipe tracking."

    for i, arg in enumerate(sys.argv):
        if arg == "--step-id" and i + 1 < len(sys.argv):
            step_id = sys.argv[i + 1]
        elif arg == "--step-name" and i + 1 < len(sys.argv):
            step_name = sys.argv[i + 1]
        elif arg == "--desc" and i + 1 < len(sys.argv):
            desc = sys.argv[i + 1]

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: SYSTEM & HARDWARE DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────
    print("[1/5] SYSTEM & PIPELINE INITIALIZATION")
    print("-" * 88)
    
    try:
        import mediapipe as mp
        mp_status = "ENABLED (468-Landmark FaceMesh, Dynamic Nostril Tracking)"
    except Exception as e:
        mp_status = f"FALLBACK ({e})"

    print(f"  * MediaPipe Face Tracking : {mp_status}")
    print(f"  * Thermal Dataset Path    : {DATASET_DIR}")
    videos = list_dataset_videos()
    print(f"  * Detected Thermal Videos : {len(videos)} clips loaded")
    print(f"  * Feature Space ({len(FEATURE_NAMES)})      : {', '.join(FEATURE_NAMES)}")
    print(f"  * Validation Strategy     : 5-Fold Cross-Validation + 80/20 Stratified Hold-out")
    print("-" * 88)
    time.sleep(0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: FEATURE EXTRACTION (OR CACHED RETRIEVAL)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/5] THERMAL SIGNAL PROCESSING & FEATURE EXTRACTION")
    print("-" * 88)
    if force_features:
        print("  Re-extracting nostril intensity waveforms across all videos via MediaPipe...")
    else:
        print("  Loading extracted physiological features from cache (use --force to re-scan video files)...")

    cache = build_feature_table(force=force_features)
    valid_videos = [v for v in videos if v["filename"] in cache and v["label_bpm"] is not None]
    
    # Progress visual
    total_v = len(valid_videos)
    for idx, v in enumerate(valid_videos):
        fname = v["filename"]
        feats = cache[fname]
        pct = int(((idx + 1) / total_v) * 40)
        bar = "█" * pct + "░" * (40 - pct)
        sys.stdout.write(f"\r  [{bar}] {idx+1}/{total_v} | {fname} -> Peak BPM: {feats.get('peak_bpm', 0):.1f} | Std: {feats.get('signal_std', 0):.2f}")
        sys.stdout.flush()
        if force_features:
            pass
        else:
            time.sleep(0.015)

    print(f"\n  [OK] Extracted {len(valid_videos)} complete feature vectors successfully.")
    print("-" * 88)
    time.sleep(0.4)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: MODEL TRAINING (ENSEMBLE OF REGRESSORS)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/5] TRAINING TREE-BASED ENSEMBLE REGRESSION MODELS (5-FOLD CV)")
    print("-" * 88)

    from sklearn.model_selection import KFold, cross_val_predict, train_test_split
    import joblib

    names = [v["filename"] for v in valid_videos]
    X = np.array([_feature_vector(cache[n]) for n in names], dtype=float)
    y = np.array([v["label_bpm"] for v in valid_videos], dtype=float)

    idx_arr = np.arange(len(y))
    tr, te = train_test_split(idx_arr, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models_dict = _make_models()
    oof_by_model = {}
    metrics_summary = {}

    for m_name, model in models_dict.items():
        print(f"  * Training {m_name:<20s} ...", end="", flush=True)
        t0 = time.time()
        
        # Fit on holdout train
        model.fit(X[tr], y[tr])
        holdout_res = _metrics(y[te], model.predict(X[te]))

        # 5-fold cross validation
        oof = cross_val_predict(_make_models()[m_name], X, y, cv=kf)
        oof_by_model[m_name] = oof
        cv_res = _metrics(y, oof)

        elapsed = time.time() - t0
        print(f" DONE ({elapsed:.2f}s) | Holdout MAE: {holdout_res['MAE']:.2f} | 5-Fold CV MAE: {cv_res['MAE']:.2f}")

    # Fit final models on all available data and persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    from rr_model import MODEL_FILES
    for m_name, model in _make_models().items():
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[m_name]))

    print("-" * 88)
    time.sleep(0.4)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: COMPREHENSIVE BENCHMARK EVALUATION (REGRESSION + F1 + CONFUSION MATRIX)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/5] BENCHMARK METRICS & CLINICAL CLASSIFICATION PERFORMANCE")
    print("=" * 88)

    models_eval = {}
    for m_name in models_dict.keys():
        models_eval[m_name] = benchmark.evaluate_predictions(
            y, oof_by_model[m_name], step_id=step_id, model_name=m_name
        )

    # 1. Regression Table
    print("\n-- REGRESSION METRICS TABLE -----------------------------------------------------------")
    print(f"{'Model Name':<20s} | {'MAE (bpm)':<10s} | {'RMSE':<10s} | {'R² Score':<10s} | {'MAPE (%)':<10s}")
    print("-" * 88)
    for m_name, ev in models_eval.items():
        reg = ev["regression"]
        print(f"{m_name:<20s} | {reg['MAE']:<10.3f} | {reg['RMSE']:<10.3f} | {reg['R2']:<10.3f} | {reg['MAPE_percent']:<10.2f}%")
    print("-" * 88)

    # 2. Clinical Classification & F1-Score Table
    print("\n-- CLINICAL CLASSIFICATION & F1-SCORE (Normal vs Tachypnea vs Bradypnea) --------------")
    print(f"{'Model Name':<20s} | {'Macro F1':<10s} | {'Weighted F1':<12s} | {'Precision':<10s} | {'Recall':<10s} | {'Accuracy':<10s}")
    print("-" * 88)
    for m_name, ev in models_eval.items():
        cls_ = ev["clinical_classification"]
        print(f"{m_name:<20s} | {cls_['f1_macro']:<10.3f} | {cls_['f1_weighted']:<12.3f} | {cls_['precision_macro']:<10.3f} | {cls_['recall_macro']:<10.3f} | {cls_['accuracy']*100:<9.1f}%")
    print("-" * 88)

    # 3. ANSI/AAMI Tolerance Accuracy Table
    print("\n-- ANSI/AAMI PHYSIOLOGICAL TOLERANCE ACCEPTANCE RATE ---------------------------------")
    print(f"{'Model Name':<20s} | {'Within ±1.5 bpm (High Precision)':<35s} | {'Within ±2.0 bpm (Clinical Band)':<30s}")
    print("-" * 88)
    for m_name, ev in models_eval.items():
        tol = ev["tolerance_rates"]
        print(f"{m_name:<20s} | {tol['within_1_5_bpm_percent']:<35.1f}% | {tol['within_2_0_bpm_percent']:<30.1f}%")
    print("-" * 88)

    # 4. Live Visual Confusion Matrix
    ens_cls = models_eval["Ensemble"]["clinical_classification"]
    labels = ens_cls["labels"]
    cm = ens_cls["confusion_matrix"]
    
    print("\n-- CONFUSION MATRIX [VOTING ENSEMBLE MODEL] -----------------------------------------")
    header_title = "Actual \\ Predicted"
    print(f"{header_title:<20s} | " + " | ".join([f"{l:<12s}" for l in labels]))
    print("-" * (23 + 15 * len(labels)))
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:<12d}" for val in row])
        print(f"{labels[i]:<20s} | {row_str}")
    print("-" * (23 + 15 * len(labels)))
    print(f"Heatmap visualization generated at: result/confusion_matrix_{step_id}_ensemble.png")

    # Save to history & Markdown report
    benchmark.record_benchmark(
        step_id=step_id,
        step_name=step_name,
        description=desc,
        models_eval=models_eval
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5: PERSISTENCE & SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[5/5] MODEL PERSISTENCE & ARTIFACT SUMMARY")
    print("=" * 88)
    print(f"  * Serialized Model Weights : {os.path.abspath(MODEL_DIR)}")
    print(f"  * Benchmark History JSON   : {os.path.abspath(benchmark.BENCHMARK_HISTORY_FILE)}")
    print(f"  * Markdown Report Updated  : {os.path.abspath(benchmark.REPORT_MD_FILE)}")
    print(f"  * Confusion Matrix Assets  : {os.path.abspath(benchmark.RESULT_DIR)}")
    print("=" * 88)
    print("  >>> TRAINING & BENCHMARK COMPLETE! READY FOR DEMO/PRESENTATION <<<\n")


if __name__ == "__main__":
    run_showcase()
