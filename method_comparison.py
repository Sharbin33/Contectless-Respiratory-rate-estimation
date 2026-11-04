"""
method_comparison.py - Conference paper experiment comparing single-method vs combined
feature sets with Random Forest and signal-processing baselines under Leave-One-Subject-Out CV.
"""
import json
import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import rr_model
import rr_signal_methods

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXP_DIR = os.path.join(MODEL_DIR, "method_comparison")
CACHE_V2_FILE = os.path.join(EXP_DIR, "features_cache_v2.json")

# 1. Feature Set Definitions (from Phase 1)
FEATURE_SETS = {
    "peak_detection": [
        "peak_bpm",
        "peak_num_peaks",
        "peak_mean_interval_sec",
        "peak_interval_std_sec",
    ],
    "autocorrelation": [
        "autocorr_bpm",
        "autocorr_lag_samples",
        "autocorr_period_sec",
        "autocorr_correlation_strength",
    ],
    "welch_psd": [
        "welch_bpm",
        "welch_dominant_freq_hz",
        "welch_band_power_ratio",
    ],
    "cwt": [
        "cwt_bpm",
        "cwt_dominant_freq_hz",
        "cwt_scale_power_ratio",
    ],
}

FEATURE_SETS["combined"] = (
    FEATURE_SETS["peak_detection"]
    + FEATURE_SETS["autocorrelation"]
    + FEATURE_SETS["welch_psd"]
    + FEATURE_SETS["cwt"]
)


def extract_v2_features_for_video(video_path):
    """Run extraction and return flat dict of all 14 features."""
    rr = rr_model.estimate_respiratory_rate(video_path, debug=True)
    dbg = rr.get("debug") or {}
    filtered = dbg.get("filtered_signal")
    fps = rr.get("fps") or 25.0
    duration = rr.get("duration_seconds") or 30.0

    if filtered is None or len(filtered) == 0:
        raise ValueError(f"No filtered signal for video: {video_path}")

    filtered = np.asarray(filtered, dtype=float)
    res = rr_signal_methods.compare_rr_methods(filtered, fps)

    # 1. Peak Detection
    pd_res = res.get("peak_detection") or {}
    peak_bpm = pd_res.get("bpm")
    if peak_bpm is None:
        peak_bpm = (len(pd_res.get("peak_indices", [])) / duration) * 60.0
    mean_int = pd_res.get("mean_interval_sec")
    if mean_int is None:
        mean_int = duration / max(1, len(pd_res.get("peak_indices", [])))
    std_int = pd_res.get("interval_std_sec")
    if std_int is None:
        std_int = 0.0

    # 2. Autocorrelation
    ac_res = res.get("autocorrelation") or {}
    ac_bpm = ac_res.get("bpm") if ac_res.get("bpm") is not None else peak_bpm
    ac_lag = ac_res.get("lag_samples") if ac_res.get("lag_samples") is not None else int(round(fps * 60.0 / max(1.0, ac_bpm)))
    ac_period = ac_res.get("period_sec") if ac_res.get("period_sec") is not None else float(ac_lag / fps)
    ac_strength = ac_res.get("correlation_strength") if ac_res.get("correlation_strength") is not None else 0.0

    # 3. Welch PSD
    w_res = res.get("welch_psd") or {}
    w_bpm = w_res.get("bpm") if w_res.get("bpm") is not None else peak_bpm
    w_freq = w_res.get("dominant_freq_hz") if w_res.get("dominant_freq_hz") is not None else float(w_bpm / 60.0)
    w_ratio = w_res.get("band_power_ratio") if w_res.get("band_power_ratio") is not None else 0.0

    # 4. CWT
    cwt_res = res.get("cwt") or {}
    cwt_bpm = cwt_res.get("bpm") if cwt_res.get("bpm") is not None else peak_bpm
    cwt_freq = cwt_res.get("dominant_freq_hz") if cwt_res.get("dominant_freq_hz") is not None else float(cwt_bpm / 60.0)
    cwt_ratio = cwt_res.get("scale_power_ratio") if cwt_res.get("scale_power_ratio") is not None else 0.0

    entry = {
        # Peak Detection
        "peak_bpm": round(float(peak_bpm), 3),
        "peak_num_peaks": int(pd_res.get("num_peaks", 0)),
        "peak_mean_interval_sec": round(float(mean_int), 3),
        "peak_interval_std_sec": round(float(std_int), 3),
        # Autocorrelation
        "autocorr_bpm": round(float(ac_bpm), 3),
        "autocorr_lag_samples": int(ac_lag),
        "autocorr_period_sec": round(float(ac_period), 3),
        "autocorr_correlation_strength": round(float(ac_strength), 3),
        # Welch PSD
        "welch_bpm": round(float(w_bpm), 3),
        "welch_dominant_freq_hz": round(float(w_freq), 4),
        "welch_band_power_ratio": round(float(w_ratio), 4),
        # CWT
        "cwt_bpm": round(float(cwt_bpm), 3),
        "cwt_dominant_freq_hz": round(float(cwt_freq), 4),
        "cwt_scale_power_ratio": round(float(cwt_ratio), 4),
    }
    return entry


def build_v2_cache(force=False):
    """Build or load features_cache_v2.json across all 32 videos."""
    os.makedirs(EXP_DIR, exist_ok=True)
    if os.path.exists(CACHE_V2_FILE) and not force:
        with open(CACHE_V2_FILE) as f:
            return json.load(f), 0.0

    cache = {}
    vids = rr_model.list_dataset_videos()
    t0 = time.time()
    for item in vids:
        fn = item["filename"]
        video_path = os.path.join(rr_model.DATASET_DIR, fn)
        print(f"[v2_cache] Extracting 14 features for {fn}...", flush=True)
        feats = extract_v2_features_for_video(video_path)
        cache[fn] = feats
    elapsed = time.time() - t0

    with open(CACHE_V2_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    return cache, elapsed


def load_dataset(feature_set_name="combined", force_cache=False):
    """Return (rows, y, subjects, filenames) for the specified feature set."""
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_name}. Available: {list(FEATURE_SETS.keys())}")

    cache, _ = build_v2_cache(force=force_cache)
    vids = rr_model.list_dataset_videos()
    keys = FEATURE_SETS[feature_set_name]

    rows = []
    y = []
    subjects = []
    filenames = []

    for v in vids:
        fn = v["filename"]
        feats = cache[fn]
        rows.append({k: feats[k] for k in keys})
        y.append(v["label_bpm"])
        subjects.append(fn.split(".")[0])
        filenames.append(fn)

    return rows, y, subjects, filenames


ALL_FILENAMES = [v["filename"] for v in rr_model.list_dataset_videos()]


def get_loso_folds():
    """
    Generate Leave-One-Subject-Out (LOSO) cross-validation folds across the 4 subjects.

    Dataset Structure:
      - Subject 1: 10 segments (1.0.mp4 - 1.9.mp4)
      - Subject 2: 10 segments (2.0.mp4 - 2.9.mp4)
      - Subject 3: 10 segments (3.0.mp4 - 3.9.mp4)
      - Subject 4:  2 segments (4.0.mp4 - 4.1.mp4)

    CAVEAT / STATISTICAL NOTE:
      Fold 4 evaluates on Subject 4, which contains only 2 segments. While pooled
      out-of-fold metrics across all 32 samples are well-defined, individual metrics
      for Fold 4 alone have high variance and are not statistically reliable in isolation.
    """
    vids = rr_model.list_dataset_videos()
    filenames = [v["filename"] for v in vids]
    subjects = [f.split(".")[0] for f in filenames]
    unique_subjects = sorted(list(set(subjects)), key=int)

    folds = []
    for i, subj in enumerate(unique_subjects):
        test_idx = [idx for idx, s in enumerate(subjects) if s == subj]
        train_idx = [idx for idx, s in enumerate(subjects) if s != subj]

        test_files = [filenames[idx] for idx in test_idx]
        train_files = [filenames[idx] for idx in train_idx]

        folds.append({
            "fold": i + 1,
            "subject": subj,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_files": train_files,
            "test_files": test_files,
        })
    return folds


LOSO_RESULTS_FILE = os.path.join(EXP_DIR, "loso_results.json")


def evaluate_loso_rf(feature_set_name):
    """
    Train and evaluate a RandomForestRegressor(n_estimators=100, random_state=42)
    under Leave-One-Subject-Out (LOSO) cross-validation across the 4 subjects.
    No hyperparameter tuning is applied (baseline RF configuration).
    """
    rows, y, subjects, filenames = load_dataset(feature_set_name)
    keys = FEATURE_SETS[feature_set_name]
    X = np.array([[float(r[k]) for k in keys] for r in rows], dtype=float)
    y = np.array(y, dtype=float)

    folds = get_loso_folds()
    oof_predictions = np.zeros(len(y), dtype=float)
    fold_summaries = {}
    per_subject_mae = {}

    for fold in folds:
        s = fold["subject"]
        tr_idx = fold["train_idx"]
        te_idx = fold["test_idx"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[tr_idx], y[tr_idx])
        preds = rf.predict(X[te_idx])
        oof_predictions[te_idx] = preds

        y_te = y[te_idx]
        mae = float(mean_absolute_error(y_te, preds))
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        r2 = float(r2_score(y_te, preds)) if len(te_idx) > 2 else None

        fold_summaries[f"fold_{fold['fold']}_subject_{s}"] = {
            "subject": s,
            "n_test": len(te_idx),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3) if r2 is not None else None,
            "predictions": {filenames[idx]: round(float(preds[i]), 2) for i, idx in enumerate(te_idx)},
        }
        per_subject_mae[s] = round(mae, 3)

    pooled_mae = float(mean_absolute_error(y, oof_predictions))
    pooled_rmse = float(np.sqrt(mean_squared_error(y, oof_predictions)))
    pooled_r2 = float(r2_score(y, oof_predictions))

    return {
        "feature_set": feature_set_name,
        "n_features": len(keys),
        "features": keys,
        "model": "RandomForestRegressor(n_estimators=100, random_state=42)",
        "tuning": "None (default baseline parameters)",
        "pooled_metrics": {
            "mae": round(pooled_mae, 3),
            "rmse": round(pooled_rmse, 3),
            "r2": round(pooled_r2, 3),
        },
        "per_subject_mae": per_subject_mae,
        "folds": fold_summaries,
        "oof_predictions": {filenames[i]: round(float(oof_predictions[i]), 2) for i in range(len(filenames))},
    }


def run_phase4():
    """Evaluate the 4 single-method + RF combinations under LOSO CV."""
    single_methods = [
        ("peak_detection", "Peak Detection + RF"),
        ("autocorrelation", "Autocorrelation + RF"),
        ("welch_psd", "Welch PSD + RF"),
        ("cwt", "CWT + RF"),
    ]

    results = {}
    if os.path.exists(LOSO_RESULTS_FILE):
        try:
            with open(LOSO_RESULTS_FILE) as f:
                results = json.load(f)
        except Exception:
            results = {}

    for key, display_name in single_methods:
        print(f"[run_phase4] Evaluating {display_name} via LOSO CV...", flush=True)
        res = evaluate_loso_rf(key)
        res["display_name"] = display_name
        results[key] = res

    with open(LOSO_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_phase5():
    """Evaluate the 5th approach: Combined (all 14 methods) + RF under LOSO CV."""
    results = {}
    if os.path.exists(LOSO_RESULTS_FILE):
        try:
            with open(LOSO_RESULTS_FILE) as f:
                results = json.load(f)
        except Exception:
            results = {}

    print("[run_phase5] Evaluating Combined (All Methods) + RF via LOSO CV...", flush=True)
    res = evaluate_loso_rf("combined")
    res["display_name"] = "Combined (All Methods) + RF"
    results["combined"] = res

    with open(LOSO_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_phase6():
    """
    Evaluate the 5 baseline approaches (no RF, no training, fixed formulas):
      1. Raw Peak Detection (peak_bpm directly)
      2. Raw Autocorrelation (autocorr_bpm directly)
      3. Raw Welch PSD (welch_bpm directly)
      4. Raw CWT (cwt_bpm directly)
      5. Consensus Median (median of the 4 valid method bpms directly)

    NOTE ON EVALUATION METHODOLOGY:
      These baseline numbers are NOT computed under LOSO cross-validation.
      Because they are fixed deterministic formulas with zero learnable parameters,
      there is no training process and no subject holdout. They are computed
      across all 32 videos directly ('Formula (in-sample, no holdout)').
    """
    with open(CACHE_V2_FILE) as f:
        cache = json.load(f)

    vids = rr_model.list_dataset_videos()
    labels_dict = {v["filename"]: v["label_bpm"] for v in vids}
    filenames = sorted(list(labels_dict.keys()))

    def eval_baseline(name, preds_dict):
        y_true = np.array([labels_dict[f] for f in filenames], dtype=float)
        y_pred = np.array([preds_dict[f] for f in filenames], dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        subjects = [f.split(".")[0] for f in filenames]
        unique_subjects = sorted(list(set(subjects)), key=int)
        per_subj_mae = {}
        for s in unique_subjects:
            idx = [i for i, subj in enumerate(subjects) if subj == s]
            subj_mae = float(mean_absolute_error(y_true[idx], y_pred[idx]))
            per_subj_mae[s] = round(subj_mae, 3)

        return {
            "display_name": name,
            "type": "Formula (in-sample, no holdout)",
            "pooled_metrics": {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 3),
            },
            "per_subject_mae": per_subj_mae,
            "predictions": {f: round(float(preds_dict[f]), 2) for f in filenames},
        }

    baselines = {}

    # 1. Raw Peak Detection
    baselines["raw_peak_detection"] = eval_baseline(
        "Raw Peak Detection",
        {f: cache[f]["peak_bpm"] for f in filenames}
    )

    # 2. Raw Autocorrelation
    baselines["raw_autocorrelation"] = eval_baseline(
        "Raw Autocorrelation",
        {f: cache[f]["autocorr_bpm"] for f in filenames}
    )

    # 3. Raw Welch PSD
    baselines["raw_welch_psd"] = eval_baseline(
        "Raw Welch PSD",
        {f: cache[f]["welch_bpm"] for f in filenames}
    )

    # 4. Raw CWT
    baselines["raw_cwt"] = eval_baseline(
        "Raw CWT",
        {f: cache[f]["cwt_bpm"] for f in filenames}
    )

    # 5. Consensus Median
    consensus_preds = {}
    for f in filenames:
        c = cache[f]
        bpms = [c[k] for k in ["peak_bpm", "autocorr_bpm", "welch_bpm", "cwt_bpm"] if c.get(k) is not None]
        consensus_preds[f] = float(np.median(bpms))

    baselines["consensus_median"] = eval_baseline(
        "Consensus Median",
        consensus_preds
    )

    # Load existing loso_results.json and append under separate key
    results = {}
    if os.path.exists(LOSO_RESULTS_FILE):
        try:
            with open(LOSO_RESULTS_FILE) as f:
                results = json.load(f)
        except Exception:
            results = {}

    results["baselines_no_training"] = baselines

    with open(LOSO_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return baselines


if __name__ == "__main__":
    if "--phase4" in sys.argv:
        print("[method_comparison] Running Phase 4: evaluating 4 single-method + RF combinations...")
        results = run_phase4()
        print("[method_comparison] Done. Saved to", LOSO_RESULTS_FILE)
    elif "--phase5" in sys.argv:
        print("[method_comparison] Running Phase 5: evaluating Combined + RF...")
        results = run_phase5()
        print("[method_comparison] Done. Saved to", LOSO_RESULTS_FILE)
    elif "--phase6" in sys.argv:
        print("[method_comparison] Running Phase 6: evaluating baselines (standalone formulas + consensus)...")
        results = run_phase6()
        print("[method_comparison] Done. Appended baselines to", LOSO_RESULTS_FILE)
    else:
        force_build = "--force" in sys.argv
        print("[method_comparison] Initializing v2 cache build...")
        cache, elapsed = build_v2_cache(force=force_build)
        print(f"[method_comparison] Done. Extracted {len(cache)} videos in {elapsed:.1f}s.")
        print(f"[method_comparison] Verified {len(get_loso_folds())} LOSO folds.")
