"""
rr_model.py -- ML respiratory-rate models (ported from V5.ipynb) for the dashboard.

Pipeline (same idea as V5.ipynb):
  thermal video -> nasal ROI signal -> bandpass + peak detection -> features
  -> Random Forest / Gradient Boosting / XGBoost / Voting ensemble -> breaths per minute.

Differences from the notebook (deliberate, see README "Respiratory rate ML"):
  * The notebook z-scores the signal *before* computing mean/std features, so those two
    features are constants (0 and 1). It also overwrites its own `breathing_rate` feature
    with the ground-truth label. Here we use informative features instead:
    num_peaks, peak_bpm, interval_bpm, spectral_bpm, signal_std, roi_detection_rate.
  * Models are evaluated with 5-fold CV in addition to the notebook's 80/20 hold-out,
    because the dataset only has 32 labelled videos.

Usage:
  python rr_model.py train      # (re)build features cache, train, save models/ + metrics.json
"""

import glob
import json
import os
import sys

import numpy as np

from respiratory_rate import estimate_respiratory_rate
import benchmark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Thermal Videos")
if not os.path.exists(DATASET_DIR):
    alt_dir = os.path.join(os.path.dirname(BASE_DIR), "final year proj", "Thermal Videos -20260919T194040Z-1-001", "Thermal Videos")
    if os.path.exists(alt_dir):
        DATASET_DIR = alt_dir
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_CACHE = os.path.join(MODEL_DIR, "features_cache.json")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

# Hand-counted breathing rates (BPM) from V5.ipynb, in sorted-filename order.
# The notebook lists 49 labels but only 32 videos (1.0 ... 4.1) are in "Thermal Videos";
# the first 32 labels are assumed to belong to those 32 videos (same order the notebook uses).
NOTEBOOK_LABELS = [
    14.5, 17, 18, 20, 21, 16, 18, 15, 19.5, 19, 19, 20, 17, 15, 17.5, 18.5, 16, 20, 23, 20,
    18.5, 19, 18, 21, 20, 19, 17, 19, 15, 18,
    19, 19, 19, 15, 14, 19, 19, 17, 15, 20,
    20, 21, 17, 15, 16, 17, 21, 16, 20,
]

FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.json")
if os.path.exists(FEATURE_NAMES_FILE):
    with open(FEATURE_NAMES_FILE) as f:
        FEATURE_NAMES = json.load(f)
else:
    FEATURE_NAMES = [
        "num_peaks", "peak_bpm", "interval_bpm", "spectral_bpm",
        "signal_std", "roi_detection_rate", "autocorr_bpm", "welch_bpm", "cwt_bpm"
    ]
MODEL_FILES = {
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Ensemble": "ensemble_model.pkl",
}


def _natural_key(name):
    return [int(p) if p.isdigit() else p for p in name.replace(".mp4", "").split(".")]


def list_dataset_videos():
    """Sorted list of dataset video file names, with the notebook label where known."""
    files = sorted(
        (os.path.basename(p) for p in glob.glob(os.path.join(DATASET_DIR, "*.mp4"))),
        key=_natural_key,
    )
    return [
        {"filename": f, "label_bpm": NOTEBOOK_LABELS[i] if i < len(NOTEBOOK_LABELS) else None}
        for i, f in enumerate(files)
    ]


def _spectral_bpm(signal, fps, low_hz=0.1, high_hz=0.5):
    """Dominant frequency (in breaths/min) of the filtered signal inside the breathing band."""
    n = len(signal)
    if n < 8 or fps <= 0:
        return None
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(signal - np.mean(signal), nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    band = (freqs >= low_hz) & (freqs <= min(high_hz, fps / 2))
    if not band.any():
        return None
    return float(freqs[band][np.argmax(spec[band])] * 60.0)


def features_from_rr_result(rr):
    """Build the ML feature vector from an estimate_respiratory_rate(debug=True) result.

    Returns (features_dict, signal_dict) or (None, None) if no usable signal.
    """
    dbg = rr.get("debug")
    fps = rr.get("fps")
    duration = rr.get("duration_seconds")
    if not dbg or not fps or not duration:
        return None, None

    filtered = np.asarray(dbg["filtered_signal"], dtype=float)
    peaks = np.asarray(dbg["peaks_indices"], dtype=int)
    num_peaks = int(len(peaks))
    peak_bpm = num_peaks / duration * 60.0  # notebook formula
    interval_bpm = rr.get("respiratory_rate_bpm")
    if interval_bpm is None:
        interval_bpm = peak_bpm
    spectral = _spectral_bpm(filtered, fps)
    if spectral is None:
        spectral = peak_bpm

    from rr_signal_methods import compare_rr_methods
    res = compare_rr_methods(filtered, fps)

    def _clean_bpm(method_key):
        val = res.get(method_key, {}).get("bpm") if res else None
        if val is None:
            return float(spectral)
        try:
            f = float(val)
            return float(spectral) if np.isnan(f) else f
        except (ValueError, TypeError):
            return float(spectral)

    autocorr_bpm = _clean_bpm("autocorrelation")
    welch_bpm = _clean_bpm("welch_psd")
    cwt_bpm = _clean_bpm("cwt")

    feats = {
        "num_peaks": num_peaks,
        "peak_bpm": round(peak_bpm, 3),
        "interval_bpm": round(float(interval_bpm), 3),
        "spectral_bpm": round(spectral, 3),
        "signal_std": round(float(dbg.get("signal_std_before_filter", 0.0)), 4),
        "roi_detection_rate": round(float(dbg.get("roi_detection_rate", 1.0)), 4),
        "autocorr_bpm": round(float(autocorr_bpm), 3),
        "welch_bpm": round(float(welch_bpm), 3),
        "cwt_bpm": round(float(cwt_bpm), 3),
    }
    sig = {
        "fps": fps,
        "time_s": (np.arange(len(filtered)) / fps).round(3).tolist(),
        "filtered": np.round(filtered, 4).tolist(),
        "peak_times_s": (peaks / fps).round(3).tolist(),
        "peak_values": np.round(filtered[peaks], 4).tolist() if num_peaks else [],
    }
    return feats, sig


def extract_features(video_path):
    """Run the RR signal pipeline on a video and return (features, signal, rr_result)."""
    rr = estimate_respiratory_rate(video_path, debug=True)
    feats, sig = features_from_rr_result(rr)
    return feats, sig, rr


def _feature_vector(feats):
    names = FEATURE_NAMES
    if os.path.exists(FEATURE_NAMES_FILE):
        try:
            with open(FEATURE_NAMES_FILE) as f:
                names = json.load(f)
        except Exception:
            names = FEATURE_NAMES
    return [feats[k] for k in names]


def build_feature_table(force=False):
    """Extract (or load cached) features for every dataset video."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    cache = {}
    if os.path.exists(FEATURE_CACHE) and not force:
        with open(FEATURE_CACHE) as f:
            cache = json.load(f)
    changed = False
    for item in list_dataset_videos():
        name = item["filename"]
        if name in cache:
            continue
        print(f"[features] {name}", flush=True)
        feats, _, rr = extract_features(os.path.join(DATASET_DIR, name))
        if feats is None:
            print(f"  skipped: {rr.get('error')}")
            continue
        cache[name] = feats
        changed = True
    if changed or not os.path.exists(FEATURE_CACHE):
        with open(FEATURE_CACHE, "w") as f:
            json.dump(cache, f, indent=1)
    return cache


def _make_models():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
    from xgboost import XGBRegressor

    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Ensemble": VotingRegressor([
            ("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)),
        ]),
    }


def _metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mse)), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train(force_features=False, step_id="step_1", step_name="MediaPipe Dynamic Tracking", description="Feature extraction with dynamic MediaPipe Face Mesh nostril tracking."):
    import joblib
    from sklearn.model_selection import KFold, cross_val_predict, train_test_split

    cache = build_feature_table(force=force_features)
    videos = [v for v in list_dataset_videos() if v["filename"] in cache and v["label_bpm"] is not None]
    names = [v["filename"] for v in videos]
    X = np.array([_feature_vector(cache[n]) for n in names], dtype=float)
    y = np.array([v["label_bpm"] for v in videos], dtype=float)
    print(f"[train] {len(y)} labelled videos, features={FEATURE_NAMES}")

    # Notebook-style hold-out (test_size=0.2, random_state=42)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {"n_videos": int(len(y)), "features": FEATURE_NAMES, "models": {}, "oof_predictions": {}}
    metrics["baselines"] = {
        "Signal peaks (notebook formula)": _metrics(y, X[:, FEATURE_NAMES.index("peak_bpm")]),
        "Signal peak-interval": _metrics(y, X[:, FEATURE_NAMES.index("interval_bpm")]),
        "Signal FFT": _metrics(y, X[:, FEATURE_NAMES.index("spectral_bpm")]),
        "Predict mean label": _metrics(y, np.full_like(y, y.mean())),
    }

    oof_by_model = {}
    for name, model in _make_models().items():
        model.fit(X[tr], y[tr])
        holdout = _metrics(y[te], model.predict(X[te]))
        oof = cross_val_predict(_make_models()[name], X, y, cv=kf)
        oof_by_model[name] = oof
        metrics["models"][name] = {"holdout": holdout, "cv5": _metrics(y, oof)}
        print(f"[train] {name:18s} holdout={holdout}  cv5={metrics['models'][name]['cv5']}")

    metrics["holdout_videos"] = [names[i] for i in te]
    for i, n in enumerate(names):
        metrics["oof_predictions"][n] = {m: round(float(p[i]), 2) for m, p in oof_by_model.items()}

    # Deployed models are fit on ALL labelled videos.
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in _make_models().items():
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[name]))
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"[train] saved models + metrics to {MODEL_DIR}")

    # Record benchmark for this improvement phase
    models_eval = {}
    for name in _make_models().keys():
        models_eval[name] = benchmark.evaluate_predictions(
            y, oof_by_model[name], step_id=step_id, model_name=name
        )

    benchmark.record_benchmark(
        step_id=step_id,
        step_name=step_name,
        description=description,
        models_eval=models_eval,
        baselines_eval=metrics.get("baselines")
    )
    return metrics


_loaded = {}


def load_models():
    if _loaded:
        return _loaded
    import joblib

    for name, fn in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fn)
        if os.path.exists(path):
            _loaded[name] = joblib.load(path)
    return _loaded


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None


def predict_ml(feats):
    """Predict BPM with every trained model. Returns {model_name: bpm} or {} if untrained."""
    models = load_models()
    if not models or feats is None:
        return {}
    x = np.array([_feature_vector(feats)], dtype=float)
    return {name: round(float(m.predict(x)[0]), 1) for name, m in models.items()}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        import tune_models
        import subprocess
        tune_models.main()
        subprocess.run([sys.executable, "make_tuning_report.py"], check=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        force_flag = "--force" in sys.argv
        
        step_id = "step_1"
        step_name = "MediaPipe Dynamic Tracking"
        desc = "Extracted signal features using dynamic MediaPipe Face Mesh nostril tracking."
        
        for i, arg in enumerate(sys.argv):
            if arg == "--step-id" and i + 1 < len(sys.argv):
                step_id = sys.argv[i + 1]
            elif arg == "--step-name" and i + 1 < len(sys.argv):
                step_name = sys.argv[i + 1]
            elif arg == "--desc" and i + 1 < len(sys.argv):
                desc = sys.argv[i + 1]
                
        train(force_features=force_flag, step_id=step_id, step_name=step_name, description=desc)
    else:
        print(__doc__)
