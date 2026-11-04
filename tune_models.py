"""tune_models.py - nested-CV hyperparameter tuning for RR regressors."""
import json
import os
import shutil
import warnings

import rr_model
import joblib
import numpy as np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
QUICK = os.environ.get("QUICK", "0") == "1"  # QUICK=1 -> fast smoke test
N_REPEATS = 1 if QUICK else 3
N_ITER_OUTER = 4 if QUICK else 25
N_ITER_FINAL = 6 if QUICK else 40
MODELS_DIR = "models"
OLD_FEATURES = ["num_peaks", "peak_bpm", "interval_bpm", "spectral_bpm",
                "signal_std", "roi_detection_rate"]
NEW_FEATURES = OLD_FEATURES + ["autocorr_bpm", "welch_bpm", "cwt_bpm"]


def load_dataset():
    """RETURN (rows, y). rows = list of dicts with feature names as keys.
    y = list of reference RR."""
    cache = json.load(open("models/features_cache.json"))
    vids = rr_model.list_dataset_videos()
    rows, y = [], []
    for v in vids:
        fn = v["filename"]
        feats = cache[fn]
        rows.append({k: feats[k] for k in NEW_FEATURES})
        y.append(v["label_bpm"])
    return rows, y


def to_matrix(rows, feature_names):
    return np.array([[float(r[f]) for f in feature_names] for r in rows])


def default_models():
    """Baseline (untuned) models. Copy the exact defaults used in rr_model.py."""
    return {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
    }


PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", 0.5, 1.0],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.85, 1.0],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 5, 10],
    },
}


def metrics_from_preds(y, preds):
    err = np.abs(y - preds)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
        "within_2_bpm_pct": float(np.mean(err <= 2) * 100),
        "within_3_bpm_pct": float(np.mean(err <= 3) * 100),
    }


def nested_cv(estimator, X, y, grid=None, n_iter=None):
    """Outer RepeatedKFold. If grid is given, tune inside each outer-train fold."""
    n_iter = n_iter or N_ITER_OUTER
    outer = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    sums = np.zeros(len(y))
    counts = np.zeros(len(y))
    fold_maes = []
    for train_idx, test_idx in outer.split(X):
        est = clone(estimator)
        if grid is not None:
            est = RandomizedSearchCV(
                est, grid, n_iter=n_iter, cv=3,
                scoring="neg_mean_absolute_error",
                random_state=RANDOM_STATE, n_jobs=-1)
        est.fit(X[train_idx], y[train_idx])
        p = est.predict(X[test_idx])
        sums[test_idx] += p
        counts[test_idx] += 1
        fold_maes.append(mean_absolute_error(y[test_idx], p))
    preds = sums / counts  # per-video out-of-fold prediction (avg over repeats)
    out = metrics_from_preds(y, preds)
    out["fold_mae_mean"] = float(np.mean(fold_maes))
    out["fold_mae_std"] = float(np.std(fold_maes))
    out["oof_preds"] = preds.tolist()
    return out


def run_config(rows, y, feature_names, tuned):
    """Evaluate RF, GB, XGB and their mean-Ensemble under one configuration."""
    X = to_matrix(rows, feature_names)
    results = {}
    for name, est in default_models().items():
        grid = PARAM_GRIDS[name] if tuned else None
        results[name] = nested_cv(est, X, y, grid)
    ens = np.mean([results[n]["oof_preds"] for n in default_models()], axis=0)
    e = metrics_from_preds(y, ens)
    e["oof_preds"] = ens.tolist()
    results["ensemble"] = e
    return results


def improvement(base, tuned):
    return {
        "mae_improvement_pct": (base["mae"] - tuned["mae"]) / base["mae"] * 100,
        "rmse_improvement_pct": (base["rmse"] - tuned["rmse"]) / base["rmse"] * 100,
        "r2_change": tuned["r2"] - base["r2"],
    }


def paired_pvalue(y, preds_a, preds_b):
    ea = np.abs(y - np.array(preds_a))
    eb = np.abs(y - np.array(preds_b))
    try:
        return float(wilcoxon(ea, eb).pvalue)
    except ValueError:  # all differences are zero
        return None


def fit_final_models(rows, y, feature_names):
    """Tune on ALL data, save models + best params. Returns importances."""
    X = to_matrix(rows, feature_names)
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_params, tuned, importances = {}, {}, {}
    for name, est in default_models().items():
        search = RandomizedSearchCV(
            est, PARAM_GRIDS[name], n_iter=N_ITER_FINAL,
            cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X, y)
        tuned[name] = search.best_estimator_
        best_params[name] = {k: (None if v is None else
                                 (v.item() if hasattr(v, "item") else v))
                             for k, v in search.best_params_.items()}
        imp = sorted(zip(feature_names, tuned[name].feature_importances_),
                     key=lambda t: -t[1])[:3]
        importances[name] = [(f, float(v)) for f, v in imp]
    ens = VotingRegressor([(n, m) for n, m in tuned.items()])
    ens.fit(X, y)
    tuned["ensemble"] = ens
    for name, model in tuned.items():
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if os.path.exists(path):  # backup once
            backup = path.replace(".pkl", "_baseline.pkl")
            if not os.path.exists(backup):
                shutil.copy(path, backup)
        joblib.dump(model, path)
    with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    return best_params, importances


def main():
    rows, y = load_dataset()
    y = np.asarray(y, dtype=float)
    A = run_config(rows, y, OLD_FEATURES, tuned=False)   # old feats, default
    B = run_config(rows, y, NEW_FEATURES, tuned=False)   # new feats, default
    C = run_config(rows, y, NEW_FEATURES, tuned=True)    # new feats, tuned
    summary = {}
    for name in ["random_forest", "gradient_boosting", "xgboost", "ensemble"]:
        summary[name] = {
            "A_old_default": {k: v for k, v in A[name].items() if k != "oof_preds"},
            "B_new_default": {k: v for k, v in B[name].items() if k != "oof_preds"},
            "C_new_tuned": {k: v for k, v in C[name].items() if k != "oof_preds"},
            "feature_gain_A_to_B": improvement(A[name], B[name]),
            "tuning_gain_B_to_C": improvement(B[name], C[name]),
            "wilcoxon_p_B_vs_C": paired_pvalue(y, B[name]["oof_preds"], C[name]["oof_preds"]),
        }
    best_params, importances = fit_final_models(rows, y, NEW_FEATURES)
    summary["best_params"] = best_params
    summary["top3_feature_importances"] = importances
    summary["n_samples"] = int(len(y))
    with open(os.path.join(MODELS_DIR, "tuning_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # per-video predictions for plots
    with open(os.path.join(MODELS_DIR, "tuning_oof.json"), "w") as f:
        json.dump({"y": y.tolist(),
                   "B_ensemble": B["ensemble"]["oof_preds"],
                   "C_ensemble": C["ensemble"]["oof_preds"]}, f)
    print(json.dumps({k: summary[k] for k in
                      ["random_forest", "gradient_boosting", "xgboost", "ensemble"]},
                     indent=2)[:6000])


if __name__ == "__main__":
    main()
