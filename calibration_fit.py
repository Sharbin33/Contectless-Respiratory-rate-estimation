"""
calibration_fit.py — Downloads the UCI Infrared Thermography Temperature Dataset,
inspects its structure, and fits a regression from forehead thermal intensity
to oral temperature. Saves the fitted model parameters for use in thermal_analysis.py.

Dataset: UCI ID 925 — "Infrared Thermography Temperature"
Source: Wang et al. (2021), 1020 subjects, 33 features, tabular CSV (no raw images).
"""

import json
import os
import sys
import numpy as np

# ── Step 1: Download & inspect the dataset ──────────────────────────────────

print("=" * 70)
print("STEP 1: Downloading UCI Infrared Thermography Temperature Dataset...")
print("=" * 70)

try:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=925)
except Exception as e:
    print(f"ERROR: Could not fetch dataset: {e}")
    print("Make sure 'ucimlrepo' is installed: pip install ucimlrepo")
    sys.exit(1)

X = dataset.data.features
y = dataset.data.targets

print(f"\nDataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target columns: {list(y.columns)}")
print(f"\nAll feature columns ({len(X.columns)}):")
for i, col in enumerate(X.columns):
    print(f"  {i+1:2d}. {col}")

print(f"\nFirst 3 rows of features:")
print(X.head(3).to_string())

print(f"\nFirst 3 rows of targets:")
print(y.head(3).to_string())

print(f"\nTarget statistics:")
print(y.describe().to_string())

# ── Step 2: Identify forehead temperature columns ───────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Identifying forehead-related columns...")
print("=" * 70)

# Look for forehead columns (FH = ForHead, FHCC = ForHead Center-Center, etc.)
forehead_cols = [c for c in X.columns if 'FH' in c.upper() or 'forehead' in c.lower()]
print(f"Forehead-related columns found: {forehead_cols}")

# Also identify all temperature columns (start with T_ or contain temp)
temp_cols = [c for c in X.columns if c.startswith('T_') or c.startswith('t_')]
print(f"\nAll T_ columns: {temp_cols}")

# Show stats for forehead columns
if forehead_cols:
    print(f"\nForehead column statistics:")
    print(X[forehead_cols].describe().to_string())

# ── Step 3: Fit regression ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Fitting calibration regression...")
print("=" * 70)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# Use the average oral temperature (aveOralF or aveOralM) as target
# Prefer aveOralF (fast mode) as it's the primary target in the dataset
target_col = None
for candidate in ['aveOralF', 'aveOralM']:
    if candidate in y.columns:
        target_col = candidate
        break

if target_col is None:
    print("ERROR: Could not find oral temperature target column!")
    print(f"Available target columns: {list(y.columns)}")
    sys.exit(1)

print(f"Using target column: '{target_col}'")

# Combine features and target, drop rows with NaN
import pandas as pd
df = pd.concat([X, y[[target_col]]], axis=1).dropna()
print(f"Samples after dropping NaN: {len(df)}")

# Try using forehead columns first, fall back to max facial temp if needed
if forehead_cols:
    # Use the first/best forehead column
    fh_col = forehead_cols[0]
    print(f"\nUsing forehead column: '{fh_col}'")
    print(f"  Range: {df[fh_col].min():.2f} to {df[fh_col].max():.2f}")
    print(f"  Mean:  {df[fh_col].mean():.2f}")
    print(f"  Std:   {df[fh_col].std():.2f}")
else:
    # Fall back: find the column most correlated with oral temp among T_ columns
    print("No forehead column found, finding best correlated T_ column...")
    correlations = {}
    for col in temp_cols:
        if col in df.columns:
            correlations[col] = abs(df[col].corr(df[target_col]))
    fh_col = max(correlations, key=correlations.get)
    print(f"Best correlated column: '{fh_col}' (r={correlations[fh_col]:.4f})")

# Print correlation
corr = df[fh_col].corr(df[target_col])
print(f"\nCorrelation between {fh_col} and {target_col}: {corr:.4f}")

# Fit linear regression: forehead_temp -> oral_temp
X_fit = df[[fh_col]].values
y_fit = df[target_col].values

lr = LinearRegression()
lr.fit(X_fit, y_fit)

y_pred_lr = lr.predict(X_fit)
mae_lr = mean_absolute_error(y_fit, y_pred_lr)
r2_lr = r2_score(y_fit, y_pred_lr)

print(f"\n--- Linear Regression Results ---")
print(f"  Slope:     {lr.coef_[0]:.6f}")
print(f"  Intercept: {lr.intercept_:.6f}")
print(f"  MAE:       {mae_lr:.4f}°F")
print(f"  R²:        {r2_lr:.4f}")

# Cross-validation
cv_scores = cross_val_score(LinearRegression(), X_fit, y_fit, cv=5, scoring='neg_mean_absolute_error')
print(f"  5-fold CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}°F")

# Also try with multiple facial temperature columns for a better fit
if len(temp_cols) > 1:
    available_temp_cols = [c for c in temp_cols if c in df.columns]
    if len(available_temp_cols) > 1:
        X_multi = df[available_temp_cols].values
        lr_multi = LinearRegression()
        lr_multi.fit(X_multi, y_fit)
        y_pred_multi = lr_multi.predict(X_multi)
        mae_multi = mean_absolute_error(y_fit, y_pred_multi)
        r2_multi = r2_score(y_fit, y_pred_multi)
        cv_multi = cross_val_score(LinearRegression(), X_multi, y_fit, cv=5, scoring='neg_mean_absolute_error')
        print(f"\n--- Multi-feature Linear Regression ({len(available_temp_cols)} features) ---")
        print(f"  MAE:       {mae_multi:.4f}°F")
        print(f"  R²:        {r2_multi:.4f}")
        print(f"  5-fold CV MAE: {-cv_multi.mean():.4f} ± {cv_multi.std():.4f}°F")

# ── Step 4: Determine if dataset temps are °F or °C ─────────────────────────

print("\n" + "=" * 70)
print("STEP 4: Determining temperature scale...")
print("=" * 70)

oral_mean = y_fit.mean()
fh_mean = df[fh_col].mean()
print(f"Oral temp mean: {oral_mean:.2f}")
print(f"Forehead temp mean: {fh_mean:.2f}")

if oral_mean > 90:
    temp_scale = "fahrenheit"
    print("=> Dataset temperatures are in FAHRENHEIT")
    # Convert slope/intercept to work with Celsius
    # If input is °F and output is °F:
    #   oral_F = slope * forehead_F + intercept
    #   oral_C = (oral_F - 32) / 1.8
    # Our pipeline works in °C, so we need the mapping:
    #   forehead_C -> oral_C
    # forehead_F = forehead_C * 1.8 + 32
    # oral_F = slope * (forehead_C * 1.8 + 32) + intercept
    # oral_C = (oral_F - 32) / 1.8
    # oral_C = (slope * (forehead_C * 1.8 + 32) + intercept - 32) / 1.8
    # oral_C = (slope * forehead_C * 1.8 + slope * 32 + intercept - 32) / 1.8
    # oral_C = slope * forehead_C + (slope * 32 + intercept - 32) / 1.8
    slope_c = lr.coef_[0]  # same slope since F->F, and C = (F-32)/1.8 is linear
    intercept_c = (lr.coef_[0] * 32 + lr.intercept_ - 32) / 1.8
    
    # Verify: convert a sample point
    sample_fh_f = fh_mean
    sample_oral_f = lr.predict([[sample_fh_f]])[0]
    sample_fh_c = (sample_fh_f - 32) / 1.8
    sample_oral_c_direct = (sample_oral_f - 32) / 1.8
    sample_oral_c_formula = slope_c * sample_fh_c + intercept_c
    print(f"  Verification: {sample_fh_c:.2f}°C -> {sample_oral_c_direct:.2f}°C (direct) vs {sample_oral_c_formula:.2f}°C (formula)")
    
    mae_c = mae_lr / 1.8  # Convert MAE from F to C
else:
    temp_scale = "celsius"
    print("=> Dataset temperatures are in CELSIUS")
    slope_c = lr.coef_[0]
    intercept_c = lr.intercept_
    mae_c = mae_lr

print(f"\nFinal calibration (°C → °C):")
print(f"  oral_temp_C = {slope_c:.6f} * forehead_skin_temp_C + {intercept_c:.6f}")
print(f"  MAE: ±{mae_c:.4f}°C")

# ── Step 5: Save calibration parameters ─────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 5: Saving calibration parameters...")
print("=" * 70)

calibration = {
    "model_type": "linear_regression",
    "description": "Forehead skin temperature (°C) to estimated oral/core temperature (°C)",
    "dataset": "UCI Infrared Thermography Temperature Dataset (ID 925)",
    "dataset_source": "Wang et al. (2021), 1020 subjects",
    "dataset_temp_scale": temp_scale,
    "feature_column_used": fh_col,
    "target_column_used": target_col,
    "n_samples": int(len(df)),
    "slope": float(slope_c),
    "intercept": float(intercept_c),
    "mae_celsius": float(mae_c),
    "r_squared": float(r2_lr),
    "correlation": float(corr),
    "forehead_temp_range_c": {
        "min": float((df[fh_col].min() - 32) / 1.8) if temp_scale == "fahrenheit" else float(df[fh_col].min()),
        "max": float((df[fh_col].max() - 32) / 1.8) if temp_scale == "fahrenheit" else float(df[fh_col].max()),
    },
    "oral_temp_range_c": {
        "min": float((y_fit.min() - 32) / 1.8) if temp_scale == "fahrenheit" else float(y_fit.min()),
        "max": float((y_fit.max() - 32) / 1.8) if temp_scale == "fahrenheit" else float(y_fit.max()),
    },
    "honesty_notes": [
        "This calibration was fitted on a specific IR camera setup (Wang et al. 2021) — different cameras/distances/environments will shift the relationship.",
        "The dataset provides pre-extracted facial temperatures, NOT raw pixel intensities from YOUR camera.",
        "Using this calibration with grayscale pixel intensity from a different thermal camera requires an additional mapping step (pixel intensity -> skin temperature) that is camera-specific.",
        "Label output as 'Calibrated (reference dataset)' — NOT 'Clinically Validated'."
    ]
}

cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_params.json")
with open(cal_path, 'w') as f:
    json.dump(calibration, f, indent=2)

print(f"Saved to: {cal_path}")
print(f"\nCalibration summary:")
print(f"  oral_C = {slope_c:.4f} * forehead_skin_C + {intercept_c:.4f}")
print(f"  MAE: ±{mae_c:.4f}°C  |  R²: {r2_lr:.4f}")
print(f"  Valid forehead input range: {calibration['forehead_temp_range_c']['min']:.1f}–{calibration['forehead_temp_range_c']['max']:.1f}°C")
print(f"\n** IMPORTANT: See honesty_notes in calibration_params.json for caveats.")
print("Done.")
