"""
train_model.py
--------------
Loads dataset.csv, trains three ML models, evaluates them,
selects the best one, and saves it as model.pkl + model_metrics.json.

Run once before launching app.py:
    python3 train_model.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset.csv …")
df = pd.read_csv("dataset.csv")

FEATURES = ["Lg_nm", "Wfin_nm", "Hfin_nm", "Tox_nm", "Vth_V",
            "VGS_V", "VDS_V", "Temp_K"]
TARGET   = "Id_A"

X = df[FEATURES].values
y = df[TARGET].values

# ── 2. Train / test split (80 / 20) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ── 3. Define models (each wrapped in a pipeline with StandardScaler) ─────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.8, random_state=42
        )),
    ]),
}

# ── 4. Train, evaluate, collect metrics ──────────────────────────────────────
results    = {}
best_name  = None
best_r2    = -np.inf
best_pipe  = None

for name, pipe in models.items():
    print(f"\nTraining: {name} …")
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    r2_train  = r2_score(y_train, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results[name] = {
        "r2_train":  round(r2_train,  6),
        "r2_test":   round(r2_test,   6),
        "mae_test":  mae_test,
        "rmse_test": rmse_test,
        "y_test":    y_test.tolist(),
        "y_pred":    y_pred_test.tolist(),
    }

    print(f"  Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
          f"RMSE={rmse_test:.3e} A")

    if r2_test > best_r2:
        best_r2   = r2_test
        best_name = name
        best_pipe = pipe

# ── 5. Extract Random Forest feature importances ─────────────────────────────
rf_pipe = models["Random Forest"]
rf_importances = rf_pipe.named_steps["model"].feature_importances_.tolist()

# ── 6. Save best model ────────────────────────────────────────────────────────
joblib.dump(best_pipe, "model.pkl")
print(f"\nBest model: {best_name}  (R²={best_r2:.4f})")
print("Saved → model.pkl")

# ── 7. Save all metrics to JSON ───────────────────────────────────────────────
metrics = {
    "best_model":      best_name,
    "features":        FEATURES,
    "n_samples":       len(df),
    "rf_importances":  rf_importances,
    "models":          results,
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved → model_metrics.json")

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n--- Model Comparison ---")
print(f"{'Model':<25} {'R²':>8}  {'RMSE (A)':>12}")
print("-" * 50)
for name, m in results.items():
    marker = " ★" if name == best_name else "  "
    print(f"{name:<25}{marker} {m['r2_test']:>8.4f}  {m['rmse_test']:>12.3e}")
