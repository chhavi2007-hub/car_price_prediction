"""
Train Car Price Prediction model from used_car_dataset.csv.

Mirrors the notebook pipeline exactly:
  - Clean kmDriven  : "98,000 km"   -> 98000.0
  - Clean AskPrice  : "₹ 1,95,000" -> 195000.0
  - Drop duplicates + NaN rows
  - Features: Brand, model, Year, kmDriven, Transmission, Owner, FuelType
  - ColumnTransformer: StandardScaler (numeric) + OneHotEncoder (categorical)
  - Regressor: RandomForestRegressor (better R2 than LinearRegression)
  - Save pipeline as model/car_price_predictor.pkl

Run:
    python train_model.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_PATH    = BASE_DIR / "used_car_dataset.csv"
MODEL_DIR    = BASE_DIR / "model"
MODEL_PATH   = MODEL_DIR / "car_price_predictor.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# ── features (same as notebook) ────────────────────────────────────────────────
FEATURES    = ["Brand", "model", "Year", "kmDriven", "Transmission", "Owner", "FuelType"]
NUMERICAL   = ["Year", "kmDriven"]
CATEGORICAL = ["Brand", "model", "Transmission", "Owner", "FuelType"]
TARGET      = "AskPrice"


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows")

    # 1. drop duplicates (notebook had 724)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] After drop_duplicates: {len(df)} rows")

    # 2. clean kmDriven: "98,000 km" -> 98000.0
    df["kmDriven"] = (
        df["kmDriven"]
        .str.replace("km", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # 3. clean AskPrice: "₹ 1,95,000" -> 195000.0
    df["AskPrice"] = (
        df["AskPrice"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # 4. drop NaN rows (kmDriven had nulls)
    df.dropna(subset=FEATURES + [TARGET], inplace=True)
    print(f"[INFO] After dropna: {len(df)} rows")
    return df


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Copy  used_car_dataset.csv  into the car_price_prediction_app/ folder."
        )

    df = load_and_clean(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── preprocessing pipeline (identical to notebook) ─────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(),                        NUMERICAL),
        ("cat", OneHotEncoder(handle_unknown="ignore"),  CATEGORICAL),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("[INFO] Training RandomForest pipeline …")
    pipeline.fit(X_train, y_train)

    # ── evaluate ────────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2, 4),
    }
    print(f"[METRICS] MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}  R²={r2:.4f}")

    # ── save model + metrics ────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[INFO] Model saved   → {MODEL_PATH}")
    print(f"[INFO] Metrics saved → {METRICS_PATH}")


if __name__ == "__main__":
    train()
