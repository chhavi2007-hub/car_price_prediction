"""
Flask REST API backend for Car Price Prediction.

Endpoints:
    GET  /health          — liveness check
    GET  /metrics         — model evaluation metrics (MAE, RMSE, R2)
    GET  /options         — valid dropdown values for Brand, Transmission, Owner, FuelType
    POST /predict         — predict car price from JSON body

Run (after training):
    python app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "model" / "car_price_predictor.pkl"
METRICS_PATH = BASE_DIR / "model" / "model_metrics.json"

FEATURES    = ["Brand", "model", "Year", "kmDriven", "Transmission", "Owner", "FuelType"]

# ── load model ─────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\nRun  python train_model.py  first."
    )

pipeline = joblib.load(MODEL_PATH)
print(f"[Flask] Model loaded from {MODEL_PATH}")

# ── static option lists (from notebook value_counts) ──────────────────────────
OPTIONS = {
    "Transmission": ["Manual", "Automatic"],
    "Owner":        ["first", "second"],
    "FuelType":     ["Petrol", "Diesel", "Hybrid/CNG"],
    "Brands": [
        "Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Mahindra",
        "Tata", "Mercedes-Benz", "Volkswagen", "BMW", "Ford",
        "Renault", "Audi", "Skoda", "Kia", "Chevrolet", "MG",
        "Nissan", "Land Rover", "Jeep", "Volvo", "Mini", "Jaguar",
        "Fiat", "Datsun", "Lexus", "Porsche", "Mitsubishi", "Isuzu",
        "Force", "Ambassador", "Rolls-Royce",
    ],
}

# ── app ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow Streamlit (different port) to call this API


@app.get("/")
def home():
    return "Car Price Prediction App is running!"

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "car_price_rf"})


@app.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        return jsonify(json.loads(METRICS_PATH.read_text()))
    return jsonify({"error": "metrics not found"}), 404


@app.get("/options")
def options():
    return jsonify(OPTIONS)


@app.post("/predict")
def predict():
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    missing = [f for f in FEATURES if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{f: body[f] for f in FEATURES}])
        predicted_price = pipeline.predict(input_df)[0]
        return jsonify({
            "predicted_price_inr": round(float(predicted_price), 2),
            "predicted_price_lakh": round(float(predicted_price) / 100_000, 2),
            "inputs": {f: body[f] for f in FEATURES},
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
