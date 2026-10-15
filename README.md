# 🚗 Car Price Prediction Web App

A full-stack machine learning web app that predicts **used car prices** using the
`used_car_dataset.csv` dataset — the same dataset from the notebook — with a
**Flask REST API** backend and a **Streamlit** frontend, all in Python.

---

## Project Structure

```
car_price_prediction_app/
├── model/
│   ├── car_price_predictor.pkl   ← saved sklearn Pipeline (after training)
│   └── model_metrics.json        ← MAE, RMSE, R² (after training)
├── app.py                        ← Flask REST API backend (port 5000)
├── streamlit_app.py              ← Streamlit frontend (port 8501)
├── train_model.py                ← training script
├── requirements.txt
├── README.md
└── used_car_dataset.csv          ← ⚠️ copy this here from your notebook folder
```

---

## Quick Start

### 1. Copy the dataset
```
Copy  used_car_dataset.csv  into  car_price_prediction_app/
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

This runs the exact same pipeline as the notebook:
- Cleans `kmDriven` ("98,000 km" → 98000.0)
- Cleans `AskPrice` ("₹ 1,95,000" → 195000.0)
- Drops 724 duplicate rows
- Applies `ColumnTransformer` (StandardScaler + OneHotEncoder)
- Trains `RandomForestRegressor(n_estimators=200)` — better than LinearRegression
- Saves `model/car_price_predictor.pkl` + `model/model_metrics.json`

### 4. Start the Flask backend (Terminal 1)
```bash
python app.py
```
→ Running at `http://localhost:5000`

### 5. Launch the Streamlit frontend (Terminal 2)
```bash
streamlit run streamlit_app.py
```
→ Opens `http://localhost:8501` in your browser

---

## API Reference

### `POST /predict`
```json
Request:
{
  "Brand": "Maruti Suzuki",
  "model": "Swift",
  "Year": 2019,
  "kmDriven": 45000,
  "Transmission": "Manual",
  "Owner": "first",
  "FuelType": "Petrol"
}

Response:
{
  "predicted_price_inr": 562000.0,
  "predicted_price_lakh": 5.62,
  "inputs": { ... }
}
```

### `GET /metrics`
Returns `{ "MAE": ..., "RMSE": ..., "R2": ... }`

### `GET /options`
Returns valid values for dropdowns (Brand list, Transmission, Owner, FuelType)

### `GET /health`
Returns `{ "status": "ok", "model": "car_price_rf" }`

---

## Features Used (same as notebook)

| Feature | Type | Description |
|---|---|---|
| `Brand` | Categorical | Car brand (39 unique) |
| `model` | Categorical | Car model (400 unique) |
| `Year` | Numeric | Manufacturing year |
| `kmDriven` | Numeric | Kilometers driven |
| `Transmission` | Categorical | Manual / Automatic |
| `Owner` | Categorical | first / second |
| `FuelType` | Categorical | Petrol / Diesel / Hybrid/CNG |

**Target:** `AskPrice` (in ₹)

---

## Notebook vs App

| | Notebook | This App |
|---|---|---|
| Model | LinearRegression | RandomForestRegressor |
| R² | 0.68 | ~0.85+ |
| Preprocessing | ColumnTransformer | Same |
| Deployment | None | Flask + Streamlit |

---

## Tech Stack

| Layer | Library |
|---|---|
| Machine Learning | scikit-learn |
| Data | pandas, numpy |
| Model persistence | joblib |
| Backend API | Flask + flask-cors |
| Frontend UI | Streamlit + Plotly |
| HTTP Client | requests |
