"""
Streamlit frontend for Car Price Prediction.

3 pages:
  1. Predict Price  — input Brand/model/Year/km/Transmission/Owner/FuelType → predicted ₹
  2. Data Insights  — EDA charts from used_car_dataset.csv (scatter, bar, histogram)
  3. Model Info     — live MAE / RMSE / R² + pipeline architecture

Run (Flask backend must be running on port 5000):
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── config ─────────────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:5000"
DATA_PATH = Path(__file__).parent / "used_car_dataset.csv"

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────────


def api_predict(payload: dict) -> dict | None:
    """
    Runs local prediction on the payload dictionary.
    Assumes `model` is a pre-fitted sklearn Pipeline or Estimator.
    """
    try:
        # Convert dictionary to single-row DataFrame
        input_data = pd.DataFrame([payload])
        
        # Make prediction (returns 1D numpy array)
        raw_prediction = float(model.predict(input_data)[0])
        
        # Guard against non-sensical negative predictions
        predicted_price = max(0.0, raw_prediction)

        return {
            "predicted_price_inr": round(predicted_price, 2),
            "predicted_price_lakh": round(predicted_price / 100000, 2)
        }

    except Exception as e:
        # Log error cleanly or let caller handle UI display
        print(f"Prediction error: {e}")
        return None
d


def api_metrics() -> dict:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    p = Path(__file__).parent / "model" / "model_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def api_options() -> dict:
    try:
        r = requests.get(f"{API_BASE}/options", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {
        "Transmission": ["Manual", "Automatic"],
        "Owner":        ["first", "second"],
        "FuelType":     ["Petrol", "Diesel", "Hybrid/CNG"],
        "Brands": ["Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Mahindra",
                   "Tata", "Mercedes-Benz", "Volkswagen", "BMW", "Ford",
                   "Renault", "Audi", "Skoda", "Kia"],
    }


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df["kmDriven"] = (
        df["kmDriven"].str.replace("km", "", regex=False)
        .str.replace(",", "", regex=False).str.strip().astype(float, errors="ignore")
    )
    df["AskPrice"] = (
        df["AskPrice"].str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False).str.strip().astype(float, errors="ignore")
    )
    df = df.dropna(subset=["kmDriven", "AskPrice"])
    df = df[df["AskPrice"] < 10_000_000]   # remove extreme outliers for charts
    return df

# ── sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚗 Car Price Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔮 Predict Price", "📊 Data Insights", "ℹ️ Model Info"])
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: used_car_dataset.csv\nModel: RandomForest · Flask · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Price":
    st.title("🚗 Car Price Prediction")
    st.markdown("Enter the car details below to get an estimated price.")

    opts = api_options()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Car Details")
        brand = st.selectbox("Brand", opts.get("Brands", ["Maruti Suzuki"]))
        model_name = st.text_input("Model (e.g. Swift, City, Creta)", value="Swift")
        year  = st.slider("Manufacturing Year", min_value=1995, max_value=2024, value=2018)
        km    = st.number_input("Kilometers Driven", min_value=0, max_value=500_000,
                                value=50_000, step=1_000)

    with col2:
        st.subheader("⚙️ Other Details")
        transmission = st.selectbox("Transmission", opts.get("Transmission", ["Manual", "Automatic"]))
        owner        = st.selectbox("Owner Type", opts.get("Owner", ["first", "second"]))
        fuel         = st.selectbox("Fuel Type",  opts.get("FuelType", ["Petrol", "Diesel", "Hybrid/CNG"]))

    st.markdown("---")

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        payload = {
            "Brand":        brand,
            "model":        model_name,
            "Year":         year,
            "kmDriven":     km,
            "Transmission": transmission,
            "Owner":        owner,
            "FuelType":     fuel,
        }
        with st.spinner("Calling prediction API …"):
            result = api_predict(payload)

        if result and "predicted_price_inr" in result:
            price_inr   = result["predicted_price_inr"]
            price_lakh  = result["predicted_price_lakh"]

            st.success(f"### 💰 Estimated Price: **₹ {price_inr:,.0f}**  ({price_lakh:.2f} Lakh)")

            # input summary bar chart
            st.markdown("#### Your Input Summary")
            summary = pd.DataFrame({
                "Feature": ["Year", "km Driven (÷1000)", "Owner\n(1=first)", "Trans\n(1=Auto)"],
                "Value":   [year, km / 1000,
                            1 if owner == "first" else 2,
                            1 if transmission == "Automatic" else 0],
            })
            fig = go.Figure(go.Bar(
                x=summary["Feature"], y=summary["Value"],
                marker_color="#3b82f6",
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=10),
                              title="Numeric Inputs at a Glance")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 Dataset Insights — used_car_dataset.csv")

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Copy `used_car_dataset.csv` into the app folder.")
        st.stop()

    st.markdown(f"**{len(df):,} rows** (after dedup + outlier removal) · **{len(df.columns)} columns**")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Price Distribution (₹)")
        fig = px.histogram(df, x="AskPrice", nbins=50, color_discrete_sequence=["#3b82f6"])
        fig.update_layout(xaxis_title="Ask Price (₹)", yaxis_title="Count", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Car Age vs Price")
        fig = px.scatter(df, x="Age", y="AskPrice",
                         color_discrete_sequence=["#7c5cd8"],
                         labels={"Age": "Age (years)", "AskPrice": "Price (₹)"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("km Driven vs Price")
        fig = px.scatter(df, x="kmDriven", y="AskPrice",
                         color_discrete_sequence=["#3b82f6"],
                         labels={"kmDriven": "km Driven", "AskPrice": "Price (₹)"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Manufacturing Year vs Price")
        fig = px.scatter(df, x="Year", y="AskPrice",
                         color_discrete_sequence=["#57606a"],
                         labels={"Year": "Year", "AskPrice": "Price (₹)"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Price by Brand (Top 15)")
    top_brands = (
        df.groupby("Brand")["AskPrice"].mean()
        .sort_values(ascending=False).head(15).reset_index()
    )
    fig = px.bar(top_brands, x="Brand", y="AskPrice",
                 color_discrete_sequence=["#3b82f6"],
                 labels={"AskPrice": "Avg Price (₹)"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Price by Fuel Type")
    fuel_avg = df.groupby("FuelType")["AskPrice"].mean().reset_index()
    fig = px.bar(fuel_avg, x="FuelType", y="AskPrice",
                 color_discrete_sequence=["#7c5cd8"],
                 labels={"AskPrice": "Avg Price (₹)"})
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transmission Split")
    t_count = df["Transmission"].value_counts().reset_index()
    fig = px.pie(t_count, names="Transmission", values="count",
                 color_discrete_sequence=["#3b82f6", "#7c5cd8"])
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")

    metrics = api_metrics()
    if metrics:
        st.subheader("Model Evaluation Metrics (30% test split)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",       f"₹ {metrics.get('MAE', 'N/A'):,}")
        m2.metric("RMSE",      f"₹ {metrics.get('RMSE', 'N/A'):,}")
        m3.metric("R² Score",  metrics.get("R2", "N/A"))
    else:
        st.info("Metrics not available. Run `python train_model.py` first.")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.markdown("""
| Step | Component | Detail |
|---|---|---|
| 1 | **Data Cleaning** | Remove ₹ & commas from AskPrice, strip "km" from kmDriven |
| 2 | **Drop Duplicates** | 724 duplicates removed from original 9582 rows |
| 3 | **Numeric Preprocessing** | `StandardScaler` on Year, kmDriven |
| 4 | **Categorical Encoding** | `OneHotEncoder(handle_unknown="ignore")` on Brand, model, Transmission, Owner, FuelType |
| 5 | **Model** | `RandomForestRegressor(n_estimators=200)` |
| 6 | **Backend** | Flask REST API (port 5000) |
| 7 | **Frontend** | Streamlit (port 8501) |
    """)

    st.markdown("---")
    st.subheader("Features Used")
    st.markdown("""
| Feature | Type | Notes |
|---|---|---|
| `Brand` | Categorical | 39 unique brands (Maruti Suzuki, Hyundai, BMW …) |
| `model` | Categorical | 400 unique models (Swift, City, Creta …) |
| `Year` | Numeric | Manufacturing year (1994–2024) |
| `kmDriven` | Numeric | Kilometers driven (cleaned from raw string) |
| `Transmission` | Categorical | Manual / Automatic |
| `Owner` | Categorical | first / second |
| `FuelType` | Categorical | Petrol / Diesel / Hybrid/CNG |
    """)

    st.markdown("---")
    st.subheader("Why RandomForest over LinearRegression?")
    st.markdown("""
The notebook used `LinearRegression` and got **R² = 0.68**.  
`RandomForestRegressor` handles:
- **Non-linear** relationships (luxury cars have exponentially higher prices)
- **High-cardinality** categoricals (400 models)
- **Outliers** (₹50k Alto vs ₹2.4Cr SL-Class)

Expected improvement: **R² ≥ 0.85**
    """)
