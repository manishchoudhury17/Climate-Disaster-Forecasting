import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Climate Disaster Forecasting", layout="wide")

st.title("🌍 Global Multi-Hazard Climate Disaster Forecasting")
st.markdown("""
Demo: LSTM + Attention model forecasting **hazard risk** from recent climate signals.
Upload your model or use the provided training script to create `models/model.h5`.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Load Data")
    default_csv = Path("data/synthetic_climate.csv")
    if default_csv.exists():
        st.caption("Found data/synthetic_climate.csv")
    file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
    elif default_csv.exists():
        df = pd.read_csv(default_csv)
    else:
        st.warning("No CSV found. Please run `python src/generate_data.py`.")
        st.stop()

    st.write(df.head())

with col2:
    st.subheader("2) Model")
    from tensorflow.keras.models import load_model
    from src.attention import Attention  # custom layer
    import joblib

    model_path = Path("models/model.h5")
    scaler_path = Path("models/scaler.pkl")
    model = None
    scaler = None

    if model_path.exists() and scaler_path.exists():
        try:
            model = load_model(model_path, custom_objects={"Attention": Attention})
            scaler = joblib.load(scaler_path)
            st.success("Loaded model and scaler from models/")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.info("No model found. Train with `python src/train.py` first.")

st.markdown("---")
st.subheader("3) Predict")
seq_len = st.slider("Sequence length", 10, 60, 30, 1)

features = ["temp_anomaly","humidity","wind_speed","precip_mm","pressure","soi_index"]
if not all(f in df.columns for f in features):
    st.error("CSV must contain the required feature columns: " + ", ".join(features))
    st.stop()

def make_sequences(X, seq_len=30):
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
    return np.array(Xs)

if model is not None and scaler is not None:
    X = scaler.transform(df[features].values)
    X_seq = make_sequences(X, seq_len=seq_len)
    if len(X_seq) == 0:
        st.warning("Not enough rows for the chosen sequence length.")
    else:
        y_pred = model.predict(X_seq, verbose=0).flatten()
        st.line_chart(y_pred)
        st.caption("Predicted hazard risk (next-step) over time window.")
else:
    st.info("Model not loaded — only data preview available.")
