# Global Multi-Hazard Climate Disaster Forecasting (LSTM + Attention + Streamlit)

A portfolio-ready project that demonstrates **sequence modeling** for multi-hazard event forecasting using an
**LSTM with an Attention layer**. Includes:
- Synthetic data generator (so recruiters can run it without external datasets)
- `train.py` to train and save a Keras model (`models/model.h5`)
- `app.py` (Streamlit) to visualize predictions and interact with the model
- Clean `README.md` and `requirements.txt`

> Swap the synthetic generator with your real dataset later (CSV with features over time).

## Quickstart

```bash
# 1) Create & activate venv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Generate synthetic dataset
python src/generate_data.py --rows 20000 --seq-len 30

# 4) Train the model (saves to models/model.h5)
python src/train.py --epochs 5 --batch-size 64

# 5) Run the dashboard
streamlit run app.py
```

## Project Structure
```
Climate-Disaster-Forecasting/
├─ data/
│  ├─ synthetic_climate.csv
├─ models/
│  └─ model.h5
├─ notebooks/
│  └─ EDA.ipynb               # (empty stub, fill if you want)
├─ src/
│  ├─ attention.py
│  ├─ generate_data.py
│  └─ train.py
├─ app.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Features & Signals (Synthetic)
- temp_anomaly, humidity, wind_speed, precip_mm, pressure, soi_index
- Rolling windows used to predict next-step **hazard_risk** in [0,1].

## Notes
- This is engineered for **resume demos**: clean code, no heavy infra.
- For real data: replace `generate_data.py` with your loader and keep the same training API.
