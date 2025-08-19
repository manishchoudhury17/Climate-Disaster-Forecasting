import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from attention import Attention

def make_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_model(seq_len, n_features):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = Attention()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic_climate.csv")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model-out", type=str, default="models/model.h5")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    features = ["temp_anomaly","humidity","wind_speed","precip_mm","pressure","soi_index"]
    target = "hazard_risk"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    y = df[target].values

    X_seq, y_seq = make_sequences(X_scaled, y, seq_len=args.seq_len)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    model = build_model(args.seq_len, len(features))
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    # Save scaler for app usage
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    main()
