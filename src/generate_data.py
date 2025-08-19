import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import random

def generate(rows=20000, seed=42):
    rng = np.random.default_rng(seed)
    # Base signals
    t = np.arange(rows)
    temp_anomaly = 0.5*np.sin(2*np.pi*t/365) + rng.normal(0, 0.3, rows)
    humidity = 60 + 20*np.sin(2*np.pi*t/180) + rng.normal(0, 5, rows)
    wind_speed = 10 + 3*np.sin(2*np.pi*t/20) + rng.normal(0, 1.5, rows)
    precip_mm = np.clip(rng.gamma(2.0, 2.0, rows) - 2, 0, None)
    pressure = 1013 + 5*np.cos(2*np.pi*t/15) + rng.normal(0, 0.8, rows)
    soi_index = 0.3*np.sin(2*np.pi*t/500) + rng.normal(0, 0.05, rows)

    # Hazard risk proxy (nonlinear mix + noise)
    hazard_risk = (0.4*np.clip(temp_anomaly, -1, 1) +
                   0.2*(precip_mm/10) +
                   0.2*(wind_speed/20) +
                   0.1*(100-humidity)/100 +
                   0.1*(1015-pressure)/10)
    hazard_risk = (hazard_risk - hazard_risk.min()) / (hazard_risk.max() - hazard_risk.min())
    hazard_risk += rng.normal(0, 0.02, rows)
    hazard_risk = np.clip(hazard_risk, 0, 1)

    df = pd.DataFrame({
        "temp_anomaly": temp_anomaly,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precip_mm": precip_mm,
        "pressure": pressure,
        "soi_index": soi_index,
        "hazard_risk": hazard_risk
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", type=str, default="data/synthetic_climate.csv")
    ap.add_argument("--seq-len", type=int, default=30, help="(kept for compatibility, not applied here)")
    args = ap.parse_args()

    df = generate(rows=args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
