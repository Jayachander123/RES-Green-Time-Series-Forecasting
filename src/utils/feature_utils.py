#!/usr/bin/env python
"""
src/utils/feature_utils.py

Usage
    python make_processed.py data/raw/nyc_taxi_weekly.csv \
                              data/processed/nyc_taxi_clean.parquet
    # same call for electricity, M5, traffic
"""
import argparse, pathlib as P, pandas as pd

def add_features(df, lags=(1, 4, 52), rolls=(4, 12)):
    df = df.copy()
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)
    for w in rolls:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
    return df.dropna().reset_index(drop=True)

def main(src, dst, lags, rolls):
    df = pd.read_csv(src, parse_dates=["Week"])
    df = add_features(df, lags=lags, rolls=rolls)
    P.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"✓ wrote {dst}  ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lags",  type=int, nargs="+", default=[1,4,52])
    ap.add_argument("--rolls", type=int, nargs="+", default=[4,12])
    a = ap.parse_args()
    main(a.src, a.dst, a.lags, a.rolls)
