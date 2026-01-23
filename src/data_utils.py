from pathlib import Path
import pandas as pd

def load_dataset(path: str, lags, rolling_windows):
    df = pd.read_parquet(Path(path))
    df["Week"] = pd.to_datetime(df["Week"])        # ensure dtype
    df = df[df["MAE"] <= 100]                      # drop sentinels

    # --- lag features -------------------------------------------------
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # --- rolling mean/std features -----------------------------------
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
        df[f"roll_std_{w}"]  = df["y"].rolling(w).std()

    df = df.dropna().reset_index(drop=True)
    return df
