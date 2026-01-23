# src/features/weekly_univariate.py
from __future__ import annotations
import pandas as pd

# -----------------------------------------------------------------
LAGS = (1, 4, 52)                       # 1 week, 4 weeks, 52 weeks
ROLL_WINDOWS = (4, 12)                  # for mean; std only on 4-week
# -----------------------------------------------------------------


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling-window statistics and calendar dummies to a weekly
    univariate series.  Returns a *new* dataframe, original is untouched.
    Required input columns: ['Week', 'y'] where Week is Monday stamp.
    """
    out = df.copy()

    # -----------------------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------------------
    for k in LAGS:
        out[f"lag_{k}"] = out["y"].shift(k)

    # -----------------------------------------------------------------
    # ROLLING MEAN / STD  (shift(1) so target week itself isn't included)
    # -----------------------------------------------------------------
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(w).mean()

    # short volatility proxy
    out["roll_std_4"] = out["y"].shift(1).rolling(4).std()

    # -----------------------------------------------------------------
    # CALENDAR DUMMIES
    # -----------------------------------------------------------------
    out["week_of_year"] = out["Week"].dt.isocalendar().week.astype(int)
    out["month"]        = out["Week"].dt.month

    # -----------------------------------------------------------------
    # DROP THE WARM-UP ROWS WITH NaNs (caused by shift / rolling)
    # -----------------------------------------------------------------
    out = out.dropna().reset_index(drop=True)
    return out


# -----------------------------------------------------------------
# convenience helper: from parquet → engineered df
def load_with_features(path: str | pd.Path) -> pd.DataFrame:
    """
    Read a cleaned parquet/csv file (columns Week,y) and append features.
    """
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, parse_dates=["Week"])
    return add_features(df)
