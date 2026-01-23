# src/etl.py
# ------------------------------------------------------------
# Data cleaning + weekly aggregation utilities for the BCR paper
# ------------------------------------------------------------
import argparse
from pathlib import Path
# from .feature_utils import add_features
from src.utils.feature_utils import add_features
import pandas as pd


# -----------------------------------------------------------------
# 1. Helpers
# -----------------------------------------------------------------
_SENTINEL_MAE_THRESHOLD = 100.0
_WEEK_ANCHOR = "W"  # Monday-anchored weekly period


def _add_week_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Attach a Monday-anchored 'Week' datetime column."""
    df["Week"] = (
        df[date_col].dt.to_period(_WEEK_ANCHOR).apply(lambda p: p.start_time)
    )
    return df


# -----------------------------------------------------------------
# 2. M5-specific loader
# -----------------------------------------------------------------
def _load_and_clean_m5(raw_sales: Path, calendar_path: Path) -> pd.DataFrame:
    """
    1. Map the wide `d_1 … d_n` columns to real dates using calendar.csv.
    2. Melt to long format (date, sales).
    3. Aggregate to weekly sums.
    Returns a DataFrame with columns ['Week', 'y'].
    """
    # 2.1 calendar for mapping d_x -> YYYY-MM-DD
    cal = pd.read_csv(calendar_path, usecols=["d", "date"], parse_dates=["date"])

    # 2.2 sales wide table
    sales = pd.read_csv(raw_sales)
    id_cols = sales.columns[:6]          # item_id, dept_id, ...
    value_cols = sales.columns[6:]       # d_1 … d_1941

    # melt to long
    long = sales.melt(
        id_vars=id_cols, value_vars=value_cols, var_name="d", value_name="y"
    )

    # join calendar to get real dates
    long = long.merge(cal, on="d", how="left")  # adds 'date'
    long = long[["date", "y"]].copy()

    # weekly aggregate
    long["date"] = pd.to_datetime(long["date"])
    long = _add_week_column(long, "date")
    weekly = long.groupby("Week", as_index=False)["y"].sum()

    return weekly


# -----------------------------------------------------------------
# 3. Generic CSV loader   (NYC-Taxi, etc.)
# -----------------------------------------------------------------
def _load_and_clean_generic(raw_path: Path) -> pd.DataFrame:

    # candidate_cols = ["date", "timestamp", "pickup_datetime"]
    candidate_cols = ["Week", "date", "timestamp", "pickup_datetime"]
    header = pd.read_csv(raw_path, nrows=0).columns
    for col in candidate_cols:
        if col in header:
            date_col = col
            break
    else:
        raise ValueError(f"No recognised date column in {raw_path}")

    # 1. read – force strict date parsing
    df = pd.read_csv(
        raw_path,
        parse_dates=[date_col],
        date_parser=lambda s: pd.to_datetime(s, errors="coerce")  # <- NEW
    )

    df = df.rename(columns={date_col: "date"})
    if df["date"].dt.to_period("W").nunique() == len(df):
        df = df.rename(columns={"date": "Week"})[["Week", "y"]]
        return df.sort_values("Week").reset_index(drop=True)
    df = df.dropna(subset=["date"])          # <- NEW  (removes bad rows)

    if "y" not in df.columns:
        raise ValueError("Expected a target column named 'y'")

    # 2. Monday-anchored week + aggregation
    df = _add_week_column(df, "date")
    weekly = (df.groupby("Week", as_index=False)["y"]
                .sum()
                .sort_values("Week")
                .reset_index(drop=True))
    return weekly



# -----------------------------------------------------------------
# 4. Public orchestration function
# -----------------------------------------------------------------
def load_and_clean(raw_path: str , out_path: str) -> None:
    """
    Dispatch to the correct loader based on the filename pattern.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.name in ("sales_train_evaluation.csv", "sales_train_validation.csv"):
        print(raw_path)
        cal_path = raw_path.with_name("calendar.csv")
        if not cal_path.exists():
            raise FileNotFoundError(
                "M5 calendar.csv not found next to sales_train_evaluation.csv"
            )
        df = _load_and_clean_m5(raw_path, cal_path)
    else:
        df = _load_and_clean_generic(raw_path)

    # df = add_features(df, lags=(1, 4, 52), rolls=(4, 12))

    df.to_parquet(f"{out_path}.parquet", index=False)
    df.to_csv(f"{out_path}.csv", index=False)
    print(f"✅ saved clean file to {out_path}  ({len(df)} rows)")


# -----------------------------------------------------------------
# 5. Tiny CLI for ad-hoc runs
# -----------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(description="Clean raw CSV → weekly Parquet")
    p.add_argument("--raw", required=True, help="Path to raw CSV")
    p.add_argument("--out", required=True, help="Path to output Parquet")
    args = p.parse_args()
    load_and_clean(args.raw, args.out)


if __name__ == "__main__":
    _cli()
