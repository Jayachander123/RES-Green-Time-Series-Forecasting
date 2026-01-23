# src/utils/metrics.py
"""
Run-level metrics helper for the RES paper.

aggregate()  -> dict with tot_benefit, tot_cost, res, n_retrains, n_steps
save_summary() -> writes that dict as pretty-printed JSON
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


REQUIRED = {"benefit", "cost"}
ALTERNATIVES = ["benefit_raw", "benefit_norm"]

import numpy as np
from sklearn.metrics import mean_squared_error

# ── NEW: RMSE helper for the ablation study ──────────────────────────
def calc_rmse(y_true, y_pred) -> float:
    """Root-mean-squared error.  y* may be list-like or NumPy array."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def aggregate(csv_path: str | Path) -> dict[str, float]:
    """
    Read the per-step log at *csv_path* and return the five headline
    numbers that go into summary.json.
    """
    df = pd.read_csv(csv_path)

    # if not REQUIRED.issubset(df.columns):
    #     missing = REQUIRED - set(df.columns)
    #     raise ValueError(f"log file missing columns: {missing}")
        # Back-compat: map raw or norm column to 'benefit' if absent
    if "benefit" not in df.columns:
        for alt in ALTERNATIVES:
            if alt in df.columns:
                df["benefit"] = df[alt]
                break

    if not REQUIRED.issubset(df.columns):
        missing = REQUIRED - set(df.columns)
        raise ValueError(f"log file missing columns: {missing}")

    tot_benefit = float(df["benefit"].sum())
    tot_cost    = float(df["cost"].sum())
    res         = float("inf") if tot_cost == 0 else tot_benefit / tot_cost

    # 'retrain' := 1/0 flag OR count cost>0 events if flag absent
    if "retrain" in df.columns:
        n_retrains = int(df["retrain"].sum())
    else:
        n_retrains = int((df["cost"] > 0).sum())

    res = dict(
        tot_benefit=tot_benefit,
        tot_cost=tot_cost,
        res=res,
        n_retrains=n_retrains,
        n_steps=int(len(df)),
    )
    return res


def save_summary(csv_path: str | Path, out_json: str | Path) -> None:
    """
    Convenience wrapper: aggregate → write JSON next to the run artefacts.
    """
    summary = aggregate(csv_path)
    Path(out_json).write_text(json.dumps(summary, indent=2))
    print(f"✓ summary written to {out_json}")


# ------------------------------------------------------------------
# Tiny CLI so you can call the helper by hand
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate run log → summary.json")
    ap.add_argument("--log", required=True, help="CSV produced during the run")
    ap.add_argument("--out", required=True, help="Path to summary.json")
    args = ap.parse_args()
    save_summary(args.log, args.out)
