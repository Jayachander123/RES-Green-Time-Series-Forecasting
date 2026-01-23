# src/evaluate.py
"""
Generic evaluation helpers.
Currently supported model kinds:
    • lightgbm   (sklearn API)
    • prophet    (yhat column)
    • arima      (pmdarima)
    • lstm       (PyTorch, sequence-to-one regression)

Extend the `evaluate()` function when you add new wrappers.
"""
from __future__ import annotations
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


# ------------------------------------------------------------------
def _split_X_y(df: pd.DataFrame):
    """Drop non-feature columns and return X, y (for tabular models)."""
    X = df.drop(columns=["Week", "y"])
    y = df["y"]
    return X, y


# ------------------------------------------------------------------
# -----------------------  L S T M   H E L P E R S  -----------------
# ------------------------------------------------------------------
def _infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Everything that is *not* Week or y is treated as a feature."""
    return [c for c in df.columns if c not in ("Week", "y")]


def _build_windows(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: List[str],
    device: str = "cpu",
):
    """
    Convert an ordered slice of df into sliding windows.

    Returns
    -------
    x_tensor : torch.FloatTensor  (batch, seq_len, n_features)
    y_tensor : torch.FloatTensor  (batch,)   – one-step-ahead target
    """
    import torch

    # Ensure chronological order (important when tuner sub-samples)
    df_sorted = df.sort_values("Week")

    # Pull feature matrix (T, F) and target vector (T,)
    x_mat = df_sorted[feature_cols].values.astype("float32")
    y_vec = df_sorted["y"].values.astype("float32")

    if len(df_sorted) <= window_size:
        raise ValueError(
            f"LSTM evaluation needs > window_size rows "
            f"(got {len(df_sorted)}, window_size={window_size})."
        )

    # Sliding window
    windows = []
    targets = []
    for idx in range(window_size, len(df_sorted)):
        windows.append(x_mat[idx - window_size : idx])
        targets.append(y_vec[idx])

    # x_tensor = torch.tensor(windows, device=device)
    x_tensor = torch.as_tensor(np.array(windows), device=device)
    y_tensor = torch.tensor(targets, device=device)

    return x_tensor, y_tensor


# ------------------------------------------------------------------
def evaluate(model: Any, df: pd.DataFrame, kind: str = "lightgbm") -> float:
    """
    Return MAE of *model* on the given DataFrame slice.
    Works for one or many rows.  Dispatches on `kind`.
    """
    if kind == "lightgbm":
        X, y_true = _split_X_y(df)
        y_pred    = model.predict(X).reshape(-1) 

    elif kind == "prophet":
        df_p      = df.rename(columns={"Week": "ds", "y": "y"})
        y_pred    = model.predict(df_p)["yhat"]
        y_true    = df_p["y"]

    elif kind == "arima":
        # pmdarima returns ndarray even for multi-step calls
        y_true    = df["y"].values
        y_pred    = model.predict(n_periods=len(df))

    elif kind in ("lstm", "gru", "transformer"):
        # ----------------------------------------------------------
        # Device handling
        import torch

        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            # Model could be a wrapper exposing `.net`
            device = next(model.net.parameters()).device  # type: ignore

        # Hyper-params (let the tuner attach them to the model if it wants)
        window_size   = getattr(model, "window_size", 30)
        feature_cols  = getattr(model, "feature_cols", _infer_feature_cols(df))

            # --- pad with training tail if df is too short -----------------
        if len(df) <= window_size:
            hist_tail = getattr(model, "history_tail", None)
            need      = window_size + 1 - len(df)   # +1 because _build_windows
            if hist_tail is not None and len(hist_tail) >= need:
                df = pd.concat([hist_tail.tail(need), df])
            else:
                raise ValueError(
                    f"LSTM evaluation needs ≥ {window_size+1} rows "
                    f"(got {len(df)} and no usable history_tail)."
                )

        # Build input/output tensors
        x_seq, y_true_t = _build_windows(
            df,
            window_size=window_size,
            feature_cols=feature_cols,
            device=device,
        )
        

        # Forward pass
        model.eval()
        with torch.no_grad():
            x_seq = (x_seq - model.mean_) / model.std_
            y_pred_t = model(x_seq)

        # Flatten possible (batch, 1) → (batch,)
        y_pred_t = y_pred_t.squeeze(-1)

        # Back to NumPy for sklearn metric
        # y_true = y_true_t.cpu().numpy()
        # y_pred = y_pred_t.cpu().numpy()
        y_true = np.atleast_1d(y_true_t.cpu().numpy())
        y_pred = np.atleast_1d(y_pred_t.cpu().numpy())

    

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    mae = float(mean_absolute_error(y_true, y_pred))

    # Short debug print so the console is not flooded
    if len(df) == 1:
        wk = df.iloc[0]["Week"]
        print(f"      MAE on week {wk.date()}: {mae:,.3f}")

    return mae


