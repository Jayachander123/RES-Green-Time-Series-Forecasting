"""
One file, many trainers.
`train(df, kind, params)` → (model_object, fit_time_seconds).

Supported kinds
  • lightgbm   – tabular features
  • prophet    – Facebook/Meta Prophet
  • arima      – pmdarima / statsmodels
  • lstm       – PyTorch sequence-to-one regression
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings 

import logging, traceback, sys

def _dump(exc: BaseException, tag="arima"):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logging.getLogger(f"res.{tag}").error("⚠️  Uncaught exception\n%s", tb)
    return exc

# ==============================================================
# make window_size safe for short series
# ==============================================================
            

def _safe_window_size(series_len: int,
                      requested: int,
                      min_size: int = 4) -> int:
    """
    Ensure the window is strictly smaller than the series.
    If `requested` fits it is returned unchanged; otherwise we shrink
    it to `max(min_size, series_len - 1)` and warn once.
    """
    if series_len > requested:
        return requested

    new_ws = max(min_size, series_len - 1)
    warnings.warn(
        f"[auto] window_size clipped from {requested} to {new_ws} "
        f"because the series has only {series_len} rows"
    )
    return new_ws


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────
def _split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["Week", "y"])
    y = df["y"]
    return X, y


def _filter_valid(fn_or_cls, params: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Return only those items from `params` that are accepted by the
    signature of the given function / class constructor.
    """
    if not params:
        return {}
    sig = inspect.signature(fn_or_cls)
    return {k: v for k, v in params.items() if k in sig.parameters}


# ──────────────────────────────────────────────────────────────
# concrete trainer functions
# ──────────────────────────────────────────────────────────────
DEFAULT_LGB_PARAMS: Dict[str, Any] = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=255,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    verbosity=-1,
    n_jobs=int(os.getenv("LGBM_NUM_THREADS", 1)),
)


def _train_lightgbm(df: pd.DataFrame, params: Dict[str, Any] | None):
    """Return fitted LightGBM model + elapsed seconds."""
    import lightgbm as lgb

    # Merge user params over defaults
    all_params = {**DEFAULT_LGB_PARAMS, **(params or {})}

    X, y = _split_X_y(df)
    t0 = time.time()
    mdl = lgb.LGBMRegressor(**all_params)
    mdl.fit(X, y)
    return mdl, time.time() - t0


def _train_prophet(df: pd.DataFrame, params: Dict[str, Any] | None):
    from prophet import Prophet

    valid = _filter_valid(Prophet.__init__, params)
    tmp = df.rename(columns={"Week": "ds", "y": "y"})

    t0 = time.time()
    mdl = Prophet(**valid).fit(tmp)
    return mdl, time.time() - t0


def _train_arima(df: pd.DataFrame, params: Dict[str, Any] | None):
    try:
        import pmdarima as pm
    
        valid = _filter_valid(pm.auto_arima, params)
    
        t0 = time.time()
        mdl = pm.auto_arima(df["y"], **valid)
        return mdl, time.time() - t0
    except Exception as e:
        print(f"Error training Arima model : {e}")
        _dump(e)
        raise


# ──────────────────────────────────────────────────────────────
#                L S T M    (PyTorch)
# ──────────────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    """
    Minimal 1-layer LSTM that outputs one number per window.
    Forward signature: x (B, T, F)  →  ŷ (B,) or (B, 1)
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=n_features,
                           hidden_size=hidden_size,
                           batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):                    # x: (B, T, F)
        h, _ = self.rnn(x)                   # h: (B, T, H)
        y = self.out(h[:, -1])               # last time-step
        return y.squeeze(-1)                 # (B,)


def _rolling_windows(
    series: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build auto-regressive (lag-only) windows.
    Returns X (N, window) and y (N,)
    """
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def _train_lstm(df: pd.DataFrame, params: Dict[str, Any] | None = None):
    p = params or {}
    # window_size  = p.get("window_size", 30)
    window_size_req = p.get("window_size", 30)
    hidden_size  = p.get("hidden_size", 64)
    epochs       = p.get("epochs", 50)
    batch_size   = p.get("batch_size", 64)
    lr           = p.get("learning_rate", 3e-3)
    grad_clip    = p.get("grad_clip", 1.0)       # 0 → no clipping

    series = df["y"].values.astype("float32")
    window_size  = _safe_window_size(len(series), window_size_req)

    # z-score -- keep parameters for replay
    mean_, std_ = series.mean(), series.std() + 1e-6
    series_n    = (series - mean_) / std_

    X, y = _rolling_windows(series_n, window_size)   # (N, T), (N,)
    X    = X[..., None]                               # add feature dim -> (N, T, 1)

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mdl = SimpleLSTM(n_features=1, hidden_size=hidden_size).to(device)
    mdl.window_size   = window_size            # ←  used by evaluate.py
    mdl.feature_cols  = ["y"]                  # ←  evaluate.py expects this
    mdl.mean_, mdl.std_ = mean_, std_

    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # ---------------- training loop -----------------
    t0 = time.time()
    mdl.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(mdl(xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), grad_clip)
            opt.step()
    train_sec = time.time() - t0
    mdl.eval()
    
    mdl.history_tail = df.tail(window_size).copy()
    return mdl, train_sec


# ──────────────────────────────────────────────────────────────
#                G R U    (PyTorch)
# ──────────────────────────────────────────────────────────────
class SimpleGRU(nn.Module):
    """
    Minimal 1-layer GRU that outputs one value per window.
    Forward: x (B, T, F) → ŷ (B,)
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.GRU(input_size=n_features,
                          hidden_size=hidden_size,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):                    # x: (B, T, F)
        h, _ = self.rnn(x)                   # h: (B, T, H)
        y = self.out(h[:, -1])               # last step
        return y.squeeze(-1)                 # (B,)


def _train_gru(df: pd.DataFrame, params: Dict[str, Any] | None = None):
    p = params or {}
    window_size_req = p.get("window_size", 30)
    hidden_size  = p.get("hidden_size", 32)
    epochs       = p.get("epochs", 30)
    batch_size   = p.get("batch_size", 64)
    lr           = p.get("learning_rate", 3e-3)
    grad_clip    = p.get("grad_clip", 1.0)

    series = df["y"].values.astype("float32")
    window_size  = _safe_window_size(len(series), window_size_req)

    # z-score normalisation (store stats on the model for replay)
    mean_, std_ = series.mean(), series.std() + 1e-6
    series_n    = (series - mean_) / std_

    X, y = _rolling_windows(series_n, window_size)   # (N, T), (N,)
    X    = X[..., None]                              # add feature dim -> (N, T, 1)

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mdl = SimpleGRU(n_features=1, hidden_size=hidden_size).to(device)
    mdl.window_size   = window_size
    mdl.feature_cols  = ["y"]          # evaluate.py uses this
    mdl.mean_, mdl.std_ = mean_, std_

    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    t0 = time.time()
    mdl.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(mdl(xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), grad_clip)
            opt.step()
    train_sec = time.time() - t0
    mdl.eval()
    mdl.history_tail = df.tail(window_size).copy()
    return mdl, train_sec


# ──────────────────────────────────────────────────────────────
#                M I N I  T R A N S F O R M E R  
# ──────────────────────────────────────────────────────────────
def _train_transformer(df: pd.DataFrame, params: Dict[str, Any] | None = None):
    """
    Bare-bones encoder-only Transformer with positional encoding.
    Performs no hyper-parameter magic; good enough to tick the
    'model-agnostic' box in the paper.  Trains on lag-only windows.
    """
    p = params or {}
    window_req  = p.get("window_size", 30)
    d_model     = p.get("d_model", 32)
    nhead       = p.get("n_heads", 4)
    epochs      = p.get("epochs", 50)
    batch_size  = p.get("batch_size", 64)
    lr          = p.get("learning_rate", 3e-3)

    import torch, math
    from torch import nn
    series = df["y"].values.astype("float32")
    window      = _safe_window_size(len(series), window_req)

    # build windows
    X, y = _rolling_windows(series, window)
    X    = X[..., None]                        # (N, T, 1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) *
                            (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TinyTF(nn.Module):
        def __init__(self):
            super().__init__()
            self.input  = nn.Linear(1, d_model)
            self.pe     = PositionalEncoding(d_model)
            encoder     = nn.TransformerEncoderLayer(d_model, nhead,
                                                     dim_feedforward=4*d_model,
                                                     batch_first=True)
            self.enc    = nn.TransformerEncoder(encoder, num_layers=2)
            self.head   = nn.Linear(d_model, 1)
        def forward(self, x):
            z = self.input(x)
            z = self.pe(z)
            z = self.enc(z)
            return self.head(z[:, -1]).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TinyTF().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.L1Loss()

    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                    batch_size=batch_size, shuffle=True)

    t0 = time.time()
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad();  loss = crit(net(xb), yb);  loss.backward();  opt.step()
    net.eval()
    net.window_size  = window            # add these four lines
    net.feature_cols = ["y"]
    net.mean_ = np.mean(series)
    net.std_  = np.std(series) + 1e-6
    net.history_tail = df.tail(window).copy()
    return net, time.time() - t0


# ──────────────────────────────────────────────────────────────
# registry + public façade
# ──────────────────────────────────────────────────────────────
TRAINERS = {
    "lightgbm": _train_lightgbm,
    "prophet":  _train_prophet,
    "arima":    _train_arima,
    "lstm":     _train_lstm,
    "gru": _train_gru,
    "transformer": _train_transformer,
}


def train(
    df: pd.DataFrame,
    kind: str,
    params: Dict[str, Any] | None = None,
) -> Tuple[Any, float]:
    """
    Generic entry point used by `pipeline.py`.

    Parameters
    ----------
    df     : DataFrame with columns ['Week', 'y', … features …]
    kind   : one of TRAINERS.keys()
    params : hyper-parameters for the chosen model (may include
             super-set; unused keys are silently ignored)
    """
    if kind not in TRAINERS:
        raise ValueError(f"Unknown model kind: {kind}. "
                         f"Available: {list(TRAINERS)}")
    return TRAINERS[kind](df, params)





