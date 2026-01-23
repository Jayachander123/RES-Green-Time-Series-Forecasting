# src/utils/stats.py
"""
Small statistical-test helpers used by the analysis notebooks.
Currently only Wilcoxon but can be extended easily.
"""

from __future__ import annotations
import numpy as np
import pandas   as pd
from scipy.stats import wilcoxon


def wilcoxon_before_after(df: pd.DataFrame,
                          before_col: str,
                          after_col : str) -> dict[str, float]:
    """
    Non-parametric Wilcoxon signed-rank test.
    Returns dict with p-value and effect size (r).
    """
    if before_col not in df.columns or after_col not in df.columns:
        raise KeyError("missing columns for Wilcoxon test")

    stat, p = wilcoxon(df[before_col], df[after_col])
    # effect size r = Z / sqrt(N).  For Wilcoxon Z≈stat when N large.
    n   = len(df)
    r   = stat / np.sqrt(n) if n else float("nan")
    return {"p": float(p), "effect_size_r": float(r)}
