#!/usr/bin/env python3
"""
Paired significance tests:  RES-λ  vs.  {FIXED-k, RANDOM-p, CARA}

Rules
-----
• If the two vectors share ≥ 2 seeds → Wilcoxon signed-rank (paired)
• Else (but each side has ≥ 2 runs) → Mann-Whitney U (independent)

Outputs one row per (dataset, baseline, λ_res) × statistic
→ tables/wilcoxon_grid.csv
"""
from __future__ import annotations
import os, argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from  scipy.stats import wilcoxon, mannwhitneyu, bootstrap, levene


# ── paths ──────────────────────────────────────────────────────────────
ROOT_TABLE = Path(os.getenv("TABLES_DIR",  "tables"))
ROOT_FIG   = Path(os.getenv("FIGURES_DIR", "figures"))
ROOT_TABLE.mkdir(parents=True, exist_ok=True)

DEC_CSV = ROOT_FIG / "all_decisions.csv"
OUT_CSV = ROOT_TABLE / "wilcoxon_grid.csv"

# ── constants ──────────────────────────────────────────────────────────
DATASETS = ["nyc_taxi_clean", "m5_sales_clean",
            "wiki_weekly",    "electricity_weekly"]

BASELINES = [("fixed",  dict(k=4)),
             ("fixed",  dict(k=8)),
             ("fixed",  dict(k=12)),
             ("random", dict(p=0.15)),
             ("random", dict(p=0.05)),
             ("cara",   dict())]

METRIC = "mae_after"        # per-week column → we will average per run
N_BOOT = 2_000
RNG    = np.random.default_rng(2024)

# ── helpers ────────────────────────────────────────────────────────────
def _filter_param(df: pd.DataFrame, param: Dict[str, Any]) -> pd.DataFrame:
    if "lambda" in param:
        df = df[np.isclose(df["lambda"], param["lambda"], atol=1e-9)]
    if "k" in param and param["k"] is not None:
        df = df[df["k"] == param["k"]]
    if "p" in param and param["p"] is not None:
        df = df[np.isclose(df["p"], param["p"], atol=1e-9)]
    return df


def _run_means(df_all: pd.DataFrame,
               dataset: str,
               policy:  str,
               extra:   Dict[str, Any],
               metric:  str) -> pd.Series:
    """Return Series indexed by seed with mean(metric) per seed."""
    df = df_all.loc[
        (df_all["dataset"] == dataset) &
        (df_all["policy_kind"].str.lower() == policy.lower())   # ❷
    ]
    df = _filter_param(df, extra)
    return (df.groupby("seed")[metric]
              .mean()
              .sort_index())


def _paired_test(res_vec: pd.Series, base_vec: pd.Series,
                 paired: bool) -> Dict[str, Any]:
    if paired:
        common = res_vec.index.intersection(base_vec.index)
        res_p, base_p = res_vec.loc[common], base_vec.loc[common]
        delta = base_p.values - res_p.values           # + ⇒ RES better
        try:                                           # ❸
            w_stat, p_val = wilcoxon(delta, alternative="greater")
        except ValueError:                             # all zeros
            w_stat, p_val = 0.0, 1.0

        ci = bootstrap((delta,), np.mean,
                       n_resamples=N_BOOT, paired=True,
                       random_state=RNG).confidence_interval
        return dict(delta_mean=delta.mean(), ci_lo=ci.low, ci_hi=ci.high,
                    stat_name="W_stat", stat_val=w_stat, p_value=p_val,
                    n_pairs=len(common), n_res=len(res_vec),
                    n_base=len(base_vec))

    # unpaired
    u_stat, p_val = mannwhitneyu(res_vec, base_vec, alternative="less")
    delta = base_vec.mean() - res_vec.mean()
    boots = [RNG.choice(base_vec, len(base_vec), replace=True).mean()
             - RNG.choice(res_vec,  len(res_vec),  replace=True).mean()
             for _ in range(N_BOOT)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return dict(delta_mean=delta, ci_lo=ci_lo, ci_hi=ci_hi,
                stat_name="U_stat", stat_val=u_stat, p_value=p_val,
                n_pairs=0, n_res=len(res_vec), n_base=len(base_vec))



# ── core ───────────────────────────────────────────────────────────────
def run_all_tests(decisions_csv: Path = DEC_CSV,
                  out_csv: Path = OUT_CSV,
                  datasets: List[str] = DATASETS,
                  baselines = BASELINES,
                  min_seeds: int = 2) -> pd.DataFrame:

    # ❶  aggregate metric across the whole replay ------------------------
    dec_all = pd.read_csv(decisions_csv)

    # numeric cast
    for col in ("lambda", "k", "p", METRIC, "seed"):
        if col in dec_all.columns:
            dec_all[col] = pd.to_numeric(dec_all[col], errors="coerce")

    if "seed" not in dec_all.columns:
        raise ValueError("CSV must contain a 'seed' column.")

    # mean(metric) per run_id (=> per seed/config)
    run_mean = (dec_all.groupby("run_id", as_index=False)[METRIC]
                       .mean()
                       .rename(columns={METRIC: f"mean_{METRIC}"}))

    # static meta columns (one per run)
    static_cols = dec_all.drop_duplicates("run_id")[[
        "run_id", "dataset", "policy_kind", "lambda", "k", "p", "seed"
    ]]

    dec = static_cols.merge(run_mean, on="run_id")
    METRIC_MEAN = f"mean_{METRIC}"

    # λ values present in RES runs
    lambda_vals = (dec.loc[dec["policy_kind"].str.lower() == "res", "lambda"]
                     .dropna().unique().tolist())
    if not lambda_vals:
        raise RuntimeError("No λ values found for policy_kind == 'res'.")

    rows: List[Dict[str, Any]] = []
    for lam in sorted(lambda_vals):
        res_param = {"lambda": lam}

        for ds in datasets:
            res_vec = _run_means(dec, ds, "res", res_param, METRIC_MEAN)

            for base_kind, base_param in baselines:
                base_vec = _run_means(dec, ds, base_kind, base_param,
                                      METRIC_MEAN)

                if len(res_vec) < min_seeds or len(base_vec) < min_seeds:
                    continue

                paired = len(res_vec.index.intersection(base_vec.index)) >= 2
                stats  = _paired_test(res_vec, base_vec, paired)

                label = (f"FIXED-k{base_param.get('k')}"  if base_kind=="fixed" else
                         f"RANDOM-p{base_param.get('p')}" if base_kind=="random" else
                         base_kind.upper())

                rows.append(dict(dataset=ds, baseline=label,
                                 lambda_res=lam,
                                 mean_res=res_vec.mean(),
                                 mean_base=base_vec.mean(),
                                 **stats))

    tbl = (pd.DataFrame(rows)
             .sort_values(["dataset", "baseline", "lambda_res"]))

    num_cols = ["mean_res", "mean_base", "stat_val", "p_value",
                "delta_mean", "ci_lo", "ci_hi"]
    tbl[num_cols] = tbl[num_cols].round(4)
    tbl.to_csv(out_csv, index=False)
    print(f"✔  Saved {len(tbl)} rows → {out_csv}")
    return tbl

# ──────────────────────────────────────────────────────────────
#  EXTRA: Levene homogeneity-of-variance test across architectures
#         (RES runs only, mid-range λ 1e-2 … 0.7)
#         → tables/levene_arch_variance.csv
# ──────────────────────────────────────────────────────────────
def levene_arch_variance(decisions_csv: Path = DEC_CSV,
                         out_csv: Path = ROOT_TABLE / "levene_arch_variance.csv"):
    df = pd.read_csv(decisions_csv)

    # keep only RES policy rows and numeric λ in the mid-range
    m = df["policy_kind"].str.lower() == "res"
    df = df[m].copy()
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    df = df[(df["lambda"] >= 1e-2) & (df["lambda"] <= 0.7)]

    if df.empty:
        print("⚠️  No RES rows for Levene test; skipping.")
        return

    # per-run mean MAE reduction (%)
    df["gain_pct"] = 100 * (df["mae_before"] - df["mae_after"]) / df["mae_before"]

    # groups by architecture
    groups = [df.loc[df["model_kind"] == m, "gain_pct"].values
              for m in sorted(df["model_kind"].unique())]

    stat, pval = levene(*groups, center="mean")

    pd.DataFrame({"statistic":[stat], "p_value":[pval]}).round(5)\
      .to_csv(out_csv, index=False)
    print(f"✓ Levene test saved → {out_csv}   (p = {pval:.3f})")


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true",
                    help="write to sanity_check/... and relax min_seeds")
    ap.add_argument("--datasets", nargs="+", metavar="DS",
                    help="datasets to include")
    args = ap.parse_args()

    if args.minimal:
        ROOT_TABLE_S = Path("sanity_check/tables"); ROOT_TABLE_S.mkdir(parents=True, exist_ok=True)
        OUT_CSV_S = ROOT_TABLE_S/"wilcoxon_grid.csv"
        DEC_CSV_S = Path("sanity_check/figures/all_decisions.csv")
        run_all_tests(decisions_csv=DEC_CSV_S, out_csv=OUT_CSV_S,
                      datasets=args.datasets or ["m5_sales_clean"],
                      min_seeds=1)
    else:
        run_all_tests(datasets=args.datasets or DATASETS)
    



if __name__ == "__main__":
    main()
