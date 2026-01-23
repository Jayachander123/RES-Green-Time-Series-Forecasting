#!/usr/bin/env python3
"""
make_res_plots.py
────────────────────────────────────────────────────────
Generates every figure used in the paper:

  01 Pareto cost-accuracy plot
  02 λ-robustness curves
  03 Architecture box-plot
  04 Dataset×model heat-map
  05 Survival curve
  06 Microscope MAE time series
  07 Stacked CPU/CO₂ bar plot
  A1 λ×dataset utility heat-map
  A2 Auto-λ   vs   best fixed-λ  bar plot  (+ CSV)

Input files (relative to project root)
  • figures/all_mlflow_runs.csv   – one row per MLflow run
  • figures/all_decisions.csv     – week-level replay log (concatenated)
Both are produced by scripts/export_decisions.py
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from typing import Optional, Callable, List
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from lifelines import KaplanMeierFitter   # fig 5
from scipy.stats import bootstrap
import argparse
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sns.set_style("whitegrid")


FIG_DIR = Path(os.getenv("FIGURES_DIR", "figures"))      # NEW
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV      = os.getenv("RUNS_CSV",      FIG_DIR/"all_mlflow_runs.csv")
DECISIONS_CSV = os.getenv("DECISIONS_CSV", FIG_DIR/"all_decisions.csv")

# ======================================================================
# helpers
# ======================================================================
def _load_runs(csv: str = RUNS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # cast numeric-looking param / metric columns once
    if "dataset" not in df.columns and "param.dataset" in df.columns:
        df["dataset"] = df["param.dataset"] 
    for c in df.columns:
        if c.startswith(("metric.", "param.")):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df



def _load_decisions(csv: str = DECISIONS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # best effort numeric coercion
    for c in ["lambda", "k", "p", "tau", "benefit", "cost", "retrained"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _finish_and_save(fig, out_png, top_pad=0.12):
    """
    Add enough space for suptitle + legend and save the figure.
    """
    # reserve some vertical margin (0–1 coordinates)
    fig.subplots_adjust(top=1 - top_pad)
    # tight_layout for the subplot region only
    fig.tight_layout()
    # now write everything, including the margin, to disk
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("saved →", out_png)


# ======================================================================
# 01  Pareto cost-accuracy trade-off
# ======================================================================
def plot_promotion_deploy_costs(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/fig00_promotions_and_deploy_costs.pdf"):



    # optional: flatten "param.*" columns
    dec = dec.rename(columns=lambda c: c.split('.', 1)[1]
                                 if c.startswith("param.") else c)

    # numeric coercion
    num_cols = ["retrained", "lambda", "k", "p", "tau", "cost"]
    for c in num_cols:
        if c in dec.columns:
            dec[c] = pd.to_numeric(dec[c], errors="coerce")

    dec["retrained"] = dec["retrained"].fillna(0).astype(int)
    dec["cost"]      = dec["cost"].fillna(0.0)

    # ────────────────────────────────
    # 1.  Pick the policy parameter
    # ────────────────────────────────
    pol = dec["policy_kind"].str.lower()

    def _ph(row):
        return f"({row.get('delta')},{row.get('lambda_ph')},{row.get('alpha')})"

    dec["param_value"] = np.select(
        [pol=="res", pol=="fixed", pol=="random", pol=="cara", pol=="ph"],
        [dec["lambda"], dec["k"],  dec["p"],
         dec["tau"].fillna(dec["lambda"]), dec.apply(_ph, axis=1)],
        default=np.nan)

    # ────────────────────────────────
    # 2.  Aggregate
    # ────────────────────────────────
    g = dec.groupby(["policy_kind","param_value"], dropna=False)

    summary = (
        g.agg(n_runs           = ("run_id", "nunique"),
              n_weeks          = ("run_id", "size"),
              total_retrains   = ("retrained", "sum"),
              total_cost_sec   = ("cost", "sum"))
          .reset_index()
    )

    summary["no_retrain_weeks"]   = summary["n_weeks"] - summary["total_retrains"]
    # summary["mean_cost_sec"]      = (summary["total_cost_sec"]
    #                                  / summary["total_retrains"].replace({0:np.nan}))
    summary["mean_cost_per_week"] = summary["total_cost_sec"] / summary["n_weeks"]

    # ────────────────────────────────
    # 3.  Order & save
    # ────────────────────────────────
    summary = (summary
               .sort_values(["policy_kind","param_value"],
                            key=lambda s: (s.str.lower() if s.dtype=="object" else s))
               .reset_index(drop=True))

    Path("tables").mkdir(exist_ok=True)
    
    summary.to_csv("tables/retrain_workload_breakdown_with_cost.csv", index=False)
    print("✓ saved → tables/retrain_workload_breakdown_with_cost.csv")

    


    USD_PER_PROMO = 40                  # labour cost per promotion
    summary_df   = summary.copy()       # ← your DataFrame

    # ------------------------------------------------------------------
    # Prep
    # ------------------------------------------------------------------
    summary_df["param_num"] = pd.to_numeric(summary_df["param_value"],
                                            errors="coerce")
    keep = summary_df["policy_kind"].isin(["always", "res", "cara"])
    plot_df = summary_df[keep].copy()

    # labels -----------------------------------------------------------
    def _lbl(r):
        if r["policy_kind"] == "always": return "ALWAYS"
        if r["policy_kind"] == "res":    return f"λ={r['param_value']}"
        return f"τ={r['param_value']}"
    plot_df["label"] = plot_df.apply(_lbl, axis=1)

    plot_df["order_key"] = (
            plot_df["policy_kind"].map({"always": 0, "res": 1, "cara": 2})*1_000
            + plot_df["param_num"].fillna(0)
    )
    plot_df = plot_df.sort_values("order_key")

    # compute deployment cost ------------------------------------------
    plot_df["deploy_usd"] = plot_df["total_retrains"] * USD_PER_PROMO

    # ------------------------------------------------------------------
    # Compact format helpers
    # ------------------------------------------------------------------
    def fmt_k(x):
        """12000 -> '12k', 1256000 -> '1.3M'"""
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x/1_000:.0f}k"
        return f"{x:.0f}"

    def fmt_int_k(x):
        if x >= 1_000:
            return f"{x/1_000:.0f}k"
        return f"{int(x)}"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    palette = {"always":"crimson","res":"royalblue","cara":"forestgreen"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 4), sharex=True)

    # left  – promotions ------------------------------------------------
    sns.barplot(data=plot_df, x="label", y="total_retrains",
                hue="policy_kind", dodge=False, palette=palette, ax=axL)

    for p in axL.patches:
        h = p.get_height()
        # axL.text(p.get_x()+p.get_width()/2, h*1.01,
        #          fmt_int_k(h), ha="center", va="bottom", fontsize=9)
        axL.text(p.get_x() + p.get_width()/2,
                h + 0.01*h,            # slight offset
                f"{h:.0f}",
                ha="center", va="bottom", fontsize=9)

    axL.set_ylabel("Promotions to production")
    axL.set_xlabel("")
    axL.set_title("How often did we redeploy?")
    axL.legend(title="Policy", frameon=False)
    plt.setp(axL.get_xticklabels(), rotation=45, ha="right")

    # right – cost ------------------------------------------------------
    sns.barplot(data=plot_df, x="label", y="deploy_usd",
                hue="policy_kind", dodge=False, palette=palette, ax=axR)

    for p in axR.patches:
        h = p.get_height()
        axR.text(p.get_x()+p.get_width()/3, h*1.01,
                 fmt_k(h), ha="center", va="bottom", fontsize=9)

    axR.set_ylabel("Cumulative labor cost")
    axR.set_xlabel("")
    axR.set_title(f"Deployment Cost in $")

    # y-axis ticks also in k-format
    axR.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt_k(y)))

    plt.setp(axR.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Deployment workload and cost – ALWAYS vs. RES(λ) vs. CARA(τ)",
                 y=0.95, fontsize=14)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print("✓ figure written to figures/promotions_and_deploy_costs.pdf")

     
    
    
    
# ======================================================================
# 01  Pareto cost-accuracy trade-off
# ======================================================================
def plot_pareto_tradeoff_single(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/fig01_pareto_tradeoff.pdf"):

    # derive replay length from week log
    week_cnt = (dec.groupby("run_id")["week"]
                  .count()
                  .rename("replay_weeks")
                  .reset_index())
    runs = runs.merge(week_cnt, on="run_id", how="left")
    runs["param.replay_weeks"] = pd.to_numeric(runs["param.replay_weeks"],
                                               errors="coerce")
    mask = runs["param.replay_weeks"].isna() | (runs["param.replay_weeks"] <= 0)
    runs.loc[mask, "param.replay_weeks"] = runs.loc[mask, "replay_weeks"]

    # compute two axes
    runs["mae_before"] = runs["metric.mean_mae_before"]
    runs["mae_after"]  = runs["metric.mean_mae_after"]
    runs["mae_pct"]    = 100 * runs["mae_after"] / runs["mae_before"]

    runs["cost_sec"] = (pd.to_numeric(runs["metric.total_retrain_sec"],
                                      errors="coerce") /
                        runs["param.replay_weeks"])

    runs = runs.replace([np.inf, -np.inf], np.nan).dropna(subset=["mae_pct",
                                                                  "cost_sec"])
    runs = runs[runs["cost_sec"] > 0]
    
    KEEP_LAM  = [0.01, 0.10, 0.30, 0.85, 1.00]
    KEEP_TAU  = [0.05, 0.30, 0.90]
    KEEP_K    = [4, 12]
    KEEP_P    = [0.05, 0.30]

    pk = runs["param.policy_kind"].str.lower()
    mask = (
        (pk != "res"   ) | runs["param.lambda"].isin(KEEP_LAM)
    ) & (
        (pk != "cara"  ) | runs["param.tau"   ].isin(KEEP_TAU)
    ) & (
        (pk != "fixed" ) | runs["param.k"     ].isin(KEEP_K)
    ) & (
        (pk != "random") | runs["param.p"     ].isin(KEEP_P)
    )
    runs = runs[mask]



    # policy label
    base = runs["param.policy_kind"].str.lower()
    runs["policy"] = np.select(
        [
            base == "always",
            base == "never",
            base == "res",
            base == "fixed",
            base == "ph",
            base == "random",
            base == "cara"
        ],
        ["ALWAYS", "NEVER", "RES", "FIXED", "PH", "RANDOM", "CARA"],
        default="OTHER",
    )

    agg = (runs
           .groupby(["dataset", "policy",
                     "param.model_kind",
                     "param.lambda", "param.tau", "param.k", "param.p"],
                    dropna=False)
           .agg(mae_mean=("mae_pct",  "mean"),
                cost_mean=("cost_sec","mean"))
           .reset_index())

    # ----- label helper --------------------------------------------
    def _lbl(row):
        if row["policy"] == "RES":
            return f"λ={row['param.lambda']}"
        if row["policy"] == "CARA":
            return f"τ={row['param.tau']}"
        if row["policy"] == "FIXED":
            return f"k={int(row['param.k'])}"
        if row["policy"] == "RANDOM":
            return f"p={row['param.p']}"
        return row["policy"]
    agg["label"] = agg.apply(_lbl, axis=1)


    datasets = ["electricity_weekly", "m5_sales_clean",
                "nyc_taxi_clean", "wiki_weekly"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        sub = agg[agg["dataset"] == ds]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.scatterplot(data=sub, x="mae_mean", y="cost_mean",
                        hue="label", style="param.model_kind",
                        s=60, ax=ax, legend=(i == 0))
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_title(ds.replace("_", " "))
        ax.set_xlabel("MAE [%] ↓")
        if i == 0:
            ax.set_ylabel("Retrain cost per week [s] ↓")
        else:
            ax.set_ylabel("")

    if axes[0].legend_ is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend_.remove()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.03),
                   ncol=len(labels), frameon=False)

    fig.suptitle("Per-Dataset Cost–Accuracy Trade-off", y=1.08)
    fig.tight_layout()
    # fig.savefig(out_png, dpi=300)
    _finish_and_save(fig, out_png, top_pad=0.20)
    agg.to_csv("tables/pareto_plot_data.csv", index=False)
    print("saved →", out_png)

# ================================================================
# Appendix – full RES λ grid  +  CARA τ grid
# • rows  = datasets   (4)
# • cols  = model kind (4)
# • colour encodes λ  or  τ   via separate colour-bars
# • marker encodes policy family (square = RES, triangle = CARA)
# ================================================================
def appendix_full_grid_pareto(runs: pd.DataFrame,
                       dec : pd.DataFrame,
                       AGG_FUN : str = "mean",
                       OUT_CSV: str = "tables/all_decisions_pareto_table_cara_res_full_grid.csv" ):


    # convert obvious numeric columns
    num_cols = ["mae_before", "mae_after", "benefit_raw", "cost",
                "lambda", "tau", "k", "p"]
    for c in num_cols:
        if c in dec.columns:
            dec[c] = pd.to_numeric(dec[c], errors="coerce")

    # ===============================================================
    # 2.  PER-RUN summary  (collapse the weeks)
    # ===============================================================
    per_run = (
        dec.groupby("run_id")
            .agg(
                dataset        = ("dataset",      "first"),
                model          = ("model_kind",   "first"),
                policy         = ("policy_kind",  "first"),
                lambda_val     = ("lambda",       "first"),
                tau_val        = ("tau",          "first"),
                mae_before     = ("mae_before",   "mean"),
                mae_after      = ("mae_after",    "mean"),
                total_retrain_sec = ("cost",      "sum"),
                replay_weeks      = ("week",      "count"),
                seed           = ("seed",         "first"),
            )
            .reset_index()
    )

    # derive accuracy [% of baseline]  and  cost / week
    per_run["mae_pct"]      = 100 * per_run["mae_after"] / per_run["mae_before"]
    per_run["cost_per_wk"]  = per_run["total_retrain_sec"] / per_run["replay_weeks"]

    # ===============================================================
    # 3.  SPLIT into RES  and  CARA
    # ===============================================================
    mask_res  = per_run["policy"].str.lower() == "res"
    mask_cara = per_run["policy"].str.lower() == "cara"

    # ===============================================================
    # 4.  AGGREGATE across seeds          ─── helper
    # ===============================================================
    def pareto_from_runs(df, thr_col, new_name):
        """
        df       : per-run DataFrame (already has mae_pct, cost_per_wk)
        thr_col  : "lambda_val" or "tau_val"
        new_name : column name in the output ("lambda" or "tau")
        """
        tbl = (
            df.groupby(["dataset", "model", thr_col])
              .agg(mae_pct        = ("mae_pct",      AGG_FUN),
                   cost_s_per_wk  = ("cost_per_wk",  AGG_FUN),
                   n_seeds        = ("run_id",       "nunique"),
                   replay_weeks   = ("replay_weeks", "first"))  # same for all seeds
              .reset_index()
              .rename(columns={thr_col: new_name})
              .sort_values(["dataset", "model", new_name])
              .round({"mae_pct": 5, "cost_s_per_wk": 6})
        )
        return tbl

    tbl_res  = pareto_from_runs(per_run[mask_res],  "lambda_val", "lambda")
    tbl_cara = pareto_from_runs(per_run[mask_cara], "tau_val",    "tau")


    # ===============================================================
    # 5.  MERGE and export to one CSV
    # ===============================================================
    summary_df = pd.concat([tbl_res.assign(policy="res"),
                            tbl_cara.assign(policy="cara")],
                           ignore_index=True)

    cols_keep = ["dataset", "model", "policy",
             "lambda", "tau", "mae_pct", "cost_s_per_wk"]

    summary_df = summary_df[cols_keep]

    # optional: round for print-friendliness
    summary_df["mae_pct"]       = summary_df["mae_pct"].round(3)
    summary_df["cost_s_per_wk"] = summary_df["cost_s_per_wk"].round(6)

    if OUT_CSV:
        summary_df.to_csv(OUT_CSV, index=False)
        print(f"\nTable saved to {OUT_CSV}")

    
    
# ------------------------------------------------------------------
# helper: draw one 4-panel Pareto plot for exactly TWO policies
# ------------------------------------------------------------------
def _pareto_two_policies(df: pd.DataFrame,
                         pol_a: str,
                         pol_b: str,
                         name_a: str = None,
                         name_b: str = None,
                         title  : str = "",
                         out_png: str = "figures/tmp.pdf"):

    PALETTE = {pol_a:"#2171b5", pol_b:"#d73027"}
    SHAPE   = {pol_a:"s",       pol_b:"^"}          # square / triangle

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    DATASETS = ["electricity_weekly", "m5_sales_clean",
                "nyc_taxi_clean",    "wiki_weekly"]

    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        if sub.empty:
            ax.set_visible(False); continue

        for pol in [pol_a, pol_b]:
            dpol = sub[sub["policy"] == pol]
            if dpol.empty: continue
            sns.scatterplot(x="mae_pct", y="cost_sec",
                            data=dpol,
                            marker=SHAPE[pol],
                            s=60,
                            color=PALETTE[pol],
                            edgecolor="white",
                            linewidth=0.5,
                            ax=ax,
                            legend=False)
        ax.set_yscale("log")
        ax.set_title(ds.replace("_"," "))
        ax.set_xlabel("MAE [%] ↓")
        ax.set_ylabel("Cost / wk [s] ↓" if ds=="electricity_weekly" else "")

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=SHAPE[p], color=PALETTE[p],
                      markersize=7, linestyle='', label=l)
               for p,l in zip([pol_a, pol_b],
                              [name_a or pol_a, name_b or pol_b])]
    fig.legend(handles=handles, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), frameon=False)

    fig.suptitle(title, y=1.12, fontsize=12)
    fig.tight_layout()
    Path(out_png).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("✓ saved →", out_png)

    # ================================================================
# helper: RES  vs  CARA, with arrows + auto-zoom
# ================================================================
def _res_vs_cara_zoom(df: pd.DataFrame,
                      title  : str = "RES vs CARA",
                      out_png: str = "figures/fig01_res_vs_cara_zoom.pdf"):

    PAL = {"RES":"#1f77b4", "CARA":"#d62728"}        # blue / red
    SHP = {"RES":"s",        "CARA":"^"}             # square / triangle
    DATASETS = ["electricity_weekly", "m5_sales_clean",
                "nyc_taxi_clean",    "wiki_weekly"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        d = df[df["dataset"] == ds]
        if d.empty:
            ax.set_visible(False); continue

        # one point per (model, policy)
        for mdl, grp in d.groupby("param.model_kind"):
            res  = grp[grp["policy"] == "RES" ]
            cara = grp[grp["policy"] == "CARA"]
            if res.empty or cara.empty: continue    # sanity

            xr, yr = res["mae_pct"].values[0],  res["cost_sec"].values[0]
            xc, yc = cara["mae_pct"].values[0], cara["cost_sec"].values[0]

            # grey arrow  CARA ➜ RES
            ax.arrow(xc, yc, xr-xc, yr-yc,
                     length_includes_head=True,
                     head_width=0.015*yc, head_length=0.012*xr,
                     lw=0.7, color="grey", alpha=0.7, zorder=1)

        # scatter the two policies on top of the arrows
        for pol in ["CARA","RES"]:           # draw CARA first, RES on top
            sns.scatterplot(data=d[d["policy"]==pol],
                            x="mae_pct", y="cost_sec",
                            marker=SHP[pol], s=75,
                            color=PAL[pol], edgecolor="white",
                            linewidth=0.6, ax=ax, legend=False, zorder=2)

        # auto-zoom  (+ 5 %)
        sub = d[d["policy"].isin(["RES","CARA"])]
        x_min, x_max = sub["mae_pct"].min(),  sub["mae_pct"].max()
        y_min, y_max = sub["cost_sec"].min(), sub["cost_sec"].max()
        ax.set_xlim(x_min*0.97, x_max*1.03)
        ax.set_ylim(y_min*0.8,  y_max*1.05)

        ax.set_yscale("log")
        ax.set_title(ds.replace("_"," "))
        ax.set_xlabel("MAE  [% of baseline] ↓")
        if ds == "electricity_weekly":
            ax.set_ylabel("Cost per week  [s] ↓")
        else:
            ax.set_ylabel("")

        # secondary y-axis: cost ratio  RES / CARA
        def _ratio(y): return y / yc        # yc is last loop value
        def _inv(r):  return r * yc
        sec = ax.secondary_yaxis('right', functions=(_ratio, _inv))
        sec.set_ylabel("Cost ratio  RES / CARA", fontsize=8)
        sec.tick_params(labelsize=7)
        sec.set_ylim(ax.get_ylim()[0]/yc, ax.get_ylim()[1]/yc)

    # legend (2 dots)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=SHP[p], color=PAL[p],
                      markersize=7, linestyle='', label=p)
                      for p in ["RES","CARA"]]
    fig.legend(handles, ["RES","CARA"], ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.05),
               frameon=False)

    fig.suptitle(title, y=1.13, fontsize=13)
    fig.tight_layout()
    Path(out_png).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("✓ saved →", out_png)



# ------------------------------------------------------------------
# 01  Pairwise Pareto plots centred on RES
# ------------------------------------------------------------------
def plot_pareto_tradeoff(runs: pd.DataFrame,
                         dec : pd.DataFrame,
                         out_dir: str = "figures"):

    # ── prepare common numeric columns ────────────────────────────
    wk_cnt = dec.groupby("run_id")["week"].count().rename("replay_weeks")
    runs = (runs.merge(wk_cnt, on="run_id", how="left")
                 .assign(replay_weeks=lambda d: d["param.replay_weeks"]
                                             .fillna(d["replay_weeks"]),
                         mae_pct   =lambda d:100*d["metric.mean_mae_after"]
                                             /d["metric.mean_mae_before"],
                         cost_sec  =lambda d:d["metric.total_retrain_sec"]
                                             /d["replay_weeks"])
                 .replace([np.inf,-np.inf],np.nan)
                 .dropna(subset=["mae_pct","cost_sec"]))

    # ── canonical policy label ────────────────────────────────────
    pk = runs["param.policy_kind"].str.lower()
    runs["policy"] = np.select(
        [pk=="always", pk=="never", pk=="res", pk=="fixed",
         pk=="ph", pk=="random", pk=="cara"],
        ["ALWAYS","NEVER","RES","FIXED","PH","RANDOM","CARA"], "OTHER")


    REP = {
        "RES"   : {"param.lambda": 0.3},   # λ = 0.30
        "CARA"  : {"param.tau"   : 0.3},   # τ = 0.30
        "FIXED" : {"param.k"     : 4   },   # k = 4
        "RANDOM": {"param.p"     : 0.15},   # p = 0.15
    }


    def _pick(d, pol):
        sub = d[d["policy"] == pol]
        if pol not in REP:                 # baseline policies
            return sub
        for col,val in REP[pol].items():
            sub = sub[np.isclose(sub[col], val, atol=1e-9)]
        return sub

    dsel = pd.concat([_pick(runs, p) for p in ["RES","CARA","ALWAYS","PH","FIXED"]])

    # ── output directory ------------------------------------------
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    # ── four pairwise plots ---------------------------------------
    # _pareto_two_policies(dsel, "RES", "CARA",
    #                      title="RES vs CARA",
    #                      out_png=out_dir/"fig01a_res_vs_cara.pdf")
    _res_vs_cara_zoom(dsel,
        title=f"RES (lambda = {REP['RES']['param.lambda']}) vs CARA (tau = {REP['CARA']['param.tau']}) – per dataset & model",
        out_png=out_dir/"fig01a_res_vs_cara_zoom.pdf")

    _pareto_two_policies(dsel, "RES", "ALWAYS",
                         title="RES vs ALWAYS",
                         out_png=out_dir/"fig01b_res_vs_always.pdf")

    _pareto_two_policies(dsel, "RES", "PH",
                         name_b="PERF-HORIZ",
                         title="RES vs Perf-Horizon",
                         out_png=out_dir/"fig01c_res_vs_ph.pdf")

    _pareto_two_policies(dsel, "RES", "FIXED",
                         name_b="FIXED k=4",
                         title="RES vs FIXED (k = 4)",
                         out_png=out_dir/"fig01d_res_vs_fixed.pdf")

    # save the filtered data for reproducibility
    dsel.to_csv("tables/pareto_pairwise_data.csv", index=False)
    print("✓ table  → tables/pareto_pairwise_data.csv")



# ===== Pareto delta table ==========================================
# =====================================================================
#  Pareto delta table – correct replay length -------------------------
# =====================================================================
def pareto_deltas(runs: pd.DataFrame,
                  dec:  pd.DataFrame,
                  out_csv: str = "tables/pareto_deltas.csv",
                  lambda_val: float = 0.3):

    runs = runs.copy()

    # 1) add replay_weeks exactly like the plotting code ---------------
    week_cnt = (dec.groupby("run_id")["week"]
                  .count()
                  .rename("replay_weeks")
                  .reset_index())
    runs = runs.merge(week_cnt, on="run_id", how="left")

    runs["param.replay_weeks"] = pd.to_numeric(
        runs["param.replay_weeks"], errors="coerce")
    mask = runs["param.replay_weeks"].isna() | (runs["param.replay_weeks"] <= 0)
    runs.loc[mask, "param.replay_weeks"] = runs.loc[mask, "replay_weeks"]

    # 2) derive the same cost / accuracy columns -----------------------
    runs["mae_pct"] = 100 * runs["metric.mean_mae_after"] \
                            / runs["metric.mean_mae_before"]
    runs["cost_sec_per_wk"] = (
        runs["metric.total_retrain_sec"] / runs["param.replay_weeks"]
    )

    # 3) helper to average across seeds --------------------------------
    def _agg(df):
        return df.groupby("run_id")[["mae_pct","cost_sec_per_wk"]].mean()

    rows = []
    for ds in ["electricity_weekly", "m5_sales_clean",
               "nyc_taxi_clean", "wiki_weekly"]:

        always = _agg(runs[(runs["dataset"] == ds) &
                           (runs["param.policy_kind"].str.lower() == "always")])

        res = _agg(runs[(runs["dataset"] == ds) &
                        (runs["param.policy_kind"].str.lower() == "res") &
                        np.isclose(runs["param.lambda"], lambda_val, atol=1e-9)])

        mae_delta   = always["mae_pct"].mean() - res["mae_pct"].mean()   # >0 ⇒ better
        cost_cut    = 1 - res["cost_sec_per_wk"].mean() / always["cost_sec_per_wk"].mean()

        rows.append(dict(dataset        = ds,
                         mae_improv_pct = round(mae_delta,   2),
                         cost_cut_pct   = round(100*cost_cut, 2)))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("✓ pareto deltas →", out_csv)



# ======================================================================
# 02  λ-robustness curves   (RES only)
# ======================================================================
def lambda_robustness(runs: pd.DataFrame,
                      out_png: str = "figures/fig02_lambda_robustness.pdf"):

    res = runs[runs["param.policy_kind"].str.lower() == "res"].copy()
    res["lambda"] = pd.to_numeric(res["param.lambda"], errors="coerce")
    res = res[res["lambda"] > 0]

    res["model"]      = res["param.model_kind"].str.upper()
    res["dataset"]    = res["param.dataset"]
    res["mae_before"] = res["metric.mean_mae_before"]
    res["mae_after"]  = res["metric.mean_mae_after"]
    res["mae_pct"]    = 100 * res["mae_after"] / res["mae_before"]

    agg = (res.groupby(["dataset", "model", "lambda"])
              .agg(mae_mean=("mae_pct", "mean"))
              .reset_index()
              .sort_values("lambda"))

    datasets = ["electricity_weekly", "m5_sales_clean",
                "nyc_taxi_clean", "wiki_weekly"]
    models = sorted(agg["model"].unique())
    palette = dict(zip(models, sns.color_palette("tab10", len(models))))
    

    # pick the λ values you really swept
    XTICKS       = [0.01, 0.03, 0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0]
    XTICKLABELS  = ["0.01", "0.03", "0.1", "0.3", "0.4",
                    "0.5", "0.6", "0.7", "0.85", "1"]


    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        sub = agg[agg["dataset"] == ds]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(data=sub, x="lambda", y="mae_mean",
                     hue="model", marker="o", palette=palette,
                     hue_order=models, ax=ax, legend=(i == 0))

        # reference line at “do nothing”
        ax.axhline(100, ls="--", color="grey", lw=0.8)
        
        LABEL_MAP = {0.01:"0.01", 0.03:"0.03", 0.10:"0.1",
             0.30:"0.3",                # 0.40 & 0.60 stay blank
             0.50:"0.5",
             0.70:"0.7", 0.85:"0.85", 1.0:"1"}
        
        # --- X-axis -----------------------------------------------------
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([LABEL_MAP.get(x, "") for x in XTICKS],
                           rotation=40, ha="right", fontsize=8)
        

        # NEW – make them readable
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.set_xlabel("λ")                        # shorter caption
        ax.set_xlim(min(XTICKS), max(XTICKS))


        # --- Y-axis ------------------------------------------------------
        ax.invert_yaxis()                         # lower MAE is better
        ax.set_title(ds.replace("_", " "))
        ax.set_xlabel("λ (retraining penalty)")

        if i == 0:
            ax.set_ylabel("MAE [% of baseline]")
        else:
            ax.set_ylabel("")

        for ax in axes: ax.axhline(100, ls="--", lw=0.6, color="grey", zorder=0)


    if axes[0].legend_ is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend_.remove()
        fig.legend(handles, labels, ncol=len(labels),
           loc="upper center", bbox_to_anchor=(0.5, 1.09),
           frameon=False)


    fig.suptitle("Figure 2 – λ-Robustness of RES Policy", y=1.01)
    fig.subplots_adjust(wspace=0.20)
    fig.tight_layout()
    # fig.savefig(out_png, dpi=300)
    _finish_and_save(fig, out_png, top_pad=0.25)
    print("saved →", out_png)


# ---------------------------------------------------------------------
# λ-robustness tables  (full grid  +  3-column short version)
# ---------------------------------------------------------------------
def table_lambda_robustness(runs: pd.DataFrame,
                            *_unused,                         # ← swallows “dec”
                            full_csv:  str = "tables/lambda_robustness.csv",
                            short_tex: str = "tables/lambda_short.csv",
                            keep_lam:  tuple = (0.10, 0.30, 0.85)):

    """
    1. Write the full λ-robustness grid (all λ that appear in *runs*)
       to *full_csv*  (wide CSV, one row per dataset-model).
    2. Write a three-column short table (λ in *keep_lam*) to *short_tex*
       in LaTeX format (booktabs-ready, no multicolumn).
    """

    # ── build base dataframe ─────────────────────────────────────────
    res = runs[runs["param.policy_kind"].str.lower() == "res"].copy()

    res["lambda"] = pd.to_numeric(res["param.lambda"], errors="coerce")
    res = res[res["lambda"] > 0]                         # drop λ = 0 / NaN

    res["mae_pct"] = (
        100 * res["metric.mean_mae_after"] / res["metric.mean_mae_before"]
    )

    # one mean per (dataset, model, λ)
    agg = (res.groupby(["param.dataset", "param.model_kind", "lambda"],
                       as_index=False)["mae_pct"]
             .mean())

    # ── full table (wide CSV) ────────────────────────────────────────
    full_tbl = (agg.pivot(index=["param.dataset", "param.model_kind"],
                          columns="lambda", values="mae_pct")
                   .sort_index(axis=1))                   # ascending λ

    Path(full_csv).parent.mkdir(parents=True, exist_ok=True)
    full_tbl.to_csv(full_csv)
    print("✓ full robustness table  →", full_csv)

    # ── short LaTeX table (three λs) ─────────────────────────────────
    short_tbl = (agg[agg["lambda"].isin(keep_lam)]
                 .pivot(index=["param.dataset", "param.model_kind"],
                        columns="lambda", values="mae_pct")
                 .reindex(columns=keep_lam))

    short_tbl.to_csv(short_tex)
    print("✓ short λ table         →", short_tex)





# ---------------------------------------------------------------------
# Auto-λ vs. best fixed-λ  (utility ratio)
# ---------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np


def auto_lambda_recovery(runs: pd.DataFrame,
                         *_unused,                      # swallows “dec”
                         out_csv: str = "tables/auto_lambda_recovery.csv"):
    """
    Compare each RES-auto-λ run (param.lambda == NaN) with the best
    fixed-λ run for the same (dataset, model).

    Output  CSV has columns:
        param.dataset , param.model_kind , auto_vs_oracle
        auto_vs_oracle = MAE_auto / (min MAE over fixed λ)
    A value of 1.00 means auto-λ exactly matches the oracle.
    """

    # ── keep only RES policy runs ────────────────────────────────────
    res = runs[runs["param.policy_kind"].str.lower() == "res"].copy()

    # ── construct the metric needed for comparison ───────────────────
    res["mae_pct"] = (
        100 * res["metric.mean_mae_after"] / res["metric.mean_mae_before"]
    )

    # ── split fixed-λ  vs  auto-λ ────────────────────────────────────
    is_auto  = res["param.lambda"].isna()
    fixed    = res[~is_auto]
    auto     = res[ is_auto]

    if auto.empty:
        print("⚠️  No auto-λ runs found; skipping export.")
        return

    # best oracle MAE for every dataset–model (min over λ)
    fixed_best = (
        fixed.groupby(["param.dataset", "param.model_kind"])["mae_pct"]
             .min()
             .rename("mae_oracle")
    )

    # mean MAE of the auto-λ runs (average over seeds)
    auto_mean = (
        auto.groupby(["param.dataset", "param.model_kind"])["mae_pct"]
            .mean()
            .rename("mae_auto")
    )

    # join & ratio
    cmp = (pd.concat([fixed_best, auto_mean], axis=1)
             .dropna()
             .assign(auto_vs_oracle=lambda df: df["mae_auto"] / df["mae_oracle"])
             .reset_index()
             .round({"auto_vs_oracle": 4}))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(out_csv, index=False)
    print("✓ auto-λ recovery →", out_csv)




def architecture_boxplot(runs: pd.DataFrame,
                         out_png: str = "figures/fig03_architecture_boxplot.pdf"):

    # ── 0. Load run table ─────────────────────────────────────────────────
    # run_df = pd.read_csv(flow_csv_in)
    run_df = runs.copy()
    
    def _to_num(col):
        try:
            return pd.to_numeric(col)
        except Exception:
            return col
    
    # numeric-cast every param / metric column
    for c in run_df.columns:
        if c.startswith(("metric.", "param.")):
            run_df[c] = _to_num(run_df[c])
    
    # ── 1. Select RES runs with “mid-range” λ  ────────────────────────────
    LOW_LAM, HIGH_LAM     = 1e-2, 0.7          # define your mid-range here
    
    res_mid = run_df[run_df["param.policy_kind"].str.lower() == "res"].copy()
    res_mid["lambda_val"] = pd.to_numeric(res_mid["param.lambda"], errors="coerce")
    
    mask = (res_mid["lambda_val"] >= LOW_LAM) & (res_mid["lambda_val"] <= HIGH_LAM)
    res_mid = res_mid[mask].copy()
    
    # ── 2. Compute per-run gain (relative error reduction) ───────────────
    res_mid["mae_before"] = res_mid["metric.mean_mae_before"]
    res_mid["mae_after"]  = res_mid["metric.mean_mae_after"]
    res_mid["gain_pct"]   = 100 * (res_mid["mae_before"] - res_mid["mae_after"]) / res_mid["mae_before"]
    
    # ── 3. Prepare plotting table  ────────────────────────────────────────
    plot_df = res_mid.assign(model = res_mid["param.model_kind"].str.upper())
    
    print("Number of runs per model architecture:")
    print(plot_df.groupby("model").size())
    
    # ── 4. Boxplot  ───────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.boxplot(data=plot_df,
                x="model", y="gain_pct",
                palette="Blues", ax=ax, showfliers=False, whis=1.5)
    
    sns.stripplot(data=plot_df, x="model", y="gain_pct",
                  color="k", size=3, alpha=.4, jitter=.25, ax=ax)
    
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    # ax.set_ylabel("Error reduction with RES  [%]")
    ax.set_ylabel("Error reduction vs. ALWAYS  [%]")

    ax.set_xlabel("Forecasting model architecture")
    ax.set_ylim(-5, ax.get_ylim()[1])
    # ax.set_title("Fig. 3 – Benefit of RES (λ mid-range) by architecture")
    ax.set_title(
        f"Fig. 3 – Benefit of RES (λ ∈ [{LOW_LAM:g}, {HIGH_LAM:g}]) by architecture", pad=10
    )
    
    
    # annotate n for each box
    for i, model in enumerate(sorted(plot_df["model"].unique())):
        n = (plot_df["model"] == model).sum()
        ax.text(i, -0.009, f"n={n}", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png,
                dpi=300, bbox_inches="tight")
    # runs is the full MLflow run-level table
    res_only = runs[runs["param.policy_kind"].str.lower() == "res"].copy()

    res_only["gain_pct"] = 100 * (
        res_only["metric.mean_mae_before"] - res_only["metric.mean_mae_after"]
    ) / res_only["metric.mean_mae_before"]

    med = (res_only
           .groupby("param.model_kind")["gain_pct"]
           .median()
           .round(1))

    med.to_frame(name="median_gain_pct").to_csv("tables/arch_medians.csv")
    print(med)



    # =======================================================
    # Fig. 4 – Best RES net utility: dataset × model heat-map
    # =======================================================
def dataset_model_heatmap(runs: pd.DataFrame,
                         out_png: str = "figures/fig04_dataset_model_heatmap.pdf"):


    # ── 0. Load run-level table ───────────────────────────────────────────
    # runs = pd.read_csv(flow_csv_in)
    
    def _to_num(col):
        try:
            return pd.to_numeric(col)
        except Exception:
            return col
    
    for c in runs.columns:
        if c.startswith(("metric.", "param.")):
            runs[c] = _to_num(runs[c])
    
    # ── 1. Keep only RES policy runs (drop sentinel λ) ────────────────────
    
    res_runs = runs[runs["param.policy_kind"].str.lower() == "res"].copy()
    res_runs["lambda_val"] = pd.to_numeric(res_runs["param.lambda"],
                                           errors="coerce")
    
    res_runs = runs[runs["param.policy_kind"].str.lower() == "res"].copy()
    res_runs["lambda_val"] = pd.to_numeric(res_runs["param.lambda"],
                                           errors="coerce")
    
    # sentinels = [ALWAYS, NEVER]
    # res_runs = res_runs[~res_runs["lambda_val"].isin(sentinels)].copy()
    
    # ── 2. Average net utility over the 3 seeds for every λ  --------------
    mean_by_seed = (
        res_runs
        .groupby(["param.dataset", "param.model_kind", "lambda_val"], as_index=False)
        .agg(res_mean=("metric.res", "mean"))
    )
    
    # ── 3. Pick λ with the highest mean utility per (dataset, model)  ------
    best = (
        mean_by_seed
        .sort_values(["param.dataset", "param.model_kind", "res_mean"],
                     ascending=[True, True, False])
        .drop_duplicates(subset=["param.dataset", "param.model_kind"])
        .rename(columns={"param.dataset": "dataset",
                         "param.model_kind": "model",
                         "res_mean": "best_res"})
    )
    
    # ── 4. Pivot to matrix form for heat-map -------------------------------
    heat = (
        best.assign(model=lambda s: s["model"].str.upper())
            .pivot(index="dataset", columns="model", values="best_res")
            .sort_index()
    )
    
    print("Best utility per dataset–model pair (rounded):\n",
          heat.round(2))
    
    Path("tables").mkdir(exist_ok=True, parents=True)
    heat.to_csv("tables/utility_raw.csv", float_format="%.3f")
    
    # ── 5. Plot ------------------------------------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    cmap = sns.diverging_palette(10, 133, as_cmap=True)   # red⇄green
    heat_log = heat.applymap(lambda u: np.sign(u) * np.log10(abs(u)+1e-9))
    
    ABS_MAX = np.nanmax(np.abs(heat_log.values))
    sns.heatmap(heat_log, annot=True, fmt=".2f", cmap=cmap, center=0, vmin=-ABS_MAX, vmax=ABS_MAX,
                linewidths=.5, cbar_kws=dict(label="log₁₀ Net utility  U"))
    
    ax.set_title("Fig. 4 – Best RES Utility by Dataset and Model")
    ax.set_xlabel("Forecasting model")
    ax.set_ylabel("Dataset")
    
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png,
                dpi=300, bbox_inches="tight")


def survival_curve(runs: pd.DataFrame, dec: pd.DataFrame, out_png: str= "figures/fig05_survival_curve.pdf"):
    """
    Generates and saves a Kaplan-Meier survival curve plot. This version is
    robust to sparse data from sanity checks.

    Args:
        runs: DataFrame of run-level metadata.
        dec: DataFrame of week-level decision logs.
        out_png: The full, explicit path to save the output PNG file.
    """
    # 0 ── Data Preparation -------------------------------------------
    decisions = dec.copy()
    
    if "retrained" not in decisions.columns or "week" not in decisions.columns:
        print("⚠️  Warning: Skipping survival curve. 'retrained' or 'week' column missing.")
        return

    decisions["retrained"] = decisions["retrained"].astype(bool)
    decisions["week_dt"]   = pd.to_datetime(decisions["week"], errors="coerce")
    
    # 1 ── Canonical paper labels ----------------------------------------
    decisions["policy_label"] = decisions["policy_kind"].str.lower().map({
        "always": "ALWAYS", "never": "NEVER", "res": "RES",
        "fixed": "FIXED", "ph": "PH", "random": "RANDOM", "cara": "CARA",
    }).fillna("OTHER")
    
    keep = ["RES", "ALWAYS", "NEVER", "FIXED", "PH", "RANDOM", "CARA"]
    decisions = decisions[decisions["policy_label"].isin(keep)].copy()
    
    if decisions.empty:
        print("⚠️  Warning: Skipping survival curve. No data left after filtering.")
        return

    # 2 ── Build lifetime table (duration, event, policy) --------------
    records = []
    for (run_id, pol), g in decisions.groupby(["run_id", "policy_label"]):
        g = g.sort_values("week_dt" if g["week_dt"].notna().all() else "week").dropna(subset=["week"])
        if g.empty: continue

        timeline = g["week_dt"].values if g["week_dt"].notna().all() else g["week"].values
        retrain_flags = g["retrained"].values
    
        # FIXED: This is the core robustness check.
        # If no retrains ever happened, the entire run is one "right-censored"
        # observation. Its lifetime is the full length of the simulation.
        if not np.any(retrain_flags):
            records.append((len(timeline), 0, pol)) # event=0 (censored)
            continue

        # This block now only runs if there is AT LEAST ONE retraining event.
        if g["week_dt"].notna().all():
            one_unit = np.timedelta64(1, "W")
            dummy = timeline[0] - one_unit
            diff_func = lambda a, b: int((a - b) / one_unit)
        else:
            dummy = timeline[0] - 1
            diff_func = lambda a, b: int(a - b)
    
        retrain_pts = timeline[retrain_flags]
        retrain_pts = np.insert(retrain_pts, 0, dummy)
    
        for prev, this in zip(retrain_pts[:-1], retrain_pts[1:]):
            records.append((max(diff_func(this, prev), 1), 1, pol)) # event=1
        
        # This is now safe because retrain_pts is guaranteed non-empty.
        tail = max(diff_func(timeline[-1], retrain_pts[-1]), 1)
        records.append((tail, 0, pol)) # event=0
    
    if not records:
        print("⚠️  Warning: Skipping survival curve. No valid lifetimes could be computed.")
        return
        
    lt_df = pd.DataFrame(records, columns=["duration", "event", "policy"])
    lt_df1 = lt_df.groupby(["policy", "duration"], as_index=False)["event"].sum()
    lt_df1.to_csv("tables/survival_curve_data.csv")
    
    # 3 ── Kaplan–Meier plot --------------------------------------------
    # FIXED: Use a robust dictionary (a map) to assign colors correctly.
    color_map = {
        "ALWAYS": "#0072B2", "CARA": "#D55E00", "FIXED": "#CC79A7",
        "NEVER": "#999999", "PH": "#009E73", "RANDOM": "#F0E442", "RES": "#56B4E9"
    }
    style_map = {"RES":"--", "CARA":"-", "PH":":", "FIXED":"-.", "RANDOM":"-",
             "ALWAYS":"-", "NEVER":"-"}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    kmf = KaplanMeierFitter()
    
    policies_in_data = sorted(lt_df["policy"].unique())
    for pol in policies_in_data:
        m = lt_df["policy"] == pol
        kmf.fit(durations=lt_df.loc[m, "duration"], event_observed=lt_df.loc[m, "event"], label=pol)
        color = color_map.get(pol, "#000000") # Default to black if policy is unknown
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=2, ls=style_map.get(pol,"-"))
    
    ax.set_xlabel("Weeks since last retrain")
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_ylabel("Survival probability P(T > t)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_title("Survival Curve of Model Versions by Policy")
    ax.legend(title="Policy", frameon=False)
    
    # FIXED: Use the full `out_png` path and ensure its parent directory exists.
    output_path = Path(out_png)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✔  Saved survival curve → {output_path}")



    # --------------------------------------------------------------------
    # Fig. 6 – Microscope MAE plot
    # Handles multiple seeds:  mode = "single"  or  mode = "average"
    # --------------------------------------------------------------------

def microscope_mae(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/fig06_microscope_mae.pdf"):
    
   # ---- slice – can be overridden via env-vars ------------------------
    DATASET     = os.getenv("MICRO_DATASET" , "m5_sales_clean")
    MODEL_KIND  = os.getenv("MICRO_MODEL"   , "lightgbm")
    POLICY_KIND = os.getenv("MICRO_POLICY"  , "res")
    LAMBDA_VAL  = float(os.getenv("MICRO_LAMBDA", 0.5))
    # ----------------------------------------------------
    
    MODE        = "single"       # "single"  or  "average"
    RUN_ID      = None           # only used in "single" mode
    # --------------------------------------------------------------------
    
    # df = pd.read_csv(all_decisions_csv)
    df = dec.copy()
    df["lambda"]   = pd.to_numeric(df["lambda"], errors="coerce")
    df["week_dt"]  = pd.to_datetime(df["week"], errors="coerce")
    df["retrained"] = df["retrained"].astype(int)          # robust cast
    
    # ---- filters --------------------------------------------------------
    mask  = df["dataset"].str.lower()     == DATASET.lower()
    mask &= df["model_kind"].str.lower()  == MODEL_KIND.lower()
    mask &= df["policy_kind"].str.lower() == POLICY_KIND.lower()
    
    if POLICY_KIND.lower() == "res" and LAMBDA_VAL is not None:
        mask &= np.isclose(df["lambda"], LAMBDA_VAL, atol=1e-9, rtol=1e-6)
    
    slice_df = df.loc[mask].copy()
    if slice_df.empty:
        raise RuntimeError("Filter left zero rows; check constants.")
    
    # ---- MODE A – one concrete seed ------------------------------------
    if MODE == "single":
        if RUN_ID is None:
            stats = (slice_df.groupby("run_id")["mae_before"].mean()
                               .reset_index(name="mean_mae"))
            med   = stats["mean_mae"].median()
            RUN_ID = stats.iloc[(stats["mean_mae"] - med).abs()
                                .argmin()]["run_id"]
            print(f"[single] chosen run_id = {RUN_ID}")
    
        dfr = (slice_df[slice_df["run_id"] == RUN_ID]
                 .sort_values("week_dt")
                 .reset_index(drop=True))
    
    # ---- MODE B – average over all seeds -------------------------------
    elif MODE == "average":
        agg = {"mae_before": "mean", "mae_after": "mean", "retrained": "max"}
        dfr = (slice_df.groupby("week_dt", as_index=False)
                         .agg(agg)
                         .sort_values("week_dt"))
        print(f"[avg] weeks aggregated: {len(dfr)} "
              f"(from {slice_df['run_id'].nunique()} seeds)")
    else:
        raise ValueError("MODE must be 'single' or 'average'")
    
    # ---- Plot -----------------------------------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    ax.plot(dfr["week_dt"], dfr["mae_before"],
            color="steelblue", lw=2, label="MAE before retrain")
    ax.plot(dfr["week_dt"], dfr["mae_after"],
        color="darkorange", lw=1.5, label="MAE after retrain")

    
    # vertical bars at retrain weeks
    for w in dfr.loc[dfr["retrained"] != 0, "week_dt"]:
        # ax.axvline(w, color="crimson", lw=0.8, alpha=0.8)
        ax.axvline(w, color="crimson", lw=0.5, alpha=0.25)
    
    # x-axis formatting
    ax.set_xlabel("Calendar week")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    
    ax.set_ylabel("MAE")
    
    subtitle = "single seed" if MODE == "single" else "mean of 3 seeds"
    policy_descr = f"{POLICY_KIND.upper()}" + \
                   (f" λ={LAMBDA_VAL}" if POLICY_KIND.lower() == "res" else "")
    ax.set_title(f"Microscope – {DATASET} | {MODEL_KIND.upper()} | "
                 f"{policy_descr} | {subtitle}")
    ax.legend(frameon=False)
    
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"saved → {out_png}")


def stacked_cost_carbon(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/fig07_stacked_cost_carbon_cpu.pdf"):


    #!/usr/bin/env python3
    """
    Fig. 7 – Year-scaled run-time / carbon breakdown (CPU-only server)
    
    • Four datasets × three seeds per policy
    • Projects weekly cost to 1-year deployment on N_SERIES concurrent streams
    • Converts wall-clock seconds → CPU-hours → kg CO₂
    """
    
    DATASETS   = ["nyc_taxi_clean", "m5_sales_clean",
                  "wiki_weekly",    "electricity_weekly"]
    MODEL_KIND = "lightgbm"
    
    POLICIES = [
        ("always", "ALWAYS"        , dict()),
        ("res"   , "RES  λ = 0.3"  , dict(lambda_val=0.3)),
        ("fixed" , "FIXED k = 4"   , dict(k_val=4)),
        ("random", "RAND  p = 0.15", dict(p_val=0.15)),
        ("ph"    , "PERF-HORIZ"    , dict()),
        ("cara" , "CARA τ = 0.3",  dict(tau_val=0.3)),
    ]
    
    # ── deployment & carbon model (CPU-only) ─────────────────────────
    N_SERIES        = 500                       # parallel time-series
    YEAR_SCALE      = 52 * N_SERIES             # weeks → 1 year × many streams
    CPU_W           = 1176                      # constant power draw of the VM
    CO2_PER_KWH     = 0.43                      # regional grid factor (kg/kWh)
    CO2_PER_CPUH    = CPU_W / 1000 * CO2_PER_KWH  # kg CO₂ per CPU-hour
    # ────────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    # COLORS = {"retrain": "#c44e52", "init_misc": "#4c72b0"}  # red / blue
    COLORS = {"retrain":"#CC6677", "init_misc":"#88CCEE"}

    # ────────────────────────────────────────────────────────────────
    # dec = pd.read_csv(all_decisions_csv)
    for col in ["lambda", "k", "p"]:
        if col in dec.columns:
            dec[col] = pd.to_numeric(dec[col], errors="coerce")
    # NEW: make step, cost, retrained numeric / boolean
    for col in ["step", "cost"]:
        if col in dec.columns:
            dec[col] = pd.to_numeric(dec[col], errors="coerce")
    
    if "retrained" in dec.columns:
        dec["retrained"] = dec["retrained"].astype(int, errors="ignore")
    
    # ────────────────────────────────────────────────────────────────
    def summarise(policy_kind, lambda_val=None, k_val=None, p_val=None, tau_val=None):
        """Return a vector with init_misc & retrain CPU-hours per year."""
        m  = dec["model_kind"].str.lower()  == MODEL_KIND.lower()
        m &= dec["policy_kind"].str.lower() == policy_kind.lower()
        m &= dec["dataset"].isin(DATASETS)
    
        if policy_kind == "res"    and lambda_val is not None:
            m &= np.isclose(dec["lambda"], lambda_val, atol=1e-9)
        if policy_kind == "fixed"  and k_val is not None:
            m &= dec["k"] == k_val
        if policy_kind == "random" and p_val is not None:
            m &= np.isclose(dec["p"], p_val, atol=1e-9)
        if policy_kind == "cara" and tau_val is not None:          
            m &= np.isclose(dec["tau"], tau_val, atol=1e-9)
    
        d = dec[m]
        if d.empty:
            raise RuntimeError(f"No rows for policy {policy_kind}")
    
        def buckets(df):
            return pd.Series({
                # first step  +  weeks without retrain
                "init_misc": df.loc[df["step"] == 1, "cost"].sum()
                           + df.loc[df["retrained"] == 0, "cost"].sum(),
                # actual retrain operations
                "retrain"  : df.loc[(df["step"] > 1) & (df["retrained"] == 1),
                                    "cost"].sum()
            })
    
        per_run = d.groupby("run_id", sort=False, group_keys=False)\
                   .apply(buckets, include_groups=False)
    
        raw_sec = per_run.sum().sum()
        hours   = per_run.sum() * YEAR_SCALE / 3600    # sec → h
        print(f"{policy_kind:<6} raw={raw_sec:7.1f} s  → {hours.sum():6.2f} CPU-h/yr")
        return hours
    
    # ────────────────────────────────────────────────────────────────
    rows = []
    for pk, label, kw in POLICIES:
        vec = summarise(pk, **kw)
        vec["label"] = label
        rows.append(vec)
    
    summary = (pd.DataFrame(rows)
                 .set_index("label")
                 [["init_misc", "retrain"]])
    
    summary["co2"] = summary.sum(axis=1) * CO2_PER_CPUH
    
    # drop blue slice if negligible everywhere (<1 %)
    if (summary["init_misc"] < 0.01 * summary[["init_misc", "retrain"]]
            .sum(axis=1)).all():
        summary = summary.drop(columns="init_misc")
        COLORS.pop("init_misc")
    
    print("\nCPU-hours/year:\n", summary.drop(columns="co2"))
    # ────────────────────────────────────────────────────────────────
    # Plot
    # ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    
    cols_plot = ["retrain"] + (["init_misc"] if "init_misc" in summary else [])
    summary[cols_plot].plot(kind="bar", stacked=True,
                            color=[COLORS[c] for c in cols_plot],
                            edgecolor="none", ax=ax)
    
    # optional stub for PERF-HORIZ if its bar would be invisible ----------
    if "PERF-HORIZ" in summary.index and summary.loc["PERF-HORIZ", cols_plot].sum() == 0:
        stub_x = summary.index.get_loc("PERF-HORIZ")
        ax.bar(stub_x, 0.02, width=0.5, color=COLORS["retrain"])
        ax.text(stub_x, 0.05, "0 kg CO₂", ha="center", va="bottom", fontsize=8)
    

    ax.set_title("Annual retrain-pipeline CPU cost – 4 datasets, 500 streams")
    ax.set_ylabel(f"CPU-hours / year  ({CO2_PER_KWH} kg CO$_2$/kWh)")


    # ax.set_ylabel(
    # f"CPU-hours per year  ({N_SERIES} streams, "
    # f"{CPU_W} W, {CO2_PER_KWH} kg CO₂/kWh)")
    ax.set_xlabel("")
#     ax.set_title(
#     f"CPU cost per year – {MODEL_KIND.upper()} "
#     f"(4 datasets, 52 weeks × {N_SERIES} streams)")
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    
    y_max = summary[cols_plot].sum(axis=1).max()
    for i, (label, row) in enumerate(summary.iterrows()):
        total_h = row[cols_plot].sum()
        ax.text(i, total_h + 0.03 * y_max,
                f"{row['co2']:.2f} kg CO₂",
                ha="center", va="bottom", fontsize=8)
    
    # remove legend when only one slice remains --------------------------
    if len(cols_plot) == 1:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    else:
        ax.legend(frameon=False, title="")
    
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print("saved →", out_png)

    
    
def lambda_dataset_heatmap(runs: pd.DataFrame, dec: pd.DataFrame,
                            out_png: str = "figures/figA1_lambda_dataset_heatmap.pdf"):

    # manual ordering – columns will be λ values now
    LAMBDA_ORDER = [0,
                0.01, 0.03, 0.10,
                0.30, 0.40, 0.50, 0.60, 0.70, 0.85,
                1.00]

    DATASET_ORDER = ["nyc_taxi_clean", "m5_sales_clean",
                     "wiki_weekly", "electricity_weekly"]

    # ------------------- load + filter -------------------------------
    dec["lambda"]  = pd.to_numeric(dec["lambda"], errors="coerce")
    dec["benefit"] = pd.to_numeric(dec["benefit"], errors="coerce")
    dec["cost"]    = pd.to_numeric(dec["cost"],    errors="coerce")

    df = dec[dec["policy_kind"].str.lower() == "res"].copy()
    df["net"] = df["benefit"] - df["cost"]

    # Σ_week net utility per run
    run_util = (df.groupby(["dataset", "lambda", "run_id"], sort=False)["net"]
                  .sum()
                  .reset_index())

    # mean over the three seeds
    mean_util = (run_util.groupby(["dataset", "lambda"], sort=False)["net"]
                           .mean()
                           .reset_index())

    # pivot → rows = dataset , columns = λ
    heat = (mean_util.pivot(index="dataset", columns="lambda", values="net")
                       .reindex(DATASET_ORDER)
                       .reindex(columns=LAMBDA_ORDER))

    # ------------------- plot ----------------------------------------
    sns.set_style("white")
    n_cols = heat.shape[1]
    fig_w = 1.8 + 0.9 * n_cols      # 0.9 inch per λ tick
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    heat_log = heat.applymap(lambda u: np.sign(u) * np.log10(abs(u) + 1e-9))
    vmax = np.nanmax(np.abs(heat_log.values))

    sns.heatmap(heat_log,
                cmap="vlag", center=0, vmin=-vmax, vmax=vmax,
                linewidths=.5, linecolor="lightgrey",
                annot=True, fmt=".1f", annot_kws=dict(size=7),
                cbar_kws=dict(label="Net utility  Σ (benefit − cost)"),
                ax=ax)

    ax.set_xlabel("λ")
    ax.set_ylabel("Dataset")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Mean net utility across seeds\n(positive ⇒ policy worthwhile)")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print("saved →", out_png)



def auto_vs_fixed_lambda(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/figA2_auto_vs_fixed.pdf"):

    out_csv: str = "tables/auto_lambda_vs_best_fixed.csv"

    #!/usr/bin/env python3
    """
    Compare RES-auto-λ to the best fixed-λ per dataset
    Metric: net utility (benefit − cost) summed over weeks
    Produces both a CSV and an optional bar plot.
    """

    
    DATASETS   = ["nyc_taxi_clean", "m5_sales_clean",
                  "wiki_weekly",    "electricity_weekly"]
    MODEL_KIND = "lightgbm"
    
    # name conventions in your CSV
    POL_AUTO   = "res_auto"          # ← adopt whatever you logged
    POL_FIXED  = "res"               # with explicit λ column
    
    # dec = pd.read_csv(all_decisions_csv)
    
    dec["lambda"]  = pd.to_numeric(dec["lambda"], errors="coerce")
    dec["benefit"] = pd.to_numeric(dec["benefit"], errors="coerce")
    dec["cost"]    = pd.to_numeric(dec["cost"],    errors="coerce")
    
    # ------------------------------------------------------------
    # helper: return net-utility DataFrame filtered by dataset + model
    # ------------------------------------------------------------
    def net_df(dataset):
        d = dec[(dec["dataset"] == dataset) &
                (dec["model_kind"] == MODEL_KIND)].copy()
        d["net"] = d["benefit"] - d["cost"]
        return d
    
    records = []
    for ds in DATASETS:
        d = net_df(ds)
    
        # ---------- fixed-λ  (lambda column is NOT null) ----------
        fixed = (d[d["lambda"].notna()]
                 .groupby(["lambda", "run_id"])["net"].sum()
                 .reset_index())
    
        # best λ = argmax over mean(net)  --------------------------
        best_lam = (fixed.groupby("lambda")["net"]
                          .mean()
                          .idxmax())
    
        net_fixed = (fixed[fixed["lambda"] == best_lam]
                     .groupby("run_id")["net"]
                     .mean()                               # average over seeds
                     .mean())                              # scalar
    
        # ---------- auto-λ  (lambda_auto column is NOT null) ------
        auto = (d[d["lambda_auto"].notna()]
                .groupby("run_id")["net"]
                .sum()
                .mean())                                   # scalar
    
        records.append(dict(dataset          = ds,
                            best_fixed_lambda = best_lam,
                            net_fixed         = net_fixed,
                            net_auto          = auto,
                            ratio_auto_fixed  = auto / net_fixed
                                                if pd.notna(auto) else np.nan))
    
    tbl = pd.DataFrame(records)
    num_cols = ["net_fixed", "net_auto", "ratio_auto_fixed"]
    tbl[num_cols] = tbl[num_cols].round(3) 
    tbl.to_csv(out_csv, index=False)
    print(tbl)
    print("\nsaved  →", out_csv)
    
    # optional bar plot ---------------------------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6,3))
    tbl_melt = tbl.melt(id_vars="dataset",
                        value_vars=["net_fixed","net_auto"],
                        var_name="policy", value_name="net")
    sns.barplot(data=tbl_melt, x="dataset", y="net",
                hue="policy", palette=["#4c72b0","#55a868"], ax=ax)
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)


    
    ax.set_ylabel("Mean net utility")
    ax.set_xlabel("")
    ax.set_title("RES: auto-λ vs. best fixed-λ")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("plot    →", out_png)
    
    from scipy.stats import wilcoxon
    delta = tbl["net_fixed"] - tbl["net_auto"]      # + ⇒ fixed better
    W,p = wilcoxon(delta, alternative="greater")    # H0: med(delta)=0
    print(f"Wilcoxon W={W}, p={p:.3f}")
    ax.text(3.7, ax.get_ylim()[1]*0.9,
        f"Wilcoxon one-sided p={p:.3f}", fontsize=8, ha="right")


def frontier_bootstrap(runs: pd.DataFrame, dec: pd.DataFrame,
                         out_png: str = "figures/fig09_frontier.pdf"):
    #!/usr/bin/env python3
    """
    Aggregate cost-benefit numbers for the frontier plot
    (RES, Fixed-4, Random-p, CARA).
    Outputs a CSV with mean and 95 % CI and updates Fig. 1.
    """
    
    OUT_TAB = Path("tables/frontier_bootstrap.csv")
    
    DATASETS = ["nyc_taxi_clean","m5_sales_clean","wiki_weekly","electricity_weekly"]
    POLICIES = ["res","fixed","random","cara"]          # keep lower-case
    UTIL_COL = "utility" 
    
    LAMBDA = 0.50
    
    # dec = pd.read_csv(CSV)
    dec = dec[dec["dataset"].isin(DATASETS) &
              dec["policy_kind"].str.lower().isin(POLICIES)].copy()
    dec[UTIL_COL] = dec["benefit"] - LAMBDA * dec["cost"]
    
    # ------------------------------------------------------------
    rows = []
    rng  = np.random.default_rng(2024)
    for (ds,pol), g in dec.groupby(["dataset","policy_kind"]):
        util = g.groupby("run_id")[UTIL_COL].sum()           # one scalar per seed
        ci   = bootstrap((util.values,),
                         np.mean, vectorized=False,
                         n_resamples=2000,
                         random_state=rng).confidence_interval
        rows.append(dict(dataset=ds,
                         policy=pol.upper(),
                         mean_util=util.mean(),
                         ci_lo=ci.low,
                         ci_hi=ci.high))
    
    front = pd.DataFrame(rows)
    order = ["nyc_taxi_clean","m5_sales_clean",
             "wiki_weekly","electricity_weekly"]
    front["dataset"] = pd.Categorical(front["dataset"], order)
    front = front.sort_values(["policy","dataset"])
    num_cols = ["mean_util", "ci_lo", "ci_hi"
    ]
    front[num_cols] = front[num_cols].round(3) 
    front.to_csv(OUT_TAB, index=False)
    print("saved →", OUT_TAB)
    
    # ------------------------------------------------------------
    # quick frontier plot (mean only, no error-bar to keep Fig.1 uncluttered)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6,4))
    for pol, df in front.groupby("policy"):
        ax.plot(df["dataset"], df["mean_util"],
                marker="o", label=pol)
        ax.fill_between(df["dataset"], df["ci_lo"], df["ci_hi"],
                    alpha=0.15, color=ax.lines[-1].get_color())
    ax.set_ylabel("Mean composite utility")
    ax.set_xlabel("")
    ax.set_title(f"Cost–benefit frontier - {LAMBDA} Lambda – four datasets")
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("saved →", out_png)



def summary_table(runs, out_tex="tables/table4_master.csv"):
    cols = ["param.dataset","param.policy_kind","metric.mean_mae_after",
            "metric.total_retrain_sec"]
    df = runs[cols].copy()
    df["util"] = runs["metric.total_benefit"] - 0.5*runs["metric.total_retrain_sec"]
    agg = (df.groupby(["param.dataset","param.policy_kind"])
             .agg(mae_after=("metric.mean_mae_after","mean"),
                  sec=("metric.total_retrain_sec","mean"),
                  util=("util","mean"))
             .round(2))
    agg.to_csv(out_tex)
    print("table →", out_tex)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def retrain_table_and_plot(runs: pd.DataFrame,
                           dec: pd.DataFrame,
                           baselines: list[str] = ("always", "cara", "fixed"),
                           out_csv: str = "tables/retrain_counts.csv",
                           out_png: str = "figures/fig08_retrain_freq.pdf"):

    """
    Build the retrain-count table AND generate one figure for every
    (dataset, model_kind) pair found in *dec*.

    Parameters
    ----------
    dec , runs
        In-memory DataFrames corresponding to all_decisions.csv and
        all_mlflow_runs.csv.
    baselines
        Policies that appear as horizontal reference lines.
    out_csv
        File path for the aggregated table.
    out_dir
        Directory where all figures will be stored.

    Returns
    -------
    agg  : pd.DataFrame
        Tidy table with columns
        [dataset, model_kind, policy_kind, lambda, fixed_k,
         mean, sd, n_runs].
    figs : dict
        {(dataset, model_kind): Figure}
    """

    # ── 1 prepare columns ────────────────────────────────────────────
    dec = dec.copy()
    dec["retrained"] = dec["retrained"].astype(int)

    runs = runs.copy()
    if "lambda" not in runs.columns:
        runs["lambda"] = pd.to_numeric(runs["param.lambda"], errors="coerce")

    if "fixed_k" not in runs.columns:
        if "k" in runs.columns:
            runs["fixed_k"] = runs["k"]
        elif "param.k" in runs.columns:
            runs["fixed_k"] = runs["param.k"]
        else:
            runs["fixed_k"] = pd.NA

    # ── 2 count triggers per run ────────────────────────────────────
    cnt = (dec.groupby("run_id", as_index=False)["retrained"]
             .sum()
             .rename(columns={"retrained": "n_retrains"}))
    

    meta_cols = ["run_id", "param.policy_kind", "param.dataset",
                 "param.model_kind", "param.lambda", "param.k", "param.p", "param.tau"]

    cnt = cnt.merge(runs[meta_cols], on="run_id", how="left")

    # ── 3 aggregate over seeds / runs ───────────────────────────────
    agg = (cnt.groupby(["param.dataset", "param.model_kind", "param.policy_kind",
                        "param.lambda", "param.k", "param.p", "param.tau"],
                       as_index=False)["n_retrains"]
             .agg(mean="mean", sd="std", n_runs="size"))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print("saved table →", out_csv)

    # ── 4 create one figure per (dataset, model) ────────────────────
    # Path(out_dir).mkdir(parents=True, exist_ok=True)
    figs: Dict[tuple[str, str], plt.Figure] = {}

    for (ds, mdl), panel in agg.groupby(["param.dataset", "param.model_kind"]):
        fig, ax = plt.subplots(figsize=(6, 4))

        # RES curve
        res_rows = panel.loc[panel["param.policy_kind"].str.lower() == "res"]
        sns.lineplot(data=res_rows,
                     x="lambda", y="mean",
                     marker="o", ax=ax, label="RES")

        # baselines
        for pol in baselines:
            mask = panel["param.policy_kind"].str.lower() == pol
            if mask.any():
                y_val = panel.loc[mask, "mean"].iloc[0]
                ax.hlines(y_val,
                          xmin=res_rows["lambda"].min(),
                          xmax=res_rows["lambda"].max(),
                          linestyles="--",
                          label=pol.upper())
                ax.text(res_rows["lambda"].max() * 1.05,
                        y_val,
                        f"{y_val:.0f}",
                        va="center")

        # cosmetics
        ax.set_xscale("log")
        ax.set_xlabel("λ  (RES only)")
        ax.set_ylabel("Mean # retrains per run")
        ax.set_title(f"{ds} – {mdl.upper()}")
        ax.legend(title="Policy")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        print("saved figure →", out_png)






# ======================================================================
# entry-point
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
            description="Generate paper figures or a minimal sanity set")
    ap.add_argument("--minimal", action="store_true",
                    help="only create the survival-curve plot")
    args = ap.parse_args()
    runs = _load_runs()
    dec  = _load_decisions()
    do_minimal = args.minimal

    if not do_minimal:
            PLOTTING_FUNCS : List[Callable] = [
            plot_promotion_deploy_costs,
            appendix_full_grid_pareto,
            plot_pareto_tradeoff,
            pareto_deltas,
            lambda_robustness,
            table_lambda_robustness,
            auto_lambda_recovery,
            architecture_boxplot,
            dataset_model_heatmap,
            survival_curve,
            microscope_mae,
            stacked_cost_carbon,
            lambda_dataset_heatmap,
            auto_vs_fixed_lambda,
            frontier_bootstrap,
            summary_table,
            retrain_table_and_plot
        ]
    else:
        print("⚠️  Minimal mode – will generate only the survival curve.")
        pass

    if do_minimal:
        survival_curve(runs=runs, dec=dec,
                         out_png= "sanity_check/figures/fig05_survival_curve.pdf")
    else:
        for func in PLOTTING_FUNCS:
            # detect which args the function expects
            if func.__code__.co_argcount >= 3:         # (runs, dec, out_png)
                func(runs, dec)
            else:                                     # (runs, out_png) etc.
                func(runs)


if __name__ == "__main__":
    main()