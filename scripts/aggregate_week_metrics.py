#!/usr/bin/env python
"""
Aggregate every week_metrics.parquet belonging to ONE MLflow experiment
into a single tidy parquet.  Standardises column names and, if necessary,
computes the per-week benefit (MAE reduction versus the always-retrain
baseline: λ == 0).
"""
import argparse, pathlib, sys, mlflow, pandas as pd

# legacy → canonical column names
STD_RENAME = {
    "cost_sec":    "cost",
    "benefit_mae": "benefit",
}

def collect(exp_name: str, out_file: str):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        sys.exit(f"❌  MLflow experiment '{exp_name}' not found")

    frames = []
    for run in client.search_runs(exp.experiment_id,
                                  "attributes.status = 'FINISHED'",
                                  max_results=10_000):
        rid = run.info.run_id
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=rid, artifact_path="week_metrics.parquet")
        except (mlflow.exceptions.MlflowException, FileNotFoundError, OSError):
            print(f"  – skipping run {rid}: week_metrics.parquet not found")
            continue          # skip runs that pre-date the WeekLogger
        df = pd.read_parquet(local) 
        if "mae_after" in df.columns and "mae" not in df.columns:
            df = df.rename(columns={"mae_after": "mae"})
        df = df.rename(columns={k: v for k, v in STD_RENAME.items()  
                        if k in df.columns})

        # attach run-level metadata that downstream scripts need
        df["run_id"]      = rid
        df["dataset"]     = run.data.params.get("dataset")
        df["model_kind"]  = run.data.params.get("model_kind")
        df["policy_kind"] = run.data.params.get("policy_kind")
        df["lambda"]      = float(run.data.params.get("lambda", "nan"))
        frames.append(df)

    if not frames:
        sys.exit("❌  No week_metrics.parquet files found in this experiment")

    out = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # derive 'benefit' if it is still missing
    # benefit = MAE(always-retrain baseline) − MAE(current run)
    # ------------------------------------------------------------------
    if "benefit" not in out.columns:
        base = (out[out["lambda"] == 0]                            
        .loc[:, ["dataset", "model_kind", "week", "mae"]]
        .rename(columns={"mae": "mae_base"})
               )
        out = out.merge(base, on=["dataset", "model_kind", "week"], how="left")
        out["benefit"] = out["mae_base"] - out["mae"]
        out.drop(columns=["mae_base"], inplace=True)
    # scale benefit to a weekly TOTAL instead of a per-row delta
    if "weekly_rows" in out.columns:
        out["benefit"] = out["benefit"] * out["weekly_rows"]
        # optional – do the same for cost if it is per-row
        # out["cost"]    = out["cost"]    * out["weekly_rows"]
    elif "n_rows" in out.columns:
        out["benefit"] = out["benefit"] * out["n_rows"]
        # out["cost"]    = out["cost"]    * out["n_rows"]
    else:
        print("⚠️  cannot scale benefit – column weekly_rows / n_rows missing")

    # final sanity: cost & benefit must exist for plotting
    if "cost" not in out.columns:
        sys.exit("❌  column 'cost' missing after aggregation")

    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)
    
    print(f"✅  Saved {len(out):,} rows → {out_file}")

        # ---- additionally save run-level summary ------------------------
    run_summary = (
        out.groupby("run_id")
           .agg(tot_benefit=("benefit", "sum"),
                tot_cost   =("cost",     "sum"),
                n_retrains =("trained",  "sum"),
                n_steps    =("week",     "count"))
           .assign(res=lambda x: x.tot_benefit / x.tot_cost)
           .reset_index()
    )
    run_summary.to_parquet(
        pathlib.Path(out_file).with_name("run_summary.parquet"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    collect(args.experiment, args.out)
