#!/usr/bin/env python3
"""
export_decisions.py  –  collects every replay_log.csv under one MLflow
experiment and concatenates them into figures/all_decisions.csv
"""

import sys, shutil, tempfile
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import argparse, sys, os
import pandas as pd
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.tracking import file_store as _fs

# ----------------------------------------------------------------------
# 0. user-editable defaults
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "res_paper_grid"
REPLAY_FILE     = "replay_log.csv"
OUTPUT_CSV      = ROOT/"figures/all_decisions.csv"
UTILITY_CSV  = ROOT/"figures/run_utility.csv"
ALL_RUNS_CSV = ROOT/"figures/all_mlflow_runs.csv"
FAILED_LIST = ROOT / "figures/failed_runs.txt"
TRACKING_URI    = None                      # None = default


def _resolve_experiment(cli: MlflowClient, name_or_id: str):
    exp = cli.get_experiment_by_name(name_or_id) or \
          cli.get_experiment(name_or_id)
    if exp is None:
        sys.exit(f"✖  No experiment named/id = {name_or_id}")
    return exp


def _iterate_runs(cli: MlflowClient, exp_id: str):
    token = None
    while True:
        page = cli.search_runs([exp_id], "", run_view_type=1,
                               max_results=10_000, page_token=token)
        for r in page:
            yield r
        token = getattr(page, "token", None)
        if not token:
            break


# ----------------------------------------------------------------------
# 1. decision-level export
# ----------------------------------------------------------------------
def export_decisions(experiment_name: str = EXPERIMENT_NAME,
                     replay_file: str     = REPLAY_FILE,
                     output_csv: str      = OUTPUT_CSV,
                     tracking_uri: str | None = TRACKING_URI) -> None:
    """
    Collect every replay_log.csv in the MLflow experiment → one big CSV.
    Writes figures/failed_runs.txt and aborts if any run is incomplete.
    """
    cli = MlflowClient(tracking_uri=tracking_uri)

    # ── resolve experiment id ------------------------------------------
    exp = cli.get_experiment_by_name(experiment_name) or \
          cli.get_experiment(experiment_name)
    if exp is None:
        sys.exit(f"✖  No experiment named/id = {experiment_name}")
    exp_id = exp.experiment_id
    print(f"Exporting runs from experiment {exp_id}  ({exp.name})")

    runs = list(_iterate_runs(cli, exp_id))
    if not runs:
        sys.exit("✖  No runs found.")

    print(f"Found {len(runs)} runs – fetching {replay_file} …")
    tmp_root = Path(tempfile.mkdtemp(prefix="mlflow_replay_"))
    frames, failed = [], []

    try:
        for run in tqdm(runs, unit="run"):
            rid = run.info.run_id
            df_step = None

            # 1️⃣ MLflow artefact download
            try:
                local = Path(cli.download_artifacts(rid, replay_file,
                                                    dst_path=tmp_root))
                df_step = pd.read_csv(local)
            except Exception:
                pass

            # 2️⃣ local outputs/… fallback (covers pickling crash case)
            if df_step is None:
                guess = list((ROOT / "outputs").glob(f"*/*/{rid}/{replay_file}"))
                if guess:
                    df_step = pd.read_csv(guess[0])

            # 3️⃣ still missing → mark as failed and continue
            if df_step is None:
                failed.append(rid)
                continue

            # ── enrich with run-level metadata ------------------------
            p = run.data.params
            df_step["run_id"]      = rid
            df_step["dataset"]     = p.get("dataset",  "")
            df_step["model_kind"]  = p.get("model_kind", "")
            df_step["policy_kind"] = p.get("policy_kind", "").lower()   # ⇐ lower
            df_step["lambda"]      = pd.to_numeric(p.get("lambda", ""), errors="coerce")
            df_step["k"]           = pd.to_numeric(p.get("k",      ""), errors="coerce")
            df_step["p"]           = pd.to_numeric(p.get("p",      ""), errors="coerce")
            df_step["seed"]        = pd.to_numeric(p.get("seed",   ""), errors="coerce")
            frames.append(df_step)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── outcome handling -----------------------------------------------
    if failed:
        FAILED_LIST.parent.mkdir(parents=True, exist_ok=True)
        FAILED_LIST.write_text("\n".join(failed))
        sys.exit(f"✖  {len(failed)} runs missing {replay_file}. "
                 f"List written to {FAILED_LIST}")
    if not frames:
        sys.exit("✖  No replay logs could be parsed.")

    big = pd.concat(frames, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(output_csv, index=False)
    print(f"✔  Export complete → {output_csv}  "
          f"({big.shape[0]:,} rows, {big.shape[1]} cols)")

# ----------------------------------------------------------------------
# 2. optional “wide” MLflow run dump
# ----------------------------------------------------------------------
def export_all_runs(
        output_csv: str = ALL_RUNS_CSV,
        only_experiment: str | None = None) -> None:
    """Flat table (one row per run, every param|metric|tag column)."""
    if not getattr(_fs, "_patched_uuid", False):
        _orig = _fs._read_persisted_run_info_dict

        def _safe_uuid(d):
            if "run_uuid" not in d and "run_id" in d:
                d["run_uuid"] = d["run_id"]
            return _orig(d)

        _fs._read_persisted_run_info_dict = _safe_uuid
        _fs._patched_uuid = True

    cli   = MlflowClient()
    rows  : List[Dict[str, Any]] = []
    counts: Counter = Counter()

    def all_experiments():
        return cli.search_experiments() if hasattr(cli, "search_experiments") \
               else cli.list_experiments()

    def fetch_runs(eid: str):
        token = None
        while True:
            page = cli.search_runs([eid], "", run_view_type=1,
                                   max_results=10_000, page_token=token)
            for r in page:
                yield r
            token = getattr(page, "token", None)
            if not token:
                break

    def flat(run) -> Dict[str, Any]:
        info, data = run.info, run.data
        out = dict(run_id=info.run_id,
                   experiment_id=info.experiment_id,
                   status=info.status,
                   start_time=info.start_time,
                   end_time=info.end_time,
                   artifact_uri=info.artifact_uri)
        out |= {f"param.{k}": v for k, v in data.params.items()}
        out |= {f"metric.{k}": v for k, v in data.metrics.items()}
        out |= {f"tag.{k}": v for k, v in data.tags.items()}
        return out

    for exp in all_experiments():
        if only_experiment and                       \
           exp.name != str(only_experiment) and      \
           exp.experiment_id != str(only_experiment):
            continue
        runs = list(fetch_runs(exp.experiment_id))
        counts[(exp.experiment_id, exp.name)] = len(runs)
        rows.extend(flat(r) for r in runs)

    print("Runs per experiment")
    for (eid, name), n in counts.most_common():
        print(f"{eid:<4} {name:<30} : {n}")
    print(f"Total runs exported : {len(rows):,}")

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✔  Saved → {output_csv}")


# ─────────────────────────────────────────────────────────────────────
# 3.  run-level utility export  →  run_utility.csv
# ─────────────────────────────────────────────────────────────────────
def export_run_utility(experiment_name: str = EXPERIMENT_NAME,
                       replay_file: str     = REPLAY_FILE,
                       output_csv: str      = UTILITY_CSV,
                       lambda_eval: float   = 0.5,
                       tracking_uri: str | None = TRACKING_URI) -> None:
    """
    utility(run) = Σ benefit_norm  −  λ_eval · Σ cost
    """
    cli  = MlflowClient(tracking_uri=tracking_uri)
    exp  = _resolve_experiment(cli, experiment_name)
    runs = list(_iterate_runs(cli, exp.experiment_id))

    rows: List[Dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="mlflow_replay_"))

    try:
        for run in tqdm(runs, unit="run"):
            rid = run.info.run_id
            try:
                local = Path(cli.download_artifacts(rid, replay_file, tmp_root))
                df = pd.read_csv(local)
            except Exception:
                continue                             # skip if no artefact

            if not {"benefit_norm", "cost"}.issubset(df.columns):
                continue                             # insufficient data

            util = df["benefit_norm"].sum() - lambda_eval * df["cost"].sum()
            p    = run.data.params

            rows.append(dict(run_id=rid,
                             seed=pd.to_numeric(p.get("seed", ""), errors="coerce"),
                             dataset=p.get("dataset", ""),
                             model_kind=p.get("model_kind", ""),
                             policy_kind=p.get("policy_kind", ""),
                             lambda_=pd.to_numeric(p.get("lambda", ""), errors="coerce"),
                             k=pd.to_numeric(p.get("k", ""), errors="coerce"),
                             p_=pd.to_numeric(p.get("p", ""), errors="coerce"),
                             utility=util))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not rows:
        sys.exit("✖  No run could provide both benefit_norm and cost.")

    df_u = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_u.to_csv(output_csv, index=False)
    print(f"✔  run_utility.csv saved     ({len(df_u):,} rows) "
          f"with λ_eval = {lambda_eval}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?",
                    default=EXPERIMENT_NAME,
                    help="MLflow experiment name or ID")
    ap.add_argument("uri", nargs="?",
                    default=TRACKING_URI,
                    help="MLflow tracking URI (file:… or http://… )")
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    if args.experiment == 'sanity_check':
        OUTPUT_CSV      = ROOT/"sanity_check/figures/all_decisions.csv"
        UTILITY_CSV  = ROOT/"sanity_check/figures/run_utility.csv"
        ALL_RUNS_CSV = ROOT/"sanity_check/figures/all_mlflow_runs.csv"
    else:
        pass

    export_decisions(args.experiment,                            
                     replay_file=REPLAY_FILE,
                     output_csv=OUTPUT_CSV,
                     tracking_uri=args.uri)                      

    export_run_utility(args.experiment,
                       replay_file=REPLAY_FILE,
                       output_csv=UTILITY_CSV,
                       tracking_uri=args.uri)

    export_all_runs(output_csv= ALL_RUNS_CSV,
                    only_experiment=args.experiment)

