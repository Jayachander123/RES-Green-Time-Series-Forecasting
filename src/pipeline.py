# src/pipeline.py
"""
Weekly Benefit-to-Cost replay driver *with*
  • verbose prints
  • defensive try/except blocks
  • MLflow experiment tracking
"""

from __future__ import annotations
import logging
import argparse, csv, json, sys, time, datetime as dt
import subprocess
from pathlib import Path
from types import SimpleNamespace
import importlib
import yaml
import mlflow
import pickle
import pandas as pd
import platform, psutil
# from functools import partial
import os
from src.utils.week_logger import WeekLogger 
from src.utils.logging_utils import make_logger
logger: logging.Logger | None = None
import tracemalloc
# from src.utils.metrics import calc_rmse
# from src.evaluate import _split_X_y      # used for RMSE calculation


# ── local helpers ────────────────────────────────────────────────
from src.features import weekly_univariate as feat_mod
from src.train     import train
from src.evaluate  import evaluate
from src import policy as policy_mod
from src.utils.emissions import track_emissions
import random, numpy as np          
from src.utils.metrics import save_summary
from src.utils.mlflow_utils import get_or_create_experiment_id



# ── util ─────────────────────────────────────────────────────────
def load_config(path: str | Path) -> dict:
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"✔  Config loaded   ({path})")
        return cfg
    except Exception as e:
        print(f"✖  Failed to read config: {e}")
        sys.exit(1)

# seed --------------------------------------------------------
def set_global_seed(seed: int | None):
    """Guarantee deterministic numpy / random / torch where available."""
    if seed is None:
        return
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        pass


def load_with_features(path: Path,
                       lags: list[int],
                       rolls: list[int]) -> pd.DataFrame:
    """
    Read a weekly series and add user-chosen lag / rolling-window features.
    The feature module (weekly_univariate.py) exposes only global constants,
    so we monkey-patch them *temporarily*.
    """
    print(f"▶  Reading data … {path}")
    try:
        df = (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path, parse_dates=["Week"])
        )
    except Exception as e:
        print(f"✖  Failed to read {path}: {e}")
        sys.exit(1)

    # ── save & patch ───────────────────────────────────────────
    backup = SimpleNamespace(LAGS=feat_mod.LAGS,
                             ROLL_WINDOWS=feat_mod.ROLL_WINDOWS)

    # (re-)import first, then patch  ← important
    importlib.reload(feat_mod)
    feat_mod.LAGS, feat_mod.ROLL_WINDOWS = tuple(lags), tuple(rolls)

    try:
        df_feat = feat_mod.add_features(df)
    except Exception as e:
        print(f"✖  Feature engineering failed: {e}")
        sys.exit(1)
    finally:
        # always restore defaults for the next call
        feat_mod.LAGS, feat_mod.ROLL_WINDOWS = backup.LAGS, backup.ROLL_WINDOWS
        importlib.reload(feat_mod)

    print(f"✔  Feature engineering done → {len(df_feat):,} rows")
    return df_feat



# ── core replay ──────────────────────────────────────────────────
def replay(cfg: dict) -> None:
    global logger
    # --------------------------------------------------------
    # A. reproducible randomness
    # --------------------------------------------------------
    seed = cfg["model"].get("seed", cfg.get("seed", 0))
    data_cfg, mdl_cfg, pol_cfg, out_cfg = (
        cfg["data"], cfg["model"], cfg["policy"], cfg["output"]
    )
    set_global_seed(seed)

    # 0. pricing helpers ------------------------------------------------
    pricing_cfg = cfg.get("pricing", {})
    price_model = pricing_cfg.get("model", "cpu_hour")
    price_vcpu_h = pricing_cfg.get("price_vcpu_h", 0.0311)
    usd_per_kwh  = pricing_cfg.get("usd_per_kwh", 0.12)
    overhead_usd = pricing_cfg.get("fixed_overhead_usd", 0.0)
    cpu_count    = psutil.cpu_count()

    def _cost_runtime(sec: float) -> float:
        return sec / 3600 * price_vcpu_h * cpu_count + overhead_usd

    def _cost_energy(kwh: float) -> float:
        return kwh * usd_per_kwh + overhead_usd

    price_fn = _cost_runtime if price_model == "cpu_hour" else _cost_energy

    # make policy & pick model kind -------------------------------------
    policy = policy_mod.make(pol_cfg)
    kind = mdl_cfg.pop("kind", "lightgbm")
    mdl_no_seed = {k: v for k, v in mdl_cfg.items() if k != "seed"}

    # --------------------------------------------------------
    # B. MLflow run
    # --------------------------------------------------------
    exp_name     = out_cfg.get("mlflow_experiment", "res_paper_grid")
    tracking_uri = "file:./mlruns"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    try:
        exp_id = str(get_or_create_experiment_id(exp_name, tracking_uri))
        mlflow.set_experiment(experiment_id=exp_id)         
    except mlflow.exceptions.MlflowException:
        # ID is stale for this store → fall back to the experiment *name*
        mlflow.set_experiment(exp_name)                      # creates if missing
        exp_id = client.get_experiment_by_name(exp_name).experiment_id

    run_name = out_cfg.get("run_name", f"replay_{Path(data_cfg['path']).stem}")

    with mlflow.start_run(run_name=run_name) as run:
        # mlflow.set_tag("git_commit",
        #                subprocess.check_output(["git", "rev-parse", "HEAD"])
        #                .strip().decode())
        mlflow.log_param("seed", seed)

        # 1. static params ---------------------------------------------
        dataset_name = Path(data_cfg["path"]).stem.split(".")[0].lower()
        static_params = {
            "history_weeks":   data_cfg["history_weeks"],
            "replay_weeks":    data_cfg["replay_weeks"],
            "lags":            data_cfg["lags"],
            "rolling_windows": data_cfg["rolling_windows"],
            **mdl_no_seed,
        }
        if "lambda" in pol_cfg:
            static_params["lambda"] = pol_cfg["lambda"]
        elif pol_cfg.get("kind") == "fixed":
            static_params["k"] = pol_cfg["k"]
        elif pol_cfg.get("kind") == "random":
            static_params["p"] = pol_cfg["p"]
        elif pol_cfg.get("kind") == "cara":          
            static_params["tau"] = pol_cfg["tau"]
        mlflow.log_params(static_params)
        mlflow.set_tag("host", platform.node())
        mlflow.set_tag("start_time_utc", dt.datetime.utcnow().isoformat()+"Z")
        mlflow.log_param("cpu_count", psutil.cpu_count())
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_kind", kind)
        mlflow.log_param("policy_kind", pol_cfg.get("kind", "res"))
        mlflow_run_id = run.info.run_id

        # paths / logger -----------------------------------------------
        out_dir = Path(out_cfg["dir"]) / mlflow_run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        week_log = WeekLogger(True, dir=out_dir)
        logger = make_logger(out_dir / "run.log",
                             level=logging.DEBUG if cfg.get("debug") else logging.INFO)
        global_log = Path("logs") / "all_runs.log"
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(global_log)
                   for h in logger.handlers):
            global_handler = logging.FileHandler(global_log, mode="a")
            global_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(global_handler)
        logger.info(f"Run directory: {out_dir}")
        track_co2 = cfg.get("track_emissions", True)
        t0_total = time.perf_counter()
        # --------------------------------------------------------
        # 3. data + features
        # --------------------------------------------------------
        df = load_with_features(Path(data_cfg["path"]),
                                data_cfg["lags"],
                                data_cfg["rolling_windows"])
        hist_n, replay_n = data_cfg["history_weeks"], data_cfg["replay_weeks"]

        if replay_n is None:
            replay_n = max(0, len(df) - hist_n)          # all remaining rows
        data_cfg["replay_weeks"] = replay_n 

        if len(df) < hist_n + replay_n:
            msg = (f"Dataset too short: need {hist_n+replay_n} rows "
                   f"but have {len(df)}.")
            logger.info(f"✖  {msg}")
            mlflow.log_param("replay_shrunk_to", len(df)-hist_n)
            replay_n = max(0, len(df) - hist_n)
            logger.info(f"⚠️  Shrinking replay_weeks to {replay_n}")
        train_df  = df.iloc[:hist_n].copy()
        replay_df = df.iloc[hist_n:hist_n + replay_n].copy()

        # --------------------------------------------------------
        # 4. initial model
        # --------------------------------------------------------
        logger.info(f"\n▶  Training initial model on {len(train_df):,} rows")

        with track_emissions(track_co2, out_dir) as eco:
            model, init_sec = train(train_df, kind, mdl_cfg)
            eco["cost_usd"] = price_fn(init_sec if price_model == "cpu_hour"
                                       else eco["kwh"])
        cand_prev = SimpleNamespace(model=model,
                                    train_sec=init_sec,
                                    eco=eco)
        policy.reset(train_df)
        mlflow.set_tag("promotion_delay_weeks", 1)
        mlflow.log_metric("init_train_sec", init_sec, step=0)
        mlflow.log_metric("init_kg_co2",    eco["kg_co2"], step=0)
        mlflow.log_metric("init_energy_kwh",eco["kwh"], step=0)
        mlflow.log_metric("init_cost_usd",  eco["cost_usd"], step=0)


        # --------------------------------------------------------
        # 5. accumulators (start at 0 to avoid double-count)
        # --------------------------------------------------------
        tot_benefit = 0.0
        eco_total = kwh_total = usd_total = 0.0
        mae_before_all, mae_after_all = [], []
        n_retrain = 0
        tot_cost   = 0

        # --------------------------------------------------------
        # 6. artefact paths
        # --------------------------------------------------------
        csv_path  = out_dir / out_cfg.get("log_csv",       "replay_log.csv")
        json_path = out_dir / out_cfg.get("summary_json",  "summary.json")

        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            # writer.writerow(["step","week","y_true","seed",
            #                  "err_before","err_after",
            #                  "benefit","train_sec",
            #                  "retrained","train_rows","lambda_auto"])
            writer.writerow(["step","week","y_true","seed",
                 "mae_before","mae_after",
                 "benefit_raw","benefit_norm","benefit",
                 "cost","retrained","train_rows","lambda_auto","tau"])


            logger.info("▶  Starting replay …")
            t_loop = time.perf_counter()

            for step in range(len(replay_df)):        # one calendar week
                this_df = replay_df.iloc[[step]]
                wk_this = this_df.iloc[0]["Week"]

                # 1. evaluate incumbent vs. buffered candidate ---------
                t_eval0 = time.perf_counter()
                mae_old = evaluate(model,            this_df, kind)
                mlflow.log_metric("eval_ms_old",
                  (time.perf_counter()-t_eval0)*1000, step=step)
                
                t_eval1 = time.perf_counter()
                mae_new = evaluate(cand_prev.model,  this_df, kind)
                mlflow.log_metric("eval_ms_new", (time.perf_counter()-t_eval1)*1000, step=step)
                
                benefit = mae_old - mae_new

                # store per-week errors for summary
                mae_before_all.append(mae_old)

                retrain = policy.should_retrain(
                                 benefit=benefit,
                                 cost_sec=cand_prev.train_sec,
                                 mae_old=mae_old,
                                 mae_new=mae_new,
                                 step_idx=step)
                mae_after_all.append(mae_new if retrain else mae_old)

                # 2. accounting: always add candidate cost -------------
                tot_cost  += cand_prev.train_sec
                eco_total += cand_prev.eco["kg_co2"]
                kwh_total += cand_prev.eco["kwh"]
                usd_total += cand_prev.eco["cost_usd"]

                benefit_log = benefit if retrain else 0.0

                if retrain:
                    model = cand_prev.model
                    n_retrain += 1
                    tot_benefit += benefit_log
                
                logged_sec  = cand_prev.train_sec
                logged_eco  = cand_prev.eco 
                

                # 3. train new candidate for next week ----------------
                train_data = pd.concat([train_df, this_df])
                with track_emissions(track_co2, out_dir) as eco:
                    if not tracemalloc.is_tracing():
                        tracemalloc.start()
                    new_model, train_sec = train(train_data, kind, mdl_cfg)
                    tracemalloc.stop()
                    eco["cost_usd"] = price_fn(train_sec if price_model == "cpu_hour"
                                               else eco["kwh"])
                cand_prev = SimpleNamespace(model=new_model,
                                            train_sec=train_sec,
                                            eco=eco)

                # 4. grow history window ------------------------------
                train_df = pd.concat([train_df, this_df])

                # guard-rail: no look-ahead
                assert wk_this not in replay_df.iloc[step+1:]["Week"].values, \
                       "Look-ahead detected!"

                # 5. external logs ------------------------------------
                week_log.log(run_id=run.info.run_id,
                             week=str(wk_this),
                             date=str(wk_this.date()),
                             trained=int(retrain),
                             mae=mae_new if retrain else mae_old,
                             cost=logged_sec)
                
                if retrain:
                    benefit_raw  = mae_old - mae_new
                    benefit_norm = benefit_raw / max(mae_old, 1e-9)
                else:
                    benefit_raw  = 0.0
                    benefit_norm = 0.0
                benefit_csv = benefit_norm if retrain else 0.0
                cost_legacy  = logged_sec  if retrain else 0.0   # 0 s on skip weeks


                writer.writerow([
                                step + 1,
                                wk_this.date(),
                                int(this_df.iloc[0]["y"]),
                                seed,
                                mae_old,
                                mae_new if retrain else mae_old,
                                benefit_raw,
                                benefit_norm,
                                benefit_csv,
                                cost_legacy,
                                int(retrain),
                                len(train_df),
                                getattr(policy, "lam", None),
                                getattr(policy, "tau", None),
                            ])



                logger.info(f"  step {step+1:3d}  {wk_this.date()}  "
                            f"{'retrain ✅' if retrain else 'skip – '}  "
                            f"benefit={benefit_log:+.4f}  "
                            f"cost={logged_sec:.2f}s")

                # MLflow metrics
                mlflow.log_metric("err_before",  mae_old,   step=step)
                mlflow.log_metric("err_after",   mae_new if retrain else mae_old, step=step)
                mlflow.log_metric("benefit",     benefit_log, step=step)
                mlflow.log_metric("train_sec",   logged_sec,  step=step)
                mlflow.log_metric("kg_co2",      logged_eco["kg_co2"], step=step)
                mlflow.log_metric("energy_kwh",  logged_eco["kwh"],    step=step)
                mlflow.log_metric("cost_usd",    logged_eco["cost_usd"],step=step)
                mlflow.log_metric("retrained",   int(retrain),          step=step)
                mlflow.log_metric("cost", cost_legacy, step=step)          # legacy alias
                mlflow.log_metric("benefit_norm", benefit_norm, step=step) # legacy alias


                try:
                    mlflow.log_metric("train_sec_next", train_sec, step=step)
                except Exception:
                    pass       # training already failed & raised earlier



            # --------------------------------------------------------
            # 7. summary & artefacts
            # --------------------------------------------------------
            loop_sec  = time.perf_counter() - t_loop
            total_sec = time.perf_counter() - t0_total

            summary = {
                "dataset":          data_cfg["path"],
                "history_weeks":    hist_n,
                "replay_weeks":     replay_n,
                "lags":             data_cfg["lags"],
                "roll_windows":     data_cfg["rolling_windows"],
                "init_train_sec":   init_sec,
                "total_retrain_sec":tot_cost,
                "total_runtime_sec":total_sec,
                "n_retrains":       n_retrain,
                "mean_mae_before":  float(pd.Series(mae_before_all).mean()),
                "mean_mae_after":   float(pd.Series(mae_after_all).mean()),
                "total_benefit":    tot_benefit,
                "timestamp":        dt.datetime.utcnow().isoformat() + "Z",
                "total_kg_co2":     eco_total,
                "total_energy_kwh": kwh_total,
                "total_cost_usd":   usd_total,
                "seed":             seed,
                "res":              tot_benefit / max(tot_cost, 1e-6),
            }
            if "lambda" in pol_cfg or "lam" in pol_cfg:
                summary["lambda"] = pol_cfg.get("lambda", pol_cfg.get("lam"))
            elif pol_cfg.get("kind") == "fixed":
                summary["k"] = pol_cfg["k"]
            elif pol_cfg.get("kind") == "random":
                summary["p"] = pol_cfg["p"]
            elif pol_cfg.get("kind") == "cara":    
                summary["tau"] = pol_cfg["tau"]

            with open(json_path, "w") as jf:
                json.dump(summary, jf, indent=2)

            # final MLflow artefacts / metrics ----------------------
            mlflow.log_metric("res", summary["res"])
            mlflow.log_metric("total_benefit", tot_benefit)
            mlflow.log_metric("total_retrain_sec", tot_cost)
            mlflow.log_metric("total_runtime_sec", total_sec)
            mlflow.log_metric("total_kg_co2", eco_total)
            mlflow.log_metric("total_cost_usd", usd_total)
            mlflow.log_metric("mean_mae_before", summary["mean_mae_before"])
            mlflow.log_metric("mean_mae_after",  summary["mean_mae_after"])
            mlflow.log_artifact(json_path)

            # save final model --------------------------------------
            try:
                model_path = out_dir / f"{kind}_final_{run.info.run_id}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path)
                mlflow.log_metric("model_size_mb",
                                  model_path.stat().st_size / 1_048_576)
            except Exception as e:
                logger.warning(f"Model not pickled: {e}")


            week_log.flush()
            logger.info("▶  Replay finished "
                        f"(runtime {total_sec:.1f}s, loop {loop_sec:.1f}s)")
        mlflow.log_artifact(csv_path)
        logger.info("▶  Replay File Logged to Mlflow.....! ")

        


# ── CLI entry-point ───────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Weekly RES replay driver")
    parser.add_argument("--config", required=True,
                        help="Path to YAML file with parameters")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg["data"]["history_weeks"] = 10
        cfg["data"]["replay_weeks"]  = 3
    replay(cfg)


if __name__ == "__main__":
    _cli()
