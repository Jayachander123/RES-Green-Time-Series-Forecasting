#!/usr/bin/env python
"""
Flexible launcher for
  • many base-dataset YAMLs
  • multiple model kinds
  • multiple retraining policies
      – RES with a λ grid
      – Fixed-k   (every k weeks)
      – Random-p  (retrain with prob. p)

Example
--------
python scripts/run_model_grid.py \
    --configs  config/m5_sales_base.yaml config/nyc_taxi_base.yaml \
    --models   lightgbm prophet arima \
    --res_lambdas   0 0.001 0.01 0.1 0.3 1 \
    --fixed_every   4 12 \
    --random_prob   0.05 0.30 \
    --experiment res_paper_grid \
    --max_workers 4
"""

from __future__ import annotations
import argparse, yaml, copy, itertools, subprocess, uuid, pathlib as P, os
from concurrent.futures import ProcessPoolExecutor

# ----------------------------------------------------------------------
# 1.  Parameter space defaults
# ----------------------------------------------------------------------
DEFAULT_KINDS       = ["arima", "lightgbm", "prophet", "transformer"]
# DEFAULT_KINDS       = ["arima"]
# DEFAULT_RES_LAMBDAS = [0, 3e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.6, 0.7, 1]
DEFAULT_RES_LAMBDAS = [0, 1e-2, 3e-2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1]
DEFAULT_FIXED_EVERY  = [4, 8, 12] 
DEFAULT_RANDOM_PROB  = [0.15, 0.05, 0.30] 
DEFAULT_PH_PARAMS = ["0.005,50,0.99"]
DEFAULT_SEEDS        = [0, 27, 45, 2024, 2025]
DEFAULT_CARA_TAU = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.9, 1]

REPO_ROOT   = P.Path(__file__).resolve().parents[1]
TMP_CFG_DIR = P.Path(os.getenv("TMP_CFG_DIR",
                               REPO_ROOT / "tmp_cfg"))
TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = P.Path(__file__).resolve().parents[1]
OUTPUT_BASE_DIR = os.getenv("OUTPUT_BASE_DIR", "outputs")


# ----------------------------------------------------------------------
# 2.  Helpers
# ----------------------------------------------------------------------
def _make_policy_dict(kind: str,
                      lam: float | None = None,
                      k: int | None = None,
                      p: float | None = None) -> tuple[dict, str]:
    """
    Return (policy_subdict, suffix_for_run_name)
    """
    # if kind == "res":
    #     return ({"kind": "res", "lambda": lam, "auto": True,
    if kind == "res":
        return ({"kind": "res", "lambda": lam, "auto": False,
                 "normalise_benefit": True,
                 "simulate_full_cost": True},
                f"res{lam}")
    if kind == "fixed":
        return ({"kind": "fixed", "k": k},      f"fixed{k}")
    if kind == "random":
        return ({"kind": "random", "p": p},     f"rnd{p}")
        
    raise ValueError(f"Unknown policy kind {kind}")

def patch_cfg(base: dict, model_kind: str, policy_cfg: dict,
              run_suffix: str, experiment: str) -> dict:
    """Return a *new* config dict with model & policy injected."""
    cfg = copy.deepcopy(base)

    # dataset name for pretty run names
    dataset_name = P.Path(cfg["data"]["path"]).stem.split("_clean")[0]

    # vary model + policy
    cfg["model"]["kind"] = model_kind
    cfg["policy"]        = policy_cfg         

    # unique run metadata
    run_uid   = uuid.uuid4().hex[:8]
    run_name  = (f"{dataset_name}_{model_kind}_{run_suffix}"
                 f"_s{cfg['model']['seed']}_{run_uid}")
    out_dir = f"{OUTPUT_BASE_DIR}/{run_name}"
    
    cfg["output"]["dir"]               = out_dir
    cfg["output"]["run_name"]          = run_name
    cfg["output"]["mlflow_experiment"] = experiment

    # (optional) descriptive MLflow tags
    cfg.setdefault("output", {}).setdefault("mlflow_tags", {})
    cfg["output"]["mlflow_tags"] |= {
        "dataset": dataset_name,
        "model":   model_kind,
        "policy":  run_suffix,
    }
    return cfg

def launch(cfg_dict: dict):
    """Write YAML to a temp file and call the pipeline once."""
    tmp_path = TMP_CFG_DIR / f"{uuid.uuid4().hex}.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    cmd = ["python", "-m", "src.pipeline", "--config", str(tmp_path)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    # To keep every temp cfg for provenance, uncomment:
    # tmp_path.rename(tmp_path.with_suffix(".used.yaml"))

# ----------------------------------------------------------------------
# 3.  CLI
# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, nargs="+",
                    help="Base YAML(s) – one per dataset.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_KINDS)

    # three policy families
    ap.add_argument("--res_lambdas", type=float, nargs="*",
                    default=DEFAULT_RES_LAMBDAS)
    ap.add_argument("--fixed_every", type=int,   nargs="*", default=DEFAULT_FIXED_EVERY)
    ap.add_argument("--random_prob", type=float, nargs="*", default=DEFAULT_RANDOM_PROB)

    ap.add_argument("--experiment", default="res_paper_grid")
    ap.add_argument("--max_workers", type=int, default=1,
                    help=">1 to run jobs in parallel")
    ap.add_argument("--ph_params", nargs="*", default=DEFAULT_PH_PARAMS,
                help="delta,lambd,alpha triples, e.g. 0.005,50,0.99 0.010,30,0.95")
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS,
                    help="Each value is written to cfg['model']['seed']")
    ap.add_argument("--cara_tau", type=float, nargs="*", default=DEFAULT_CARA_TAU)
    return ap.parse_args()

# ----------------------------------------------------------------------
# 4.  Main driver
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Build the Cartesian product of jobs
    # ------------------------------------------------------------------
    jobs: list[dict] = []

    # Pre-load all base YAMLs once
    args.configs = [p for p in args.configs if p.strip()]
    base_cfgs = {p: yaml.safe_load(open(p)) for p in args.configs}

    for yaml_path, model_kind, seed in itertools.product(
            args.configs, args.models, args.seeds):
        base = base_cfgs[yaml_path]
        base = copy.deepcopy(base)          
        base.setdefault("model", {})["seed"] = seed

        # res λ grid
        for lam in args.res_lambdas:
            #1) fixed-λ grid
            pol_dict, suffix = _make_policy_dict("res", lam=lam)
            jobs.append(patch_cfg(base, model_kind, pol_dict,
                                  suffix, args.experiment))
        # 2) single auto-λ variant  ───────────────────
        pol_auto = {"kind": "res", "auto": True,
                    "normalise_benefit": True,
                    "simulate_full_cost": True}
        jobs.append(patch_cfg(base, model_kind, pol_auto,
                              "res_auto", args.experiment))


        # Fixed-k
        for k in args.fixed_every:
            pol_dict, suffix = _make_policy_dict("fixed", k=k)
            jobs.append(patch_cfg(base, model_kind, pol_dict,
                                  suffix, args.experiment))

        # Random-p
        for p in args.random_prob:
            pol_dict, suffix = _make_policy_dict("random", p=p)
            jobs.append(patch_cfg(base, model_kind, pol_dict,
                                  suffix, args.experiment))
        #cara-t
        for tau in args.cara_tau:
            pol_dict = {"kind": "cara", "tau": tau}
            suffix   = f"cara{tau}"
            jobs.append(patch_cfg(base, model_kind, pol_dict, suffix, args.experiment))

        # never retrain
        for policy in ("never",):
            pol_dict = {"kind": policy}
            jobs.append(patch_cfg(base, model_kind, pol_dict, policy, args.experiment))

        # always retrain
        for policy in ("always",):
            pol_dict = {"kind": policy}
            jobs.append(patch_cfg(base, model_kind, pol_dict, policy, args.experiment))

        

        # ph_param_list = args.ph_params or [
        #     f"{d},{l},{a}" for (d, l, a) in DEFAULT_PH_PARAMS
        # ]
        ph_param_list = args.ph_params or DEFAULT_PH_PARAMS

        for triplet in ph_param_list:
            d, l, a = map(float, triplet.split(","))
            pol_dict = {"kind": "ph", "delta": d, "lambd": l, "alpha": a}
            suffix   = f"ph{d}_{l}_{a}"
            jobs.append(
                patch_cfg(base, model_kind, pol_dict, suffix, args.experiment)
            )

    print(f"Prepared {len(jobs)} jobs.")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    if args.max_workers == 1:
        for cfg in jobs:
            launch(cfg)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            ex.map(launch, jobs)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
