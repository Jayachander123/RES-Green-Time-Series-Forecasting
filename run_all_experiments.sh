#!/usr/bin/env bash
# ===============================================================
# run_all_experiments.sh   
# ===============================================================
set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------
# 0.  Thread & MLflow environment (add once, here)
# ---------------------------------------------------------------
export OMP_NUM_THREADS=1          # OpenMP (NumPy, SciPy)
export MKL_NUM_THREADS=1          # Intel MKL
export NUMEXPR_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1   # Apple Accelerate
export LGBM_NUM_THREADS=1         # LightGBM
export MLFLOW_EXPERIMENT_NAME="res_paper_grid"

# ---------------------------------------------------------------
# user settings
# ---------------------------------------------------------------
CONFIG_DIR="config"            # folder that contains YAMLs
OUT_BASE="outputs"             
WEEK_PARQ="${OUT_BASE}/week_level_all.parquet"
PYTHON="python"                
MAX_WORKERS=27 
SEED_LIST=(0 27 45 2024 2025)

BACKUP_DIR="backups"           
mkdir -p "$BACKUP_DIR"       

TS=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------
# helper: archive a directory if non-empty, then recreate it
# ---------------------------------------------------------------
archive_if_present () {
  local src="$1" ; local tag="$2"

  if [[ -d "$src" && $(du -s "$src" | cut -f1) -ne 0 ]]; then
      local dest="${BACKUP_DIR}/$(basename "$src")_${TS}"
      mv "$src" "$dest"
      echo "ðŸ§¹  archived $tag â†’ $dest/"
  fi
  mkdir -p "$src"              # recreate empty original
}

# ---------------------------------------------------------------
# clean-slate step
# ---------------------------------------------------------------
# archive_if_present "mlruns"      "MLflow store"
# archive_if_present "${OUT_BASE}" "run artefacts"
# archive_if_present "figures"     "figures"

# ---------------------------------------------------------------
# discover YAMLs
# ---------------------------------------------------------------
shopt -s nullglob
YAMLS=("${CONFIG_DIR}"/*.yaml)
shopt -u nullglob
[[ ${#YAMLS[@]} -gt 0 ]] || { echo "âŒ  no YAMLs in ${CONFIG_DIR}"; exit 1; }

# ---------------------------------------------------------------
# read the MLflow experiment name from the *first* YAML
# ---------------------------------------------------------------
FIRST_YAML="${YAMLS[0]}"
EXPERIMENT_NAME=$(
  $PYTHON - <<PY
import yaml, pathlib, sys
with open("${FIRST_YAML}") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("output", {}).get("mlflow_experiment", "default"), end="")
PY
)
echo "   MLflow experiment name: '${EXPERIMENT_NAME}'"

# ---------------------------------------------------------------
# run experiments (week logger enabled by default)
# ---------------------------------------------------------------
for CFG in "${YAMLS[@]}"; do
    name=$(basename "$CFG")
    if [[ "$name" == *base* ]]; then
        echo "🚀  λ-grid   ➜  $name"
        $PYTHON src/run_model_grid.py \
                --configs "$CFG" \
                --max_workers "$MAX_WORKERS" \
                --seeds "${SEED_LIST[@]}"
    else
        echo "ðŸš€  baseline âžœ  $name"
        $PYTHON -m src.pipeline --config "$CFG"
    fi
done
echo "  all experiments finished"

# ---------------------------------------------------------------
# aggregate week_metrics.parquet â†’ one Parquet
# ---------------------------------------------------------------
echo "  aggregating weekly logs"
$PYTHON -m scripts.export_decisions

# ---------------------------------------------------------------
# build all figures (run-level + week-level)
# ---------------------------------------------------------------
echo "  making paper figures"
$PYTHON -m scripts.make_res_plots 

# ---------------------------------------------------------------
# paired t-test & confidence interval
# ---------------------------------------------------------------
echo "  paired t-test"
$PYTHON -m scripts.stats_paired_test 

# ---------------------------------------------------------------
# paired t-test & confidence interval
# ---------------------------------------------------------------
echo "  Wilcoxon Summary table"
$PYTHON -m scripts.create_summary 

# ---------------------------------------------------------------
# recap
# ---------------------------------------------------------------
echo
echo "  Done!"
echo "     MLflow runs     : ./mlruns/"
echo "     Aggregated weeks: ${WEEK_PARQ}"
echo "     Figures         : ./figures/"
echo "      Stats table     : tables/wilcoxon_grid.csv""