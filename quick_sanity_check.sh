#!/usr/bin/env bash
###############################################################################
# quick_sanity_check.sh – ≤15-min end-to-end run for reviewers
###############################################################################
set -euo pipefail
IFS=$'\n\t'

echo "────────────────────────────────────────────────────────"
echo " 🚀  Quick Sanity Check (wall-time ≈ 15 min)"
echo "────────────────────────────────────────────────────────"

# ── 0. Folder layout ────────────────────────────────────────────────────────
BOX="sanity_check"            # all artefacts live under this folder
mkdir -p "$BOX"
mkdir -p logs

export OUTPUT_BASE_DIR="$BOX/outputs"
export FIGURES_DIR="$BOX/figures"
export TABLES_DIR="$BOX/tables"
export TMP_CFG_DIR="$BOX/tmp_cfg"


for d in "$OUTPUT_BASE_DIR" "$FIGURES_DIR" "$TABLES_DIR" "$TMP_CFG_DIR"; do
    rm -rf "$d" && mkdir -p "$d"
done

# ── 1. Thread limits (same as full run) ─────────────────────────────────────
for v in OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_MAX_THREADS \
         OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS LGBM_NUM_THREADS
do  export $v=1; done
export LGBM_NUM_THREADS=4

# ── 2. MLflow ─ same store (./mlruns/) but different experiment name ────────
rm -f .mlflow_experiment_id               # delete cached numeric ID
export MLFLOW_EXPERIMENT_NAME="sanity_check"

# ── 3. Miniature grid: ONE LightGBM RES run only ────────────────────────────
echo "▶ Running miniature experiment …"
python src/run_model_grid.py \
       --configs   config/m5_sales_base.yaml \
       --models    lightgbm \
       --seeds     0 \
       --res_lambdas 0.3 \
       --fixed_every   4 \
       --random_prob   0.15 \
       --cara_tau      0.03 \
       --ph_params     '0.005,50,0.99' \
       --experiment "$MLFLOW_EXPERIMENT_NAME" \
       --max_workers 4

# ── 4. Post-processing (same sequence as full run) ──────────────────────────
echo "▶ Aggregating weekly logs";  python -m scripts.export_decisions sanity_check
echo "▶ Generating figures";       python -m scripts.make_res_plots --minimal
echo "▶ Statistical tests";        python -m scripts.stats_paired_test --minimal --datasets m5_sales_clean

# ── 5. Recap ────────────────────────────────────────────────────────────────
echo
echo "🎉  Sanity run finished"
echo "────────────────────────────────────────────────────────"
echo " MLflow experiment : sanity_check   (stored in ./mlruns/)"
echo " Outputs           : $OUTPUT_BASE_DIR/"
echo " Figures           : $FIGURES_DIR/"
echo " Tables            : $TABLES_DIR/"
echo "────────────────────────────────────────────────────────"
