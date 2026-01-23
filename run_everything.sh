#!/usr/bin/env bash
###############################################################################
# run_everything.sh ─ single entry point to reproduce the paper end-to-end.
#
# Phases
#   1. Data acquisition & preparation      → scripts/fetch_and_prepare.sh
#   2. Training, evaluation, statistics    → run_all_experiments.sh
###############################################################################
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

FETCH_SCRIPT="${ROOT_DIR}/scripts/fetch_and_prepare.sh"
EXP_SCRIPT="${ROOT_DIR}/run_all_experiments.sh"

if [[ ! -x "${FETCH_SCRIPT}" || ! -x "${EXP_SCRIPT}" ]]; then
  echo "  Error: required scripts are missing or not executable:" >&2
  echo "   ${FETCH_SCRIPT}" >&2
  echo "   ${EXP_SCRIPT}"  >&2
  exit 1
fi

echo "================================================================="
echo " STARTING FULL REPRODUCTION PIPELINE"
echo "================================================================="
echo

# ────────────────────────────────────────────────────────────────
echo "STARTING DATA ACQUISITION AND PREPARATION ---"
bash "${FETCH_SCRIPT}"
echo "  DATA ACQUISITION AND PREPARATION  COMPLETE."
echo

# ────────────────────────────────────────────────────────────────
echo "STARTING EXPERIMENT EXECUTION AND ANALYSIS ---"
bash "${EXP_SCRIPT}"
echo "EXPERIMENT EXECUTION AND ANALYSIS  COMPLETE."
echo

echo "================================================================="
echo "Done!  The entire pipeline finished successfully."
echo "================================================================="
