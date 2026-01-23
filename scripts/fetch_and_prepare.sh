#!/usr/bin/env bash

set -euo pipefail

# --- Download all raw data ---
echo "--- Starting Data Download ---"
bash scripts/get_nyc_taxi.sh
bash scripts/get_m5_sales.sh
bash scripts/get_electricity.sh
bash scripts/get_wtraffic.sh #
echo "--- All raw data downloaded to data/raw/ ---"
echo ""

# --- Process all raw data into clean, weekly format ---
echo "--- Starting Data Processing ---"
python -m src.etl --raw data/raw/nyc_taxi_daily.csv --out data/processed/nyc_taxi_clean

python -m src.etl --raw data/raw/sales_train_evaluation.csv --out data/processed/m5_sales_clean

python -m src.etl --raw data/raw/electricity_weekly_raw.csv --out data/processed/electricity_weekly

python -m src.etl --raw data/raw/wiki_weekly_raw.csv --out data/processed/wiki_weekly
echo "--- All processed data saved to data/processed/ ---"
echo ""

echo "✅ Data preparation complete."
