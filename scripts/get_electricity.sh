#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw

echo "⇣ Downloading Australian electricity load data (UCI)..."
DATA_URL="https://zenodo.org/records/16627581/files/electricity_weekly.csv?download=1"

curl -L -o data/raw/electricity_weekly_raw.csv "$DATA_URL"

echo "✓ Downloaded raw electricity data to data/raw/electricity_load.zip"