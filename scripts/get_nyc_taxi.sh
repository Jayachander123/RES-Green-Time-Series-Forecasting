#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw

echo "⇣ Downloading pre-generated NYC Taxi daily ride counts (2019-2021)..."

DATA_URL="https://zenodo.org/records/16626495/files/nyc_taxi_daily.csv?download=1"

curl -L -o data/raw/nyc_taxi_daily.csv "$DATA_URL"

echo "✓ Wrote daily NYC taxi counts to data/raw/nyc_taxi_daily.csv"

