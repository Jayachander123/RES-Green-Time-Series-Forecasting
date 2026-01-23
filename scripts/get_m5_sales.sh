#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw/m5

echo "⇣ Downloading M5 sales data from Zenodo..."

DATA_URL="https://zenodo.org/records/16627920/files/sales_train_evaluation.csv?download=1"

curl -L -o data/raw/sales_train_evaluation.csv "$DATA_URL"

echo "✓ Downloaded M5 raw files to data/raw/sales_train_evaluation.csv"


