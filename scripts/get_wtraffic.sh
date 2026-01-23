#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw

echo "⇣ Downloading Wikipedia traffic Zenodo"
DATA_URL="https://zenodo.org/records/16627671/files/wiki_weekly.csv?download=1"

curl -L -o data/raw/wiki_weekly_raw.csv "$DATA_URL"

echo "⇣ Wikipedia traffic Downloaded"
