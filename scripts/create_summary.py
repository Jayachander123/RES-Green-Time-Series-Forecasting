import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_DIR  = "tables"
DEFAULT_CSV_PATH = Path("tables") / "wilcoxon_grid.csv"
SANITY_CSV_PATH  = Path("sanity_check") / "tables" / "wilcoxon_grid.csv"
ALPHA = 0.05

def create_summary_table(filename: str, out_put:str):
    """
    Reads a Wilcoxon test results CSV and generates a 'Win-Loss-Tie' summary table.
    """
    print(f"▶️  Reading data from '{filename}'...")
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found. Please make sure '{filename}' is in the same directory.")
        return

    # --- 1. Group Baselines ---
    def get_baseline_group(baseline_name):
        if 'FIXED' in baseline_name: return 'Fixed-k'
        if 'RANDOM' in baseline_name: return 'Random-p'
        if 'CARA' in baseline_name: return 'CARA'
        if 'page_hinkley' in baseline_name.lower() or 'ph' in baseline_name.lower(): return 'Page-Hinkley'
        return baseline_name

    df['baseline_group'] = df['baseline'].apply(get_baseline_group)

    # --- 2. Defined Win-Loss-Tie Logic ---
    # A 'Win' for RES is when its error is lower, meaning delta_mean > 0.
    # The 'delta_mean' column is 'mean_base - mean_res'.
    
    conditions = [
        (df['p_value'] < ALPHA) & (df['delta_mean'] > 0),  # RES Win: p-value significant AND RES error is lower
        (df['p_value'] < ALPHA) & (df['delta_mean'] < 0),  # RES Loss: p-value significant AND RES error is higher
    ]

    choices = ['RES Win', 'RES Loss']
    df['result'] = np.select(conditions, choices, default='Tie')

    print("✔  Win-Loss-Tie results calculated with corrected logic.")

    # --- 3. Create and Polish the Summary Table ---
    summary_table = pd.crosstab(df['baseline_group'], df['result'])
    
    desired_columns = ['RES Win', 'RES Loss', 'Tie']
    for col in desired_columns:
        if col not in summary_table.columns:
            summary_table[col] = 0
    summary_table = summary_table[desired_columns]
    summary_table.to_csv(out_put)
    
    
    # --- 4. Print Results and LaTeX Code ---
    print("\n" + "="*40)
    print("      CORRECTED SUMMARY RESULTS")
    print("="*40)
    print(summary_table)
    print("="*40)

def main():
    ap = argparse.ArgumentParser(
            description="Generate paper tables or a minimal sanity set")
    ap.add_argument("--minimal", action="store_true",
                    help="generate tables in sanity_check folder")
    args = ap.parse_args()
    do_minimal = args.minimal
    if do_minimal:
        ROOT = Path(__file__).resolve().parents[1]
        ROOT_TABLE = Path(os.getenv("TABLES_DIR",  "sanity_check/tables"))
        OUT_CSV = ROOT_TABLE/"wilcoxon_summary.csv"
        
        CSV_FILENAME = SANITY_CSV_PATH
        out_put = OUT_CSV
    else:
        ROOT_TABLE = Path(os.getenv("TABLES_DIR",  "tables"))
        OUT_CSV = ROOT_TABLE/"wilcoxon_summary.csv"
        CSV_FILENAME = DEFAULT_CSV_PATH
        out_put = OUT_CSV
    create_summary_table(filename=CSV_FILENAME, out_put=out_put)
 

if __name__ == '__main__':
    main()

   