 # Reproducibility Package for "Retraining-Efficiency Score (RES)"

This package contains the code and data necessary to reproduce the results in our paper.

We provide two primary ways to run the experiments:

1. **Quick Sanity Check:** A fast, lightweight run on a single dataset that validates the entire pipeline in \~15 minutes on a standard laptop. **(Recommended for reviewers)**.

2. **Full Experiment Suite:** The complete set of 1920 experiments, which requires a powerful machine (e.g., 224-vCPU cloud VM) and 5+ hours.

---

## Hardware & Software Requirements

* **OS:** Linux / macOS
* **Python:** 3.10+
* **Hardware (Sanity Check):** Standard laptop with at least 8GB RAM.
* **Hardware (Full Run):** The original experiments were run on a 224-vCPU Google Cloud VM (`n2d-highcpu-224`).

---

## 1. Installation

**Option A: Using Docker (Recommended for Reproducibility)**

```bash
# Build the Docker image
docker build -t res_paper .

# Run Sanity Check (Linux/macOS)
docker run --rm \
  -v "$(pwd)/sanity_check/figures":/app/sanity_check/figures \
  -v "$(pwd)/sanity_check/tables":/app/sanity_check/tables \
  -v "$(pwd)/sanity_check/mlruns":/app/mlruns \
  -w /app \
  res_paper \
  bash ./quick_sanity_check.sh

# (Optional) Run Sanity Check (Linux/macOS) and capture logs for verification:
docker run --rm \
  -v "$(pwd)/sanity_check/figures":/app/sanity_check/figures \
  -v "$(pwd)/sanity_check/tables":/app/sanity_check/tables \
  -v "$(pwd)/sanity_check/mlruns":/app/mlruns \
  -w /app \
  res_paper \
  bash -x quick_sanity_check.sh > sanity_trace.log 2>&1


# Run Sanity Check (Windows CMD)
docker build -t res_paper .
docker run --rm -v "%cd%/sanity_check/figures:/app/sanity_check/figures" -v "%cd%/sanity_check/tables:/app/sanity_check/tables" res_paper bash ./quick_sanity_check.sh

# Run Sanity Check (Windows PowerShell)
docker build -t res_paper .
docker run --rm -v "${PWD}/sanity_check/figures:/app/sanity_check/figures" -v "${PWD}/sanity_check/tables:/app/sanity_check/tables" res_paper bash ./quick_sanity_check.sh
```


**Option B: Using virtual environment + pip**

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # For macOS/Linux
# venv\Scripts\activate   # For Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make the script executable
chmod +x quick_sanity_check.sh

# 4. Run the quick check
./quick_sanity_check.sh
```

### 🍎 macOS Users Note
If you are running the experiments locally (without Docker) on macOS, you must install `libomp` (OpenMP) for LightGBM to work.

```bash
brew install libomp
```
---


## 2. Full Experiment Suite (1920 Experiments)

Run the entire suite of 1920 model retraining experiments, generating all figures and statistical tables from the paper.

**Option A: Using Docker (Recommended for Reproducibility)**

#### 💻 Linux / macOS

```bash
# 1. Build the Docker image
docker build -t res_paper .

# 2. Run the full experiment suite
docker run --rm -it \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/figures:/app/figures" \
  -v "$(pwd)/tables:/app/tables" \
  res_paper \
  bash ./run_all_experiments.sh
```
```bash
# (Optional) 2.1 Run Full experiment (Linux/macOS) and capture logs for verification:

docker run --rm -it \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/figures:/app/figures" \
  -v "$(pwd)/tables:/app/tables" \
  res_paper \
  bash -x ./run_all_experiments.sh > run_trace.log 2>&1
```


#### 🪟 Windows Command Prompt (CMD)

```cmd
:: 1. Build the Docker image
docker build -t res_paper .

:: 2. Run the full experiment suite
docker run --rm -it ^
  -v "%cd%/outputs:/app/outputs" ^
  -v "%cd%/mlruns:/app/mlruns" ^
  -v "%cd%/figures:/app/figures" ^
  -v "%cd%/tables:/app/tables" ^
  res_paper ^
  bash ./run_all_experiments.sh
```

#### 🪟 Windows PowerShell

```powershell
# 1. Build the Docker image
docker build -t res_paper .

# 2. Run the full experiment suite
docker run --rm -it `
  -v "${PWD}/outputs:/app/outputs" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -v "${PWD}/figures:/app/figures" `
  -v "${PWD}/tables:/app/tables" `
  res_paper `
  bash ./run_all_experiments.sh
```

**Option B: Using virtual environment + pip**

The script below performs a clean-slate run, executes all experiment combinations, generates logs, figures, and statistical summaries.

```bash
# 1. Make it executable
chmod +x run_all_experiments.sh

# 2. Run all experiments
./run_all_experiments.sh
```

This will:

* Run 1920 experiments
* Log to `mlruns/`
* Store weekly metrics in `outputs/*/*/week_level_all.parquet`
* Save visualizations to `figures/`
* Save statistical tests to `tables/`

**Expected runtime:** \~5+ hours on a high-core machine (224 GB RAM vCPU) with 27 MAX_WORKERS.

---

## 3. Data Availability

* **Processed Data:** Included in this package for convenience.
* **Raw Data:** Also available at \[Zenodo DOI / Link] (will be provided upon publication). If strict reproducibility from raw is needed, run: 

```bash
bash ./scripts/fetch_and_prepare.sh
```

## 4. Run Everything - optional

* **The run_everything.sh script:** is the recommended entry point for a clean-slate reproduction. It automatically handles Phase 1 (Data Fetching & Prep) and Phase 2 (Experiment Execution).

```bash

# 1. Make the master script executable
chmod +x run_everything.sh

# 2. Run the full pipeline (Data + Experiments)
./run_everything.sh
```

This will:

1. Download and Prepare Data: Runs scripts/fetch_and_prepare.sh to ensure datasets are ready.
2. Run 1920 Experiments: Executes the full grid search.
3. Generate Artifacts: Logs to mlruns/, aggregates results to outputs/, and saves figures to figures/.
4. Expected runtime: ~5+ hours on a high-core machine (224 vCPU) with 27 MAX_WORKERS.


---

## Outputs

| Type               | Location                                |
| ------------------ | --------------------------------------- |
| MLflow             | `./mlruns/`                             |
| Logs               | `./outputs/`                            |
| Figures            | `./figures/` or `sanity_check/figures/` |
| Tables             | `./tables/` or `sanity_check/tables/`   |
| All mlflow runs    | `figures/all_mlflow_runs.csv` or  `sanity_check/figures/all_mlflow_runs.csv` |
| All decisions runs | `figures/all_decisions.csv` or  `sanity_check/figures/all_decisions.csv` |

---

## Notes for Reviewers

* If you're short on time, use the **Quick Sanity Check** only.
* If you want to validate robustness, run the **Full Experiment Suite**.
* For any issues, please run the relevant command (mentioned above to capture and verify logs) and refer to `sanity_trace.log` or contact the authors.

---

Thank you for reviewing our submission!
