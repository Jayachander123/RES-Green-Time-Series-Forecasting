# Datasets

This directory contains the raw data used in our study.

## NYC Taxi Daily Rides (`nyc_taxi_daily.csv`)

This file contains the total number of yellow cab rides per day from 2019-01-01 to 2021-12-31. It was generated once by the authors using the following BigQuery SQL query and is hosted on Zenodo for reproducibility.

**DOI:** [https://doi.org/10.5281/zenodo.16626495]

**Provenance Query:**
```sql
WITH trips AS (
  SELECT DATE(pickup_datetime) AS d
  FROM  `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_*`
  WHERE _TABLE_SUFFIX BETWEEN '2019' AND '2021'
        AND pickup_datetime >= '2019-01-01'
        AND pickup_datetime <  '2022-01-01'
)
SELECT d AS date, COUNT(*) AS y
FROM trips
GROUP BY d
ORDER BY d



# Pre-Generated Datasets for [Retraining-Efficiency Score (RES): A Simple, Cost-Aware Retraining Rule for
Compute-Efficient, Accurate Forecasting]

This record contains the three pre-generated time series datasets used in the experiments for our paper.

The datasets were processed from their original sources into a consistent weekly format. They are provided here to ensure full reproducibility of our results without requiring users to have API keys or run complex data extraction pipelines. The exact scripts used to generate these files are included in the paper's code repository.

---

### 1. Weekly Electricity Consumption (Zone MT_001)
**DOI:** [https://doi.org/10.5281/zenodo.16627581] 

*   **File:** `electricity_weekly.csv`
*   **Description:** A single time series of total weekly electricity consumption for a single, representative zone (`MT_001`).
*   **Source:** [UCI Machine Learning Repository: ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
*   **Provenance:** The original 15-minute interval data for the `MT_001` zone was aggregated into weekly (Sunday-anchored) sums.

---
### 2. Weekly Wikipedia Page Views (Demo Series)
**DOI:** [https://doi.org/10.5281/zenodo.16627671]

*   **File:** `wiki_weekly.csv`
*   **Description:** A single time series of weekly page views.
*   **Source:** [Kaggle: Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
*   **Provenance:** This series corresponds to the **first data row** of the `train_1.csv` file, which is the page `2NE1_zh.wikipedia.org_all-access_spider`. The daily data was aggregated into weekly (Sunday-anchored) sums. This series was chosen as a representative example for the experiment.

---

### 3. Total Weekly Sales for All M5 Products
**DOI:** [https://doi.org/10.5281/zenodo.16627920]
    
*   **File:** `m5_sales_weekly.csv`
*   **Description:** A single time series representing the total aggregated weekly sales for all 30,490 products in the competition.
*   **Source:** [Kaggle: M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)
*   **Provenance:** The original daily sales data across all products was summed for each day and then aggregated into weekly (Sunday-anchored) totals.
