import time, pandas as pd, json
from src.train import train
from src.evaluate import evaluate
from src.pipeline import load_with_features
from pathlib import Path

df = load_with_features(Path("data/processed/m5_sales_clean.parquet"),
                        lags=[1,4,52], rolls=[4,12])
train_df = df.iloc[:-52]           # 80 weeks history
test_df  = df.iloc[-52:]           # dummy future

t0 = time.perf_counter()
model, t_train = train(train_df, "lightgbm", {})
train_sec = time.perf_counter() - t0

t0 = time.perf_counter()
_ = evaluate(model, test_df, "lightgbm")
valid_sec = time.perf_counter() - t0

t0 = time.perf_counter()
# deployment stub – nothing to do
time.sleep(0.1)
deploy_sec = time.perf_counter() - t0

pd.DataFrame(dict(stage=["train","valid","deploy"],
                  seconds=[train_sec, valid_sec, deploy_sec]))\
  .to_csv("figures/profile_cost_breakdown.csv", index=False)
print("Profile saved.")
