# src/mlflow_utils.py
from __future__ import annotations
import os
from pathlib import Path
from filelock import FileLock
import mlflow


def get_or_create_experiment_id(
    name: str = "res_paper_grid",
    tracking_uri: str | None = None,
) -> int:
    """
    Return the numeric experiment_id for *name*.
    • Creates the experiment exactly once, even under heavy concurrency.
    • Caches the ID in a tiny text file so next startup is O(1).
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    cache_file = Path(mlflow.get_tracking_uri()) / f".exp_{name}.id"

    # 1️⃣  fast path: use cached ID if the directory exists
    if cache_file.exists():
        return int(cache_file.read_text().strip())

    # 2️⃣  slow path: serialise creation with a lock
    lock_path = str(cache_file) + ".lock"
    with FileLock(lock_path):
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
        else:
            exp_id = exp.experiment_id
        cache_file.write_text(str(exp_id))
        return exp_id
