import pandas as pd
import mlflow
from pathlib import Path

class WeekLogger:
    """
    Collect (run, week) rows and, at the end of the run, persist them
    as an artefact.  Purely additive: if the caller never touches this
    class nothing new is logged and behaviour is 100 % identical to the
    old pipeline.
    """
    def __init__(self, enabled: bool = True, dir: str | Path = "."):
        self.enabled = enabled
        self.dir     = Path(dir)
        self._rows = []

    def log(self, **kw):
        if self.enabled:
            self._rows.append(kw)

    def flush(self, filename: str = "week_metrics.parquet"):
        if not (self.enabled and self._rows):
            return
        out = self.dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(out, index=False)
        mlflow.log_artifact(out)
        mlflow.set_tag("has_week_metrics", "true")
        mlflow.log_metric("weekly_rows", len(self._rows))
