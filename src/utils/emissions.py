# src/utils/emissions.py
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Dict


try:
    from codecarbon import EmissionsTracker      # pip install codecarbon
    HAVE_CC = True
except ImportError:
    HAVE_CC = False

_KG_PER_KWH = 0.475          # US average; CodeCarbon uses the same




@contextmanager
def track_emissions(enabled: bool,
                    out_dir: Path,
                    country: str = "USA") -> Iterator[Dict[str, float]]:
    """
    Context-manager that measures energy / CO₂ for a code block.
    If CodeCarbon is missing *or* enabled == False → returns zeros.
    """
    stats = {"kg_co2": 0.0, "kwh": 0.0}

    if not enabled or not HAVE_CC:
        yield stats
        return

    tracker = EmissionsTracker(
        output_dir=str(out_dir),
        measure_power_secs=1,
        log_level="error",
    )
    tracker.start()
    yield stats
    kg = tracker.stop()

    kwh = kg / _KG_PER_KWH
    stats.update({"kg_co2": kg,
                  "kwh":     kwh,
                  })
