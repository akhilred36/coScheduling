from pathlib import Path
import sys

import pandas as pd


AUDIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT_DIR.parent))
from metrics import metrics_by_method


predictions = pd.read_csv(
    AUDIT_DIR / "primary_full_default" / "directional_predictions.csv"
)
rows = []
for stratum, frame in [
    ("true_equals_1", predictions[predictions["true_slowdown"].eq(1.0)]),
    ("true_above_1", predictions[predictions["true_slowdown"].gt(1.0)]),
]:
    result = metrics_by_method(frame)
    result.insert(1, "stratum", stratum)
    rows.append(result)
pd.concat(rows, ignore_index=True).to_csv(
    AUDIT_DIR / "post_holdout" / "performance_by_floor_stratum.csv", index=False
)
