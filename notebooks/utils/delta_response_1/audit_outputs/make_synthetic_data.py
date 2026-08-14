from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "synthetic_data"
ROOT.mkdir(exist_ok=True)

jobs = pd.DataFrame(
    [
        ["a", 1.0, 0.10, 10.0, 100.0],
        ["b", 2.0, 0.20, 30.0, 400.0],
        ["c", 4.0, 0.40, 80.0, 1200.0],
    ],
    columns=["job_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
)
inhibitors = pd.DataFrame(
    [
        ["i0", 0.5, 0.05, 5.0, 50.0],
        ["i1", 1.0, 0.10, 12.0, 130.0],
        ["i2", 2.0, 0.25, 35.0, 500.0],
        ["i3", 3.0, 0.35, 60.0, 900.0],
        ["i4", 5.0, 0.55, 100.0, 1800.0],
        ["i5", 8.0, 0.75, 160.0, 3000.0],
    ],
    columns=["inhib_id", "mpi_time", "comm_frac", "total_msgs", "total_bytes"],
)

response_rows = []
for victim_index, victim in enumerate(jobs["job_id"]):
    for inhibitor_index, inhibitor in enumerate(inhibitors["inhib_id"]):
        log_slowdown = 0.04 + 0.03 * victim_index + 0.02 * inhibitor_index
        log_slowdown += 0.005 * victim_index * inhibitor_index
        response_rows.append([victim, inhibitor, float(np.exp(log_slowdown))])
responses = pd.DataFrame(response_rows, columns=["job_id", "inhib_id", "slowdown"])

pair_rows = []
for left in range(len(jobs)):
    for right in range(left, len(jobs)):
        log_left = 0.05 + 0.025 * left + 0.018 * right + 0.004 * left * right
        log_right = 0.05 + 0.025 * right + 0.018 * left + 0.004 * left * right
        pair_rows.append(
            [
                jobs.iloc[left]["job_id"],
                jobs.iloc[right]["job_id"],
                float(np.exp(log_left)),
                float(np.exp(log_right)),
            ]
        )
pairs = pd.DataFrame(
    pair_rows, columns=["jobA_id", "jobB_id", "slowdown_A", "slowdown_B"]
)

jobs.to_csv(ROOT / "jobs.csv", index=False)
inhibitors.to_csv(ROOT / "inhibitors.csv", index=False)
responses.to_csv(ROOT / "job_inh.csv", index=False)
pairs.to_csv(ROOT / "pair.csv", index=False)
