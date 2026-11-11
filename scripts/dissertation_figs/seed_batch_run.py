"""Seed batch job pool: runs seed_batch/jobs.tsv with W single-threaded workers, resumable.

usage: seed_batch_run.py [workers] [part ...]   (default 7 workers, all parts)
Order: one 1W restricted timing task, full-data jobs (longest first), restricted (longest first), restricted_socket.
A job is skipped if its out_dir/run_report.json says complete. Each job logs to seed_batch/logs/<name>.log; finished jobs
append name, rc, seconds, start, end to seed_batch/done.tsv.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path.home() / "hpcResearch/experiment_outputs/seed_batch"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 7
PARTS = sys.argv[2:] or ["full", "restricted", "restricted_socket"]
TIMING = "R_blind_1w_mpi_s06_8"
J = pd.read_csv(OUT / "jobs.tsv", sep="\t", keep_default_na=False)
J = J[J.part.isin(PARTS)]


def complete(out: str) -> bool:
    r = Path(out) / "run_report.json"
    try:
        return json.loads(r.read_text()).get("status") == "complete"
    except Exception:
        return False


rank = {"full": 1, "restricted": 2, "restricted_socket": 3}
J = J.assign(pri=[0 if n == TIMING else rank[p] for n, p in zip(J.name, J.part)])
J = J.sort_values(["pri", "est_seconds"], ascending=[True, False])
todo = [r for r in J.itertuples() if not complete(r.out_dir)]
print(f"{len(todo)} jobs to run ({len(J) - len(todo)} already complete), {W} workers", flush=True)
(OUT / "logs").mkdir(exist_ok=True)
lock = threading.Lock()
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def run(r):
    t0 = time.time()
    with open(OUT / "logs" / f"{r.name}.log", "w") as fh:
        rc = subprocess.call(json.loads(r.argv), cwd=r.cwd or None, stdout=fh, stderr=subprocess.STDOUT, env=env)
    t1 = time.time()
    with lock:
        with open(OUT / "done.tsv", "a") as fh:
            fh.write(f"{r.name}\t{rc}\t{t1 - t0:.0f}\t{time.strftime('%T', time.localtime(t0))}\t"
                     f"{time.strftime('%T', time.localtime(t1))}\n")
    print(f"{r.name} rc={rc} {t1 - t0:.0f}s", flush=True)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(run, todo))
print("all done", flush=True)
