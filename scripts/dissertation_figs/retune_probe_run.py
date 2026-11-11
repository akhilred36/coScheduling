"""Re-tune probe job pool: run jobs_<stage>.tsv files with W single-threaded workers (resumable; nice 19 inherited).

usage: retune_probe_run.py W stage [stage ...]
Skips jobs whose run_report.json says complete. Logs to retune_probe/logs/<name>.log; appends to retune_probe/done.tsv.
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

OUT = Path.home() / f"hpcResearch/experiment_outputs/{os.environ.get('PROBE', 'retune')}_probe"
W = int(sys.argv[1])
J = pd.concat([pd.read_csv(OUT / f"jobs_{s}.tsv", sep="\t", keep_default_na=False) for s in sys.argv[2:]])
# longest first: context variants and long budgets
J = J.assign(pri=[-(("context" in n) * 2 + ("_e240" in n or "_e120" in n)) for n in J.name]).sort_values("pri")


def complete(out: str) -> bool:
    try:
        return json.loads((Path(out) / "run_report.json").read_text()).get("status") == "complete"
    except Exception:
        return False


todo = [r for r in J.itertuples() if not complete(r.out_dir)]
print(f"{len(todo)} jobs to run, {W} workers", flush=True)
(OUT / "logs").mkdir(exist_ok=True)
lock = threading.Lock()
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def run(r):
    t0 = time.time()
    with open(OUT / "logs" / f"{r.name}.log", "w") as fh:
        rc = subprocess.call(json.loads(r.argv), cwd=r.cwd or None, stdout=fh, stderr=subprocess.STDOUT, env=env)
    t1 = time.time()
    with lock, open(OUT / "done.tsv", "a") as fh:
        fh.write(f"{r.name}\t{rc}\t{t1 - t0:.0f}\t{time.strftime('%T', time.localtime(t0))}\t"
                 f"{time.strftime('%T', time.localtime(t1))}\n")
    print(f"{r.name} rc={rc} {t1 - t0:.0f}s", flush=True)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(run, todo))
print("stage done", flush=True)
