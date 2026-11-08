"""Placement (socket) and inhibitor-coverage diagnostics from raw run logs.

1. Socket placement: mpibind verbose output ('mpibind: task   0 ... cpus N') in each
   job's output log gives the first PU of rank 0 on the first node. On Dane
   (2 x 56-core SPR, SMT-2, 224 PUs) standard Linux numbering is assumed:
   0-55 socket 0 HT0, 56-111 socket 1 HT0, 112-167 socket 0 HT1, 168-223 socket 1 HT1.
2. Within-configuration effect of socket on log T_MPI and log T_app.
3. Inhibitor time coverage of the application run in App-Inhibitor co-runs:
   min(1, inhibitor AppTime / app AppTime) (both start in the same second).

Requires run_data/ (raw logs). Writes output/placement_numbers.md.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import APPS, NAME, OUT_DIR, PROCESSED, REPO, read

RUN = REPO / "run_data"
PAT = re.compile(r"mpibind: task\s+0 .*cpus (\d+)")
out = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=out)


def first_pu(log: Path):
    try:
        with log.open(errors="ignore") as fh:
            for line in fh:
                m = PAT.search(line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        return None
    return None


def socket_of(pu):
    return None if pu is None else int((pu % 112) >= 56)


# --------------------------------------------------------------------------- scan
rows = []
for d in sorted((RUN / "experiments_coscheduling_inhib_coscheduled").glob("*/data/*")):
    parts = d.name.split("_")  # 8_app_inhib_m_w_d_sp_rep
    rows.append(dict(App=parts[1], m=int(parts[3]), w=int(parts[4]), sp=round(float(parts[6]), 9),
                     rep=int(parts[7]), app_sock=socket_of(first_pu(d / "app_output.log")),
                     inh_sock=socket_of(first_pu(d / "inhib_output.log"))))
ai_s = pd.DataFrame(rows)
rows = []
for d in sorted((RUN / "experiments_coscheduling_app_coscheduled").glob("*/data/8_*")):
    nn, a, b, rep = d.name.split("_")
    if a == b:
        continue
    rows.append(dict(A=a, B=b, rep=int(rep), sock_A=socket_of(first_pu(d / f"output_{a}.log")),
                     sock_B=socket_of(first_pu(d / f"output_{b}.log"))))
aa_s = pd.DataFrame(rows)
rows = []
for d in sorted((RUN / "experiments_coscheduling_isolated").glob("*/data/8_*")):
    rows.append(dict(run=d.name, pu=first_pu(d / "output.log")))
iso_s = pd.DataFrame(rows)
iso_s["socket"] = iso_s.pu.map(socket_of)
OUT_DIR.mkdir(exist_ok=True)
ai_s.to_csv(OUT_DIR / "socket_app_inhibitor.csv", index=False)
aa_s.to_csv(OUT_DIR / "socket_app_app.csv", index=False)
iso_s.to_csv(OUT_DIR / "socket_isolated.csv", index=False)

p("## Socket placement")
p(f"Isolated 8-node app runs: first PU counts {iso_s.pu.value_counts().to_dict()}; socket counts {iso_s.socket.value_counts().to_dict()}")
p(f"App-Inhibitor runs: (app socket, inhibitor socket) counts {ai_s.groupby(['app_sock','inh_sock']).size().to_dict()}")
p(f"App-App non-self 8-node runs: (A socket, B socket) counts {aa_s.groupby(['sock_A','sock_B']).size().to_dict()}")
p()

# --------------------------------------------------------------------------- effects
ai = read(PROCESSED / "data_inhib_coscheduled.csv").rename(columns={
    "Inhib Message Size": "m", "Inhib Wait Time (us)": "w", "Inhib Comm Sparsity": "sp", "Run Iteration": "rep"})
ai["sp"] = ai.sp.round(9)
ai = ai.merge(ai_s, on=["App", "m", "w", "sp", "rep"], how="inner")
keys = ["App", "m", "w", "sp"]


def within_effect(frame, group, sock_col, val_col):
    f = frame.copy()
    f["dm"] = np.log(f[val_col]) - f.groupby(group)[val_col].transform(lambda x: np.log(x).mean())
    mixed = f.groupby(group)[sock_col].transform("nunique") == 2
    f = f[mixed]
    per = f.groupby(["victim", sock_col])["dm"].mean().unstack()
    per["socket1 - socket0 (log)"] = per[1] - per[0]
    e = f.groupby(sock_col)["dm"].mean()
    return e[1] - e[0], len(f), per["socket1 - socket0 (log)"]


ai["victim"] = ai.App
p("## Within-configuration socket effect (socket 1 minus socket 0, mean log difference)")
res = {}
for label, col in (("T_MPI", "MPI Time Average"), ("T_app", "App Time Average")):
    eff, n, per = within_effect(ai, keys, "app_sock", col)
    res[f"App-Inhibitor {label}"] = per
    p(f"App-Inhibitor {label}: {eff:+.4f} (x{np.exp(eff):.3f}), rows in mixed configs = {n}")
aa = read(PROCESSED / "data_coscheduled.csv")
aa = aa[aa["Num Nodes"] == 8].rename(columns={"App A": "A", "App B": "B", "Run Iteration": "rep"})
aa = aa.merge(aa_s, on=["A", "B", "rep"], how="inner")
long = pd.concat([
    pd.DataFrame({"pair": aa.A + "|" + aa.B, "victim": aa.A, "sock": aa.sock_A,
                  "T_MPI": aa["App A MPI Time Average"], "T_app": aa["App A Time Average"]}),
    pd.DataFrame({"pair": aa.A + "|" + aa.B, "victim": aa.B, "sock": aa.sock_B,
                  "T_MPI": aa["App B MPI Time Average"], "T_app": aa["App B Time Average"]})])
for label in ("T_MPI", "T_app"):
    eff, n, per = within_effect(long, ["pair", "victim"], "sock", label)
    res[f"App-App {label}"] = per
    p(f"App-App {label}: {eff:+.4f} (x{np.exp(eff):.3f}), rows in mixed directions = {n}")
p()
t = pd.DataFrame(res).loc[APPS].rename(index=NAME).round(3)
p("Per-app socket effect (log; negative = faster on socket 1):")
p(t.to_string())
share = long.groupby(["pair", "victim"]).sock.mean().value_counts().sort_index()
p(f"\nApp-App directional outcomes by share of the 4 replicates with the victim on socket 1: {share.to_dict()}")
p()

# --------------------------------------------------------------------------- coverage
p("## Inhibitor runtime coverage of the application run (App-Inhibitor)")
ai_all = read(PROCESSED / "data_inhib_coscheduled.csv")
ai_all["cov"] = np.minimum(1, ai_all["Inhib App Time Average"] / ai_all["App Time Average"])
iso = read(PROCESSED / "data_isolated_agg.csv")
iso8 = iso[iso["Num Nodes"] == 8].set_index("App")["App Time Average Mean"]
r_used = {}
for a in APPS:
    s = next((RUN / "experiments_coscheduling_inhib_coscheduled").glob(f"*/slurm_scripts/8_{a}_inhib_*.slurm"))
    r_used[a] = float(re.search(r" -r ([0-9.]+)", s.read_text()).group(1))
g = ai_all.groupby("App")["cov"]
t = pd.DataFrame({"-r used (s)": pd.Series(r_used), "isolated 8-node T_app (s)": iso8,
                  "median coverage": g.median(), "p10 coverage": g.quantile(0.1),
                  "frac runs coverage<0.95": g.apply(lambda x: (x < 0.95).mean())}).loc[APPS]
t.index = [NAME[a] for a in t.index]
p(t.round(3).to_string())

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "placement_numbers.md").write_text(out.getvalue())
print("wrote", OUT_DIR / "placement_numbers.md")
