"""Socket batch 2, part A facts (A1, A2): socket assignment of every co-run, from all mpibind task lines.

A1  per run: socket set of every listed task of each job (mpibind prints one node's block, 56 tasks); is the
    victim socket the complement of the aggressor socket?
A2  replicate counts per (directed pair / app x inhibitor, victim socket) cell.

Writes output/socket_corun_assign.csv (one row per co-run job) and output/socket_aware_facts.md.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from common import NAME, OUT_DIR, REPO

RUN = REPO / "run_data"
TASK = re.compile(r"mpibind: task\s+(\d+) .*cpus (\d+)")
log = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=log)


def sockets(path: Path):
    """(set of sockets over listed tasks, n tasks, sorted unique hyperthread indices)."""
    s, ht, n = set(), set(), 0
    try:
        with path.open(errors="ignore") as fh:
            for line in fh:
                m = TASK.search(line)
                if m:
                    pu = int(m.group(2))
                    s.add(int(pu % 112 >= 56))
                    ht.add(int(pu >= 112))
                    n += 1
    except FileNotFoundError:
        return None, 0, None
    return s, n, ht


def one(sset):
    return next(iter(sset)) if sset and len(sset) == 1 else (-1 if sset else None)


rows = []
for d in sorted((RUN / "experiments_coscheduling_inhib_coscheduled").glob("*/data/*")):
    parts = d.name.split("_")  # 8_app_inhib_m_w_d_sp_rep
    sa, na, ha = sockets(d / "app_output.log")
    si, ni, hi = sockets(d / "inhib_output.log")
    rows.append(dict(campaign="AI", ts=d.parent.parent.name, run=d.name, victim=parts[1],
                     aggressor=f"inh_{parts[3]}_{parts[4]}_{round(float(parts[6]), 9)}",
                     m=int(parts[3]), w=int(parts[4]), sp=round(float(parts[6]), 9), rep=int(parts[7]),
                     n_mpip=len(list((d / "mpip_profiles").glob("*.mpiP"))) if (d / "mpip_profiles").exists() else 0,
                     s_V=one(sa), s_X=one(si), nV=na, nX=ni, htV=str(sorted(ha or [])), htX=str(sorted(hi or []))))
for d in sorted((RUN / "experiments_coscheduling_app_coscheduled").glob("*/data/8_*")):
    nn, a, b, rep = d.name.split("_")
    if a == b:
        continue
    sa, na, ha = sockets(d / f"output_{a}.log")
    sb, nb, hb = sockets(d / f"output_{b}.log")
    for v, x, sv, sx, nv, nx, hv, hx in ((a, b, sa, sb, na, nb, ha, hb), (b, a, sb, sa, nb, na, hb, ha)):
        rows.append(dict(campaign="AA", ts=d.parent.parent.name, run=d.name, victim=v, aggressor=x, rep=int(rep),
                         n_mpip=len(list((d / "mpip_profiles").glob("*.mpiP"))),
                         s_V=one(sv), s_X=one(sx), nV=nv, nX=nx, htV=str(sorted(hv or [])),
                         htX=str(sorted(hx or []))))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "socket_corun_assign.csv", index=False)

p("# Socket batch 2, part A facts")
p()
p("## A1. Is the victim socket always the complement of the aggressor socket?")
p("Source: every `mpibind: task <i> ... cpus <PU>` line in each job's output log (mpibind prints one node's "
  "56-task block per job, so this covers one of the 8 nodes per run). Socket = PU mod 112 >= 56.")
for c, g in df.groupby("campaign"):
    lab = {"AI": "App-Inhibitor", "AA": "App-App (non-self, 8 nodes; directed rows, 2 per run)"}[c]
    p(f"\n### {lab}: {len(g)} rows, {g.run.nunique()} run dirs")
    p(f"- tasks listed per job: victim {sorted(g.nV.unique())}, aggressor {sorted(g.nX.unique())}")
    p(f"- hyperthread index sets: victim {g.htV.value_counts().to_dict()}, aggressor {g.htX.value_counts().to_dict()}")
    p(f"- jobs spanning both sockets: victim {int((g.s_V == -1).sum())}, aggressor {int((g.s_X == -1).sum())}; "
      f"no placement lines: victim {int(g.s_V.isna().sum())}, aggressor {int(g.s_X.isna().sum())}")
    ok = g.s_V.isin([0, 1]) & g.s_X.isin([0, 1])
    comp = ok & (g.s_V != g.s_X)
    p(f"- (s_V, s_X) counts: {g[ok].groupby(['s_V', 's_X']).size().to_dict()}")
    p(f"- complement holds in {int(comp.sum())}/{len(g)} rows; exceptions: {len(g) - int(comp.sum())}")
    ex = g[~comp]
    if len(ex):
        p("  exceptions (run, s_V, s_X, n_mpip): " + "; ".join(
            f"{r.ts}/{r.run} ({r.s_V}, {r.s_X}, {r.n_mpip})" for r in ex.itertuples()))
p()

p("## A2. Replicates per socket cell")
aa = df[(df.campaign == "AA") & df.s_V.isin([0, 1])]
cnt = aa.groupby(["victim", "aggressor"]).s_V.agg(n0=lambda s: int((s == 0).sum()), n1=lambda s: int((s == 1).sum()))
p("### App-App: directed non-self outcomes (victim, aggressor) = 90")
p(f"- (n reps victim on sock0, sock1) distribution: {cnt.value_counts().sort_index().to_dict()}")
p(f"- outcomes with 0 reps on one socket: {int(((cnt.n0 == 0) | (cnt.n1 == 0)).sum())} "
  f"(all on sock0: {int((cnt.n1 == 0).sum())}, all on sock1: {int((cnt.n0 == 0).sum())})")
p(f"- socket-resolved cells (outcome x victim socket with >= 1 rep): {int((cnt.n0 > 0).sum() + (cnt.n1 > 0).sum())}; "
  f"cells with exactly 1 rep: {int((cnt.n0 == 1).sum() + (cnt.n1 == 1).sum())}")
one_sided = cnt[(cnt.n0 == 0) | (cnt.n1 == 0)].reset_index()
p("- one-sided outcomes: " + ", ".join(f"{NAME[r.victim]}<-{NAME[r.aggressor]} ({r.n0}/{r.n1})"
                                        for r in one_sided.itertuples()))
per_v = aa.groupby("victim").s_V.mean()
p("- share of victim reps on sock1 per app: " + ", ".join(f"{NAME[k]} {v:.2f}" for k, v in per_v.items()))

ai = df[(df.campaign == "AI") & df.s_V.isin([0, 1])]
PANEL = [(2000000, 1000000, 0.3), (20000, 100000, 0.1), (200, 200000, 0.002237136), (2500000, 200000, 0.002237136),
         (2500000, 600000, 0.002237136), (400000, 1000000, 0.5), (40000, 1000000, 0.1), (40000, 1000000, 0.7),
         (500, 5000, 0.05)]
key = list(zip(ai.m, ai.w, ai.sp))
ai = ai.assign(panel=[k in set(PANEL) for k in key])
for lab, g in (("90 panel rows (10 apps x 9 panel inhibitors)", ai[ai.panel]), ("full grid", ai)):
    c = g.groupby(["victim", "m", "w", "sp"]).s_V.agg(n0=lambda s: int((s == 0).sum()),
                                                       n1=lambda s: int((s == 1).sum()))
    p(f"\n### App-Inhibitor, {lab}: {len(c)} cells (app x inhibitor)")
    p(f"- (n0, n1) distribution: {c.value_counts().sort_index().to_dict()}")
    p(f"- cells with 0 reps on one socket: {int(((c.n0 == 0) | (c.n1 == 0)).sum())} "
      f"(all sock0 {int((c.n1 == 0).sum())}, all sock1 {int((c.n0 == 0).sum())}); "
      f"resolved cells with >= 1 rep: {int((c.n0 > 0).sum() + (c.n1 > 0).sum())}; "
      f"cells with exactly 1 rep: {int((c.n0 == 1).sum() + (c.n1 == 1).sum())}")
    p("- share of app reps on sock1 per app: " + ", ".join(
        f"{NAME[k]} {v:.2f}" for k, v in g.groupby("victim").s_V.mean().items()))
(OUT_DIR / "socket_aware_facts.md").write_text(log.getvalue())
