"""Host sets per run from mpiP 'MPI Task Assignment' headers (all campaigns). Writes output/run_hosts.csv."""
import re
from pathlib import Path
import pandas as pd
from common import OUT_DIR, REPO
RX = re.compile(r"^@ MPI Task Assignment\s*:\s*\d+\s+(\S+)")
rows = []
for camp in ("isolated", "inhib_isolated", "inhib_coscheduled", "app_coscheduled", "isolated_socket"):
    for f in (REPO / "run_data").glob(f"experiments_coscheduling_{camp}/*/data*/**/*.mpiP"):
        hosts = set()
        with f.open(errors="ignore") as fh:
            for i, line in enumerate(fh):
                m = RX.match(line)
                if m:
                    hosts.add(m.group(1))
                elif hosts or i > 2000:
                    break
        rows.append(dict(campaign=camp, ts=f.parts[-5] if camp != "isolated_socket" else "", run=f.parent.parent.name,
                         file=f.name, hosts=" ".join(sorted(hosts))))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "run_hosts.csv", index=False)
d261 = df[df.hosts.str.contains(r"\bdane261\b")]
print(df.groupby("campaign").size().to_dict())
print("runs on dane261 by campaign:", d261.groupby("campaign").size().to_dict())
print(d261[["campaign", "ts", "run", "file"]].to_string())
