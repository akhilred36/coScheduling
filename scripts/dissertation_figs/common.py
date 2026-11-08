"""Shared data loading and figure style for the dissertation node-sharing chapter.

Canonical data = the current rebuilt snapshot produced by
notebooks/utils/delta_response_filteredInhibs_biDirectional/prepare_bidirectional_data.py
(MPI-time targets) and its *_totalTime sibling (total-time targets), both built
from processed_data/ at branch dane @ 5eea7198.

Run with: /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/python
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "processed_data"
BIDIR = REPO / "notebooks/utils/delta_response_filteredInhibs_biDirectional"
BIDIR_TT = REPO / "notebooks/utils/delta_response_filteredInhibs_biDirectional_totalTime"
FIG_DIR = Path("/home/akhil/hpcResearch/repos/Dissertation/figures/node_sharing")
OUT_DIR = Path(__file__).resolve().parent / "output"

APPS = ["amg", "beatnik", "fiesta", "kripke", "laghos", "lammps", "minife",
        "minivite", "quicksilver", "tricount"]
NAME = {"amg": "AMG2023", "beatnik": "Beatnik", "fiesta": "FIESTA", "kripke": "Kripke",
        "laghos": "Laghos", "lammps": "LAMMPS", "minife": "MiniFE", "minivite": "miniVite",
        "quicksilver": "Quicksilver", "tricount": "TriC"}
ONE_NODE_APPS = ["beatnik", "fiesta", "lammps", "minife", "minivite", "tricount"]

# Selection features exactly as select_inhibitors.communication_features.
SEL_FEATURES = ["comm_frac", "log1p_message_rate", "log1p_byte_rate", "log1p_mean_message_size"]
SEL_LABELS = ["comm. fraction", "log(1+msg rate)", "log(1+byte rate)", "log(1+mean msg size)"]


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()


# ----------------------------------------------------------------------------- profiles
def jobs() -> pd.DataFrame:
    """Isolated 8-node application profiles (canonical training/jobs.csv)."""
    return read(BIDIR / "data/training/jobs.csv").set_index("job_id").loc[APPS]


def inhibitors() -> pd.DataFrame:
    return read(BIDIR / "data/training/inhibitors.csv")


def job_inh() -> pd.DataFrame:
    """App-Inhibitor MPI-time responses (canonical, 2,870 aggregate configs)."""
    return read(BIDIR / "data/training/job_inh.csv")


def selection_features(frame: pd.DataFrame) -> np.ndarray:
    mpi = frame["mpi_time"].to_numpy(float)
    frac = frame["comm_frac"].to_numpy(float)
    msgs = frame["total_msgs"].to_numpy(float)
    size = frame["total_bytes"].to_numpy(float)
    runtime = mpi / frac
    mean_size = np.divide(size, msgs, out=np.zeros_like(size), where=msgs > 0)
    return np.column_stack([frac, np.log1p(msgs / runtime), np.log1p(size / runtime),
                            np.log1p(mean_size)])


# ----------------------------------------------------------------------------- App-App
def pairs_mpi() -> pd.DataFrame:
    return read(BIDIR / "data/evaluation/pair.csv")


def pairs_total() -> pd.DataFrame:
    return read(BIDIR_TT / "data/evaluation/pair.csv")


def directional(pairs: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    """Expand pair rows into directional outcomes: victim, aggressor, slowdown, raw ratio.

    raw_col is 'raw_mpi_time_ratio' or 'raw_app_time_ratio'.
    """
    rows = []
    for r in pairs.itertuples(index=False):
        d = r._asdict()
        rows.append(dict(victim=d["jobA_id"], aggressor=d["jobB_id"], endpoint="A",
                         slowdown=d["slowdown_A"], raw=d[f"{raw_col}_A"]))
        rows.append(dict(victim=d["jobB_id"], aggressor=d["jobA_id"], endpoint="B",
                         slowdown=d["slowdown_B"], raw=d[f"{raw_col}_B"]))
    out = pd.DataFrame(rows)
    out["self"] = out.victim == out.aggressor
    return out


def directional_from_processed(num_nodes: int, time: str = "MPI") -> pd.DataFrame:
    """Directional slowdowns straight from processed_data aggregates (used for 1-node)."""
    iso = read(PROCESSED / "data_isolated_agg.csv")
    iso = iso[iso["Num Nodes"] == num_nodes].set_index("App")
    co = read(PROCESSED / "data_coscheduled_agg.csv")
    co = co[co["Num Nodes"] == num_nodes]
    col = "MPI Time Average Mean" if time == "MPI" else "Time Average Mean"
    iso_col = "MPI Time Average Mean" if time == "MPI" else "App Time Average Mean"
    rows = []
    for r in co.itertuples(index=False):
        d = dict(zip(co.columns, r))
        for v, x in (("A", "B"), ("B", "A")):
            victim, aggr = d[f"App {v}"], d[f"App {x}"]
            if time == "MPI":
                co_t = d[f"App {v} MPI Time Average Mean"]
            else:
                co_t = d[f"App {v} Time Average Mean"]
            raw = co_t / iso.loc[victim, iso_col]
            rows.append(dict(victim=victim, aggressor=aggr, endpoint=v, raw=raw,
                             slowdown=max(1.0, raw)))
    out = pd.DataFrame(rows)
    out["self"] = out.victim == out.aggressor
    return out


# ----------------------------------------------------------------------------- style
def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# Okabe-Ito colorblind-safe palette.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
      "grey": "#999999"}
FULL_W, HALF_W = 6.0, 3.0


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    print("wrote", FIG_DIR / name)
