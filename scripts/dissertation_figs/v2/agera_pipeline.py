"""Plan v2 prediction study: 8 methods x {1W, BD} x {MPI, app time} x {socket-aware, blind} x tiers.

Self-contained (numpy, pandas, scikit-learn, xgboost, torch); reads only the export bundle written by v2_build.py.

Unit = one socket-resolved directed App-App cell (victim A, co-runner B, victim socket s_V; self-pairs pooled, s_V=0.5).
Target y = UNCLIPPED ln(T_corun / T_base_JMs) for the chosen timer. Nothing is clipped anywhere.

Tiers / splits
  full        split k (k = seed): 11 of the 55 unordered co-runs (both directions, all socket cells) are test,
              drawn with numpy default_rng(1000 + k). Init seed = k. One fit per method per split.
  restricted  split j in 0..119 = j-th C(10,3) held-out triple (itertools order over the APPS list).
              Train = rows among the 7 known apps. Test = one_known (exactly one held-out endpoint) and
              zero_shot (both endpoints held out); one fit scores both. 10 init seeds (deterministic learners: 1 fit).

Methods
  App-App regressors (trained on the split's training rows): ridge, lasso, svr, rf, xgb, mlp.
     Features (z-scored on training rows): 1W = A panel y (9) + B panel y (9) + A, B base profiles (4+4) + bit;
     BD adds A, B inhibitor slowdown caused (9+9). Panel features use the target's timer.
  Inhibitor-only response models (trained on App-Inh panel cells of the training apps; no App-App labels):
     absolute  R(V,X) = h(e_X) + u(e_V).v(e_X), shared encoder 5->8->4 (4 base features + own socket bit), rank 2
     delta     same potential trained on within-victim differences; prediction = R(A,B) + aggregated residual of
               A's panel cells (y_j - R(A, I_j)), aggregation uniform / median / kernel(T).
     Plain Huber (delta=1) on unclipped targets. No BD form (form = "none").
  Socket: aware = victim bit s_V (co-runner/inhibitor bit 1 - s_V for the response models); blind = constant 0.

Hyperparameters: nested CV on the split's training data only, init seed 0, then frozen across seeds.
  regressors: full = GroupKFold(5) over unordered pairs; restricted = leave-one-known-app-out (validation rows have
  exactly one endpoint = the left-out app). Response models: leave-one-training-app-out over App-Inh panel cells;
  delta validates each cell with leave-one-config-out anchors. Criterion: pooled validation log MAE.

Outputs (under --out): preds/<task>.csv.gz, hp/<task>.csv (every grid point's CV score + chosen flag),
  done/<task>.json (timings, versions), run_config.json. Re-running skips finished tasks (resume).

Usage (agera):
  export CUDA_VISIBLE_DEVICES=""
  python scripts/dissertation_figs/v2/agera_pipeline.py --bundle scripts/dissertation_figs/agera_bundle \
      --out scripts/dissertation_figs/agera_out/v2_full --workers 78
  pilot:  --tiers full,restricted --splits-full 0,1 --splits-restricted 0,57 --seeds 0,1 --workers 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

APPS = ["amg", "beatnik", "fiesta", "kripke", "laghos", "lammps", "minife", "minivite", "quicksilver", "tricount"]
REGRESSORS = ["ridge", "lasso", "svr", "rf", "xgb", "mlp"]
RESPONSE = ["absolute", "delta"]
DETERMINISTIC = {"ridge", "lasso", "svr"}
TARGETS = ["mpi", "tt"]
SOCKETS = ["aware", "blind"]
FORMS = ["1w", "bd"]
MAX_WORKERS = 80
TUNE_SEED = 0
FULL_TEST_PAIRS = 11

# ----------------------------------------------------------------------------- grids (approved A4)
# Grids: A4 as approved, extended once after the B6 pilot edge check (pilot_a; rule: ordered axes with >= 3 values
# whose edge wins > 30% of tasks get two more steps at the same spacing; 2-value axes only when one side wins >= 2/3;
# natural bounds kept). Extensions are marked "ext".
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5)],  # ext 1e4, 1e5
    "lasso": [{"alpha": a} for a in (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3)],  # ext 1, 3
    "svr": [{"C": c, "epsilon": e, "gamma": g}
            for c in (0.1, 1, 10, 100, 1000) for e in (0.05, 0.2) for g in ("scale", "0.3/p")],  # ext C 100, 1000
    "rf": [{"max_depth": d, "min_samples_leaf": l, "max_features": f}
           for d in (1, 2, 3, 6, None) for l in (1, 3, 5) for f in (0.33, 1.0)],  # ext depth 2, 1
    "xgb": [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_child_weight": w}
            for n in (10, 30, 100, 300) for d in (1, 2, 4) for lr in (0.003, 0.01, 0.03, 0.1)
            for w in (1, 5)],  # ext n 30, 10; depth 1; lr 0.01, 0.003
    # epochs are checkpoints of one training run per (hidden, wd)
    "mlp": [{"hidden": h, "weight_decay": wd, "epochs": e}
            for h in ("16", "32-16") for wd in (1e-4, 1e-2) for e in (10, 30, 100, 300, 1000)],  # ext 30, 10
    "absolute": [{"lr": lr, "weight_decay": wd, "epochs": e}
                 for lr in (1e-3, 3e-3, 1e-2, 3e-2) for wd in (1e-4, 1e-2, 0.1, 1.0)
                 for e in (10, 20, 40, 80, 160)],  # ext lr 1e-2, 3e-2; wd 0.1, 1 (capped: AdamW wd > 1 degenerate)
    "delta": [{"lr": lr, "weight_decay": wd, "epochs": e, "agg": a}
              for lr in (1e-3, 3e-3, 1e-2, 3e-2) for wd in (1e-4, 1e-2) for e in (10, 20, 40, 80, 160, 320, 640)
              for a in ("uniform", "median", "kernel_0.3", "kernel_1", "kernel_3")],  # ext lr 1e-2, 3e-2; ep 320, 640
}
MLP_LR = 3e-3
RESP_BATCHES, RESP_BATCH = 4, 128


# ----------------------------------------------------------------------------- data
class Bundle:
    def __init__(self, root: Path):
        man = json.loads((root / "manifest.json").read_text())
        for name, meta in man["files"].items():
            got = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if got != meta["sha256"]:
                raise SystemExit(f"bundle checksum mismatch for {name}")
        self.manifest = man
        self.panel = man["panel"]
        self.apps = pd.read_csv(root / "apps.csv").set_index("app").loc[APPS]
        self.inh = pd.read_csv(root / "panel_inhibitors.csv").set_index("config").loc[self.panel]
        self.panel_app = pd.read_csv(root / "panel_app.csv")
        self.resolved = pd.read_csv(root / "appinh_resolved.csv")
        rows = pd.read_csv(root / "appapp_rows.csv")
        rows = rows.sort_values(["victim", "corunner", "s_V"]).reset_index(drop=True)
        rows["row_id"] = np.arange(len(rows))
        self.rows = rows
        self.pairs = sorted(rows.pair.unique())

    def phys(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.log1p(frame.mpi_time), frame.comm_frac, np.log1p(frame.total_msgs),
                                np.log1p(frame.total_bytes)]).astype(float)

    def features(self, target: str, form: str, socket: str) -> tuple[np.ndarray, list[str]]:
        py = self.panel_app.pivot(index="app", columns="config", values=f"y_{target}").loc[APPS, self.panel]
        pc = self.panel_app.pivot(index="app", columns="config", values=f"c_{target}").loc[APPS, self.panel]
        prof = pd.DataFrame(self.phys(self.apps), index=APPS, columns=["lmpi", "frac", "lmsg", "lbytes"])
        r = self.rows
        blocks, names = [], []
        for end, col in (("A", "victim"), ("B", "corunner")):
            blocks.append(py.loc[r[col]].to_numpy())
            names += [f"{end}_panel_{c}" for c in self.panel]
        for end, col in (("A", "victim"), ("B", "corunner")):
            blocks.append(prof.loc[r[col]].to_numpy())
            names += [f"{end}_{c}" for c in prof.columns]
        if form == "bd":
            for end, col in (("A", "victim"), ("B", "corunner")):
                blocks.append(pc.loc[r[col]].to_numpy())
                names += [f"{end}_caused_{c}" for c in self.panel]
        bit = r.s_V.to_numpy(float) if socket == "aware" else np.zeros(len(r))
        blocks.append(bit[:, None])
        names.append("s_V")
        return np.hstack(blocks), names


def full_split(bundle: Bundle, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + k)
    test_pairs = set(rng.choice(np.array(bundle.pairs), FULL_TEST_PAIRS, replace=False).tolist())
    te = bundle.rows.pair.isin(test_pairs).to_numpy()
    return ~te, te


def restricted_split(bundle: Bundle, j: int):
    held = set(list(itertools.combinations(APPS, 3))[j])
    r = bundle.rows
    n = r.victim.isin(held).astype(int) + r.corunner.isin(held).astype(int)
    return (n == 0).to_numpy(), (n == 1).to_numpy(), (n == 2).to_numpy(), sorted(held)


def log_mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# ----------------------------------------------------------------------------- regressors
def set_torch(seed: int):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch


def mlp_fit_predict(X, y, Xte_list, hp_list, seed):
    """Train one MLP per (hidden, wd); return predictions at each requested epoch checkpoint.

    hp_list share hidden/wd and differ only in epochs. Returns list (aligned with hp_list) of lists of arrays."""
    torch = set_torch(seed)
    hidden = [int(h) for h in hp_list[0]["hidden"].split("-")]
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=hp_list[0]["weight_decay"])
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    tes = [torch.as_tensor(x, dtype=torch.float32) for x in Xte_list]
    want = {hp["epochs"]: i for i, hp in enumerate(hp_list)}
    out = [None] * len(hp_list)
    for ep in range(1, max(want) + 1):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(Xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
        if ep in want:
            net.eval()
            with torch.no_grad():
                out[want[ep]] = [net(t).squeeze(-1).numpy().astype(float) for t in tes]
    return out


def make_sklearn(method: str, hp: dict, seed: int, p: int):
    if method == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=hp["alpha"])
    if method == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=hp["alpha"], max_iter=100_000)
    if method == "svr":
        from sklearn.svm import SVR
        g = "scale" if hp["gamma"] == "scale" else 0.3 / p
        return SVR(kernel="rbf", C=hp["C"], epsilon=hp["epsilon"], gamma=g)
    if method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=500, max_depth=hp["max_depth"],
                                     min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
                                     random_state=seed, n_jobs=1)
    if method == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                            learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                            device="cpu", n_jobs=1, random_state=seed)
    raise ValueError(method)


SCALE_Y = {"ridge", "lasso", "svr", "mlp"}


def regressor_fit_predict(method, X, y, Xte_list, hps, seed):
    """Fit on (X, y) for every hp in hps; return list over hps of list over Xte_list of predictions."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs, tes = (X - mu) / sd, [(x - mu) / sd for x in Xte_list]
    if method in SCALE_Y:
        ym, ys = y.mean(), y.std()
        ys = ys if ys > 1e-12 else 1.0
    else:
        ym, ys = 0.0, 1.0
    yz = (y - ym) / ys
    if method == "mlp":
        res = [None] * len(hps)
        groups = {}
        for i, hp in enumerate(hps):
            groups.setdefault((hp["hidden"], hp["weight_decay"]), []).append(i)
        for idx in groups.values():
            outs = mlp_fit_predict(Xs, yz, tes, [hps[i] for i in idx], seed)
            for i, o in zip(idx, outs):
                res[i] = [v * ys + ym for v in o]
        return res
    res = []
    for hp in hps:
        m = make_sklearn(method, hp, seed, X.shape[1])
        m.fit(Xs, yz)
        res.append([m.predict(t).astype(float) * ys + ym for t in tes])
    return res


def regressor_cv_folds(bundle, train_mask, tier):
    r = bundle.rows
    idx = np.flatnonzero(train_mask)
    folds = []
    if tier == "full":
        from sklearn.model_selection import GroupKFold
        g = r.pair.to_numpy()[idx]
        for tr, va in GroupKFold(n_splits=5).split(idx, groups=g):
            folds.append((idx[tr], idx[va]))
    else:
        known = sorted(set(r.victim[idx]) | set(r.corunner[idx]))
        for a in known:
            inv = (r.victim.to_numpy()[idx] == a).astype(int) + (r.corunner.to_numpy()[idx] == a).astype(int)
            folds.append((idx[inv == 0], idx[inv == 1]))
    return folds


def run_regressor(bundle, method, form, target, socket, tier, split, seeds, masks):
    X, _ = bundle.features(target, form, socket)
    y = bundle.rows[f"y_{target}"].to_numpy(float)
    train = masks["train"]
    grid = GRIDS[method]
    # ---- tune (seed 0)
    folds = regressor_cv_folds(bundle, train, tier)
    abs_err = np.zeros(len(grid))
    n_val = 0
    for tr, va in folds:
        preds = regressor_fit_predict(method, X[tr], y[tr], [X[va]], grid, TUNE_SEED)
        for i, pr in enumerate(preds):
            abs_err[i] += np.abs(y[va] - pr[0]).sum()
        n_val += len(va)
    cv = abs_err / n_val
    best = int(np.argmin(cv))
    hp_tab = pd.DataFrame([{**{f"hp_{k}": v for k, v in g.items()}, "cv_log_mae": c, "chosen": i == best,
                            "n_folds": len(folds), "n_val": n_val} for i, (g, c) in enumerate(zip(grid, cv))])
    # ---- fit + predict
    test_names = [k for k in masks if k != "train"]
    fit_seeds = [seeds[0]] if method in DETERMINISTIC else seeds
    out = []
    for s in fit_seeds:
        preds = regressor_fit_predict(method, X[train], y[train], [X[masks[k]] for k in test_names], [grid[best]], s)[0]
        for k, pr in zip(test_names, preds):
            out.append(pred_frame(bundle, masks[k], pr, k, s, method in DETERMINISTIC))
    return pd.concat(out, ignore_index=True), hp_tab


def pred_frame(bundle, mask, pred, test_tier, seed, det):
    r = bundle.rows[mask]
    return pd.DataFrame({"row_id": r.row_id.to_numpy(), "test_tier": test_tier, "seed": seed,
                         "deterministic": det, "y_pred": np.asarray(pred, float)})


# ----------------------------------------------------------------------------- response models
class ResponseData:
    """Scaled profiles (+ socket bits) for apps and panel inhibitors; App-Inh cells of a set of apps."""

    def __init__(self, bundle: Bundle, train_apps, target, socket):
        frame = pd.concat([bundle.apps.loc[train_apps], bundle.inh], ignore_index=True)
        v = bundle.phys(frame)
        self.mu, self.sd = v.mean(0), v.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        self.app_prof = {a: (bundle.phys(bundle.apps.loc[[a]])[0] - self.mu) / self.sd for a in APPS}
        self.inh_prof = {c: (bundle.phys(bundle.inh.loc[[c]])[0] - self.mu) / self.sd for c in bundle.panel}
        self.aware = socket == "aware"
        self.target = target
        self.cells = bundle.resolved

    def bit(self, s):
        return float(s) if self.aware else 0.0

    def victim_vec(self, app, s):
        return np.append(self.app_prof[app], self.bit(s))

    def inh_vec(self, c, s_victim):
        return np.append(self.inh_prof[c], self.bit(1 - s_victim))

    def app_aggr_vec(self, app, s_victim):
        return np.append(self.app_prof[app], self.bit(1 - s_victim))

    def cell_arrays(self, apps):
        c = self.cells[self.cells.app.isin(apps)].reset_index(drop=True)
        V = np.stack([self.victim_vec(a, s) for a, s in zip(c.app, c.s_V)])
        A = np.stack([self.inh_vec(k, s) for k, s in zip(c.config, c.s_V)])
        return c, V, A, c[f"y_{self.target}"].to_numpy(float)


def make_potential(torch):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4),
                                           torch.nn.ReLU())

        def forward(self, x):
            return self.net(x)

    class LowRankPotential(torch.nn.Module):
        """R(V, X) = h(E(X)) + <u(E(V)), v(E(X))> (legacy architecture, input 4 base + socket bit)."""

        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.global_head = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
            self.victim_head = torch.nn.Linear(4, 2)
            self.aggressor_head = torch.nn.Linear(4, 2)

        def forward(self, v, x):
            ev, ex = self.encoder(v), self.encoder(x)
            return self.global_head(ex).squeeze(-1) + (self.victim_head(ev) * self.aggressor_head(ex)).sum(-1)

    return LowRankPotential()


def train_response(kind, V, A, y, groups, lr, wd, checkpoints, seed, on_checkpoint):
    """Legacy sampling: victim app uniformly, then (absolute) one of its cells / (delta) two distinct cells."""
    torch = set_torch(seed)
    model = make_potential(torch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    apps = sorted(groups)
    members = [np.asarray(groups[a]) for a in apps]
    Vt, At, yt = (torch.as_tensor(np.array(x, dtype=np.float32)) for x in (V, A, y))
    huber = torch.nn.functional.huber_loss
    for ep in range(1, max(checkpoints) + 1):
        model.train()
        for _ in range(RESP_BATCHES):
            who = rng.integers(len(apps), size=RESP_BATCH)
            if kind == "absolute":
                i = np.array([members[w][rng.integers(len(members[w]))] for w in who])
                ii = torch.as_tensor(i)
                pred = model(Vt[ii], At[ii])
                loss = huber(pred, yt[ii], delta=1.0)
            else:
                pairs = [rng.choice(members[w], 2, replace=False) for w in who]
                ia = torch.as_tensor(np.array([p[0] for p in pairs]))
                iq = torch.as_tensor(np.array([p[1] for p in pairs]))
                pred = model(Vt[iq], At[iq]) - model(Vt[ia], At[ia])
                loss = huber(pred, yt[iq] - yt[ia], delta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in checkpoints:
            model.eval()
            on_checkpoint(ep, model, torch)
    return model


def potentials(model, torch, V, X):
    with torch.no_grad():
        return model(torch.as_tensor(V, dtype=torch.float32), torch.as_tensor(X, dtype=torch.float32)).numpy()


def aggregate(res, dist, agg):
    if agg == "uniform":
        return float(res.mean())
    if agg == "median":
        return float(np.median(res))
    T = float(agg.split("_")[1])
    lg = -(dist ** 2) / T
    w = np.exp(lg - lg.max())
    return float((w / w.sum()) @ res)


AGGS = ["uniform", "median", "kernel_0.3", "kernel_1", "kernel_3"]


def delta_predict(model, torch, rd: ResponseData, victim, s_v, aggr_prof_scaled, aggr_vec, anchor_cells):
    """anchor_cells: DataFrame of the victim's App-Inh cells used as anchors. Returns {agg: prediction}."""
    Va = np.stack([rd.victim_vec(victim, s) for s in anchor_cells.s_V])
    Aa = np.stack([rd.inh_vec(c, s) for c, s in zip(anchor_cells.config, anchor_cells.s_V)])
    res = anchor_cells[f"y_{rd.target}"].to_numpy(float) - potentials(model, torch, Va, Aa)
    dist = np.array([np.linalg.norm(rd.inh_prof[c] - aggr_prof_scaled) for c in anchor_cells.config])
    base = float(potentials(model, torch, rd.victim_vec(victim, s_v)[None], aggr_vec[None])[0])
    return {a: base + aggregate(res, dist, a) for a in AGGS}


def run_response(bundle, kind, target, socket, tier, split, seeds, masks, train_apps):
    grid = GRIDS[kind]
    train_keys = sorted({(g["lr"], g["weight_decay"]) for g in grid})
    checkpoints = sorted({g["epochs"] for g in grid})
    # ---- tune: leave-one-training-app-out on App-Inh panel cells (seed 0)
    err = {}  # (lr, wd, ep, agg) -> [sum abs err, n]
    for held in train_apps:
        fit_apps = [a for a in train_apps if a != held]
        rd = ResponseData(bundle, fit_apps, target, socket)
        cells, V, A, y = rd.cell_arrays(fit_apps)
        groups = {a: np.flatnonzero(cells.app.to_numpy() == a) for a in fit_apps}
        vc, Vv, Av, yv = rd.cell_arrays([held])
        for lr, wd in train_keys:
            def cb(ep, model, torch, lr=lr, wd=wd):
                if kind == "absolute":
                    pr = potentials(model, torch, Vv, Av)
                    e = err.setdefault((lr, wd, ep, None), [0.0, 0])
                    e[0] += np.abs(yv - pr).sum()
                    e[1] += len(yv)
                    return
                for i, row in vc.iterrows():
                    anchors = vc[vc.config != row.config]
                    preds = delta_predict(model, torch, rd, held, row.s_V, rd.inh_prof[row.config],
                                          rd.inh_vec(row.config, row.s_V), anchors)
                    for a, pv in preds.items():
                        e = err.setdefault((lr, wd, ep, a), [0.0, 0])
                        e[0] += abs(yv[i] - pv)
                        e[1] += 1
            train_response(kind, V, A, y, groups, lr, wd, checkpoints, TUNE_SEED, cb)
    rows = []
    for g in grid:
        e = err[(g["lr"], g["weight_decay"], g["epochs"], g.get("agg"))]
        rows.append({**{f"hp_{k}": v for k, v in g.items()}, "cv_log_mae": e[0] / e[1], "n_folds": len(train_apps),
                     "n_val": e[1]})
    hp_tab = pd.DataFrame(rows)
    best = int(hp_tab.cv_log_mae.to_numpy().argmin())
    hp_tab["chosen"] = np.arange(len(hp_tab)) == best
    hp = grid[best]
    # ---- final fits
    rd = ResponseData(bundle, train_apps, target, socket)
    cells, V, A, y = rd.cell_arrays(train_apps)
    groups = {a: np.flatnonzero(cells.app.to_numpy() == a) for a in train_apps}
    out = []
    r = bundle.rows
    for s in seeds:
        holder = {}
        train_response(kind, V, A, y, groups, hp["lr"], hp["weight_decay"], [hp["epochs"]], s,
                       lambda ep, model, torch: holder.update(model=model, torch=torch))
        model, torch = holder["model"], holder["torch"]
        for k, m in masks.items():
            if k == "train":
                continue
            sub = r[m]
            preds = []
            for v, b, sv in zip(sub.victim, sub.corunner, sub.s_V):
                aggr_vec = rd.app_aggr_vec(b, sv)
                if kind == "absolute":
                    preds.append(float(potentials(model, torch, rd.victim_vec(v, sv)[None], aggr_vec[None])[0]))
                else:
                    anchors = rd.cells[rd.cells.app == v]  # the victim's panel probes (known or held out)
                    preds.append(delta_predict(model, torch, rd, v, sv, rd.app_prof[b], aggr_vec, anchors)[hp["agg"]])
            out.append(pred_frame(bundle, m, preds, k, s, False))
    return pd.concat(out, ignore_index=True), hp_tab


# ----------------------------------------------------------------------------- tasks
def task_name(t):
    return f"{t['tier']}_s{t['split']:03d}_{t['target']}_{t['socket']}_{t['method']}_{t['form']}"


def build_tasks(args):
    tasks = []
    for tier in args.tiers:
        splits = args.splits_full if tier == "full" else args.splits_restricted
        for split in splits:
            for target in args.targets:
                for socket in args.sockets:
                    for method in args.methods:
                        forms = ["none"] if method in RESPONSE else args.forms
                        for form in forms:
                            tasks.append(dict(tier=tier, split=split, target=target, socket=socket, method=method,
                                              form=form))
    # heavy tasks first for better packing
    weight = {"delta": 0, "absolute": 1, "mlp": 2, "rf": 3, "xgb": 4, "svr": 5, "lasso": 6, "ridge": 7}
    tasks.sort(key=lambda t: (weight[t["method"]], t["tier"], t["split"]))
    return tasks


_BUNDLE = None


def _init(bundle_dir, xgb_grid=None):
    global _BUNDLE
    _BUNDLE = Bundle(Path(bundle_dir))
    if xgb_grid is not None:  # grids must be passed explicitly: forkserver/spawn workers re-import the module
        GRIDS["xgb"] = xgb_grid


def run_task(t, out_dir: str, seeds):
    out = Path(out_dir)
    name = task_name(t)
    t0 = time.time()
    b = _BUNDLE
    if t["tier"] == "full":
        tr, te = full_split(b, t["split"])
        masks = {"train": tr, "full": te}
        seeds = [t["split"]]  # full data: seed k = split k
        train_apps = list(APPS)  # every app is known in the full-data tier
    else:
        tr, ok, zs, held = restricted_split(b, t["split"])
        masks = {"train": tr, "one_known": ok, "zero_shot": zs}
        train_apps = [a for a in APPS if a not in held]
    if t["method"] in RESPONSE:
        preds, hp = run_response(b, t["method"], t["target"], t["socket"], t["tier"], t["split"], seeds, masks,
                                 train_apps)
    else:
        preds, hp = run_regressor(b, t["method"], t["form"], t["target"], t["socket"], t["tier"], t["split"],
                                  seeds, masks)
    for k, v in t.items():
        preds.insert(0, k, v)
        hp.insert(0, k, v)
    preds.to_csv(out / "preds" / f"{name}.csv.gz", index=False, float_format="%.10g")
    hp.to_csv(out / "hp" / f"{name}.csv", index=False, float_format="%.10g")
    info = {**t, "seconds": time.time() - t0, "n_pred": len(preds)}
    (out / "done" / f"{name}.json").write_text(json.dumps(info))
    return name, info["seconds"]


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "torch", "xgboost"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            v[mod] = f"missing ({e.__class__.__name__})"
    return v


def parse_list(s, cast=str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and cast is int:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(cast(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiers", default="full,restricted")
    ap.add_argument("--splits-full", default="0-9", help="full-data splits (= seeds), e.g. 0-9")
    ap.add_argument("--splits-restricted", default="0-119")
    ap.add_argument("--seeds", default="0-9", help="init seeds for the restricted tiers")
    ap.add_argument("--methods", default=",".join(REGRESSORS + RESPONSE))
    ap.add_argument("--forms", default="1w,bd")
    ap.add_argument("--targets", default="mpi,tt")
    ap.add_argument("--sockets", default="aware,blind")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xgb-depths", default="1,2,4",
                    help="xgb max_depth grid (follow-up rerun used 1,2,4,6,8; other xgb axes unchanged)")
    ap.add_argument("--dry-run", action="store_true", help="list tasks and exit")
    args = ap.parse_args()
    args.tiers = parse_list(args.tiers)
    args.splits_full = parse_list(args.splits_full, int)
    args.splits_restricted = parse_list(args.splits_restricted, int)
    args.seeds = parse_list(args.seeds, int)
    args.methods = parse_list(args.methods)
    args.forms = parse_list(args.forms)
    args.targets = parse_list(args.targets)
    args.sockets = parse_list(args.sockets)
    depths = parse_list(args.xgb_depths, int)
    if depths != [1, 2, 4]:
        GRIDS["xgb"] = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_child_weight": w}
                        for n in (10, 30, 100, 300) for d in depths for lr in (0.003, 0.01, 0.03, 0.1)
                        for w in (1, 5)]
    assert set(args.methods) <= set(REGRESSORS + RESPONSE), args.methods
    assert all(0 <= s < 120 for s in args.splits_restricted)
    workers = max(1, min(args.workers, MAX_WORKERS, os.cpu_count() or 1))

    for d in ("preds", "hp", "done", "errors"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    todo = [t for t in tasks if not (args.out / "done" / f"{task_name(t)}.json").exists()]
    cfg = {"argv": sys.argv, "workers": workers, "versions": versions(), "n_tasks": len(tasks),
           "grids": GRIDS, "mlp_lr": MLP_LR, "resp_batches": [RESP_BATCHES, RESP_BATCH], "tune_seed": TUNE_SEED,
           "seeds": args.seeds, "bundle_manifest": json.loads((args.bundle / "manifest.json").read_text())}
    (args.out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"{len(tasks)} tasks, {len(todo)} to run, {workers} workers; versions {cfg['versions']}", flush=True)
    if args.dry_run:
        for t in todo[:20]:
            print(task_name(t))
        return
    if "xgb" in args.methods and cfg["versions"]["xgboost"].startswith("missing"):
        raise SystemExit("xgboost is not installed; drop xgb via --methods or install it")
    t_start = time.time()
    fails = 0

    def heartbeat(done, state):
        el = time.time() - t_start
        (args.out / "status.json").write_text(json.dumps({
            "state": state, "pid": os.getpid(), "workers": workers, "tasks_total": len(tasks),
            "tasks_todo_at_start": len(todo), "done_this_run": done, "failures": fails,
            "elapsed_s": round(el), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    heartbeat(0, "running")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(args.bundle), GRIDS["xgb"])) as ex:
        futs = {ex.submit(run_task, t, str(args.out), args.seeds): t for t in todo}
        for n, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                name, sec = f.result()
                print(f"[{n}/{len(todo)}] {name} {sec:.1f}s  (elapsed {time.time() - t_start:.0f}s)", flush=True)
            except Exception:  # noqa: BLE001
                fails += 1
                (args.out / "errors" / f"{task_name(t)}.txt").write_text(traceback.format_exc())
                print(f"[{n}/{len(todo)}] FAILED {task_name(t)}", flush=True)
            heartbeat(n, "running")
    heartbeat(len(todo), "finished" if not fails else "finished_with_failures")
    print(f"finished in {time.time() - t_start:.0f}s; failures {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
