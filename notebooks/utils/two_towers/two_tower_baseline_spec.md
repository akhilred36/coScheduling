# Two-Tower Baseline: MPI Co-Scheduling Slowdown Prediction

## 0. Objective

Build a PyTorch model that predicts the communication-performance slowdown of an
application when co-scheduled with another entity (an inhibitor config, or —
at evaluation time — another real application).

Train on abundant **App × Inhibitor** data. Evaluate (zero-shot, no further
training) on scarce **App × App** data. This works because inhibitors and
apps are described by the *same* communication-profile features, so a single
shared encoder can be trained mostly on inhibitors and still generalize to
real apps acting as the "aggressor."

Implement this as a **two-tower (Siamese) neural network**:
- One shared encoder maps any entity's communication profile to an embedding.
- Two small projection heads split that embedding into a "victim role"
  representation and an "aggressor role" representation.
- An interaction head combines a victim embedding and an aggressor embedding
  into a predicted slowdown.

Do not implement few-shot/anchor-based transitive inference in this file —
this spec is for the **baseline absolute-regression model only**.

---

## 1. Input data

Four CSV files. Replace the bracketed paths with the actual locations given
to you at runtime — do not hardcode assumptions about where they live.

### `jobs_df` — `<PATH_TO_JOBS_CSV>`
Isolated-run profile for each real application (~10 rows).

| column | type | description |
|---|---|---|
| `job_id` | str | application name, unique |
| `mpi_time` | float | total time spent in MPI calls |
| `comm_frac` | float | fraction of total runtime spent communicating, in [0,1] |
| `total_msgs` | int | total messages sent |
| `total_bytes` | int | total bytes sent |

### `inhibitors_df` — `<PATH_TO_INHIBITORS_CSV>`
Isolated-run profile for each synthetic inhibitor config (~200 rows).

| column | type | description |
|---|---|---|
| `inhib_id` | str | inhibitor config name, unique |
| `msg_size` | float | inhibitor control knob (synthetic — **do not use as a model feature**, see §2.1) |
| `wait_time` | float | inhibitor control knob (synthetic — **do not use as a model feature**) |
| `comm_sparsity` | float | inhibitor control knob (synthetic — **do not use as a model feature**) |
| `mpi_time` | float | same semantics as in `jobs_df` |
| `comm_frac` | float | same semantics as in `jobs_df` |
| `total_msgs` | int | same semantics as in `jobs_df` |
| `total_bytes` | int | same semantics as in `jobs_df` |

### `job_inh_df` — `<PATH_TO_JOB_INH_CSV>`
Co-scheduling result: app is the **victim**, inhibitor is the **aggressor**.
This is the main training set (~2000 rows, roughly 10 apps × 200 inhibitors).

| column | type | description |
|---|---|---|
| `job_id` | str | victim application, foreign key into `jobs_df` |
| `inhib_id` | str | aggressor inhibitor, foreign key into `inhibitors_df` |
| `slowdown` | float | slowdown of `job_id` caused by co-scheduling with `inhib_id` (≥ 1 typically, ratio of co-run time to isolated time for the communication metric) |

### `pair_df` — `<PATH_TO_PAIR_CSV>`
Real App × App co-scheduling result. This is the **final holdout evaluation
set** — never use it for training or hyperparameter tuning. 55 rows (10
choose 2, plus self-pairs where `jobA_id == jobB_id`), each row containing
both directional slowdowns.

| column | type | description |
|---|---|---|
| `jobA_id` | str | foreign key into `jobs_df` |
| `jobB_id` | str | foreign key into `jobs_df` |
| `slowdown_A` | float | slowdown of A when co-scheduled with B (A is victim, B is aggressor) |
| `slowdown_B` | float | slowdown of B when co-scheduled with A (B is victim, A is aggressor) |

---

## 2. Preprocessing

### 2.1 Feature columns — critical constraint

The shared encoder must accept **only the four features that exist for both
apps and inhibitors**:

```
FEATURE_COLS = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
```

Do **not** feed `msg_size`, `wait_time`, or `comm_sparsity` into the encoder.
Those columns only exist for inhibitors, not for real apps. If the encoder
depends on them, it cannot be applied to `pair_df` at evaluation time. It's
fine to load/keep those columns in the dataframe for inspection, just exclude
them from the feature tensor.

### 2.2 Transforms

For each of the 4 feature columns, apply in this order:

1. **Log transform** on `mpi_time`, `total_msgs`, `total_bytes` (all
   heavy-tailed, non-negative): `x' = log1p(x)`. Leave `comm_frac` untouched
   (it's already bounded in [0,1]).
2. **Standardization** (z-score): `x'' = (x' - mean) / std`.

The mean/std for standardization must be computed **once**, from the union of
`jobs_df` and `inhibitors_df` isolated profiles (after log1p), and reused
everywhere — training, validation, and the final `pair_df` evaluation. Do not
refit normalization stats per fold in a way that leaks information; refit
only on the training portion of `jobs_df`/`inhibitors_df` for each CV fold
(see §5), and always apply that fold's stats to build every entity's
embedding input including validation and eval entities.

Concretely, write a small helper:

```python
class ProfileNormalizer:
    def __init__(self, feature_cols=("mpi_time", "comm_frac", "total_msgs", "total_bytes"),
                 log_cols=("mpi_time", "total_msgs", "total_bytes")):
        self.feature_cols = list(feature_cols)
        self.log_cols = set(log_cols)
        self.mean_ = None
        self.std_ = None

    def _transform_log(self, df):
        df = df.copy()
        for c in self.log_cols:
            df[c] = np.log1p(df[c].astype(float))
        return df

    def fit(self, df):
        df = self._transform_log(df)
        self.mean_ = df[self.feature_cols].mean().values.astype(np.float32)
        self.std_ = df[self.feature_cols].std().values.astype(np.float32)
        self.std_[self.std_ < 1e-6] = 1e-6
        return self

    def transform(self, df):
        df = self._transform_log(df)
        x = df[self.feature_cols].values.astype(np.float32)
        return (x - self.mean_) / self.std_
```

Fit this once on `pd.concat([jobs_df[FEATURE_COLS], inhibitors_df[FEATURE_COLS]])`
using the training-fold subset of apps only (inhibitors are never held out —
see §5), then use `.transform()` everywhere.

### 2.3 Target transform

Model predicts `log(slowdown)`, not raw slowdown. Recover raw slowdown at
evaluation time with `exp(...)`. This keeps the loss well-scaled (slowdown is
a ratio, and errors should be judged multiplicatively, not additively).

```python
y = np.log(job_inh_df["slowdown"].values.astype(np.float32))
```

---

## 3. Building the training dataset

Merge `job_inh_df` with `jobs_df` (on `job_id`) and `inhibitors_df` (on
`inhib_id`) to produce, for every row, a `(victim_profile_vec, aggressor_profile_vec, log_slowdown)` triple.

```python
class VictimAggressorDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df, jobs_df, inhib_df, normalizer):
        # pairs_df has columns: job_id, inhib_id, slowdown
        job_feats = normalizer.transform(jobs_df.set_index("job_id").loc[pairs_df["job_id"]])
        inh_feats = normalizer.transform(inhib_df.set_index("inhib_id").loc[pairs_df["inhib_id"]])
        self.victim = torch.tensor(job_feats, dtype=torch.float32)
        self.aggressor = torch.tensor(inh_feats, dtype=torch.float32)
        self.y = torch.tensor(np.log(pairs_df["slowdown"].values.astype(np.float32)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.victim[idx], self.aggressor[idx], self.y[idx]
```

Wrap in a standard `DataLoader` (`shuffle=True` for training,
`shuffle=False` for validation), batch size given in §6.

---

## 4. Model architecture

All four raw features become a 4-dim input vector after normalization.

```python
class SharedEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden=64, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwoTowerSlowdownModel(nn.Module):
    def __init__(self, in_dim=4, hidden=64, emb_dim=32, role_dim=16, interaction_hidden=32):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, hidden, emb_dim)   # shared weights, used for BOTH roles
        self.victim_head = nn.Linear(emb_dim, role_dim)
        self.aggressor_head = nn.Linear(emb_dim, role_dim)
        self.interaction = nn.Sequential(
            nn.Linear(role_dim * 3, interaction_hidden),  # [z_v, z_a, z_v * z_a]
            nn.ReLU(),
            nn.Linear(interaction_hidden, 1),
        )

    def forward(self, victim_profile, aggressor_profile):
        z_victim_full = self.encoder(victim_profile)
        z_aggr_full = self.encoder(aggressor_profile)   # SAME encoder call, not a separate module
        z_v = self.victim_head(z_victim_full)
        z_a = self.aggressor_head(z_aggr_full)
        interaction_input = torch.cat([z_v, z_a, z_v * z_a], dim=-1)
        log_slowdown_pred = self.interaction(interaction_input).squeeze(-1)
        return log_slowdown_pred
```

**Important implementation detail**: `self.encoder` must be a single module
instance called on both `victim_profile` and `aggressor_profile` — this is
what makes it a Siamese network. Do not create two separate encoder modules
with independently-initialized weights.

---

## 5. Training procedure

- **Loss**: `nn.HuberLoss(delta=1.0)` (more robust to outlier slowdowns than
  plain MSE) applied to `(log_slowdown_pred, log_slowdown_true)`.
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)`.
- **Batch size**: 128 (dataset is small, ~2000 rows — one epoch is a handful
  of batches).
- **Epochs**: up to 300, with early stopping (patience 20 epochs on
  validation loss, restore best checkpoint).
- **Seed everything** (`torch.manual_seed`, `np.random.seed`) for
  reproducibility; report results averaged over at least 3 seeds.

### 5.1 Model selection via Leave-One-App-Out cross-validation

Because there are only 10 apps, use **Leave-One-App-Out (LOAO)** CV on
`job_inh_df` for hyperparameter tuning and early-stopping decisions:

For each of the 10 apps `A_i`:
1. Training fold = all rows of `job_inh_df` where `job_id != A_i`.
2. Validation fold = all rows of `job_inh_df` where `job_id == A_i`.
3. Fit `ProfileNormalizer` on `jobs_df` rows excluding `A_i`, plus all of
   `inhibitors_df` (inhibitors are never excluded — there are enough of them
   that they don't need to be held out, and this maximizes normalizer
   stability).
4. Train the model on the training fold, track validation loss on the held
   out app's rows, apply early stopping.
5. Record validation metrics (§7) for `A_i`.

Average metrics across all 10 folds to select hyperparameters. This
tests whether the model generalizes to a victim app's profile it has never
seen — the closest proxy available for the real generalization target
(unseen aggressor profiles) without touching `pair_df`.

After hyperparameters are chosen, train one **final model** on *all* of
`job_inh_df` (no held-out app), using early stopping against a random 10%
row-level split purely for stopping, not for tuning.

**Do not use `pair_df` at any point during steps 5.1 — it is reserved
entirely for final evaluation in §6.**

---

## 6. Final evaluation on `pair_df`

`pair_df` gives one row per unordered app pair with both directional
slowdowns. Expand it into two directional records before scoring, matching
the `(victim, aggressor, slowdown)` shape used in training:

```python
def expand_pair_df(pair_df):
    a_as_victim = pair_df.rename(columns={
        "jobA_id": "job_id", "jobB_id": "inhib_id_like", "slowdown_A": "slowdown"
    })[["job_id", "inhib_id_like", "slowdown"]]

    b_as_victim = pair_df.rename(columns={
        "jobB_id": "job_id", "jobA_id": "inhib_id_like", "slowdown_B": "slowdown"
    })[["job_id", "inhib_id_like", "slowdown"]]

    directional = pd.concat([a_as_victim, b_as_victim], ignore_index=True)
    directional = directional.rename(columns={"inhib_id_like": "aggressor_job_id"})
    return directional  # columns: job_id (victim), aggressor_job_id (aggressor), slowdown
```

This yields up to 110 directional rows (fewer if self-pairs, where
`jobA_id == jobB_id`, are excluded — decide whether to include self-pairs;
if included, both `victim_profile` and `aggressor_profile` come from the same
row of `jobs_df`).

Build victim and aggressor feature tensors by looking up **both** columns in
`jobs_df` (not `inhibitors_df` — every entity here is a real app), apply the
same `ProfileNormalizer` used at training time, run the final model forward,
and compare `exp(log_slowdown_pred)` against the true `slowdown` column.

---

## 7. Metrics

Report all of the following, both per-LOAO-fold (§5.1, on
`job_inh_df`) and on the final `pair_df` holdout (§6):

- **MAE** and **RMSE** in log-slowdown space (`log_pred` vs `log_true`).
- **MAE** and **RMSE** in raw slowdown space (`exp(log_pred)` vs raw
  `slowdown`), since that's the human-interpretable quantity.
- **Spearman rank correlation** between predicted and true slowdown — useful
  given how small the `pair_df` holdout is (55–110 points), where a rank
  metric is more robust than an error metric.
- Print/save a full table of individual `pair_df` predictions (victim,
  aggressor, true slowdown, predicted slowdown, absolute error) for manual
  inspection — this is small enough to eyeball entirely.

---

## 8. Suggested project layout

```
project/
  data.py        # loading CSVs, ProfileNormalizer, VictimAggressorDataset, expand_pair_df
  model.py        # SharedEncoder, TwoTowerSlowdownModel
  train.py        # LOAO cross-validation loop, final model training, checkpoint saving
  evaluate.py      # loads final checkpoint, runs §6/§7 evaluation on pair_df, prints report
  utils.py         # seeding, metric functions (mae, rmse, spearman)
  main.py          # CLI entry point: python main.py --jobs_csv ... --inhibitors_csv ... --job_inh_csv ... --pair_csv ...
```

`main.py` should accept the four CSV paths as command-line arguments (do not
hardcode paths), run training end-to-end, and print the final `pair_df`
evaluation report.

---

## 9. Dependencies

```
torch
pandas
numpy
scipy        # for spearmanr
```

---

## 10. Edge cases and gotchas

- **Self-pairs** (`jobA_id == jobB_id` in `pair_df`): decide up front whether
  to include them in evaluation; if included, victim and aggressor profiles
  are identical vectors, which is a valid (if degenerate) input to the model.
- **Missing co-scheduling combinations**: `job_inh_df` may not be a complete
  10×200 grid — do not assume every app has a row for every inhibitor; build
  the dataset from whatever rows actually exist via the merge, don't try to
  construct a dense grid.
- **`total_msgs` / `total_bytes` of zero**: `log1p(0) = 0` is fine, no
  special-casing needed.
- **Normalizer leakage**: never fit `ProfileNormalizer` using `pair_df` or
  using the held-out app's isolated row within a LOAO fold.
- **Do not implement any few-shot/anchor-based delta model here** — that is
  a separate, more advanced approach; this spec is the baseline two-tower
  regression model only.
