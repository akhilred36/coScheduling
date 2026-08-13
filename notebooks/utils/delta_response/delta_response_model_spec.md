# Relative/Delta Model: Transitive Slowdown Inference via Known Anchors

## 0. Objective

This is the second model in a two-part system (the first being the absolute
two-tower baseline — see `two_tower_baseline_spec.md`). It answers a
different question:

> Given App A's *known* slowdown when co-scheduled with some inhibitor J,
> and the communication profile of a new co-scheduling partner K, predict
> App A's slowdown against K.

Instead of predicting slowdown from scratch, this model predicts the
**relative change (delta)** in log-slowdown caused by switching the
aggressor from J (known outcome) to K (unknown outcome), then adds that
delta onto the known value. At evaluation time, this lets you triangulate an
App-App slowdown prediction using the ~200 known App-Inhibitor slowdowns as
"anchors," even though no App-App training data was used.

Reuse the data-loading and normalization code from the baseline spec
(`ProfileNormalizer`, `FEATURE_COLS`, CSV schemas) — do not reimplement
those, import/reuse them.

---

## 1. Input data

Same four CSVs as the baseline spec. Re-state the paths as CLI arguments:

- `<PATH_TO_JOBS_CSV>` → `jobs_df`: `job_id, mpi_time, comm_frac, total_msgs, total_bytes`
- `<PATH_TO_INHIBITORS_CSV>` → `inhibitors_df`: `inhib_id, msg_size, wait_time, comm_sparsity, mpi_time, comm_frac, total_msgs, total_bytes`
- `<PATH_TO_JOB_INH_CSV>` → `job_inh_df`: `job_id, inhib_id, slowdown` — this is the sole source of training data for this model.
- `<PATH_TO_PAIR_CSV>` → `pair_df`: `jobA_id, jobB_id, slowdown_A, slowdown_B` — final holdout, used only for evaluation exactly as in the baseline spec (§6 there).

As with the baseline, only `FEATURE_COLS = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]` are ever fed into the encoder. Never use `msg_size`, `wait_time`, `comm_sparsity` as model inputs.

---

## 2. Core architectural idea: a response score, not a direct predictor

Define a **response score** `R(z_v, z_a)` — a scalar function of a victim
embedding and an aggressor embedding — using the *same* shared-encoder +
role-head structure as the baseline model (`SharedEncoder`, `victim_head`,
`aggressor_head`; you may either import and reuse the baseline's trained
encoder as a warm start, or train a fresh one end-to-end here — implement
fresh training by default, and leave a `--init_from_baseline_ckpt` optional
CLI flag as a nice-to-have, not required).

The key design constraint: **predict the delta between two aggressors as
the difference of two response scores evaluated with the same victim**:

```
predicted_delta(A, J, K) = R(z_v(A), z_a(K)) - R(z_v(A), z_a(J))
```

This has two properties worth preserving exactly, because the transitive
inference in §6 depends on them:

1. **Antisymmetry**: swapping J and K negates the prediction automatically,
   since it's a literal subtraction. Do not implement the delta as an
   independent MLP over `[z_v, z_j, z_k]` — that would not guarantee
   antisymmetry and would make anchors inconsistent with each other.
2. **Path-consistency**: because `predicted_delta(A,J,K) = R(z_v,z_k) - R(z_v,z_j)`,
   chaining through a third point cancels algebraically:
   `predicted_delta(A,J,K) = predicted_delta(A,J,X) + predicted_delta(A,X,K)`
   for any X. This is what makes "transitive" inference through many anchors
   sensible rather than arbitrary.

The final predicted log-slowdown, grounded in a known observation
`log_slowdown_J`, is a **residual correction on top of the known value**:

```
predicted_log_slowdown(A, K | anchor J) = log_slowdown_J + [R(z_v(A), z_a(K)) - R(z_v(A), z_a(J))]
```

```python
class ResponseScoreModel(nn.Module):
    def __init__(self, in_dim=4, hidden=64, emb_dim=32, role_dim=16, score_hidden=32):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, hidden, emb_dim)  # same class as baseline; single shared instance
        self.victim_head = nn.Linear(emb_dim, role_dim)
        self.aggressor_head = nn.Linear(emb_dim, role_dim)
        self.response_head = nn.Sequential(
            nn.Linear(role_dim * 3, score_hidden),  # [z_v, z_a, z_v * z_a], same interaction shape as baseline
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

    def response_score(self, victim_profile, aggressor_profile):
        z_v = self.victim_head(self.encoder(victim_profile))
        z_a = self.aggressor_head(self.encoder(aggressor_profile))
        interaction_input = torch.cat([z_v, z_a, z_v * z_a], dim=-1)
        return self.response_head(interaction_input).squeeze(-1)  # shape (batch,)

    def forward(self, victim_profile, aggressor_j_profile, aggressor_k_profile):
        r_j = self.response_score(victim_profile, aggressor_j_profile)
        r_k = self.response_score(victim_profile, aggressor_k_profile)
        return r_k - r_j  # predicted delta = predicted_log_slowdown_k - predicted_log_slowdown_j
```

Note `response_score` alone is *not* trained to equal `log(slowdown)`
directly — it's only ever trained through differences (see §4). Its absolute
scale is arbitrary; only differences of it are meaningful. This is
intentional: it lets per-app systematic biases (e.g. a fixed measurement
offset for a particular app) cancel out of the loss instead of having to be
modeled.

---

## 3. Building the delta-pair training set

Training examples are **triplets of two inhibitors observed against the same
app**: `(A, J, K)` where `job_inh_df` has rows for both `(A, J)` and `(A, K)`.
Per app with ~200 inhibitors, that's up to `C(200, 2) ≈ 19,900` possible
triplets, and with 10 apps up to ~199,000 total — do not materialize this
combinatorially as a static tensor. Instead, sample triplets on the fly.

```python
class DeltaPairDataset(torch.utils.data.Dataset):
    """
    Randomly samples (A, J, K) triplets where J, K are two different
    inhibitors both co-scheduled with the same app A, from job_inh_df.
    Virtual dataset: __len__ controls how many samples are drawn per epoch,
    not a fixed materialized set.
    """
    def __init__(self, job_inh_df, jobs_df, inhib_df, normalizer,
                 samples_per_epoch=20000, seed=0):
        self.job_inh_df = job_inh_df.reset_index(drop=True)
        self.by_job = {
            job_id: g.reset_index(drop=True)
            for job_id, g in job_inh_df.groupby("job_id")
            if len(g) >= 2   # need at least 2 inhibitors observed for this app
        }
        self.job_ids = list(self.by_job.keys())
        self.jobs_feat = normalizer.transform(jobs_df.set_index("job_id"))
        self.jobs_index = {jid: i for i, jid in enumerate(jobs_df["job_id"])}
        self.inhib_feat = normalizer.transform(inhib_df.set_index("inhib_id"))
        self.inhib_index = {iid: i for i, iid in enumerate(inhib_df["inhib_id"])}
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        job_id = self.rng.choice(self.job_ids)
        group = self.by_job[job_id]
        j_idx, k_idx = self.rng.choice(len(group), size=2, replace=False)
        row_j, row_k = group.iloc[j_idx], group.iloc[k_idx]

        victim = self.jobs_feat[self.jobs_index[job_id]]
        aggr_j = self.inhib_feat[self.inhib_index[row_j["inhib_id"]]]
        aggr_k = self.inhib_feat[self.inhib_index[row_k["inhib_id"]]]

        log_slowdown_j = np.log(row_j["slowdown"])
        log_slowdown_k = np.log(row_k["slowdown"])
        target_delta = log_slowdown_k - log_slowdown_j

        return (
            torch.tensor(victim, dtype=torch.float32),
            torch.tensor(aggr_j, dtype=torch.float32),
            torch.tensor(aggr_k, dtype=torch.float32),
            torch.tensor(target_delta, dtype=torch.float32),
        )
```

Re-sampling with `np.random.default_rng` inside `__getitem__` means each
epoch effectively sees a fresh random set of triplets. Set `samples_per_epoch`
to something like 20,000–50,000; this is a hyperparameter, not a hard
requirement.

---

## 4. Training procedure

- **Loss**: `nn.HuberLoss(delta=1.0)` between `model(victim, aggr_j, aggr_k)`
  (predicted delta) and `target_delta`.
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)`.
- **Batch size**: 256 (triplets are cheap; can afford a larger batch than the
  baseline).
- **Epochs**: up to 300 with early stopping (patience 20 epochs on
  validation loss).
- Seed everything, average results over ≥3 seeds, same as baseline.

### 4.1 Leave-One-App-Out cross-validation

Identical philosophy to the baseline spec: for each app `A_i`, build the
`DeltaPairDataset` only from `job_inh_df` rows where `job_id != A_i` for
training, and only from rows where `job_id == A_i` for validation triplets
(sample validation triplets the same way, but fix the seed so the validation
set is stable across epochs — e.g. materialize a fixed 2,000-triplet
validation set once per fold rather than resampling it).

Fit `ProfileNormalizer` per fold exactly as in the baseline (excluding
`A_i`'s isolated row from the app-side fit, keeping all inhibitors).

Use LOAO results to pick hyperparameters (embedding dim, hidden sizes,
learning rate, `samples_per_epoch`). Then train one final model on all of
`job_inh_df`. **Never touch `pair_df` during this phase.**

---

## 5. Direct (non-transitive) sanity check

Before moving to the App-App evaluation, sanity-check the delta model on its
own held-out triplets from §4.1: report MAE/RMSE of `predicted_delta` vs
`target_delta`, and MAE/RMSE of the reconstructed
`log_slowdown_J + predicted_delta` vs the true `log_slowdown_K`. This isolates
model quality from the anchor-aggregation logic in §6.

---

## 6. Transitive App-App evaluation on `pair_df`

For each directional evaluation record `(A, B, true_slowdown)` built by
`expand_pair_df` (reuse from the baseline spec — same function, same
directional expansion), do the following:

### 6.1 Gather anchors

Collect all rows of `job_inh_df` where `job_id == A`. Each such row
`(A, J, slowdown_J)` is one candidate anchor. (With ~200 inhibitors, there
should be close to 200 anchors per app — use all of them, don't subsample at
evaluation time.)

### 6.2 Per-anchor prediction

For every anchor J:

```python
z_a_J = aggressor_head(encoder(profile(J)))     # from inhibitors_df
z_a_B = aggressor_head(encoder(profile(B)))     # from jobs_df — B is a real app here
r_J = response_head(victim=A, aggressor=J)
r_B = response_head(victim=A, aggressor=B)
predicted_log_slowdown_from_J = log(slowdown_J) + (r_B - r_J)
```

(This is exactly `forward()` from §2 with `aggressor_j=J, aggressor_k=B`,
plus adding back `log(slowdown_J)`.)

### 6.3 Aggregate across anchors

Combine all per-anchor predictions into a single estimate using a
**similarity-weighted average**, where an anchor J is trusted more if its
aggressor embedding is close to B's aggressor embedding:

```python
def aggregate_predictions(anchor_log_preds, z_a_anchors, z_a_target, temperature=1.0):
    """
    anchor_log_preds: (num_anchors,) predicted log-slowdown from each anchor
    z_a_anchors: (num_anchors, role_dim) aggressor embeddings of each anchor inhibitor
    z_a_target: (role_dim,) aggressor embedding of the target (App B)
    """
    dists_sq = ((z_a_anchors - z_a_target[None, :]) ** 2).sum(dim=-1)
    weights = torch.softmax(-dists_sq / temperature, dim=0)
    return (weights * anchor_log_preds).sum()
```

`temperature` is a hyperparameter — tune it on the LOAO validation
folds by holding out one app's own row from `job_inh_df` at a time, treating
it as a fake "target," and checking whether the anchor-weighted prediction
(using that app's *other* inhibitor rows as anchors) recovers its true
slowdown well. This gives you a way to tune `temperature` without ever
touching `pair_df`.

Implement and report **both**:
- unweighted mean/median over all anchors (simplest possible aggregation,
  useful as an ablation), and
- the similarity-weighted average above.

### 6.4 Final prediction

`predicted_slowdown(A, B) = exp(aggregate_predictions(...))`. Compare against
`true_slowdown` from `pair_df` using the same metrics as the baseline (§7
below).

---

## 7. Metrics

Same as the baseline spec:
- MAE/RMSE in log-slowdown space and raw slowdown space.
- Spearman rank correlation.
- Full per-pair prediction table for `pair_df` (victim, target, true
  slowdown, predicted slowdown, absolute error, and — useful for debugging —
  the standard deviation of per-anchor predictions before aggregation, as a
  crude uncertainty estimate; large spread across anchors means B's profile
  wasn't well represented by the inhibitor sweep).

Also report metrics from both the baseline (absolute) model and this delta
model side by side on the same `pair_df` rows, so the two approaches are
directly comparable.

---

## 8. Suggested project layout (extends the baseline layout)

```
project/
  data.py          # (shared) CSV loading, ProfileNormalizer, expand_pair_df
  model.py         # (shared) SharedEncoder; baseline's TwoTowerSlowdownModel; this file's ResponseScoreModel
  delta_data.py    # DeltaPairDataset
  train_delta.py   # LOAO CV loop + final training for ResponseScoreModel
  evaluate_delta.py # loads final checkpoint, runs §6/§7 transitive evaluation on pair_df
  main_delta.py    # CLI entry point: python main_delta.py --jobs_csv ... --inhibitors_csv ... --job_inh_csv ... --pair_csv ...
```

Reuse `SharedEncoder` and `ProfileNormalizer` from the baseline's `model.py`/`data.py` rather than duplicating them.

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

- **Apps with fewer than 2 inhibitor rows** in `job_inh_df` can't form
  training triplets — exclude them from `DeltaPairDataset` (handled by the
  `len(g) >= 2` filter above) but keep them for §6 evaluation only if they
  have at least one anchor.
- **Anchor with `slowdown_J == 0` or negative**: shouldn't occur physically
  (slowdown should be ≥ some positive value), but guard against
  `log(0)`/`log(negative)` producing `-inf`/`nan` — assert positivity when
  loading `job_inh_df`.
- **`temperature` too small**: softmax collapses to a single nearest anchor,
  which is high-variance; too large and it degenerates to an unweighted mean,
  discarding useful similarity information. Treat it as a real hyperparameter
  to sweep, not a fixed constant.
- **Self-pairs in `pair_df`** (`jobA_id == jobB_id`): B's own profile is
  identical to A's; anchors are still A's own `job_inh_df` rows, this works
  without special-casing.
- **Don't retrain `ProfileNormalizer` separately from the baseline's** unless
  intentionally — for a fair side-by-side comparison in §7, use the same
  final normalizer (fit on all of `jobs_df` + `inhibitors_df`) for both
  models' final evaluation runs.
