# Delta Response Model Pipeline

## Overview

The Delta Response Model is a transitive slowdown inference system that predicts the slowdown of an application when co-scheduled with a new partner, using known slowdowns as "anchors." Instead of predicting absolute slowdown from scratch, this model predicts the **relative change (delta)** in log-slowdown caused by switching from a known inhibitor to a new target.

## Core Idea

Given:
- App A's **known** slowdown when co-scheduled with inhibitor J
- The communication profile of a new co-scheduling partner K

The model predicts:
```
predicted_log_slowdown(A, K | anchor J) = log_slowdown_J + [R(z_v(A), z_a(K)) - R(z_v(A), z_a(J))]
```

Where `R(z_v, z_a)` is a response score function. This approach triangulates App-App slowdown predictions using ~200 known App-Inhibitor slowdowns as anchors, even without direct App-App training data.

## Key Properties

1. **Antisymmetry**: Swapping inhibitors J and K negates the prediction automatically, since delta is computed as literal subtraction.

2. **Path-consistency**: Chaining through intermediate points cancels algebraically:
   `predicted_delta(A,J,K) = predicted_delta(A,J,X) + predicted_delta(A,X,K)`

This enables reliable transitive inference through multiple anchors.

## Architecture

### ResponseScoreModel

```
┌─────────────────────────────────────────────────────────────┐
│                    SharedEncoder                            │
│              (transforms profile → embedding)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼────────┐        ┌────────▼────────┐
│ victim_head    │        │ aggressor_head  │
│ (emb → role)   │        │ (emb → role)    │
└───────┬────────┘        └────────┬────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
              [z_v, z_a, z_v*z_a]
                      │
              ┌───────▼────────┐
              │ response_head  │
              │  → scalar R    │
              └────────────────┘
```

The response score `R` is not trained to equal `log(slowdown)` directly—it's only trained through differences. Its absolute scale is arbitrary; only differences are meaningful.

## Pipeline Components

### 1. Data Loading (`data.py`)

- **ProfileNormalizer**: Standardizes feature columns with log-transform
- **FEATURE_COLS**: `["mpi_time", "comm_frac", "total_msgs", "total_bytes"]`
- **expand_pair_df**: Converts pairwise slowdown data into directional records

### 2. Delta Pair Dataset (`delta_data.py`)

- **DeltaPairDataset**: Virtual dataset that samples `(A, J, K)` triplets on-the-fly
- Each triplet contains: victim app A, two inhibitors J and K both observed with A
- Target: `log_slowdown_K - log_slowdown_J`
- Samples ~20,000-50,000 triplets per epoch via random sampling

### 3. Training (`train_delta.py`)

**Training Procedure:**
- **Loss**: HuberLoss(delta=1.0) between predicted and target delta
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Batch size**: 256
- **Epochs**: Up to 300 with early stopping (patience=20)

**Leave-One-App-Out (LOAO) Cross-Validation:**
- For each app A_i, train on all other apps' inhibitor data
- Validate on triplets involving A_i
- Tune hyperparameters (embedding dim, hidden sizes, learning rate, temperature)
- Average results over ≥3 seeds
- Final model trained on all `job_inh_df` data

**Metrics Computed:**
- Delta MAE/RMSE (predicted vs target delta)
- Reconstructed MAE/RMSE (log_slowdown_J + predicted_delta vs true log_slowdown_K)
- Spearman correlation

### 4. Evaluation (`evaluate_delta.py`)

**Transitive App-App Evaluation:**

For each directional record `(A, B, true_slowdown)`:

1. **Gather anchors**: All `(A, J, slowdown_J)` rows from `job_inh_df`
2. **Per-anchor prediction**: 
   ```
   pred_log_slowdown_from_J = log(slowdown_J) + [R(A,B) - R(A,J)]
   ```
3. **Aggregate across anchors** using:
   - **Similarity-weighted average**: Weights based on aggressor embedding similarity
   - **Unweighted mean/median**: Simple baseline aggregation

**Aggregation Formula:**
```python
dists_sq = ((z_a_anchors - z_a_target[None, :]) ** 2).sum(dim=-1)
weights = softmax(-dists_sq / temperature, dim=0)
final_pred = sum(weights * anchor_predictions)
```

Temperature is a tunable hyperparameter that controls the sharpness of the weighting.

**Final Output:**
- `predicted_slowdown(A, B) = exp(aggregate_predictions(...))`
- Full prediction table with: victim, target, true slowdown, predicted slowdown, absolute error, anchor std (uncertainty estimate)

## Usage

### Training
```bash
python train_delta.py \
    --jobs_csv jobs.csv \
    --inhibitors_csv inhibitors.csv \
    --job_inh_csv job_inh.csv \
    --pair_csv pair.csv \
    --output_dir outputs
```

### Evaluation
```bash
python evaluate_delta.py \
    --checkpoint outputs/delta_model_seed0.pt \
    --jobs_csv jobs.csv \
    --inhibitors_csv inhibitors.csv \
    --job_inh_csv job_inh.csv \
    --pair_csv pair.csv \
    --output_dir outputs \
    --temperature 1.0
```

### Main CLI
```bash
python main_delta.py \
    --jobs_csv jobs.csv \
    --inhibitors_csv inhibitors.csv \
    --job_inh_csv job_inh.csv \
    --pair_csv pair.csv \
    --stage train \
    --emb_dim 32 --hidden 64 --role_dim 16 \
    --lr 1e-3 --batch_size 256 --max_epochs 300

python main_delta.py \
    --jobs_csv jobs.csv \
    --inhibitors_csv inhibitors.csv \
    --job_inh_csv job_inh.csv \
    --pair_csv pair.csv \
    --stage evaluate \
    --checkpoint outputs/delta_model_seed0.pt \
    --temperature 1.0
```

## File Structure

```
delta_response/
├── data.py          # CSV loading, ProfileNormalizer, expand_pair_df
├── model.py         # SharedEncoder, TwoTowerSlowdownModel, ResponseScoreModel
├── delta_data.py    # DeltaPairDataset for triplet sampling
├── train_delta.py   # LOAO CV loop + final training
├── evaluate_delta.py # Transitive evaluation on pair_df
├── main_delta.py    # CLI entry point
└── README.md        # This file
```

## Input Data Format

Four CSV files with the same schema as the baseline model:

1. **jobs.csv**: `job_id, mpi_time, comm_frac, total_msgs, total_bytes`
2. **inhibitors.csv**: `inhib_id, msg_size, wait_time, comm_sparsity, mpi_time, comm_frac, total_msgs, total_bytes`
3. **job_inh.csv**: `job_id, inhib_id, slowdown` — training data source
4. **pair.csv**: `jobA_id, jobB_id, slowdown_A, slowdown_B` — evaluation holdout

## Metrics

Evaluated in both log-space and raw space:
- **MAE/RMSE** in log-slowdown and raw slowdown
- **Spearman rank correlation**
- **Anchor standard deviation** as uncertainty estimate

Results compared side-by-side with baseline absolute model for direct comparison.

## Dependencies

```
torch
pandas
numpy
scipy (for spearmanr)
```

## Edge Cases & Considerations

1. **Apps with <2 inhibitors**: Excluded from training (can't form triplets) but kept for evaluation if they have ≥1 anchor
2. **Non-positive slowputs**: Guard against `log(0)`/`log(negative)` — assert positivity when loading data
3. **Temperature tuning**: Too small → collapses to single anchor (high variance); too large → degenerate to unweighted mean
4. **Normalization**: Use same `ProfileNormalizer` (fit on all jobs + inhibitors) for fair comparison with baseline
