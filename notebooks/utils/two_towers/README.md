# Two-Tower Baseline: MPI Co-Scheduling Slowdown Prediction

A PyTorch implementation of a two-tower neural network that predicts communication-performance slowdown when applications are co-scheduled.

## Overview

This pipeline predicts the slowdown an application experiences when co-scheduled with another entity (either a synthetic inhibitor config or another real application). The model is trained on abundant **App × Inhibitor** data and evaluated zero-shot on scarce **App × App** data.

## Architecture

The model uses a Siamese two-tower architecture:

- **Shared Encoder**: Maps any entity's communication profile to a 32-dim embedding using a 3-layer MLP (4→64→64→32)
- **Role Projections**: Two separate heads map the shared embedding into victim and aggressor role representations (32→16)
- **Interaction Head**: Combines victim and aggressor embeddings with their element-wise product to predict log-slowdown

```
Communication Profile (4 features)
            ↓
    Shared Encoder (4 → 64 → 64 → 32)
            ↓
    ┌───────┴───────┐
    ↓               ↓
Victim Head     Aggressor Head
(32 → 16)       (32 → 16)
    └───────┬───────┘
            ↓
  Concat([z_v, z_a, z_v * z_a])
            ↓
    Interaction Network
    (48 → 32 → 1)
            ↓
     Log Slowdown
```

## Input Data

Four CSV files are required:

| File | Description | Rows |
|------|-------------|------|
| `jobs.csv` | Isolated-run profiles for real applications | ~10 |
| `inhibitors.csv` | Isolated-run profiles for synthetic inhibitors | ~200 |
| `job_inh.csv` | Co-scheduling results (app victim × inhibitor aggressor) | ~2000 |
| `pair.csv` | Real App × App co-scheduling (evaluation holdout) | 55 |

### Feature Columns

Only 4 features are used as model input (shared by both apps and inhibitors):
- `mpi_time`: Total time spent in MPI calls
- `comm_frac`: Fraction of runtime spent communicating [0,1]
- `total_msgs`: Total messages sent
- `total_bytes`: Total bytes sent

**Note**: Inhibitor-specific control knobs (`msg_size`, `wait_time`, `comm_sparsity`) are excluded from the model.

## Preprocessing

1. **Log transform** on `mpi_time`, `total_msgs`, `total_bytes`: `x' = log1p(x)`
2. **Standardization** (z-score) using mean/std computed from training data

Target is `log(slowdown)` for stable training.

## Training Procedure

- **Loss**: Huber Loss (δ=1.0) on log-slowdown
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Batch size**: 128
- **Early stopping**: Patience=20 epochs, max 300 epochs

### Leave-One-App-Out Cross-Validation

Due to limited data (~10 apps), LOAO CV is used:
- Each fold holds out all pairs involving one app
- Tests generalization to unseen victim apps
- Averaged over 3+ random seeds (default: 42, 123, 456)

### Final Model

Trained on all `job_inh.csv` data with a random 90/10 train/val split for early stopping.

## Evaluation

Evaluated on `pair.csv` (never used in training/tuning):
- Expands bidirectional pairs into directional (victim, aggressor) format
- Predicts slowdown for all app pairs
- Reports metrics in both log and raw slowdown space

### Metrics

- **MAE/RMSE** in log-slowdown space
- **MAE/RMSE** in raw slowdown space
- **Spearman rank correlation**

## Project Structure

```
two_towers/
├── main.py       # CLI entry point, full pipeline
├── data.py       # Data loading, normalizer, datasets
├── model.py      # TwoTowerSlowdownModel architecture
├── train.py      # Training loops, LOAO CV
├── evaluate.py   # Standalone evaluation script
└── utils.py      # Seeding, metric computation
```

## Usage

```bash
python main.py \
  --jobs_csv data/jobs.csv \
  --inhibitors_csv data/inhibitors.csv \
  --job_inh_csv data/job_inh.csv \
  --pair_csv data/pair.csv \
  --hidden 64 \
  --emb_dim 32 \
  --role_dim 16 \
  --lr 1e-3 \
  --batch_size 128 \
  --patience 20 \
  --max_epochs 300 \
  --seeds 42 123 456
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden` | 64 | Hidden layer size in encoder |
| `--emb_dim` | 32 | Embedding dimension |
| `--role_dim` | 16 | Role projection dimension |
| `--lr` | 1e-3 | Learning rate |
| `--batch_size` | 128 | Batch size |
| `--patience` | 20 | Early stopping patience |
| `--max_epochs` | 300 | Maximum training epochs |
| `--seeds` | 42 123 456 | Random seeds for CV |

## Dependencies

```
torch
pandas
numpy
scipy
tqdm
```

## Key Design Decisions

1. **Shared encoder**: Ensures inhibitors and apps use the same feature space, enabling zero-shot transfer
2. **Role-specific heads**: Allows different treatment of victim vs. aggressor roles
3. **Element-wise product**: Captures multiplicative interactions between victim and aggressor embeddings
4. **Log-target**: Stabilizes training for ratio-based slowdown metric
5. **LOAO CV**: Properly evaluates generalization to unseen apps

## Results

The model outputs:
- Aggregate metrics (LOAO CV averages, final holdout metrics)
- Per-fold performance breakdown
- Individual predictions table saved to `predictions.csv`
