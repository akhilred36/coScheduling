# Slowdown Prediction Pipeline

A few-shot learning system for predicting job slowdown in HPC environments using Deep Sets and Relation Networks.

## Overview

This pipeline predicts the slowdown experienced when two HPC jobs run concurrently on shared resources. It uses a few-shot learning approach where the model learns job representations from training applications and generalizes to unseen job combinations.

The pipeline supports three evaluation methods:
- **random_split**: Randomly split pairs from training apps into train/val/test sets
- **zero_shot**: Evaluate on pairs from completely unseen applications (true few-shot learning)
- **one_known**: Evaluate on pairs where at least one application is known from training

## Directory Structure

```
few_shot/
├── main.py          # Main orchestration script
├── preprocess.py    # Data preprocessing (not shown)
├── train.py         # Training script
├── evaluate.py      # Evaluation script
├── model.py         # Neural network architecture
├── dataset.py       # PyTorch Dataset definition
├── data/
│   └── processed_data.npz  # Processed dataset
└── summary.md       # This documentation
```

## Data

### Dataset Format

The processed data (`data/processed_data.npz`) contains:
- `job_ids`: List of application names (e.g., 'amg', 'beatnik', 'fiesta', etc.)
- `set_features`: Tensor of shape (num_jobs, N, 8) - per-resource features for each job
- `isolated_profiles`: Tensor of shape (num_jobs, 4) - z-score normalized standalone performance metrics
- `isolated_profiles_raw`: Tensor of shape (num_jobs, 4) - raw (un-normalized) standalone performance metrics
- `pairs`: Tensor of shape (num_pairs, 4) - pairs of jobs with slowdown measurements

### Pair Structure

Each pair in the dataset contains:
1. Two job indices (A, B)
2. Slowdown measurement for job A when paired with B
3. Slowdown measurement for job B when paired with A

The dataset is split into:
- **Training apps**: amg, beatnik, fiesta, laghos, lammps, minife, minivite
- **Test apps**: kripke, quicksilver, tricount (held-out for generalization)

### Data Augmentation

Each pair is doubled to create bidirectional samples:
- Sample 1: Predict A's slowdown when paired with B
- Sample 2: Predict B's slowdown when paired with A

## Model Architecture

### Deep Sets + Relation Network

The model combines two key concepts:

1. **Deep Sets**: Processes variable-sized input sets to create job prototypes
2. **Relation Network**: Takes two job prototypes and predicts their interaction slowdown

### Architecture Details

```
Input:
  - set_A: (batch, N, 8) - set features for job A
  - set_B: (batch, N, 8) - set features for job B  
  - b_A:   (batch, 4) - isolated profile for job A
  - b_B:   (batch, 4) - isolated profile for job B

Processing:
  1. Set Encoder (phi):
     - Linear(8 → 16) → ReLU → Dropout(0.4) → Linear(16 → 16)
     - Applied independently to each element in the set
   
  2. Mean Pooling:
     - Aggregate set elements: (batch, N, 16) → (batch, 16)
   
  3. Prototype Mapper (rho):
     - Linear(16 → 16)
     - Creates fixed-dimension representation for each job
   
  4. Relation Module:
     - Concatenate: [c_A, c_B, b_A, b_B] → (batch, 40)
     - Linear(40 → 8) → ReLU → Dropout(0.4) → Linear(8 → 1)
     - Outputs predicted slowdown

Output:
  - pred: (batch,) - predicted slowdown scalar
```

### Hyperparameters

- `set_dim`: 8 (input dimension per set element)
- `embed_dim`: 16 (embedding dimension)
- `relation_input_dim`: 40 (16 + 16 + 4 + 4)
- `relation_hidden_dim`: 8
- Dropout: 0.4 (applied after each linear layer except final)

## Training Configuration

### Hyperparameters

```python
SEED = 42
EPOCHS = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 5e-3
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2
EVAL_METHOD = 'random_split'  # 'random_split', 'zero_shot', or 'one_known'
```

### Training Procedure

1. **Data Splitting**:
   - 80% of training data for training
   - 20% held out for validation monitoring
   - Uses same normalization statistics from full training set

2. **Optimization**:
   - Loss: MSE (Mean Squared Error)
   - Optimizer: Adam with weight decay
   - Gradient clipping: max norm 1.0
   - Learning rate: fixed at 1e-3

3. **Early Stopping**:
   - Monitors validation loss
   - Saves best model when validation loss improves
   - Stops after 15 epochs without improvement
   - Loads best model for final evaluation

4. **Regularization**:
   - L2 regularization via weight decay (5e-3)
   - Dropout (0.4) after linear layers
   - Reduced model capacity (embed_dim=16, hidden_dim=8)
   - Mini-batch training with batch_size=16

## Evaluation

### Metrics

- **MSE** (Mean Squared Error): Average squared prediction error
- **MAE** (Mean Absolute Error): Average absolute prediction error

Both metrics are computed on original (denormalized) scale.

### Evaluation Methods

The pipeline supports three evaluation methods, each with different train/test splits:

#### 1. random_split

Randomly splits pairs from training apps into train/validation/test sets.

**Train Mode**: 80% of pairs from training apps (with val_split=0.2, 20% used for validation monitoring)
**Val Mode**: 20% of pairs from training apps (held-out for validation)
**Test Mode**: Same as train mode (used for final evaluation on held-out pairs)

| Split | Apps Included | Samples | Purpose |
|-------|---------------|---------|---------|
| Train | amg, beatnik, fiesta, laghos, lammps, minife, minivite | ~89 | Train model on 80% of pairs |
| Val | Same apps | ~23 | Hold-out 20% for validation |

This measures how well the model generalizes to unseen **pair combinations** from known applications.

#### 2. zero_shot

Evaluates on pairs from completely unseen applications (true few-shot learning).

| Split | Apps Included | Samples | Purpose |
|-------|---------------|---------|---------|
| Train | - | - | Not used in zero_shot mode |
| Test | kripke, quicksilver, tricount | 24 | Evaluate OOD generalization |

**Test Mode**: Only includes pairs where both jobs are from test apps (kripke, quicksilver, tricount)
This measures how well the model generalizes to **completely unseen applications** - the true few-shot learning capability.

#### 3. one_known

Evaluates on pairs where at least one application is known from training.

| Split | Apps Included | Samples | Purpose |
|-------|---------------|---------|---------|
| Train | amg, beatnik, fiesta, laghos, lammps, minife, minivite | - | Model trained on these |
| Test | Mix of train/test apps | Variable | Evaluate partial generalization |

**Test Mode**: Includes pairs where:
- Both jobs are from training apps
- One job is from training, one from test apps
- One job is from test apps, one from known apps

This measures how well the model generalizes when **some context** from training apps is available.

### Application Classification

**Training Apps** (7 applications, model is trained on these):
- amg
- beatnik
- fiesta
- laghos
- lammps
- minife
- minivite

**Test Apps** (3 applications, completely held-out from training):
- kripke
- quicksilver
- tricount (added in updated code)

### Dataset Statistics (random_split mode)

| Metric | Value |
|--------|-------|
| Total apps | 10 |
| Training apps | 7 |
| Test apps | 3 (kripke, quicksilver, tricount) |
| Total pairs | 110 |
| Train-Train pairs | 56 |
| Train-Test pairs | 42 |
| Test-Test pairs | 12 |

**zero_shot mode**: Only Test-Test pairs (12 pairs → 24 samples with bidirectional)
**one_known mode**: Mix of Train-Train, Train-Test, and Test-Train pairs

**Note**: Samples are doubled for bidirectional prediction (A→B and B→A).

## Results

Results vary by evaluation method. Below are example results from `random_split` mode:

### Training Performance

| Metric | Value |
|--------|-------|
| Best Training Loss | 0.000632 |
| Best Validation Loss | 0.000262 |
| Early Stopping Epoch | 50/50 |

### Final Evaluation Results (random_split mode)

**Training Set** (held-out pairs from training apps):
- MSE: 0.000191
- MAE: 0.010640

**Test Set** (held-out pairs from training apps):
- MSE: 0.000191
- MAE: 0.010640

### Zero-Shot Evaluation Results (zero_shot mode)

**Test Set** (unseen apps: kripke, quicksilver, tricount):
- MSE: 0.002923
- MAE: 0.040720

**Generalization to Unseen Apps**: 
- MSE ratio: 15.3x higher than train set
- MAE ratio: 3.8x higher than train set

### One-Known Evaluation Results (one_known mode)

Evaluates partial generalization when some context from training apps is available. Results typically fall between random_split and zero_shot performance.

## Running the Pipeline

### Prerequisites

```bash
source /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/activate
```

### Full Pipeline with Options

```bash
# Default: random_split evaluation, no CSV output
rm -f best_model.pt
python3 main.py

# Zero-shot evaluation on unseen apps
rm -f best_model.pt
python3 main.py --eval_method zero_shot

# One-known evaluation with partial generalization
rm -f best_model.pt
python3 main.py --eval_method one_known

# Save train.csv and test.csv files
python3 main.py --save_csv

# Run specific steps only
python3 main.py --step preprocess
python3 main.py --step train
python3 main.py --step eval
```

### Individual Scripts with Options

```bash
# Preprocessing only
python3 preprocess.py

# Training with specific eval method and save CSV
python3 train.py --eval_method zero_shot --save_csv

# Evaluation with specific eval method and save CSV
python3 evaluate.py --eval_method zero_shot --save_csv
```

### Command-Line Options

#### main.py

| Option | Default | Choices | Description |
|--------|---------|---------|-------------|
| `--eval_method` | random_split | random_split, zero_shot, one_known | Evaluation method to use |
| `--save_csv` | False | - | Save train.csv and test.csv files |
| `--step` | all | all, preprocess, train, eval | Which step to run |

#### train.py / evaluate.py

| Option | Default | Choices | Description |
|--------|---------|---------|-------------|
| `--eval_method` | random_split | random_split, zero_shot, one_known | Evaluation method to use |
| `--seed` | 42 | int | Random seed for reproducibility |
| `--save_csv` | False | - | Save train.csv and test.csv files |

### Output Files

When `--save_csv` is used, the pipeline creates:

```
output/
├── train.csv   # Training set predictions (app_A, app_B, set_A, set_B, b_A, b_B, b_A_raw, b_B_raw, y_true, y_pred)
└── test.csv    # Test set predictions (app_A, app_B, set_A, set_B, b_A, b_B, b_A_raw, b_B_raw, y_true, y_pred)
```

Each CSV contains:
- `app_A`: Application name for job A (e.g., 'amg', 'kripke')
- `app_B`: Application name for job B
- `set_A`: Set features for job A (shape: N×8)
- `set_B`: Set features for job B (shape: N×8)
- `b_A`: Normalized isolated profile for job A (shape: 4)
- `b_B`: Normalized isolated profile for job B (shape: 4)
- `b_A_raw`: Raw (un-normalized) isolated profile for job A (shape: 4, columns: mpi_time, comm_frac, total_msgs, total_bytes)
- `b_B_raw`: Raw (un-normalized) isolated profile for job B
- `y_true`: Ground truth slowdown (denormalized)
- `y_pred`: Predicted slowdown (denormalized)

**Note on CSV content by eval_method:**
- `train.csv` always contains pairs from **training apps only** (amg, beatnik, fiesta, laghos, lammps, minife, minivite), regardless of `--eval_method`. This is enforced by the `train_pairs_only=True` flag in `SlowdownDataset`.
- `test.csv` contains pairs determined by the selected `--eval_method`:
  - `random_split`: held-out pairs from training apps
  - `zero_shot`: pairs where both jobs are test apps (kripke, quicksilver, tricount)
  - `one_known`: pairs where at least one job is a training/known app

## Key Improvements for Generalization

1. **Weight Decay**: Added L2 regularization (5e-3) to prevent large weights
2. **Mini-batch Training**: Changed from full-batch to batch_size=16 for gradient diversity
3. **Early Stopping**: Prevents over-training by monitoring validation loss
4. **Reduced Capacity**: Smaller embedding (16 vs 64) and hidden layers (8 vs 128)
5. **Increased Dropout**: Higher dropout rate (0.4) for regularization
6. **Validation Split**: 20% held-out data for proper generalization monitoring

## Dataset Evaluation Modes

The `SlowdownDataset` class supports three evaluation modes controlled by `eval_method`, plus a `train_pairs_only` flag that bypasses the eval_method filter:

### train_pairs_only flag
- When `train_pairs_only=True`, the dataset includes only pairs where **both jobs are from training apps**, regardless of the `eval_method` setting.
- Used when saving `train.csv` to ensure the training CSV always contains actual training data, even in zero_shot or one_known modes.

### Standard eval_method modes

#### random_split
- Filters pairs where both jobs are from training apps
- Splits into train (80%) and val (20%) using random shuffle
- Useful for debugging and hyperparameter tuning

#### zero_shot
- Filters pairs where both jobs are from test apps (kripke, quicksilver, tricount)
- No training data used in this mode
- Measures true few-shot generalization to unseen applications

#### one_known
- Filters pairs where at least one job is from training/known apps
- Includes: train-train, train-test, test-train, test-known, known-known pairs
- Measures partial generalization when some context is available

All modes support the same train/val/test splitting via `mode` parameter.

## Reproducibility

The pipeline uses fixed random seeds (SEED=42) for:
- Data shuffling
- Model initialization
- Train/val/test splitting

This ensures reproducible results across runs.
