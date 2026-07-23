# Slowdown Prediction Pipeline

A few-shot learning system for predicting job slowdown in HPC environments using Deep Sets and Relation Networks.

## Overview

This pipeline predicts the slowdown experienced when two HPC jobs run concurrently on shared resources. It uses a few-shot learning approach where the model learns job representations from training applications and generalizes to unseen job combinations.

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
- `isolated_profiles`: Tensor of shape (num_jobs, 4) - standalone performance metrics
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
EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 5e-3
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2
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

### Evaluation Protocol

The pipeline performs two evaluations:

#### 1. Training Set Evaluation (Held-out Pairs)

Tests generalization to **new job combinations** from training apps.

| Split | Apps Included | Samples | Purpose |
|-------|---------------|---------|---------|
| Training | amg, beatnik, fiesta, laghos, lammps, minife, minivite | 89 | Train model on 80% of pairs |
| Validation | Same apps | 23 | Hold-out 20% for validation |

This measures how well the model generalizes to unseen **pair combinations** from known applications.

#### 2. Test Set Evaluation (Unseen Apps)

Tests generalization to **completely unseen applications** (true few-shot learning).

| Split | Apps Included | Samples | Purpose |
|-------|---------------|---------|---------|
| Training | amg, beatnik, fiesta, laghos, lammps, minife, minivite | - | Model trained on these |
| Test | kripke, quicksilver, tricount | 24 | Evaluate OOD generalization |

This measures how well the model generalizes to **completely unseen applications** - the true few-shot learning capability.

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
- tricount

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total apps | 10 |
| Training apps | 7 |
| Test apps | 3 |
| Total pairs | 110 |
| Train-Train pairs | 56 |
| Train-Test pairs | 42 |
| Test-Test pairs | 12 |
| Test samples (Test-Test × 2 directions) | 24 |

**Note**: The 24 test samples come from 12 test-test pairs (kripke-quicksilver, kripke-tricount, quicksilver-tricount combinations), each doubled for bidirectional prediction.

## Results

### Training Performance

| Metric | Value |
|--------|-------|
| Best Training Loss | 0.000632 |
| Best Validation Loss | 0.000262 |
| Early Stopping Epoch | 50/50 |

### Final Evaluation Results

**Training Set** (held-out pairs from training apps):
- MSE: 0.000191
- MAE: 0.010640

**Test Set** (unseen apps: kripke, quicksilver, tricount):
- MSE: 0.002923
- MAE: 0.040720

**Generalization Ratio**: 
- MSE ratio: 15.3x
- MAE ratio: 3.8x

### Overfitting Analysis

- Train MSE: 0.00019, Test MSE: 0.00292 (unseen apps)
- Train/Test ratio: 15.3x (significant improvement)

## Running the Pipeline

### Prerequisites

```bash
source /home/akhil/hpcResearch/python_venvs/ml_analysis/bin/activate
```

### Full Pipeline

```bash
rm -f best_model.pt
python3 main.py
```

This runs preprocessing (if needed), training, and evaluation sequentially.

### Individual Scripts

```bash
# Preprocessing only
python3 preprocess.py

# Training only
python3 train.py

# Evaluation only
python3 evaluate.py
```

## Key Improvements for Generalization

1. **Weight Decay**: Added L2 regularization (5e-3) to prevent large weights
2. **Mini-batch Training**: Changed from full-batch to batch_size=16 for gradient diversity
3. **Early Stopping**: Prevents over-training by monitoring validation loss
4. **Reduced Capacity**: Smaller embedding (16 vs 64) and hidden layers (8 vs 128)
5. **Increased Dropout**: Higher dropout rate (0.4) for regularization
6. **Validation Split**: 20% held-out data for proper generalization monitoring

## Reproducibility

The pipeline uses fixed random seeds (SEED=42) for:
- Data shuffling
- Model initialization
- Train/val/test splitting

This ensures reproducible results across runs.
