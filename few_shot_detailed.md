# Few-Shot Learning for HPC Co-Scheduling: Detailed Documentation

## Overview

This notebook implements a few-shot learning approach to predict how HPC applications will perform when co-scheduled together. The core idea is to learn a "signature" for each application based on how it responds to various inhibitor workloads, then use these signatures to predict performance degradation (slowdown) when two applications run simultaneously.

This is particularly useful in HPC environments where running experiments to test all possible application combinations would be prohibitively expensive. Instead, we can learn from a limited set of experiments and generalize to new applications.

---

## Part 1: Importing Libraries

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
```

**Purpose**: This cell imports all necessary libraries:
- **NumPy**: For numerical operations and array manipulations
- **Pandas**: For data manipulation and analysis
- **PyTorch**: For building and training neural networks
- **scikit-learn**: For data preprocessing (StandardScaler) and model evaluation
- **tqdm**: For progress bars during training
- **warnings**: To suppress non-critical warnings and keep output clean

---

## Part 2: Loading Raw Data

```python
jobs_raw_df = pd.read_csv("../processed_data/data_isolated_agg.csv", index_col=0)
jobs_raw_df = jobs_raw_df[jobs_raw_df["Num Nodes"] == 8]
inhibitors_raw_df = pd.read_csv("../processed_data/data_inhib_isolated_agg.csv", index_col=0)
inhibitors_raw_df = inhibitors_raw_df[inhibitors_raw_df["Num Nodes"] == 8]
job_inh_raw_df = pd.read_csv("../processed_data/data_inhib_coscheduled_agg.csv", index_col=0)
job_inh_raw_df = job_inh_raw_df[job_inh_raw_df["Num Nodes"] == 8]
pair_raw_df = pd.read_csv("../processed_data/data_coscheduled_agg.csv", index_col=0)
pair_raw_df = pair_raw_df[pair_raw_df["Num Nodes"] == 8]
```

**Purpose**: Loads four different datasets from CSV files, all filtered to only include experiments run on 8 nodes:

1. **data_isolated_agg.csv**: Performance data for individual applications running alone (baseline)
2. **data_inhib_isolated_agg.csv**: Performance data for "inhibitor" applications running alone
3. **data_inhib_coscheduled_agg.csv**: Performance when a main application runs together with an inhibitor
4. **data_coscheduled_agg.csv**: Performance when two applications run together as equal partners

Filtering to 8 nodes ensures consistent hardware conditions across all experiments.

---

## Part 3: Data Preprocessing - Creating Structured Datasets

This cell creates four processed DataFrames from the raw data:

### 3a: Jobs DataFrame (Individual Applications)
```python
jobs_df["job_id"] = jobs_raw_df["App"]
jobs_df["mpi_time"] = jobs_raw_df["MPI Time Average Mean"]
jobs_df["comm_frac"] = jobs_raw_df["MPI Time Average Mean"] / jobs_raw_df["App Time Average Mean"]
jobs_df["total_msgs"] = jobs_raw_df["Total Messages Sent Mean"]
jobs_df["total_bytes"] = jobs_raw_df["Total Bytes Sent Mean"]
```

**Features extracted for each application**:
- `job_id`: Name/identifier of the application
- `mpi_time`: Absolute MPI execution time (seconds)
- `comm_frac`: Fraction of total time spent in communication (MPI)
- `total_msgs`: Total number of messages sent
- `total_bytes`: Total bytes sent

### 3b: Inhibitors DataFrame (Inhibitor Applications)
```python
inhibitors_df["inh_id"] = (selected_data.astype(str).agg('_'.join, axis=1))
inhibitors_df["msg_size"] = inhibitors_raw_df["Inhib Message Size"]
inhibitors_df["wait_time"] = inhibitors_raw_df["Inhib Wait Time (us)"]
inhibitors_df["comm_sparsity"] = inhibitors_raw_df["Inhib Comm Sparsity"]
inhibitors_df["mpi_time"] = inhibitors_raw_df["Inhib MPI Time Average Mean"]
inhibitors_df["comm_frac"] = inhibitors_raw_df["Inhib MPI Time Average Mean"] / inhibitors_raw_df["Inhib App Time Average Mean"]
inhibitors_df["total_msgs"] = inhibitors_raw_df["Inhib Total Messages Sent Mean"]
inhibitors_df["total_bytes"] = inhibitors_raw_df["Inhib Total Bytes Sent Mean"]
```

**Features for inhibitors**:
- `inh_id`: Unique identifier created from configuration parameters
- Configuration: `msg_size`, `wait_time`, `comm_sparsity` (describes the inhibitor workload)
- Profiling: Same metrics as jobs (mpi_time, comm_frac, total_msgs, total_bytes)

### 3c: Job-Inhibitor Coscheduling DataFrame
```python
for i in job_inh_raw_df.iloc:
    iso_time = (jobs_df[jobs_df["job_id"] == i["App"]])["mpi_time"].max()
    coscheduled_time = i["MPI Time Average Mean"]
    if (coscheduled_time < iso_time):
        slowdown = 1
    else:
        slowdown = coscheduled_time/iso_time
    job_inh_df["slowdown"] = job_inh_slowdowns
```

**Purpose**: Calculates the **slowdown** experienced by a job when co-scheduled with an inhibitor:
- **Slowdown formula**: `coscheduled_time / isolated_time`
- If coscheduled time is less than isolated (rare), slowdown = 1 (no penalty)
- This metric represents how much slower the job runs due to interference

### 3d: Job-Job Pairs DataFrame
```python
for i in pair_raw_df.iloc:
    iso_time = (jobs_df[jobs_df["job_id"] == i["App A"]])["mpi_time"].max()
    coscheduled_time = i["App A MPI Time Average Mean"]
    if (coscheduled_time < iso_time):
        slowdown = 1
    else:
        slowdown = coscheduled_time/iso_time
    pair_df["slowdown_A"] = pair_slowdowns
```

**Purpose**: Similar to above, but calculates slowdown for App A when paired with App B in a coscheduling experiment.

---

## Part 4: Feature Engineering and Normalization

This is a critical section where data is prepared for machine learning:

### 4a: Feature Columns Definition
```python
job_feat_cols = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
inh_config_cols = ["msg_size", "wait_time", "comm_sparsity"]
inh_prof_cols = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
```

Defines which features will be used for each component.

### 4b: Creating Inhibitor Feature Vectors
```python
inhibitors_df["inh_features"] = inhibitors_df.apply(
    lambda row: np.concatenate([row[inh_config_cols].values, row[inh_prof_cols].values]),
    axis=1
)
```

**Purpose**: Combines configuration and profiling features into a single 7-dimensional vector for each inhibitor:
- 3 configuration features + 4 profiling features = 7 total dimensions

### 4c: Building Job-to-Inhibitor Mappings
```python
for job_id in job_ids:
    job_data = job_inh_df[job_inh_df["job_id"] == job_id]
    for _, row in job_data.iterrows():
        inh_feat_scaled = inh_feat_scaler.transform(inh_feat.reshape(1, -1)).flatten()
        inhib_list.append({
            "inh_id": inh_id,
            "inh_feat": inh_feat_scaled,  # 7-dim
            "slowdown": slowdown
        })
```

**Purpose**: For each job, creates a list of how it responded to different inhibitors:
- Each entry contains: inhibitor ID, normalized inhibitor features, and the observed slowdown
- This forms the "training data" for learning each job's signature

**Data Quality Checks**:
- Skips rows where inhibitor data is missing
- Skips rows where slowdown is NaN
- Tracks skipped experiments (12 in this case)
- Filters to only jobs with valid data (10 valid jobs)

### 4d: Feature Normalization
```python
inh_feat_scaler = StandardScaler()
inh_feat_scaler.fit(inh_feat_matrix)

slowdown_scaler = StandardScaler()
slowdown_scaler.fit(np.array(slowdowns_all).reshape(-1, 1))

job_feat_scaler = StandardScaler()
job_feat_scaler.fit(job_feat_matrix)
```

**Purpose**: Normalizes all features using StandardScaler (zero mean, unit variance):
- **inh_feat_scaler**: Normalizes inhibitor features across all inhibitors
- **slowdown_scaler**: Normalizes slowdown values for consistent training
- **job_feat_scaler**: Normalizes isolated job features

Normalization is crucial for neural network training as it ensures all features contribute equally.

---

## Part 5: Neural Network Architecture

### 5a: JobSignatureEncoder (Deep Sets Architecture)

```python
class JobSignatureEncoder(nn.Module):
    def __init__(self, inh_feat_dim=7, hidden_dim=64, latent_dim=64):
        self.element_encoder = nn.Sequential(
            nn.Linear(inh_feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
```

**Purpose**: Encodes a job's responses to multiple inhibitors into a fixed-size latent vector (job signature).

**Key Design Principles**:
1. **Deep Sets Architecture**: Permutation-invariant neural network that processes sets
2. **Element Encoder**: Processes each (inhibitor, slowdown) pair independently
   - Input: 7 inhibitor features + 1 slowdown = 8 dimensions
   - Two hidden layers with 64 neurons and ReLU activation
3. **Aggregator**: Combines all encoded elements via mean pooling
   - Takes the mean of all encoded inhibitor responses
   - Produces a single 64-dimensional latent vector (job signature)
4. **Mean Pooling**: Makes the model permutation-invariant (order of inhibitors doesn't matter)
5. **Masking**: Handles variable numbers of inhibitors per job

**Forward Pass**:
```python
def forward(self, inh_feats, slowdowns, mask=None):
    x = torch.cat([inh_feats, slowdowns], dim=-1)  # Concatenate features + slowdown
    encoded = self.element_encoder(x)  # Encode each inhibitor response
    pooled = encoded.mean(dim=1)  # Average across all inhibitors
    job_latent = self.aggregator(pooled)  # Produce final signature
    return job_latent
```

### 5b: PairSlowdownPredictor

```python
class PairSlowdownPredictor(nn.Module):
    def __init__(self, latent_dim=64, job_feat_dim=4, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + job_feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
```

**Purpose**: Takes two job signatures and their isolated features to predict pairwise slowdown.

**Input**: 
- zA, zB: 64-dim job signatures (latent representations)
- jobA_feat, jobB_feat: 4-dim isolated features (mpi_time, comm_frac, etc.)
- Combined input: 64×2 + 4×2 = 136 dimensions

**Architecture**:
- Hidden layer 1: 136 → 64 neurons with ReLU
- Dropout (20%): Prevents overfitting
- Hidden layer 2: 64 → 64 neurons with ReLU
- Dropout (20%): Prevents overfitting
- Output layer: 64 → 1 (predicted slowdown)

### 5c: FullModel (Complete Architecture)

```python
class FullModel(nn.Module):
    def __init__(self, inh_feat_dim=7, latent_dim=64, job_feat_dim=4):
        self.job_encoder = JobSignatureEncoder(inh_feat_dim, hidden_dim=64, latent_dim=latent_dim)
        self.pair_predictor = PairSlowdownPredictor(latent_dim, job_feat_dim, hidden_dim=64)
```

**Purpose**: Combines both components into a single model that:
1. Encodes Job A's responses to inhibitors into a signature
2. Encodes Job B's responses to inhibitors into a signature
3. Predicts the slowdown when A and B are co-scheduled

**Forward Pass**:
```python
def forward(self, jobA_inh_feats, jobA_slowdowns, jobA_feat,
            jobB_inh_feats, jobB_slowdowns, jobB_feat,
            maskA=None, maskB=None):
    zA = self.job_encoder(jobA_inh_feats, jobA_slowdowns, maskA)
    zB = self.job_encoder(jobB_inh_feats, jobB_slowdowns, maskB)
    return self.pair_predictor(zA, zB, jobA_feat, jobB_feat)
```

---

## Part 6: Dataset Preparation

### 6a: Data Preparation Functions

```python
def build_job_tensor(job_id):
    # Create padded tensors for job's inhibitor data
    inh_feats = np.zeros((max_inhibitors, 7), dtype=np.float32)
    slowdowns = np.zeros((max_inhibitors, 1), dtype=np.float32)
    mask = np.zeros(max_inhibitors, dtype=bool)
    mask[:n] = True  # Mark valid entries
    
    # Normalize and convert to tensors
```

**Purpose**: Converts each job's data into fixed-size padded tensors:
- **Padding**: Jobs may have different numbers of inhibitor experiments (max: 235)
- **Mask**: Tracks which entries are valid vs. padding
- **Normalization**: Applies the previously fitted scalers

### 6b: PairDataset Class

```python
class PairDataset(Dataset):
    def __init__(self, pair_df, job_tensors):
        self.pair_df = pair_df
        self.job_tensors = job_tensors
    
    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        # Returns all inputs needed for one pair prediction
        return {
            "jobA_inh_feats": torch.tensor(inhA),
            "jobA_slowdowns": torch.tensor(slowA),
            "jobA_mask": torch.tensor(maskA),
            "jobA_feat": torch.tensor(featA),
            # ... same for jobB
            "target": torch.tensor(target)
        }
```

**Purpose**: PyTorch Dataset class that provides one batch at a time during training. Each sample contains:
- All inhibitor features and slowdowns for Job A (padded to max length)
- All inhibitor features and slowdowns for Job B (padded to max length)
- Masks to distinguish real data from padding
- Isolated features for both jobs
- Target: actual slowdown from experiment

---

## Part 7: Training Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 64
batch_size = min(8, len(pair_df))
epochs = 500
lr = 1e-3
weight_decay = 1e-4
loss_fn = nn.MSELoss()
```

**Hyperparameters**:
- **Device**: Uses GPU if available, otherwise CPU
- **latent_dim**: Size of job signature (64 dimensions)
- **batch_size**: 8 samples per gradient update (small due to limited data)
- **epochs**: Up to 500 training iterations
- **lr (learning rate)**: 0.001 - controls step size during optimization
- **weight_decay**: 0.0001 - L2 regularization to prevent overfitting
- **MSELoss**: Mean Squared Error loss for regression task

### Training Function

```python
def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    for batch in dataloader:
        pred = model(...)
        loss = loss_fn(pred, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

**Training Steps**:
1. Set model to training mode
2. For each batch:
   - Forward pass: get predictions
   - Calculate loss
   - Zero gradients
   - Backward pass: compute gradients
   - **Gradient clipping**: Prevent exploding gradients (max norm = 1.0)
   - Update model parameters

### Evaluation Function

```python
def eval_model(model, dataloader, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(...)
        # Calculate MAE, MAPE on original scale
```

**Evaluation Steps**:
1. Set model to evaluation mode
2. No gradient computation (faster, less memory)
3. Calculate metrics on original (unnormalized) scale:
   - **MSE Loss**: Mean Squared Error
   - **MAE (Mean Absolute Error)**: Average absolute difference
   - **MAPE (Mean Absolute Percentage Error)**: Average percentage error

---

## Part 8: Training Strategy - Train on Subset, Test on Unseen Jobs

This is the core "few-shot learning" aspect:

### 8a: Defining Train/Test Split

```python
train_job_ids = valid_job_ids[:7]  # Train on amg, beatnik, fiesta, kripke, laghos, lammps, minife
test_job_ids = [j for j in valid_job_ids if j not in train_job_ids]  # Test on minivite, quicksilver, tricount
```

**Key Concept**: Train on 7 jobs, evaluate on 3 completely **unseen** jobs. This tests whether the model can generalize to applications it has never encountered before.

### 8b: Creating Dataset Splits

```python
train_pair_df = pair_df[both jobs in training set]
test_pair_df = pair_df[at least one job is unseen]
val_pair_df = 10% of training pairs (for early stopping)
```

- **Training set**: Only pairs where both jobs were seen during training
- **Test set**: Pairs where at least one job was unseen (tests generalization)
- **Validation set**: Subset of training pairs used to monitor overfitting

### 8c: Training Loop with Early Stopping

```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss, val_mae, val_mape, _, _ = eval_model(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 100:
        break  # Early stopping
```

**Early Stopping**: Stops training if validation loss doesn't improve for 100 epochs:
- Saves the best model state
- Prevents overfitting to training data
- Model loaded: best performing on validation set

---

## Part 9: Evaluation Metrics

### Training Set Evaluation
```python
r2_train = 1 - (ss_res_train / ss_tot_train)
```
- **R² Score**: Proportion of variance explained by the model (1.0 = perfect)
- **MAE**: Average absolute error (in original units)
- **MAPE**: Average percentage error

### Test Set Evaluation
Same metrics calculated on **unseen jobs** to measure generalization ability.

### Additional Analysis
- **One unseen job**: Pairs with 1 trained job + 1 unseen job
- **Both jobs unseen**: Pairs with 2 unseen jobs (hardest case)

This breakdown reveals how well the model generalizes:
- If performance degrades significantly with unseen jobs → model hasn't learned transferable features
- If performance remains good → job signatures capture generalizable characteristics

---

## Part 10: Visualization

Creates a comprehensive 3x3 grid of visualizations:

1. **Training Curves**: Shows how loss decreases over epochs
2. **Training Predictions vs Actual**: Scatter plot with perfect prediction line
3. **Test Predictions vs Actual**: Same for test set, color-coded by pair type
4. **Training Error Distribution**: Histogram of prediction errors
5. **Test Error Distribution**: Same for test set
6. **MAE by Job**: Bar chart showing error per job
7. **Box Plot Comparison**: Compares error distributions between train and test
8. **APE Distribution**: Percentage error distributions
9. **Summary Table**: All key metrics side-by-side

---

## Summary: How Few-Shot Learning Works Here

### The Problem
In HPC environments, you can't experiment with all possible application combinations. With N applications, there are O(N²) pairs to test.

### The Solution
1. **Learn Job Signatures**: For each job, collect how it responds to many different inhibitor workloads
2. **Encode Signatures**: Use a neural network to compress these responses into a fixed-size latent vector
3. **Predict Pairwise Interference**: Use two job signatures to predict their combined performance

### Why It's "Few-Shot"
- Train on 7 jobs (limited data)
- Test on 3 completely new jobs (zero examples during training)
- Model must have learned **general patterns** of how jobs interfere, not just memorized specific pairs

### Key Innovation: Deep Sets Architecture
- Processes variable numbers of inhibitor responses
- Permutation-invariant (order doesn't matter)
- Produces fixed-size job signature regardless of how many inhibitor experiments exist
- These signatures capture the essential characteristics that determine interference behavior

---

## Results Interpretation (from notebook output)

**Training Set Performance**:
- R² = 0.8147: Model explains 81.5% of variance in training data
- MAE = 0.0407: Average error of ~4% slowdown
- MAPE = 3.67%: Average percentage error of 3.67%

**Test Set Performance (Unseen Jobs)**:
- R² = -0.4402: Model performs worse than predicting the mean (struggles to generalize)
- MAE = 0.1608: Average error of ~16% slowdown
- MAPE = 13.71%: Average percentage error of 13.71%

**Analysis**:
- The model fits training data well but generalizes poorly to unseen jobs
- This suggests the current approach may need:
  - More training data (more jobs)
  - Different architecture
  - Better feature representation
  - Data augmentation

The breakdown shows:
- **One unseen job**: MAE = 0.1536, MAPE = 12.92%
- **Both jobs unseen**: MAE = 0.1859, MAPE = 16.46%

Both cases show significant degradation compared to training, indicating the model hasn't learned fully transferable job signatures.
