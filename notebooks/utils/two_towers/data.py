import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr


def load_csvs(jobs_csv: str, inhibitors_csv: str, job_inh_csv: str, pair_csv: str):
    """Load all four CSV files."""
    jobs_df = pd.read_csv(jobs_csv)
    inhibitors_df = pd.read_csv(inhibitors_csv)
    job_inh_df = pd.read_csv(job_inh_csv)
    pair_df = pd.read_csv(pair_csv)
    return jobs_df, inhibitors_df, job_inh_df, pair_df


FEATURE_COLS = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
LOG_COLS = ["mpi_time", "total_msgs", "total_bytes"]


class ProfileNormalizer:
    def __init__(self, feature_cols=FEATURE_COLS, log_cols=LOG_COLS):
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


class VictimAggressorDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df, jobs_df, inhib_df, normalizer):
        job_feats = normalizer.transform(jobs_df.set_index("job_id").loc[pairs_df["job_id"]])
        inh_feats = normalizer.transform(inhib_df.set_index("inhib_id").loc[pairs_df["inhib_id"]])
        self.victim = torch.tensor(job_feats, dtype=torch.float32)
        self.aggressor = torch.tensor(inh_feats, dtype=torch.float32)
        self.y = torch.tensor(np.log(pairs_df["slowdown"].values.astype(np.float32)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.victim[idx], self.aggressor[idx], self.y[idx]


def expand_pair_df(pair_df):
    """Expand pair_df into directional records."""
    a_as_victim = pair_df.rename(columns={
        "jobA_id": "job_id", "jobB_id": "aggressor_job_id", "slowdown_A": "slowdown"
    })[["job_id", "aggressor_job_id", "slowdown"]]

    b_as_victim = pair_df.rename(columns={
        "jobB_id": "job_id", "jobA_id": "aggressor_job_id", "slowdown_B": "slowdown"
    })[["job_id", "aggressor_job_id", "slowdown"]]

    directional = pd.concat([a_as_victim, b_as_victim], ignore_index=True)
    return directional


def evaluate_on_pair_df(model, jobs_df, pair_df, normalizer, device):
    """Evaluate model on pair_df and return predictions and metrics."""
    directional = expand_pair_df(pair_df)
    
    job_list = directional["job_id"].tolist() + directional["aggressor_job_id"].tolist()
    all_job_feats = normalizer.transform(jobs_df.set_index("job_id").loc[list(set(job_list))])
    
    job_to_feats = dict(zip(jobs_df["job_id"].str.strip().tolist(), all_job_feats))
    
    victim_list = directional["job_id"].tolist()
    aggressor_list = directional["aggressor_job_id"].tolist()
    
    victim_feats = torch.tensor([job_to_feats[jid.strip()] for jid in victim_list], dtype=torch.float32).to(device)
    aggressor_feats = torch.tensor([job_to_feats[jid.strip()] for jid in aggressor_list], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        log_slowdown_pred = model(victim_feats, aggressor_feats).cpu().numpy()
    
    true_slowdown = directional["slowdown"].values.astype(np.float32)
    pred_slowdown = np.exp(log_slowdown_pred)
    true_log = np.log(true_slowdown)
    
    mae_log = np.mean(np.abs(log_slowdown_pred - true_log))
    rmse_log = np.sqrt(np.mean((log_slowdown_pred - true_log) ** 2))
    mae_raw = np.mean(np.abs(pred_slowdown - true_slowdown))
    rmse_raw = np.sqrt(np.mean((pred_slowdown - true_slowdown) ** 2))
    spearman_corr, _ = spearmanr(pred_slowdown, true_slowdown)
    
    results_df = directional.copy()
    results_df["log_pred"] = log_slowdown_pred
    results_df["pred_slowdown"] = pred_slowdown
    results_df["abs_error"] = np.abs(pred_slowdown - true_slowdown)
    
    metrics = {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "mae_raw": mae_raw,
        "rmse_raw": rmse_raw,
        "spearman_corr": spearman_corr
    }
    
    return results_df, metrics
