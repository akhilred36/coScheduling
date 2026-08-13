"""
DeltaPairDataset for sampling (A, J, K) triplets on the fly.
"""

import numpy as np
import torch


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
            if len(g) >= 2
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
