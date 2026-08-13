"""
Shared data loading, normalization, and preprocessing utilities for delta response model.
Reuse from baseline spec: ProfileNormalizer, FEATURE_COLS, expand_pair_df.
"""

import numpy as np
import pandas as pd
import torch


FEATURE_COLS = ["mpi_time", "comm_frac", "total_msgs", "total_bytes"]
LOG_COLS = ("mpi_time", "total_msgs", "total_bytes")


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


def load_csvs(jobs_csv, inhibitors_csv, job_inh_csv, pair_csv):
    jobs_df = pd.read_csv(jobs_csv)
    inhibitors_df = pd.read_csv(inhibitors_csv)
    job_inh_df = pd.read_csv(job_inh_csv)
    pair_df = pd.read_csv(pair_csv)
    return jobs_df, inhibitors_df, job_inh_df, pair_df


def expand_pair_df(pair_df):
    """
    Expand pair_df into directional records.
    Returns dataframe with columns: job_id (victim), aggressor_job_id (aggressor), slowdown.
    """
    a_as_victim = pair_df.rename(columns={
        "jobA_id": "job_id",
        "jobB_id": "inhib_id_like",
        "slowdown_A": "slowdown"
    })[["job_id", "inhib_id_like", "slowdown"]]

    b_as_victim = pair_df.rename(columns={
        "jobB_id": "job_id",
        "jobA_id": "inhib_id_like",
        "slowdown_B": "slowdown"
    })[["job_id", "inhib_id_like", "slowdown"]]

    directional = pd.concat([a_as_victim, b_as_victim], ignore_index=True)
    directional = directional.rename(columns={"inhib_id_like": "aggressor_job_id"})
    return directional


def load_and_prepare_data(jobs_csv, inhibitors_csv, job_inh_csv, pair_csv):
    jobs_df, inhibitors_df, job_inh_df, pair_df = load_csvs(
        jobs_csv, inhibitors_csv, job_inh_csv, pair_csv
    )

    normalizer = ProfileNormalizer()
    normalizer.fit(pd.concat([jobs_df, inhibitors_df], ignore_index=False))

    return {
        "jobs_df": jobs_df,
        "inhibitors_df": inhibitors_df,
        "job_inh_df": job_inh_df,
        "pair_df": pair_df,
        "normalizer": normalizer,
    }
