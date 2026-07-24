#!/usr/bin/env python3
"""
Data Preprocessing Script
Load raw CSV files, filter inhibitors, and save processed tensors.
"""

import pandas as pd
import numpy as np
import os

def main():
    data_dir = "data"
    output_path = os.path.join(data_dir, "processed_data.npz")
    
    # Check if already processed
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping preprocessing.")
        return
    
    print("Loading CSV files...")
    
    # Load DataFrames
    jobs_df = pd.read_csv(os.path.join(data_dir, "jobs.csv"))
    inhibitors_df = pd.read_csv(os.path.join(data_dir, "inhibitors.csv"))
    job_inh_df = pd.read_csv(os.path.join(data_dir, "job_inh.csv"))
    pair_df = pd.read_csv(os.path.join(data_dir, "pair.csv"))
    
    # Get unique job_ids
    unique_jobs = jobs_df['job_id'].unique()
    print(f"Number of unique jobs: {len(unique_jobs)}")
    
    # Find common inhibitors: inhibitors present for ALL jobs
    jobs_per_inhib = job_inh_df.groupby('inhib_id')['job_id'].nunique()
    common_inhibs = jobs_per_inhib[jobs_per_inhib == len(unique_jobs)].index.tolist()
    n_common = len(common_inhibs)
    print(f"Number of common inhibitors (present for all jobs): {n_common}")
    
    # Filter job_inh to common inhibitors only
    job_inh_filtered = job_inh_df[job_inh_df['inhib_id'].isin(common_inhibs)].copy()
    
    # Sort inhibitors to ensure consistent ordering
    common_inhibs_sorted = sorted(common_inhibs)
    inhib_id_to_idx = {inhib_id: idx for idx, inhib_id in enumerate(common_inhibs_sorted)}
    
    # Build set tensor for each job
    # Features: [slowdown, msg_size, wait_time, comm_sparsity, mpi_time, comm_frac, total_msgs, total_bytes]
    print("Building set features...")
    
    # Join job_inh with inhibitors to get inhibitor features
    job_inh_with_features = job_inh_filtered.merge(
        inhibitors_df[['inhib_id', 'msg_size', 'wait_time', 'comm_sparsity', 'mpi_time', 'comm_frac', 'total_msgs', 'total_bytes']],
        on='inhib_id',
        how='left'
    )
    
    # Create job_id to index mapping
    job_id_to_idx = {job_id: idx for idx, job_id in enumerate(unique_jobs)}
    idx_to_job_id = {idx: job_id for job_id, idx in job_id_to_idx.items()}
    
     # Build set_features: (10, N_common, 8)
    n_jobs = len(unique_jobs)
    set_features = np.zeros((n_jobs, n_common, 8), dtype=np.float32)
    
    # Collect all feature values for normalization
    all_mpi_times_iso = []
    all_comm_fracs_iso = []
    all_total_msgs_iso = []
    all_total_bytes_iso = []
    
    for _, row in jobs_df.iterrows():
        all_mpi_times_iso.append(row['mpi_time'])
        all_comm_fracs_iso.append(row['comm_frac'])
        all_total_msgs_iso.append(row['total_msgs'])
        all_total_bytes_iso.append(row['total_bytes'])
    
    mpi_time_mean_iso, mpi_time_std_iso = np.mean(all_mpi_times_iso), np.std(all_mpi_times_iso) or 1.0
    comm_frac_mean_iso, comm_frac_std_iso = np.mean(all_comm_fracs_iso), np.std(all_comm_fracs_iso) or 1.0
    total_msgs_mean_iso, total_msgs_std_iso = np.mean(all_total_msgs_iso), np.std(all_total_msgs_iso) or 1.0
    total_bytes_mean_iso, total_bytes_std_iso = np.mean(all_total_bytes_iso), np.std(all_total_bytes_iso) or 1.0
    
    print("Isolated profile normalization stats:")
    print(f"  mpi_time: mean={mpi_time_mean_iso:.4f}, std={mpi_time_std_iso:.4f}")
    print(f"  comm_frac: mean={comm_frac_mean_iso:.4f}, std={comm_frac_std_iso:.4f}")
    print(f"  total_msgs: mean={total_msgs_mean_iso:.4f}, std={total_msgs_std_iso:.4f}")
    print(f"  total_bytes: mean={total_bytes_mean_iso:.4f}, std={total_bytes_std_iso:.4f}")
    
    # Collect all feature values for normalization
    all_slowdowns = []
    all_msg_sizes = []
    all_wait_times = []
    all_comm_sparsities = []
    all_mpi_times_inh = []
    all_comm_fracs_inh = []
    all_total_msgs_inh = []
    all_total_bytes_inh = []
    
    for _, row in job_inh_with_features.iterrows():
        all_slowdowns.append(row['slowdown'])
        all_msg_sizes.append(row['msg_size'])
        all_wait_times.append(row['wait_time'])
        all_comm_sparsities.append(row['comm_sparsity'])
        all_mpi_times_inh.append(row['mpi_time'])
        all_comm_fracs_inh.append(row['comm_frac'])
        all_total_msgs_inh.append(row['total_msgs'])
        all_total_bytes_inh.append(row['total_bytes'])
    
    slowdown_mean, slowdown_std = np.mean(all_slowdowns), np.std(all_slowdowns) or 1.0
    msg_size_mean, msg_size_std = np.mean(all_msg_sizes), np.std(all_msg_sizes) or 1.0
    wait_time_mean, wait_time_std = np.mean(all_wait_times), np.std(all_wait_times) or 1.0
    comm_sparsity_mean, comm_sparsity_std = np.mean(all_comm_sparsities), np.std(all_comm_sparsities) or 1.0
    mpi_time_mean_inh, mpi_time_std_inh = np.mean(all_mpi_times_inh), np.std(all_mpi_times_inh) or 1.0
    comm_frac_mean_inh, comm_frac_std_inh = np.mean(all_comm_fracs_inh), np.std(all_comm_fracs_inh) or 1.0
    total_msgs_mean_inh, total_msgs_std_inh = np.mean(all_total_msgs_inh), np.std(all_total_msgs_inh) or 1.0
    total_bytes_mean_inh, total_bytes_std_inh = np.mean(all_total_bytes_inh), np.std(all_total_bytes_inh) or 1.0
    
    print("Set feature normalization stats:")
    print(f"  slowdown: mean={slowdown_mean:.4f}, std={slowdown_std:.4f}")
    print(f"  msg_size: mean={msg_size_mean:.4f}, std={msg_size_std:.4f}")
    print(f"  wait_time: mean={wait_time_mean:.4f}, std={wait_time_std:.4f}")
    print(f"  comm_sparsity: mean={comm_sparsity_mean:.4f}, std={comm_sparsity_std:.4f}")
    print(f"  mpi_time (inh): mean={mpi_time_mean_inh:.4f}, std={mpi_time_std_inh:.4f}")
    print(f"  comm_frac: mean={comm_frac_mean_inh:.4f}, std={comm_frac_std_inh:.4f}")
    print(f"  total_msgs: mean={total_msgs_mean_inh:.4f}, std={total_msgs_std_inh:.4f}")
    print(f"  total_bytes: mean={total_bytes_mean_inh:.4f}, std={total_bytes_std_inh:.4f}")
    
    for job_idx, job_id in enumerate(unique_jobs):
        job_data = job_inh_with_features[job_inh_with_features['job_id'] == job_id]
        # Sort by inhib_id to ensure consistent order
        job_data = job_data.sort_values('inhib_id')
        
        for _, row in job_data.iterrows():
            inhib_idx = inhib_id_to_idx[row['inhib_id']]
            set_features[job_idx, inhib_idx, 0] = (row['slowdown'] - slowdown_mean) / slowdown_std
            set_features[job_idx, inhib_idx, 1] = (row['msg_size'] - msg_size_mean) / msg_size_std
            set_features[job_idx, inhib_idx, 2] = (row['wait_time'] - wait_time_mean) / wait_time_std
            set_features[job_idx, inhib_idx, 3] = (row['comm_sparsity'] - comm_sparsity_mean) / comm_sparsity_std
            set_features[job_idx, inhib_idx, 4] = (row['mpi_time'] - mpi_time_mean_inh) / mpi_time_std_inh
            set_features[job_idx, inhib_idx, 5] = (row['comm_frac'] - comm_frac_mean_inh) / comm_frac_std_inh
            set_features[job_idx, inhib_idx, 6] = (row['total_msgs'] - total_msgs_mean_inh) / total_msgs_std_inh
            set_features[job_idx, inhib_idx, 7] = (row['total_bytes'] - total_bytes_mean_inh) / total_bytes_std_inh
    
    # Build isolated profiles: (10, 4)
    # Columns: mpi_time, comm_frac, total_msgs, total_bytes
    print("Building isolated profiles...")
    isolated_profiles = np.zeros((n_jobs, 4), dtype=np.float32)
    isolated_profiles_raw = np.zeros((n_jobs, 4), dtype=np.float32)
    
    for _, row in jobs_df.iterrows():
        job_idx = job_id_to_idx[row['job_id']]
        isolated_profiles_raw[job_idx, 0] = row['mpi_time']
        isolated_profiles_raw[job_idx, 1] = row['comm_frac']
        isolated_profiles_raw[job_idx, 2] = row['total_msgs']
        isolated_profiles_raw[job_idx, 3] = row['total_bytes']
        isolated_profiles[job_idx, 0] = (row['mpi_time'] - mpi_time_mean_iso) / mpi_time_std_iso
        isolated_profiles[job_idx, 1] = (row['comm_frac'] - comm_frac_mean_iso) / comm_frac_std_iso
        isolated_profiles[job_idx, 2] = (row['total_msgs'] - total_msgs_mean_iso) / total_msgs_std_iso
        isolated_profiles[job_idx, 3] = (row['total_bytes'] - total_bytes_mean_iso) / total_bytes_std_iso
    
  # Build pair targets
    print("Building pair targets...")
    pairs_list = []
    seen_pairs = set()
    for _, row in pair_df.iterrows():
        job_a = row['jobA_id']
        job_b = row['jobB_id']
        slowdown_a = row['slowdown_A']
        slowdown_b = row['slowdown_B']
        
        idx_a = job_id_to_idx[job_a]
        idx_b = job_id_to_idx[job_b]
        
        # Skip duplicate self-pairs
        if idx_a == idx_b:
            pair_key = (idx_a, idx_b)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
        
        # Add both directions for different jobs, only once for same job
        if idx_a != idx_b:
            pair_key_ab = (idx_a, idx_b)
            pair_key_ba = (idx_b, idx_a)
            if pair_key_ab not in seen_pairs and pair_key_ba not in seen_pairs:
                pairs_list.append([idx_a, idx_b, float(slowdown_a), float(slowdown_b)])
                pairs_list.append([idx_b, idx_a, float(slowdown_b), float(slowdown_a)])
                seen_pairs.add(pair_key_ab)
                seen_pairs.add(pair_key_ba)
        else:
            pairs_list.append([idx_a, idx_b, float(slowdown_a), float(slowdown_b)])
    
    pairs = np.array(pairs_list, dtype=np.float32)
    
   # Save as NPZ
    print(f"Saving to {output_path}...")
    np.savez(
        output_path,
        job_ids=np.array(unique_jobs, dtype=object),
        set_features=set_features,
        isolated_profiles=isolated_profiles,
        isolated_profiles_raw=isolated_profiles_raw,
        pairs=pairs,
        inhibitor_ids=np.array(common_inhibs_sorted, dtype=object),
        slowdown_mean=np.float32(slowdown_mean),
        slowdown_std=np.float32(slowdown_std),
        msg_size_mean=np.float32(msg_size_mean),
        msg_size_std=np.float32(msg_size_std),
        wait_time_mean=np.float32(wait_time_mean),
        wait_time_std=np.float32(wait_time_std),
        comm_sparsity_mean=np.float32(comm_sparsity_mean),
        comm_sparsity_std=np.float32(comm_sparsity_std),
        mpi_time_mean_inh=np.float32(mpi_time_mean_inh),
        mpi_time_std_inh=np.float32(mpi_time_std_inh),
        comm_frac_mean_inh=np.float32(comm_frac_mean_inh),
        comm_frac_std_inh=np.float32(comm_frac_std_inh),
        total_msgs_mean_inh=np.float32(total_msgs_mean_inh),
        total_msgs_std_inh=np.float32(total_msgs_std_inh),
        total_bytes_mean_inh=np.float32(total_bytes_mean_inh),
        total_bytes_std_inh=np.float32(total_bytes_std_inh),
        mpi_time_mean_iso=np.float32(mpi_time_mean_iso),
        mpi_time_std_iso=np.float32(mpi_time_std_iso),
        comm_frac_mean_iso=np.float32(comm_frac_mean_iso),
        comm_frac_std_iso=np.float32(comm_frac_std_iso),
        total_msgs_mean_iso=np.float32(total_msgs_mean_iso),
        total_msgs_std_iso=np.float32(total_msgs_std_iso),
        total_bytes_mean_iso=np.float32(total_bytes_mean_iso),
        total_bytes_std_iso=np.float32(total_bytes_std_iso)
    )
    
    print(f"Done! Processed data saved to {output_path}")
    print(f"  - job_ids: {unique_jobs}")
    print(f"  - set_features shape: {set_features.shape}")
    print(f"  - isolated_profiles shape: {isolated_profiles.shape}")
    print(f"  - isolated_profiles_raw shape: {isolated_profiles_raw.shape}")
    print(f"  - pairs shape: {pairs.shape}")

if __name__ == "__main__":
    main()
