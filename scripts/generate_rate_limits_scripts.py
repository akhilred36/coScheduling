#!/usr/bin/env python3
import os
import time

experiments_dir = "/p/lustre2/alasandagutt1/experiments_rate_limits"
base_repo_path = "/g/g90/alasandagutt1/repos/coScheduling"

if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

timestamp_dir = os.path.join(experiments_dir, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(timestamp_dir)

data_dir = os.path.join(timestamp_dir, "data")
os.makedirs(data_dir)

slurm_scripts_dir = os.path.join(timestamp_dir, "slurm_scripts")
os.makedirs(slurm_scripts_dir)

num_cpus = 896
num_nodes = 8
mem = "240G"
walltime = "24:00:00"
ntasks_per_node = 112
mail_user = "aalasand1@unm.edu"

for i in range(10):
    experiment_name = f"rate_limits_{i}"

    exp_data_dir = os.path.join(data_dir, experiment_name)
    os.makedirs(exp_data_dir)

    slurm_script_path = os.path.join(slurm_scripts_dir, f"{experiment_name}.slurm")

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name {experiment_name}
#SBATCH --mail-user {mail_user}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {experiment_name}.out
#SBATCH --error {experiment_name}.err
#SBATCH --ntasks {num_cpus}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --nodes {num_nodes}
#SBATCH --mem {mem}
#SBATCH --time {walltime}
#SBATCH --partition pbatch
#SBATCH --distribution block:block

module load gcc/10
module load openmpi
module load cmake

export DATA_DIR={exp_data_dir}

srun --exclusive -n 448 --mem 119G --nodes 8 --ntasks-per-node 56 --distribution=block:block --mpibind=on,v {base_repo_path}/build/map_bandwidths -o $DATA_DIR/bandwidths.csv
srun --exclusive -n 448 --mem 119G --nodes 8 --ntasks-per-node 56 --distribution=block:block --mpibind=on,v {base_repo_path}/build/map_rate_limits -o $DATA_DIR/rate_limits.csv
"""

    with open(slurm_script_path, "w") as f:
        f.write(slurm_content)

    print(f"Created {slurm_script_path}")

print("\nAll SLURM scripts created successfully!")
print(f"Experiments directory: {timestamp_dir}")
