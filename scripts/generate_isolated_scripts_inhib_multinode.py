#!/usr/bin/env python3
"""
Script to generate isolated SLURM scripts for running various inhib configurations.
"""

import os
import json
import itertools
import time
from datetime import datetime

# =============================================================================
# User-modifiable variables
# =============================================================================
num_nodes = 8
redundant_runs = 4
walltime = "00:10:00"
email = "aalasand1@unm.edu"
num_cpus = 56
mem = 240
inhib_runtime = "3600" # in seconds # in seconds # in seconds # in seconds

# =============================================================================
# Important variables, file paths, modules
# =============================================================================
modules = ["gcc/10", "openmpi", "cmake"]
spack_setup = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
spack_env_dir = "/g/g90/alasandagutt1/spack_envs/beatnik/"
mpip_path = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
run_configs_path = "/g/g90/alasandagutt1/repos/coScheduling/run_configs/"
base_repo_path = "/g/g90/alasandagutt1/repos/coScheduling/"
experiments_path = "/p/lustre2/alasandagutt1/experiments_coscheduling_inhib_isolated/"

# =============================================================================
# Main script logic
# =============================================================================

# Step 1: Check if experiments_path exists, create if not, and cd into it
if not os.path.exists(experiments_path):
    os.makedirs(experiments_path)
    print(f"Created experiments directory: {experiments_path}")
os.chdir(experiments_path)
print(f"Changed to directory: {os.getcwd()}")

# Step 2: Create a new directory with current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp_dir = os.path.join(experiments_path, timestamp)
os.makedirs(timestamp_dir)
os.chdir(timestamp_dir)
print(f"Created and changed to timestamp directory: {timestamp_dir}")

# Step 3: Create slurm_scripts and data directories
slurm_scripts_dir = os.path.join(timestamp_dir, "slurm_scripts")
os.makedirs(slurm_scripts_dir)
print(f"Created slurm_scripts directory: {slurm_scripts_dir}")

data_dir = os.path.join(timestamp_dir, "data")
os.makedirs(data_dir)
print(f"Created data directory: {data_dir}")

# Step 4: Read inhib config
config_file = os.path.join(run_configs_path, f"{num_nodes}_nodes_inhib.json")
with open(config_file, 'r') as f:
    config = json.load(f)

inhib = config.get("inhib", {})
inhib_path = inhib.get("path", "")
inhib_exec = inhib.get("exec", "")
inhib_args = inhib.get("args", {})

# Build full inhib executable path
inhib_full_path = os.path.join(base_repo_path, inhib_path, inhib_exec)

# Step 5: Generate all combinations of inhib arguments
inhib_arg_keys = list(inhib_args.keys())
inhib_arg_values = [inhib_args[key] for key in inhib_arg_keys]
inhib_combinations = list(itertools.product(*inhib_arg_values))

# Step 6: Iterate through all inhib combinations
for i, inhib_combo in enumerate(inhib_combinations):
    inhib_args_dict = dict(zip(inhib_arg_keys, inhib_combo))
    
    # Generate experiment name
    params_str = "_".join([str(val) for val in inhib_combo])
    experiment_name = f"{num_nodes}_inhib_{params_str}"
    
    # Step 6.1: Iterate through redundant runs
    for r in range(redundant_runs):
        # Create filenames and directories
        slurm_filename = f"{experiment_name}_{r}.slurm"
        slurm_filepath = os.path.join(slurm_scripts_dir, slurm_filename)

        data_subdir_name = f"{experiment_name}_{r}"
        data_subdir = os.path.join(data_dir, data_subdir_name)
        os.makedirs(data_subdir)

        mpip_profiles_dir = os.path.join(data_subdir, "mpip_profiles")
        os.makedirs(mpip_profiles_dir)

        # Step 6.1.3: Write SLURM header
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name {experiment_name}_{r}
#SBATCH --mail-user {email}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {experiment_name}_{r}.out
#SBATCH --error {experiment_name}_{r}.err
#SBATCH --ntasks {num_cpus*2*num_nodes}
#SBATCH --ntasks-per-node {num_cpus*2}
#SBATCH --cpus-per-task 1
#SBATCH --nodes {num_nodes}
#SBATCH --mem {mem}G
#SBATCH --time {walltime}
#SBATCH --partition pbatch
#SBATCH --distribution block:cyclic
"""

        # Step 6.1.4: Add module loads, spack setup, and execution command
        # Load modules
        for module in modules:
            slurm_content += f"module load {module}\n"

        # Source spack setup
        slurm_content += f"source {spack_setup}\n"

        # Activate spack environment
        slurm_content += f"spack env activate {spack_env_dir}\n"

        # Build inhib arguments string
        inhib_args_str = ""
        for key, value in inhib_args_dict.items():
            if key == "-r":
                # Skip -r for isolated runs (applicable for coscheduled only)
                continue
            inhib_args_str += f" {key} {value}"

        # Build the full command with LD_PRELOAD and MPIP
        output_log = os.path.join(data_subdir, "output.log")

        srun_command = f"time srun -n {num_cpus*num_nodes} --ntasks-per-node {num_cpus} --nodes {num_nodes} --mem {int(mem/2)-1}G --distribution=block:block --mpibind=on,v env LD_PRELOAD=\"{mpip_path}\" MPIP=\"-f {mpip_profiles_dir}\" {inhib_full_path}{inhib_args_str} -r {inhib_runtime} > {output_log} 2>&1"
        slurm_content += srun_command + "\n"

        # Write the SLURM file
        with open(slurm_filepath, 'w') as f:
            f.write(slurm_content)

        print(f"Created SLURM script: {slurm_filepath}")
        print(f"Created data directory: {data_subdir}")

print("All SLURM scripts generated successfully!")
