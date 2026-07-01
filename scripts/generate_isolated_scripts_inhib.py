#!/usr/bin/env python3
"""
Script to generate isolated SLURM scripts for running inhibitor configurations.
"""

import os
import json
import itertools
import time
from datetime import datetime

# =============================================================================
# User-modifiable variables
# =============================================================================
node_choices = [8]
redundant_runs = 4
walltime = "00:20:00"
email = "aalasand1@unm.edu"
num_cpus = 112
mem = "240G"
INHIB_RUNTIME = "3600" # seconds

# =============================================================================
# Important variables, file paths, modules
# =============================================================================
modules = ["gcc/10", "openmpi", "cmake"]
spack_setup = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
spack_env_dir = "/g/g90/alasandagutt1/spack_envs/beatnik/"
network_inhibitor_path = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
run_configs_path = "/g/g90/alasandagutt1/repos/coScheduling/run_configs/"
base_repo_path = "/g/g90/alasandagutt1/repos/coScheduling/"
experiments_path = "/p/lustre2/alasandagutt1/experiments_coscheduling_isolated_inhib/"

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

# Step 4: Iterate through node_choices
for num_nodes in node_choices:
    # Step 4.1: Read run_configs_path/<num_nodes>_nodes_inhib.json
    config_file = os.path.join(run_configs_path, f"{num_nodes}_nodes_inhib.json")
    with open(config_file, 'r') as f:
        config = json.load(f)

    inhib_config = config.get("inhib", {})

    # Step 4.2: Get inhibitor arguments
    inhib_args = inhib_config.get("args", {})
    inhib_path = inhib_config.get("path", "")
    inhib_exec = inhib_config.get("exec", "")

    # Build full inhibitor path
    inhib_full_path = os.path.join(base_repo_path, inhib_path, inhib_exec)

    # Step 4.3: Generate all combinations of inhibitor arguments
    inhib_arg_keys = list(inhib_args.keys())
    inhib_arg_values = [inhib_args[key] for key in inhib_arg_keys]

    inhib_combinations = list(itertools.product(*inhib_arg_values))

    # Step 4.4: Iterate through all combinations
    for inhib_combo in inhib_combinations:
        inhib_args_dict = dict(zip(inhib_arg_keys, inhib_combo))
        
        # Step 4.5: Iterate through redundant runs
        for i in range(redundant_runs):
            # Create filenames and directories
            inhib_args_str = " ".join([f"{key} {value}" for key, value in inhib_args_dict.items()])
            params_str = "_".join([str(v) for v in inhib_combo])
            slurm_filename = f"{num_nodes}_inhib_{params_str}_{i}.slurm"
            slurm_filepath = os.path.join(slurm_scripts_dir, slurm_filename)

            data_subdir_name = f"{num_nodes}_inhib_{params_str}_{i}"
            data_subdir = os.path.join(data_dir, data_subdir_name)
            os.makedirs(data_subdir)

            mpip_profiles_dir = os.path.join(data_subdir, "mpip_profiles")
            os.makedirs(mpip_profiles_dir)

            # Step 4.5.1: Write SLURM header
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name {slurm_filename}
#SBATCH --mail-user {email}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {slurm_filename}.out
#SBATCH --error {slurm_filename}.err
#SBATCH --ntasks {num_cpus}
#SBATCH --ntasks-per-node {num_cpus}
#SBATCH --nodes {num_nodes}
#SBATCH --mem {mem}
#SBATCH --time {walltime}
#SBATCH --partition pbatch
#SBATCH --distribution block:block
"""

            # Step 4.5.2: Add module loads, spack setup, and execution command
            # Load modules
            for module in modules:
                slurm_content += f"module load {module}\n"

            # Source spack setup
            slurm_content += f"source {spack_setup}\n"

            # Activate spack environment
            slurm_content += f"spack env activate {spack_env_dir}\n"

            # Build the execution command with network inhibitor
            output_log = os.path.join(data_subdir, "output.log")

            srun_command = f"time srun --exclusive -n {num_cpus} --distribution=block:block --cpu-bind=cores --mpibind=on,v env LD_PRELOAD=\"{network_inhibitor_path}\" MPIP=\"-f {mpip_profiles_dir}\" {inhib_full_path} {inhib_args_str} -r {INHIB_RUNTIME}> {output_log} 2>&1"
            slurm_content += srun_command + "\n"

            # Write the SLURM file
            with open(slurm_filepath, 'w') as f:
                f.write(slurm_content)

            print(f"Created SLURM script: {slurm_filepath}")
            print(f"Created data directory: {data_subdir}")

print("All SLURM scripts generated successfully!")
