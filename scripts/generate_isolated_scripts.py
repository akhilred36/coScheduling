#!/usr/bin/env python3
"""
Script to generate isolated SLURM scripts for running applications with MPIP profiling.
"""

import os
import json
import time
from datetime import datetime

# =============================================================================
# User-modifiable variables
# =============================================================================
node_choices = [1]
redundant_runs = 4
walltime = "00:30:00"
email = "aalasand1@unm.edu"
num_cpus = 56
mem = "120G"

# =============================================================================
# Important variables, file paths, modules
# =============================================================================
modules = ["gcc/10", "openmpi", "cmake"]
spack_setup = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
spack_env_dir = "/g/g90/alasandagutt1/spack_envs/beatnik/"
mpip_path = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
run_configs_path = "/g/g90/alasandagutt1/repos/coScheduling/run_configs/"
base_repo_path = "/g/g90/alasandagutt1/"
experiments_path = "/p/lustre2/alasandagutt1/experiments_coscheduling_isolated/"

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
    # Step 4.1: Read run_configs_path/<num_nodes>_nodes.json
    config_file = os.path.join(run_configs_path, f"{num_nodes}_nodes.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    apps = config.get("apps", {})
    
    # Step 4.2: Iterate through all applications
    for app_name, app_info in apps.items():
        # Step 4.3: Iterate through redundant runs
        for i in range(redundant_runs):
            # Create filenames and directories
            slurm_filename = f"{num_nodes}_{app_name}_{i}.slurm"
            slurm_filepath = os.path.join(slurm_scripts_dir, slurm_filename)
            
            data_subdir_name = f"{num_nodes}_{app_name}_{i}"
            data_subdir = os.path.join(data_dir, data_subdir_name)
            os.makedirs(data_subdir)
            
            mpip_profiles_dir = os.path.join(data_subdir, "mpip_profiles")
            os.makedirs(mpip_profiles_dir)
            
            # Step 4.3.3: Write SLURM header
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name {num_nodes}_{app_name}_{i}_isolated
#SBATCH --mail-user {email}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {num_nodes}_{app_name}_{i}_isolated.out
#SBATCH --error {num_nodes}_{app_name}_{i}_isolated.err
#SBATCH --ntasks {num_cpus} 
#SBATCH --ntasks-per-node {num_cpus}
#SBATCH --nodes {num_nodes}
#SBATCH --mem {mem}
#SBATCH --time {walltime}
#SBATCH --partition pbatch
#SBATCH --distribution block:block
"""
            
            # Step 4.3.4: Add module loads, spack setup, and execution command
            # Load modules
            for module in modules:
                slurm_content += f"module load {module}\n"
            
            # Source spack setup
            slurm_content += f"source {spack_setup}\n"
            
            # Activate spack environment
            slurm_content += f"spack env activate {spack_env_dir}\n"
            
            # Build the execution command
            exec_path = app_info.get("exec", "")
            args = app_info.get("args", {})
            
            # Convert args dict to command line arguments
            args_str = ""
            for key, value in args.items():
                if key == "":
                    # Handle empty key (for fiesta)
                    args_str += f" {value}"
                else:
                    args_str += f" {key} {value}"
            
            # Build the full command with LD_PRELOAD and MPIP
            # Paths in JSON are relative to base_repo_path
            exec_full_path = os.path.join(base_repo_path, app_info.get("path", ""), exec_path)
            output_log = os.path.join(data_subdir, "output.log")
            
            srun_command = f"srun -n {num_cpus} --distribution=block:block --cpu-bind=cores env LD_PRELOAD=\"{mpip_path}\" MPIP=\"-f {mpip_profiles_dir}\" {exec_full_path}{args_str} > {output_log} 2>&1"
            slurm_content += srun_command + "\n"
            
            # Write the SLURM file
            with open(slurm_filepath, 'w') as f:
                f.write(slurm_content)
            
            print(f"Created SLURM script: {slurm_filepath}")
            print(f"Created data directory: {data_subdir}")

print("All SLURM scripts generated successfully!")
