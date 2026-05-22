#!/usr/bin/env python3
"""
Script to generate isolated SLURM scripts for co-scheduling experiments.
This script creates SLURM job scripts that run pairs of applications concurrently
with MPIP profiling enabled.
"""

import os
import json
import subprocess
from datetime import datetime
from itertools import combinations_with_replacement

# =============================================================================
# USER-MODIFIABLE VARIABLES
# =============================================================================
node_choices = [1]
redundant_runs = 4
walltime = "00:45:00"
email = "aalasand1@unm.edu"
num_cpus = 112
mem = "240G"

# =============================================================================
# IMPORTANT VARIABLES (DO NOT MODIFY)
# =============================================================================
modules = ["gcc/10", "openmpi", "cmake"]
spack_setup = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
spack_env_dir = "/g/g90/alasandagutt1/spack_envs/beatnik/"
mpip_path = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
run_configs_path = "/g/g90/alasandagutt1/repos/coScheduling/run_configs/"
base_repo_path = "/g/g90/alasandagutt1/repos/coScheduling/"
experiments_path = "/p/lustre2/alasandagutt1/experiments_coscheduling_app_coscheduled/"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_app_combinations(apps_dict, redundant_runs):
    """
    Generate all unique app pairs (a, b) where a != b to avoid duplicates.
    beatnik+fiesta is the same as fiesta+beatnik, so we only generate one.
    """
    app_names = list(apps_dict.keys())
    pairs = []

    # Use combinations_with_replacement to get all unique pairs
    # This ensures we don't have duplicates like (a, b) and (b, a)
    for a, b in combinations_with_replacement(app_names, 2):
        # Skip when a == b (same app paired with itself)
        if a != b:
            for i in range(redundant_runs):
                pairs.append((a, b, i))

    return pairs

def format_args(args_dict):
    """
    Format the arguments dictionary into a command-line string.
    """
    formatted_args = []
    for key, value in args_dict.items():
        if key == "":
            # Handle empty key (for fiesta)
            formatted_args.append(str(value))
        else:
            formatted_args.append(f"{key} {value}")
    return " ".join(formatted_args)

def generate_slurm_header(num_nodes, app_a, app_b, run_id):
    """
    Generate the SLURM header for a job script.
    """
    job_name = f"{num_nodes}_{app_a}_{app_b}_{run_id}_isolated"
    output_file = f"{job_name}.out"
    error_file = f"{job_name}.err"

    header = f"""#!/bin/bash
#SBATCH --job-name {job_name}
#SBATCH --mail-user {email}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {output_file}
#SBATCH --error {error_file}
#SBATCH --ntasks {num_cpus}
#SBATCH --ntasks-per-node {num_cpus}
#SBATCH --nodes {num_nodes}
#SBATCH --mem {mem}
#SBATCH --time {walltime}
#SBATCH --partition pbatch
#SBATCH --distribution block:cyclic
"""
    return header

def generate_slurm_content(num_nodes, app_a, app_b, run_id, apps_dict, data_dir):
    """
    Generate the full SLURM script content.
    """
    header = generate_slurm_header(num_nodes, app_a, app_b, run_id)

    # Module loads and spack setup
    script_lines = [header]
    script_lines.append("")

    # Load modules
    for module in modules:
        script_lines.append(f"module load {module}")

    script_lines.append("")
    script_lines.append(f"source {spack_setup}")
    script_lines.append(f"spack env activate {spack_env_dir}")
    script_lines.append("")

    # Get app configurations
    app_a_config = apps_dict[app_a]
    app_b_config = apps_dict[app_b]

    # Calculate resources for each app (half of allocated)
    tasks_per_app = num_cpus // 2
    mem_per_app = mem  # Note: mem is a string, we keep it as is

    # Format arguments for each app
    args_a = format_args(app_a_config["args"])
    args_b = format_args(app_b_config["args"])

    # Build full paths for executables
    exec_a = os.path.join(base_repo_path, app_a_config["path"], app_a_config["exec"])
    exec_b = os.path.join(base_repo_path, app_b_config["path"], app_b_config["exec"])

    # MPIP profile directory
    mpip_profile_dir = os.path.join(data_dir, "mpip_profiles")

    # Run app a
    srun_a = f'time srun --exclusive -n {tasks_per_app} --mem {mem_per_app} --distribution=block:block --cpu-bind=cores env LD_PRELOAD="{mpip_path}" MPIP="-f {mpip_profile_dir}" {exec_a} {args_a} &'
    script_lines.append(srun_a)

    # Run app b
    srun_b = f'time srun --exclusive -n {tasks_per_app} --mem {mem_per_app} --distribution=block:block --cpu-bind=cores env LD_PRELOAD="{mpip_path}" MPIP="-f {mpip_profile_dir}" {exec_b} {args_b} &'
    script_lines.append(srun_b)

    # Wait for both to complete
    script_lines.append("")
    script_lines.append("wait")

    return "\n".join(script_lines) + "\n"

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    # Step 1: Check if experiments_path exists, create if not, and cd into it
    experiments_base = experiments_path.rsplit("/", 1)[0]  # /p/lustre2/alasandagutt1
    experiments_dir_name = "experiments_coscheduling_app_coscheduled"
    experiments_full_path = experiments_path

    if not os.path.exists(experiments_full_path):
        os.makedirs(experiments_full_path, exist_ok=True)
        print(f"Created directory: {experiments_full_path}")

    os.chdir(experiments_full_path)
    print(f"Changed to directory: {os.getcwd()}")

    # Step 2: Create a new directory with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(experiments_full_path, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    os.chdir(timestamp_dir)
    print(f"Created and changed to timestamp directory: {timestamp_dir}")

    # Step 3: Create slurm_scripts and data directories
    slurm_scripts_dir = os.path.join(timestamp_dir, "slurm_scripts")
    data_dir = os.path.join(timestamp_dir, "data")
    os.makedirs(slurm_scripts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created slurm_scripts directory: {slurm_scripts_dir}")
    print(f"Created data directory: {data_dir}")

    # Step 4: Iterate through node_choices
    for num_nodes in node_choices:
        # Read the JSON configuration file
        json_file = os.path.join(run_configs_path, f"{num_nodes}_nodes.json")
        print(f"Reading configuration from: {json_file}")

        with open(json_file, 'r') as f:
            config = json.load(f)

        apps_dict = config["apps"]
        print(f"Found applications: {list(apps_dict.keys())}")

        # Precompute app combinations to avoid duplicates
        app_combinations = get_app_combinations(apps_dict, redundant_runs)
        print(f"Generated {len(app_combinations)} app combinations")

        # Iterate through all app pairs
        for app_a, app_b, run_id in app_combinations:
            # Create SLURM script
            slurm_filename = f"{num_nodes}_{app_a}_{app_b}_{run_id}.slurm"
            slurm_filepath = os.path.join(slurm_scripts_dir, slurm_filename)

            # Create data directory for this run
            run_data_dir = os.path.join(data_dir, f"{num_nodes}_{app_a}_{app_b}_{run_id}")
            os.makedirs(run_data_dir, exist_ok=True)

            # Create mpip_profiles subdirectory
            mpip_profiles_dir = os.path.join(run_data_dir, "mpip_profiles")
            os.makedirs(mpip_profiles_dir, exist_ok=True)

            # Generate and write SLURM script
            slurm_content = generate_slurm_content(
                num_nodes, app_a, app_b, run_id, apps_dict, run_data_dir
            )

            with open(slurm_filepath, 'w') as f:
                f.write(slurm_content)

            print(f"Created SLURM script: {slurm_filepath}")
            print(f"Created data directory: {run_data_dir}")
            print(f"Created mpip_profiles directory: {mpip_profiles_dir}")

if __name__ == "__main__":
    main()
