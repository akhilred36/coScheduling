#!/usr/bin/env python3
"""
Script to generate SLURM experiment scripts for co-scheduling network inhibitor
with all available applications on the Dane HPC system.
"""

import json
import os
import itertools
from datetime import datetime


# Constants
WORKSPACE_PATH = "/g/g90/alasandagutt1/repos/coScheduling/"
EXPERIMENTS_DIR = "/p/lustre2/alasandagutt1/experiments_coscheduling_inhib_coscheduled/"
NUM_NODES = 1
NUM_CPUS = 112
MEMORY = "240G"
WALLTIME = "00:45:00"
MAIL_USER = "aalasand1@unm.edu"
MODULES = ["gcc/10", "openmpi", "cmake"]
SPACK_ENV_PATH = "/g/g90/alasandagutt1/spack_envs/beatnik/"
SPACK_SETUP_ENV = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
NETWORK_INHIBITOR_EXEC = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
MPIP_FLAGS = "-f"

# Paths to config files
CONFIG_APPS = "run_configs/1_nodes.json"
CONFIG_INHIB = "run_configs/1_nodes_inhib.json"


def load_json_config(filepath):
    """Load and parse a JSON configuration file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_experiment_name(app_name, inhib_args, r):
    """
    Generate experiment name in format:
    1_<app_name>_inhib_<m_value>_<w_value>_<i_value>_<c_value>_<s_value>_<r>
    Only values are used, not argument keys.
    """
    inhib_params = []
    for key, value in inhib_args.items():
        inhib_params.append(str(value))
    
    params_str = "_".join(inhib_params)
    return f"1_{app_name}_inhib_{params_str}_{r}"


def generate_slurm_script(experiment_name, data_dir, inhib_exec, inhib_args_str, 
                          app_exec, app_args_str, mpip_prof_path):
    """
    Generate the content of a SLURM script for the experiment.
    """
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name {experiment_name}
#SBATCH --mail-user {MAIL_USER}
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output {experiment_name}.out
#SBATCH --error {experiment_name}.err
#SBATCH --ntasks {NUM_CPUS}
#SBATCH --ntasks-per-node {NUM_CPUS}
#SBATCH --nodes {NUM_NODES}
#SBATCH --mem {MEMORY}
#SBATCH --time {WALLTIME}
#SBATCH --partition pbatch
#SBATCH --distribution block:block

# Load modules
module load {' '.join(MODULES)}

# Activate Spack environment
source {SPACK_SETUP_ENV} && spack env activate {SPACK_ENV_PATH}

# Create mpip profiles directory
mkdir -p {mpip_prof_path}

# Run network inhibitor in background (runs forever until killed)
srun --exclusive -n 56 --mem 119G --distribution=block:block --cpu-bind=cores {inhib_exec} {inhib_args_str} > {data_dir}/inhib_output.log 2>&1 &

# Run the application with MPIP profiling in foreground
time srun --exclusive -n 56 --mem 119G --distribution=block:block --cpu-bind=cores env LD_PRELOAD="{NETWORK_INHIBITOR_EXEC}" MPIP="{MPIP_FLAGS} {mpip_prof_path}" {app_exec} {app_args_str} > {data_dir}/app_output.log 2>&1

# Cancel the job to kill the background inhibitor process
scancel $SLURM_JOB_ID
"""
    return slurm_content


def main():
    """Main function to generate all experiment scripts."""
    
    # Load configuration files
    apps_config = load_json_config(CONFIG_APPS)
    inhib_config = load_json_config(CONFIG_INHIB)
    
    apps = apps_config["apps"]
    inhib = inhib_config["inhib"]
    
    # Get inhibitor executable path and name
    inhib_path = inhib["path"]
    inhib_exec = inhib["exec"]
    inhib_args = inhib["args"]
    
    # Construct full absolute path for inhibitor executable
    inhib_full_path = f"{WORKSPACE_PATH}{inhib_path}{inhib_exec}"
    
    # Store app paths for later use
    app_paths = {}
    for app_name, app_info in apps.items():
        app_paths[app_name] = f"{WORKSPACE_PATH}{app_info['path']}{app_info['exec']}"
    
    # Create experiments directory if it doesn't exist
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)
        print(f"Created experiments directory: {EXPERIMENTS_DIR}")
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(EXPERIMENTS_DIR, timestamp)
    os.makedirs(timestamp_dir)
    print(f"Created timestamp directory: {timestamp_dir}")
    
    # Create data and slurm_scripts subdirectories
    data_dir = os.path.join(timestamp_dir, "data")
    slurm_scripts_dir = os.path.join(timestamp_dir, "slurm_scripts")
    os.makedirs(data_dir)
    os.makedirs(slurm_scripts_dir)
    print(f"Created data directory: {data_dir}")
    print(f"Created slurm_scripts directory: {slurm_scripts_dir}")
    
    # Generate all combinations of inhibitor arguments
    inhib_arg_keys = list(inhib_args.keys())
    inhib_arg_values = [inhib_args[key] for key in inhib_arg_keys]
    
    inhib_combinations = list(itertools.product(*inhib_arg_values))
    
    # Generate experiment scripts
    scripts_generated = 0
    for inhib_combo in inhib_combinations:
        inhib_args_dict = dict(zip(inhib_arg_keys, inhib_combo))
        
        for r in range(0, 4):  # r from 0 to 3 inclusive
            for app_name, app_info in apps.items():
                # Generate experiment name
                experiment_name = generate_experiment_name(app_name, inhib_args_dict, r)
                
                # Build inhibitor arguments string
                inhib_args_str = " ".join([f"{key} {value}" for key, value in inhib_args_dict.items()])
                
                # Build app arguments string
                app_args_list = []
                for key, value in app_info["args"].items():
                    if key:  # Skip empty keys
                        app_args_list.append(f"{key} {value}")
                    else:
                        app_args_list.append(value)
                app_args_str = " ".join(app_args_list)
                
                # Create data subdirectory for this experiment
                experiment_data_dir = os.path.join(data_dir, experiment_name)
                os.makedirs(experiment_data_dir)
                
                # Create mpip_profiles subdirectory
                mpip_prof_path = os.path.join(experiment_data_dir, "mpip_profiles")
                os.makedirs(mpip_prof_path)
                
                # Generate SLURM script
                slurm_script_path = os.path.join(slurm_scripts_dir, f"{experiment_name}.slurm")
                slurm_content = generate_slurm_script(
                    experiment_name=experiment_name,
                    data_dir=experiment_data_dir,
                    inhib_exec=inhib_full_path,
                    inhib_args_str=inhib_args_str,
                    app_exec=app_paths[app_name],
                    app_args_str=app_args_str,
                    mpip_prof_path=mpip_prof_path
                )
                
                with open(slurm_script_path, 'w') as f:
                    f.write(slurm_content)
                
                scripts_generated += 1
                print(f"Generated: {experiment_name}.slurm")
    
    print(f"\nTotal scripts generated: {scripts_generated}")
    print(f"Experiments directory: {EXPERIMENTS_DIR}")
    print(f"Timestamp directory: {timestamp_dir}")
    print(f"Data directory: {data_dir}")
    print(f"SLURM scripts directory: {slurm_scripts_dir}")


if __name__ == "__main__":
    main()
