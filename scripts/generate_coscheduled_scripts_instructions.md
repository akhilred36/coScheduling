# Generate isolated scripts
Write a python script, scripts/generate_coscheduled_scripts.py to do as follows. Make sure to ask any questions if instructions are unclear.

## Important variables, file paths, modules
1) modules are: 
```gcc/10 openmpi cmake```
2) "spack_setup" is: ```/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh```
3) "spack_env_dir" is: ```/g/g90/alasandagutt1/spack_envs/beatnik/```
4) "mpip_path" is:```/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so```
5) "run_configs_path" is:```/g/g90/alasandagutt1/repos/coScheduling/run_configs/```. The format of .json files in this directory is ```<num_nodes>_nodes.json```. For example, ```1_nodes.json``` refers to configurations for 1 node runs
6) "base_repo_path" is:```/g/g90/alasandagutt1/```. All of the paths shown in run_configs_path json files are relative to "base_repo_path"
7) "experiments_path" is ```/p/lustre2/alasandagutt1/experiments_coscheduling_app_coscheduled/```

## Script variables
The following variables can be modified by the user. Set them at a convenient location at the top of the script:
1) "node_choices": set to [1]
2) "redundant_runs": set to 4
3) "walltime": set to "00:45:00"
4) "email": set to "aalasand1@unm.edu"
5) "num_cpus": set to 112
6) "mem": "240G"

## The script
1) At the beginning of the script, check if "experiments_path" exists. If it doesn't exist, create the "experiments_coscheduling_app_coscheduled" directory in "experiments_path" and cd into it. If it exists, simply cd into it
2) Create a new directory with a name that is the current timestamp, and cd into it
3) Create a directory called "slurm_scripts", and create a directory called "data".
4) Iterate through "node_choices", and the iterating variable is called "num_nodes":
    1) Read ```run_configs_path/<num_nodes>_nodes.json``` and retrieve all of the information pertaining to the applications, their inputs, paths, executables, etc.
    2) Iterate through all of the applications twice in a nested fashion, so that the iterating variables is "a" and "b". The goal is to pair every app a with every app b. Something to note here is that a="beatnik",b="fiesta" is functionally the same as a="fiesta",b="beatnik". Precompute these combinations in advance to avoid these functional duplicates:
        1) Iterate from i=0 to i < redundant_runs:
            1) Create a file in "slurm_scripts" titled "num_nodes_a_b_i.slurm" where num_nodes, a, b, and i are the variables from the iterations.
            2) Create a directory in "data" titled "num_nodes_a_b_i" using the same variables. Create a directory inside this newly created directory titled "mpip_profiles"
            3) Now, write the following header to the created slurm file:
            ```
            #!/bin/bash
            #SBATCH --job-name <num_nodes>_<a>_<b>_<i>_isolated
            #SBATCH --mail-user <email>
            #SBATCH --mail-type FAIL,TIME_LIMIT
            #SBATCH --output <num_nodes>_<a>_<b>_<i>_isolated.out
            #SBATCH --error <num_nodes>_<a>_<b>_<i>_isolated.err
            #SBATCH --ntasks <num_cpus> 
            #SBATCH --ntasks-per-node <num_cpus>
            #SBATCH --nodes <num_nodes>
            #SBATCH --mem <mem>
            #SBATCH --time <walltime>
            #SBATCH --partition pbatch
            #SBATCH --distribution block:cyclic
            ```
            4) Now do the following:
                1) module load from the module var declared above
                2) source "spack_setup"
                3) spack env activate "spack_env_dir"
                4) Run the following for app "a" and redirect the output to "output_a.log" in the appropriate "num_nodes_a_b_i" directory pertaining to this particular run. The "exec" and "args" are derived from the .json file. The resources for this srun are half of what were allocated in the batch script.
                ```
                time srun --exclusive -n num_cpus/2 --mem mem/2 --distribution=block:block --cpu-bind=cores env LD_PRELOAD="mpip_path" MPIP="-f <path/to/num_nodes_a_b_i/mpip_profiles>" ./exec <args> &
                ```
            5) Run the following for app "b" and redirect the output to "output_a.log" in the appropriate "num_nodes_a_b_i" directory pertaining to this particular run. The "exec" and "args" are derived from the .json file. The resources for this srun are half of what were allocated in the batch script.
                ```
                time srun --exclusive -n num_cpus/2 --mem mem/2 --distribution=block:block --cpu-bind=cores env LD_PRELOAD="mpip_path" MPIP="-f <path/to/num_nodes_a_b_i/mpip_profiles>" ./exec <args> &
                ```
            6) Now wait for them to complete:
            ```wait```