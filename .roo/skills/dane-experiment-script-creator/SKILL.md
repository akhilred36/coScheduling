---
name: dane-experiment-script-creator
description: Use this skill when the user needs to create experiment scripts to run on the Dane HPC system
modeSlugs:
  - architect
  - code
---

# Dane Experiment Script Creator

## Instructions

### The python script
1) You will be writing a python script in scripts/ . The user will provide you with the exact name of the script, but prompt them if they forget to do that.
2) This python script will be creating individual slurm scripts for experiments which can then all be submitted in one big batch.

### What the script does
1) The script creates a directory in /p/lustre2/alasandagutt1/ for the experiments. The name of this directory will be provided by the user, but prompt them if they forget to do this. I will henceforth refer to this directory for experiments as "experiments_dir".
2) The script should create experiments_dir in /p/lustre2/alasandagutt1 only if it doesn't already exist.
3) The script should create a subdirectory in experiments_dir which is given the name of the current timestamp. I will henceforth refer to this subdirectory as timestamp_dir.
4) The script should create two directories inside timestamp_dir: "data", and "slurm_scripts".
5) The slurm_scripts directory is where the script will write the individual slurm scripts to, and the data directory will contain several subdirectories, each that will contain the output of the corresponding slurm script after it finishes executing. Each subdirectory and the corresponding slurm script will have the same name. The format of this name will be provided by the user. For example, if the slurm script is named "run_1_1.slurm", there will be a corresponding directory in data/ named "run_1_1", inside which there will be output data related to the experiment. I will henceforth refer to this name as "experiment_name"
6) It is very important that you use absolute paths for everything.

### Slurm script format
1) The following is the slurm script header format:
```
#!/bin/bash
#SBATCH --job-name <experiment_name>
#SBATCH --mail-user aalasand1@unm.edu
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output <experiment_name>.out
#SBATCH --error <experiment_name>.err
#SBATCH --ntasks <num_cpus> 
#SBATCH --ntasks-per-node <num_cpus>
#SBATCH --nodes <num_nodes>
#SBATCH --mem <mem>
#SBATCH --time <walltime>
#SBATCH --partition pbatch
#SBATCH --distribution block:block
```
2) All of the information in angular brackets (<>) will be provided by the user. They may either be constant values, or the script may be iterating through some lists to generate them. The user will explain what the particular experiment creation script should do.
3) After the slurm header, the slurm script will load the necessary modules, and/or activate the appropriate spack environment. The user will give you this information, but prompt them if they forget.
4) After the environemnt is set, the script will run the main srun(s). The user will provide you with details of these. The output of the sruns should be redirected to the corresponding data directory. The name of the output(s) will be determined by the user.