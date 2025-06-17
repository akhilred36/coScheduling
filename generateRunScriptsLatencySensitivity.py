# Latency sensitivity

from os import mkdir, listdir, chdir, getcwd

ping_pong_path="/u/aalasand1/hpcResearch/coScheduling/build"
network_inhib_path = "/u/aalasand1/hpcResearch/coScheduling/build/"

inhib_msg_sizes = [0, 100, 1000, 10000, 100000, 1000000]
inhib_wait_times = [0, 10, 100, 1000]

executables = {
    "beatnik": "./rocketrig",
    "fiesta": "./fiesta",
    "lammps": "./lmp",
    "lulesh": "./lulesh",
    "minife": "./miniFE.x",
    "inhib": "./networkInhib"
}

inhib_exec = executables["inhib"]

for i in range(10):
    for msg_size in inhib_msg_sizes:
        for wait_time in inhib_wait_times:
            chdir("experiment_scripts_latencySensitivity")
            dir_name = f"coScheduled_latencySensitivity_{msg_size}_{wait_time}_{i}"
            mkdir(dir_name)
            chdir("../")
            chdir("slurm_scripts_latencySensitivity")
            slurm_file = open(f"{dir_name}.slurm", "w")
            slurm_file.write(f"#!/bin/bash\n")
            slurm_file.write(f"#SBATCH --job-name {dir_name}\n")
            slurm_file.write(f"#SBATCH --mail-user aalasand1@unm.edu\n")
            slurm_file.write(f"#SBATCH --mail-type FAIL,TIME_LIMIT\n")
            slurm_file.write(f"#SBATCH --output {dir_name}.out\n")
            slurm_file.write(f"#SBATCH --error {dir_name}.err\n")
            slurm_file.write(f"#SBATCH --ntasks 4\n")
            slurm_file.write(f"#SBATCH --ntasks-per-node 2\n")
            slurm_file.write(f"#SBATCH --ntasks-per-socket 2\n")
            slurm_file.write(f"#SBATCH --nodes 2\n")
            slurm_file.write(f"#SBATCH --cpus-per-task 1\n")
            slurm_file.write(f"#SBATCH --mem 240G\n")
            slurm_file.write(f"#SBATCH --time 00:45:00\n")
            slurm_file.write(f"#SBATCH --partition cpu\n")
            slurm_file.write(f"#SBATCH --account bckq-delta-cpu\n")
            slurm_file.write(f"module load openmpi gcc/11 cmake\n")
            slurm_file.write(f"cd {network_inhib_path}\n")
            if (msg_size > 0):
                slurm_file.write(f"srun --ntasks 2 --ntasks-per-node 1 --nodes 2 --ntasks-per-socket 1 --cpus-per-task 1 --mem 200G {inhib_exec} -m {msg_size} -w {wait_time} &\n")
            slurm_file.write(f"cd {ping_pong_path}\n")
            for j in range(250):
                slurm_file.write(f"srun -n 2 --mem 30G --cpus-per-task 1 --ntasks-per-node 1 --nodes 2 --ntasks-per-socket 1 ./pingPong_latency >> ../experiment_scripts_latencySensitivity/{dir_name}/output.log\n")
            slurm_file.write("scancel $SLURM_JOB_ID\n")
            slurm_file.write("wait\n")
            slurm_file.close()
            chdir("../")
