from os import mkdir, listdir, chdir, getcwd
from math import floor, ceil

spack_path="/g/g20/bacon4/spack/share/spack/setup-env.sh"
#spack_env_dir="/p/lustre1/bacon4/coScheduling/spackstuff"
spack_env_dir="/g/g20/bacon4/coScheduling/spackstuff"
#mpip_path="/p/lustre1/bacon4/coScheduling/mpip-3.5/libmpip.so"
mpip_path="/g/g20/bacon4/coScheduling/mpip-3.5/libmpip.so"

RUNTIME_OVERLAP_THRESHOLD=0.4
MAX_PROC_PER_NODE = 8

apps = ["beatnik", "fiesta", "lammps", "lulesh", "minife"]

total_proc_choices = [1,8,27,64]

executables = {
        "beatnik": "./rocketrig",
        "fiesta": "./fiesta",
        "lammps": "./lmp",
        "lulesh": "./lulesh",
        "minife": "./miniFE.x",
        "inhib": "./networkInhib"
        }

app_paths = {
        "beatnik": "/p/lustre1/bacon4/coScheduling/apps/beatnik/build/examples",
        "fiesta": "/p/lustre1/bacon4/coScheduling/apps/fiesta/build",
        "lammps": "/p/lustre1/bacon4/coScheduling/apps/lammps/build",
        "lulesh": "/p/lustre1/bacon4/coScheduling/apps/lulesh/kokkos-no-uvm/build",
        "minife": "/p/lustre1/bacon4/coScheduling/apps/miniFE/kokkos/build",
        }



network_inhib_path = "/p/lustre1/bacon4/coScheduling/build/"

def app_inputs(app, num_procs):
    if (app == "beatnik"):
        return "-n 2048 -w "+str(num_procs) + "-F 0"
    elif (app == "fiesta"):
        if (num_procs == 1):
            return "../input/fiesta_1_60_5000ts.lua"
        if (num_procs == 8):
            return "../input/fiesta_8_120_5000ts.lua"
        if (num_procs == 27):
            return "../input/fiesta_27_180_5000ts.lua"
        if (num_procs == 64):
            return "../input/fiesta_64_240_5000ts.lua"
    elif (app == "lammps"):
        if (num_procs == 1):
            return "-in ../../examinimd/input/in_130_0.9.lj"
        if (num_procs == 8):
            return "-in ../../examinimd/input/in_260_0.9.lj"
        if (num_procs == 27):
            return "-in ../../examinimd/input/in_390_0.9.lj"
        if (num_procs == 64):
            return "-in ../../examinimd/input/in_520_0.9.lj"
    elif (app == "lulesh"):
        return "-s 110"
    elif (app == "minife"):
        if (num_procs == 1):
            return "--nx=150 --ny=150 --nz=150"
        if (num_procs == 8):
            return "--nx=300 --ny=300 --nz=300"
        if (num_procs == 27):
            return "--nx=450 --ny=450 --nz=450"
        if (num_procs == 64):
            return "--nx=600 --ny=600 --nz=600"

# Generate co-scheduling run scripts

inhib_msg_sizes = [1000]
inhib_wait_times = [0]

for app_1 in apps:
    app_1_active_dir = app_paths[app_1]
    app_1_exec = executables[app_1]
    inhib_exec = executables["inhib"]
    for num_procs in total_proc_choices:
        for msg_size in inhib_msg_sizes:
            for wait_time in inhib_wait_times:
                app_1_inputs = app_inputs(app_1, num_procs)
                for i in range(1, 4):
                    num_nodes = ceil(num_procs/MAX_PROC_PER_NODE)
                    ntasks_per_node = ceil((num_procs*2)/num_nodes)
                    # Create directories
                    chdir("experiment_scripts_inhibitor")
                    dir_name = f"coScheduledInhib_{app_1}_{num_procs}_{num_nodes}_{msg_size}_{wait_time}_{i}"
                    mkdir(dir_name)
                    chdir(dir_name)
                    mkdir("mpipProfiles")
                    dir_path = getcwd()
                    chdir("../../")
                    # Create Slurm scripts
                    chdir("slurm_scripts_inhibitor")
                    slurm_file = open(f"{dir_name}.slurm", "w")
                    slurm_file.write(f"#!/bin/bash\n")
                    slurm_file.write(f"#SBATCH --job-name {dir_name}\n")
                    slurm_file.write(f"#SBATCH --mail-user aalasand1@unm.edu\n")
                    slurm_file.write(f"#SBATCH --mail-type FAIL,TIME_LIMIT\n")
                    slurm_file.write(f"#SBATCH --output {dir_name}.out\n")
                    slurm_file.write(f"#SBATCH --error {dir_name}.err\n")
                    slurm_file.write(f"#SBATCH --ntasks {num_procs*2}\n")
                    slurm_file.write(f"#SBATCH --ntasks-per-node {ntasks_per_node}\n")
                    slurm_file.write(f"#SBATCH --nodes {num_nodes}\n")
                    slurm_file.write(f"#SBATCH --cpus-per-task 1\n")
                    slurm_file.write(f"#SBATCH --mem 240G\n")
                    slurm_file.write(f"#SBATCH --time 02:00:00\n")
                    slurm_file.write(f"#SBATCH --partition pbatch\n")
                    # TODO: Add module loads
                    slurm_file.write(f"source {spack_path}\n")
                    slurm_file.write(f"spack env activate -d {spack_env_dir}\n")
                    slurm_file.write(f"module load gcc/11.2.1 cmake/3.23.1 openmpi/4.1.2\n")
                    slurm_file.write(f"cd {network_inhib_path}\n")
                    # slurm_file.write(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74/lib/\n")
                    slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {int(ntasks_per_node/2)} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {inhib_exec} -m {msg_size} -w {wait_time} &\n")
                    slurm_file.write(f"cd {app_1_active_dir}\n")
                    # TODO Update LD_PRELOAD paths
                    slurm_file.write(f"export LD_PRELOAD={mpip_path}\n")
                    slurm_file.write(f"export MPIP=\"-f {dir_path}/mpipProfiles\"\n")
                    slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {int(ntasks_per_node/2)} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_1_exec} {app_1_inputs} | grep measuredTime >> {dir_path}/{app_1}_time.log\n")
                    slurm_file.write("scancel $SLURM_JOB_ID\n")
                    slurm_file.write("wait\n")
                    slurm_file.close()
                    chdir("../")
