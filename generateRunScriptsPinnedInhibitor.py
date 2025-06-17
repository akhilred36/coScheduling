from os import mkdir, listdir, chdir, getcwd
from math import floor, ceil

#apps = ["beatnik", "fiesta", "lammps", "lulesh", "minife"]
apps = ["beatnik"]

total_proc_per_node_choices = [1,4,8,16]
total_node_choices = [1,2,4]

executables = {
        "beatnik": "./rocketrig",
        "fiesta": "./fiesta",
        "lammps": "./lmp",
        "lulesh": "./lulesh",
        "minife": "./miniFE.x",
        "inhib": "./networkInhib"
        }

app_paths = {
        "beatnik": "/u/aalasand1/hpcResearch/coScheduling/apps/beatnik/build/examples",
        "fiesta": "/u/aalasand1/hpcResearch/coScheduling/apps/fiesta/build",
        "lammps": "/u/aalasand1/hpcResearch/coScheduling/apps/lammps/build",
        "lulesh": "/u/aalasand1/hpcResearch/coScheduling/apps/lulesh/kokkos-no-uvm/build",
        "minife": "/u/aalasand1/hpcResearch/coScheduling/apps/miniFE/kokkos/src",
        }

network_inhib_path = "/u/aalasand1/hpcResearch/coScheduling/build/"

def app_inputs(app, num_procs):
    if (app == "beatnik"):
        return "-n 2048 -w "+str(num_procs)
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

inhib_msg_sizes = [100, 1000]
inhib_wait_times = [0, 10]

# Specify pinnings for number of procs selected
# Format: app_proc_1_cpu,app_proc_2_cpu,...,app_proc_n_cpu:inhib_proc_1_cpu,inhib_proc_2_cpu,...,inhib_proc_n_cpu
proc_per_node_pinnings_map = {1: "1:0",
        4: "1,2,3,4:0",
        8: "1,2,3,4,65,66,67,68:0,64",
        16: "1,2,3,4,33,34,35,36,65,66,67,68,97,98,99,100:0,32,64,96"}

for app_1 in apps:
    app_1_active_dir = app_paths[app_1]
    app_1_exec = executables[app_1]
    inhib_exec = executables["inhib"]
    for procs_per_node in total_proc_per_node_choices:
        for num_nodes in total_node_choices:
            for msg_size in inhib_msg_sizes:
                for wait_time in inhib_wait_times:
                    app_1_inputs = app_inputs(app_1, procs_per_node*num_nodes)
                    for i in range(0, 1):
                        pinnings_map = proc_per_node_pinnings_map[procs_per_node]
                        inhib_num_procs_per_node = len(pinnings_map.split(":")[1].split(","))
                        inhib_proc_mapping = pinnings_map.split(":")[1]
                        app_proc_mapping = pinnings_map.split(":")[0]
                        # Create directories
                        chdir("experiment_scripts_pinnedCPUs_inhibitor")
                        dir_name = f"coScheduledPinnedInhib_{app_1}_{num_nodes}_{procs_per_node}_{msg_size}_{wait_time}_{i}"
                        mkdir(dir_name)
                        chdir(dir_name)
                        dir_path = getcwd()
                        chdir("../../")
                        # Create Slurm scripts
                        chdir("slurm_scripts_pinnedCPUs_inhibitor")
                        slurm_file = open(f"{dir_name}.slurm", "w")
                        slurm_file.write('#!/bin/bash\n')
                        slurm_file.write(f'#SBATCH --job-name {dir_name}\n')
                        slurm_file.write(f'#SBATCH --nodes {num_nodes}\n')
                        slurm_file.write(f'#SBATCH --mem 240G\n')
                        slurm_file.write(f'#SBATCH --time 02:00:0\n')
                        slurm_file.write(f'#SBATCH --partition cpu\n')
                        slurm_file.write(f'#SBATCH --output {dir_name}.out\n')
                        slurm_file.write(f'#SBATCH --exclusive\n')
                        slurm_file.write(f'#SBATCH --account bckq-delta-cpu\n')
                        slurm_file.write(f'cd {network_inhib_path}\n')
                        slurm_file.write(f'srun -n {inhib_num_procs_per_node*num_nodes} --ntasks-per-node {inhib_num_procs_per_node} --nodes {num_nodes} --mem 120G --overlap --cpu-bind=verbose,v,map_cpu:{inhib_proc_mapping} {inhib_exec} -m {msg_size} -w {wait_time} &\n')
                        slurm_file.write(f'network_pid=$!\n')
                        slurm_file.write(f'cd {app_1_active_dir}\n')
                        slurm_file.write(f'sleep 10\n')
                        slurm_file.write(f'srun -n {procs_per_node*num_nodes} --ntasks-per-node {procs_per_node} --nodes {num_nodes} --mem 120G --cpu-bind=verbose,v,map_cpu:{app_proc_mapping} {app_1_active_dir}/{app_1_exec} {app_1_inputs} | grep measuredTime > {dir_path}/{app_1}_time.log\n')
                        slurm_file.write(f'kill $network_pid\n')
                        slurm_file.write(f'wait\n')
                        slurm_file.close()
                        chdir("../")
