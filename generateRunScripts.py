from os import mkdir, listdir, chdir, getcwd
from math import floor, ceil


spack_path="/g/g20/bacon4/spack/share/spack/setup-env.sh"
spack_env_dir="/g/g20/bacon4/coScheduling/spackstuff"
mpip_path="/g/g20/bacon4/spack/opt/spack/linux-rhel8-sapphirerapids/gcc-11.2.1/mpip-3.5-lubjpwtspfd7splvdcvgkko23svfsso3/lib/libmpiP.so"

RUNTIME_OVERLAP_THRESHOLD=0.4
MAX_PROC_PER_NODE = 8
apps = ["beatnik", "fiesta", "lammps", "lulesh", "minife"]
total_proc_choices = [1,8,27,64]

single_proc_runtimes = {
        "beatnik": 1363.89,
        "fiesta": 938,
        "lammps": 568.718,
        "lulesh": 543.766,
        "minife": 37.0947
        }

executables = {
        "beatnik": "./rocketrig",
        "fiesta": "./fiesta",
        "lammps": "./lmp",
        "lulesh": "./lulesh",
        "minife": "./miniFE.x",
        }

app_paths = {
        "beatnik": "/g/g20/bacon4/coScheduling/apps/beatnik/build/examples",
        "fiesta": "/g/g20/bacon4/coScheduling/apps/fiesta/build",
        "lammps": "/g/g20/bacon4/coScheduling/apps/lammps/build",
        "lulesh": "/g/g20/bacon4/coScheduling/apps/lulesh/kokkos-no-uvm/build",
        "minife": "/g/g20/bacon4/coScheduling/apps/miniFE/kokkos/build",
        }

def app_inputs(app, num_procs):
    if (app == "beatnik"):
        return "-n 2048 -w "+str(num_procs)+" -F 0"
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

def generate_combinations(apps):
    combinations = []
    for app1 in apps:
        for app2 in apps:
            if (app1 != app2):
                comb = set([app1,app2])
                if (comb not in combinations):
                    combinations.append(comb)
    return combinations

# Generate co-scheduling run scripts
combinations = generate_combinations(apps)
for c in combinations:
    app_1 = c.pop()
    app_2 = c.pop()
    app_1_active_dir = app_paths[app_1]
    app_2_active_dir = app_paths[app_2]
    app_1_exec = executables[app_1]
    app_2_exec = executables[app_2]
    app_1_singleProcRuntime = single_proc_runtimes[app_1]
    app_2_singleProcRuntime = single_proc_runtimes[app_2]
    print(f"App 1: {app_1}, App 2: {app_2}")
    for num_procs in total_proc_choices:
        app_1_inputs = app_inputs(app_1, num_procs)
        app_2_inputs = app_inputs(app_2, num_procs)
        for i in range(1,4):
            num_nodes = ceil(num_procs/MAX_PROC_PER_NODE)
            ntasks_per_node = ceil((num_procs)/num_nodes)
            # Create directories
            chdir("experiment_scripts")
            dir_name = f"coScheduled_{app_1}_{app_2}_{num_procs}_{num_nodes}_{i}"
            mkdir(dir_name)
            chdir(dir_name)
            mkdir("mpipProfiles")
            dir_path = getcwd()
            chdir("../../")
            # Create Slurm scripts
            chdir("slurm_scripts")
            slurm_file = open(f"{dir_name}.slurm", "w")
            slurm_file.write(f"#!/bin/bash\n")
            slurm_file.write(f"#SBATCH --job-name {dir_name}\n")
            slurm_file.write(f"#SBATCH --mail-user aalasand1@unm.edu\n")
            slurm_file.write(f"#SBATCH --mail-type FAIL,TIME_LIMIT\n")
            slurm_file.write(f"#SBATCH --output {dir_name}.out\n")
            slurm_file.write(f"#SBATCH --error {dir_name}.err\n")
            slurm_file.write(f"#SBATCH --ntasks {num_procs*2}\n")
            slurm_file.write(f"#SBATCH --ntasks-per-node {ntasks_per_node*2}\n")
            slurm_file.write(f"#SBATCH --nodes {num_nodes}\n")
            slurm_file.write(f"#SBATCH --cpus-per-task 1\n")
            slurm_file.write(f"#SBATCH --mem 240G\n")
            slurm_file.write(f"#SBATCH --time 03:00:00\n")
            slurm_file.write(f"#SBATCH --partition pbatch\n")
            # TODO: Add module loads
            slurm_file.write(f"source {spack_path}\n")
            slurm_file.write(f"spack env activate -d {spack_env_dir}\n")
            slurm_file.write(f"module load gcc/11.2.1 cmake/3.23.1 openmpi/4.1.2\n")
            # TODO Update LD_PRELOAD paths
            # slurm_file.write(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74/lib/\n")
            slurm_file.write(f"export LD_PRELOAD={mpip_path}\n")
            slurm_file.write(f"export MPIP=\"-f {dir_path}/mpipProfiles\"\n")
            # slurm_file.write(f"export FI_CXI_DEFAULT_CQ_SIZE=131072\nexport FI_CXI_OFLOW_BUF_SIZE=8388608\nexport FI_CXI_CQ_FILL_PERCENT=20\n")
            if (app_1_singleProcRuntime < RUNTIME_OVERLAP_THRESHOLD*app_2_singleProcRuntime):
                slurm_file.write(f"cd {app_2_active_dir}\n")
                slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_2_exec} {app_2_inputs} | grep measuredTime >> {dir_path}/{app_2}_time.log &\n")
                slurm_file.write(f"app2PID=$!\n")
                slurm_file.write(f"cd {app_1_active_dir}\n")
                slurm_file.write(f"while ps -p $app2PID > /dev/null\n")
                slurm_file.write(f"do\n")
                slurm_file.write(f"\tsrun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_1_exec} {app_1_inputs} | grep measuredTime >> {dir_path}/{app_1}_time.log\n")
                slurm_file.write("done\n")
                slurm_file.write("wait\n")
            elif (app_2_singleProcRuntime < RUNTIME_OVERLAP_THRESHOLD*app_1_singleProcRuntime):
                slurm_file.write(f"cd {app_1_active_dir}\n")
                slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_1_exec} {app_1_inputs} | grep measuredTime >> {dir_path}/{app_1}_time.log &\n")
                slurm_file.write(f"app1PID=$!\n")
                slurm_file.write(f"cd {app_2_active_dir}\n")
                slurm_file.write(f"while ps -p $app1PID > /dev/null\n")
                slurm_file.write(f"do\n")
                slurm_file.write(f"\tsrun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_2_exec} {app_2_inputs} | grep measuredTime >> {dir_path}/{app_2}_time.log\n")
                slurm_file.write("done\n")
                slurm_file.write("wait\n")
            else:
                slurm_file.write(f"cd {app_1_active_dir}\n")
                slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_1_exec} {app_1_inputs} | grep measuredTime >> {dir_path}/{app_1}_time.log &\n")
                slurm_file.write(f"cd {app_2_active_dir}\n")
                slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_2_exec} {app_2_inputs} | grep measuredTime >> {dir_path}/{app_2}_time.log &\n")
                slurm_file.write("wait\n")
            slurm_file.close()
            chdir("../")

# Generate single app run scripts
combinations = generate_combinations(apps)
for app in apps:
    app_active_dir = app_paths[app]
    app_exec = executables[app]
    print(f"App: {app}")
    for num_procs in total_proc_choices:
        app_input = app_inputs(app, num_procs)
        for i in range(1,4):
            num_nodes = ceil(num_procs/MAX_PROC_PER_NODE)
            ntasks_per_node = ceil((num_procs)/num_nodes)
            # Create directories
            chdir("experiment_scripts")
            dir_name = f"nonCoScheduled_{app}_{num_procs}_{num_nodes}_{i}"
            mkdir(dir_name)
            chdir(dir_name)
            mkdir("mpipProfiles")
            dir_path = getcwd()
            chdir("../../")

            # Create Slurm scripts
            chdir("slurm_scripts")
            slurm_file = open(f"{dir_name}.slurm", "w")
            slurm_file.write(f"#!/bin/bash\n")
            slurm_file.write(f"#SBATCH --job-name {dir_name}\n")
            slurm_file.write(f"#SBATCH --mail-user aalasand1@unm.edu\n")
            slurm_file.write(f"#SBATCH --mail-type FAIL,TIME_LIMIT\n")
            slurm_file.write(f"#SBATCH --output {dir_name}.out\n")
            slurm_file.write(f"#SBATCH --error {dir_name}.err\n")
            slurm_file.write(f"#SBATCH --ntasks {num_procs}\n")
            slurm_file.write(f"#SBATCH --ntasks-per-node {ntasks_per_node*2}\n")
            slurm_file.write(f"#SBATCH --nodes {num_nodes}\n")
            slurm_file.write(f"#SBATCH --cpus-per-task 1\n")
            slurm_file.write(f"#SBATCH --mem 240G\n")
            slurm_file.write(f"#SBATCH --time 03:00:00\n")
            slurm_file.write(f"#SBATCH --partition pbatch\n")
            # TODO: Add module loads
            slurm_file.write(f"module load gcc/11.2.1 cmake/3.23.1 openmpi/4.1.2\n")
            slurm_file.write(f"source {spack_path}\n")
            slurm_file.write(f"spack env activate -d {spack_env_dir}\n")
            # TODO Update LD_PRELOAD paths
            # slurm_file.write(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74/lib/\n")
            slurm_file.write(f"export LD_PRELOAD={mpip_path}\n")
            slurm_file.write(f"export MPIP=\"-f {dir_path}/mpipProfiles\"\n")
            slurm_file.write(f"export MPIP=\"-f {dir_path}/mpipProfiles\"\n")
            # slurm_file.write(f"export FI_CXI_DEFAULT_CQ_SIZE=131072\nexport FI_CXI_OFLOW_BUF_SIZE=8388608\nexport FI_CXI_CQ_FILL_PERCENT=20\n")
            slurm_file.write(f"cd {app_active_dir}\n")
            slurm_file.write(f"srun --ntasks {num_procs} --ntasks-per-node {ntasks_per_node} --nodes {num_nodes} --cpus-per-task 1 --mem 120G {app_exec} {app_input} | grep measuredTime >> {dir_path}/{app}_time.log &\n")
            slurm_file.write("wait\n")
            slurm_file.close()
            chdir("../")
