from os import chdir, listdir, mkdir, remove
from shutil import rmtree

chdir("experiment_scripts")
all_files = listdir(".")

for f in all_files:
    if "minife" not in f:
        rmtree(f)

chdir("../experiment_scripts_inhibitor")
all_files = listdir(".")

for f in all_files:
    if "minife" not in f:
        rmtree(f)

chdir("../slurm_scripts")
all_files = listdir(".")

for f in all_files:
    if "minife" not in f:
        remove(f)

chdir("../slurm_scripts_inhibitor")
all_files = listdir(".")

for f in all_files:
    if "minife" not in f:
        remove(f)
