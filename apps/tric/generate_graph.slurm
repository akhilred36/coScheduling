#!/bin/bash
#SBATCH --job-name generate_graph
#SBATCH --mail-user aalasand1@unm.edu
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output generate_graph.out
#SBATCH --error generate_graph.err
#SBATCH --ntasks 448
#SBATCH --ntasks-per-node 112
#SBATCH --cpus-per-task 1
#SBATCH --nodes 4
#SBATCH --mem 240G
#SBATCH --time 02:00:00
#SBATCH --partition pbatch

module load gcc/10
module load openmpi
module load cmake

cd /g/g90/alasandagutt1/repos/coScheduling/apps/tric
srun -n 448 ./generate_graph_mpi 72000000 2160000000 /p/lustre2/alasandagutt1/tric_graphs/72000000_2160000000.bin
