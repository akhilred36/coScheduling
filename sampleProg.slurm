#!/bin/bash
#SBATCH --job-name sampleProg
#SBATCH --mail-user aalasand1@unm.edu
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output sampleProg.out
#SBATCH --error sampleProg.err
#SBATCH --mem 10G
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=00:02:00              # Time limit hrs:min:sec
#SBATCH -p debug,condo,cup-ecs,general

cd /users/aalasand/carc-scratch/workspace/coScheduling
srun -n 1 --mem 4G ./sampleProg1.sh &
app1PID=$!
while ps -p $app1PID > /dev/null
do
    srun -n 1 --mem 4G ./sampleProg2.sh
done
