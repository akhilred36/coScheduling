srun --mpi=pmi2 --ntasks 1 --gpus-per-task=1 -p cup-ecs -x hopper049 ./lmp -in ../../kokkos-miniapps/ExaMiniMD/input/in_100_0.4.lj -sf kk -kokkos on g 1
