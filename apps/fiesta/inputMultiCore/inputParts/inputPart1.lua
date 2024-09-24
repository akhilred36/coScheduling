--
-- 3D Ideal Expansion

title = "3D Idealized Expansion"

--Output Options
write_freq = 0                      --Solution Write Interval
stat_freq = 8000

--Restart Options
restart = 0                           --Whether or not to use restart file
time = 0.0                            --Start time of simulation
tstart = 0                            --Start time index of simulation
restartName = "restart-000000.h5"   --Restart File Name

--MPI Options
mpi = "gpu-aware"		      --Use CUDA-aware MPI

--Gas Properties
R = 8.314462                          --Universal Gas Constant [J/(K*mol)]
ns = 1                                --Number of Gas Species
species_names={"Air"}
gamma = {1.409}                        --Array of Species Ratio of Specific Heats
M = {0.02897}                         --Array of Species Molar Masses [kg/mol]
mu = {2.928e-5}
visc=0
scheme="weno5"
grid="cartesian"
buoyancy = 0

--Time
nt = 10000                             --Time Step at which to end simulation
dt = 1e-7                             --Time Step Size [s]
