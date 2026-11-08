## Socket placement
Isolated 8-node app runs: first PU counts {0: 24, 112: 16}; socket counts {0: 40}
App-Inhibitor runs: (app socket, inhibitor socket) counts {(0, 1): 6105, (1, 0): 5375}
App-App non-self 8-node runs: (A socket, B socket) counts {(0, 1): 89, (1, 0): 91}

## Within-configuration socket effect (socket 1 minus socket 0, mean log difference)
App-Inhibitor T_MPI: -0.1222 (x0.885), rows in mixed configs = 9998
App-Inhibitor T_app: -0.0199 (x0.980), rows in mixed configs = 9998
App-App T_MPI: -0.0660 (x0.936), rows in mixed directions = 304
App-App T_app: -0.0057 (x0.994), rows in mixed directions = 304

Per-app socket effect (log; negative = faster on socket 1):
             App-Inhibitor T_MPI  App-Inhibitor T_app  App-App T_MPI  App-App T_app
victim                                                                             
AMG2023                   -0.280               -0.024         -0.189         -0.008
Beatnik                    0.006               -0.002          0.004         -0.001
FIESTA                    -0.260               -0.030         -0.312         -0.015
Kripke                    -0.133               -0.017          0.000         -0.001
Laghos                    -0.145               -0.042         -0.057         -0.013
LAMMPS                    -0.112               -0.009         -0.064         -0.005
MiniFE                    -0.203               -0.020         -0.077         -0.004
miniVite                  -0.038               -0.020          0.017          0.002
Quicksilver               -0.048               -0.029         -0.041         -0.019
TriC                      -0.004               -0.006          0.005          0.003

App-App directional outcomes by share of the 4 replicates with the victim on socket 1: {0.0: 7, 0.25: 21, 0.5: 34, 0.75: 21, 1.0: 7}

## Inhibitor runtime coverage of the application run (App-Inhibitor)
             -r used (s)  isolated 8-node T_app (s)  median coverage  p10 coverage  frac runs coverage<0.95
AMG2023            343.0                    342.500            0.980         0.934                    0.148
Beatnik            371.0                    397.734            0.857         0.719                    1.000
FIESTA             299.0                    359.092            0.844         0.801                    1.000
Kripke             355.0                    354.500            0.997         0.959                    0.090
Laghos             332.0                    331.238            0.856         0.730                    0.660
LAMMPS             266.0                    354.750            0.749         0.729                    1.000
MiniFE             384.0                    388.250            0.970         0.937                    0.171
miniVite           442.0                    371.250            1.000         1.000                    0.003
Quicksilver        295.0                    295.000            0.997         0.964                    0.071
TriC               281.0                    378.000            0.681         0.619                    1.000
