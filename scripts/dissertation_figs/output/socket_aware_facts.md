# Socket batch 2, part A facts

## A1. Is the victim socket always the complement of the aggressor socket?
Source: every `mpibind: task <i> ... cpus <PU>` line in each job's output log (mpibind prints one node's 56-task block per job, so this covers one of the 8 nodes per run). Socket = PU mod 112 >= 56.

### App-App (non-self, 8 nodes; directed rows, 2 per run): 360 rows, 180 run dirs
- tasks listed per job: victim [np.int64(56)], aggressor [np.int64(56)]
- hyperthread index sets: victim {'[1]': 360}, aggressor {'[1]': 360}
- jobs spanning both sockets: victim 0, aggressor 0; no placement lines: victim 0, aggressor 0
- (s_V, s_X) counts: {(0, 1): 180, (1, 0): 180}
- complement holds in 360/360 rows; exceptions: 0

### App-Inhibitor: 11480 rows, 11480 run dirs
- tasks listed per job: victim [np.int64(56)], aggressor [np.int64(56)]
- hyperthread index sets: victim {'[1]': 11480}, aggressor {'[1]': 11480}
- jobs spanning both sockets: victim 0, aggressor 0; no placement lines: victim 0, aggressor 0
- (s_V, s_X) counts: {(0, 1): 6105, (1, 0): 5375}
- complement holds in 11480/11480 rows; exceptions: 0

## A2. Replicates per socket cell
### App-App: directed non-self outcomes (victim, aggressor) = 90
- (n reps victim on sock0, sock1) distribution: {(0, 4): 7, (1, 3): 21, (2, 2): 34, (3, 1): 21, (4, 0): 7}
- outcomes with 0 reps on one socket: 14 (all on sock0: 7, all on sock1: 7)
- socket-resolved cells (outcome x victim socket with >= 1 rep): 166; cells with exactly 1 rep: 42
- one-sided outcomes: AMG2023<-Laghos (4/0), AMG2023<-MiniFE (4/0), Beatnik<-FIESTA (4/0), Beatnik<-Laghos (4/0), Beatnik<-miniVite (0/4), FIESTA<-Beatnik (0/4), FIESTA<-Kripke (4/0), Kripke<-FIESTA (0/4), Laghos<-AMG2023 (0/4), Laghos<-Beatnik (0/4), MiniFE<-AMG2023 (0/4), miniVite<-Beatnik (4/0), miniVite<-Quicksilver (0/4), Quicksilver<-miniVite (4/0)
- share of victim reps on sock1 per app: AMG2023 0.42, Beatnik 0.47, FIESTA 0.58, Kripke 0.47, Laghos 0.50, LAMMPS 0.64, MiniFE 0.47, miniVite 0.47, Quicksilver 0.42, TriC 0.56

### App-Inhibitor, 90 panel rows (10 apps x 9 panel inhibitors): 90 cells (app x inhibitor)
- (n0, n1) distribution: {(0, 4): 3, (1, 3): 14, (2, 2): 32, (3, 1): 33, (4, 0): 8}
- cells with 0 reps on one socket: 11 (all sock0 8, all sock1 3); resolved cells with >= 1 rep: 169; cells with exactly 1 rep: 47
- share of app reps on sock1 per app: AMG2023 0.33, Beatnik 0.47, FIESTA 0.50, Kripke 0.28, Laghos 0.22, LAMMPS 0.47, MiniFE 0.53, miniVite 0.50, Quicksilver 0.36, TriC 0.53

### App-Inhibitor, full grid: 2870 cells (app x inhibitor)
- (n0, n1) distribution: {(0, 4): 133, (1, 3): 639, (2, 2): 1065, (3, 1): 796, (4, 0): 237}
- cells with 0 reps on one socket: 370 (all sock0 237, all sock1 133); resolved cells with >= 1 rep: 5370; cells with exactly 1 rep: 1435
- share of app reps on sock1 per app: AMG2023 0.46, Beatnik 0.46, FIESTA 0.48, Kripke 0.47, Laghos 0.46, LAMMPS 0.50, MiniFE 0.45, miniVite 0.47, Quicksilver 0.45, TriC 0.48
