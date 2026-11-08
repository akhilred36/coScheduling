## C1 noise_cv.pdf
(a) app order (by CV T_MPI, desc): Laghos, LAMMPS, FIESTA, AMG2023, miniVite, Beatnik, Kripke, TriC, MiniFE, Quicksilver
(b) n: isolated inhibitor configs 287, App-Inhibitor configs 2870, App-App endpoint outcomes 110 (incl. self)

## C2 slowdown_ecdf.pdf: log x, 90 non-self outcomes each; dotted line at 1.05
ECDF at 1.05: MPI 0.489, total 0.933

## C3 appapp_heatmap.pdf: rows sorted by victim mean (off-diagonal, desc), cols by aggressor mean (desc); diagonal = mean of the two self-pair copies, boxed
rows: Laghos, Beatnik, AMG2023, MiniFE, miniVite, TriC, Kripke, FIESTA, LAMMPS, Quicksilver
cols: TriC, Beatnik, AMG2023, miniVite, Kripke, FIESTA, MiniFE, LAMMPS, Laghos, Quicksilver
appapp_heatmap_total.pdf rows: Laghos, Beatnik, TriC, miniVite, Quicksilver, AMG2023, MiniFE, Kripke, FIESTA, LAMMPS | cols: TriC, Beatnik, AMG2023, miniVite, FIESTA, MiniFE, LAMMPS, Kripke, Laghos, Quicksilver

## C4 slowdown_vs_profile.pdf (90 non-self MPI outcomes; Spearman on clipped slowdown)
(a) Aggressor byte rate
(B/s): rho = 0.300, p = 0.00408
(b) Aggressor message rate
(msgs/s): rho = 0.049, p = 0.649
(c) Victim comm. fraction
($T_\mathrm{MPI}/T_\mathrm{app}$): rho = 0.091, p = 0.396
Hollow grey markers = floor outcomes (slowdown = 1). Victim comm. fraction panel uses log x as requested.

## C5 inhib_response.pdf
Victims picked by mean log MPI slowdown over all 287 inhibitors (rank 1 = least):
  Quicksilver 0.017, FIESTA 0.108, TriC 0.172, Kripke 0.191, LAMMPS 0.211, miniVite 0.296, Beatnik 0.411, MiniFE 0.548, Laghos 0.633, AMG2023 0.677
  chosen: AMG2023 (most susceptible), miniVite (6th of 10, median-ish), Quicksilver (least)
(a) slices: sparsity = 1.0, wait in {0, 1e3, 1e5, 1e6} us, msg size = base grid {2e3, 4e3, 2e4, 4e4, 2e5, 4e5, 2e6, 4e6} B
(b) slices: wait = 0, msg size in {2 KB, 40 KB, 400 KB, 4 MB}, sparsity in {0.1, 0.3, 0.5, 0.7, 1.0} (base grid)
Both from the base block only (fully populated). y = clipped MPI slowdown (log). Note the App-Inhibitor caveats: inhibitor runtime coverage (AMG2023 0.98, miniVite 1.00, Quicksilver 1.00 median) and socket effect (AMG2023 -0.28 log).
  AMG2023: slowdown range in (a) slice 1.00-5.81; in (b) slice 1.09-4.21
  miniVite: slowdown range in (a) slice 1.02-1.98; in (b) slice 1.07-1.83
  Quicksilver: slowdown range in (a) slice 1.00-1.11; in (b) slice 1.00-1.14

## C6 coverage_pca.pdf
PCA fit on apps + all 287 inhibitors, features standardized by the pooled mean/SD: comm. fraction, log(1+msg rate), log(1+byte rate), log(1+mean msg size)
Explained variance: PC1 0.593, PC2 0.336, PC3 0.071, PC4 0.000 (PC1+PC2 0.929)
Loadings (PC1, PC2): comm. fraction: +0.55, -0.29; log(1+msg rate): +0.00, -0.85; log(1+byte rate): +0.62, -0.12; log(1+mean msg size): +0.56, +0.42
Note: the 4 features are nearly collinear (log byte rate ~ log msg rate + log mean msg size; PC4 = 0.000), so a 4-D hull is degenerate. Hull membership is tested in the 3 independent features (comm. fraction, log msg rate, log mean msg size) by LP feasibility.
Distances: RMS standardized distance with the apps-only scaler (the select_inhibitors.py metric; 1.0 = one app SD per feature on average).
             nearest inhibitor dist             nearest inhibitor  nearest panel dist       nearest panel inhibitor  inside 3-D hull of grid  inside 3-D hull of panel outside grid range on
App                                                                                                                                                                                         
AMG2023                       0.155            8_20000_100000_0.1               0.155            8_20000_100000_0.1                    False                     False                     -
Beatnik                       0.147          8_400000_1000000_0.5               0.147          8_400000_1000000_0.5                     True                     False                     -
FIESTA                        0.470  8_2500000_200000_0.002237136               0.470  8_2500000_200000_0.002237136                    False                     False                     -
Kripke                        0.175  8_2500000_600000_0.002237136               0.175  8_2500000_600000_0.002237136                    False                     False       log(1+msg rate)
Laghos                        0.225               8_500_5000_0.05               0.225               8_500_5000_0.05                     True                     False                     -
LAMMPS                        0.110  8_2500000_600000_0.002237136               0.110  8_2500000_600000_0.002237136                    False                     False       log(1+msg rate)
MiniFE                        0.106           8_40000_1000000_0.1               0.106           8_40000_1000000_0.1                     True                      True                     -
miniVite                      0.291           8_40000_1000000_0.7               0.291           8_40000_1000000_0.7                     True                      True                     -
Quicksilver                   1.348      8_200_200000_0.002237136               1.348      8_200_200000_0.002237136                    False                     False                     -
TriC                          0.159         8_2000000_1000000_0.3               0.159         8_2000000_1000000_0.3                     True                     False                     -

Panel inhibitor knob settings (msg size B, wait us, sparsity):
  8_2000000_1000000_0.3: 2000000 B, 1000000 us, 0.3
  8_20000_100000_0.1: 20000 B, 100000 us, 0.1
  8_200_200000_0.002237136: 200 B, 200000 us, 0.00223714
  8_2500000_200000_0.002237136: 2500000 B, 200000 us, 0.00223714
  8_2500000_600000_0.002237136: 2500000 B, 600000 us, 0.00223714
  8_400000_1000000_0.5: 400000 B, 1000000 us, 0.5
  8_40000_1000000_0.1: 40000 B, 1000000 us, 0.1
  8_40000_1000000_0.7: 40000 B, 1000000 us, 0.7
  8_500_5000_0.05: 500 B, 5000 us, 0.05

## C7 scale_1node.pdf: 30 non-self directional outcomes each, same 6 apps (Beatnik, FIESTA, LAMMPS, MiniFE, miniVite, TriC)
KS two-sample: D = 0.300, p = 0.135

## C8 socket_effect.pdf
Per config (App-Inhibitor) / directional outcome (App-App) having replicates on both sockets: mean log T_MPI(socket 1) - mean log T_MPI(socket 0). Box = IQR, whiskers = 5-95%, line = median; orange dots = App-App directional outcomes, orange tick = their median. Apps ordered by App-Inhibitor median.
             App-Inh n  App-Inh median  App-Inh mean  App-Inh frac<0  App-App n  App-App median
AMG2023            258          -0.243        -0.315           0.787          7          -0.182
FIESTA             258          -0.237        -0.303           0.957          7          -0.365
LAMMPS             247          -0.165        -0.132           0.765          9          -0.158
Laghos             248          -0.155        -0.170           0.843          7          -0.047
MiniFE             240          -0.104        -0.238           0.792          8          -0.063
Kripke             249          -0.063        -0.156           0.835          8          -0.019
Quicksilver        245          -0.052        -0.056           0.996          8          -0.051
TriC               260          -0.011        -0.007           0.562          9           0.008
miniVite           240          -0.004        -0.040           0.508          7          -0.000
Beatnik            255           0.005         0.007           0.455          6          -0.004
All App-Inhibitor configs: n = 2500, median -0.065, mean -0.141, frac<0 0.750, Wilcoxon p = 5.9e-169
All App-App directions: n = 76, median -0.039, mean -0.072, frac<0 0.750, Wilcoxon p = 1e-05
