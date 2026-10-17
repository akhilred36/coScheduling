# Bidirectional Variant Diagrams

One diagram per joint-family variant in `pipeline.py:VARIANT_REGISTRY`, showing
how each is trained and how it produces an App-App prediction. The `legacy_*`
variants are documented in `schematic.md` Part B.

## Shared notation and steps

| Symbol | Meaning |
|---|---|
| `x_V` | 4 standardized isolated-profile features: `comm_frac`, `log1p` message rate, `log1p` byte rate, `log1p` mean message size (fold-local scaler) |
| `e_V = f(x_V)` | shared encoder embedding (same network for applications and inhibitors) |
| `R(V, X)` | latent log slowdown of victim `V` under aggressor `X`: `b(e_V) + h(e_X) + <u(e_V), v(e_X)>` |
| `y` | observed log slowdown, `log(max(1, T_MPI_co / T_MPI_iso))`; `y = 0` is the floor (censored) |
| `D_t(V, X)` | bandwidth head, `g([e_V, e_X, u, v, u*v])`, predicting the rate degradation of endpoint `V` for task `t` |
| panel | the 9 selected inhibitors; every training and calibration label comes from App-Inhibitor runs on this panel |

Steps every variant shares:

- **Episodes.** Pick a training application `A` uniformly, then two distinct
  panel inhibitors: `J` (anchor) and `K` (query). For each one, draw one replicate
  row. A batch holds 128 episodes, and an epoch is 4 batches.
- **Response losses.** The censored absolute Huber loss on `R(A,J)` and `R(A,K)`:
  floor rows are penalized only if the prediction is above 0. The identified
  censored delta Huber loss on `R(A,K) - R(A,J)`: a pair where both rows are at
  the floor adds nothing. Each has weight 1.
- **Training schedule.** Crossed validation (hold out one application × one
  inhibitor block, 5 seeds) picks the number of epochs (median best epoch) and
  the calibration `λ ∈ {0.1, 1, 10}`. The final models are then trained on all
  training applications with 5 seeds, AdamW, lr 1e-3, weight decay 1e-4.
- **Output.** App-App directions `A→B` and `B→A` are evaluated separately,
  with the same response function. Latent predictions are averaged over seeds,
  then clipped at the floor: `ŷ = max(0, mean latent)`.
- **Calibration (`_calibrated` methods).** Fit one offset `c` per victim:
  `argmin_c Σ_J w_J ℓ_J(c) + λ/2 c²`, where `ℓ_J` is `Huber(R(A,J)+c − y_AJ)`
  for exact anchors and `Huber(max(0, R(A,J)+c))` for floor anchors, and
  `w_J` gives each inhibitor equal weight. Then predict `R(A,B) + c`.

Selected settings in `outputs/full_comparison` are listed under each diagram.

---

## 1. `joint_profile`

Absolute + delta response model with a victim-level term. It has no auxiliary
tasks and no fingerprints.

```mermaid
flowchart TB
  subgraph TRAIN["Training"]
    direction TB
    T0["App-Inhibitor rows on the 9-inhibitor panel<br/>(training applications only)"]
    T1["Episode: victim app A,<br/>anchor inhibitor J, query inhibitor K (J ≠ K)"]
    T2["Profiles x_A, x_J, x_K"]
    T3["Shared encoder<br/>e_A, e_J, e_K"]
    T4["R(A,J), R(A,K)<br/>b(e_A) + h(e_X) + ⟨u(e_A), v(e_X)⟩"]
    L1["Absolute loss<br/>censored Huber vs y_AJ, y_AK"]
    L2["Delta loss<br/>censored Huber on R(A,K) − R(A,J)<br/>vs y_AK − y_AJ"]
    LT["Total = 1·absolute + 1·delta<br/>AdamW, epochs fixed by crossed CV"]
    T0 --> T1 --> T2 --> T3 --> T4
    T4 --> L1 --> LT
    T4 --> L2 --> LT
  end

  subgraph INFER["App-App inference: direction A→B"]
    direction TB
    I0["Profiles x_A, x_B"]
    I1["Trained encoder + response head"]
    I2["Direct: R(A,B)"]
    C0["A's 9 panel measurements y_AJ"]
    C1["R(A,J) for each panel inhibitor"]
    C2["Floor-aware offset c<br/>argmin Σ w_J ℓ_J(c) + λ/2 c²"]
    I3["Calibrated: R(A,B) + c"]
    O1["Mean over 5 seeds → max(0, ·)<br/>joint_profile_direct / _calibrated"]
    I0 --> I1 --> I2
    C0 --> C1 --> C2
    I1 --> C1
    I2 --> I3
    C2 --> I3
    I2 --> O1
    I3 --> O1
  end

  TRAIN -.->|"checkpoints, 5 seeds"| INFER
```

Selected: 1 epoch, `λ = 1`.

---

## 2. `bidirectional_profile`

`joint_profile` plus a bandwidth head trained on both endpoints of every
App-Inhibitor run: how much the application's rates degrade (forward), and how
much the inhibitor's rates degrade (reverse).

```mermaid
flowchart TB
  subgraph TRAIN["Training"]
    direction TB
    T0["App-Inhibitor rows on the 9-inhibitor panel<br/>slowdown + app-side and inhibitor-side<br/>rate degradations (bw_wall, bw_mpi, msg_wall)"]
    T1["Episode: victim app A,<br/>anchor inhibitor J, query inhibitor K"]
    T3["Shared encoder<br/>e_A, e_J, e_K"]
    T4["Response head<br/>R(A,J), R(A,K)"]
    T5F["Bandwidth head, forward<br/>D_t(A,J), D_t(A,K)<br/>app endpoint degradation"]
    T5R["Bandwidth head, reverse<br/>D_t(J,A), D_t(K,A)<br/>inhibitor endpoint degradation"]
    L1["Absolute loss"]
    L2["Delta loss"]
    L3["Auxiliary loss<br/>masked Huber, scaled per task/endpoint<br/>mean within task, then across tasks"]
    LT["Total = 1·absolute + 1·delta + 0.1·auxiliary"]
    T0 --> T1 --> T3
    T3 --> T4
    T3 --> T5F
    T3 --> T5R
    T4 --> L1 --> LT
    T4 --> L2 --> LT
    T5F --> L3
    T5R --> L3
    L3 --> LT
  end

  subgraph INFER["App-App inference: direction A→B"]
    direction TB
    I0["Profiles x_A, x_B"]
    I1["Trained encoder + response head"]
    I2["Direct: R(A,B)"]
    C0["A's 9 panel measurements y_AJ"]
    C2["Floor-aware offset c from R(A,J) vs y_AJ"]
    I3["Calibrated: R(A,B) + c"]
    O1["Mean over 5 seeds → max(0, ·)<br/>bidirectional_profile_direct / _calibrated"]
    AX["Diagnostics only: D_t(A,B) and D_t(B,A)<br/>(not used in the slowdown prediction)"]
    I0 --> I1 --> I2 --> I3 --> O1
    I2 --> O1
    C0 --> C2 --> I3
    I1 --> C2
    I1 --> AX
  end

  TRAIN -.->|"checkpoints, 5 seeds"| INFER
```

Selected: 2 epochs, `λ = 1`. The reverse supervision affects the slowdown
prediction only through the shared encoder and the `u`, `v` heads.

---

## 3. `bidirectional_profile_wall` (ablation)

Same as `bidirectional_profile`, but the auxiliary tasks are only the
wall-clock-normalized rates (`bw_wall`, `msg_wall`). MPI-time throughput
(`bw_mpi`) is dropped because it shares a denominator with the slowdown target.

```mermaid
flowchart TB
  subgraph TRAIN["Training"]
    direction TB
    T0["App-Inhibitor rows on the 9-inhibitor panel<br/>slowdown + app-side and inhibitor-side<br/>bw_wall, msg_wall degradations only"]
    T1["Episode: victim app A,<br/>anchor inhibitor J, query inhibitor K"]
    T3["Shared encoder<br/>e_A, e_J, e_K"]
    T4["Response head<br/>R(A,J), R(A,K)"]
    T5F["Bandwidth head, forward<br/>D_t(A,·), t ∈ {bw_wall, msg_wall}"]
    T5R["Bandwidth head, reverse<br/>D_t(·,A), t ∈ {bw_wall, msg_wall}"]
    L1["Absolute loss"]
    L2["Delta loss"]
    L3["Auxiliary loss (2 tasks × 2 endpoints)"]
    LT["Total = 1·absolute + 1·delta + 0.1·auxiliary"]
    T0 --> T1 --> T3
    T3 --> T4
    T3 --> T5F
    T3 --> T5R
    T4 --> L1 --> LT
    T4 --> L2 --> LT
    T5F --> L3
    T5R --> L3
    L3 --> LT
  end

  subgraph INFER["App-App inference: direction A→B"]
    direction TB
    I0["Profiles x_A, x_B"]
    I1["Trained encoder + response head"]
    I2["Direct: R(A,B)"]
    C0["A's 9 panel measurements y_AJ"]
    C2["Floor-aware offset c from R(A,J) vs y_AJ"]
    I3["Calibrated: R(A,B) + c"]
    O1["Mean over 5 seeds → max(0, ·)<br/>bidirectional_profile_wall_direct / _calibrated"]
    I0 --> I1 --> I2 --> I3 --> O1
    I2 --> O1
    C0 --> C2 --> I3
    I1 --> C2
  end

  TRAIN -.->|"checkpoints, 5 seeds"| INFER
```

Selected: 2 epochs, `λ = 1`.

---

## 4. `bidirectional_context`

`bidirectional_profile` plus a fingerprint: the application's embedding is
adjusted by a permutation-invariant summary of its own measured panel
responses. Inhibitors never get a fingerprint.

```mermaid
flowchart TB
  subgraph TRAIN["Training"]
    direction TB
    T0["App-Inhibitor rows on the 9-inhibitor panel<br/>slowdown + forward and reverse rate degradations"]
    T1["Episode: victim app A, anchor J, query K,<br/>support set S ⊂ panel, n_S ∈ 0..7, S ∩ {J,K} = ∅<br/>(S empty with probability 0.25)"]
    K0["One token per inhibitor in S:<br/>inhibitor profile, mean y, floor fraction,<br/>app-side and inhibitor-side degradation<br/>(3 tasks, scaled) + availability masks"]
    K1["Token MLP → masked mean pool z_A"]
    K2["Fusion M(z_A, present, log(1+n_S))<br/>exactly 0 when S is empty"]
    T3["Embeddings<br/>e_A = f(x_A) + M(...),  e_J = f(x_J),  e_K = f(x_K)"]
    T4["Response head<br/>R(A,J), R(A,K)"]
    T5["Bandwidth head<br/>forward D_t(A,·) and reverse D_t(·,A)"]
    LT["Total = 1·absolute + 1·delta + 0.1·auxiliary<br/>(J, K labels never appear in S)"]
    T0 --> T1
    T1 --> K0 --> K1 --> K2 --> T3
    T1 --> T3
    T3 --> T4 --> LT
    T3 --> T5 --> LT
  end

  subgraph INFER["App-App inference: direction A→B"]
    direction TB
    F0["A's 9 panel rows → fingerprint of A<br/>B's 9 panel rows → fingerprint of B"]
    I1["Trained encoder with context fusion<br/>e_A, e_B both include fingerprints"]
    I2["Direct: R(A,B)"]
    X0["Cross-fit A's panel into 3 groups C_1..C_3"]
    X1["View k: fingerprint of A from panel minus C_k<br/>offset c_k fitted on anchors C_k only"]
    X2["View latent: R_k(A,B) + c_k"]
    X3["Calibrated: mean over the 3 views"]
    O1["Mean over 5 seeds → max(0, ·)<br/>bidirectional_context_direct / _calibrated"]
    F0 --> I1 --> I2 --> O1
    X0 --> X1 --> X2 --> X3 --> O1
    I1 --> X1
  end

  TRAIN -.->|"checkpoints, 5 seeds"| INFER
```

Selected: 2 epochs, `λ = 1`. The cross-fitting means no panel label both
shapes the fingerprint and calibrates the offset within the same view. In the
calibrated views, B (the aggressor) keeps its full 9-inhibitor fingerprint.

---

## 5. `bidirectional_context_forward` (ablation)

Same as `bidirectional_context`, except the fingerprint tokens hold only the
application-side degradations. The inhibitor-side (reverse) evidence is left
out of the fingerprint. The auxiliary loss still trains on both directions.

```mermaid
flowchart TB
  subgraph TRAIN["Training"]
    direction TB
    T0["App-Inhibitor rows on the 9-inhibitor panel<br/>slowdown + forward and reverse rate degradations"]
    T1["Episode: victim app A, anchor J, query K,<br/>support set S ⊂ panel, n_S ∈ 0..7, S ∩ {J,K} = ∅"]
    K0["One token per inhibitor in S:<br/>inhibitor profile, mean y, floor fraction,<br/>app-side degradation only (3 tasks) + masks"]
    K1["Token MLP → masked mean pool z_A"]
    K2["Fusion M(z_A, present, log(1+n_S))"]
    T3["Embeddings<br/>e_A = f(x_A) + M(...),  e_J, e_K without context"]
    T4["Response head<br/>R(A,J), R(A,K)"]
    T5["Bandwidth head<br/>forward D_t(A,·) and reverse D_t(·,A)"]
    LT["Total = 1·absolute + 1·delta + 0.1·auxiliary"]
    T0 --> T1
    T1 --> K0 --> K1 --> K2 --> T3
    T1 --> T3
    T3 --> T4 --> LT
    T3 --> T5 --> LT
  end

  subgraph INFER["App-App inference: direction A→B"]
    direction TB
    F0["Forward-only fingerprints of A and B<br/>from their 9 panel rows"]
    I1["Trained encoder with context fusion"]
    I2["Direct: R(A,B)"]
    X0["Cross-fit A's panel into 3 groups"]
    X1["View k: fingerprint from panel minus C_k,<br/>offset c_k fitted on C_k"]
    X3["Calibrated: mean over views of R_k(A,B) + c_k"]
    O1["Mean over 5 seeds → max(0, ·)<br/>bidirectional_context_forward_direct / _calibrated"]
    F0 --> I1 --> I2 --> O1
    X0 --> X1 --> X3 --> O1
    I1 --> X1
  end

  TRAIN -.->|"checkpoints, 5 seeds"| INFER
```

Selected: 1 epoch, `λ = 10`. Compare it with `bidirectional_context` to see
what the reverse evidence adds through the fingerprint.

---

## How the variants differ

| Variant | Victim level `b` | Delta loss | Auxiliary tasks (forward + reverse) | Fingerprint tokens | Calibration |
|---|:-:|:-:|---|---|---|
| `joint_profile` | ✓ | ✓ | — | — | offset on all 9 anchors |
| `bidirectional_profile` | ✓ | ✓ | bw_wall, bw_mpi, msg_wall | — | offset on all 9 anchors |
| `bidirectional_profile_wall` | ✓ | ✓ | bw_wall, msg_wall | — | offset on all 9 anchors |
| `bidirectional_context` | ✓ | ✓ | bw_wall, bw_mpi, msg_wall | app + inhibitor endpoints | 3-view cross-fit |
| `bidirectional_context_forward` | ✓ | ✓ | bw_wall, bw_mpi, msg_wall | app endpoint only | 3-view cross-fit |
