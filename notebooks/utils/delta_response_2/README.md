# Delta Response 2

Delta Response 2 is a standalone, sealed `random_split` pipeline for predicting
directional MPI co-scheduling slowdown from application profiles and
App-Inhibitor measurements. It develops and freezes a model without loading
App-App outcomes.

The package supports only the all-application condition:

- All ten application profiles and valid App-Inhibitor responses may contribute
  to final fitting.
- App-App labels do not influence fitting, candidate selection, checkpoint
  choice, OOD calibration, or model scaling.
- Historical App-App evaluation is a separate retrospective procedure defined
  in `experiments.md`.

## Documentation

Read these files before changing or evaluating the pipeline:

- `schematic.md`: model architecture, censored objective, inference corrections,
  validation design, and the actually frozen winner.
- `experiments.md`: autonomous-agent instructions for reproducing the historical
  static `random_split` evaluation used for Delta Response 1.
- `scientific_spec.md`: prespecified development estimands and selection policy.
- `AUDIT_REPRODUCTION.md`: reproduction of the inherited App-Inhibitor summary.
- `delta_response_2.md`: design history, evidence, and scientific constraints.

## What Changed

Compared with Delta Response 1, this implementation adds:

- A convex victim correction that can use exact and floor-censored anchors.
- Explicit legacy exact-anchor candidates for a controlled ablation.
- Label-free application-profile-weighted validation views.
- Profile-space and complete mechanism-family holdouts.
- Compact fixed-step neural training without query-selected checkpoints.
- Informative censored-pair sampling that excludes floor-floor episodes.
- Intercept, direct-anchor, censored linear, tree, and absolute-neural baselines.
- Equal-fold multi-view selection, rank gates, seed-stability gates, Pareto
  filtering, and a grouped one-standard-error complexity preference.
- Immutable input/code/config/fold seals, resumable fold shards, validated final
  checkpoints, and an exact completion-output roster.

## Environment

Run commands from:

```text
notebooks/utils
```

Use the requested Conda environment:

```bash
conda run -n hpcResearch python --version
```

The completed run used Python 3.11.8, PyTorch 2.4.1+cu121, NumPy 1.23.5,
pandas 2.2.1, SciPy 1.10.1, and scikit-learn 1.2.2. CUDA initialization was
unavailable during that run, so `experiments/compact_v1` is sealed to a
CPU-visible environment.

## Inputs

Defaults resolve to `few_shot/data/`:

```text
jobs.csv        application profiles
inhibitors.csv  synthetic-inhibitor profiles and mechanism metadata
job_inh.csv     directional App-Inhibitor slowdown responses
```

There is intentionally no pair-data argument in `plan`, `run`, `status`, or
`predict-profile`.

## Tests

Run the redesigned suite before creating a new plan:

```bash
conda run -n hpcResearch python -m unittest discover \
  -s delta_response_2/tests -v
```

The current suite covers censor inequalities, potential invariants, fixed-step
budgets, label-free weights, fold leakage, query-label independence, manifests,
completion seals, checkpoint rosters, and frozen replay.

## Fresh Full Run

Planning writes immutable input hashes, source hashes, resolved configuration,
candidate definitions, inhibitor blocks, and every fold role. Use a new empty
root for every new plan:

```bash
conda run -n hpcResearch python -m delta_response_2.run_experiments plan \
  --root delta_response_2/experiments/compact_v2 \
  --config delta_response_2/configs/compact_search.json

conda run -n hpcResearch python -m delta_response_2.run_experiments run \
  --root delta_response_2/experiments/compact_v2

conda run -n hpcResearch python -m delta_response_2.run_experiments status \
  --root delta_response_2/experiments/compact_v2
```

Keep device visibility identical for all three commands. If CPU execution is
required, prefix every command with `CUDA_VISIBLE_DEVICES=""`. The run is
resumable: completed fold shards are hash-verified and skipped.

Do not edit top-level package Python files, the resolved config, source inputs,
or plan manifests after `plan`. Any such edit correctly invalidates the plan.
Create another root instead of weakening verification.

## Completed Reference Run

The completed five-seed run is:

```text
delta_response_2/experiments/compact_v1
```

Verify it with the same CPU visibility used when it was planned:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n hpcResearch \
  python -m delta_response_2.run_experiments status \
  --root delta_response_2/experiments/compact_v1
```

It completed 400 fold-seed tasks for 53 candidates across profile and mechanism
folds. The selected method is `current_delta_ood_h8_e4`, a five-seed compact
low-rank potential with an OOD-gated exact-anchor correction. Its worst primary
App-Inhibitor development log MAE is `0.166523`.

These are development results. They are not App-App transfer estimates.

## Frozen Profile Replay

Replay the frozen recipe for two known profile IDs:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n hpcResearch \
  python -m delta_response_2.run_experiments predict-profile \
  --root delta_response_2/experiments/compact_v1 \
  --victim-id amg \
  --aggressor-id beatnik \
  --device cpu
```

The command verifies the plan, exact completed-file roster, all hashes, final
checkpoint schema, configured seed roster, and recipe binding before inference.
It returns each seed's latent prediction and the latent-space ensemble. It
accepts IDs only and cannot load an App-App outcome file.

## Main Artifacts

Inside a completed root:

```text
seal.json                                  immutable development-plan seal
completion_seal.json                       final completion seal
output_hashes.json                         exact declared output roster and hashes
run_report.json                            machine-readable run summary
FINAL_REPORT.md                            concise scientific report
frozen_recipe.json                         selected recipe and checkpoint binding
fold_manifest.csv                          row-level fold roles and scaler membership
candidate_manifest.csv                     complete candidate registry
tasks/oof_predictions.csv                  all seed-specific OOF predictions
tasks/training_curves.csv                   fixed-step learning curves
tasks/final_checkpoints/                    validated final artifacts
consolidated/candidate_selection.csv        gates and selection objectives
consolidated/ensemble_metrics.csv           latent-ensemble development metrics
consolidated/metrics_per_victim.csv         victim-level diagnostics
consolidated/paired_victim_bootstrap.csv    descriptive development comparisons
```

Never place evaluation files inside a completed root. Completion verification
rejects undeclared additions. Use a sibling output root as required by
`experiments.md`.

## Historical Static Evaluation

To evaluate the frozen recipe on the same historical static `random_split`
dataset used by Delta Response 1, follow `experiments.md` exactly. That protocol:

- Verifies the frozen model before pair-data access.
- Keeps evaluation-only code and outputs outside `compact_v1`.
- Uses the same 55 pair rows, 90 primary non-self directions, metrics, and
  1,000-draw unordered-pair bootstrap.
- Reports the result as retrospective and diagnostic because the historical
  pair table is already contaminated.
- Prohibits pair-informed retraining, calibration, threshold changes, candidate
  selection, or acceptance claims.

## App-Inhibitor Audit

The inherited saved App-Inhibitor predictions can be re-audited without any
App-App input:

```bash
conda run -n hpcResearch python -m delta_response_2.audit_saved \
  --saved-predictions delta_response_1/experiments/random_split_model_search_v1/capacity_limit/crossed_validation_predictions.csv \
  --output-dir delta_response_2/audit_outputs/app_inhibitor_reproduction_v3
```

The output directory must not already exist; choose a new versioned path for a
later rerun.

See `AUDIT_REPRODUCTION.md` for the recorded estimands and values.
