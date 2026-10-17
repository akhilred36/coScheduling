# Corrected Pipeline Verification

## Test Commands

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n hpcResearch \
  python -m unittest discover -s tests -v
```

Result: 14 tests passed.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n hpcResearch \
  python audit_outputs/audit_tests.py
```

Result: 9 independent audit tests passed.

Execution deviation: the first compatibility run of the pre-existing audit
test harness wrote to its fixed `audit_outputs/synthetic_crossed_validation/`
directory and overwrote that synthetic run's CV artifacts. No real or
pair-derived artifact was involved. The harness was then repaired to use a
temporary directory under `audit_outputs/`, so subsequent runs do not overwrite
historical outputs.

## Synthetic End-To-End Reproducibility

The following command was run twice, changing only the output directory from
`corrected_repro_run_1` to `corrected_repro_run_2`:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n hpcResearch python pipeline.py \
  --jobs-csv audit_outputs/synthetic_data/jobs.csv \
  --inhibitors-csv audit_outputs/synthetic_data/inhibitors.csv \
  --job-inh-csv audit_outputs/synthetic_data/job_inh.csv \
  --pair-csv audit_outputs/synthetic_data/pair.csv \
  --output-dir audit_outputs/corrected_repro_run_1 \
  --cv-seeds 0 1 --final-seeds 0 1 \
  --inhibitor-blocks 2 --max-epochs 3 --patience 2 \
  --batches-per-epoch 2 --bootstrap-samples 100
```

Results:

- Every artifact was byte-identical except `run_report.json`.
- The two reports were identical after removing `elapsed_seconds`.
- Every saved synthetic prediction had finite latent/observed log values,
  `predicted_log_slowdown >= 0`, and `predicted_slowdown >= 1`.
- All saved synthetic metric and bootstrap estimate/interval fields were finite.
- Low-rank and generic aggregation both selected kernel.
- Selected final epochs were low-rank 3, generic 3, and absolute 1.

Deviations from defaults:

- CV seeds: two (`0, 1`) rather than five.
- Final seeds: two (`0, 1`) rather than five.
- Inhibitor blocks: 2 rather than 4.
- Maximum epochs: 3 rather than 80.
- Patience: 2 rather than 10.
- Batches per epoch: 2 rather than 4.
- Bootstrap samples: 100 rather than 1000.
- Input data was deterministic synthetic data under `audit_outputs/`.

## App-Inhibitor-Only Crossed Validation

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n hpcResearch python pipeline.py \
  --output-dir audit_outputs/app_inhib_only_corrected \
  --skip-holdout \
  --cv-seeds 0 1 --final-seeds 0 1 \
  --inhibitor-blocks 4 --max-epochs 12 --patience 4 \
  --batches-per-epoch 2
```

Results:

- Runtime: 177.28 seconds on CUDA for the final recorded run.
- Valid rows: 2,327 of 2,339; 12 missing-profile rows quarantined.
- Censored floor rows: 249.
- Model/fold/seed fits: 240, 80 for each model kind.
- Finite clipped-target CV prediction records checked: 344,396.
- Low-rank selected kernel at temperature 0.1, grouped log-MAE 0.19919,
  grouped standard error 0.03296.
- Generic selected kernel at temperature 0.1, grouped log-MAE 0.20638,
  grouped standard error 0.03407.
- Model-specific uniform/direct validation log-MAE was 0.26309 for low-rank,
  0.28386 for generic, and 0.33489 for absolute response.
- Selected final epochs were 1 independently for low-rank, generic, and
  absolute-response models.
- OOD calibration used 2,327 query-to-uncensored-support distances in one
  inhibitor-only base-feature coordinate system.
- `run_report.json` records `holdout_opened_after_final_training: false`,
  `pair_rows: null`, and `directional_outcomes: null`.

Deviations from defaults:

- Added `--skip-holdout`.
- CV seeds: two (`0, 1`) rather than five.
- Final seeds: two (`0, 1`) rather than five.
- Maximum epochs: 12 rather than 80.
- Patience: 4 rather than 10.
- Batches per epoch: 2 rather than 4.
- The default four blocks, model candidate, feature set, optimizer settings,
  aggregation grids, clustering algorithm/seed, batch size, and automatic
  device selection were retained. Bootstrap settings were not used because no
  pair evaluation occurred.

## Unresolved By Design

The historical real App-App dataset was already exposed by prior smoke runs.
That contamination and the previous negative post-holdout result cannot be
repaired in code. This repair did not open or evaluate that dataset and did not
use prior App-App performance to choose changes. No claim of improved App-App
performance is made. A newly collected untouched App-App dataset is required.

The clipped CSVs also do not identify latent speedup magnitude. The corrected
censored formulation uses the known one-sided boundary information and enforces
the operational support, but it cannot recover information that was discarded
before storage.
