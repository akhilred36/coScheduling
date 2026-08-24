"""Fixed protocol values for the Delta Response 2 static evaluation."""

from __future__ import annotations

from pathlib import Path


STATIC_EVALUATION_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = STATIC_EVALUATION_DIR.parent
UTILS_DIR = PACKAGE_DIR.parent

FROZEN_ROOT = PACKAGE_DIR / "experiments" / "compact_v1"
DATA_DIR = UTILS_DIR / "few_shot" / "data"
JOBS_PATH = DATA_DIR / "jobs.csv"
INHIBITORS_PATH = DATA_DIR / "inhibitors.csv"
JOB_INH_PATH = DATA_DIR / "job_inh.csv"
# This path is deliberately constructed lexically. Preparation must never call
# resolve, stat, exists, open, or hash on it.
PAIR_PATH = DATA_DIR / "pair.csv"

ARCHIVE_ROOT = (
    UTILS_DIR
    / "delta_response_1"
    / "experiments"
    / "random_split_model_search_v1"
    / "final_random_split"
)
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "experiments" / "static_random_split_evaluation_v1"

EVIDENCE_LABEL = "retrospective static random_split evaluation"
EVIDENCE_STATUS = (
    "historical diagnostic dataset; development-only transfer evidence"
)
FUTURE_EVIDENCE_REQUIREMENT = (
    "A newly collected immutable App-App holdout remains necessary for a fresh "
    "transfer claim."
)

EXPECTED_INPUT_HASHES = {
    "jobs.csv": "356a3136a2d32527e75eada31841807d0ccc13dfe48ad0540664666938c0179c",
    "inhibitors.csv": "a8db332145b560f51c2176bc8378a423cb1a318bb9a7bfcc31788b1addf12573",
    "job_inh.csv": "cb721f466161e816e3a2711b9e6d807b4c7b72113af506a0eca4d483513d2118",
    "pair.csv": "369381f33b9689bc7fbc2a5b3d8d285fc4a604c5480968391c4c84a9b24b20c0",
}
INPUT_PATHS = {
    "jobs.csv": JOBS_PATH,
    "inhibitors.csv": INHIBITORS_PATH,
    "job_inh.csv": JOB_INH_PATH,
    "pair.csv": PAIR_PATH,
}

EXPECTED_ARCHIVE_HASHES = {
    "directional_predictions.csv": "85f0cecfc55e3dee634ecf9ad9598b957f45378e5d4e0865f84c260289fb2725",
    "evaluation_pair_manifest.csv": "0eb6822506fdd20f1c5cfaa110fede545498fc81200568dae1949f685d03c8f1",
    "requested_run_config.json": "a191f6c85f66b1c39d28f8df22417ae2e5738412beb42f7f43e85e56e5e9d736",
    "selection.json": "e3276b0ff0c8fd41baa1ba0f02a927ccbd53fc2ce40e9b1eeeba6a3567272cb2",
    "run_report.json": "21371fe7f8a684747b7ea56f4ab7a305430b7a52ac0f5a866c05536a23522a2a",
}

SELECTED_CANDIDATE_ID = "current_delta_ood_h8_e4__de045c14efb9"
FROZEN_RECIPE_SHA256 = "3b95393dba19ddb851bf974e793b1669d769f6cd6b11388ba7818d486e9d8628"
FROZEN_PLAN_SEAL_SHA256 = "1cd46f81278203e1c565c063c8bc95eee69784067935a40433c4ae3095aea340"
FROZEN_COMPLETION_DIGEST = "114c3003053cd89ea7e656dc09ca84202cf09d715e708af8a0fabf72a82e116f"

SEEDS = (0, 1, 2, 3, 4)
VALID_APP_INHIBITOR_ROWS = 2623
STATIC_PAIR_ROWS = 55
SELF_PAIR_ROWS = 10
NON_SELF_CLUSTERS = 45
DIRECTIONAL_ROWS = 110
PRIMARY_DIRECTIONAL_ROWS = 90
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 923
BOOTSTRAP_QUANTILES = (0.025, 0.975)

TRAINING_SETTINGS = {
    "feature_set": "base",
    "hidden_dim": 32,
    "embedding_dim": 16,
    "rank": 2,
    "optimizer_steps": 80,
    "batch_size": 128,
    "checkpoint_interval": 20,
    "learning_rate": 0.003,
    "weight_decay": 0.001,
    "linear_ridge": 0.01,
    "legacy_ood_temperature": 4.0,
    "legacy_ood_quantile": 0.90,
    "legacy_ood_fallback": "uniform",
    "censored_shrinkage": 10.0,
    "query_gamma": 0.75,
    "device": "cpu",
}

METHOD_CANDIDATES = {
    "delta2_selected": SELECTED_CANDIDATE_ID,
    "delta2_constant_1": "not_applicable",
    "delta2_victim_inhibitor_median": "not_applicable",
    "delta2_nearest_anchor": "not_applicable",
    "delta2_profile_only_ood_h32_e16": "current_delta_ood_h32_e16__726110c4a776",
    "delta2_censored_zero_h32_e16": "censored_correction_delta_h32_e16_zero_l10p0__e12c04b3e734",
    "delta2_query_shrinkage_h32_e16": "censored_correction_delta_with_query_shrinkage_h32_e16_l10p0_g0p75__2571628c7df2",
    "delta2_absolute_neural_h32_e16": "regularized_absolute_neural_h32_e16__a452e916ac46",
    "delta2_absolute_linear_r0p01": "regularized_absolute_linear_r0p01__a7f8020fd65b",
}
DELTA2_METHODS = tuple(METHOD_CANDIDATES)
SEEDED_METHODS = (
    "delta2_selected",
    "delta2_profile_only_ood_h32_e16",
    "delta2_censored_zero_h32_e16",
    "delta2_query_shrinkage_h32_e16",
    "delta2_absolute_neural_h32_e16",
)
ANALYTIC_METHODS = (
    "delta2_constant_1",
    "delta2_victim_inhibitor_median",
    "delta2_nearest_anchor",
)
DETERMINISTIC_METHODS = (*ANALYTIC_METHODS, "delta2_absolute_linear_r0p01")
SHARED_POTENTIAL_METHODS = (
    "delta2_profile_only_ood_h32_e16",
    "delta2_censored_zero_h32_e16",
    "delta2_query_shrinkage_h32_e16",
)

ARCHIVE_METHODS = (
    "constant_1",
    "victim_inhibitor_median",
    "nearest_anchor",
    "absolute_response",
    "delta_single_anchor",
    "delta_uniform",
    "delta_median",
    "delta_kernel",
    "delta_ood_kernel",
    "generic_potential",
)
NAMESPACED_ARCHIVE_METHODS = tuple(f"delta1_{value}" for value in ARCHIVE_METHODS)

PAIR_REQUIRED_COLUMNS = ("jobA_id", "jobB_id", "slowdown_A", "slowdown_B")
PAIR_MANIFEST_COLUMNS = (
    "evaluation_method",
    "pair_row_id",
    "pair_cluster_id",
    "jobA_id",
    "jobB_id",
    "self_pair",
    "jobA_known",
    "jobB_known",
    "known_endpoint_count",
    "included",
    "evidence_label",
)

SEED_PREDICTION_COLUMNS = (
    "evaluation_method",
    "pair_row_id",
    "pair_cluster_id",
    "direction",
    "victim_id",
    "aggressor_id",
    "self_pair",
    "method",
    "candidate_id",
    "seed",
    "latent_predicted_log_slowdown",
    "predicted_log_slowdown",
    "predicted_slowdown",
    "true_log_slowdown",
    "true_slowdown",
)
DIRECTIONAL_PREDICTION_COLUMNS = (
    "evaluation_method",
    "pair_row_id",
    "pair_cluster_id",
    "direction",
    "victim_id",
    "aggressor_id",
    "self_pair",
    "method",
    "candidate_id",
    "latent_predicted_log_slowdown",
    "predicted_log_slowdown",
    "predicted_slowdown",
    "true_log_slowdown",
    "true_slowdown",
    "ensemble_seed_count",
    "model_seed_log_std",
)
CORRECTION_DIAGNOSTIC_COLUMNS = (
    "anchor_count",
    "exact_anchor_count",
    "floor_anchor_count",
    "nearest_anchor_distance",
    "nearest_exact_anchor_distance",
    "ood_alpha",
    "correction",
    "correction_prior",
    "correction_active_floor_violations",
)

ERROR_METRICS = (
    "log_mae",
    "log_mse",
    "log_rmse",
    "raw_mae",
    "raw_mape",
    "raw_mse",
    "raw_rmse",
    "median_absolute_log_error",
    "median_multiplicative_error",
)
BOOTSTRAP_METRICS = (
    *ERROR_METRICS,
    "signed_log_bias",
    "calibration_intercept",
    "calibration_slope",
    "spearman",
    "pearson",
    "mean_true_log_slowdown",
    "mean_predicted_log_slowdown",
)
METRIC_ESTIMATES = (*BOOTSTRAP_METRICS, "overprediction_count", "underprediction_count")
METRIC_FLAGS = ("calibration_defined", "spearman_defined", "pearson_defined")

DIAGNOSTIC_ARTIFACT_FILES = (
    "profile_scaler.json",
    "distance_profile_scaler.json",
    "delta_h32_e16_seed0.pt",
    "delta_h32_e16_seed1.pt",
    "delta_h32_e16_seed2.pt",
    "delta_h32_e16_seed3.pt",
    "delta_h32_e16_seed4.pt",
    "delta_h32_e16_training_curves.csv",
    "delta_h32_e16_ood_reference.csv",
    "absolute_neural_h32_e16_seed0.pt",
    "absolute_neural_h32_e16_seed1.pt",
    "absolute_neural_h32_e16_seed2.pt",
    "absolute_neural_h32_e16_seed3.pt",
    "absolute_neural_h32_e16_seed4.pt",
    "absolute_neural_h32_e16_training_curves.csv",
    "absolute_linear_r0p01.json",
)
DIAGNOSTIC_METADATA_FILES = ("artifact_hashes.json", "preparation_state.json")
DIAGNOSTIC_FILES = (*DIAGNOSTIC_ARTIFACT_FILES, *DIAGNOSTIC_METADATA_FILES)

PREPARATION_TOP_LEVEL_FILES = (
    "evaluation_plan.json",
    "evaluation_preparation_seal.json",
    "input_hashes.json",
)
EVALUATION_PRODUCT_FILES = (
    "evaluation_pair_manifest.csv",
    "seed_predictions.csv",
    "directional_predictions.csv",
    "metrics_non_self.csv",
    "metrics_non_self_per_victim.csv",
    "metrics_non_self_macro_victim.csv",
    "metrics_all.csv",
    "metrics_self.csv",
    "metrics_self_averaged.csv",
    "metrics_by_seed.csv",
    "seed_stability.csv",
    "cluster_bootstrap_non_self.csv",
    "paired_method_differences_non_self.csv",
    "archived_delta1_comparison.csv",
    "design_diagnostic_comparison.csv",
    "run_report.json",
    "STATIC_RANDOM_SPLIT_REPORT.md",
)
COMPLETION_METADATA_FILES = ("output_hashes.json", "completion_seal.json")


def prepared_roster() -> set[str]:
    return set(PREPARATION_TOP_LEVEL_FILES) | {
        f"diagnostic_checkpoints/{name}" for name in DIAGNOSTIC_FILES
    }


def completed_roster() -> set[str]:
    return prepared_roster() | set(EVALUATION_PRODUCT_FILES) | set(
        COMPLETION_METADATA_FILES
    )
