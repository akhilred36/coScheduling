"""Prespecified design comparisons, hypotheses, and static report rendering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import EVIDENCE_LABEL, EVIDENCE_STATUS, FUTURE_EVIDENCE_REQUIREMENT
from .bootstrap import ERROR_SIGN_CONVENTION, SIGN_CONVENTION


def _method_row(metrics: pd.DataFrame, method: str) -> pd.Series:
    selected = metrics[metrics["method"].eq(method)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one primary metric row for {method}")
    return selected.iloc[0]


def _paired_row(
    paired: pd.DataFrame, first: str, second: str, metric: str
) -> pd.Series:
    selected = paired[
        paired["first_method"].eq(first)
        & paired["second_method"].eq(second)
        & paired["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"missing paired contrast {first} - {second} for {metric}")
    return selected.iloc[0]


def design_diagnostic_table(
    metrics: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    references = {
        "delta2_selected": "delta1_delta_ood_kernel",
        "delta2_constant_1": "delta2_selected",
        "delta2_victim_inhibitor_median": "delta2_selected",
        "delta2_nearest_anchor": "delta2_selected",
        "delta2_profile_only_ood_h32_e16": "delta2_selected",
        "delta2_censored_zero_h32_e16": "delta2_profile_only_ood_h32_e16",
        "delta2_query_shrinkage_h32_e16": "delta2_profile_only_ood_h32_e16",
        "delta2_absolute_neural_h32_e16": "delta2_selected",
        "delta2_absolute_linear_r0p01": "delta2_selected",
    }
    roles = {
        "delta2_selected": "primary_frozen_recipe",
        "delta2_constant_1": "pair_independent_baseline",
        "delta2_victim_inhibitor_median": "pair_independent_baseline",
        "delta2_nearest_anchor": "pair_independent_baseline",
        "delta2_profile_only_ood_h32_e16": "fixed_profile_winner_diagnostic",
        "delta2_censored_zero_h32_e16": "all_anchor_censoring_diagnostic",
        "delta2_query_shrinkage_h32_e16": "fixed_query_shrinkage_diagnostic",
        "delta2_absolute_neural_h32_e16": "absolute_model_diagnostic",
        "delta2_absolute_linear_r0p01": "absolute_model_diagnostic",
    }
    rows: list[dict[str, object]] = []
    for method, reference in references.items():
        metric_row = _method_row(metrics, method)
        log_difference = _paired_row(paired, method, reference, "log_mae")
        bias_difference = _paired_row(paired, method, reference, "signed_log_bias")
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "evaluation_method": "random_split",
                "scope": "non_self_ensemble",
                "method": method,
                "candidate_id": metric_row["candidate_id"],
                "role": roles[method],
                "reference_method": reference,
                "log_mae": metric_row["log_mae"],
                "raw_mae": metric_row["raw_mae"],
                "signed_log_bias": metric_row["signed_log_bias"],
                "spearman": metric_row["spearman"],
                "spearman_defined": metric_row["spearman_defined"],
                "calibration_intercept": metric_row["calibration_intercept"],
                "calibration_slope": metric_row["calibration_slope"],
                "calibration_defined": metric_row["calibration_defined"],
                "overprediction_count": metric_row["overprediction_count"],
                "underprediction_count": metric_row["underprediction_count"],
                "log_mae_difference": log_difference["estimate_difference"],
                "log_mae_difference_ci_lower": log_difference["ci_lower"],
                "log_mae_difference_ci_upper": log_difference["ci_upper"],
                "signed_bias_difference": bias_difference["estimate_difference"],
                "signed_bias_difference_ci_lower": bias_difference["ci_lower"],
                "signed_bias_difference_ci_upper": bias_difference["ci_upper"],
                "difference_definition": SIGN_CONVENTION,
                "error_metric_sign_convention": ERROR_SIGN_CONVENTION,
            }
        )
    return pd.DataFrame(rows)


def hypothesis_results(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    selected = _method_row(metrics, "delta2_selected")
    profile = _method_row(metrics, "delta2_profile_only_ood_h32_e16")
    censored = _method_row(metrics, "delta2_censored_zero_h32_e16")

    h1_pair = _paired_row(
        paired,
        "delta2_censored_zero_h32_e16",
        "delta2_profile_only_ood_h32_e16",
        "signed_log_bias",
    )
    h1_moved_toward_zero = abs(float(censored["signed_log_bias"])) < abs(
        float(profile["signed_log_bias"])
    )
    h1_fewer_over = int(censored["overprediction_count"]) < int(
        profile["overprediction_count"]
    )
    if (
        float(profile["signed_log_bias"]) > 0
        and h1_moved_toward_zero
        and h1_fewer_over
        and float(h1_pair["ci_upper"]) < 0
    ):
        h1_status = "supportive"
    elif abs(float(censored["signed_log_bias"])) > abs(
        float(profile["signed_log_bias"])
    ):
        h1_status = "contrary"
    else:
        h1_status = "weakened"

    h2_pair = _paired_row(
        paired,
        "delta2_selected",
        "delta2_profile_only_ood_h32_e16",
        "log_mae",
    )
    if float(h2_pair["ci_upper"]) < 0:
        h2_status = "supportive"
    elif float(h2_pair["ci_lower"]) > 0:
        h2_status = "rejected"
    else:
        h2_status = "weakened"

    stability_log = stability[stability["metric"].eq("log_mae")].set_index("method")
    selected_stability = stability_log.loc["delta2_selected"]
    profile_stability = stability_log.loc["delta2_profile_only_ood_h32_e16"]
    sd_no_larger = float(selected_stability["seed_standard_deviation"]) <= float(
        profile_stability["seed_standard_deviation"]
    )
    worst_no_larger = float(selected_stability["seed_worst"]) <= float(
        profile_stability["seed_worst"]
    )
    if sd_no_larger and worst_no_larger:
        h3_status = "supportive"
    elif not sd_no_larger and not worst_no_larger:
        h3_status = "rejected"
    else:
        h3_status = "weakened"

    h4_interval = _bootstrap_row(bootstrap, "delta2_selected", "signed_log_bias")
    if (
        float(selected["signed_log_bias"]) > 0
        and float(h4_interval["ci_lower"]) > 0
        and int(selected["overprediction_count"]) > int(selected["underprediction_count"])
    ):
        h4_status = "supportive"
    elif float(selected["signed_log_bias"]) <= 0:
        h4_status = "rejected"
    else:
        h4_status = "weakened"

    h5_methods = [
        "delta2_constant_1",
        "delta2_absolute_neural_h32_e16",
        "delta2_absolute_linear_r0p01",
        "delta1_absolute_response",
    ]
    h5_rows = [
        _paired_row(paired, method, "delta2_selected", "log_mae")
        for method in h5_methods
    ]
    h5_support = [
        row
        for row in h5_rows
        if float(row["estimate_difference"]) < 0 and float(row["ci_upper"]) < 0
    ]
    h5_overlap = [
        row
        for row in h5_rows
        if float(row["estimate_difference"]) < 0
        and float(row["ci_lower"]) <= 0 <= float(row["ci_upper"])
    ]
    h5_status = "supportive" if h5_support else ("weakened" if h5_overlap else "rejected")

    return {
        "H1": {
            "status": h1_status,
            "legacy_signed_bias": float(profile["signed_log_bias"]),
            "censored_signed_bias": float(censored["signed_log_bias"]),
            "legacy_overprediction_count": int(profile["overprediction_count"]),
            "censored_overprediction_count": int(censored["overprediction_count"]),
            "paired_signed_bias_difference": float(h1_pair["estimate_difference"]),
            "paired_interval": [float(h1_pair["ci_lower"]), float(h1_pair["ci_upper"])],
            "interpretation": "all-anchor censoring contrast on shared potential checkpoints",
        },
        "H2": {
            "status": h2_status,
            "selected_minus_profile_log_mae": float(h2_pair["estimate_difference"]),
            "paired_interval": [float(h2_pair["ci_lower"]), float(h2_pair["ci_upper"])],
            "interpretation": "selector contrast; it does not isolate a selection component",
        },
        "H3": {
            "status": h3_status,
            "selected_seed_log_mae_sd": float(
                selected_stability["seed_standard_deviation"]
            ),
            "profile_seed_log_mae_sd": float(
                profile_stability["seed_standard_deviation"]
            ),
            "selected_seed_log_mae_worst": float(selected_stability["seed_worst"]),
            "profile_seed_log_mae_worst": float(profile_stability["seed_worst"]),
            "interpretation": "descriptive capacity stability; no training-regimen effect is identified",
        },
        "H4": {
            "status": h4_status,
            "selected_signed_bias": float(selected["signed_log_bias"]),
            "signed_bias_interval": [
                float(h4_interval["ci_lower"]),
                float(h4_interval["ci_upper"]),
            ],
            "overprediction_count": int(selected["overprediction_count"]),
            "underprediction_count": int(selected["underprediction_count"]),
            "interpretation": "conditional scale-mismatch observation, not proof of its cause",
        },
        "H5": {
            "status": h5_status,
            "comparisons": {
                method: {
                    "log_mae_difference": float(row["estimate_difference"]),
                    "paired_interval": [float(row["ci_lower"]), float(row["ci_upper"])],
                }
                for method, row in zip(h5_methods, h5_rows)
            },
            "interpretation": "consistent with conditional domain shift only; no causal mechanism is identified",
        },
    }


def _bootstrap_row(
    bootstrap: pd.DataFrame,
    method: str,
    metric: str,
) -> dict[str, float]:
    selected = bootstrap[
        bootstrap["method"].eq(method) & bootstrap["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"missing bootstrap interval for {method}/{metric}")
    row = selected.iloc[0]
    return {
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
    }


def _f(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def render_report(
    metrics: pd.DataFrame,
    per_victim: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    hypotheses: dict[str, dict[str, Any]],
) -> str:
    selected = _method_row(metrics, "delta2_selected")
    d1_ood = _method_row(metrics, "delta1_delta_ood_kernel")
    constant = _method_row(metrics, "delta2_constant_1")
    d1_absolute = _method_row(metrics, "delta1_absolute_response")
    profile = _method_row(metrics, "delta2_profile_only_ood_h32_e16")
    censored = _method_row(metrics, "delta2_censored_zero_h32_e16")
    shrunk = _method_row(metrics, "delta2_query_shrinkage_h32_e16")
    absolute_neural = _method_row(metrics, "delta2_absolute_neural_h32_e16")
    absolute_linear = _method_row(metrics, "delta2_absolute_linear_r0p01")

    selected_d1 = _paired_row(
        paired, "delta2_selected", "delta1_delta_ood_kernel", "log_mae"
    )
    selected_constant = _paired_row(
        paired, "delta2_selected", "delta2_constant_1", "log_mae"
    )
    selected_absolute = _paired_row(
        paired, "delta2_selected", "delta1_absolute_response", "log_mae"
    )
    absolute_neural_selected = _paired_row(
        paired,
        "delta2_absolute_neural_h32_e16",
        "delta2_selected",
        "log_mae",
    )
    absolute_linear_selected = _paired_row(
        paired,
        "delta2_absolute_linear_r0p01",
        "delta2_selected",
        "log_mae",
    )
    censored_legacy_bias = _paired_row(
        paired,
        "delta2_censored_zero_h32_e16",
        "delta2_profile_only_ood_h32_e16",
        "signed_log_bias",
    )
    selected_victims = per_victim[per_victim["method"].eq("delta2_selected")]
    selected_seed = stability[
        stability["method"].eq("delta2_selected")
        & stability["metric"].eq("log_mae")
    ].iloc[0]
    selected_bias_interval = bootstrap[
        bootstrap["method"].eq("delta2_selected")
        & bootstrap["metric"].eq("signed_log_bias")
    ].iloc[0]

    h1 = hypotheses["H1"]
    h2 = hypotheses["H2"]
    h3 = hypotheses["H3"]
    h4 = hypotheses["H4"]
    h5 = hypotheses["H5"]
    h5_text = "; ".join(
        f"`{method}` minus selected {_f(values['log_mae_difference'])} "
        f"[{_f(values['paired_interval'][0])}, {_f(values['paired_interval'][1])}]"
        for method, values in h5["comparisons"].items()
    )
    hypothesis_lines = [
        f"- **H1: {h1['status']}.** Legacy versus censored signed bias is {_f(h1['legacy_signed_bias'])} versus {_f(h1['censored_signed_bias'])}; censored-minus-legacy is {_f(h1['paired_signed_bias_difference'])} [{_f(h1['paired_interval'][0])}, {_f(h1['paired_interval'][1])}], with overprediction counts {h1['legacy_overprediction_count']} versus {h1['censored_overprediction_count']}. {h1['interpretation']}.",
        f"- **H2: {h2['status']}.** Selected-minus-profile-winner log MAE is {_f(h2['selected_minus_profile_log_mae'])} [{_f(h2['paired_interval'][0])}, {_f(h2['paired_interval'][1])}]. {h2['interpretation']}.",
        f"- **H3: {h3['status']}.** Selected versus `32/16` seed log-MAE SD is {_f(h3['selected_seed_log_mae_sd'])} versus {_f(h3['profile_seed_log_mae_sd'])}; worst seed is {_f(h3['selected_seed_log_mae_worst'])} versus {_f(h3['profile_seed_log_mae_worst'])}. {h3['interpretation']}.",
        f"- **H4: {h4['status']}.** Selected signed bias is {_f(h4['selected_signed_bias'])} [{_f(h4['signed_bias_interval'][0])}, {_f(h4['signed_bias_interval'][1])}], with {h4['overprediction_count']} overpredictions and {h4['underprediction_count']} underpredictions. {h4['interpretation']}.",
        f"- **H5: {h5['status']}.** {h5_text}. {h5['interpretation']}.",
    ]

    return "\n".join(
        [
            "# Delta Response 2 Static Random-Split Report",
            "",
            "## Evidence Status",
            "",
            f"This is a **{EVIDENCE_LABEL}** on a {EVIDENCE_STATUS}. It is not a fresh transfer claim.",
            FUTURE_EVIDENCE_REQUIREMENT,
            "No static outcome was used to fit a model, set a prior, choose a checkpoint, calibrate OOD behavior, or select a method.",
            "",
            "## Primary Result",
            "",
            "The primary estimand contains 90 non-self directions from 45 original unordered pair clusters. Seed predictions were averaged in latent log space and clipped once after averaging.",
            "",
            "| Method | Log MAE | Raw MAE | Signed log bias | Spearman | Calibration slope | Over / under |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| `delta2_selected` | {_f(selected['log_mae'])} | {_f(selected['raw_mae'])} | {_f(selected['signed_log_bias'])} | {_f(selected['spearman'])} | {_f(selected['calibration_slope'])} | {int(selected['overprediction_count'])} / {int(selected['underprediction_count'])} |",
            f"| `delta1_delta_ood_kernel` | {_f(d1_ood['log_mae'])} | {_f(d1_ood['raw_mae'])} | {_f(d1_ood['signed_log_bias'])} | {_f(d1_ood['spearman'])} | {_f(d1_ood['calibration_slope'])} | {int(d1_ood['overprediction_count'])} / {int(d1_ood['underprediction_count'])} |",
            f"| `delta2_constant_1` | {_f(constant['log_mae'])} | {_f(constant['raw_mae'])} | {_f(constant['signed_log_bias'])} | {_f(constant['spearman'])} | {_f(constant['calibration_slope'])} | {int(constant['overprediction_count'])} / {int(constant['underprediction_count'])} |",
            f"| `delta1_absolute_response` | {_f(d1_absolute['log_mae'])} | {_f(d1_absolute['raw_mae'])} | {_f(d1_absolute['signed_log_bias'])} | {_f(d1_absolute['spearman'])} | {_f(d1_absolute['calibration_slope'])} | {int(d1_absolute['overprediction_count'])} / {int(d1_absolute['underprediction_count'])} |",
            "",
            "## Prespecified Questions",
            "",
            f"1. **Frozen Delta Response 2 versus frozen Delta Response 1 OOD delta.** The log-MAE difference (`delta2_selected - delta1_delta_ood_kernel`) is {_f(selected_d1['estimate_difference'])}, with paired 45-cluster interval [{_f(selected_d1['ci_lower'])}, {_f(selected_d1['ci_upper'])}]. Negative values favor Delta Response 2.",
            f"2. **No-degradation and absolute magnitude baselines.** The selected-minus-constant log-MAE difference is {_f(selected_constant['estimate_difference'])} [{_f(selected_constant['ci_lower'])}, {_f(selected_constant['ci_upper'])}], and selected-minus-inherited-absolute is {_f(selected_absolute['estimate_difference'])} [{_f(selected_absolute['ci_lower'])}, {_f(selected_absolute['ci_upper'])}]. Their raw MAEs are {_f(constant['raw_mae'])} and {_f(d1_absolute['raw_mae'])}, versus {_f(selected['raw_mae'])} for the selected model.",
            f"3. **Ordering.** The selected-model Spearman estimate is {_f(selected['spearman'])} (defined={bool(selected['spearman_defined'])}); Pearson is {_f(selected['pearson'])}. This rank evidence is reported separately from magnitude accuracy.",
            f"4. **Scale diagnostics.** Signed bias is {_f(selected['signed_log_bias'])}, with interval [{_f(selected_bias_interval['ci_lower'])}, {_f(selected_bias_interval['ci_upper'])}]. Calibration is `y = {_f(selected['calibration_intercept'])} + {_f(selected['calibration_slope'])} p`; mean predicted log slowdown is {_f(selected['mean_predicted_log_slowdown'])} versus mean truth {_f(selected['mean_true_log_slowdown'])}, with {int(selected['overprediction_count'])} overpredictions.",
            f"5. **Seed and victim stability.** Five-seed non-self log MAE has population SD {_f(selected_seed['seed_standard_deviation'])} and worst seed {_f(selected_seed['seed_worst'])}. Across ten victims, selected-model log MAE ranges from {_f(selected_victims['log_mae'].min())} to {_f(selected_victims['log_mae'].max())}.",
            f"6. **All-anchor censoring versus legacy correction.** On shared `32/16` checkpoints, legacy bias is {_f(profile['signed_log_bias'])} and censored-zero bias is {_f(censored['signed_log_bias'])}; overprediction counts are {int(profile['overprediction_count'])} and {int(censored['overprediction_count'])}. The censored-minus-legacy signed-bias difference is {_f(censored_legacy_bias['estimate_difference'])} [{_f(censored_legacy_bias['ci_lower'])}, {_f(censored_legacy_bias['ci_upper'])}].",
            f"7. **Fixed query shrinkage.** Profile-only and query-shrinkage log MAEs are {_f(profile['log_mae'])} and {_f(shrunk['log_mae'])}; Spearman estimates are {_f(profile['spearman'])} and {_f(shrunk['spearman'])}. The observed scale and rank movement is descriptive. This compound contrast changes censoring, population-prior use, and gamma together, so it cannot isolate a gamma effect and is not a static-data selection.",
            f"8. **Absolute versus delta response.** Absolute-neural and censored-linear log MAEs are {_f(absolute_neural['log_mae'])} and {_f(absolute_linear['log_mae'])}, compared with {_f(selected['log_mae'])} for the selected delta model and {_f(d1_absolute['log_mae'])} for the inherited absolute response. Absolute-neural minus selected is {_f(absolute_neural_selected['estimate_difference'])} [{_f(absolute_neural_selected['ci_lower'])}, {_f(absolute_neural_selected['ci_upper'])}], and censored-linear minus selected is {_f(absolute_linear_selected['estimate_difference'])} [{_f(absolute_linear_selected['ci_lower'])}, {_f(absolute_linear_selected['ci_upper'])}]. Negative values favor the absolute model.",
            "9. **Paired uncertainty.** Every interval uses the same 1,000 NumPy draws of 45 unordered pair clusters with seed 923. Both directions are repeated whenever a cluster is drawn. Intervals crossing zero are treated as uncertain rather than directional evidence.",
            "10. **Design interpretation.** The hypothesis decisions below follow the prespecified rules. Observed contrasts are separated from causal explanations; domain-shift mechanisms remain hypotheses.",
            "",
            "## H1-H5 Decisions",
            "",
            *hypothesis_lines,
            "",
            "## Scope And Limitations",
            "",
            "Self-pair results are secondary diagnostics and are not pooled into the primary 90-direction result. Macro-victim estimates weight each of the ten victims equally. Undefined rank or calibration estimates are blank in CSV tables and retain explicit defined flags.",
            "The archived Delta Response 1 predictions were hash-verified and read only after the new Delta Response 2 prediction artifacts were written and hashed. Archived rows remain in a separate namespaced file.",
            FUTURE_EVIDENCE_REQUIREMENT,
            "",
        ]
    )
