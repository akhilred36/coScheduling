"""Post-freeze loading and namespace isolation for archived DR1 outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ARCHIVE_METHODS,
    ARCHIVE_ROOT,
    DIRECTIONAL_ROWS,
    EVIDENCE_LABEL,
    EXPECTED_ARCHIVE_HASHES,
    NAMESPACED_ARCHIVE_METHODS,
    STATIC_PAIR_ROWS,
)
from .integrity import require_regular_file, sha256_file


MERGE_KEYS = (
    "pair_row_id",
    "pair_cluster_id",
    "direction",
    "victim_id",
    "aggressor_id",
)


def verify_archive_hashes() -> dict[str, str]:
    """Hash every declared archive file before parsing any one of them."""
    if ARCHIVE_ROOT.is_symlink() or not ARCHIVE_ROOT.is_dir():
        raise RuntimeError("archived Delta Response 1 root is absent or a symlink")
    observed: dict[str, str] = {}
    for name in EXPECTED_ARCHIVE_HASHES:
        path = ARCHIVE_ROOT / name
        require_regular_file(path)
        observed[name] = sha256_file(path)
    mismatches = {
        name: (observed[name], expected)
        for name, expected in EXPECTED_ARCHIVE_HASHES.items()
        if observed[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"archived Delta Response 1 hash mismatch: {mismatches}")
    return observed


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()


def _validate_archived_manifest(new_manifest: pd.DataFrame) -> None:
    archived = _read_csv(ARCHIVE_ROOT / "evaluation_pair_manifest.csv")
    if "evaluation_method" in archived:
        archived = archived[archived["evaluation_method"].eq("random_split")]
    required = {"pair_row_id", "jobA_id", "jobB_id"}
    missing = sorted(required - set(archived.columns))
    if missing:
        raise ValueError(f"archived pair manifest is missing columns: {missing}")
    endpoints = archived[["pair_row_id", "jobA_id", "jobB_id"]].copy()
    endpoints["pair_row_id"] = pd.to_numeric(
        endpoints["pair_row_id"], errors="raise"
    ).astype(int)
    endpoints["jobA_id"] = endpoints["jobA_id"].astype(str)
    endpoints["jobB_id"] = endpoints["jobB_id"].astype(str)
    endpoints = endpoints.drop_duplicates().sort_values("pair_row_id").reset_index(drop=True)
    expected = (
        new_manifest[["pair_row_id", "jobA_id", "jobB_id"]]
        .sort_values("pair_row_id")
        .reset_index(drop=True)
    )
    if len(endpoints) != STATIC_PAIR_ROWS or not endpoints.equals(expected):
        raise RuntimeError("new pair manifest does not match archived row IDs and endpoints")


def load_archived_comparison(
    new_manifest: pd.DataFrame,
    canonical_directions: pd.DataFrame,
    verified_hashes: dict[str, str],
) -> pd.DataFrame:
    if verified_hashes != EXPECTED_ARCHIVE_HASHES:
        raise RuntimeError("archive parsing requires all five expected hashes")
    _validate_archived_manifest(new_manifest)
    archive = _read_csv(ARCHIVE_ROOT / "directional_predictions.csv")
    required = {
        "evaluation_method",
        "pair_row_id",
        "pair_cluster_id",
        "direction",
        "victim_id",
        "aggressor_id",
        "method",
        "anchor_budget",
        "predicted_log_slowdown",
        "true_slowdown",
    }
    missing = sorted(required - set(archive.columns))
    if missing:
        raise ValueError(f"archived directional predictions are missing columns: {missing}")
    archive = archive[
        archive["evaluation_method"].eq("random_split")
        & archive["anchor_budget"].astype(str).str.lower().eq("all")
        & archive["method"].astype(str).isin(ARCHIVE_METHODS)
    ].copy()
    archive["archive_method"] = archive["method"].astype(str)
    archive["method"] = "delta1_" + archive["archive_method"]
    if set(archive["method"]) != set(NAMESPACED_ARCHIVE_METHODS):
        raise RuntimeError("archived method namespace is incomplete")

    archive["pair_row_id"] = pd.to_numeric(
        archive["pair_row_id"], errors="raise"
    ).astype(int)
    archive["pair_cluster_id"] = pd.to_numeric(
        archive["pair_cluster_id"], errors="raise"
    ).astype(int)
    archive["direction"] = archive["direction"].astype(str)
    archive["victim_id"] = archive["victim_id"].astype(str)
    archive["aggressor_id"] = archive["aggressor_id"].astype(str)
    archive["self_pair"] = archive["victim_id"].eq(archive["aggressor_id"])
    if archive[list(MERGE_KEYS) + ["method"]].duplicated().any():
        raise RuntimeError("archived comparison has duplicate method directions")
    counts = archive["method"].value_counts()
    if set(counts.index) != set(NAMESPACED_ARCHIVE_METHODS) or not counts.eq(
        DIRECTIONAL_ROWS
    ).all():
        raise RuntimeError("each archived method must contain exactly 110 directions")

    archive = archive.rename(
        columns={
            "self_pair": "self_pair_archive",
            "true_slowdown": "true_slowdown_archive",
        }
    )
    canonical = canonical_directions[
        [*MERGE_KEYS, "self_pair", "true_log_slowdown", "true_slowdown"]
    ].rename(
        columns={
            "self_pair": "self_pair_canonical",
            "true_log_slowdown": "true_log_slowdown_canonical",
            "true_slowdown": "true_slowdown_canonical",
        }
    )
    merged = archive.merge(
        canonical,
        on=list(MERGE_KEYS),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("archive directions do not match the validated static pair table")
    if not merged["self_pair_archive"].eq(merged["self_pair_canonical"]).all():
        raise RuntimeError("archive self-pair flags differ from validated membership")

    archived_true = pd.to_numeric(merged["true_slowdown_archive"], errors="coerce").to_numpy(dtype=float)
    canonical_true = merged["true_slowdown_canonical"].to_numpy(dtype=float)
    if not np.isfinite(archived_true).all() or not np.allclose(
        archived_true, canonical_true, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("archive and validated pair outcomes differ")
    archived_log = np.log(archived_true)
    if not np.isfinite(archived_log).all() or not np.allclose(
        archived_log,
        merged["true_log_slowdown_canonical"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("archive true log outcomes differ from validated outcomes")

    predicted_log = pd.to_numeric(
        merged["predicted_log_slowdown"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(predicted_log).all() or (predicted_log < 0).any():
        raise RuntimeError("archive predicted log slowdowns are invalid")
    if "predicted_slowdown" in merged.columns:
        predicted_raw = pd.to_numeric(
            merged["predicted_slowdown"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(predicted_raw).all() or not np.allclose(
            predicted_raw, np.exp(predicted_log), rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError("archive raw and log predictions are inconsistent")
    else:
        predicted_raw = np.exp(predicted_log)

    if "latent_predicted_log_slowdown" in merged.columns:
        latent_log = pd.to_numeric(
            merged["latent_predicted_log_slowdown"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(latent_log).all():
            raise RuntimeError("archive latent predictions are non-finite")
    else:
        latent_log = predicted_log
    if "model_seed_log_std" in merged.columns:
        model_seed_log_std = pd.to_numeric(
            merged["model_seed_log_std"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(model_seed_log_std).all() or (model_seed_log_std < 0).any():
            raise RuntimeError("archive model-seed standard deviations are invalid")
    else:
        model_seed_log_std = np.zeros(len(merged), dtype=float)

    result = pd.DataFrame(
        {
            "evaluation_method": "random_split",
            "pair_row_id": merged["pair_row_id"].to_numpy(dtype=int),
            "pair_cluster_id": merged["pair_cluster_id"].to_numpy(dtype=int),
            "direction": merged["direction"].astype(str).to_numpy(),
            "victim_id": merged["victim_id"].astype(str).to_numpy(),
            "aggressor_id": merged["aggressor_id"].astype(str).to_numpy(),
            "self_pair": merged["self_pair_canonical"].to_numpy(dtype=bool),
            "archive_method": merged["archive_method"].astype(str).to_numpy(),
            "method": merged["method"].astype(str).to_numpy(),
            "candidate_id": "archived_delta_response_1",
            "latent_predicted_log_slowdown": latent_log,
            "predicted_log_slowdown": predicted_log,
            "predicted_slowdown": predicted_raw,
            "true_log_slowdown": merged["true_log_slowdown_canonical"].to_numpy(dtype=float),
            "true_slowdown": canonical_true,
            "ensemble_seed_count": 5,
            "model_seed_log_std": model_seed_log_std,
            "anchor_budget": "all",
            "evidence_label": EVIDENCE_LABEL,
        }
    )
    return result.sort_values(
        ["pair_row_id", "direction", "archive_method"], kind="stable"
    ).reset_index(drop=True)
