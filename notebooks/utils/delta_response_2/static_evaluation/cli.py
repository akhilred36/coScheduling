"""Non-interactive command-line interface for static evaluation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import (
    COMPLETION_METADATA_FILES,
    DEFAULT_OUTPUT_ROOT,
    EVIDENCE_LABEL,
    completed_roster,
    prepared_roster,
)
from .evaluation import evaluate, verify_complete
from .integrity import read_canonical_json, regular_file_roster
from .preparation import prepare, require_fixed_root, verify_preparation


def _fixed_root(path: Path) -> Path:
    return require_fixed_root(path)


def command_prepare(args: argparse.Namespace) -> dict[str, object]:
    return prepare(_fixed_root(args.root), device=args.device)


def command_evaluate(args: argparse.Namespace) -> dict[str, object]:
    return evaluate(_fixed_root(args.root), device=args.device)


def command_status(args: argparse.Namespace) -> dict[str, Any]:
    root = _fixed_root(args.root)
    completion = root / "completion_seal.json"
    if completion.is_symlink():
        raise RuntimeError("completion seal may not be a symlink")
    if completion.is_file():
        report = verify_complete(root)
        selected = report["selected_non_self_metrics"]
        return {
            "status": "complete",
            "root": str(root),
            "evidence_label": EVIDENCE_LABEL,
            "selected_candidate_id": report["selected_candidate_id"],
            "selected_non_self_log_mae": selected["log_mae"],
            "selected_non_self_raw_mae": selected["raw_mae"],
            "selected_signed_log_bias": selected["signed_log_bias"],
            "selected_spearman": selected["spearman"],
            "selected_calibration_slope": selected["calibration_slope"],
            "hypotheses": {
                name: result["status"]
                for name, result in report["hypotheses"].items()
            },
            "verification": "all_artifacts_verified",
        }
    run_state_path = root / "run_report.json"
    if run_state_path.is_symlink():
        raise RuntimeError("run report may not be a symlink")
    if run_state_path.is_file():
        verify_preparation(root, require_prepared_roster=False)
        actual = regular_file_roster(root)
        allowed = completed_roster() - set(COMPLETION_METADATA_FILES)
        if not prepared_roster().issubset(actual) or not actual.issubset(allowed):
            raise RuntimeError("incomplete evaluation has an invalid artifact roster")
        state = read_canonical_json(run_state_path)
        return {
            "status": "evaluation_incomplete",
            "root": str(root),
            "evidence_label": EVIDENCE_LABEL,
            "recorded_stage": state.get("status"),
            "app_app_outcome_access_started": True,
            "pair_access_status": "may_have_been_opened",
            "verification": "preparation_verified_but_completion_seal_absent",
        }
    prepared = verify_preparation(root, require_prepared_roster=True)
    return {
        "status": "prepared",
        "root": str(root),
        "evidence_label": EVIDENCE_LABEL,
        "app_app_outcome_opened": False,
        "selected_candidate_id": prepared["recipe"]["selected"]["candidate_id"],
        "diagnostic_artifact_count": len(prepared["artifact_hashes"]),
        "verification": "preparation_verified_without_pair_access",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Delta Response 2 retrospective static evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function, help_text in (
        ("prepare", command_prepare, "fit diagnostics and seal without pair access"),
        ("evaluate", command_evaluate, "evaluate only after preparation verifies"),
        ("status", command_status, "verify prepared or complete state"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        if name in {"prepare", "evaluate"}:
            command.add_argument("--device", choices=["cpu"], default="cpu")
        command.set_defaults(function=function)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
