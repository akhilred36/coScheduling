"""Plan, execute, and inspect the Delta Response 2 random-split search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .frozen import predict_frozen_recipe
from .manifests import create_plan, verify_completion, verify_plan
from .pipeline import run_experiment


PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE = PACKAGE_DIR.parents[2]
DEFAULT_DATA_DIR = PACKAGE_DIR.parent / "few_shot" / "data"


def _config(path: Path | None) -> ExperimentConfig:
    config = ExperimentConfig() if path is None else ExperimentConfig.from_json(path)
    config.validate()
    return config


def command_plan(args: argparse.Namespace) -> None:
    create_plan(
        args.root.resolve(),
        config=_config(args.config),
        jobs_csv=args.jobs_csv.resolve(),
        inhibitors_csv=args.inhibitors_csv.resolve(),
        job_inh_csv=args.job_inh_csv.resolve(),
        package_dir=PACKAGE_DIR,
        workspace=WORKSPACE,
    )
    print(f"Sealed App-Inhibitor experiment plan: {args.root.resolve()}")


def command_run(args: argparse.Namespace) -> None:
    run_experiment(args.root.resolve(), PACKAGE_DIR)


def command_status(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    verify_plan(root, PACKAGE_DIR)
    report_path = root / "run_report.json"
    if report_path.exists():
        verify_completion(root)
        report = json.loads(report_path.read_text())
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": "planned", "root": str(root)}, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    prediction = predict_frozen_recipe(
        args.root.resolve(),
        args.victim_id,
        args.aggressor_id,
        device=args.device,
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create immutable folds and hashes")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--config", type=Path)
    plan.add_argument("--jobs-csv", type=Path, default=DEFAULT_DATA_DIR / "jobs.csv")
    plan.add_argument(
        "--inhibitors-csv", type=Path, default=DEFAULT_DATA_DIR / "inhibitors.csv"
    )
    plan.add_argument(
        "--job-inh-csv", type=Path, default=DEFAULT_DATA_DIR / "job_inh.csv"
    )
    plan.set_defaults(function=command_plan)

    run = subparsers.add_parser("run", help="run all candidates and freeze a recipe")
    run.add_argument("--root", type=Path, required=True)
    run.set_defaults(function=command_run)

    status = subparsers.add_parser("status", help="verify the seal and show status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(function=command_status)

    predict = subparsers.add_parser(
        "predict-profile", help="replay a frozen recipe for two sealed profile IDs"
    )
    predict.add_argument("--root", type=Path, required=True)
    predict.add_argument("--victim-id", required=True)
    predict.add_argument("--aggressor-id", required=True)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict.set_defaults(function=command_predict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
