"""
Main CLI entry point for delta response model.
"""

import argparse
import os
import sys

from train_delta import main as train_main
from evaluate_delta import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(description="Delta Response Model - Train and Evaluate")
    parser.add_argument("--jobs_csv", type=str, required=True)
    parser.add_argument("--inhibitors_csv", type=str, required=True)
    parser.add_argument("--job_inh_csv", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--stage", type=str, choices=["train", "evaluate"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)

    # Training arguments
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--role_dim", type=int, default=16)
    parser.add_argument("--score_hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)

    # Evaluation arguments
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    if args.stage == "train":
        import train_delta
        train_delta.main(args)
    elif args.stage == "evaluate":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for evaluation stage")
            sys.exit(1)
        import evaluate_delta
        evaluate_delta.main(args)


if __name__ == "__main__":
    main()
