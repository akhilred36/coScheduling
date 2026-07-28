#!/usr/bin/env python3
"""
Experiment runner for few-shot learning pipeline.
Runs experiments across all three evaluation modes: random_split, one_known, zero_shot.
"""

import subprocess
import itertools
import os
import argparse
from tqdm import tqdm


def run_random_split_experiments(venv_path=None):
    """Run random_split experiments with varying train/test ratios."""
    base_dir = "random_split"
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    iterations = 50
    
    if venv_path:
        python_cmd = os.path.join(venv_path, "bin", "python3")
    else:
        python_cmd = "python3"
    
    for ratio in tqdm(ratios, desc="Train/val ratios", position=0):
        for iter_idx in range(iterations):
            save_dir = f"{base_dir}/random_split_{iter_idx}"
            os.makedirs(save_dir, exist_ok=True)
            output_log = os.path.join(save_dir, "output.log")
            cmd = [
                python_cmd, "main.py",
                "--eval_method", "random_split",
                "--train_split", str(ratio),
                "--save_csv", save_dir
            ]
            with open(output_log, "w") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)
    
    print(f"\nCompleted {len(ratios) * iterations} random_split experiments")


def get_all_apps():
    """Return list of all applications."""
    return ["beatnik", "amg", "fiesta", "kripke", "laghos", "lammps", "minife", "minivite", "quicksilver", "tricount"]


def run_one_known_experiments(venv_path=None):
    """Run one_known experiments with all combinations of training apps."""
    base_dir = "one_known"
    apps = get_all_apps()
    sizes = range(3, 9)  # 3 to 8 apps in training set
    
    if venv_path:
        python_cmd = os.path.join(venv_path, "bin", "python3")
    else:
        python_cmd = "python3"
    
    for size in tqdm(sizes, desc="Training set sizes", position=0):
        combinations = list(itertools.combinations(apps, size))
        for combo in tqdm(combinations, desc=f"Size {size}", position=1, leave=False):
            save_dir = f"{base_dir}/one_known_{'_'.join(combo)}"
            os.makedirs(save_dir, exist_ok=True)
            output_log = os.path.join(save_dir, "output.log")
            training_apps = ",".join(combo)
            
            cmd = [
                python_cmd, "main.py",
                "--eval_method", "one_known",
                "--training_apps", training_apps,
                "--save_csv", save_dir
            ]
            with open(output_log, "w") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)
    
    total = sum(len(list(itertools.combinations(apps, size))) for size in sizes)
    print(f"\nCompleted {total} one_known experiments")


def run_zero_shot_experiments(venv_path=None):
    """Run zero_shot experiments with all combinations of training apps."""
    base_dir = "zero_shot"
    apps = get_all_apps()
    sizes = range(3, 9)  # 3 to 8 apps in training set
    
    if venv_path:
        python_cmd = os.path.join(venv_path, "bin", "python3")
    else:
        python_cmd = "python3"
    
    for size in tqdm(sizes, desc="Training set sizes", position=0):
        combinations = list(itertools.combinations(apps, size))
        for combo in tqdm(combinations, desc=f"Size {size}", position=1, leave=False):
            save_dir = f"{base_dir}/zero_shot_{'_'.join(combo)}"
            os.makedirs(save_dir, exist_ok=True)
            output_log = os.path.join(save_dir, "output.log")
            training_apps = ",".join(combo)
            
            cmd = [
                python_cmd, "main.py",
                "--eval_method", "zero_shot",
                "--training_apps", training_apps,
                "--save_csv", save_dir
            ]
            with open(output_log, "w") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)
    
    total = sum(len(list(itertools.combinations(apps, size))) for size in sizes)
    print(f"\nCompleted {total} zero_shot experiments")


def main():
    """Run all experiments."""
    parser = argparse.ArgumentParser(description="Run few-shot learning experiments")
    parser.add_argument("--venv", type=str, default=None,
                        help="Path to Python virtual environment (e.g., /path/to/venv/bin/python3)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "random_split", "one_known", "zero_shot"],
                        help="Which experiment mode to run (default: all)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting Few-Shot Learning Experiments")
    print("=" * 60)
    
    if args.venv:
        print(f"Using virtual environment: {args.venv}\n")
    
    if args.mode in ["all", "random_split"]:
        print("\n" + "=" * 60)
        print("MODE 1: random_split")
        print("=" * 60)
        run_random_split_experiments(args.venv)
    
    if args.mode in ["all", "one_known"]:
        print("\n" + "=" * 60)
        print("MODE 2: one_known")
        print("=" * 60)
        run_one_known_experiments(args.venv)
    
    if args.mode in ["all", "zero_shot"]:
        print("\n" + "=" * 60)
        print("MODE 3: zero_shot")
        print("=" * 60)
        run_zero_shot_experiments(args.venv)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
