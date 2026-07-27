#!/usr/bin/env python3
"""
Main Orchestration Script
Ties together preprocessing, training, and evaluation.
"""

import argparse
import os
import subprocess
import sys

DATA_PATH = "data/processed_data.npz"
PREPROCESS_SCRIPT = "preprocess.py"
TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "evaluate.py"


def main():
    parser = argparse.ArgumentParser(description='Main orchestration script for slowdown prediction pipeline')
    parser.add_argument('--eval_method', type=str, default='random_split',
                        choices=['random_split', 'zero_shot', 'one_known'],
                        help='Evaluation method to use')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/test split fraction for random_split evaluation (default: 0.8)')
    parser.add_argument('--training_apps', type=str, default='amg,beatnik,fiesta,laghos,lammps,minife,minivite',
                        help='Comma-separated list of training applications (default: amg,beatnik,fiesta,laghos,lammps,minife,minivite)')
    parser.add_argument('--save_csv', type=str, nargs='?', const='output', default=None,
                        help='Save train.csv and test.csv files to specified directory (default: output/)')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'preprocess', 'train', 'eval'],
                        help='Which step to run')
    args = parser.parse_args()
    
    if args.eval_method != 'random_split' and args.train_split != 0.8:
        print("Error: --train_split can only be used with --eval_method random_split")
        sys.exit(1)
    
    print("=" * 60)
    print("Slowdown Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocessing
    if args.step in ['all', 'preprocess']:
        print("\n[Step 1/3] Data Preprocessing...")
        if os.path.exists(DATA_PATH):
            print(f"Found existing {DATA_PATH}, skipping preprocessing.")
        else:
            print(f"Running preprocessing script: {PREPROCESS_SCRIPT}")
            result = subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
    
    # Step 2: Training
    if args.step in ['all', 'train']:
        print("\n[Step 2/3] Training...")
        print(f"Running training script: {TRAIN_SCRIPT}")
        train_cmd = [sys.executable, TRAIN_SCRIPT, '--eval_method', args.eval_method, '--train_split', str(args.train_split), '--training_apps', args.training_apps]
        if args.save_csv:
            train_cmd.append('--save_csv')
            train_cmd.append(args.save_csv)
        result = subprocess.run(train_cmd, check=True)
    
    # Step 3: Evaluation
    if args.step in ['all', 'eval']:
        print("\n[Step 3/3] Evaluation...")
        print(f"Running evaluation script: {EVAL_SCRIPT}")
        eval_cmd = [sys.executable, EVAL_SCRIPT, '--eval_method', args.eval_method, '--train_split', str(args.train_split), '--training_apps', args.training_apps]
        if args.save_csv:
            eval_cmd.append('--save_csv')
            eval_cmd.append(args.save_csv)
        result = subprocess.run(eval_cmd, check=True)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
