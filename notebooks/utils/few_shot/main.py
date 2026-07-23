#!/usr/bin/env python3
"""
Main Orchestration Script
Ties together preprocessing, training, and evaluation.
"""

import os
import subprocess
import sys


# Training configuration
TRAIN_APPS = ['amg', 'beatnik', 'fiesta', 'kripke', 'laghos', 
              'lammps', 'minife', 'minivite', 'quicksilver']

DATA_PATH = "data/processed_data.npz"
PREPROCESS_SCRIPT = "preprocess.py"
TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "evaluate.py"


def main():
    print("=" * 60)
    print("Slowdown Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocessing
    print("\n[Step 1/3] Data Preprocessing...")
    if os.path.exists(DATA_PATH):
        print(f"Found existing {DATA_PATH}, skipping preprocessing.")
    else:
        print(f"Running preprocessing script: {PREPROCESS_SCRIPT}")
        result = subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
    
    # Step 2: Training
    print("\n[Step 2/3] Training...")
    print(f"Running training script: {TRAIN_SCRIPT}")
    result = subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
    
    # Step 3: Evaluation
    print("\n[Step 3/3] Evaluation...")
    print(f"Running evaluation script: {EVAL_SCRIPT}")
    result = subprocess.run([sys.executable, EVAL_SCRIPT], check=True)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
