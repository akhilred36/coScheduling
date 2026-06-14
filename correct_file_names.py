#!/usr/bin/env python3
import os
import sys
import re


def parse_filename(filename):
    """Parse the filename to extract components."""
    # Format: <num_nodes>_<app_name>_<inhib>_<msgSize>_<waitTime>_-1_d_<sparsity>_<runIter>.slurm
    pattern = r'^(\d+)_(.+)_(.+)_(.+)_(\d+)_-1_d_(.+?)_(\d+)\.slurm$'
    match = re.match(pattern, filename)
    if match:
        return {
            'num_nodes': int(match.group(1)),
            'app_name': match.group(2),
            'inhib': match.group(3),
            'msgSize': match.group(4),
            'waitTime': int(match.group(5)),
            'sparsity': match.group(6),
            'runIter': int(match.group(7))
        }
    return None


def format_filename(parsed, new_num_nodes):
    """Reformat the filename with new num_nodes."""
    return f"{new_num_nodes}_{parsed['app_name']}_{parsed['inhib']}_{parsed['msgSize']}_{parsed['waitTime']}_-1_d_{parsed['sparsity']}_{parsed['runIter']}.slurm"


def main():
    if len(sys.argv) != 2:
        print("Usage: python correct_file_names.py <filepath>")
        sys.exit(1)

    base_path = sys.argv[1]
    slurm_dir = os.path.join(base_path, "slurm_scripts")
    data_dir = os.path.join(base_path, "data")

    if not os.path.exists(slurm_dir):
        print(f"Error: slurm_scripts directory not found at {slurm_dir}")
        sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        sys.exit(1)

    for filename in os.listdir(slurm_dir):
        if not filename.endswith(".slurm"):
            continue

        filepath = os.path.join(slurm_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        if "#SBATCH --nodes 8" not in content:
            continue

        parsed = parse_filename(filename)
        if not parsed:
            print(f"Warning: Could not parse filename {filename}")
            continue

        new_filename = format_filename(parsed, 8)
        new_filepath = os.path.join(slurm_dir, new_filename)

        os.rename(filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

        old_data_dirname = filename.replace(".slurm", "")
        old_data_dir = os.path.join(data_dir, old_data_dirname)

        if os.path.exists(old_data_dir):
            new_data_dirname = new_filename.replace(".slurm", "")
            new_data_dir = os.path.join(data_dir, new_data_dirname)
            os.rename(old_data_dir, new_data_dir)
            print(f"Renamed directory: {old_data_dirname} -> {new_data_dirname}")
        else:
            print(f"Warning: Data directory {old_data_dirname} not found")


if __name__ == "__main__":
    main()
