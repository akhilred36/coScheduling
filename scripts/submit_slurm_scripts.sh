#!/bin/bash

# Check if a directory path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Error: No directory path provided."
    exit 1
fi

# Store the provided directory path
DIR="$1"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Initialize counters
total_files=0
submitted_count=0
failed_count=0

# Iterate through all .slurm files in the directory
for slurm_file in "$DIR"/*.slurm; do
    # Check if any .slurm files were found (glob might not match anything)
    if [ -f "$slurm_file" ]; then
        total_files=$((total_files + 1))
        echo "Submitting: $slurm_file"
        if sbatch "$slurm_file"; then
            submitted_count=$((submitted_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    fi
done

# Print summary
echo ""
echo "=== Summary ==="
echo "Total .slurm files found: $total_files"
echo "Successfully submitted: $submitted_count"
echo "Failed to submit: $failed_count"
