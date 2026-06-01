#!/usr/bin/env python3
import os
import re

base_dir = "/home/akhil/hpcResearch/repos/coScheduling/sample_data/experiments_coscheduling_inhib_coscheduled/20260529_102306/slurm_scripts"

pattern = re.compile(r'^(\d+)_(.+)\.(err|out|slurm)$')

for name in os.listdir(base_dir):
    match = pattern.match(name)
    if match:
        old_path = os.path.join(base_dir, name)
        if os.path.isfile(old_path):
            new_name = "8_" + match.group(2) + "." + match.group(3)
            new_path = os.path.join(base_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {name} -> {new_name}")
