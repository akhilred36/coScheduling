#!/usr/bin/env python3
import os
import re

base_dir = "/home/akhil/hpcResearch/repos/coScheduling/sample_data/experiments_coscheduling_inhib_coscheduled/20260529_102306/data"

for name in os.listdir(base_dir):
    old_path = os.path.join(base_dir, name)
    if os.path.isdir(old_path):
        match = re.match(r'^(\d+)_(.+)$', name)
        if match:
            new_name = "8_" + match.group(2)
            new_path = os.path.join(base_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {name} -> {new_name}")
