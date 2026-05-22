"""
CoScheduling Runs Parser Module

This module provides utilities to parse coscheduling experiment run directories
and extract performance metrics from mpiP output files.
"""

import re
import pandas as pd
from typing import List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from .mpip_parse import MPIPParser


@dataclass
class RunMetrics:
    """Represents the metrics extracted from a single run directory."""
    app: str
    num_nodes: int
    run_iteration: int
    app_time_average: float
    mpi_time_average: float
    total_messages_sent: int


class CoSchedulingRunsParser:
    """
    Parser for coscheduling experiment run directories.
    
    This class parses directories containing mpiP output files and extracts
    performance metrics for each run configuration.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the CoScheduling Runs Parser.
        
        Args:
            base_path: Optional path to the base directory containing run subdirectories.
        """
        self.base_path: Optional[Path] = Path(base_path) if base_path else None
        self._runs: List[RunMetrics] = []
    
    def parse_isolated(self, directory: Union[str, Path]) -> pd.DataFrame:
        """
        Parse an isolated experiment directory and extract metrics.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<app>_<runIteration>/
                mpip_profiles/
                    <filename>.mpiP
            ...
        
        Args:
            directory: Path to the experiment data directory.
            
        Returns:
            pandas DataFrame with columns:
            - "App": Application name
            - "Num Nodes": Number of nodes used
            - "Run Iteration": Run iteration number
            - "App Time Average": Average AppTime across all tasks (seconds)
            - "MPI Time Average": Average MPITime across all tasks (seconds)
            - "Total Messages Sent": Sum of all messages sent
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Iterate over all subdirectories matching the pattern <numNodes>_<app>_<runIteration>
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            # Parse the directory name: <numNodes>_<app>_<runIteration>
            match = re.match(r'^(\d+)_(.+)_(\d+)$', subdir.name)
            if not match:
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            run_iteration = int(match.group(3))
            
            # Find the .mpiP file in mpip_profiles subdirectory
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if not mpiP_files:
                continue
            
            # Parse the first .mpiP file found
            mpiP_file = mpiP_files[0]
            parser = MPIPParser(mpiP_file)
            
            # Extract metrics from MPI Time section
            if parser.mpi_time_df is not None:
                app_time_avg = parser.mpi_time_df['app_time'].mean()
                mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
            else:
                app_time_avg = 0.0
                mpi_time_avg = 0.0
            
            # Extract total messages sent from aggregate sent statistics
            total_messages_sent = 0
            if parser.aggregate_sent_df is not None:
                total_messages_sent = parser.aggregate_sent_df['count'].sum()
            
            # Create RunMetrics object
            metrics = RunMetrics(
                app=app,
                num_nodes=num_nodes,
                run_iteration=run_iteration,
                app_time_average=app_time_avg,
                mpi_time_average=mpi_time_avg,
                total_messages_sent=total_messages_sent
            )
            self._runs.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "App": [r.app for r in self._runs],
            "Num Nodes": [r.num_nodes for r in self._runs],
            "Run Iteration": [r.run_iteration for r in self._runs],
            "App Time Average": [r.app_time_average for r in self._runs],
            "MPI Time Average": [r.mpi_time_average for r in self._runs],
            "Total Messages Sent": [r.total_messages_sent for r in self._runs]
        })
        
        return df
