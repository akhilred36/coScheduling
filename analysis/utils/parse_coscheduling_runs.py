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
    
    def parse_coscheduled(self, directory: Union[str, Path]) -> pd.DataFrame:
        """
        Parse a coscheduled experiment directory and extract metrics for paired applications.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<appA>_<appB>_<runIteration>/
                mpip_profiles/
                    <execA>.mpiP
                    <execB>.mpiP
            ...
        
        Args:
            directory: Path to the experiment data directory.
            
        Returns:
            pandas DataFrame with columns:
            - "Num Nodes": Number of nodes used
            - "App A": First application name
            - "App B": Second application name
            - "Run Iteration": Run iteration number
            - "App A Time Average": Average AppTime across all tasks for App A (seconds)
            - "App B Time Average": Average AppTime across all tasks for App B (seconds)
            - "App A MPI Time Average": Average MPITime across all tasks for App A (seconds)
            - "App B MPI Time Average": Average MPITime across all tasks for App B (seconds)
            - "App A Total Messages Sent": Sum of all messages sent for App A
            - "App B Total Messages Sent": Sum of all messages sent for App B
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Load the run config to map exec names to app names
        run_config_path = Path("../run_configs/1_nodes.json")
        if not run_config_path.exists():
            raise FileNotFoundError(f"Run config file not found: {run_config_path}")
        
        import json
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        
        # Create reverse mapping: exec -> app
        exec_to_app = {}
        for app_name, app_info in run_config.get("apps", {}).items():
            exec_name = app_info.get("exec")
            if exec_name:
                exec_to_app[exec_name] = app_name
        
        # Iterate over all subdirectories matching the pattern <numNodes>_<appA>_<appB>_<runIteration>
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            # Parse the directory name: <numNodes>_<appA>_<appB>_<runIteration>
            match = re.match(r'^(\d+)_(.+)_(.+)_(\d+)$', subdir.name)
            if not match:
                continue
            
            num_nodes = int(match.group(1))
            app_a_name = match.group(2)
            app_b_name = match.group(3)
            run_iteration = int(match.group(4))
            
            # Find the .mpiP files in mpip_profiles subdirectory
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if len(mpiP_files) != 2:
                continue
            
            # Parse both .mpiP files and identify which app each belongs to
            app_a_metrics = None
            app_b_metrics = None
            
            for mpiP_file in mpiP_files:
                # Extract exec name from filename (e.g., "fiesta.56.712080.1.mpiP" -> "fiesta")
                filename = mpiP_file.stem
                exec_name = filename.split('.')[0]
                
                # Map exec name to app name
                app_name = exec_to_app.get(exec_name)
                if not app_name:
                    continue
                
                # Parse the mpiP file
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
                    total_messages_sent = int(parser.aggregate_sent_df['count'].sum())
                
                metrics = RunMetrics(
                    app=app_name,
                    num_nodes=num_nodes,
                    run_iteration=run_iteration,
                    app_time_average=app_time_avg,
                    mpi_time_average=mpi_time_avg,
                    total_messages_sent=total_messages_sent
                )
                
                # Assign to App A or App B based on app name
                if app_name == app_a_name:
                    app_a_metrics = metrics
                elif app_name == app_b_name:
                    app_b_metrics = metrics
            
            # Only add if both apps were found and parsed
            if app_a_metrics is not None and app_b_metrics is not None:
                self._runs.append({
                    "Num Nodes": num_nodes,
                    "App A": app_a_name,
                    "App B": app_b_name,
                    "Run Iteration": run_iteration,
                    "App A Time Average": app_a_metrics.app_time_average,
                    "App B Time Average": app_b_metrics.app_time_average,
                    "App A MPI Time Average": app_a_metrics.mpi_time_average,
                    "App B MPI Time Average": app_b_metrics.mpi_time_average,
                    "App A Total Messages Sent": app_a_metrics.total_messages_sent,
                    "App B Total Messages Sent": app_b_metrics.total_messages_sent
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(self._runs)
        
        return df
